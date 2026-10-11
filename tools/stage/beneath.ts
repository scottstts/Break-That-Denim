/**
 * Headless run of the staged "Beneath" experience with the product's mechanics and renderer: the
 * same Stage drives the clip, the creature's body (its claws pierce), its holds and sound cues as in the worker.
 *   node tools/gpu/run.mjs tools/stage/beneath.ts [name] [width height]
 * FRAMES=t1,t2,... stills at those times (default every 1 s); VIDEO=1 every frame at 30 Hz into
 * artifacts/captures/stage/<name>-frames (ffmpeg it); UNTIL=t stops early; STATS=1 prints counters;
 * CAM='{"eye":[..],"target":[..],"up":[..]}' renders from a debug camera instead of the stage's; LIT=1 lights
 * it flat (the product's key, a bright fill, no film) to see the mechanics;
 * AUDIO=file.json records the acoustic records, holds and sound cues (tools/stage/audio.ts renders them).
 * TRACE=1 prints the rupture count every 0.1 s. DUMPAT=t dumps the state then (session.ts DUMP format);
 * WATCH=n prints node n every frame.
 * DEBRIS=1 prints the detached material every 0.1 s; DEBRISAT=t1,t2,.. also what it rests on then
 * (capsule, cloth, nothing) and each piece's height, velocity and closest skin.
 * ASYNC=1 requests connectivity without waiting, as the worker does; SUBSTEPS=n sets the number
 * per presented frame (default 4; the worker varies it up to 5).
 * TOPOLOGY_AUDIT=1 independently checks connectivity and both claw anchors every frame; AUDIT_NODE
 * selects the centre node to trace (default 11550). Saves <name>-topology.json without awaiting
 * diagnostic readbacks between frames.
 * The laugh moves the jaw as heard in the browser on the desktop it was laid out on (PACE, default
 * DESKTOP_PACE: simulated seconds per wall second).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { AcousticLog } from '../../src/engine/acousticLog';
import { ACOUSTIC_SLOTS, RECORD_FLOATS, SOURCES, decodeRecord, fracturePosition } from '../../src/audio/record';
import type { StageEvent } from '../../src/stage/beneath';
import { resolve } from 'node:path';
import { cameraMatrices, Renderer } from '../../src/rendering/renderer';
import { transformPoint } from '../../src/rendering/math';
import { BodyContact } from '../../src/simulation/gpu/body';
import { readBuffer } from '../../src/gpu/util';
import { Interaction } from '../../src/simulation/gpu/interaction';
import { GPU_DEFAULTS, RELEASE_REFINEMENT } from '../../src/simulation/gpu/solver';
import { Topology } from '../../src/simulation/gpu/topology';
import { CreatureRenderer } from '../../src/stage/creatureRenderer';
import { Puppet } from '../../src/stage/puppet';
import { Rig } from '../../src/stage/rig';
import { Stage } from '../../src/stage/stage';
import { CAPTURES, drain, gpuDevice, saveTexture } from '../gpu/env';
import { settledSolver } from '../gpu/scripts/render';
import { loudness, loudnessAt } from '../../src/audio/envelope';
import { DESKTOP_PACE, LAUGH_AT } from '../../src/stage/beneath';
import { decodeAudio, loadClip, loadCreature } from './env';
import { TopologyAudit } from '../studies/topologyAudit';

export default async function main(args: string[]): Promise<void> {
  const name = args[0] ?? 'beneath';
  const width = Number(args[1] ?? 1200);
  const height = Number(args[2] ?? 750);
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const H = GPU_DEFAULTS.h;
  const interaction = new Interaction(device, solver, scene);
  const body = new BodyContact(device, solver, scene);
  solver.postCommit = (pass) => body.encodePierce(pass);
  solver.preSubstep = (pass) => interaction.encodeSubstep(pass);
  solver.preAssemble = (dispatch) => {
    body.encode(dispatch);
    interaction.encode(dispatch);
  };
  const topology = new Topology(device, solver, scene, () => interaction.coreNodes);
  const audit = process.env.TOPOLOGY_AUDIT ? new TopologyAudit(solver, interaction, Number(process.env.AUDIT_NODE ?? 11550)) : null;
  await topology.request();
  const format: GPUTextureFormat = 'rgba8unorm';
  const renderer = new Renderer(device, format, solver, scene);
  renderer.resize(width, height);
  const { model, images } = loadCreature();
  const rig = new Rig(model);
  const puppet = new Puppet(rig);
  const creature = new CreatureRenderer(renderer, model, images);
  let refine = 0;
  const acoustic = process.env.AUDIO ? new AcousticLog(device, solver.counters) : null;
  const audioRecords: number[][] = [];
  const holds: [number, number, boolean][] = [];
  const cues: StageEvent[] = [];
  const laugh = loudness(decodeAudio('public/exprerience_sounds/creepy_laugh.mp3', 48000), 48000);
  const pace = Number(process.env.PACE ?? DESKTOP_PACE);
  let simTime = 0;
  const stage = new Stage({
    device,
    scene,
    solver,
    interaction,
    renderer,
    body,
    creature,
    puppet,
    clip: loadClip(),
    analyse: () => topology.request(),
    letGo: (slot) => {
      interaction.letGo(slot);
      void topology.request().then(() => {
        interaction.release(slot);
        refine = Math.round(RELEASE_REFINEMENT.duration / H);
      });
    },
    cue: (e) => {
      cues.push(e);
      if (process.env.STATS) console.log(`  cue ${e.kind} at ${e.t.toFixed(2)}`);
    },
    held: (slot, held) => {
      holds.push([audioRecords.length, slot, held]);
      if (process.env.STATS) console.log(`  slot ${slot} ${held ? 'held' : 'released'}`);
    },
    laugh: () => loudnessAt(laugh, (simTime - LAUGH_AT) / pace),
  });
  const until = Number(process.env.UNTIL ?? stage.duration);
  const video = !!process.env.VIDEO;
  const times = (process.env.FRAMES ?? '').split(',').filter(Boolean).map(Number);
  if (!video && times.length === 0) for (let t = 1; t <= until; t += 1) times.push(t);
  const dir = video ? `stage/${name}-frames` : 'stage';
  if (video) mkdirSync(resolve(CAPTURES, dir), { recursive: true });
  const tex = device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const counters = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  let events = 0;
  let frame = 0;
  let next = 0;
  let simMs = 0;
  const perFrame = Number(process.env.SUBSTEPS ?? Math.round(1 / 30 / H));
  for (let k = 0; stage.time < until; k++) {
    const t0 = performance.now();
    if (k % perFrame === 0) acoustic?.begin();
    simTime = stage.time + H;
    stage.substep(H);
    const enc = device.createCommandEncoder();
    solver.encodeSubstep(enc, refine-- > 0 ? RELEASE_REFINEMENT.newtonIterations : 0);
    acoustic?.record(enc);
    device.queue.submit([enc.finish()]);
    if ((k + 1) % perFrame !== 0) continue;
    await device.queue.onSubmittedWorkDone();
    const rec = await acoustic?.end();
    if (rec) {
      const out = new Float32Array(RECORD_FLOATS);
      for (let q = 0; q < rec.count; q++) {
        const pans = Array.from({ length: SOURCES }, (_, src) => fracturePosition(rec.raw, q * ACOUSTIC_SLOTS, src)?.[0] ?? 0);
        decodeRecord(rec.raw, q * ACOUSTIC_SLOTS, out, 0, H, pans);
        audioRecords.push(Array.from(out));
      }
    }
    simMs += performance.now() - t0;
    // ruptures since the last analysis: the worker requests one from its rest probe
    const ce = device.createCommandEncoder();
    ce.copyBufferToBuffer(solver.counters, 0, counters, 0, 64);
    device.queue.submit([ce.finish()]);
    await counters.mapAsync(GPUMapMode.READ);
    const c = new Uint32Array(counters.getMappedRange().slice(0));
    counters.unmap();
    // TRACE=1: ruptures every 0.1 s (where the cloth fails, and how fast)
    // DUMPAT=t: the state then (session.ts DUMP format: x, spans, tails, yarn field) to <name>-<t>.state
    if (process.env.DUMPAT && Math.abs(stage.time - Number(process.env.DUMPAT)) < 0.017) {
      const parts = await Promise.all([solver.x, solver.segs, solver.tails, solver.field].map((b) => readBuffer(device, b)));
      const out = new Uint8Array(parts.reduce((t, b) => t + b.byteLength, 0));
      let at = 0;
      for (const b of parts) {
        out.set(new Uint8Array(b), at);
        at += b.byteLength;
      }
      writeFileSync(resolve(CAPTURES, `${name}-${process.env.DUMPAT}.state`), out);
    }
    // WATCH=n: node n every frame (state, position, velocity)
    if (process.env.WATCH) {
      const n = Number(process.env.WATCH);
      const xs = new Float32Array(await readBuffer(device, solver.x));
      const vs = new Float32Array(await readBuffer(device, solver.v));
      console.log(`watch t=${stage.time.toFixed(3)} state ${xs[4 * n + 3]} x (${[0, 1, 2].map((i) => (xs[4 * n + i] * 1e3).toFixed(1)).join(', ')}) mm v (${[0, 1, 2].map((i) => vs[4 * n + i].toFixed(3)).join(', ')}) m/s`);
    }
    // DEBRIS=1: detached material still in the scene every 0.1 s: nodes, height range, mean vertical
    // speed, nodes slower than 1 cm/s, retired nodes
    if (process.env.DEBRIS && frame % 3 === 0) {
      const xs = new Float32Array(await readBuffer(device, solver.x));
      const vs = new Float32Array(await readBuffer(device, solver.v));
      let cnt = 0;
      let slow = 0;
      let retired = 0;
      let zlo = Infinity;
      let zhi = -Infinity;
      let vz = 0;
      for (let n = 0; n < solver.N; n++) {
        if (xs[4 * n + 3] > 1.5) retired++;
        else if (xs[4 * n + 3] > 0.5) {
          cnt++;
          zlo = Math.min(zlo, xs[4 * n + 2]);
          zhi = Math.max(zhi, xs[4 * n + 2]);
          vz += vs[4 * n + 2];
          if (Math.hypot(vs[4 * n], vs[4 * n + 1], vs[4 * n + 2]) < 0.01) slow++;
        }
      }
      // DEBRISAT=t1,t2,..: what each debris node rests against then (nearest capsule and its gap to the skin)
      if (process.env.DEBRISAT && process.env.DEBRISAT.split(',').some((v) => Math.abs(stage.time - Number(v)) < 0.051)) {
        const caps = puppet.capsules();
        const names = puppet.capsuleJoints().map((j) => rig.names[j]);
        const tally = new Map<string, number>();
        const sg = new Uint32Array(await readBuffer(device, solver.segs));
        const nx = scene.layers[0].nx;
        const act = (q: number) => q >= 0 && (sg[8 * q + 5] & 3) === 1;
        const cloth: number[] = [];
        for (let m = 0; m < solver.N; m++) if (xs[4 * m + 3] < 0.5) cloth.push(m);
        const example = new Map<string, number>();
        for (let n = 0; n < solver.N; n++) {
          if (!(xs[4 * n + 3] > 0.5 && xs[4 * n + 3] < 1.5)) continue;
          const q = [xs[4 * n], xs[4 * n + 1], xs[4 * n + 2]];
          let best = Infinity;
          let who = '';
          caps.forEach((c, k) => {
            const e = [0, 1, 2].map((i) => c.b[i] - c.a[i]);
            const ee = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            const t = ee > 1e-12 ? Math.min(1, Math.max(0, ((q[0] - c.a[0]) * e[0] + (q[1] - c.a[1]) * e[1] + (q[2] - c.a[2]) * e[2]) / ee)) : 0;
            const gap = Math.hypot(...[0, 1, 2].map((i) => q[i] - c.a[i] - t * e[i])) - c.radius;
            if (gap < best) {
              best = gap;
              who = `${names[k]}${c.active ? '' : ' (inactive)'}`;
            }
          });
          let near = Infinity;
          for (const m of cloth) near = Math.min(near, Math.hypot(xs[4 * m] - q[0], xs[4 * m + 1] - q[1], xs[4 * m + 2] - q[2]));
          // a span still joining it to a node that is cloth
          const linked = [[2 * n, n + 1], [2 * n + 1, n + nx], [2 * (n - 1), n - 1], [2 * (n - nx) + 1, n - nx]].some(([sIdx, m]) => m >= 0 && m < solver.N && act(sIdx) && xs[4 * m + 3] < 0.5);
          const speed = Math.hypot(vs[4 * n], vs[4 * n + 1], vs[4 * n + 2]);
          const key = best < 0.004 ? `on ${who}` : linked ? 'linked to cloth' : near < 0.004 ? 'on cloth' : speed < 0.02 ? 'free and still' : 'free and moving';
          if (!example.has(key)) example.set(key, n);
          tally.set(key, (tally.get(key) ?? 0) + 1);
        }
        console.log(`debris at ${stage.time.toFixed(2)}: ${[...tally].map(([k, v]) => `${v} ${k} (e.g. ${example.get(k)})`).join(', ')}`);
        // pieces: debris joined by active spans; per piece its size, centroid height, mean velocity, nearest capsule gap
        const seen = new Set<number>();
        for (let n0 = 0; n0 < solver.N; n0++) {
          if (seen.has(n0) || !(xs[4 * n0 + 3] > 0.5 && xs[4 * n0 + 3] < 1.5)) continue;
          const piece = [n0];
          seen.add(n0);
          for (let k = 0; k < piece.length; k++) {
            const n = piece[k];
            for (const [sIdx, m] of [[2 * n, n + 1], [2 * n + 1, n + nx], [2 * (n - 1), n - 1], [2 * (n - nx) + 1, n - nx]]) {
              if (m < 0 || m >= solver.N || seen.has(m) || !act(sIdx) || !(xs[4 * m + 3] > 0.5 && xs[4 * m + 3] < 1.5)) continue;
              seen.add(m);
              piece.push(m);
            }
          }
          const mean = (o: number, a: Float32Array) => piece.reduce((acc, n) => acc + a[4 * n + o], 0) / piece.length;
          let gap = Infinity;
          let on = '';
          for (const n of piece) caps.forEach((c, ci) => {
            const e = [0, 1, 2].map((i) => c.b[i] - c.a[i]);
            const ee = e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
            const t = ee > 1e-12 ? Math.min(1, Math.max(0, ((xs[4 * n] - c.a[0]) * e[0] + (xs[4 * n + 1] - c.a[1]) * e[1] + (xs[4 * n + 2] - c.a[2]) * e[2]) / ee)) : 0;
            const g = Math.hypot(...[0, 1, 2].map((i) => xs[4 * n + i] - c.a[i] - t * e[i])) - c.radius;
            if (g < gap) {
              gap = g;
              on = names[ci];
            }
          });
          if (piece.length >= 3 && piece.length <= 10) {
            const ext = new Float32Array(await readBuffer(device, solver.ext));
            const meta = new Float32Array(await readBuffer(device, (solver.contact as unknown as { meta: GPUBuffer }).meta));
            for (const n of piece) {
              const f = (o: ArrayLike<number>, k: number) => [0, 1, 2].map((i) => o[k + i].toFixed(3)).join(', ');
              console.log(`    node ${n} (i=${n % nx} j=${Math.floor(n / nx)}) z ${(xs[4 * n + 2] * 1e3).toFixed(1)} mm v (${f(vs, 4 * n)}) ext g (${f(ext, 12 * n)}) bound n (${f(meta, 8 * n)}) share ${(meta[8 * n + 3] * 1e3).toFixed(2)} mm recip (${f(meta, 8 * n + 4)}) ${(meta[8 * n + 7] * 1e3).toFixed(2)} mm`);
            }
          }
          const vp = cameraMatrices(stage.frame(), width / height).viewProj;
          const cw = transformPoint(vp, [0, 1, 2].map((o) => mean(o, xs)) as [number, number, number]);
          const pix = `pixel (${((0.5 + 0.5 * cw[0] / cw[3]) * width).toFixed(0)}, ${((0.5 - 0.5 * cw[1] / cw[3]) * height).toFixed(0)})`;
          if (piece.length >= 3) console.log(`  ${pix} first node ${piece[0]}`);
          if (piece.length >= 3) console.log(`  piece of ${piece.length} at (${[0, 1, 2].map((o) => (mean(o, xs) * 1e3).toFixed(0)).join(', ')}) mm, v (${[0, 1, 2].map((o) => mean(o, vs).toFixed(3)).join(', ')}) m/s, closest capsule ${on} at ${(gap * 1e3).toFixed(1)} mm`);
        }
      }
      if (cnt + retired > 0) console.log(`debris t=${stage.time.toFixed(2)} ${cnt} nodes z ${(zlo * 1e3).toFixed(0)}..${(zhi * 1e3).toFixed(0)} mm mean vz ${(vz / Math.max(cnt, 1)).toFixed(3)} m/s slow ${slow} retired ${retired}`);
    }
    if (process.env.TRACE && frame % 3 === 0) console.log(`trace t=${stage.time.toFixed(2)} ruptures=${c[0]} freed=${topology.freedCount}`);
    if (c[0] + c[1] !== events) {
      events = c[0] + c[1];
      if (process.env.ASYNC) void topology.request();
      else await topology.request();
    }
    audit?.record(stage.time);
    const camera = process.env.CAM ? { ...stage.frame(), ...(JSON.parse(process.env.CAM) as object) } : stage.frame();
    if (process.env.NOCREATURE) creature.visible = false;
    if (process.env.LIT) {
      Object.assign(renderer, { keyOffset: [-0.05, 0.6, 0.3], keyAim: [0, 0.12, 0], keyIntensity: 6, keyCone: 0.9, keyInner: 1, keyLeak: 0.05, keyColor: [1, 1, 1], ambient: [0.6, 0.6, 0.6], exposure: 1, freeze: false });
      renderer.post = { grade: 0, grain: 0, vignette: 0, glitch: 0, glitchSeed: 0 };
    }
    const capture = video || (next < times.length && stage.time >= times[next] - 1e-6);
    const re = device.createCommandEncoder();
    renderer.encode(re, camera, stage.time, tex.createView());
    device.queue.submit([re.finish()]);
    if (capture) {
      const label = video ? `${dir}/${String(frame).padStart(4, '0')}` : `${dir}/${name}-${stage.time.toFixed(2)}`;
      await saveTexture(device, tex, label);
      if (!video) {
        // where the face is in the frame (NDC, y up) and how far from the lens
        const m = cameraMatrices(camera, width / height);
        const face = puppet.where('frontface');
        const q = transformPoint(m.viewProj, face);
        const dist = Math.hypot(face[0] - m.eye[0], face[1] - m.eye[1], face[2] - m.eye[2]);
        if (process.env.WATCH) {
          const n = Number(process.env.WATCH);
          const xs = new Float32Array(await readBuffer(device, solver.x));
          const w = transformPoint(m.viewProj, [xs[4 * n], xs[4 * n + 1], xs[4 * n + 2]]);
          console.log(`watch ${n} in frame ${label}: pixel (${((0.5 + 0.5 * w[0] / w[3]) * width).toFixed(0)}, ${((0.5 - 0.5 * w[1] / w[3]) * height).toFixed(0)})`);
        }
        console.log(`t=${stage.time.toFixed(2)} ruptures=${c[0]} pulled=${c[1]} rejected=${c[12]} freed=${topology.freedCount} face ndc=(${(q[0] / q[3]).toFixed(2)}, ${(q[1] / q[3]).toFixed(2)}) at ${dist.toFixed(2)} m ${label}`);
        next++;
      }
    }
    frame++;
    if (frame % 30 === 0) await drain(device);
  }
  console.log(`simulated ${stage.time.toFixed(2)} s in ${(simMs / 1000).toFixed(1)} s of substeps (${((simMs / frame) * 1).toFixed(1)} ms per frame of ${perFrame} substeps)`);
  stage.stop();
  if (audit) {
    await audit.finish();
    console.log('topology audit summary', JSON.stringify(audit.summary()));
    writeFileSync(resolve(CAPTURES, `${name}-topology.json`), JSON.stringify({ summary: audit.summary(), rows: audit.rows }));
  }
  if (process.env.AUDIO) writeFileSync(resolve(process.env.AUDIO), JSON.stringify({ h: H, records: audioRecords, holds, cues }));
  device.destroy();
}
