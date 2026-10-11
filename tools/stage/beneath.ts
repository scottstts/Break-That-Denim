/**
 * Headless run of the staged "Beneath" experience with the product's mechanics and renderer: the
 * same Stage drives the clip, the creature's body (its claws pierce), its holds and sound cues as in the worker.
 *   node tools/gpu/run.mjs tools/stage/beneath.ts [name] [width height]
 * FRAMES=t1,t2,... stills at those times (default every 1 s); VIDEO=1 every frame at 30 Hz into
 * artifacts/captures/stage/<name>-frames (ffmpeg it); UNTIL=t stops early; STATS=1 prints counters;
 * CAM='{"eye":[..],"target":[..],"up":[..]}' renders from a debug camera instead of the stage's; LIT=1 lights
 * it flat (the product's key, a bright fill, no film) to see the mechanics;
 * AUDIO=file.json records the acoustic records, holds and sound cues (tools/stage/audio.ts renders them).
 * TRACE=1 prints the rupture count every 0.1 s.
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
  const perFrame = Math.round(1 / 30 / H);
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
    if (process.env.TRACE && frame % 3 === 0) console.log(`trace t=${stage.time.toFixed(2)} ruptures=${c[0]} freed=${topology.freedCount}`);
    if (c[0] + c[1] !== events) {
      events = c[0] + c[1];
      await topology.request();
    }
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
        console.log(`t=${stage.time.toFixed(2)} ruptures=${c[0]} pulled=${c[1]} rejected=${c[12]} freed=${topology.freedCount} face ndc=(${(q[0] / q[3]).toFixed(2)}, ${(q[1] / q[3]).toFixed(2)}) at ${dist.toFixed(2)} m ${label}`);
        next++;
      }
    }
    frame++;
    if (frame % 30 === 0) await drain(device);
  }
  console.log(`simulated ${stage.time.toFixed(2)} s in ${(simMs / 1000).toFixed(1)} s of substeps (${((simMs / frame) * 1).toFixed(1)} ms per frame of ${perFrame} substeps)`);
  stage.stop();
  if (process.env.AUDIO) writeFileSync(resolve(process.env.AUDIO), JSON.stringify({ h: H, records: audioRecords, holds, cues }));
  device.destroy();
}
