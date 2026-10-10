/**
 * Scripted interactive session (development diagnostics): drives the product's grip with the
 * worker's handle logic through a gesture script (grab, drag, yank, release, regrab), checks the
 * state every presented frame (non-finite values, runaway nodes, damage gating) and renders selected
 * frames into one contact sheet (artifacts/captures/<name>.png).
 *
 *   node tools/gpu/run.mjs tools/gpu/scripts/session.ts <script> [name] [every]
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { Interaction } from '../../../src/simulation/gpu/interaction';
import { Topology } from '../../../src/simulation/gpu/topology';
import { GRIP, pinchScale } from '../../../src/simulation/grip';
import { CAPTURES, drain, encodePng, gpuDevice } from '../env';
import { settledSolver } from './render';

/** Gesture steps in world metres relative to the grab point; times in seconds. */
type Step =
  | { grab: [number, number] }
  | { to: [number, number, number]; t: number }
  | { release: true }
  | { wait: number }
  // cut a U-shaped flap of half-size `half` lattice cells (hinge on its +v side) at a world point
  | { cut: [number, number]; half: number }
  // cut a square piece free on all four sides and lay it `lift` m above the sheet, `shift` m along +x
  | { scrap: [number, number]; half: number; lift: number; shift: number }
  // fold the cut flap over its hinge: wrapped around a fold of this radius (m), then flat on top
  | { foldover: number };

/** Gesture lengths are authored at 10 cm framing and scaled to the scene (about 35 cm view). */
const GESTURE_SCALE = 3.2;

const SCRIPTS_10CM: Record<string, Step[]> = {
  // drag in the screen plane through the worn zone, release, regrab and yank the other way
  yank: [
    { grab: [0.003, -0.002] },
    { to: [0.03, 0.0, 0.004], t: 0.6 },
    { to: [0.045, -0.01, 0.01], t: 0.4 },
    { release: true },
    { wait: 0.5 },
    { grab: [-0.008, 0.004] },
    { to: [-0.04, 0.02, 0.006], t: 0.5 },
    { to: [-0.04, -0.03, 0.02], t: 0.5 },
    { release: true },
    { wait: 0.6 },
  ],
  // repeated tugs at one spot, out of plane
  tug: [
    { grab: [0.006, 0.0] },
    { to: [0.0, 0.0, 0.02], t: 0.4 },
    { to: [0.0, 0.0, 0.002], t: 0.3 },
    { to: [0.01, 0.0, 0.03], t: 0.4 },
    { release: true },
    { wait: 0.5 },
  ],
  // untouched drape: must stay at rest
  rest: [{ wait: Number(process.env.WAIT ?? 1.5) }],
  // a fast sideways drag from GRAB (10 cm units, default worn centre), held and released: does the
  // cloth tear at the grip while pulling?
  tear: [
    { grab: (process.env.GRAB ?? '0.003,-0.002').split(',').map(Number) as [number, number] },
    { to: [-0.045, 0.0, 0.0006], t: 0.5 },
    { wait: 0.4 },
    { release: true },
    { wait: 0.8 },
  ],
  // fixed torn geometry: a flap cut on three sides drops through its opening and rings down
  // (solver and damping comparisons independent of how a tear happens to run)
  flap: [{ cut: [0.0, 0.0], half: 8 }, { wait: 2.5 }],
  // self-contact: a cut flap folded back over its hinge and dropped onto the sheet must rest on it,
  // not fall through
  // debris against the cloth: a piece cut free is laid above intact cloth and falls onto it
  scrap: [{ scrap: [-0.035, 0.03], half: 4, lift: 0.01, shift: 0.03 }, { wait: 1.2 }],
  // self-contact, deterministic: the cut flap is laid folded over the sheet 4 mm above it and drops
  drop: [{ cut: [-0.035, 0.03], half: 8 }, { foldover: 0.0025 }, { wait: 1.5 }],
  fold: [
    { cut: [0.0, 0.0], half: 8 },
    { grab: [0.0, -0.0085] },
    { to: [0.0, 0.02, 0.012], t: 0.6 },
    { to: [0.0, 0.035, 0.006], t: 0.5 },
    { release: true },
    { wait: 1.5 },
  ],
  // one in-plane drag through the worn zone, held
  pull: [
    { grab: [0.003, -0.002] },
    { to: [0.03, 0.0, 0.004], t: 0.6 },
    { wait: 0.4 },
  ],
  // a slow stretch of the worn zone toward the viewer and sideways, held, then released
  open: [
    { grab: [0.004, -0.003] },
    { to: [0.012, -0.006, 0.02], t: 1.2 },
    { wait: 0.6 },
    { to: [0.02, -0.01, 0.03], t: 1.0 },
    { release: true },
    { wait: 0.8 },
  ],
  // intact cloth far from the worn zone
  // strong in-plane pull through the worn zone, held, released and rung down (tear while pulling,
  // not on release; settle time)
  snap: [
    { grab: [0.003, -0.002] },
    { to: [-0.03, 0.0, 0.004], t: 0.6 },
    { wait: 0.3 },
    { release: true },
    { wait: 1.5 },
  ],
  // grab at GRAB and pull by PULL (10 cm units, x,y,z) over 0.8 s, hold, release: a user opening the
  // worn patch they see (cores of the worn cells: tools/gpu/scripts/cores in the notes)
  pluck: [
    { grab: (process.env.GRAB ?? '0.0105,-0.0063').split(',').map(Number) as [number, number] },
    { to: (process.env.PULL ?? '0.012,0,0.012').split(',').map(Number) as [number, number, number], t: 0.8 },
    { wait: 0.3 },
    { release: true },
    { wait: 0.8 },
  ],
  // a user working the worn patch: plucks at each worn-through core and between them, cumulative
  worn: [
    { grab: [0.0105, -0.0063] },
    { to: [0.015, 0.0, 0.01], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [-0.0035, -0.0042] },
    { to: [-0.004, -0.014, 0.012], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.0007, 0.0058] },
    { to: [-0.008, 0.012, 0.014], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.004, -0.005] },
    { to: [0.0, -0.002, 0.02], t: 0.7 },
    { release: true },
    { wait: 0.6 },
  ],
  // the worn session, then a second round of pulls at the opened holes' edges in other directions
  worn2: [
    { grab: [0.0105, -0.0063] },
    { to: [0.015, 0.0, 0.01], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [-0.0035, -0.0042] },
    { to: [-0.004, -0.014, 0.012], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.0007, 0.0058] },
    { to: [-0.008, 0.012, 0.014], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.004, -0.005] },
    { to: [0.0, -0.002, 0.02], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.014, -0.004] },
    { to: [0.012, 0.01, 0.012], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [-0.007, -0.006] },
    { to: [-0.012, 0.004, 0.012], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.003, 0.009] },
    { to: [0.006, 0.016, 0.01], t: 0.7 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.009, -0.009] },
    { to: [0.004, -0.02, 0.012], t: 0.7 },
    { release: true },
    { wait: 0.6 },
  ],
  // a user opening the worn cells one by one: a pinch at each cell's core pulled outward from the
  // cluster (mostly in plane, slightly lifted), released, then a pull at the cluster's middle
  cells: [
    { grab: [-0.0026, 0.0138] },
    { to: [-0.004, 0.014, 0.008], t: 0.9 },
    { release: true },
    { wait: 0.3 },
    { grab: [-0.0088, -0.0048] },
    { to: [-0.014, -0.006, 0.008], t: 0.9 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.0182, -0.0043] },
    { to: [0.014, -0.004, 0.008], t: 0.9 },
    { release: true },
    { wait: 0.3 },
    { grab: [0.0005, 0.0033] },
    { to: [0.01, -0.012, 0.006], t: 0.9 },
    { release: true },
    { wait: 0.3 },
    { grab: [-0.0078, 0.0307] },
    { to: [-0.004, 0.012, 0.006], t: 0.9 },
    { release: true },
    { wait: 0.3 },
    { grab: [-0.0203, 0.0095] },
    { to: [-0.012, 0.0, 0.006], t: 0.9 },
    { release: true },
    { wait: 1.0 },
  ],
  // a hard drag through the worn zone, released, then left to settle for a long time (restlessness)
  settle: [
    { grab: [0.003, -0.002] },
    { to: [0.0, 0.035, 0.012], t: 0.7 },
    { to: [0.01, 0.05, 0.02], t: 0.4 },
    { release: true },
    { wait: Number(process.env.WAIT ?? 3) },
  ],
  // a long, hard drag that rips a large hole with hanging flaps, then a long settle
  rip: [
    { grab: [0.003, -0.002] },
    { to: [0.0, 0.03, 0.01], t: 0.5 },
    { to: [0.0, 0.07, 0.02], t: 0.6 },
    { release: true },
    { wait: Number(process.env.WAIT ?? 4) },
  ],
  // a mouse drag as the product maps it: in-plane targets only (AUTOLIFT=1 adds the worker's drag lift),
  // pulled out from the worn zone, held, then released
  drag: [
    { grab: [0.004, -0.003] },
    { to: [0.025, 0.012, 0], t: 0.8 },
    { wait: 0.5 },
    { release: true },
    { wait: 0.6 },
  ],
  // a user hauling on the cloth: a fast, long drag in the screen plane (AUTOLIFT=1), held at full reach,
  // then released (does the grip hold, where does the cloth fail)
  haul: [
    { grab: (process.env.GRAB ?? '0.004,-0.003').split(',').map(Number) as [number, number] },
    { to: (process.env.PULL ?? '0.06,0.02,0').split(',').map(Number) as [number, number, number], t: 0.5 },
    { wait: Number(process.env.HOLD ?? 1.5) },
    { release: true },
    { wait: 1.0 },
  ],
  // a user playing roughly: hard, fast drags in different directions across the worn zone, each held
  // a moment and released (robustness: rejections, runaway nodes, settling, threads)
  maul: [
    { grab: [0.004, -0.003] },
    { to: [0.05, 0.03, 0], t: 0.4 },
    { wait: 0.5 },
    { release: true },
    { wait: 0.4 },
    { grab: [-0.01, 0.01] },
    { to: [-0.06, 0.03, 0], t: 0.4 },
    { wait: 0.5 },
    { release: true },
    { wait: 0.4 },
    { grab: [0.012, -0.012] },
    { to: [0.0, -0.06, 0], t: 0.4 },
    { wait: 0.5 },
    { release: true },
    { wait: 0.4 },
    { grab: [-0.004, -0.004] },
    { to: [0.03, -0.05, 0], t: 0.3 },
    { to: [-0.03, 0.04, 0], t: 0.5 },
    { release: true },
    { wait: 2.0 },
  ],
  intact: [
    { grab: [-0.035, 0.03] },
    { to: [-0.03, 0.0, 0.015], t: 0.8 },
    { release: true },
    { wait: 0.5 },
  ],
};

const scaleStep = (st: Step): Step =>
  'scrap' in st ? { ...st, scrap: [st.scrap[0] * GESTURE_SCALE, st.scrap[1] * GESTURE_SCALE] } :
  'cut' in st ? { cut: [st.cut[0] * GESTURE_SCALE, st.cut[1] * GESTURE_SCALE], half: st.half } :
  'foldover' in st ? st :
  'grab' in st ? { grab: [st.grab[0] * GESTURE_SCALE, st.grab[1] * GESTURE_SCALE] } : 'to' in st ? { to: [st.to[0] * GESTURE_SCALE, st.to[1] * GESTURE_SCALE, st.to[2] * GESTURE_SCALE], t: st.t } : st;
const SCRIPTS: Record<string, Step[]> = Object.fromEntries(Object.entries(SCRIPTS_10CM).map(([k, v]) => [k, v.map(scaleStep)]));

const H_FRAME = 1 / 30;
const HANDLE_MAX_SPEED = 2.5;

export default async function main(args: string[]): Promise<void> {
  const scriptName = args[0] ?? 'yank';
  const name = args[1] ?? `session-${scriptName}`;
  const every = Number(args[2] ?? 6);
  const script = SCRIPTS[scriptName];
  if (!script) throw new Error(`unknown script ${scriptName}`);
  // GRIP='{"radius":0.008,"maxForce":300}' overrides the grip parameters
  Object.assign(GRIP, JSON.parse(process.env.GRIP ?? '{}') as object);
  const device = await gpuDevice();
  // NOMASK=1: plain bilinear prolongation (comparison with connectivity-aware masking)
  const noMask = process.env.NOMASK ? { patch: (m: string, c: string) => (m === 'multigrid' ? c.replace('      if (ok) {', '      if (true) {') : c) } : {};
  const { solver, scene } = await settledSolver(device, { ...(JSON.parse(process.env.SOLVER ?? '{}') as object), ...noMask });
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (pass) => interaction.encodeSubstep(pass);
  solver.preAssemble = (dispatch) => interaction.encode(dispatch);
  const topology = process.env.NO_TOPOLOGY ? null : new Topology(device, solver, scene);
  let events = 0;
  const format: GPUTextureFormat = 'rgba8unorm';
  const [W, Hh] = (process.env.SIZE ?? "960x600").split("x").map(Number);
  const renderer = new Renderer(device, format, solver, scene);
  renderer.resize(W, Hh);
  const tex = device.createTexture({ size: [W, Hh], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const snaps = (process.env.SNAPS ?? '').split(',').filter((q) => q.length > 0).map(Number).sort((a, b) => a - b);
  const [snapW, snapH] = (process.env.SNAPSIZE ?? '1320x1624').split('x').map(Number);
  const snapTex = device.createTexture({ size: [snapW, snapH], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const h = solver.params.h;
  const perFrame = Math.round(H_FRAME / h);
  const x0 = new Float32Array(await readBuffer(device, solver.x));
  const N = scene.nodeCount;

  let handle: [number, number, number] | null = null;
  let anchor: [number, number, number] = [0, 0, 0];
  const tiles: Uint8Array[] = [];
  const labels: string[] = [];
  let frame = 0;
  let t = 0;
  const log: string[] = [];

  const grab = (p: [number, number]) => {
    let picked = -1;
    let best = 1e9;
    for (let n = 0; n < N; n++) {
      const d = Math.hypot(lastX[4 * n] - p[0], lastX[4 * n + 1] - p[1]);
      if (d < best && !scene.pinned[n]) {
        best = d;
        picked = n;
      }
    }
    anchor = [lastX[4 * picked], lastX[4 * picked + 1], lastX[4 * picked + 2]];
    grabbed = picked;
    handle = [...anchor];
    grabT = t;
    interaction.attach(picked, anchor);
    console.log(`  grab node ${picked} at ${anchor.map((q) => q.toFixed(4)).join(',')}`);
  };
  let lastX = x0;
  let grabbed = -1;
  // scrap fixture: the piece's centre node (its height against the sheet below it is logged)
  let scrapCentre = -1;
  let grabT = 0;
  let gripLost = false;
  let cutBox: number[] | null = null;

  const runFrame = async (goal: [number, number, number] | null) => {
    // per-frame maximum speed
    const ce = device.createCommandEncoder();
    ce.clearBuffer(solver.counters, 8, 4);
    device.queue.submit([ce.finish()]);
    for (let k = 0; k < perFrame; k++) {
      if (handle && goal) {
        const d = [goal[0] - handle[0], goal[1] - handle[1], goal[2] - handle[2]];
        const len = Math.hypot(d[0], d[1], d[2]);
        const st = HANDLE_MAX_SPEED * h;
        for (let q = 0; q < 3; q++) handle[q] = len <= st ? goal[q] : handle[q] + (d[q] * st) / len;
        interaction.setTarget(handle);
        if (!process.env.NOPINCH) interaction.setPinch(pinchScale(t + k * h - grabT));
      }
      const hunting = process.env.HUNT && t > Number(process.env.HUNT);
      if (hunting) snapshot(device, solver, true);
      if (process.env.LSPROBE && t + k * h > Number(process.env.LSPROBE)) {
        await lineProbe(device, solver, interaction, `t=${(t + (k + 1) * h).toFixed(4)}`);
        continue;
      }
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
      if (hunting && (await hunt(device, solver, `t=${(t + (k + 1) * h).toFixed(4)}`))) {
        snapshot(device, solver, false);
        await replay(device, solver, interaction);
        process.exit(0);
      }
    }
    await drain(device);
    t += perFrame * h;
    const xs = new Float32Array(await readBuffer(device, solver.x));
    const v = new Float32Array(await readBuffer(device, solver.v));
    if (handle && interaction.coreNodes.length > 0 && !gripLost && interaction.coreNodes.every((n) => xs[4 * n + 3] > 0.5)) {
      gripLost = true;
      console.log(`  grip core became debris at t=${(t).toFixed(2)} (the pinched piece tore out)`);
    }
    const c = new Uint32Array(await readBuffer(device, solver.counters, 64));
    lastX = xs;
    if (topology && c[0] + c[1] !== events) {
      events = c[0] + c[1];
      await topology.request(interaction.coreNodes);
    }
    let bad = 0;
    let far = 0;
    let maxDisp = 0;
    let gated = 0;
    for (let n = 0; n < N; n++) {
      const a = xs[4 * n];
      const b = xs[4 * n + 1];
      const z = xs[4 * n + 2];
      if (!Number.isFinite(a + b + z)) {
        bad++;
        continue;
      }
      const d = Math.hypot(a - x0[4 * n], b - x0[4 * n + 1], z - x0[4 * n + 2]);
      maxDisp = Math.max(maxDisp, d);
      if (d > 0.3) far++;
      if (v[4 * n + 3] > solver.params.historyTolerance) gated++;
    }
    if (scrapCentre >= 0) {
      const c = scrapCentre;
      let best = 1e9;
      let zs = 0;
      for (let m = 0; m < N; m++) {
        if (xs[4 * m + 3] > 0.5) continue;
        const d = Math.hypot(xs[4 * m] - xs[4 * c], xs[4 * m + 1] - xs[4 * c + 1]);
        if (d < best) [best, zs] = [d, xs[4 * m + 2]];
      }
      console.log(`  scrap centre x ${(xs[4 * c] * 1e3).toFixed(1)} z ${(xs[4 * c + 2] * 1e3).toFixed(2)} mm v ${Array.from(v.slice(4 * c, 4 * c + 3)).map((q) => q.toFixed(3)).join(',')} (state ${xs[4 * c + 3]}), sheet below ${(zs * 1e3).toFixed(2)} mm at ${(best * 1e3).toFixed(1)} mm in plan`);
    }
    if (process.env.TRACK) {
      if (handle) console.log(`  handle ${handle.map((q) => q.toFixed(4)).join(',')}`);
      for (const tn of process.env.TRACK.split(',').map(Number)) {
        const k = interaction.patchNodes.indexOf(tn);
        console.log(`  node ${tn} x=${Array.from(xs.slice(4 * tn, 4 * tn + 4)).map((q) => q.toFixed(4))} v=${Array.from(v.slice(4 * tn, 4 * tn + 3)).map((q) => q.toFixed(3))} grip=${k >= 0}`);
      }
    }
    if (process.env.STATS) {
      // kinetic energy split: interior cloth, hole-edge cloth (a missing span), debris; fast-node counts
      const sg = new Uint32Array(await readBuffer(device, solver.segs));
      const act = (q: number) => (sg[8 * q + 5] & 3) === 1;
      let keIn = 0;
      let keEdge = 0;
      let keDeb = 0;
      let fastIn = 0;
      let fastEdge = 0;
      let top = -1;
      let topV = 0;
      let topDeg = 0;
      const fastByDeg = [0, 0, 0, 0, 0];
      for (const L of scene.layers) {
        for (let jj = 0; jj < L.ny; jj++) {
          for (let ii = 0; ii < L.nx; ii++) {
            const n = L.nodeOffset + jj * L.nx + ii;
            if (scene.pinned[n] || xs[4 * n + 3] > 1.5) continue;
            const sp = Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]);
            const ke = 0.5 * scene.mass[n] * sp * sp;
            if (xs[4 * n + 3] > 0.5) {
              keDeb += ke;
              continue;
            }
            const deg = (ii < L.nx - 1 && act(2 * n) ? 1 : 0) + (ii > 0 && act(2 * (n - 1)) ? 1 : 0) + (jj < L.ny - 1 && act(2 * n + 1) ? 1 : 0) + (jj > 0 && act(2 * (n - L.nx) + 1) ? 1 : 0);
            const edge = deg < 4;
            if (edge) keEdge += ke;
            else keIn += ke;
            if (sp > 0.02) {
              if (edge) fastEdge++;
              else fastIn++;
            }
            if (sp > 0.02) fastByDeg[deg]++;
            if (sp > topV) {
              topV = sp;
              top = n;
              topDeg = deg;
            }
          }
        }
      }
      console.log(`  KE uJ in=${(keIn * 1e6).toFixed(1)} edge=${(keEdge * 1e6).toFixed(1)} deb=${(keDeb * 1e6).toFixed(1)} fast(>2cm/s) in=${fastIn} edge=${fastEdge} top=${top} deg=${topDeg} v=${topV.toFixed(3)} fastByDegree=${fastByDeg.join('/')}`);
    }
    const line = `t=${t.toFixed(2)} rupt=${c[0]} pull=${c[1]} rejected=${c[12]} freed=${topology?.freedCount ?? 0} vmax=${new Float32Array(c.buffer)[2].toFixed(3)} maxDisp=${(maxDisp * 1e3).toFixed(1)}mm far=${far} nonfinite=${bad} gatedNodes=${gated}`;
    log.push(line);
    if (frame % every === 0 || bad > 0) console.log(line);
    {
      // every frame is rendered (the loose threads' dynamics advance with the presented frames, as in
      // the product); every `every`-th is captured
      const enc = device.createCommandEncoder();
      renderer.encode(enc, { ...DEFAULT_CAMERA, ...(JSON.parse(process.env.CAMERA ?? "{}") as object) }, t, tex.createView());
      device.queue.submit([enc.finish()]);
    }
    if (process.env.FRAMES) {
      // FRAMES=dir: every presented frame at SNAPSIZE / SNAPCAMERA into artifacts/captures/<dir>/
      // (ffmpeg -framerate 30 -i %05d.png for a clip)
      const dir = resolve(CAPTURES, process.env.FRAMES);
      mkdirSync(dir, { recursive: true });
      renderer.resize(snapW, snapH);
      const enc = device.createCommandEncoder();
      renderer.encode(enc, { ...DEFAULT_CAMERA, ...(JSON.parse(process.env.SNAPCAMERA ?? '{}') as object) }, t, snapTex.createView());
      device.queue.submit([enc.finish()]);
      writeFileSync(resolve(dir, `${String(frame).padStart(5, '0')}.png`), encodePng(snapW, snapH, await pixels(device, snapTex)));
      renderer.resize(W, Hh);
    }
    if (snaps.length > 0 && t >= snaps[0] - 1e-6) {
      // SNAPS='1.2,3.4': full-size stills at those times (SNAPSIZE, SNAPCAMERA; reference framing by
      // default), e.g. the cloth while a pull is held
      const st = snaps.shift()!;
      renderer.resize(snapW, snapH);
      const enc = device.createCommandEncoder();
      renderer.encode(enc, { ...DEFAULT_CAMERA, distance: 1.086, ...(JSON.parse(process.env.SNAPCAMERA ?? '{}') as object) }, t, snapTex.createView());
      device.queue.submit([enc.finish()]);
      writeFileSync(resolve(CAPTURES, `${name}-t${st}.png`), encodePng(snapW, snapH, await pixels(device, snapTex)));
      renderer.resize(W, Hh);
    }
    if (frame % every === 0) {
      const px = await pixels(device, tex);
      if (process.env.MOTION && tiles.length > 0) {
        // image change since the previous rendered frame: mean absolute difference and the fraction
        // of pixels that changed visibly (restlessness of the presented image, independent of KE)
        const prev = tiles[tiles.length - 1];
        let sum = 0;
        let moved = 0;
        for (let q = 0; q < px.length; q += 4) {
          const d = Math.abs(px[q] - prev[q]) + Math.abs(px[q + 1] - prev[q + 1]) + Math.abs(px[q + 2] - prev[q + 2]);
          sum += d;
          if (d > 24) moved++;
        }
        console.log(`  MOTION t=${t.toFixed(2)} meanAbs=${(sum / (px.length / 4)).toFixed(3)} changed=${((100 * moved) / (px.length / 4)).toFixed(3)}%`);
      }
      tiles.push(px);
      labels.push(t.toFixed(2));
    }
    frame++;
    return bad;
  };

  // ruptures before the last release, after it, and when the motion settled afterwards
  let rupturesAtRelease = 0;
  let releaseT = -1;
  let settledT = -1;
  const settleProbe = async () => {
    const c = new Uint32Array(await readBuffer(device, solver.counters, 16));
    const vmax = new Float32Array(c.buffer)[2];
    if (releaseT >= 0 && settledT < 0 && vmax < 0.02) settledT = t;
    if (releaseT >= 0 && vmax >= 0.02) settledT = -1;
    return c[0];
  };
  const baseRun = runFrame;
  const runFrameS = async (goal: [number, number, number] | null) => {
    const r = await baseRun(goal);
    await settleProbe();
    return r;
  };
  for (const s of script) {
    if ('foldover' in s) {
      const L = scene.layers[0];
      const [ia, ib, ja, jb] = cutBox!;
      const xs = new Float32Array(await readBuffer(device, solver.x));
      // each column wraps around a cylinder of radius R at its hinge node, then lies flat on top
      // (span lengths preserved; arc length m L along the flap)
      const R = s.foldover;
      for (let i = ia; i <= ib; i++) {
        const h = jb * L.nx + i;
        const b1 = (jb - 1) * L.nx + i;
        const e = [0, 1, 2].map((q) => xs[4 * b1 + q] - xs[4 * h + q]);
        e[2] = 0;
        const el = Math.hypot(e[0], e[1]);
        const Lr = el;
        for (let q = 0; q < 2; q++) e[q] /= el;
        for (let j = ja; j < jb; j++) {
          const n = j * L.nx + i;
          const arc = (jb - j) * Lr;
          let along: number;
          let up: number;
          if (arc <= Math.PI * R) {
            along = R * Math.sin(arc / R);
            up = R * (1 - Math.cos(arc / R));
          } else {
            along = -(arc - Math.PI * R);
            up = 2 * R;
          }
          xs[4 * n] = xs[4 * h] + e[0] * along;
          xs[4 * n + 1] = xs[4 * h + 1] + e[1] * along;
          xs[4 * n + 2] = xs[4 * h + 2] + up;
        }
      }
      solver.setState(xs);
      lastX = xs;
    } else if ('scrap' in s) {
      let at = 0;
      let best = 1e9;
      for (let n = 0; n < N; n++) {
        const d = Math.hypot(lastX[4 * n] - s.scrap[0], lastX[4 * n + 1] - s.scrap[1]);
        if (d < best) [best, at] = [d, n];
      }
      const L = scene.layers[0];
      const i0 = at % L.nx;
      const j0 = Math.floor(at / L.nx);
      const a = s.half;
      // cut every yarn of the bordering spans in the field too, so the renderer sees the piece separate
      const sg = new Uint32Array(await readBuffer(device, solver.segs));
      const fw = new Uint32Array(await readBuffer(device, solver.field));
      const F = scene.field[0];
      const cutSpan = (i: number, j: number, f: number) => {
        sg[8 * (2 * (j * L.nx + i) + f) + 5] |= 2;
        sg[8 * (2 * (j * L.nx + i) + f) + 3] = 0xffffffff;
        const P = f === 1 ? F.bv : F.bu;
        const B = f === 1 ? F.bu : F.bv;
        for (let q = 0; q < B; q++) {
          for (let p = 0; p < P; p++) {
            const c = f === 1 ? F.offset + (j * F.bv + Math.floor(F.bv / 2) + p) * F.K + i * F.bu + q : F.offset + (j * F.bv + q) * F.K + i * F.bu + Math.floor(F.bu / 2) + p;
            fw[c] |= 0xff << (8 * f);
          }
        }
      };
      for (let j = j0 - a; j <= j0 + a; j++) {
        cutSpan(i0 - a - 1, j, 0);
        cutSpan(i0 + a, j, 0);
      }
      for (let i = i0 - a; i <= i0 + a; i++) {
        cutSpan(i, j0 - a - 1, 1);
        cutSpan(i, j0 + a, 1);
      }
      device.queue.writeBuffer(solver.segs, 0, sg);
      device.queue.writeBuffer(solver.field, 0, fw);
      device.queue.writeBuffer(solver.fieldEvents, 0, new Uint32Array([1]));
      const xs = new Float32Array(await readBuffer(device, solver.x));
      for (let j = j0 - a; j <= j0 + a; j++) {
        for (let i = i0 - a; i <= i0 + a; i++) {
          const n = j * L.nx + i;
          xs[4 * n] += s.shift;
          xs[4 * n + 2] += s.lift;
        }
      }
      solver.setState(xs);
      lastX = xs;
      await topology?.request([]);
      cutBox = [i0 - a, i0 + a, j0 - a, j0 + a];
      scrapCentre = at;
    } else if ('cut' in s) {
      let at = 0;
      let best = 1e9;
      for (let n = 0; n < N; n++) {
        const d = Math.hypot(lastX[4 * n] - s.cut[0], lastX[4 * n + 1] - s.cut[1]);
        if (d < best) [best, at] = [d, n];
      }
      const L = scene.layers[0];
      const i0 = at % L.nx;
      const j0 = Math.floor(at / L.nx);
      const sg = new Uint32Array(await readBuffer(device, solver.segs));
      const cutSpan = (i: number, j: number, f: number) => {
        sg[8 * (2 * (j * L.nx + i) + f) + 5] |= 2 | (0xf << 8);
      };
      const a = s.half;
      for (let j = j0 - a; j < j0 + a; j++) {
        cutSpan(i0 - a - 1, j, 0);
        cutSpan(i0 + a, j, 0);
      }
      for (let i = i0 - a; i <= i0 + a; i++) cutSpan(i, j0 - a - 1, 1);
      device.queue.writeBuffer(solver.segs, 0, sg);
      await topology?.request([]);
      cutBox = [i0 - a, i0 + a, j0 - a, j0 + a];
    } else if ('grab' in s) grab(s.grab);
    else if ('release' in s) {
      rupturesAtRelease = (await settleProbe()) ?? 0;
      releaseT = t;
      settledT = -1;
      interaction.release();
      await topology?.request([]);
      handle = null;
      gripLost = false;
    } else if ('wait' in s) {
      for (let q = 0; q < Math.round(s.wait / H_FRAME); q++) await runFrameS(null);
    } else {
      const goal: [number, number, number] = [anchor[0] + s.to[0], anchor[1] + s.to[1], anchor[2] + s.to[2]];
      // the product's drag lift (worker.ts LIFT_PER_DRAG): the handle rises toward the viewer with the
      // in-plane distance dragged from the grab point
      if (process.env.AUTOLIFT) goal[2] += Math.min(0.05, 0.35 * Math.hypot(goal[0] - anchor[0], goal[1] - anchor[1]));
      // the pointer target moves linearly over the step; the handle follows at bounded speed
      const start = handle ? ([...handle] as [number, number, number]) : goal;
      const frames = Math.max(1, Math.round(s.t / H_FRAME));
      for (let q = 1; q <= frames; q++) {
        const a = q / frames;
        await runFrameS([start[0] + (goal[0] - start[0]) * a, start[1] + (goal[1] - start[1]) * a, start[2] + (goal[2] - start[2]) * a]);
      }
    }
  }
  if (process.env.RENDERCOST) {
    // GPU time of a frame of the final (torn) state at the output size, per pass
    const qs = device.createQuerySet({ type: 'timestamp', count: 8 });
    const res = device.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rd = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    for (let k = 0; k < 3; k++) {
      const enc = device.createCommandEncoder();
      renderer.encode(enc, { ...DEFAULT_CAMERA, ...(JSON.parse(process.env.CAMERA ?? '{}') as object) }, t, tex.createView(), qs);
      enc.resolveQuerySet(qs, 0, 8, res, 0);
      enc.copyBufferToBuffer(res, 0, rd, 0, 64);
      device.queue.submit([enc.finish()]);
      await drain(device);
    }
    await rd.mapAsync(GPUMapMode.READ);
    const tq = new BigUint64Array(rd.getMappedRange().slice(0));
    console.log('RENDER ' + ['prep', 'shadow', 'main', 'post'].map((nm, k) => `${nm} ${(Number(tq[2 * k + 1] - tq[2 * k]) / 1e6).toFixed(2)} ms`).join(', '));
  }
  if (process.env.RESTLESS) {
    // nodes still moving at the end: where they are, how they are attached (degree, and the size of
    // the cloth piece reachable from them within 12 spans)
    const xs = new Float32Array(await readBuffer(device, solver.x));
    const v = new Float32Array(await readBuffer(device, solver.v));
    const sg = new Uint32Array(await readBuffer(device, solver.segs));
    const L = scene.layers[0];
    const act = (q: number) => (sg[8 * q + 5] & 3) === 1;
    const nbrs = (n: number) => {
      const i = n % L.nx;
      const j = Math.floor(n / L.nx);
      const o: number[] = [];
      if (i < L.nx - 1 && act(2 * n)) o.push(n + 1);
      if (i > 0 && act(2 * (n - 1))) o.push(n - 1);
      if (j < L.ny - 1 && act(2 * n + 1)) o.push(n + L.nx);
      if (j > 0 && act(2 * (n - L.nx) + 1)) o.push(n - L.nx);
      return o.filter((m) => xs[4 * m + 3] < 0.5);
    };
    const fast: string[] = [];
    let count = 0;
    for (let n = 0; n < N; n++) {
      if (xs[4 * n + 3] > 0.5) continue;
      const sp = Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]);
      if (sp < 0.01) continue;
      count++;
      if (fast.length >= 12) continue;
      const seen = new Set([n]);
      let front = [n];
      for (let d = 0; d < 12 && front.length; d++) {
        const nf: number[] = [];
        for (const a of front) for (const b of nbrs(a)) if (!seen.has(b)) { seen.add(b); nf.push(b); }
        front = nf;
      }
      fast.push(`(${n % L.nx},${Math.floor(n / L.nx)}) v=${(sp * 100).toFixed(1)}cm/s deg=${nbrs(n).length} reach12=${seen.size}`);
    }
    console.log(`RESTLESS ${count} nodes above 1 cm/s`, fast);
  }
  const finalRuptures = new Uint32Array(await readBuffer(device, solver.counters, 16))[0];
  if (cutBox) {
    // flap nodes lying over the sheet (within 2 mm in plan of a sheet node) but below it: passed through
    const L = scene.layers[0];
    const inFlap = (n: number) => {
      const i = n % L.nx;
      const j = Math.floor(n / L.nx);
      return i >= cutBox![0] && i <= cutBox![1] && j >= cutBox![2] && j <= cutBox![3];
    };
    let over = 0;
    let below = 0;
    for (let n = 0; n < N; n++) {
      if (!inFlap(n) || lastX[4 * n + 3] > 0.5) continue;
      let best = 1e9;
      let z = 0;
      for (let m = 0; m < N; m++) {
        if (inFlap(m) || lastX[4 * m + 3] > 0.5) continue;
        const d = Math.hypot(lastX[4 * m] - lastX[4 * n], lastX[4 * m + 1] - lastX[4 * n + 1]);
        if (d < best) [best, z] = [d, lastX[4 * m + 2]];
      }
      if (best > 0.002) continue;
      over++;
      if (lastX[4 * n + 2] < z - 0.0005) below++;
    }
    console.log(`FOLD flap nodes over the sheet ${over}, of them below it ${below}`);
  }
  {
    // newly broken spans by rest-plane distance from the grabbed node
    const sg = new Uint32Array(await readBuffer(device, solver.segs));
    const bins = [0, 0, 0, 0];
    const g = grabbed >= 0 ? grabbed : 0;
    for (let q = 0; q < 2 * N; q++) {
      if ((sg[8 * q + 5] & 2) === 0 || scene.segFlags[q] & 2) continue;
      const a = q >> 1;
      const d = Math.hypot(scene.restUV[2 * a] - scene.restUV[2 * g], scene.restUV[2 * a + 1] - scene.restUV[2 * g + 1]);
      bins[d < 0.02 ? 0 : d < 0.05 ? 1 : d < 0.1 ? 2 : 3]++;
    }
    console.log(`BROKEN by distance from grip: <2cm ${bins[0]}, 2-5cm ${bins[1]}, 5-10cm ${bins[2]}, >10cm ${bins[3]}`);
  }
  const summary = `SUMMARY ruptures before release ${rupturesAtRelease}, after ${finalRuptures - rupturesAtRelease}; settled (vmax < 2 cm/s) ${settledT >= 0 ? (settledT - releaseT).toFixed(2) + ' s after release' : 'never'}`;
  console.log(summary);
  log.push(summary);
  mkdirSync(CAPTURES, { recursive: true });
  writeFileSync(resolve(CAPTURES, `${name}.log`), log.join('\n'));
  if (process.env.DUMP) {
    // final state for offline analysis (still.ts): x (vec4 per node), segs (8 x 32-bit per span),
    // tails (4 per node), yarn field (u32 per crossing)
    const xb = new Uint8Array(await readBuffer(device, solver.x));
    const sb = new Uint8Array(await readBuffer(device, solver.segs));
    const tb = new Uint8Array(await readBuffer(device, solver.tails));
    const fb = new Uint8Array(await readBuffer(device, solver.field));
    const out = new Uint8Array(xb.length + sb.length + tb.length + fb.length);
    out.set(xb, 0);
    out.set(sb, xb.length);
    out.set(tb, xb.length + sb.length);
    out.set(fb, xb.length + sb.length + tb.length);
    writeFileSync(resolve(CAPTURES, `${name}.state`), out);
  }
  const path = resolve(CAPTURES, `${name}.png`);
  writeFileSync(path, sheet(tiles, W, Hh, 3));
  console.log(path, labels.join(' '));
  device.destroy();
}

async function pixels(device: GPUDevice, tex: GPUTexture): Promise<Uint8Array> {
  const w = tex.width;
  const h = tex.height;
  const bpr = Math.ceil((w * 4) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange());
  const out = new Uint8Array(4 * w * h);
  for (let y = 0; y < h; y++) out.set(src.subarray(y * bpr, y * bpr + 4 * w), 4 * w * y);
  buf.unmap();
  buf.destroy();
  return out;
}

/** Tile frames (cols per row) into one PNG. */
function sheet(tiles: Uint8Array[], w: number, h: number, cols: number): Uint8Array {
  const rows = Math.ceil(tiles.length / cols);
  const W = cols * w;
  const H = rows * h;
  const out = new Uint8Array(4 * W * H);
  tiles.forEach((t, k) => {
    const ox = (k % cols) * w;
    const oy = Math.floor(k / cols) * h;
    for (let y = 0; y < h; y++) out.set(t.subarray(4 * w * y, 4 * w * (y + 1)), 4 * ((oy + y) * W + ox));
  });
  return encodePng(W, H, out);
}

/** Report non-finite entries of the solver state (development: NaN hunting). */
async function hunt(device: GPUDevice, solver: import('../../../src/simulation/gpu/solver').GpuSolver, tag: string): Promise<boolean> {
  const bufs: Array<[string, GPUBuffer]> = [
    ['x', solver.x],
    ['v', solver.v],
    ['dx', solver.dx],
    ['delta', solver.delta],
    ['ext', solver.ext],
    ['pred', solver.pred],
    ['segs', solver.segs],
    ['tails', solver.tails],
    ['p', solver.p],
    ['Ap', solver.Ap],
    ['rec', solver.rec],
    ['scal', solver.scal],
    ['partial', solver.partial],
    ['partial4', solver.partial4],
    ...solver.levels.flatMap((L, k): Array<[string, GPUBuffer]> => [[`op${k}`, L.op], [`X${k}`, L.X], [`B${k}`, L.B], [`R${k}`, L.R]]),
  ];
  const out: string[] = [];
  for (const [nm, b] of bufs) {
    const f = new Float32Array(await readBuffer(device, b));
    let first = -1;
    let cnt = 0;
    for (let q = 0; q < f.length; q++) {
      if (!Number.isFinite(f[q])) {
        if (first < 0) first = q;
        cnt++;
      }
    }
    if (cnt) out.push(`${nm}: ${cnt} (first ${first})`);
  }
  const c = new Float32Array(await readBuffer(device, solver.counters, 48));
  if (out.length) console.log(tag, out.join(', '), 'alpha', c[3], 'updates', Array.from(c.slice(4, 8)));
  return out.length > 0;
}

const saved = new Map<GPUBuffer, GPUBuffer>();
/** Save (or restore) the state a substep starts from. */
function snapshot(device: GPUDevice, solver: import('../../../src/simulation/gpu/solver').GpuSolver, save: boolean): void {
  const enc = device.createCommandEncoder();
  const all = [solver.x, solver.v, solver.dx, solver.delta, solver.ext, solver.pred, solver.segs, solver.tails, solver.p, solver.Ap, solver.rec, solver.scal, solver.partial, solver.partial4, solver.counters, ...solver.levels.flatMap((L) => [L.op, L.X, L.B, L.R])];
  for (const b of all) {
    let c = saved.get(b);
    if (!c) {
      c = device.createBuffer({ size: b.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      saved.set(b, c);
    }
    if (save) enc.copyBufferToBuffer(b, 0, c, 0, b.size);
    else enc.copyBufferToBuffer(c, 0, b, 0, b.size);
  }
  device.queue.submit([enc.finish()]);
}

/** Re-run one substep stage by stage, reporting the first stage that produces non-finite values. */
async function replay(device: GPUDevice, solver: import('../../../src/simulation/gpu/solver').GpuSolver, interaction: Interaction): Promise<void> {
  const s = solver as unknown as Record<string, (...a: unknown[]) => void>;
  const stage = async (name: string, f: (pass: GPUComputePassEncoder) => void) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    f(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
    await hunt(device, solver, name);
  };
  const enc = device.createCommandEncoder();
  enc.clearBuffer(solver.counters, 16, 32);
  device.queue.submit([enc.finish()]);
  await hunt(device, solver, 'restored');
  await stage('predict', (p) => solver.encodePredict(p));
  await stage('gripLead', (p) => interaction.encodeSubstep(p));
  for (let it = 0; it < solver.params.newtonMax; it++) {
    await stage(`grip${it}`, (p) => interaction.encode((pipe, group, g) => { p.setPipeline(pipe); p.setBindGroup(0, group); p.dispatchWorkgroups(g); }));
    await stage(`assemble${it}`, (p) => solver.encodeAssemble(p, it === 0));
    if (it === 0) {
      const mx = async (nm: string, b: GPUBuffer, stride = 4) => {
        const f = new Float32Array(await readBuffer(device, b));
        let m = 0;
        let at = 0;
        for (let q = 0; q < f.length; q++) if (Math.abs(f[q]) > m) [m, at] = [Math.abs(f[q]), q];
        console.log(`  ${nm}: max |v| ${m.toExponential(3)} at ${at} (node ${Math.floor(at / stride)})`);
      };
      await mx('B', solver.levels[0].B);
      await mx('op0', solver.levels[0].op, 72);
      for (let k = 1; k < solver.levels.length; k++) await mx(`op${k}`, solver.levels[k].op, 200);
      await stage('clear', (p) => s.dispatch.call(solver, p, 'clear2', 'clear', Math.ceil(solver.N / 256)));
      await stage('vcycle', (p) => solver.encodeVCycle(p, 0));
      for (let k = 0; k < solver.levels.length; k++) await mx(`X${k}`, solver.levels[k].X);
      const x = new Float32Array(await readBuffer(device, solver.x));
      console.log('  x of node 232', Array.from(x.slice(928, 932)));
    }
    await stage(`pcg${it}`, (p) => solver.encodePCG(p));
    await stage(`line${it}`, (p) => solver.encodeLineSearch(p));
    await stage(`apply${it}`, (p) => s.dispatch.call(solver, p, 'newtonApply', `newtonApply${it}`, Math.ceil(solver.N / 128)));
  }
  await stage('finalize', (p) => solver.encodeFinalize(p));
}

/**
 * One substep run stage by stage (LSPROBE=t0 from that time on): per Newton iteration the directional
 * derivative g.d of the incremental potential along the PCG direction, the evaluated energy changes at
 * steps 1, 1/2, 1/4, 1/8 with their rounding bounds, the chosen step, and the nodes contributing most
 * to an ascent.
 */
async function lineProbe(device: GPUDevice, solver: import('../../../src/simulation/gpu/solver').GpuSolver, interaction: Interaction, tag: string): Promise<void> {
  const s = solver as unknown as Record<string, (...a: unknown[]) => void>;
  const run = (f: (pass: GPUComputePassEncoder) => void) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    f(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
  };
  const c0 = device.createCommandEncoder();
  const contact = solver.contact as unknown as { encodeClear(e: GPUCommandEncoder): void; encodeTerms(p: GPUComputePassEncoder): void; encodeDetect(p: GPUComputePassEncoder): void };
  if (solver.contactEnabled) contact.encodeClear(c0);
  c0.clearBuffer(solver.counters, 16, 32);
  device.queue.submit([c0.finish()]);
  run((p) => {
    solver.encodePredict(p);
    interaction.encodeSubstep(p);
    if (solver.contactEnabled) contact.encodeDetect(p);
  });
  for (let it = 0; it < solver.params.newtonMax; it++) {
    run((p) => {
      if (solver.contactEnabled) contact.encodeTerms(p);
      interaction.encode((pipe, group, g) => {
        p.setPipeline(pipe);
        p.setBindGroup(0, group);
        p.dispatchWorkgroups(g);
      });
      solver.encodeAssemble(p, it === 0);
      solver.encodePCG(p);
      solver.encodeLineSearch(p);
    });
    const B = new Float32Array(await readBuffer(device, solver.levels[0].B));
    const D = new Float32Array(await readBuffer(device, solver.delta));
    const P4 = new Float32Array(await readBuffer(device, solver.partial4));
    const C = new Float32Array(await readBuffer(device, solver.counters, 64));
    const xs = new Float32Array(await readBuffer(device, solver.x));
    let gd = 0;
    const contrib: Array<[number, number]> = [];
    for (let n = 0; n < solver.N; n++) {
      const q = -(B[4 * n] * D[4 * n] + B[4 * n + 1] * D[4 * n + 1] + B[4 * n + 2] * D[4 * n + 2]);
      gd += q;
      if (q > 0) contrib.push([q, n]);
    }
    contrib.sort((a, b) => b[0] - a[0]);
    const e = [0, 0, 0, 0];
    const m = [0, 0, 0, 0];
    for (let w = 0; w < 256; w++) for (let k = 0; k < 4; k++) {
      e[k] += P4[8 * w + k];
      m[k] += P4[8 * w + 4 + k];
    }
    if (e[3] > 4e-6 * m[3]) {
      const dbg = await perNodeEnergy(device, solver);
      const ranked = Array.from({ length: solver.N }, (_, n) => n).sort((a, b) => dbg[8 * b + 3] - dbg[8 * a + 3]).slice(0, 3);
      const xs2 = new Float32Array(await readBuffer(device, solver.x));
      const dx2 = new Float32Array(await readBuffer(device, solver.dx));
      const sg = new Float32Array(await readBuffer(device, solver.segs));
      const sgu = new Uint32Array(sg.buffer);
      const L = solver.scene.layers[0];
      for (const n of ranked) {
        const spans = [0, 1].map((f) => `f${f} rest=${(sg[8 * (2 * n + f)] * 1e3).toFixed(3)}mm intact=${sg[8 * (2 * n + f) + 1].toFixed(3)} flags=${sgu[8 * (2 * n + f) + 5]}`).join(' ');
        console.log(`    node ${n} (i=${n % L.nx} j=${Math.floor(n / L.nx)}) w=${xs2[4 * n + 3]} dE=[${Array.from(dbg.slice(8 * n, 8 * n + 4)).map((v) => v.toExponential(2)).join(' ')}] cat(ax,visc,bend,shear)=[${Array.from(dbg.slice(8 * n + 4, 8 * n + 8)).map((v) => v.toExponential(2)).join(' ')}] dx=${(Math.hypot(dx2[4 * n], dx2[4 * n + 1], dx2[4 * n + 2]) * 1e3).toFixed(3)}mm |d|=${Math.hypot(D[4 * n], D[4 * n + 1], D[4 * n + 2]).toExponential(2)} ${spans}`);
      }
    }
    const top = contrib.slice(0, 4).map(([q, n]) => `${n}(${q.toExponential(1)} w=${xs[4 * n + 3]})`).join(' ');
    console.log(`  LS ${tag} it${it} g.d=${gd.toExponential(3)} e=[${e.map((v) => v.toExponential(2)).join(' ')}] tol=[${m.map((v) => (4e-6 * v).toExponential(1)).join(' ')}] alpha=${C[3]} | ascent nodes ${top}`);
    run((p) => s.dispatch.call(solver, p, 'newtonApply', `newtonApply${it}`, Math.ceil(solver.N / 128)));
  }
  run((p) => solver.encodeFinalize(p));
  await device.queue.onSubmittedWorkDone();
}

let dbgMod: { pipe: GPUComputePipeline; group: GPUBindGroup; buf: GPUBuffer } | null = null;
/** Per-node energy changes of the line search (energy.wgsl nodeDelta) at the current Newton iterate. */
async function perNodeEnergy(device: GPUDevice, solver: import('../../../src/simulation/gpu/solver').GpuSolver): Promise<Float32Array> {
  if (!dbgMod) {
    const { ComputeModule, createBuffer } = await import('../../../src/gpu/util');
    const common = (await import('../../../src/simulation/gpu/wgsl/common.wgsl?raw')).default;
    const drag = (await import('../../../src/simulation/gpu/wgsl/drag.wgsl?raw')).default;
    const energy = (await import('../../../src/simulation/gpu/wgsl/energy.wgsl?raw')).default;
    // per category (axial elastic, axial viscous, bending, shear): the change at the shortest step
    const tag = (src: string, c: number) => {
      const at = en.indexOf(src);
      if (at < 0) throw new Error(`energy probe: ${src} not found`);
      const inner = src.slice('add1(k, '.length, -2);
      en = en.replace(src, `{ let pv = ${inner}; add1(k, pv); if (k == 3u) { cat[${c}] += pv; } }`);
    };
    let en = energy.replace('var<private> mag: vec4<f32>;', 'var<private> mag: vec4<f32>;\nvar<private> cat: vec4<f32>;');
    tag('add1(k, 0.5 * (T0 + Ta) * r.x);', 0);
    tag('add1(k, 0.5 * c * r.x * (r.y + l0 - 2.0 * ln));', 1);
    tag('add1(k, 0.5 * ke * bendPsiDelta(q0, q0 + dq, dq, ab));', 2);
    tag('add1(k, 0.5 * (f0 + shearLaw(ks, jam, Ld, rr.y).x) * rr.x);', 3);
    const code = common + drag + en + `
@group(0) @binding(10) var<storage, read_write> dbg: array<vec4<f32>>;
@compute @workgroup_size(128)
fn perNode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.counts.x) { return; }
  cat = vec4<f32>(0.0);
  nodeDelta(gid.x);
  dbg[2u * gid.x] = acc;
  dbg[2u * gid.x + 1u] = cat;
}`;
    const mod = new ComputeModule(device, 'energyDbg', code);
    const buf = createBuffer(device, 'dbg', 32 * solver.N);
    const group = mod.bindGroup('dbg', { 0: solver.simParams, 1: solver.nodes, 2: solver.x, 3: solver.dx, 4: solver.pred, 5: solver.segs, 6: solver.delta, 7: solver.ext, 8: solver.partial4, 9: solver.counters, 10: buf });
    dbgMod = { pipe: mod.pipeline('perNode'), group, buf };
  }
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(dbgMod.pipe);
  pass.setBindGroup(0, dbgMod.group);
  pass.dispatchWorkgroups(Math.ceil(solver.N / 128));
  pass.end();
  device.queue.submit([enc.finish()]);
  return new Float32Array(await readBuffer(device, dbgMod.buf));
}
