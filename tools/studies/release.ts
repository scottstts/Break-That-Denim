/**
 * Grab, stretch, hold, release: how the held material recoils. The product's picker and hand
 * (tools/gpu/grab.ts) grab the cloth over world point GRAB, drag it by PULL (m, from the grab point)
 * over PULLT seconds, hold for HOLD seconds and release; the run continues AFTER seconds.
 *
 * Every substep after the release it records the held material's centroid (grip core) along the
 * recoil direction u (from where it finally rests to where it was released): s(t) = (c(t) - c_end).u.
 * Overshoot is how far it swings past rest as a fraction of s(0); rings counts sign changes beyond 2 %
 * of s(0); settle is the time after which |s| stays within 5 % of s(0). Also the cloth's kinetic
 * energy and largest speed (live, unpinned nodes), ruptures and line-search rejections.
 *
 *   GRAB=-0.04,0.05 PULL=0.03,0,0 node tools/gpu/run.mjs tools/studies/release.ts [label]
 *   SOLVER='{"newtonIterations":6,"newtonMax":6,"pcgIterations":4}'  GRIP='{"pinch":1}'  TRACE=1
 */
import { readBuffer } from '../../src/gpu/util';
import { GRIP } from '../../src/simulation/grip';
import { Topology } from '../../src/simulation/gpu/topology';
import { Interaction } from '../../src/simulation/gpu/interaction';
import { settledSolver } from '../gpu/scripts/render';
import { drain, gpuDevice } from '../gpu/env';
import { grabAt } from '../gpu/grab';

const nums = (s: string) => s.split(',').map(Number) as [number, number, number];

export default async function main(args: string[]): Promise<void> {
  const label = args[0] ?? 'release';
  Object.assign(GRIP, JSON.parse(process.env.GRIP ?? '{}') as object);
  const grabXY = nums(process.env.GRAB ?? '-0.04,0.05');
  const pull = nums(process.env.PULL ?? '0.03,0,0');
  const pullT = Number(process.env.PULLT ?? 0.5);
  const hold = Number(process.env.HOLD ?? 0.4);
  const after = Number(process.env.AFTER ?? 1.0);
  // RAMP=s: the fingers open over this time (hold falls linearly to zero), then the grip ends
  const ramp = Number(process.env.RAMP ?? 0);
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device, JSON.parse(process.env.SOLVER ?? '{}') as object);
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (p) => interaction.encodeSubstep(p);
  solver.preAssemble = (d) => interaction.encode(d);
  const topology = new Topology(device, solver, scene, () => interaction.coreNodes);
  await topology.request();
  const N = scene.nodeCount;
  const h = solver.params.h;
  const rampSteps = Math.round(ramp / h);
  // ITERS=n over WINDOW s from the release (default the ramp): event-targeted Newton iterations
  const iters = Number(process.env.ITERS ?? 0);
  // MAXITERS=n: further iterations up to n while the last update exceeds TOL (m), decided on the GPU
  const maxIters = Number(process.env.MAXITERS ?? 0);
  if (process.env.TOL) solver.setNewtonTolerance(Number(process.env.TOL));
  let executed = 0;
  let refinedSteps = 0;
  const iterSteps = Math.round(Number(process.env.WINDOW ?? ramp) / h);
  let x = new Float32Array(await readBuffer(device, solver.x));
  let near = 0;
  let best = Infinity;
  for (let n = 0; n < N; n++) {
    const d = Math.hypot(x[4 * n] - grabXY[0], x[4 * n + 1] - grabXY[1]);
    if (d < best && x[4 * n + 3] < 0.5) [best, near] = [d, n];
  }
  const g = await grabAt(interaction, [grabXY[0], grabXY[1], x[4 * near + 2]], 0);
  if (!g) throw new Error('nothing to grab there');
  const core = [...interaction.core(0)];
  const centroid = (xs: Float32Array) => {
    const c = [0, 0, 0];
    for (const n of core) for (let q = 0; q < 3; q++) c[q] += xs[4 * n + q] / core.length;
    return c;
  };
  const holdSteps = Math.round((pullT + hold) / h);
  const total = holdSteps + Math.round(after / h);
  const trace: { t: number; c: number[]; ke: number; vmax: number }[] = [];
  for (let k = 0; k < total; k++) {
    if (k === holdSteps && process.env.STRAIN) {
      // strain of the spans inside the held patch just before the release
      const xs = new Float32Array(await readBuffer(device, solver.x));
      const sg = new Float32Array(await readBuffer(device, solver.segs));
      const nx = scene.layers[0].nx;
      const inPatch = new Set(interaction.patchNodes);
      const strains: number[] = [];
      for (const n of inPatch) {
        for (let f = 0; f < 2; f++) {
          const b = f === 0 ? n + 1 : n + nx;
          if (!inPatch.has(b)) continue;
          const l = Math.hypot(xs[4 * b] - xs[4 * n], xs[4 * b + 1] - xs[4 * n + 1], xs[4 * b + 2] - xs[4 * n + 2]);
          strains.push(l / sg[8 * (2 * n + f)] - 1);
        }
      }
      strains.sort((a, b) => a - b);
      console.log(`  patch span strain: min ${strains[0].toFixed(3)} median ${strains[strains.length >> 1].toFixed(3)} max ${strains[strains.length - 1].toFixed(3)} (${strains.length} spans)`);
    }
    if (k >= holdSteps && k < holdSteps + rampSteps) interaction.setHold(1 - (k - holdSteps + 1) / (rampSteps + 1));
    if (k === holdSteps + rampSteps) {
      interaction.release(0);
      void topology.request();
    }
    if (k < holdSteps) {
      const a = Math.min(1, ((k + 1) * h) / pullT);
      g.hand.step(h, [a * pull[0], a * pull[1], a * pull[2]]);
      interaction.setTarget(g.hand.target);
      interaction.setPinch(g.hand.pinch);
    }
    const enc = device.createCommandEncoder();
    const refined = k >= holdSteps && k < holdSteps + iterSteps;
    solver.encodeSubstep(enc, refined ? iters : 0, refined ? maxIters : 0);
    device.queue.submit([enc.finish()]);
    // every substep from just before the release, every 4th before that
    if (k < holdSteps - 1 && k % 4 !== 3) continue;
    await drain(device);
    x = new Float32Array(await readBuffer(device, solver.x));
    const v = new Float32Array(await readBuffer(device, solver.v));
    if (refined) {
      // Newton iterations this substep actually ran (their update slots are non-zero)
      const cf = new Float32Array(await readBuffer(device, solver.counters, 64));
      for (let it = 0; it < 8; it++) if (cf[4 + it] > 0) executed++;
      refinedSteps++;
    }
    let ke = 0;
    let vmax = 0;
    for (let n = 0; n < N; n++) {
      if (x[4 * n + 3] > 0.5 || scene.pinned[n]) continue;
      const s2 = v[4 * n] ** 2 + v[4 * n + 1] ** 2 + v[4 * n + 2] ** 2;
      ke += 0.5 * scene.mass[n] * s2;
      vmax = Math.max(vmax, Math.sqrt(s2));
    }
    trace.push({ t: (k + 1) * h - (pullT + hold), c: centroid(x), ke, vmax });
  }
  const counters = new Uint32Array(await readBuffer(device, solver.counters, 64));
  const end = trace[trace.length - 1].c;
  const rel = trace.find((r) => r.t >= -1e-9)!;
  const u = [0, 1, 2].map((q) => rel.c[q] - end[q]);
  const s0 = Math.hypot(...u);
  const post = trace.filter((r) => r.t >= 0);
  const s = post.map((r) => [0, 1, 2].reduce((a, q) => a + ((r.c[q] - end[q]) * u[q]) / s0, 0));
  let rings = 0;
  let sign = 1;
  for (const q of s) {
    if (Math.abs(q) < 0.02 * s0) continue;
    const sg = Math.sign(q);
    if (sg !== sign) rings++;
    sign = sg;
  }
  let settle = 0;
  for (let k = 0; k < s.length; k++) if (Math.abs(s[k]) > 0.05 * s0) settle = post[k].t;
  const before = trace.filter((r) => r.t < 0);
  // kinetic energy regained after the first trough of the recoil (a bounce), as a share of its peak
  let peak = 0;
  let trough = Infinity;
  let rebound = 0;
  let phase = 0;
  for (const r of post) {
    if (phase === 0) {
      if (r.ke >= peak) peak = r.ke;
      else phase = 1;
    }
    if (phase === 1) {
      if (r.ke <= trough) trough = r.ke;
      else phase = 2;
    }
    if (phase === 2) rebound = Math.max(rebound, r.ke - trough);
  }
  if (process.env.TRACE) for (let k = 0; k < post.length; k += 2) console.log(`  t=${post[k].t.toFixed(4)} s=${(s[k] * 1e3).toFixed(2)}mm ke=${(post[k].ke * 1e6).toFixed(2)}uJ vmax=${post[k].vmax.toFixed(3)}`);
  const out = {
    label,
    node: g.node,
    wear: scene.wear[g.node],
    releasedFromRestMm: s0 * 1e3,
    overshoot: -Math.min(...s) / s0,
    rings,
    settleS: settle,
    holdKEuJ: before[before.length - 1].ke * 1e6,
    postPeakKEuJ: Math.max(...post.map((r) => r.ke)) * 1e6,
    postPeakSpeed: Math.max(...post.map((r) => r.vmax)),
    rebound: rebound / Math.max(peak, 1e-30),
    meanIterations: refinedSteps > 0 ? executed / refinedSteps : 0,
    ruptures: counters[0],
    pulledOut: counters[1],
    rejected: counters[12],
  };
  console.log(JSON.stringify(out));
  device.destroy();
}
