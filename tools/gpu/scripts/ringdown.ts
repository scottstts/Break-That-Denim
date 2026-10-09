/** After a grab-pull-release, track the max nodal speed per 0.1 s window (sleep criterion check). */
import { readBuffer } from '../../../src/gpu/util';
import { Interaction } from '../../../src/simulation/gpu/interaction';
import { GPU_DEFAULTS, type GpuSolverParams } from '../../../src/simulation/gpu/solver';
import { drain, gpuDevice } from '../env';
import { settledSolver } from './render';

export default async function main(args: string[]): Promise<void> {
  const device = await gpuDevice();
  const { solver: base, scene } = await settledSolver(device);
  const deep = args[0] === 'deep' || args[0] === 'conv';
  const solver = deep ? new (base.constructor as typeof import('../../../src/simulation/gpu/solver').GpuSolver)(device, scene, { ...GPU_DEFAULTS, maxCoarsestNodes: 48, coarseSweeps: 6, ...(args[0] === 'conv' ? { newtonIterations: 6, newtonMax: 6, pcgIterations: 4 } : {}) }) : base;
  if (deep) {
    const enc0 = device.createCommandEncoder();
    enc0.copyBufferToBuffer(base.x, 0, solver.x, 0, base.x.size);
    device.queue.submit([enc0.finish()]);
  }
  const variant: Partial<GpuSolverParams> = args[0] === 'tight' ? { newtonMax: 4, newtonTolerance: 1e-6 } : {};
  Object.assign(solver.params, variant);
  if (args[1] === 'nohist') solver.setHistoryEnabled(false);
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (pass) => interaction.encodeSubstep(pass);
  solver.preAssemble = (dispatch) => interaction.encode(dispatch);
  const xs = new Float32Array(await readBuffer(device, solver.x));
  let picked = 0;
  let best = 1e9;
  for (let n = 0; n < scene.nodeCount; n++) {
    const d = Math.hypot(xs[4 * n] + 0.03, xs[4 * n + 1] - 0.02);
    if (d < best) {
      best = d;
      picked = n;
    }
  }
  const p0: [number, number, number] = [xs[4 * picked], xs[4 * picked + 1], xs[4 * picked + 2]];
  interaction.attach(picked, p0);
  const h = GPU_DEFAULTS.h;
  const out: string[] = [];
  for (let k = 0; k < 120 * 4; k++) {
    const t = k * h;
    if (t < 0.25) interaction.setTarget([p0[0], p0[1], p0[2] + 0.04 * t]);
    if (Math.abs(t - 0.5) < h / 2) interaction.release();
    const enc = device.createCommandEncoder();
    if (k % 12 === 0) enc.clearBuffer(solver.counters, 8, 4);
    solver.encodeSubstep(enc);
    device.queue.submit([enc.finish()]);
    if (k % 12 === 11) {
      await drain(device);
      const c = new Float32Array(await readBuffer(device, solver.counters, 16));
      out.push(`${((k + 1) * h).toFixed(1)}:${(c[2] * 1e3).toFixed(2)}`);
    }
  }
  const v = new Float32Array(await readBuffer(device, solver.v));
  const x = new Float32Array(await readBuffer(device, solver.x));
  const L = scene.layers[0];
  const top: Array<[number, number]> = [];
  for (let n = 0; n < scene.nodeCount; n++) top.push([Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]), n]);
  top.sort((a, b) => b[0] - a[0]);
  for (const [, n] of top.slice(0, 6)) {
    console.log(`node (${n % L.nx},${Math.floor(n / L.nx)}) v (${(v[4 * n] * 1e3).toFixed(2)}, ${(v[4 * n + 1] * 1e3).toFixed(2)}, ${(v[4 * n + 2] * 1e3).toFixed(2)}) mm/s at (${(x[4 * n] * 1e3).toFixed(1)}, ${(x[4 * n + 1] * 1e3).toFixed(1)}, ${(x[4 * n + 2] * 1e3).toFixed(1)}) mm, wear ${scene.wear[n].toFixed(2)}`);
  }
  let vs = 0;
  for (let n = 0; n < scene.nodeCount; n++) vs += Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]);
  console.log('mean speed mm/s', ((vs / scene.nodeCount) * 1e3).toFixed(3));
  console.log(`${args[0] ?? 'prod'} max speed mm/s per 0.1 s (release at 0.5 s):\n${out.join(' ')}`);
  device.destroy();
}
