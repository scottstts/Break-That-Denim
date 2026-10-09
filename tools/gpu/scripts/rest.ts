/** Rest-state study: settle the default drape and track max nodal speed per 0.1 s window. */
import { readBuffer } from '../../../src/gpu/util';
import { GpuSolver, GPU_DEFAULTS } from '../../../src/simulation/gpu/solver';
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { drain, gpuDevice } from '../env';

export default async function main(args: string[]): Promise<void> {
  const damp = args[0] !== 'nodamp';
  const device = await gpuDevice();
  const scene = compileScene(defaultSceneSpecs());
  const conv = args[1] === 'conv';
  const solver = new GpuSolver(device, scene, conv ? { ...GPU_DEFAULTS, newtonIterations: 8, newtonMax: 8, pcgIterations: 4, maxCoarsestNodes: 48, coarseSweeps: 6 } : { ...GPU_DEFAULTS });
  solver.setHistoryEnabled(false);
  const windows: string[] = [];
  const steps = 12 * 50; // 5 s
  for (let k = 0; k < steps; k++) {
    const enc = device.createCommandEncoder();
    if (k % 12 === 0) enc.clearBuffer(solver.counters, 8, 4);
    solver.encodeSubstep(enc);
    if (damp && k % 12 === 11 && k < 360) enc.clearBuffer(solver.v);
    device.queue.submit([enc.finish()]);
    if (k % 12 === 11) {
      await drain(device);
      const c = new Float32Array(await readBuffer(device, solver.counters, 16));
      windows.push(`${((k + 1) / 120).toFixed(1)}s:${(c[2] * 1e3).toFixed(2)}`);
    }
  }
  console.log(`max speed (mm/s) per 0.1 s window, kinetic damping ${damp ? 'until 3 s' : 'off'}:\n` + windows.join(' '));
  device.destroy();
}
