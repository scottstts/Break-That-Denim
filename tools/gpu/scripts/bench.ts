/** Substep cost of the default scene with the production schedule (throttled; development only). */
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { GpuSolver, GPU_DEFAULTS } from '../../../src/simulation/gpu/solver';
import { drain, gpuDevice } from '../env';

async function time(device: GPUDevice, solver: GpuSolver, reps = 10): Promise<number> {
  const times: number[] = [];
  for (let k = 0; k < reps; k++) {
    const t0 = performance.now();
    const enc = device.createCommandEncoder();
    solver.encodeSubstep(enc);
    device.queue.submit([enc.finish()]);
    times.push(await drain(device, t0));
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(reps / 2)];
}

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const scene = compileScene(defaultSceneSpecs());
  for (const tol of [1e9, 0]) {
    const solver = new GpuSolver(device, scene, { ...GPU_DEFAULTS, newtonTolerance: tol });
    await time(device, solver, 2);
    const levels = solver.levels.map((l) => `${l.layers[0].nx}x${l.layers[0].ny}`).join(' ');
    console.log(`nodes ${scene.nodeCount} levels ${levels}  ${tol > 1 ? 'mandatory Newton only' : 'all optional Newton on'}: ${(await time(device, solver)).toFixed(2)} ms/substep`);
  }
  device.destroy();
}
