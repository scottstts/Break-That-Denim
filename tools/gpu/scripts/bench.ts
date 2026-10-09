/**
 * Substep cost of the default scene (development only). Substeps are submitted back to back in
 * batches of four, as the worker does per presented frame; the median batch time is reported.
 *   SOLVER='{"pcgIterations":2}' node tools/gpu/run.mjs tools/gpu/scripts/bench.ts
 */
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { GpuSolver, GPU_DEFAULTS } from '../../../src/simulation/gpu/solver';
import { drain, gpuDevice } from '../env';

const BATCH = 4;

async function time(device: GPUDevice, solver: GpuSolver, reps = 12): Promise<number> {
  const times: number[] = [];
  for (let k = 0; k < reps; k++) {
    const t0 = performance.now();
    for (let b = 0; b < BATCH; b++) {
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
    }
    times.push((await drain(device, t0)) / BATCH);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(reps / 2)];
}

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const scene = compileScene(defaultSceneSpecs());
  const solver = new GpuSolver(device, scene, { ...GPU_DEFAULTS, ...(JSON.parse(process.env.SOLVER ?? '{}') as object) });
  await time(device, solver, 2);
  const levels = solver.levels.map((l) => `${l.layers[0].nx}x${l.layers[0].ny}`).join(' ');
  console.log(`nodes ${scene.nodeCount} levels ${levels}: ${(await time(device, solver)).toFixed(2)} ms/substep`);
  device.destroy();
}
