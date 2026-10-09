/** Precompute the settled default drape as a static asset (public/drape.bin). */
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../../src/gpu/util';
import { encodeDrape, settleDrape } from '../../../src/simulation/gpu/settle';
import { GpuSolver, GPU_DEFAULTS } from '../../../src/simulation/gpu/solver';
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { gpuDevice, ROOT } from '../env';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const scene = compileScene(defaultSceneSpecs());
  const solver = new GpuSolver(device, scene, GPU_DEFAULTS);
  await settleDrape(device, solver);
  const x = new Float32Array(await readBuffer(device, solver.x));
  const c = new Float32Array(await readBuffer(device, solver.counters, 16));
  const file = resolve(ROOT, 'public/drape.bin');
  writeFileSync(file, encodeDrape(solver, x));
  console.log(`wrote ${file} (${scene.nodeCount} nodes); speed at end ${c[2]}`);
  device.destroy();
}
