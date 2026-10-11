/**
 * Substep cost on a moving sheet (development): the settled drape is given a transverse wave
 * (0.3 m/s amplitude, 4 cm wavelength) before each batch of four substeps, so contact detection works
 * on moving nodes as during a drag. Median batch time per substep.
 */
import { readBuffer } from '../../src/gpu/util';
import { settledSolver } from '../gpu/scripts/render';
import { drain, gpuDevice } from '../gpu/env';
import { FLOOR_Z } from '../../src/simulation/floor';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const { solver } = await settledSolver(device);
  // FLOOR=1: with the floor collision free play has
  if (process.env.FLOOR) solver.setFloor(FLOOR_Z);
  const x0 = new Float32Array(await readBuffer(device, solver.x));
  const v = new Float32Array(x0.length);
  for (let n = 0; n < solver.N; n++) v[4 * n + 2] = 0.3 * Math.sin((2 * Math.PI * x0[4 * n]) / 0.04);
  const times: number[] = [];
  for (let r = 0; r < 14; r++) {
    solver.setState(x0, v);
    await drain(device);
    const t0 = performance.now();
    for (let b = 0; b < 4; b++) {
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
    }
    times.push((await drain(device, t0)) / 4);
  }
  times.sort((a, b) => a - b);
  console.log(`moving sheet: ${times[7].toFixed(2)} ms/substep`);
  device.destroy();
}
