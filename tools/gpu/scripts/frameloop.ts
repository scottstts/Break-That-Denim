/**
 * Wall time of the product's frame loop (development): per presented frame, SUB substeps (default 4,
 * natural time at 30 frames/s with h = 1/120) and one render at W x H, submitted as the worker does.
 * Reports the median frame time and the simulation-time / wall-time ratio it allows at 30 frames/s.
 *   SUB=4 node tools/gpu/run.mjs tools/gpu/scripts/frameloop.ts [width height]
 */
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { gpuDevice } from '../env';
import { settledSolver } from './render';

export default async function main(args: string[]): Promise<void> {
  const width = Number(args[0] ?? 1600);
  const height = Number(args[1] ?? 1000);
  const sub = Number(process.env.SUB ?? 4);
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device, JSON.parse(process.env.SOLVER ?? '{}') as object);
  const renderer = new Renderer(device, 'rgba8unorm', solver, scene);
  renderer.resize(width, height);
  const tex = device.createTexture({ size: [width, height], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const times: number[] = [];
  for (let f = 0; f < 40; f++) {
    const t0 = performance.now();
    for (let k = 0; k < sub; k++) {
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
    }
    const enc = device.createCommandEncoder();
    renderer.encode(enc, DEFAULT_CAMERA, f / 30, tex.createView());
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    if (f >= 5) times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  const med = times[times.length >> 1];
  const h = solver.params.h;
  console.log(`frame ${med.toFixed(1)} ms (${sub} substeps + render): ${(1000 / med).toFixed(1)} frames/s, sim/wall ${((sub * h * 1000) / Math.max(med, 1000 / 30)).toFixed(2)} at <= 30 frames/s`);
  device.destroy();
}
