/** GPU time of one rendered frame (timestamp queries around the frame's passes). */
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { drain, gpuDevice } from '../env';
import { settledSolver } from './render';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const renderer = new Renderer(device, 'rgba8unorm', solver, scene);
  const [w, h] = [1789, 1117];
  renderer.resize(w, h);
  const tex = device.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const times: number[] = [];
  for (let k = 0; k < 8; k++) {
    const t0 = performance.now();
    const enc = device.createCommandEncoder();
    renderer.encode(enc, DEFAULT_CAMERA, 0, tex.createView());
    device.queue.submit([enc.finish()]);
    times.push(await drain(device, t0));
  }
  times.sort((a, b) => a - b);
  const qs = device.createQuerySet({ type: 'timestamp', count: 8 });
  const res = device.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const rd = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  renderer.encode(enc, DEFAULT_CAMERA, 0, tex.createView(), qs);
  enc.resolveQuerySet(qs, 0, 8, res, 0);
  enc.copyBufferToBuffer(res, 0, rd, 0, 64);
  device.queue.submit([enc.finish()]);
  await rd.mapAsync(GPUMapMode.READ);
  const t = new BigUint64Array(rd.getMappedRange().slice(0));
  console.log(['prep', 'shadow', 'main', 'post'].map((n, k) => `${n} ${(Number(t[2 * k + 1] - t[2 * k]) / 1e6).toFixed(2)} ms`).join(', '));
  console.log(`frame at ${w}x${h}: median ${times[4].toFixed(2)} ms (wall, includes submit latency)`);
  device.destroy();
}
