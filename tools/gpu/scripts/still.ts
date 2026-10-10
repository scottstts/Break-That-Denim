/**
 * Render a dumped session state (session.ts with DUMP=1) from any camera, for iterating on the
 * appearance of a fixed torn state.
 *   node tools/gpu/run.mjs tools/gpu/scripts/still.ts <state name> <out name> [width height]
 * CAMERA='{"distance":0.3,"target":[0.01,0,0]}' overrides the default camera; RENDERCOST=1 also reports
 * per-pass and whole-frame GPU time (minimum over repeated frames; pass
 * boundaries overlap on tile-based GPUs, so the frame total is the reliable figure); with REBUILD=1
 * every measured frame also rebuilds the field-derived state (edge field, loose threads).
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { CAPTURES, drain, gpuDevice, saveTexture } from '../env';
import { settledSolver } from './render';

export default async function main(args: string[]): Promise<void> {
  const stateName = args[0];
  const out = args[1] ?? `${stateName}-still`;
  const width = Number(args[2] ?? 1400);
  const height = Number(args[3] ?? 900);
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const N = scene.nodeCount;
  const data = readFileSync(resolve(CAPTURES, `${stateName}.state`));
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const xBytes = 16 * N;
  const sBytes = solver.segs.size;
  device.queue.writeBuffer(solver.x, 0, buf, 0, xBytes);
  device.queue.writeBuffer(solver.segs, 0, buf, xBytes, sBytes);
  const tBytes = solver.tails.size;
  if (buf.byteLength >= xBytes + sBytes + tBytes) device.queue.writeBuffer(solver.tails, 0, buf, xBytes + sBytes, tBytes);
  if (buf.byteLength >= xBytes + sBytes + tBytes + solver.field.size) device.queue.writeBuffer(solver.field, 0, buf, xBytes + sBytes + tBytes, solver.field.size);
  const format: GPUTextureFormat = 'rgba8unorm';
  const renderer = new Renderer(device, format, solver, scene);
  renderer.resize(width, height);
  // LIGHT='{"keyOffset":[..],"keyIntensity":..,"exposure":..,"ambient":[..],"keyAim":[..],"keyCone":..}'
  Object.assign(renderer, JSON.parse(process.env.LIGHT ?? '{}') as object);
  const camera = { ...DEFAULT_CAMERA, ...(JSON.parse(process.env.CAMERA ?? '{}') as object) };
  const tex = device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  // SETTLE seconds (default 2) of frames at 30 Hz first: the loose threads' dynamics come to rest
  const settle = Number(process.env.SETTLE ?? 2);
  const frames = Math.max(2, Math.round(settle * 30));
  for (let k = 0; k < frames; k++) {
    const enc = device.createCommandEncoder();
    renderer.encode(enc, camera, k / 30, tex.createView());
    device.queue.submit([enc.finish()]);
    if (k % 8 === 7) await drain(device);
  }
  console.log(await saveTexture(device, tex, out));
  if (process.env.RENDERCOST) {
    const qs = device.createQuerySet({ type: 'timestamp', count: 8 });
    const res = device.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const names = ['prep', 'shadow', 'main', 'post'];
    const samples: number[][] = names.map(() => []);
    const total: number[] = [];
    const frames = 15;
    for (let k = 0; k < frames; k++) {
      const rd = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder();
      // REBUILD=1: rebuild the field-derived state every frame (the cost while yarns are separating), as
      // the solver's field event triggers it; REBUILD=reset: from scratch, as after a reset
      if (process.env.REBUILD === 'reset') renderer.resetTopology(enc);
      else if (process.env.REBUILD) device.queue.writeBuffer(solver.fieldEvents, 0, new Uint32Array([1]));
      renderer.encode(enc, camera, (frames + k) / 30, tex.createView(), qs);
      enc.resolveQuerySet(qs, 0, 8, res, 0);
      enc.copyBufferToBuffer(res, 0, rd, 0, 64);
      device.queue.submit([enc.finish()]);
      await drain(device);
      await rd.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(rd.getMappedRange().slice(0));
      names.forEach((_, i) => samples[i].push(Number(t[2 * i + 1] - t[2 * i]) / 1e6));
      total.push(Number(t[7] - t[0]) / 1e6);
      rd.destroy();
    }
    console.log('RENDER ' + names.map((n, i) => `${n} ${Math.min(...samples[i]).toFixed(2)} ms`).join(', ') + `, frame ${Math.min(...total).toFixed(2)} ms`);
  }
  device.destroy();
}
