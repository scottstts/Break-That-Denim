/**
 * Map of the cloth surface material of a dumped state (session.ts DUMP=1) as the renderer derives
 * it: per crossing, white = both yarns are surface, pale blue = pick only, dark blue = warp only,
 * orange = a loose thread there, black = nothing. Cropped to the damage; rows stretched to the pick /
 * warp pitch ratio.
 *   node tools/gpu/run.mjs tools/gpu/scripts/surfacemap.ts <state> <out name> [pixels per crossing]
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { CAPTURES, encodePng, gpuDevice } from '../env';
import { settledSolver } from './render';

export default async function main(args: string[]): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const N = scene.nodeCount;
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  let o = 0;
  for (const b of [solver.x, solver.segs, solver.tails, solver.field]) {
    const size = b === solver.x ? 16 * N : b.size;
    device.queue.writeBuffer(b, 0, buf, o, size);
    o += size;
  }
  const renderer = new Renderer(device, 'rgba8unorm', solver, scene);
  renderer.resize(64, 64);
  const tex = device.createTexture({ size: [64, 64], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const enc = device.createCommandEncoder();
  renderer.encode(enc, DEFAULT_CAMERA, 0, tex.createView());
  device.queue.submit([enc.finish()]);
  const bits = new Uint32Array(await readBuffer(device, (renderer as unknown as { threadBits: GPUBuffer }).threadBits));
  const field = new Uint32Array(await readBuffer(device, solver.field));
  const F = scene.field[0];
  const S = Number(args[2] ?? 2);
  const surf = (c: number, f: number) => ((field[c] >> (8 * f)) & 0xff) < 254 && ((bits[c >> 4] >> (2 * (c & 15) + f)) & 1) === 0;
  const thread = (c: number, f: number) => ((field[c] >> (8 * f)) & 0xff) < 254 && ((bits[c >> 4] >> (2 * (c & 15) + f)) & 1) === 1;
  let k0 = 1e9, k1 = -1, m0 = 1e9, m1 = -1;
  for (let m = 0; m < F.M; m++) for (let k = 0; k < F.K; k++) {
    const c = m * F.K + k;
    if (!surf(c, 0) || !surf(c, 1)) [k0, k1, m0, m1] = [Math.min(k0, k), Math.max(k1, k), Math.min(m0, m), Math.max(m1, m)];
  }
  k0 = Math.max(0, k0 - 20); k1 = Math.min(F.K - 1, k1 + 20); m0 = Math.max(0, m0 - 14); m1 = Math.min(F.M - 1, m1 + 14);
  const W = (k1 - k0 + 1) * S, H = Math.round((m1 - m0 + 1) * S * 1.53);
  const rgba = new Uint8Array(4 * W * H);
  for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) {
    const k = k0 + Math.floor(xx / S), m = m1 - Math.floor(y / (S * 1.53));
    const c = m * F.K + k;
    const p = surf(c, 0), w = surf(c, 1);
    let col = p && w ? [235, 235, 235] : p ? [150, 190, 235] : w ? [40, 60, 150] : [0, 0, 0];
    if (!p && !w && (thread(c, 0) || thread(c, 1))) col = [230, 140, 40];
    const q = 4 * (y * W + xx);
    rgba.set([col[0], col[1], col[2], 255], q);
  }
  const path = resolve(CAPTURES, `${args[1]}.png`);
  writeFileSync(path, encodePng(W, H, rgba));
  console.log(path, { k0, k1, m0, m1 });
  device.destroy();
}
