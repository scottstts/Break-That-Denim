/**
 * Thread runs of a dumped state (session.ts DUMP=1) that no lattice chain carries but are held at both
 * ends: the renderer draws them as straight lines between their ends (thread_geom.wgsl runCenter).
 *   node tools/gpu/run.mjs tools/studies/straightruns.ts <state name>
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../src/rendering/renderer';
import { CAPTURES, drain, gpuDevice } from '../gpu/env';
import { settledSolver } from '../gpu/scripts/render';

export default async function main(args: string[]): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const N = scene.nodeCount;
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  let o = 0;
  for (const b of [solver.x, solver.segs, solver.tails, solver.field]) {
    device.queue.writeBuffer(b, 0, buf, o, b.size);
    o += b.size;
  }
  const renderer = new Renderer(device, 'rgba8unorm', solver, scene);
  renderer.resize(320, 200);
  const tex = device.createTexture({ size: [320, 200], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  for (let k = 0; k < 3; k++) {
    const enc = device.createCommandEncoder();
    renderer.encode(enc, DEFAULT_CAMERA, k / 30, tex.createView());
    device.queue.submit([enc.finish()]);
  }
  await drain(device);
  const r = renderer as unknown as { threadCount: GPUBuffer; runs: GPUBuffer; runGeo: GPUBuffer };
  const count = new Uint32Array(await readBuffer(device, r.threadCount))[0];
  const runs = new Uint32Array(await readBuffer(device, r.runs, 16 * count));
  const geo = new Float32Array(await readBuffer(device, r.runGeo, 7 * 16 * count));
  let both = 0;
  let straight = 0;
  const lens: number[] = [];
  const reasons: string[] = [];
  const hanging: number[] = [];
  const xs = new Float32Array(await readBuffer(device, solver.x));
  const sg = new Uint32Array(await readBuffer(device, solver.segs));
  const L = scene.layers[0];
  const F = scene.field[0];
  // why chainCarries (thread_prep.wgsl) fails: a span that does not hold (rupture: all yarns cut;
  // otherwise slip pull-out), or nodes of different pieces
  const reason = (rx: number, ry: number, rz: number): string => {
    const f = (rx >> 4) & 1;
    const yarn = ry & 0xffff;
    const pos = ry >>> 16;
    const n = rz;
    const Bal = f === 0 ? F.bu : F.bv;
    const Bcr = f === 0 ? F.bv : F.bu;
    const len = f === 0 ? L.nx : L.ny;
    const drawA = pos - 1;
    const drawB = pos + n;
    const b = Math.floor(yarn / Bcr);
    const at = (a: number) => (f === 0 ? b * L.nx + a : a * L.nx + b);
    const cl = (v: number) => Math.min(Math.max(Math.floor((v - 0.5 * (Bal - 1)) / Bal), 0), len - 2);
    const a0 = cl(drawA);
    const a1 = cl(drawB);
    const w0 = xs[4 * at(a0) + 3];
    for (let a = a0; a <= a1; a++) {
      const m = at(a);
      const fl = sg[8 * (2 * m + f) + 5];
      if ((fl & 3) !== 1) {
        const mask = sg[8 * (2 * m + f) + 3];
        return mask === (1 << Bcr) - 1 || mask === 0xffffffff ? 'rupture' : `slip(mask ${mask.toString(2)})`;
      }
      if (Math.abs(xs[4 * m + 3] - w0) > 0.5 || Math.abs(xs[4 * at(a + 1) + 3] - w0) > 0.5) return 'pieces';
    }
    return 'none?';
  };
  for (let q = 0; q < count; q++) {
    const heldA = ((runs[4 * q] >> 5) & 3) !== 1;
    const heldB = ((runs[4 * q] >> 7) & 3) !== 1;
    const g = 28 * q;
    if (!heldA || !heldB) {
      if (heldA !== heldB && geo[g] < 0.5) hanging.push(((runs[4 * q + 2] + 0.5) * (((runs[4 * q] >> 4) & 1) === 0 ? scene.layers[0].weave.pitchU : scene.layers[0].weave.pitchV)) * 1e3);
      continue;
    }
    both++;
    if (geo[g] > 0.5) continue;
    straight++;
    reasons.push(reason(runs[4 * q], runs[4 * q + 1], runs[4 * q + 2]));
    lens.push(Math.hypot(geo[g + 8] - geo[g + 4], geo[g + 9] - geo[g + 5], geo[g + 10] - geo[g + 6]) * 1e3);
  }
  lens.sort((a, b) => b - a);
  console.log(`${args[0]}: runs ${count}, held at both ends ${both}, straight (no carrying chain) ${straight}; longest mm ${lens.slice(0, 8).map((v) => v.toFixed(1)).join(' ')}; reasons ${JSON.stringify(reasons.reduce<Record<string, number>>((acc, r) => ((acc[r] = (acc[r] ?? 0) + 1), acc), {}))}`);
  hanging.sort((a, b) => b - a);
  console.log(`  hanging (one end held, no carrying chain): ${hanging.length}; longest mm ${hanging.slice(0, 10).map((v) => v.toFixed(1)).join(' ')}; over 5 mm ${hanging.filter((v) => v > 5).length}`);
  void N;
  device.destroy();
}
