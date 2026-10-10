/**
 * Free thread tails of a dumped session state (session.ts DUMP=1) against the cloth: for every run
 * held at one end beyond its chain's reach (thread_tail.wgsl), whether its drawn tail passes through
 * an intact cell (the collision faces as two triangles; faces within two cells of its root, the cloth
 * it comes out of, are not tested). Compares the solved tails with the procedural hang they replaced
 * (from the same root and direction: bending to the vertical within 1.5 mm).
 *   node tools/gpu/run.mjs tools/studies/tails.ts <state name>
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../src/rendering/renderer';
import { CAPTURES, drain, gpuDevice } from '../gpu/env';
import { settledSolver } from '../gpu/scripts/render';
import { segTri } from './crossings';

type V = [number, number, number];
const RUN_GEO = 9;
const TAIL_PTS = 17;

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
  {
    const enc = device.createCommandEncoder();
    solver.encodeContactRefresh(enc);
    device.queue.submit([enc.finish()]);
  }
  for (let k = 0; k < 3; k++) {
    const enc = device.createCommandEncoder();
    renderer.encode(enc, DEFAULT_CAMERA, k / 30, tex.createView());
    device.queue.submit([enc.finish()]);
  }
  await drain(device);
  const r = renderer as unknown as { threadCount: GPUBuffer; runs: GPUBuffer; runGeo: GPUBuffer; tails: GPUBuffer; tailCount: GPUBuffer };
  const count = new Uint32Array(await readBuffer(device, r.threadCount))[0];
  const runs = new Uint32Array(await readBuffer(device, r.runs, 16 * count));
  const geo = new Float32Array(await readBuffer(device, r.runGeo, RUN_GEO * 16 * count));
  const nTails = new Uint32Array(await readBuffer(device, r.tailCount))[0];
  const tails = new Float32Array(await readBuffer(device, r.tails, 16 * TAIL_PTS * Math.max(1, Math.min(nTails, 8192))));
  const x = new Float32Array(await readBuffer(device, solver.x));
  const segs = new Uint32Array(await readBuffer(device, solver.segs));
  const L = scene.layers[0];
  const P = (n: number): V => [x[4 * n], x[4 * n + 1], x[4 * n + 2]];
  const active = (n: number, f: number) => (segs[8 * (2 * n + f) + 5] & 3) === 1;
  const cloth = (n: number) => x[4 * n + 3] < 0.5;
  // intact cells, hashed by bounding box on 4 mm cells
  const cell = 4e-3;
  const grid = new Map<string, number[]>();
  const key = (a: number, b: number, c: number) => `${a},${b},${c}`;
  for (let j = 0; j + 1 < L.ny; j++) {
    for (let i = 0; i + 1 < L.nx; i++) {
      const n = L.nodeOffset + j * L.nx + i;
      const c = [n, n + 1, n + L.nx, n + L.nx + 1];
      if (!c.every(cloth) || !(active(n, 0) && active(n, 1) && active(n + 1, 1) && active(n + L.nx, 0))) continue;
      const lo = [1e9, 1e9, 1e9];
      const hi = [-1e9, -1e9, -1e9];
      for (const m of c) for (let q = 0; q < 3; q++) {
        lo[q] = Math.min(lo[q], x[4 * m + q]);
        hi[q] = Math.max(hi[q], x[4 * m + q]);
      }
      for (let a = Math.floor(lo[0] / cell); a <= Math.floor(hi[0] / cell); a++)
        for (let b = Math.floor(lo[1] / cell); b <= Math.floor(hi[1] / cell); b++)
          for (let z = Math.floor(lo[2] / cell); z <= Math.floor(hi[2] / cell); z++) {
            const k = key(a, b, z);
            const list = grid.get(k);
            if (list) list.push(n);
            else grid.set(k, [n]);
          }
    }
  }
  const ij = (n: number) => [(n - L.nodeOffset) % L.nx, Math.floor((n - L.nodeOffset) / L.nx)];
  const through = (pts: V[], root: number): boolean => {
    const [ri, rj] = ij(root);
    for (let s = 0; s + 1 < pts.length; s++) {
      const p = pts[s], q = pts[s + 1];
      const seen = new Set<number>();
      const steps = Math.ceil(Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) / (0.5 * cell)) + 1;
      for (let u = 0; u <= steps; u++) {
        const t = u / steps;
        const k = key(Math.floor((p[0] + t * (q[0] - p[0])) / cell), Math.floor((p[1] + t * (q[1] - p[1])) / cell), Math.floor((p[2] + t * (q[2] - p[2])) / cell));
        for (const f of grid.get(k) ?? []) {
          if (seen.has(f)) continue;
          seen.add(f);
          const [fi, fj] = ij(f);
          if (Math.abs(fi - ri) <= 2 && Math.abs(fj - rj) <= 2) continue;
          const [c00, c10, c01, c11] = [f, f + 1, f + L.nx, f + L.nx + 1].map(P);
          if (segTri(p, q, c00, c10, c11) >= 0 || segTri(p, q, c00, c11, c01) >= 0) return true;
        }
      }
    }
    return false;
  };
  let hanging = 0, solved = 0, oldThrough = 0, newThrough = 0;
  const lens: number[] = [];
  for (let q = 0; q < count; q++) {
    const g = RUN_GEO * 4 * q;
    const heldA = ((runs[4 * q] >> 5) & 3) !== 1;
    const heldB = ((runs[4 * q] >> 7) & 3) !== 1;
    const len = geo[g + 3];
    if (geo[g] > 0.5 || heldA === heldB || len <= 0.1e-3) continue;
    hanging++;
    lens.push(len * 1e3);
    const root = Math.round(geo[g + 28 + 3]);
    const q0: V = [geo[g + 28], geo[g + 29], geo[g + 30]];
    const d0: V = [geo[g + 32], geo[g + 33], geo[g + 34]];
    // the procedural hang (thread_geom.wgsl before tails): bends from d0 toward -z within 1.5 mm
    const old: V[] = [q0];
    let c: V = [...q0];
    const steps = 6;
    for (let k = 0; k < steps; k++) {
      const sm = ((k + 0.5) / steps) * len;
      const w = sm / (sm + 1.5e-3);
      const d: V = [d0[0] * (1 - w), d0[1] * (1 - w), d0[2] * (1 - w) - w];
      const dl = Math.hypot(d[0], d[1], d[2]);
      c = [c[0] + (d[0] / dl) * (len / steps), c[1] + (d[1] / dl) * (len / steps), c[2] + (d[2] / dl) * (len / steps)];
      old.push(c);
    }
    if (through(old, root)) oldThrough++;
    const slot = Math.round(geo[g + 2]);
    if (slot < 0) continue;
    solved++;
    const pts: V[] = [];
    for (let k = 0; k < TAIL_PTS; k++) pts.push([tails[4 * (slot * TAIL_PTS + k)], tails[4 * (slot * TAIL_PTS + k) + 1], tails[4 * (slot * TAIL_PTS + k) + 2]]);
    if (through(pts, root)) newThrough++;
  }
  lens.sort((a, b) => b - a);
  console.log(`${args[0]}: free tails ${hanging} (solved ${solved}); longest mm ${lens.slice(0, 6).map((v) => v.toFixed(1)).join(' ')}; over 5 mm ${lens.filter((v) => v > 5).length}`);
  console.log(`  through the cloth: procedural hang ${oldThrough}, solved tails ${newThrough}`);
  void N;
  device.destroy();
}
