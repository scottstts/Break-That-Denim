/**
 * Interpenetration census of a dumped session state (session.ts DUMP=1): active spans that pass
 * through an intact cell (the collision faces, as two triangles), split by whether the span borders
 * an intact cell itself (sheet) or not (a strand: a thread across an opening). Spans and cells that
 * share a node are not tested.
 *   node tools/gpu/run.mjs tools/studies/crossings.ts <state name>
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { compileScene, defaultSceneSpecs } from '../../src/simulation/scene';
import { CAPTURES } from '../gpu/env';

type V = [number, number, number];

/** Parameter t in (0, 1) where segment p-q crosses triangle (a, b, c), or -1. */
export function segTri(p: V, q: V, a: V, b: V, c: V): number {
  const e1: V = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const e2: V = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const d: V = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
  const h: V = [d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]];
  const det = e1[0] * h[0] + e1[1] * h[1] + e1[2] * h[2];
  if (Math.abs(det) < 1e-18) return -1;
  const f = 1 / det;
  const s: V = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
  const u = f * (s[0] * h[0] + s[1] * h[1] + s[2] * h[2]);
  if (u < 0 || u > 1) return -1;
  const qv: V = [s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]];
  const v = f * (d[0] * qv[0] + d[1] * qv[1] + d[2] * qv[2]);
  if (v < 0 || u + v > 1) return -1;
  const t = f * (e2[0] * qv[0] + e2[1] * qv[1] + e2[2] * qv[2]);
  return t > 0 && t < 1 ? t : -1;
}

export function census(scene: ReturnType<typeof compileScene>, x: Float32Array, segs: Uint32Array): { sheet: number; strand: number; strandSpans: number; sheetSpans: number; excluded: number } {
  const L = scene.layers[0];
  const N = scene.nodeCount;
  const active = (n: number, f: number) => (segs[8 * (2 * n + f) + 5] & 3) === 1;
  const cloth = (n: number) => x[4 * n + 3] < 0.5;
  const P = (n: number): V => [x[4 * n], x[4 * n + 1], x[4 * n + 2]];
  const faceOk = new Uint8Array(N);
  for (let j = 0; j + 1 < L.ny; j++) {
    for (let i = 0; i + 1 < L.nx; i++) {
      const n = L.nodeOffset + j * L.nx + i;
      const c = [n, n + 1, n + L.nx, n + L.nx + 1];
      if (!c.every(cloth)) continue;
      if (active(n, 0) && active(n, 1) && active(n + 1, 1) && active(n + L.nx, 0)) faceOk[n] = 1;
    }
  }
  // faces hashed by bounding box on 4 mm cells
  const cell = 4e-3;
  const grid = new Map<string, number[]>();
  const key = (a: number, b: number, c: number) => `${a},${b},${c}`;
  for (let n = 0; n < N; n++) {
    if (!faceOk[n]) continue;
    const c = [n, n + 1, n + L.nx, n + L.nx + 1];
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
  let sheet = 0;
  let strand = 0;
  let strandSpans = 0;
  let sheetSpans = 0;
  // crossings of a face within self-contact's lattice exclusion window of either span end
  // (selfcontact.wgsl detect: di, dj in [-3, 2]), which contact never tests
  let excluded = 0;
  const inWindow = (node: number, face: number) => {
    const di = (face % L.nx) - (node % L.nx);
    const dj = Math.floor(face / L.nx) - Math.floor(node / L.nx);
    return di >= -3 && di <= 2 && dj >= -3 && dj <= 2;
  };
  let near = false;
  for (let j = 0; j < L.ny; j++) {
    for (let i = 0; i < L.nx; i++) {
      const n = L.nodeOffset + j * L.nx + i;
      for (let f = 0; f < 2; f++) {
        if ((f === 0 && i + 1 >= L.nx) || (f === 1 && j + 1 >= L.ny) || !active(n, f)) continue;
        const m = f === 0 ? n + 1 : n + L.nx;
        if (!cloth(n) || !cloth(m)) continue;
        // a span borders the cells on either side of it
        const sides = f === 0 ? [n, j > 0 ? n - L.nx : -1] : [n, i > 0 ? n - 1 : -1];
        const isSheet = sides.some((c) => c >= 0 && faceOk[c]);
        if (isSheet) sheetSpans++;
        else strandSpans++;
        const p = P(n);
        const q = P(m);
        const seen = new Set<number>();
        let hit = false;
        const steps = Math.ceil(Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) / (0.5 * cell)) + 1;
        for (let s = 0; s <= steps && !hit; s++) {
          const t = s / steps;
          const k = key(Math.floor((p[0] + t * (q[0] - p[0])) / cell), Math.floor((p[1] + t * (q[1] - p[1])) / cell), Math.floor((p[2] + t * (q[2] - p[2])) / cell));
          for (const fc of grid.get(k) ?? []) {
            if (seen.has(fc)) continue;
            seen.add(fc);
            const c = [fc, fc + 1, fc + L.nx, fc + L.nx + 1];
            if (c.includes(n) || c.includes(m)) continue;
            const [c00, c10, c01, c11] = c.map(P);
            if (segTri(p, q, c00, c10, c11) >= 0 || segTri(p, q, c00, c11, c01) >= 0) {
              hit = true;
              near = inWindow(n, fc) || inWindow(m, fc);
              break;
            }
          }
        }
        if (hit) {
          if (isSheet) sheet++;
          else strand++;
          if (near) excluded++;
        }
      }
    }
  }
  return { sheet, strand, strandSpans, sheetSpans, excluded };
}

export default async function main(args: string[]): Promise<void> {
  const scene = compileScene(defaultSceneSpecs());
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  const N = scene.nodeCount;
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const x = new Float32Array(buf, 0, 4 * N);
  const segs = new Uint32Array(buf, 16 * N, 16 * N);
  const r = census(scene, x, segs);
  console.log(`${args[0]}: spans through a cell: sheet ${r.sheet} / ${r.sheetSpans}, strand ${r.strand} / ${r.strandSpans} (${r.excluded} within the contact's lattice exclusion)`);
}
