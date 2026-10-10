/**
 * Fold census of a dumped session state (session.ts DUMP=1) around a grip: on rings of rest radius r
 * about the grabbed material point, how much of the material gathered there went into buckling (out of
 * plane) and how much into in-plane compression, and the wrinkles the buckling made.
 *   rest  = the ring's rest circumference
 *   chord = its current length through the nodes in 3D (what the yarns hold: in-plane compression)
 *   flat  = its length projected on the mean plane of the ring (the material's gathering)
 * in-plane share = (rest - chord) / (rest - flat): 0 when every bit of gathering buckles (cloth), 1 when
 * the sheet squeezes like rubber. Wrinkles: the ring's height off a fitted low-order shape (the tent of
 * the pull), their count (sign changes / 2) and RMS amplitude. Spans: the compressed fraction near the
 * grip.
 *   node tools/gpu/run.mjs tools/studies/folds.ts <state name> <grab x,y (m, rest world)> (SCENE as for the session)
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { compileScene } from '../../src/simulation/scene';
import { CAPTURES, experimentSpecs } from '../gpu/env';

export default async function main(args: string[]): Promise<void> {
  const scene = compileScene(experimentSpecs());
  const N = scene.nodeCount;
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  const x = new Float32Array(data.buffer.slice(data.byteOffset, data.byteOffset + 16 * N));
  const [gx, gy] = (args[1] ?? '0,0').split(',').map(Number);
  const L = scene.layers[0];
  // the grabbed material point: the node whose rest position is nearest the grab
  let g = 0;
  let best = 1e9;
  for (let n = 0; n < N; n++) {
    const d = Math.hypot(scene.position[3 * n] - gx, scene.position[3 * n + 1] - gy);
    if (d < best) {
      best = d;
      g = n;
    }
  }
  const P = (n: number) => [x[4 * n], x[4 * n + 1], x[4 * n + 2]];
  const R = (n: number) => [scene.position[3 * n], scene.position[3 * n + 1]];
  const [g0x, g0y] = R(g);
  console.log(`grip node ${g} at rest (${g0x.toFixed(3)}, ${g0y.toFixed(3)}), now (${P(g).map((v) => v.toFixed(3)).join(', ')})`);
  const cell = Math.max(L.material.pitchU, L.material.pitchV);
  for (const r of [0.01, 0.02, 0.03, 0.05, 0.08, 0.12]) {
    // ring nodes: rest distance within half a cell of r, ordered by rest angle
    const ring: { n: number; a: number }[] = [];
    for (let n = 0; n < N; n++) {
      if (x[4 * n + 3] >= 0.5) continue;
      const [rx, ry] = R(n);
      const d = Math.hypot(rx - g0x, ry - g0y);
      if (Math.abs(d - r) < 0.5 * cell) ring.push({ n, a: Math.atan2(ry - g0y, rx - g0x) });
    }
    if (ring.length < 16) continue;
    ring.sort((p, q) => p.a - q.a);
    const pts = ring.map((p) => P(p.n));
    const m = pts.length;
    // mean plane of the ring: centroid and normal (Newell)
    const c = [0, 0, 0];
    for (const p of pts) for (let k = 0; k < 3; k++) c[k] += p[k] / m;
    const nrm = [0, 0, 0];
    for (let i = 0; i < m; i++) {
      const a = pts[i], b = pts[(i + 1) % m];
      nrm[0] += (a[1] - b[1]) * (a[2] + b[2]);
      nrm[1] += (a[2] - b[2]) * (a[0] + b[0]);
      nrm[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    const nl = Math.hypot(nrm[0], nrm[1], nrm[2]);
    for (let k = 0; k < 3; k++) nrm[k] /= nl;
    let chord = 0, flat = 0;
    const h: number[] = [];
    for (let i = 0; i < m; i++) {
      const a = pts[i], b = pts[(i + 1) % m];
      const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      chord += Math.hypot(d[0], d[1], d[2]);
      const dn = d[0] * nrm[0] + d[1] * nrm[1] + d[2] * nrm[2];
      flat += Math.hypot(d[0] - dn * nrm[0], d[1] - dn * nrm[1], d[2] - dn * nrm[2]);
      h.push((a[0] - c[0]) * nrm[0] + (a[1] - c[1]) * nrm[1] + (a[2] - c[2]) * nrm[2]);
    }
    // rest circumference through the same nodes
    let rest = 0;
    for (let i = 0; i < m; i++) {
      const [ax, ay] = R(ring[i].n);
      const [bx, by] = R(ring[(i + 1) % m].n);
      rest += Math.hypot(bx - ax, by - ay);
    }
    // wrinkles: height off the low-order shape (Fourier 0..2 in the rest angle)
    const ang = ring.map((p) => p.a);
    const coef: number[][] = [];
    for (let k = 0; k <= 2; k++) {
      let a = 0, b = 0;
      for (let i = 0; i < m; i++) {
        a += (h[i] * Math.cos(k * ang[i]) * 2) / m;
        b += (h[i] * Math.sin(k * ang[i]) * 2) / m;
      }
      coef.push([k === 0 ? a / 2 : a, b]);
    }
    const res = h.map((v, i) => v - coef.reduce((s, [a, b], k) => s + a * Math.cos(k * ang[i]) + b * Math.sin(k * ang[i]), 0));
    let sc = 0;
    for (let i = 0; i < m; i++) if (Math.sign(res[i]) !== Math.sign(res[(i + 1) % m])) sc++;
    const rms = Math.sqrt(res.reduce((s, v) => s + v * v, 0) / m);
    const gathered = rest - flat;
    const share = gathered > 1e-6 ? (rest - chord) / gathered : NaN;
    console.log(
      `r ${(100 * r).toFixed(0)} cm: ${m} nodes, gathered ${((100 * gathered) / rest).toFixed(1)}% of the ring, in-plane share ${share.toFixed(2)}; ` +
        `wrinkles ${(sc / 2).toFixed(0)} (wavelength ${((100 * flat) / Math.max(sc / 2, 1)).toFixed(1)} cm), rms ${(1000 * rms).toFixed(2)} mm`,
    );
  }
}
