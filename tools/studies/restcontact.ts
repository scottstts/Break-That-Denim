/**
 * Does a dumped state come to rest? (session.ts DUMP=1: positions, spans, tails and yarn field; the
 * velocities start at zero.) Runs REST_SECONDS (default 2) of substeps without any hand and prints, per
 * 0.1 s, the largest speed and kinetic energy of attached cloth and of detached pieces, and the
 * fastest node. Motion that keeps going from rest is energy the solve injects (a detached thread lying
 * on a flap chattering against it, say). SOLVER (JSON) overrides solver parameters;
 * WATCH=n also prints node n. NOPIECES=1 leaves the detached pieces unlabelled (no rigid correction).
 *   node tools/gpu/run.mjs tools/studies/restcontact.ts <state name>
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../src/gpu/util';
import { settledSolver } from '../gpu/scripts/render';
import { Topology } from '../../src/simulation/gpu/topology';
import { CAPTURES, drain, gpuDevice } from '../gpu/env';

export default async function main(args: string[]): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device, JSON.parse(process.env.SOLVER ?? '{}') as object);
  const N = solver.N;
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  let o = 0;
  const take = (bytes: number) => {
    const out = buf.slice(o, o + bytes);
    o += bytes;
    return out;
  };
  const x = new Float32Array(take(16 * N));
  device.queue.writeBuffer(solver.segs, 0, take(64 * N));
  device.queue.writeBuffer(solver.tails, 0, take(16 * N));
  device.queue.writeBuffer(solver.field, 0, take(solver.field.size));
  solver.setState(x);
  // the detached pieces as the solver sees them in a session (rigid-mode correction); NOPIECES=1 skips
  if (!process.env.NOPIECES) await new Topology(device, solver, scene).request();
  if (process.env.WATCH) {
    // the faces nearest the watched node at the start: intact cells (four spans) of nodes in the solve
    const n = Number(process.env.WATCH);
    const sg = new Uint32Array(await readBuffer(device, solver.segs));
    const act = (q: number) => (sg[8 * q + 5] & 3) === 1;
    const L = scene.layers[0];
    const P = (q: number) => [x[4 * q], x[4 * q + 1], x[4 * q + 2]];
    const near: Array<[number, number]> = [];
    for (let j = 0; j + 1 < L.ny; j++) for (let i = 0; i + 1 < L.nx; i++) {
      const f = L.nodeOffset + j * L.nx + i;
      const c = [f, f + 1, f + L.nx, f + L.nx + 1];
      if (c.some((q) => x[4 * q + 3] > 1.5) || !(act(2 * f) && act(2 * f + 1) && act(2 * (f + 1) + 1) && act(2 * (f + L.nx)))) continue;
      const m = [0, 1, 2].map((k) => c.reduce((t, q) => t + P(q)[k], 0) / 4);
      near.push([Math.hypot(...[0, 1, 2].map((k) => m[k] - P(n)[k])), f]);
    }
    // active spans between a detached node and a node of the cloth (the analysis should leave none)
    let bridges = 0;
    for (let q = 0; q < N; q++) {
      for (const [f, m] of [[0, q + 1], [1, q + L.nx]]) {
        if (m >= N || !act(2 * q + f)) continue;
        const sq = x[4 * q + 3];
        const sm = x[4 * m + 3];
        if ((sq > 0.5 && sq < 1.5 && sm < 0.5) || (sm > 0.5 && sm < 1.5 && sq < 0.5)) bridges++;
      }
    }
    console.log(`  spans joining detached material to cloth: ${bridges}`);
    near.sort((u, v) => u[0] - v[0]);
    for (const [d, f] of near.slice(0, 4)) {
      const c = [f, f + 1, f + L.nx, f + L.nx + 1];
      const sf = new Float32Array(sg.buffer);
      const spans = [2 * f, 2 * f + 1, 2 * (f + 1) + 1, 2 * (f + L.nx)];
      console.log(`    spans intact ${spans.map((q) => sf[8 * q + 1].toFixed(2)).join(' ')}, cut yarns ${spans.map((q) => (sg[8 * q + 3] >>> 0).toString(2).split('').filter((b) => b === '1').length).join(' ')}`);
      console.log(`  face ${f} (i=${(f - L.nodeOffset) % L.nx} j=${Math.floor((f - L.nodeOffset) / L.nx)}) centroid ${(d * 1e3).toFixed(1)} mm from ${n}; corner states ${c.map((q) => x[4 * q + 3]).join(' ')}; corners at ${c.map((q) => '(' + P(q).map((v) => (v * 1e3).toFixed(0)).join(',') + ')').join(' ')}`);
    }
  }
  const h = solver.params.h;
  const steps = Math.round(0.1 / h);
  const seconds = Number(process.env.REST_SECONDS ?? process.env.SECONDS ?? 2);
  for (let k = 1; k <= Math.round(seconds / 0.1); k++) {
    for (let q = 0; q < steps; q++) {
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
    }
    await drain(device);
    const xs = new Float32Array(await readBuffer(device, solver.x));
    const v = new Float32Array(await readBuffer(device, solver.v));
    const c = new Uint32Array(await readBuffer(device, solver.counters, 64));
    const ke = [0, 0];
    const vmax = [0, 0];
    let top = -1;
    for (let n = 0; n < N; n++) {
      const w = xs[4 * n + 3];
      if (w > 1.5 || scene.pinned[n]) continue;
      const s = w > 0.5 ? 1 : 0;
      const sp = Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]);
      ke[s] += 0.5 * scene.mass[n] * sp * sp;
      if (sp > vmax[s]) vmax[s] = sp;
      if (sp >= Math.max(...vmax)) top = n;
    }
    if (process.env.WATCH) {
      const n = Number(process.env.WATCH);
      console.log(`  watch ${n}: state ${xs[4 * n + 3]} x (${[0, 1, 2].map((i) => (xs[4 * n + i] * 1e3).toFixed(1)).join(', ')}) mm v (${[0, 1, 2].map((i) => v[4 * n + i].toFixed(3)).join(', ')}) m/s`);
    }
    console.log(`t=${(0.1 * k).toFixed(1)} cloth vmax ${vmax[0].toFixed(4)} KE ${(1e9 * ke[0]).toFixed(2)} nJ, detached vmax ${vmax[1].toFixed(4)} KE ${(1e9 * ke[1]).toFixed(2)} nJ, fastest ${top} (state ${xs[4 * top + 3]}), ruptures ${c[0]}, rejected ${c[12]}`);
  }
  device.destroy();
}
