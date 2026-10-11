// Floor friction probe: a 5 x 5 detached patch lying on the floor (at its resting depth), sent sliding
// at V (m/s) and spinning at OMEGA (rad/s); per substep its span strain, spin, mean velocity, height,
// line-search step and largest Newton update per iteration.
//   node tools/gpu/run.mjs tools/studies/floorslide.ts
// STEPS=n substeps; SOLVER='{..}' solver overrides; FLOORLAW='{..}' overrides FLOOR_CONTACT; NOFLOOR=1
// no floor; NOTOPO=1 no piece labels (no rigid modes); LSDUMP=1 line-search energies; DELTA=1 Newton
// directions; NOKR=1 rigid frame without internal rotation stiffness; NOCURVE=1 linear (not rotation)
// update path for pieces.
import { readBuffer } from '../../src/gpu/util';
import { Topology } from '../../src/simulation/gpu/topology';
import { FLOOR_CONTACT } from '../../src/simulation/floor';
import { DENIM } from '../../src/simulation/material';
import { drain, gpuDevice } from '../gpu/env';
import { settledSolver } from '../gpu/scripts/render';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device, { ...(JSON.parse(process.env.SOLVER ?? '{}') as object), patch: (m: string, c: string) => (m === 'rigid' && process.env.NOCURVE ? c.replace('w = vec4<f32>(pieceStep[2u * p + 1u].xyz, 1.0);', '') : m === 'rigid' && process.env.NOKR ? c.replace('e[tri(3u + i, 3u + j)] += Kr[i][j];', '') : m === 'energy' && process.env.LSDUMP ? c.replace('counters[3] = bitcast<u32>(a);', 'counters[3] = bitcast<u32>(a); counters[13] = bitcast<u32>(e.x); counters[14] = bitcast<u32>(e.w); counters[15] = bitcast<u32>(tol.x);') : c) });
  const topology = new Topology(device, solver, scene);
  const nx = scene.layers[0].nx;
  const segs = new Uint32Array(await readBuffer(device, solver.segs));
  const rest = new Float32Array(segs.buffer);
  const fz = -0.02;
  const delta = 0.5 * DENIM.thickness + FLOOR_CONTACT.bulge;
  // FLOORLAW='{"friction":0}' overrides the contact law
  Object.assign(FLOOR_CONTACT, JSON.parse(process.env.FLOORLAW ?? '{}') as object);
  solver.setFloor(process.env.NOFLOOR ? null : fz);
  const xs = new Float32Array(await readBuffer(device, solver.x));
  for (let n = 0; n < solver.N; n++) xs[4 * n + 3] = 2;
  const v = new Float32Array(4 * solver.N);
  const V = Number(process.env.V ?? 0.25);
  const c0 = 60 * nx + 60;
  const nodes: number[] = [];
  for (let q = 0; q < 2 * solver.N; q++) segs[8 * q + 5] |= 2;
  for (let j = 0; j < 5; j++) {
    for (let i = 0; i < 5; i++) {
      const n = c0 + j * nx + i;
      nodes.push(n);
      xs.set([(i - 2) * rest[16 * n], (j - 2) * rest[16 * n + 8], fz + delta - Math.sqrt((scene.mass[n] * 9.81 * delta) / FLOOR_CONTACT.stiffness), 0], 4 * n);
      const W = Number(process.env.OMEGA ?? 0);
      v.set([V - W * (j - 2) * rest[16 * n + 8], W * (i - 2) * rest[16 * n], 0, 0], 4 * n);
      if (i < 4) { segs[8 * 2 * n + 5] = 1; segs[8 * 2 * n + 3] = 0; rest[8 * 2 * n + 1] = 1; }
      if (j < 4) { segs[8 * (2 * n + 1) + 5] = 1; segs[8 * (2 * n + 1) + 3] = 0; rest[8 * (2 * n + 1) + 1] = 1; }
    }
  }
  topology.reset();
  solver.setState(xs, v);
  device.queue.writeBuffer(solver.segs, 0, segs);
  if (!process.env.NOTOPO) await topology.request([]);
  const steps = Number(process.env.STEPS ?? 12);
  const mean = (a: Float32Array, q: number) => nodes.reduce((s, n) => s + a[4 * n + q], 0) / nodes.length;
  for (let k = 0; k < steps; k++) {
    const enc = device.createCommandEncoder();
    solver.encodeSubstep(enc);
    device.queue.submit([enc.finish()]);
    await drain(device);
    const x = new Float32Array(await readBuffer(device, solver.x));
    const vv = new Float32Array(await readBuffer(device, solver.v));
    const c = new Uint32Array(await readBuffer(device, solver.counters));
    const alpha = new Float32Array(c.buffer)[3];
    if (process.env.DELTA) {
      const dl = new Float32Array(await readBuffer(device, solver.delta));
      const pr = new Float32Array(await readBuffer(device, solver.pred));
      console.log(`  delta z ${[0, 12, 24].map((q) => (1000 * dl[4 * nodes[q] + 2]).toFixed(4)).join(' ')} mm, xy ${(1000 * dl[4 * nodes[12]]).toFixed(4)} ${(1000 * dl[4 * nodes[12] + 1]).toFixed(4)}, pred z ${(1000 * pr[4 * nodes[12] + 2]).toFixed(4)} mm`);
    }
    const cx = mean(x, 0);
    const cy = mean(x, 1);
    let lz = 0;
    let i2 = 0;
    for (const n of nodes) {
      const rx = x[4 * n] - cx;
      const ry = x[4 * n + 1] - cy;
      lz += rx * vv[4 * n + 1] - ry * vv[4 * n];
      i2 += rx * rx + ry * ry;
    }
    let strain = 0;
    for (const n of nodes) {
      const i = (n - c0) % nx;
      if (i < 4) strain = Math.max(strain, Math.hypot(x[4 * n] - x[4 * n + 4], x[4 * n + 1] - x[4 * n + 5], x[4 * n + 2] - x[4 * n + 6]) / rest[16 * n] - 1);
    }
    console.log(`${k}: strain ${(100 * strain).toFixed(4)} %, spin ${(lz / i2).toFixed(3)} rad/s, vx ${mean(vv, 0).toFixed(4)} vz ${mean(vv, 2).toFixed(4)} height ${(1000 * (mean(x, 2) - fz)).toFixed(4)} mm, alpha ${alpha}, rejected ${c[12]} upd ${[4, 5, 6, 7].map((q) => (1e6 * new Float32Array(c.buffer)[q]).toFixed(2)).join('/')} um${process.env.LSDUMP ? ` e1 ${new Float32Array(c.buffer)[13].toExponential(3)} e1/8 ${new Float32Array(c.buffer)[14].toExponential(3)} tol ${new Float32Array(c.buffer)[15].toExponential(3)}` : ''}, state ${x[4 * c0 + 3]}`);
  }
  console.log(`Coulomb deceleration per substep ${(FLOOR_CONTACT.friction * 9.81 * solver.params.h).toFixed(4)} m/s`);
}
