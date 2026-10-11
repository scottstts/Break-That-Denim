/**
 * Detached-material contracts on the production kernels (headless: Node + Dawn). A piece the
 * connectivity analysis detached keeps its mechanics: its intact spans resist extension under
 * differential motion and under partial support, and a free piece translates, rotates and falls as one
 * body. Bending viscosity is internal: it does not resist a rigid rotation.
 *   npm run test:gpu
 * Exits non-zero on failure.
 */
import { ComputeModule, createBuffer, readBuffer } from '../../src/gpu/util';
import { Topology } from '../../src/simulation/gpu/topology';
import type { GpuSolver } from '../../src/simulation/gpu/solver';
import common from '../../src/simulation/gpu/wgsl/common.wgsl?raw';
import drag from '../../src/simulation/gpu/wgsl/drag.wgsl?raw';
import assemble from '../../src/simulation/gpu/wgsl/assemble.wgsl?raw';
import { drain, gpuDevice } from '../../tools/gpu/env';
import { settledSolver } from '../../tools/gpu/scripts/render';

const results: Array<{ name: string; ok: boolean }> = [];
function check(name: string, ok: boolean, detail: string): void {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
}

type Vec3 = [number, number, number];
const SEG_EXISTS = 1;
const SEG_BROKEN = 2;

interface Fixture {
  solver: GpuSolver;
  topology: Topology;
  initial: Float32Array;
  segs0: Uint32Array;
  rest: (n: number, f: number) => number;
  nx: number;
}

/**
 * Only `placed` nodes in the scene (every other node retired), only `spans` (owner, family) intact,
 * and the given velocities; the connectivity analysis then detaches what no anchor reaches.
 */
async function isolate(fx: Fixture, placed: Map<number, Vec3>, spans: Array<[number, number]>, velocity = new Map<number, Vec3>(), anchors: number[] = []): Promise<void> {
  const { solver, topology, initial, segs0 } = fx;
  const xs = initial.slice();
  for (let n = 0; n < solver.N; n++) xs[4 * n + 3] = 2;
  for (const [n, p] of placed) xs.set([...p, 0], 4 * n);
  const segs = segs0.slice();
  const fs = new Float32Array(segs.buffer);
  for (let q = 0; q < 2 * solver.N; q++) segs[8 * q + 5] |= SEG_BROKEN;
  for (const [n, f] of spans) {
    const q = 2 * n + f;
    segs[8 * q + 5] = SEG_EXISTS;
    segs[8 * q + 3] = 0;
    fs[8 * q + 1] = 1;
  }
  const v = new Float32Array(4 * solver.N);
  for (const [n, q] of velocity) v.set(q, 4 * n);
  topology.reset();
  solver.setState(xs, v);
  fx.solver.device.queue.writeBuffer(solver.segs, 0, segs);
  await topology.request(anchors);
}

function substeps(solver: GpuSolver, k: number): void {
  for (let q = 0; q < k; q++) {
    const enc = solver.device.createCommandEncoder();
    solver.encodeSubstep(enc);
    solver.device.queue.submit([enc.finish()]);
  }
}

const vec = (a: Float32Array, n: number): Vec3 => [a[4 * n], a[4 * n + 1], a[4 * n + 2]];
const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** Largest |length / rest - 1| over the given spans. */
function worstStrain(fx: Fixture, x: Float32Array, spans: Array<[number, number]>): number {
  let worst = 0;
  for (const [n, f] of spans) {
    const b = n + (f === 0 ? 1 : fx.nx);
    worst = Math.max(worst, Math.abs(dist(vec(x, n), vec(x, b)) / fx.rest(n, f) - 1));
  }
  return worst;
}

function setGravity(solver: GpuSolver, g: Vec3): void {
  solver.device.queue.writeBuffer(solver.simParams, 4, new Float32Array(g));
}

/** Two detached nodes joined by one intact span, moving apart along it at 0.2 m/s (zero gravity). */
async function differential(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const a = 30 * nx + 30;
  const L = fx.rest(a, 0);
  setGravity(solver, [0, 0, 0]);
  await isolate(fx, new Map([[a, [-L / 2, 0, 0.08]], [a + 1, [L / 2, 0, 0.08]]]), [[a, 0]], new Map([[a, [-0.1, 0, 0]], [a + 1, [0.1, 0, 0]]]));
  let worst = 0;
  for (let k = 0; k < 6; k++) {
    substeps(solver, 5);
    await drain(solver.device);
    worst = Math.max(worst, worstStrain(fx, new Float32Array(await readBuffer(solver.device, solver.x)), [[a, 0]]));
  }
  const x = new Float32Array(await readBuffer(solver.device, solver.x));
  const segs = new Uint32Array(await readBuffer(solver.device, solver.segs));
  const flags = segs[8 * (2 * a) + 5];
  check('detached: an intact span pulled apart by its nodes resists the extension', x[4 * a + 3] === 1 && worst < 0.01 && flags === SEG_EXISTS, `state ${x[4 * a + 3]}, largest strain ${(100 * worst).toFixed(3)} % over 0.25 s, flags ${flags}`);
}

/**
 * The same pair dropped across the edge of a mounted face: one end lands on it, the other is past the
 * edge. The landed end's contact must load the piece through its span, not stretch it into a connector
 * between a supported and a falling node: it tips over the edge as one piece (and may fall out of the
 * scene, retired).
 */
async function edge(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const a = 30 * nx + 30;
  const L = fx.rest(a, 0);
  const face = 10 * nx + 10;
  const corners = [face, face + 1, face + nx, face + nx + 1];
  const nodes0 = new Uint32Array(await readBuffer(solver.device, solver.nodes));
  const pinned = nodes0.slice();
  for (const n of corners) pinned[4 * n + 1] |= 1;
  solver.device.queue.writeBuffer(solver.nodes, 0, pinned);
  setGravity(solver, [0, 0, -9.81]);
  const placed = new Map<number, Vec3>([[a, [-L / 2, 0, 0.008]], [a + 1, [L / 2, 0, 0.008]]]);
  corners.forEach((n, i) => placed.set(n, [i % 2 ? -0.0015 : -0.00492, i < 2 ? -0.00174 : 0.00174, 0]));
  await isolate(fx, placed, [[a, 0], [face, 0], [face, 1], [face + 1, 1], [face + nx, 0]], new Map(), corners);
  let worst = 0;
  let landed = false;
  for (let k = 0; k < 12; k++) {
    substeps(solver, 4);
    await drain(solver.device);
    const x = new Float32Array(await readBuffer(solver.device, solver.x));
    worst = Math.max(worst, worstStrain(fx, x, [[a, 0]]));
    if (x[4 * a + 2] < 0.002) landed = true;
  }
  const x = new Float32Array(await readBuffer(solver.device, solver.x));
  check('detached: a pair landing across a face edge tips off as one piece', landed && worst < 0.05 && x[4 * a + 3] > 0.5, `landed ${landed}, largest strain ${(100 * worst).toFixed(2)} %, ends at z ${(1000 * x[4 * a + 2]).toFixed(1)} / ${(1000 * x[4 * a + 6]).toFixed(1)} mm after 0.4 s`);
  solver.device.queue.writeBuffer(solver.nodes, 0, nodes0);
}

/**
 * A detached 4 x 4 piece thrown with a spin about its normal: it keeps its shape and its spin, and falls.
 * The spin it loses is backward Euler's numerical dissipation, first order in h (16, 9, 5 % at h, h / 2,
 * h / 4), not a resistance of the solve.
 */
async function tumble(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const c0 = 40 * nx + 40;
  const placed = new Map<number, Vec3>();
  const velocity = new Map<number, Vec3>();
  const spans: Array<[number, number]> = [];
  const pu = fx.rest(c0, 0);
  const pv = fx.rest(c0, 1);
  const omega = 10;
  for (let j = 0; j < 4; j++) {
    for (let i = 0; i < 4; i++) {
      const n = c0 + j * nx + i;
      const p: Vec3 = [(i - 1.5) * pu, (j - 1.5) * pv, 0.1];
      placed.set(n, p);
      velocity.set(n, [-omega * p[1], omega * p[0], 0]);
      if (i < 3) spans.push([n, 0]);
      if (j < 3) spans.push([n, 1]);
    }
  }
  setGravity(solver, [0, 0, -9.81]);
  await isolate(fx, placed, spans, velocity);
  let worst = 0;
  for (let k = 0; k < 6; k++) {
    substeps(solver, Math.round(0.25 / 6 / solver.params.h));
    await drain(solver.device);
    worst = Math.max(worst, worstStrain(fx, new Float32Array(await readBuffer(solver.device, solver.x)), spans));
  }
  const x = new Float32Array(await readBuffer(solver.device, solver.x));
  const v = new Float32Array(await readBuffer(solver.device, solver.v));
  const nodes = [...placed.keys()];
  const com = [0, 1, 2].map((q) => nodes.reduce((s, n) => s + x[4 * n + q], 0) / nodes.length);
  const vcom = [0, 1, 2].map((q) => nodes.reduce((s, n) => s + v[4 * n + q], 0) / nodes.length);
  // spin about the vertical through the centre of mass: sum r x v / sum r^2 (equal masses)
  let lz = 0;
  let i2 = 0;
  for (const n of nodes) {
    const rx = x[4 * n] - com[0];
    const ry = x[4 * n + 1] - com[1];
    lz += rx * (v[4 * n + 1] - vcom[1]) - ry * (v[4 * n] - vcom[0]);
    i2 += rx * rx + ry * ry;
  }
  const spin = lz / i2;
  const detached = nodes.every((n) => x[4 * n + 3] === 1);
  const t = 6 * Math.round(0.25 / 6 / solver.params.h) * solver.params.h;
  check('detached: a free piece keeps its shape, spins and falls', detached && worst < 0.02 && spin > 0.8 * omega && vcom[2] < -0.5 * 9.81 * t, `largest strain ${(100 * worst).toFixed(2)} %, spin ${spin.toFixed(2)} of ${omega} rad/s, fall ${(-vcom[2]).toFixed(3)} m/s (free fall ${(9.81 * t).toFixed(3)}) after ${t.toFixed(2)} s`);
}

/**
 * A detached strand of five crossings laid along a small cloth patch mounted at its rim (a trampoline,
 * 0.95 mm below the strand: inside the contact distance), from rest under gravity. Both settle. With
 * nothing but inertia along its normal, the strand bounced: each touchdown, Newton's first iteration
 * (the penalty linearized outside the contact distance) let gravity sink it deep and the penalty kicked
 * it back, at centimetres per second, unless the substep starts it at its touchdown balance
 * (selfcontact.wgsl touchdown).
 */
async function strandAtRest(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const c0 = 60 * nx + 60;
  const pu = fx.rest(c0, 0);
  const pv = fx.rest(c0, 1);
  const placed = new Map<number, Vec3>();
  const spans: Array<[number, number]> = [];
  const rim: number[] = [];
  const M = 7;
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < M; i++) {
      const n = c0 + j * nx + i;
      placed.set(n, [(i - 3) * pu, (j - 3) * pv, 0]);
      if (i < M - 1) spans.push([n, 0]);
      if (j < M - 1) spans.push([n, 1]);
      if (i === 0 || j === 0 || i === M - 1 || j === M - 1) rim.push(n);
    }
  }
  // the strand: a row of the lattice far from the patch, laid diagonally across it
  const s0 = 90 * nx + 90;
  const strand: number[] = [];
  for (let i = 0; i < 5; i++) {
    const n = s0 + i;
    strand.push(n);
    placed.set(n, [(i - 2) * pu * 0.7, (i - 2) * pu * 0.7, 0.00095]);
    if (i < 4) spans.push([n, 0]);
  }
  const nodes0 = new Uint32Array(await readBuffer(solver.device, solver.nodes));
  const pinned = nodes0.slice();
  for (const n of rim) pinned[4 * n + 1] |= 1;
  solver.device.queue.writeBuffer(solver.nodes, 0, pinned);
  setGravity(solver, [0, 0, -9.81]);
  await isolate(fx, placed, spans, new Map(), rim);
  let late = 0;
  for (let k = 0; k < 12; k++) {
    substeps(solver, 10);
    await drain(solver.device);
    if (k < 6) continue;
    const v = new Float32Array(await readBuffer(solver.device, solver.v));
    for (const n of placed.keys()) late = Math.max(late, Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]));
  }
  const x = new Float32Array(await readBuffer(solver.device, solver.x));
  const gap = Math.min(...strand.map((n) => x[4 * n + 2] - x[4 * (c0 + 3 * nx + 3) + 2]));
  check('detached: a strand lying on cloth comes to rest with it', strand.every((n) => x[4 * n + 3] === 1) && late < 0.002 && gap > 0, `largest speed ${(1000 * late).toFixed(2)} mm/s over 0.5-1 s, strand ${(1000 * gap).toFixed(2)} mm above the patch centre`);
  solver.device.queue.writeBuffer(solver.nodes, 0, nodes0);
}

/**
 * The shipped bending triple (assemble.wgsl) on a bent triple whose step is a rigid rotation: the
 * viscous part of its gradient (all but ke r) vanishes; a step that bends it further is resisted.
 */
async function bendViscosity(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const a = 30 * nx + 30;
  const La = fx.rest(a, 0);
  const Lc = fx.rest(a + 1, 0);
  const out = createBuffer(solver.device, 'test.bend', 32);
  const probe = `
@group(0) @binding(10) var<storage, read_write> probeOut: array<vec4<f32>>;
@compute @workgroup_size(1)
fn probeBend() {
  let t = triple(${a}u, ${a + 1}u, ${a + 2}u, 0u, P.layers[0]);
  probeOut[0] = vec4<f32>(tripleGrad(t) - t.ke * t.r, length(t.r));
  probeOut[1] = vec4<f32>(t.ke * t.r, length(t.rn));
}`;
  const mod = new ComputeModule(solver.device, 'test.bend', common + drag + assemble + probe);
  const group = mod.bindGroup('test.bend', { 0: solver.simParams, 1: solver.nodes, 2: solver.x, 3: solver.dx, 5: solver.segs, 10: out });
  // bent by 0.3 rad at the middle node, both spans at rest length
  const th = 0.3;
  const pa: Vec3 = [-La, 0, 0];
  const pb: Vec3 = [0, 0, 0];
  const pc: Vec3 = [Lc * Math.cos(th), Lc * Math.sin(th), 0];
  const rotZ = (p: Vec3, ang: number): Vec3 => [p[0] * Math.cos(ang) - p[1] * Math.sin(ang), p[0] * Math.sin(ang) + p[1] * Math.cos(ang), p[2]];
  const rotX = (p: Vec3, ang: number): Vec3 => [p[0], p[1] * Math.cos(ang) - p[2] * Math.sin(ang), p[1] * Math.sin(ang) + p[2] * Math.cos(ang)];
  const run = async (move: (p: Vec3) => Vec3) => {
    const xs = fx.initial.slice();
    const d = new Float32Array(4 * solver.N);
    [pa, pb, pc].forEach((p, k) => {
      xs.set([...p, 0], 4 * (a + k));
      const q = move(p);
      d.set([q[0] - p[0], q[1] - p[1], q[2] - p[2], 0], 4 * (a + k));
    });
    solver.setState(xs);
    solver.device.queue.writeBuffer(solver.dx, 0, d);
    const segs = fx.segs0.slice();
    for (const q of [2 * a, 2 * (a + 1)]) segs[8 * q + 5] = SEG_EXISTS;
    solver.device.queue.writeBuffer(solver.segs, 0, segs);
    const enc = solver.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(mod.pipeline('probeBend'));
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(1);
    pass.end();
    solver.device.queue.submit([enc.finish()]);
    const r = new Float32Array(await readBuffer(solver.device, out));
    return { visc: Math.hypot(r[0], r[1], r[2]), elastic: Math.hypot(r[4], r[5], r[6]) };
  };
  // rigid: 30 degrees about the vertical, then about the first span's axis
  const rigid = await run((p) => rotX(rotZ(p, Math.PI / 6), Math.PI / 6));
  // bending further: the far node turns about the middle node by another 0.1 rad
  const bend = await run((p) => (p === pc ? rotZ(p, 0.1) : p));
  check('bending viscosity: a rigid rotation is not resisted, further bending is', rigid.visc < 1e-4 * bend.visc && bend.visc > 0, `viscous gradient ${rigid.visc.toExponential(2)} N (rigid) vs ${bend.visc.toExponential(2)} N (bend), elastic ${bend.elastic.toExponential(2)} N`);
  solver.device.queue.writeBuffer(solver.segs, 0, fx.segs0);
  solver.device.queue.writeBuffer(solver.dx, 0, new Float32Array(4 * solver.N));
}

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device, JSON.parse(process.env.SOLVER ?? '{}') as object);
  const initial = new Float32Array(await readBuffer(device, solver.x));
  const segs0 = new Uint32Array(await readBuffer(device, solver.segs));
  const restF = new Float32Array(segs0.buffer);
  const fx: Fixture = {
    solver,
    topology: new Topology(device, solver, scene),
    initial,
    segs0,
    rest: (n, f) => restF[8 * (2 * n + f)],
    nx: scene.layers[0].nx,
  };
  await differential(fx);
  await edge(fx);
  await tumble(fx);
  await strandAtRest(fx);
  await bendViscosity(fx);
  device.destroy();
  const failed = results.filter((r) => !r.ok).length;
  console.log(`=== ${results.length - failed}/${results.length} passed`);
  if (failed) throw new Error(`${failed} detached-material check(s) failed`);
}
