/**
 * Floor contact on the production kernels (headless: Node + Dawn). A piece torn out of the sheet falls
 * onto the floor and comes to rest lying on it; a piece thrown at it never passes below the cloth's half
 * thickness; a piece sliding or spinning on it stops as Coulomb friction stops it; a long flap hanging
 * from the mount lies along it where it reaches it.
 *   npm run test:gpu
 * Exits non-zero on failure.
 */
import { readBuffer } from '../../src/gpu/util';
import { Topology } from '../../src/simulation/gpu/topology';
import type { GpuSolver } from '../../src/simulation/gpu/solver';
import { FLOOR_CONTACT, FLOOR_Z } from '../../src/simulation/floor';
import { DENIM } from '../../src/simulation/material';
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
const HALF = 0.5 * DENIM.thickness;
const DELTA = HALF + FLOOR_CONTACT.bulge;

interface Fixture {
  solver: GpuSolver;
  topology: Topology;
  initial: Float32Array;
  segs0: Uint32Array;
  rest: (n: number, f: number) => number;
  nx: number;
  /** Height above the floor at which node n rests on it (its weight carried by the penalty). */
  resting: (n: number) => number;
}

function substeps(solver: GpuSolver, k: number): void {
  for (let q = 0; q < k; q++) {
    const enc = solver.device.createCommandEncoder();
    solver.encodeSubstep(enc);
    solver.device.queue.submit([enc.finish()]);
  }
}

/**
 * Run `seconds` in chunks; per chunk the lowest height of `nodes` above the floor and (from `lateFrom`
 * s on) their largest speed.
 */
async function watch(fx: Fixture, nodes: number[], seconds: number, lateFrom: number, floorZ: number): Promise<{ lowest: number; late: number; x: Float32Array; v: Float32Array }> {
  const { solver } = fx;
  const chunk = 6;
  const chunks = Math.round(seconds / solver.params.h / chunk);
  let lowest = Infinity;
  let late = 0;
  let x = new Float32Array(0);
  let v = new Float32Array(0);
  for (let k = 0; k < chunks; k++) {
    substeps(solver, chunk);
    await drain(solver.device);
    x = new Float32Array(await readBuffer(solver.device, solver.x));
    v = new Float32Array(await readBuffer(solver.device, solver.v));
    for (const n of nodes) {
      if (x[4 * n + 3] > 1.5) continue;
      lowest = Math.min(lowest, x[4 * n + 2] - floorZ);
      if ((k + 1) * chunk * solver.params.h >= lateFrom) late = Math.max(late, Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]));
    }
  }
  return { lowest, late, x, v };
}

/** Largest |length / rest - 1| over the intact spans among `nodes`. */
function worstStrain(fx: Fixture, x: Float32Array, spans: Array<[number, number]>): number {
  let worst = 0;
  for (const [n, f] of spans) {
    const b = n + (f === 0 ? 1 : fx.nx);
    const d = Math.hypot(x[4 * n] - x[4 * b], x[4 * n + 1] - x[4 * b + 1], x[4 * n + 2] - x[4 * b + 2]);
    worst = Math.max(worst, Math.abs(d / fx.rest(n, f) - 1));
  }
  return worst;
}

/** Only `placed` nodes in the scene (every other one retired), only `spans` intact, with velocities. */
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
  solver.device.queue.writeBuffer(solver.segs, 0, segs);
  await topology.request(anchors);
}

/** An M x M detached patch centred at (0, 0, z) with all its spans, and its spans. */
function patch(fx: Fixture, c0: number, M: number, z: number): { placed: Map<number, Vec3>; spans: Array<[number, number]> } {
  const pu = fx.rest(c0, 0);
  const pv = fx.rest(c0, 1);
  const placed = new Map<number, Vec3>();
  const spans: Array<[number, number]> = [];
  for (let j = 0; j < M; j++) {
    for (let i = 0; i < M; i++) {
      const n = c0 + j * fx.nx + i;
      placed.set(n, [(i - (M - 1) / 2) * pu, (j - (M - 1) / 2) * pv, z]);
      if (i < M - 1) spans.push([n, 0]);
      if (j < M - 1) spans.push([n, 1]);
    }
  }
  return { placed, spans };
}

/** A piece cut out of the mounted sheet falls the 12 cm to the floor and comes to rest lying on it. */
async function torn(fx: Fixture): Promise<void> {
  const { solver, nx, topology, initial, segs0 } = fx;
  const i0 = 70;
  const j0 = 40;
  const inside = (i: number, j: number) => i >= i0 && i < i0 + 8 && j >= j0 && j < j0 + 7;
  const piece: number[] = [];
  const spans: Array<[number, number]> = [];
  const segs = segs0.slice();
  for (let j = j0 - 1; j < j0 + 8; j++) {
    for (let i = i0 - 1; i < i0 + 9; i++) {
      const n = j * nx + i;
      if (inside(i, j) !== inside(i + 1, j)) segs[8 * (2 * n) + 5] |= SEG_BROKEN;
      if (inside(i, j) !== inside(i, j + 1)) segs[8 * (2 * n + 1) + 5] |= SEG_BROKEN;
      if (!inside(i, j)) continue;
      piece.push(n);
      if (inside(i + 1, j)) spans.push([n, 0]);
      if (inside(i, j + 1)) spans.push([n, 1]);
    }
  }
  solver.setFloor(FLOOR_Z);
  topology.reset();
  solver.setState(initial);
  solver.device.queue.writeBuffer(solver.segs, 0, segs);
  await topology.request();
  const r = await watch(fx, piece, 1.5, 1.0, FLOOR_Z);
  const live = piece.filter((n) => r.x[4 * n + 3] < 1.5);
  const highest = Math.max(...live.map((n) => r.x[4 * n + 2] - FLOOR_Z));
  const strain = worstStrain(fx, r.x, spans);
  check(
    'floor: a piece torn out of the sheet lands on the floor and rests there',
    live.length === piece.length && r.lowest >= HALF * 0.999 && highest < DELTA && r.late < 0.002 && strain < 0.02,
    `${live.length} of ${piece.length} in the scene, lowest ${(1000 * r.lowest).toFixed(3)} mm, highest ${(1000 * highest).toFixed(3)} mm above the floor (contact ${(1000 * DELTA).toFixed(2)} mm), largest speed ${(1000 * r.late).toFixed(2)} mm/s over 1-1.5 s, largest strain ${(100 * strain).toFixed(2)} %`,
  );
  topology.reset();
  solver.setState(initial);
  solver.device.queue.writeBuffer(solver.segs, 0, segs0);
}

/** A 5 x 5 piece thrown flat and tilted at the floor at 4 m/s from 1 cm: never below half its thickness. */
async function thrown(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const fz = -0.02;
  solver.setFloor(fz);
  const { placed, spans } = patch(fx, 40 * nx + 40, 5, fz + 0.01);
  // tilted 30 degrees about x: one edge lands first
  for (const [n, p] of placed) placed.set(n, [p[0], p[1] * Math.cos(0.5), fz + 0.01 + p[1] * Math.sin(0.5)]);
  const velocity = new Map<number, Vec3>([...placed.keys()].map((n) => [n, [0.3, 0, -4]]));
  await isolate(fx, placed, spans, velocity);
  const r = await watch(fx, [...placed.keys()], 0.6, 0.4, fz);
  const strain = worstStrain(fx, r.x, spans);
  check('floor: a piece thrown at the floor never passes below half its thickness', r.lowest >= HALF * 0.999 && strain < 0.03, `lowest ${(1000 * r.lowest).toFixed(3)} mm above the floor (half thickness ${(1000 * HALF).toFixed(2)}), largest strain ${(100 * strain).toFixed(2)} %, largest speed ${(1000 * r.late).toFixed(2)} mm/s over 0.4-0.6 s`);
}

/** A piece lying on the floor, sent sliding at 0.25 m/s: it stops within the Coulomb distance v^2 / (2 mu g). */
async function slide(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const fz = -0.02;
  solver.setFloor(fz);
  const { placed, spans } = patch(fx, 60 * nx + 60, 5, fz + fx.resting(60 * nx + 62));
  const v0 = 0.25;
  const velocity = new Map<number, Vec3>([...placed.keys()].map((n) => [n, [v0, 0, 0]]));
  await isolate(fx, placed, spans, velocity);
  const nodes = [...placed.keys()];
  const x0 = nodes.reduce((s, n) => s + placed.get(n)![0], 0) / nodes.length;
  const r = await watch(fx, nodes, 0.4, 0.3, fz);
  const x1 = nodes.reduce((s, n) => s + r.x[4 * n], 0) / nodes.length;
  const coulomb = (v0 * v0) / (2 * FLOOR_CONTACT.friction * 9.81);
  const ratio = (x1 - x0) / coulomb;
  check('floor: a piece sliding on the floor stops within the Coulomb distance', ratio > 0.8 && ratio < 1.1 && r.late < 0.002 && r.lowest >= HALF * 0.999, `slid ${(1000 * (x1 - x0)).toFixed(2)} mm (Coulomb ${(1000 * coulomb).toFixed(2)} mm, ratio ${ratio.toFixed(3)}), largest speed ${(1000 * r.late).toFixed(2)} mm/s over 0.3-0.4 s`);
}

/**
 * A piece spinning flat on the floor at 10 rad/s: Coulomb friction stops it within its torque's time,
 * w0 / (mu g sum m r / sum m r^2) (a piece in the solve's first iterations kept spinning for a second).
 */
async function spin(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const fz = -0.02;
  solver.setFloor(fz);
  const c0 = 60 * nx + 60;
  const { placed, spans } = patch(fx, c0, 5, fz + fx.resting(c0 + 2 * nx + 2));
  const w0 = 10;
  const velocity = new Map<number, Vec3>([...placed].map(([n, p]) => [n, [-w0 * p[1], w0 * p[0], 0]]));
  await isolate(fx, placed, spans, velocity);
  const nodes = [...placed.keys()];
  const sr = nodes.reduce((t, n) => t + Math.hypot(...placed.get(n)!.slice(0, 2)), 0);
  const sr2 = nodes.reduce((t, n) => t + placed.get(n)!.slice(0, 2).reduce((q, c) => q + c * c, 0), 0);
  const coulomb = w0 / ((FLOOR_CONTACT.friction * 9.81 * sr) / sr2);
  let stopped = Infinity;
  let w = w0;
  for (let k = 1; k <= Math.round(0.1 / solver.params.h); k++) {
    substeps(solver, 1);
    await drain(solver.device);
    const x = new Float32Array(await readBuffer(solver.device, solver.x));
    const v = new Float32Array(await readBuffer(solver.device, solver.v));
    const c = [0, 1].map((q) => nodes.reduce((t, n) => t + x[4 * n + q], 0) / nodes.length);
    let lz = 0;
    let i2 = 0;
    for (const n of nodes) {
      const rx = x[4 * n] - c[0];
      const ry = x[4 * n + 1] - c[1];
      lz += rx * v[4 * n + 1] - ry * v[4 * n];
      i2 += rx * rx + ry * ry;
    }
    w = lz / i2;
    if (Math.abs(w) < 0.01 * w0) {
      stopped = k * solver.params.h;
      break;
    }
  }
  check('floor: a piece spinning on the floor stops within the Coulomb time', stopped <= 2 * coulomb, `stopped after ${(1000 * stopped).toFixed(1)} ms (Coulomb ${(1000 * coulomb).toFixed(1)} ms), spin ${w.toFixed(3)} rad/s`);
}

/**
 * A 3-wide strip mounted at its top 4 cm above the floor and 10 cm long, laid out hanging straight down
 * past the floor's height (the part below it starts on the floor, bent out along it): it comes to rest
 * hanging, then lying along the floor, without passing below it.
 */
async function trailing(fx: Fixture): Promise<void> {
  const { solver, nx } = fx;
  const fz = -0.04;
  solver.setFloor(fz);
  const c0 = 20 * nx + 100;
  const pv = fx.rest(c0, 1);
  const pu = fx.rest(c0, 0);
  const L = 30;
  const placed = new Map<number, Vec3>();
  const spans: Array<[number, number]> = [];
  const top: number[] = [];
  const ground = fz + fx.resting(c0 + 1);
  for (let j = 0; j < L; j++) {
    for (let i = 0; i < 3; i++) {
      const n = c0 + j * nx + i;
      // down to the floor, then along it (+y)
      const drop = -j * pv;
      const p: Vec3 = drop >= ground ? [(i - 1) * pu, 0, drop] : [(i - 1) * pu, ground - drop, ground];
      placed.set(n, p);
      if (i < 2) spans.push([n, 0]);
      if (j < L - 1) spans.push([n, 1]);
      if (j === 0) top.push(n);
    }
  }
  const nodes0 = new Uint32Array(await readBuffer(solver.device, solver.nodes));
  const pinned = nodes0.slice();
  for (const n of top) pinned[4 * n + 1] |= 1;
  solver.device.queue.writeBuffer(solver.nodes, 0, pinned);
  await isolate(fx, placed, spans, new Map(), top);
  const nodes = [...placed.keys()];
  const r = await watch(fx, nodes, 1.5, 1.0, fz);
  const lying = nodes.filter((n) => r.x[4 * n + 2] - fz < DELTA).length;
  const strain = worstStrain(fx, r.x, spans);
  check('floor: a flap hanging from the mount lies along the floor where it reaches it', r.lowest >= HALF * 0.999 && lying > 0 && r.late < 0.002 && strain < 0.02, `${lying} of ${nodes.length} nodes lying on it, lowest ${(1000 * r.lowest).toFixed(3)} mm above it, largest speed ${(1000 * r.late).toFixed(2)} mm/s over 1-1.5 s, largest strain ${(100 * strain).toFixed(2)} %`);
  solver.device.queue.writeBuffer(solver.nodes, 0, nodes0);
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
    resting: (n) => DELTA - Math.sqrt((scene.mass[n] * 9.81 * DELTA) / FLOOR_CONTACT.stiffness),
  };
  await torn(fx);
  await thrown(fx);
  await slide(fx);
  await spin(fx);
  await trailing(fx);
  device.destroy();
  const failed = results.filter((r) => !r.ok).length;
  console.log(`=== ${results.length - failed}/${results.length} passed`);
  if (failed) throw new Error(`${failed} floor check(s) failed`);
}
