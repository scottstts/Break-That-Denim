/**
 * GPU mechanics vs the f64 reference on identical states (headless: Node + Dawn).
 *   npm run test:gpu
 * Exits non-zero on failure. GPU work is submitted in short chunks.
 */
import { readBuffer } from '../../src/gpu/util';
import { DENIM } from '../../src/simulation/material';
import { assembleFineOperator, buildHierarchy, offsetIndex, pcg, smoothLines, STENCIL, vcycle, type MGConfig } from '../../src/simulation/reference/newtonMultigrid';
import { ReferenceSolver } from '../../src/simulation/reference/solver';
import { compileScene, SEG_BROKEN, type LayerSpec } from '../../src/simulation/scene';
import { COARSE_ROW, FINE_ROW, GpuSolver, GPU_DEFAULTS } from '../../src/simulation/gpu/solver';
import { gpuDevice } from '../../tools/gpu/env';

const results: Array<{ name: string; ok: boolean }> = [];
function check(name: string, ok: boolean, detail: string): void {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
}

/** The reference schedule matching GPU_DEFAULTS. */
const MG: MGConfig = { maxCoarsestNodes: GPU_DEFAULTS.maxCoarsestNodes, coarseSweeps: GPU_DEFAULTS.coarseSweeps, lineColors: 3, lagCoarse: true, lineSearch: true };

/** Production schedule without self-contact (the f64 reference has no contact). */
const PARAMS = { ...GPU_DEFAULTS, selfContact: false };

function coupon(nu: number, nv: number, extra: Partial<LayerSpec> = {}): LayerSpec {
  return {
    material: DENIM,
    bundle: [3, 2],
    sizeU: (nu - 1) * 3 * DENIM.pitchU,
    sizeV: (nv - 1) * 2 * DENIM.pitchV,
    z: 0,
    angle: 0.3,
    compressU: 0.03,
    compressV: 0.02,
    shear: 0.02,
    collarRows: 2,
    wear: { center: [0, 0], axis: [1, 0], radii: [1, 1], amount: 0, cellSize: 0, featureScale: 0.01, defectThreshold: 2, defectDensity: 0 },
    wearStrengthLoss: [0, 0],
    seed: 7,
    perturbation: 0.4e-3,
    ...extra,
  };
}

/**
 * Operator rows (nodes) touched by an active span whose strain at the predicted iterate lies within
 * rounding of the axial law's tension/compression kink, where the tangent jumps (eaCompress -> eaLow)
 * and f32 and f64 may legitimately take different branches. Crossing slip relaxes spans next to free or
 * mounted boundaries to zero tension, so such spans occur in torn fixtures. Branch boundaries are
 * tested separately (design: numerical thresholds), not inside smooth-operator comparisons.
 */
function kinkRows(cpu: ReferenceSolver, N: number): Set<number> {
  const rows = new Set<number>();
  for (let sIdx = 0; sIdx < 2 * N; sIdx++) {
    if (!cpu.segActive(sIdx)) continue;
    const a = sIdx >> 1;
    const b = cpu.segOther(sIdx);
    let l = 0;
    for (let q = 0; q < 3; q++) l += (cpu.x[3 * b + q] + cpu.dx[3 * b + q] - cpu.x[3 * a + q] - cpu.dx[3 * a + q]) ** 2;
    if (Math.abs(Math.sqrt(l) / cpu.restLength[sIdx] - 1) < 1e-6) {
      rows.add(a);
      rows.add(b);
    }
  }
  return rows;
}

/** maxAbsRelToScale over operator rows (STENCIL x 9 entries per node) not in `skip`. */
function maxAbsRelToScaleRows(a: ArrayLike<number>, b: ArrayLike<number>, skip: Set<number>): number {
  const row = STENCIL * 9;
  let scale = 0;
  let e = 0;
  for (let q = 0; q < b.length; q++) {
    scale = Math.max(scale, Math.abs(b[q]));
    if (!skip.has(Math.floor(q / row))) e = Math.max(e, Math.abs(a[q] - b[q]));
  }
  return e / Math.max(scale, 1e-30);
}

function maxAbsRelToScale(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let scale = 0;
  let e = 0;
  for (let q = 0; q < b.length; q++) {
    scale = Math.max(scale, Math.abs(b[q]));
    e = Math.max(e, Math.abs(a[q] - b[q]));
  }
  return e / Math.max(scale, 1e-30);
}

async function readVec(device: GPUDevice, buf: GPUBuffer, n: number): Promise<Float64Array> {
  const f = new Float32Array(await readBuffer(device, buf, 16 * n));
  const o = new Float64Array(3 * n);
  for (let k = 0; k < n; k++) for (let q = 0; q < 3; q++) o[3 * k + q] = f[4 * k + q];
  return o;
}

function writeVec(device: GPUDevice, buf: GPUBuffer, v: Float64Array): void {
  const n = v.length / 3;
  const f = new Float32Array(4 * n);
  for (let k = 0; k < n; k++) for (let q = 0; q < 3; q++) f[4 * k + q] = v[3 * k + q];
  device.queue.writeBuffer(buf, 0, f);
}

const FINE_OFFS: Array<[number, number]> = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]];

/** GPU operator rows (vec4-aligned fine or coarse layout) -> 25-offset CPU layout. */
async function readOperator(device: GPUDevice, buf: GPUBuffer, N: number, fine: boolean): Promise<Float64Array> {
  const stride = fine ? FINE_ROW : COARSE_ROW;
  const f = new Float32Array(await readBuffer(device, buf, N * stride * 4));
  const A = new Float64Array(N * STENCIL * 9);
  const put = (n: number, di: number, dj: number, b: number) => {
    const [xx, xy, xz, yy, yz, zz] = [f[b], f[b + 1], f[b + 2], f[b + 3], f[b + 4], f[b + 5]];
    A.set([xx, xy, xz, xy, yy, yz, xz, yz, zz], (n * STENCIL + offsetIndex(di, dj)) * 9);
  };
  const scal = (n: number, di: number, dj: number, v: number) => A.set([v, 0, 0, 0, v, 0, 0, 0, v], (n * STENCIL + offsetIndex(di, dj)) * 9);
  for (let n = 0; n < N; n++) {
    const r = n * stride;
    put(n, 0, 0, r);
    if (fine) {
      for (let k = 0; k < 8; k++) put(n, FINE_OFFS[k][0], FINE_OFFS[k][1], r + 8 + 8 * k);
      scal(n, 2, 0, f[r + 6]);
      scal(n, -2, 0, f[r + 7]);
      scal(n, 0, 2, f[r + 14]);
      scal(n, 0, -2, f[r + 15]);
    } else {
      for (let k = 0; k < 24; k++) {
        const o = k >= 12 ? k + 1 : k;
        put(n, (o % 5) - 2, Math.floor(o / 5) - 2, r + 8 + 8 * k);
      }
    }
  }
  return A;
}

async function submit(device: GPUDevice, fn: (pass: GPUComputePassEncoder) => void): Promise<void> {
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  fn(pass);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}

function copyState(cpu: ReferenceSolver, N: number): { xs: Float32Array; vs: Float32Array } {
  const xs = new Float32Array(4 * N);
  const vs = new Float32Array(4 * N);
  for (let n = 0; n < N; n++) {
    for (let q = 0; q < 3; q++) {
      xs[4 * n + q] = cpu.x[3 * n + q];
      vs[4 * n + q] = cpu.v[3 * n + q];
    }
  }
  // the CPU continues from the f32-rounded state so both start identically
  for (let n = 0; n < N; n++) {
    for (let q = 0; q < 3; q++) {
      cpu.x[3 * n + q] = xs[4 * n + q];
      cpu.v[3 * n + q] = vs[4 * n + q];
    }
  }
  return { xs, vs };
}

async function correctness(device: GPUDevice): Promise<void> {
  console.log('--- correctness: coupon vs f64 reference (production schedule)');
  const scene = compileScene([coupon(48, 40)]);
  const params = PARAMS;
  const cpu = new ReferenceSolver(scene, { h: params.h, gravity: params.gravity, newtonIterations: params.newtonIterations, pcgIterations: params.pcgIterations, mg: MG });
  cpu.historyTolerance = params.historyTolerance;
  for (let k = 0; k < 12; k++) cpu.substep();
  const gpu = new GpuSolver(device, scene, params);
  const N = scene.nodeCount;
  const { xs, vs } = copyState(cpu, N);
  gpu.setState(xs, vs);
  gpu.setSegments(cpu.restLength, cpu.maxExt, cpu.segFlags, cpu.tails);

  cpu.predict();
  await submit(device, (p) => {
    gpu.encodePredict(p);
    gpu.encodeAssemble(p);
  });
  const Acpu = assembleFineOperator(cpu);
  const Agpu = await readOperator(device, gpu.levels[0].op, N, true);
  // f32 positions shift strains by ~1e-5; on the steep decrimping ramp of the axial tangent that moves
  // diagonal entries by a few 1e-5 of the operator scale (input rounding, not kernel error)
  check('fine operator blocks', maxAbsRelToScale(Agpu, Acpu.A) < 5e-5, `maxAbs/scale=${maxAbsRelToScale(Agpu, Acpu.A).toExponential(2)}`);
  const g = new Float64Array(3);
  const bCpu = new Float64Array(3 * N);
  for (let n = 0; n < N; n++) {
    cpu.nodeGradient(n, g);
    for (let q = 0; q < 3; q++) bCpu[3 * n + q] = -g[q];
  }
  const bGpu = await readVec(device, gpu.levels[0].B, N);
  check('newton rhs', maxAbsRelToScale(bGpu, bCpu) < 1e-3, `maxAbs/scale=${maxAbsRelToScale(bGpu, bCpu).toExponential(2)}`);

  // Galerkin coarse operators from the GPU fine operator (isolates RAP)
  Acpu.A.set(Agpu);
  const levels = buildHierarchy(Acpu, MG);
  check('level count', levels.length === gpu.levels.length, `cpu=${levels.length} gpu=${gpu.levels.length}`);
  for (let k = 1; k < Math.min(levels.length, gpu.levels.length); k++) {
    const Ak = await readOperator(device, gpu.levels[k].op, gpu.levels[k].N, false);
    const e = maxAbsRelToScale(Ak, levels[k].A.A);
    check(`galerkin level ${k}`, e < 1e-5, `maxAbs/scale=${e.toExponential(2)}`);
  }

  const zc = new Float64Array(3 * N);
  vcycle(levels, 0, zc, bGpu, 1, 1, params.coarseSweeps, smoothLines, 3);
  writeVec(device, gpu.levels[0].B, bGpu);
  device.queue.writeBuffer(gpu.levels[0].X, 0, new Float32Array(4 * N));
  await submit(device, (p) => gpu.encodeVCycle(p, 0));
  const zg = await readVec(device, gpu.levels[0].X, N);
  check('V-cycle', maxAbsRelToScale(zg, zc) < 5e-3, `maxAbs/scale=${maxAbsRelToScale(zg, zc).toExponential(2)}`);

  const dc = new Float64Array(3 * N);
  pcg(levels, bGpu, dc, params.pcgIterations, undefined, smoothLines, MG);
  writeVec(device, gpu.levels[0].B, bGpu);
  await submit(device, (p) => gpu.encodePCG(p));
  const dg = await readVec(device, gpu.delta, N);
  check('PCG step', maxAbsRelToScale(dg, dc) < 2e-2, `maxAbs/scale=${maxAbsRelToScale(dg, dc).toExponential(2)}`);

  // line-search energy changes along the GPU direction
  const eCpu = cpu.energyChange(dg, [1, 0.5, 0.25]);
  await submit(device, (p) => gpu.encodeLineSearch(p));
  const part = new Float32Array(await readBuffer(device, gpu.partial4));
  const eGpu = [0, 0, 0];
  for (let k = 0; k < part.length / 4; k++) for (let q = 0; q < 3; q++) eGpu[q] += part[4 * k + q];
  const eScale = Math.max(...eCpu.map(Math.abs));
  const eErr = Math.max(...eGpu.map((v, q) => Math.abs(v - eCpu[q]))) / eScale;
  check('line-search energy changes', eErr < 1e-2, `gpu [${eGpu.map((v) => v.toExponential(3)).join(', ')}] cpu [${eCpu.map((v) => v.toExponential(3)).join(', ')}]`);

  await dynamics(device, scene, cpu, gpu, 'production schedule', 0.25);
}

/**
 * Trajectory agreement from the same state. With a converged schedule the GPU must track the f64
 * reference closely; the production schedule is inexact, so f32 differences can select different
 * buckling paths of this unsettled coupon and only gross agreement is required.
 */
async function dynamics(device: GPUDevice, scene: ReturnType<typeof compileScene>, cpu: ReferenceSolver, gpu: GpuSolver, label: string, tol: number): Promise<void> {
  const N = scene.nodeCount;
  const { xs, vs } = copyState(cpu, N);
  gpu.setState(xs, vs);
  gpu.setSegments(cpu.restLength, cpu.maxExt, cpu.segFlags, cpu.tails);
  const steps = 60;
  for (let k = 0; k < steps; k++) cpu.substep();
  for (let k = 0; k < steps; k += 4) {
    const enc = device.createCommandEncoder();
    for (let q = 0; q < 4; q++) gpu.encodeSubstep(enc);
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  const xg = await readVec(device, gpu.x, N);
  let moved = 0;
  for (let q = 0; q < xg.length; q++) moved = Math.max(moved, Math.abs(cpu.x[q] - xs[((q / 3) | 0) * 4 + (q % 3)]));
  let err = 0;
  for (let q = 0; q < xg.length; q++) err = Math.max(err, Math.abs(xg[q] - cpu.x[q]));
  check(`dynamics ${steps} substeps, ${label}`, err < tol * moved, `max |x_gpu - x_cpu| = ${(err * 1e6).toFixed(2)} um, max motion ${(moved * 1e6).toFixed(1)} um`);
}

async function convergedDynamics(device: GPUDevice): Promise<void> {
  console.log('--- converged schedule dynamics');
  const scene = compileScene([coupon(48, 40)]);
  const params = { ...PARAMS, newtonIterations: 5, newtonMax: 5, pcgIterations: 3 };
  const cpu = new ReferenceSolver(scene, { h: params.h, gravity: params.gravity, newtonIterations: 5, pcgIterations: 3, mg: MG });
  cpu.historyTolerance = params.historyTolerance;
  for (let k = 0; k < 12; k++) cpu.substep();
  const gpu = new GpuSolver(device, scene, params);
  await dynamics(device, scene, cpu, gpu, 'converged schedule', 0.03);
}

/**
 * Torn topology: a flap cut on three sides plus a slit. Exercises the shear diagonals and air drag of
 * hole-edge nodes and the connectivity-aware prolongation (Galerkin levels, V-cycle, PCG).
 */
async function torn(device: GPUDevice): Promise<void> {
  console.log('--- torn coupon: flap and slit');
  const scene = compileScene([coupon(48, 40)]);
  const params = PARAMS;
  const cpu = new ReferenceSolver(scene, { h: params.h, gravity: params.gravity, newtonIterations: params.newtonIterations, pcgIterations: params.pcgIterations, mg: MG });
  cpu.historyTolerance = params.historyTolerance;
  const nx = scene.layers[0].nx;
  const cut = (i: number, j: number, f: number) => (cpu.segFlags[2 * (j * nx + i) + f] |= SEG_BROKEN);
  for (let j = 12; j < 26; j++) {
    cut(14, j, 0);
    cut(31, j, 0);
  }
  for (let i = 15; i <= 31; i++) cut(i, 11, 1);
  for (let i = 6; i < 13; i++) cut(i, 30, 1);
  for (let k = 0; k < 12; k++) cpu.substep();
  const gpu = new GpuSolver(device, scene, params);
  const N = scene.nodeCount;
  const { xs, vs } = copyState(cpu, N);
  gpu.setState(xs, vs);
  gpu.setSegments(cpu.restLength, cpu.maxExt, cpu.segFlags, cpu.tails);
  cpu.predict();
  await submit(device, (p) => {
    gpu.encodePredict(p);
    gpu.encodeAssemble(p);
  });
  const A = assembleFineOperator(cpu);
  const Agpu = await readOperator(device, gpu.levels[0].op, N, true);
  const kinks = kinkRows(cpu, N);
  const eOp = maxAbsRelToScaleRows(Agpu, A.A, kinks);
  check('torn: fine operator', eOp < 5e-5, `${eOp.toExponential(2)} (${kinks.size} rows at the axial kink excluded)`);
  A.A.set(Agpu);
  const levels = buildHierarchy(A, MG);
  for (let k = 1; k < Math.min(levels.length, gpu.levels.length); k++) {
    const Ak = await readOperator(device, gpu.levels[k].op, gpu.levels[k].N, false);
    const e = maxAbsRelToScale(Ak, levels[k].A.A);
    check(`torn: galerkin level ${k}`, e < 1e-5, `maxAbs/scale=${e.toExponential(2)}`);
  }
  const b = await readVec(device, gpu.levels[0].B, N);
  const zc = new Float64Array(3 * N);
  vcycle(levels, 0, zc, b, 1, 1, params.coarseSweeps, smoothLines, 3);
  device.queue.writeBuffer(gpu.levels[0].X, 0, new Float32Array(4 * N));
  await submit(device, (p) => gpu.encodeVCycle(p, 0));
  const zg = await readVec(device, gpu.levels[0].X, N);
  check('torn: V-cycle', maxAbsRelToScale(zg, zc) < 5e-3, `maxAbs/scale=${maxAbsRelToScale(zg, zc).toExponential(2)}`);
  const dc = new Float64Array(3 * N);
  pcg(levels, b, dc, params.pcgIterations, undefined, smoothLines, MG);
  await submit(device, (p) => gpu.encodePCG(p));
  const dg = await readVec(device, gpu.delta, N);
  check('torn: PCG step', maxAbsRelToScale(dg, dc) < 2e-2, `maxAbs/scale=${maxAbsRelToScale(dg, dc).toExponential(2)}`);
  const eCpu = cpu.energyChange(dg, [1, 0.5, 0.25]);
  await submit(device, (p) => gpu.encodeLineSearch(p));
  const part = new Float32Array(await readBuffer(device, gpu.partial4));
  const eGpu = [0, 0, 0];
  for (let k = 0; k < part.length / 4; k++) for (let q = 0; q < 3; q++) eGpu[q] += part[4 * k + q];
  const eScale = Math.max(...eCpu.map(Math.abs));
  const eErr = Math.max(...eGpu.map((v, q) => Math.abs(v - eCpu[q]))) / eScale;
  check('torn: line-search energy changes', eErr < 1e-2, `gpu [${eGpu.map((v) => v.toExponential(3)).join(', ')}] cpu [${eCpu.map((v) => v.toExponential(3)).join(', ')}]`);
}

/** Long yarn lines (as in the full scene): GPU smoother and PCG against the f64 reference. */
async function longLines(device: GPUDevice): Promise<void> {
  console.log('--- long lines: 115 x 30 and 30 x 164 coupons');
  for (const [nu, nv] of [
    [115, 30],
    [30, 164],
  ]) {
    const scene = compileScene([coupon(nu, nv)]);
    const params = PARAMS;
    const cpu = new ReferenceSolver(scene, { h: params.h, gravity: params.gravity, newtonIterations: 1, pcgIterations: params.pcgIterations, mg: MG });
    const gpu = new GpuSolver(device, scene, params);
    const N = scene.nodeCount;
    cpu.predict();
    await submit(device, (p) => {
      gpu.encodePredict(p);
      gpu.encodeAssemble(p);
    });
    const A = assembleFineOperator(cpu);
    const Agpu = await readOperator(device, gpu.levels[0].op, N, true);
    check(`long ${nu}x${nv}: fine operator`, maxAbsRelToScale(Agpu, A.A) < 2e-5, `${maxAbsRelToScale(Agpu, A.A).toExponential(2)}`);
    A.A.set(Agpu);
    const levels = buildHierarchy(A, MG);
    const b = await readVec(device, gpu.levels[0].B, N);
    const xc = new Float64Array(3 * N);
    smoothLines(levels[0].A, xc, b);
    device.queue.writeBuffer(gpu.levels[0].X, 0, new Float32Array(4 * N));
    await submit(device, (p) => gpu.encodeSmooth(p, 0, false));
    const xg = await readVec(device, gpu.levels[0].X, N);
    check(`long ${nu}x${nv}: one smoothing sweep`, maxAbsRelToScale(xg, xc) < 1e-3, `${maxAbsRelToScale(xg, xc).toExponential(2)}`);
    const dc = new Float64Array(3 * N);
    pcg(levels, b, dc, params.pcgIterations, undefined, smoothLines, MG);
    await submit(device, (p) => gpu.encodePCG(p));
    const dg = await readVec(device, gpu.delta, N);
    check(`long ${nu}x${nv}: PCG step`, maxAbsRelToScale(dg, dc) < 2e-2, `${maxAbsRelToScale(dg, dc).toExponential(2)}`);
  }
}

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  await correctness(device);
  await convergedDynamics(device);
  await torn(device);
  await longLines(device);
  device.destroy();
  const failed = results.filter((r) => !r.ok).length;
  console.log(`=== ${results.length - failed}/${results.length} passed`);
  if (failed) throw new Error(`${failed} GPU check(s) failed`);
}
