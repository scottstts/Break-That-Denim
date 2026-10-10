/**
 * Accuracy study (CPU f64 reference): timestep and multigrid schedule on the bundled denim lattice.
 * A worn coupon is pulled by a prescribed patch (lift + drag), held, released and left to ring down.
 * Each configuration is compared with an over-resolved run at matching times.
 *
 *   node tools/gpu/run.mjs tools/studies/timestep.ts <config> [out.json]
 * configs: truth | h60 | h120 | h60-lean | h120-lean | ...
 */
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { DENIM } from '../../src/simulation/material';
import { MG_DEFAULTS, newtonStepPCG, type MGConfig } from '../../src/simulation/reference/newtonMultigrid';
import { ReferenceSolver } from '../../src/simulation/reference/solver';
import { compileScene, type LayerSpec } from '../../src/simulation/scene';
import { GRIP, gripPatch } from '../../src/simulation/grip';

const OUT = resolve(import.meta.dirname, '../../artifacts/studies');
const LEAN: MGConfig = { maxCoarsestNodes: 400, coarseSweeps: 2, lineColors: 2 };
const LEAN3: MGConfig = { maxCoarsestNodes: 400, coarseSweeps: 2, lineColors: 3 };
const LAG3: MGConfig = { ...LEAN3, lagCoarse: true };
const LAG3LS: MGConfig = { ...LAG3, lineSearch: true };

const CONFIGS: Record<string, { h: number; newton: number; pcg: number; mg: MGConfig; tol?: number; histTol?: number }> = {
  'al-p2-t20': { h: 1 / 120, newton: 8, pcg: 2, mg: LEAN, tol: 2e-5 },
  'al-p2-t50': { h: 1 / 120, newton: 8, pcg: 2, mg: LEAN, tol: 5e-5 },
  'al-p2-cap4': { h: 1 / 120, newton: 4, pcg: 2, mg: LEAN, tol: 2e-6 },
  'al-p2-cap3': { h: 1 / 120, newton: 3, pcg: 2, mg: LEAN, tol: 2e-6 },
  'al-p3-cap4': { h: 1 / 120, newton: 4, pcg: 3, mg: LEAN, tol: 2e-6 },
  'l3-p2-t10': { h: 1 / 120, newton: 6, pcg: 2, mg: LEAN3, tol: 1e-5 },
  'l3-p2-t10-c4': { h: 1 / 120, newton: 4, pcg: 2, mg: LEAN3, tol: 1e-5 },
  'l3-p1-c4': { h: 1 / 120, newton: 4, pcg: 1, mg: LEAN3, tol: 1e-5 },
  'l3-p1-c6': { h: 1 / 120, newton: 6, pcg: 1, mg: LEAN3, tol: 1e-5 },
  'g-p1-c4': { h: 1 / 120, newton: 4, pcg: 1, mg: LAG3, tol: 1e-5 },
  'g-p1-c5': { h: 1 / 120, newton: 5, pcg: 1, mg: LAG3, tol: 1e-5 },
  'g-p2-c4': { h: 1 / 120, newton: 4, pcg: 2, mg: LAG3, tol: 1e-5 },
  'g-fixed2x2': { h: 1 / 120, newton: 2, pcg: 2, mg: LEAN3 },
  'gd-p1-c4': { h: 1 / 120, newton: 4, pcg: 1, mg: LAG3, tol: 1e-5, histTol: 2e-6 },
  'gd-p1-c3': { h: 1 / 120, newton: 3, pcg: 1, mg: LAG3, tol: 1e-5, histTol: 2e-6 },
  'gd-p1-c2': { h: 1 / 120, newton: 2, pcg: 1, mg: LAG3, histTol: 2e-6 },
  'gd5-p1-c3': { h: 1 / 120, newton: 3, pcg: 1, mg: LAG3, tol: 1e-5, histTol: 5e-6 },
  'ls-p1-c2': { h: 1 / 120, newton: 2, pcg: 1, mg: LAG3LS },
  'ls-p1-c2-hist': { h: 1 / 120, newton: 2, pcg: 1, mg: LAG3LS, histTol: 2e-6 },
  'ls-p1-c3': { h: 1 / 120, newton: 3, pcg: 1, mg: LAG3LS },
  'h120-adapt': { h: 1 / 120, newton: 8, pcg: 3, mg: MG_DEFAULTS, tol: 2e-6 },
  'h120-adapt-lean': { h: 1 / 120, newton: 8, pcg: 3, mg: LEAN, tol: 2e-6 },
  'h120-adapt-lean-p2': { h: 1 / 120, newton: 8, pcg: 2, mg: LEAN, tol: 2e-6 },
  'h120-adapt5': { h: 1 / 120, newton: 8, pcg: 3, mg: MG_DEFAULTS, tol: 5e-6 },
  truth: { h: 1 / 480, newton: 3, pcg: 8, mg: MG_DEFAULTS },
  h60: { h: 1 / 60, newton: 2, pcg: 3, mg: MG_DEFAULTS },
  h120: { h: 1 / 120, newton: 2, pcg: 3, mg: MG_DEFAULTS },
  'h60-lean': { h: 1 / 60, newton: 2, pcg: 3, mg: LEAN },
  'h60-lean-p2': { h: 1 / 60, newton: 2, pcg: 2, mg: LEAN },
  'h120-lean': { h: 1 / 120, newton: 2, pcg: 3, mg: LEAN },
  'h120-lean-p2': { h: 1 / 120, newton: 2, pcg: 2, mg: LEAN },
  'h60-n3': { h: 1 / 60, newton: 3, pcg: 2, mg: LEAN },
  'h60-conv': { h: 1 / 60, newton: 6, pcg: 10, mg: MG_DEFAULTS },
  'h120-conv': { h: 1 / 120, newton: 6, pcg: 10, mg: MG_DEFAULTS },
  'h240-conv': { h: 1 / 240, newton: 4, pcg: 10, mg: MG_DEFAULTS },
  'h240': { h: 1 / 240, newton: 2, pcg: 3, mg: MG_DEFAULTS },
};

const T_PULL = 0.2;
const T_HOLD = 0.1;
const T_END = 0.7;
const SAMPLE = 1 / 60;

function spec(): LayerSpec {
  const bu = 3;
  const bv = 2;
  const nu = 46;
  const nv = 46;
  return {
    material: DENIM,
    bundle: [bu, bv],
    sizeU: (nu - 1) * bu * DENIM.pitchU,
    sizeV: (nv - 1) * bv * DENIM.pitchV,
    z: 0,
    angle: -0.4,
    compressU: 0.02,
    compressV: 0.01,
    shear: 0.02,
    collarRows: 2,
    wear: { center: [0, 0], axis: [1, -1], radii: [0.012, 0.009], amount: 0.9, cellSize: 0.009, featureScale: 0.01, defectThreshold: 0.85, defectDensity: 0.4, sparseFrom: 2 },
    wearStrengthLoss: [0.55, 0.93],
    seed: 11,
    perturbation: 0.5e-3,
  };
}

/** Handle goal displacement (m); the handle approaches it at a bounded speed like the app's. */
const GOAL: [number, number, number] = [0.006, 0.003, 0.009];
const HANDLE_SPEED = 0.15;

export default async function main(args: string[]): Promise<void> {
  const name = args[0] ?? 'h120';
  const cfg = CONFIGS[name];
  if (!cfg) throw new Error(`unknown config ${name}`);
  const scene = compileScene([spec()]);
  const L = scene.layers[0];
  const picked = Math.floor(L.ny / 2) * L.nx + Math.floor(L.nx / 2);
  const S = new ReferenceSolver(scene, { h: cfg.h, gravity: [0, 0, -9.81], newtonIterations: cfg.newton, pcgIterations: cfg.pcg, mg: cfg.mg });
  // Common settled start: damage-free drape relaxation with kinetic damping (cached).
  const settledFile = resolve(OUT, 'settled.json');
  if (!existsSync(settledFile)) {
    const R = new ReferenceSolver(scene, { h: 1 / 240, gravity: [0, 0, -9.81], newtonIterations: 3, pcgIterations: 8 });
    R.historyEnabled = false;
    for (let k = 0; k < 240; k++) {
      R.substep();
      for (let q = 0; q < R.v.length; q++) R.v[q] *= 0.9;
    }
    let vmax = 0;
    for (let q = 0; q < R.v.length; q++) vmax = Math.max(vmax, Math.abs(R.v[q]));
    console.log('settled; residual max |v| component', vmax.toExponential(2));
    mkdirSync(OUT, { recursive: true });
    writeFileSync(settledFile, JSON.stringify(Array.from(R.x)));
  }
  S.x.set(JSON.parse(readFileSync(settledFile, 'utf8')) as number[]);
  S.historyTolerance = cfg.histTol ?? Infinity;
  const patch = gripPatch(scene, picked);
  if (!patch) throw new Error('no grip patch');
  const p0: [number, number, number] = [S.x[3 * picked], S.x[3 * picked + 1], S.x[3 * picked + 2]];
  const offset = new Float64Array(3 * patch.nodes.length);
  patch.nodes.forEach((n, k) => {
    for (let q = 0; q < 3; q++) offset[3 * k + q] = S.x[3 * n + q] - p0[q];
  });
  const handle: [number, number, number] = [...p0];
  S.setGrip({ nodes: Int32Array.from(patch.nodes), w: Float64Array.from(patch.weights), offset, goal: handle, k: patch.stiffness });
  let released = false;
  const iters: number[] = [];
  let increases = 0;
  let worst = 0;
  const samples: number[][] = [];
  const ruptures: number[] = [];
  const steps = Math.round(T_END / cfg.h);
  const per = Math.round(SAMPLE / cfg.h);
  const t0 = performance.now();
  for (let k = 0; k <= steps; k++) {
    const t = k * cfg.h;
    if (k % per === 0) {
      samples.push(Array.from(S.x, (v) => Math.fround(v)));
      ruptures.push(S.brokenCount);
    }
    if (k === steps) break;
    const tn = t + cfg.h;
    if (!released && tn > T_PULL + T_HOLD) {
      released = true;
      S.setGrip(null);
    }
    if (!released) {
      const g = [p0[0] + GOAL[0], p0[1] + GOAL[1], p0[2] + GOAL[2]];
      const dd = [g[0] - handle[0], g[1] - handle[1], g[2] - handle[2]];
      const len = Math.hypot(dd[0], dd[1], dd[2]);
      const step = HANDLE_SPEED * cfg.h;
      for (let q = 0; q < 3; q++) handle[q] = len <= step ? g[q] : handle[q] + (dd[q] * step) / len;
      // bounded force: the handle may lead the gripped material by at most GRIP.maxLead
      const c = S.gripCentroid()!;
      const lead = [handle[0] - c[0], handle[1] - c[1], handle[2] - c[2]];
      const ll = Math.hypot(lead[0], lead[1], lead[2]);
      if (ll > GRIP.maxLead) for (let q = 0; q < 3; q++) handle[q] = c[q] + (lead[q] * GRIP.maxLead) / ll;
    }
    if (cfg.tol === undefined) S.substep();
    else {
      S.predict();
      let it = 0;
      while (it < cfg.newton) {
        const e0 = S.potential();
        const m = newtonStepPCG(S, cfg.pcg, cfg.mg);
        const e1 = S.potential();
        if (e1 > e0 + 1e-12 * Math.abs(e0)) {
          increases++;
          worst = Math.max(worst, (e1 - e0) / Math.abs(e0));
        }
        it++;
        if (it >= 2 && m < cfg.tol) break;
      }
      iters[it] = (iters[it] ?? 0) + 1;
      S.finalize();
    }
  }
  const wall = (performance.now() - t0) / 1000;
  if (iters.length) console.log('  newton iteration histogram', JSON.stringify(iters), 'energy increases', increases, 'worst rel', worst.toExponential(2));
  mkdirSync(OUT, { recursive: true });
  const file = resolve(OUT, `timestep-${name}.json`);
  writeFileSync(file, JSON.stringify({ name, cfg, wall, ruptures, pulledOut: S.pulledOutCount, samples }));
  console.log(`${name}: ${steps} substeps in ${wall.toFixed(1)} s, ruptures ${S.brokenCount}, pulled out ${S.pulledOutCount}`);
  const truthFile = resolve(OUT, 'timestep-truth.json');
  if (name !== 'truth' && existsSync(truthFile)) {
    const truth = JSON.parse(readFileSync(truthFile, 'utf8')) as { samples: number[][]; ruptures: number[] };
    const series: string[] = [];
    let maxErr = 0;
    let maxErrRelease = 0;
    const nS = Math.min(samples.length, truth.samples.length);
    for (let q = 0; q < nS; q++) {
      const a = samples[q];
      const b = truth.samples[q];
      let e = 0;
      for (let n = 0; n < a.length / 3; n++) e = Math.max(e, Math.hypot(a[3 * n] - b[3 * n], a[3 * n + 1] - b[3 * n + 1], a[3 * n + 2] - b[3 * n + 2]));
      maxErr = Math.max(maxErr, e);
      if (q % 3 === 0) series.push(`${(q * SAMPLE).toFixed(2)}:${(e * 1e3).toFixed(2)}/${ruptures[q]}-${truth.ruptures[q]}`);
      if (q * SAMPLE > T_PULL + T_HOLD) maxErrRelease = Math.max(maxErrRelease, e);
    }
    console.log(`  max position error ${(maxErr * 1e3).toFixed(2)} mm (after release ${(maxErrRelease * 1e3).toFixed(2)} mm); truth ruptures ${truth.ruptures[truth.ruptures.length - 1]}`);
    console.log('  t:err(mm)/ruptures-truth ' + series.join(' '));
  }
}
