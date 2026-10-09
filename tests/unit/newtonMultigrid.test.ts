import { describe, expect, it } from 'vitest';
import { DENIM } from '../../src/simulation/material';
import {
  assembleFineOperator,
  buildHierarchy,
  buildProlongation,
  galerkin,
  offsetIndex,
  smoothLines,
  solvePenta,
  STENCIL,
  vcycle,
  pcg,
  type StencilOperator,
} from '../../src/simulation/reference/newtonMultigrid';
import { ReferenceSolver } from '../../src/simulation/reference/solver';
import { compileScene, type LayerSpec } from '../../src/simulation/scene';

function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}

function coupon(nu: number, nv: number): LayerSpec {
  return {
    material: DENIM,
    sizeU: (nu - 1) * DENIM.pitchU,
    sizeV: (nv - 1) * DENIM.pitchV,
    z: 0,
    angle: 0.3,
    compressU: 0.02,
    compressV: 0.01,
    shear: 0.01,
    collarRows: 2,
    wear: { center: [0, 0], axis: [1, 0], radii: [1, 1], amount: 0, cellSize: 0, defectThreshold: 2, defectDensity: 0 },
    wearStrengthLoss: [0, 0],
    seed: 3,
    perturbation: 0.3e-3,
  };
}

function toDense(A: StencilOperator): Float64Array {
  const n3 = 3 * A.N;
  const M = new Float64Array(n3 * n3);
  for (let n = 0; n < A.N; n++) {
    for (let dj = -2; dj <= 2; dj++) {
      for (let di = -2; di <= 2; di++) {
        const m = A.nbr(n, di, dj);
        if (m < 0) continue;
        const base = (n * STENCIL + offsetIndex(di, dj)) * 9;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) M[(3 * n + a) * n3 + 3 * m + b] = A.A[base + a * 3 + b];
      }
    }
  }
  return M;
}

describe('Newton multigrid building blocks', () => {
  it('block pentadiagonal solve matches the system', () => {
    const r = rng(9);
    const n = 9;
    const D = new Float64Array(9 * n);
    const E = new Float64Array(9 * n);
    const F = new Float64Array(9 * n);
    // diagonally dominant symmetric system
    for (let k = 0; k < n; k++) {
      for (let q = 0; q < 9; q++) {
        E[9 * k + q] = r();
        F[9 * k + q] = 0.5 * r();
      }
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) D[9 * k + a * 3 + b] = a === b ? 12 : 0.3 * (a + b - 2);
    }
    const rhs = new Float64Array(3 * n).map(() => r());
    const x = new Float64Array(3 * n);
    solvePenta(n, D, E, F, rhs, x);
    // residual
    for (let k = 0; k < n; k++) {
      for (let a = 0; a < 3; a++) {
        let s = 0;
        for (let b = 0; b < 3; b++) {
          s += D[9 * k + a * 3 + b] * x[3 * k + b];
          if (k + 1 < n) s += E[9 * k + a * 3 + b] * x[3 * (k + 1) + b];
          if (k + 2 < n) s += F[9 * k + a * 3 + b] * x[3 * (k + 2) + b];
          if (k - 1 >= 0) s += E[9 * (k - 1) + b * 3 + a] * x[3 * (k - 1) + b];
          if (k - 2 >= 0) s += F[9 * (k - 2) + b * 3 + a] * x[3 * (k - 2) + b];
        }
        expect(Math.abs(s - rhs[3 * k + a])).toBeLessThan(1e-12);
      }
    }
  });

  it('assembled operator is symmetric and Galerkin coarse operator equals P^T A P', () => {
    const scene = compileScene([coupon(11, 9)]);
    const S = new ReferenceSolver(scene);
    S.predict();
    const A = assembleFineOperator(S);
    const M = toDense(A);
    const n3 = 3 * A.N;
    let asym = 0;
    let mag = 0;
    for (let i = 0; i < n3; i++) {
      for (let j = 0; j < n3; j++) {
        asym = Math.max(asym, Math.abs(M[i * n3 + j] - M[j * n3 + i]));
        mag = Math.max(mag, Math.abs(M[i * n3 + j]));
      }
    }
    expect(asym / mag).toBeLessThan(1e-12);
    const { coarse, P } = buildProlongation(A);
    const C = galerkin(A, coarse, P);
    const Mc = toDense(C);
    const nc3 = 3 * C.N;
    // dense P (fine-free rows only)
    const Pd = new Float64Array(n3 * nc3);
    for (let f = 0; f < A.N; f++) {
      for (let k = 0; k < 4; k++) {
        const c = P.idx[4 * f + k];
        if (c < 0) continue;
        for (let q = 0; q < 3; q++) Pd[(3 * f + q) * nc3 + 3 * c + q] = P.w[4 * f + k];
      }
    }
    // Mfree: zero fixed rows/cols
    const Mf = Float64Array.from(M);
    for (let n = 0; n < A.N; n++) {
      if (!A.fixed[n]) continue;
      for (let q = 0; q < 3; q++) {
        for (let j = 0; j < n3; j++) {
          Mf[(3 * n + q) * n3 + j] = 0;
          Mf[j * n3 + 3 * n + q] = 0;
        }
      }
    }
    let err = 0;
    let cmag = 0;
    for (let I = 0; I < nc3; I++) {
      for (let J = 0; J < nc3; J++) {
        if (C.fixed[Math.floor(I / 3)] || C.fixed[Math.floor(J / 3)]) continue;
        let s = 0;
        for (let a = 0; a < n3; a++) {
          const pa = Pd[a * nc3 + I];
          if (pa === 0) continue;
          for (let b = 0; b < n3; b++) s += pa * Mf[a * n3 + b] * Pd[b * nc3 + J];
        }
        err = Math.max(err, Math.abs(s - Mc[I * nc3 + J]));
        cmag = Math.max(cmag, Math.abs(s));
      }
    }
    expect(err / cmag).toBeLessThan(1e-10);
  });

  it('smoothing and V-cycles reduce the residual of the Newton system', () => {
    const scene = compileScene([coupon(40, 30)]);
    const S = new ReferenceSolver(scene);
    S.predict();
    const A = assembleFineOperator(S);
    const r = rng(4);
    const b = new Float64Array(3 * A.N).map(() => r());
    for (let n = 0; n < A.N; n++) if (A.fixed[n]) b[3 * n] = b[3 * n + 1] = b[3 * n + 2] = 0;
    const norm = (x: Float64Array) => {
      const Ax = new Float64Array(x.length);
      A.apply(x, Ax);
      let s = 0;
      for (let q = 0; q < x.length; q++) s += (b[q] - Ax[q]) ** 2;
      return Math.sqrt(s);
    };
    const x = new Float64Array(3 * A.N);
    const r0 = norm(x);
    smoothLines(A, x, b);
    expect(norm(x)).toBeLessThan(r0);
    const levels = buildHierarchy(A);
    expect(levels.length).toBeGreaterThan(2);
    const y = new Float64Array(3 * A.N);
    vcycle(levels, 0, y, b);
    expect(norm(y)).toBeLessThan(0.5 * r0);
    // PCG with the symmetric V-cycle: energy-norm error against a manufactured solution
    const xs = new Float64Array(3 * A.N).map(() => 1e-5 * r());
    for (let n = 0; n < A.N; n++) if (A.fixed[n]) xs[3 * n] = xs[3 * n + 1] = xs[3 * n + 2] = 0;
    const bs = new Float64Array(3 * A.N);
    A.apply(xs, bs);
    const enorm = (x: Float64Array) => {
      const e = x.map((v, q) => v - xs[q]);
      const Ae = new Float64Array(e.length);
      A.apply(e, Ae);
      let t = 0;
      for (let q = 0; q < e.length; q++) t += e[q] * Ae[q];
      return Math.sqrt(t);
    };
    const z = new Float64Array(3 * A.N);
    const e0 = enorm(z);
    pcg(levels, bs, z, 3);
    expect(enorm(z) / e0).toBeLessThan(5e-3);
  });
});
