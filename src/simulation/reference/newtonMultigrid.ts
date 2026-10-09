import { bendPsi, shearLaw } from '../laws';
import { FAMILY_U } from '../material';
import type { LayerInfo } from '../scene';
import { bracket, coarsenLayers, shouldCoarsen, type LatticeLayer } from '../lattice';
import type { ReferenceSolver } from './solver';

/**
 * Projected-Newton step solved with Galerkin geometric multigrid (f64 reference).
 *
 * Operators are stored as radius-2 lattice stencils of symmetric 3x3 blocks (25 offsets per node).
 * The fine operator is the projected (PSD) Hessian of the incremental potential; coarse operators are
 * P^T A P with bilinear prolongation over every second yarn of each family, which keeps the stencil
 * radius at 2. Smoothing is zebra-colored exact block-tridiagonal yarn-line solves with same-pass
 * couplings majorized onto the diagonal (Jacobi within a color, mirroring the GPU). PCG with one
 * symmetric V-cycle as preconditioner removes the slow out-of-plane modes that bilinear transfers
 * leave for the 4th-order bending operator.
 */

/** Multigrid schedule (mirrored by the GPU solver). */
export interface MGConfig {
  /** Coarsen while a level has more nodes than this (the coarsest level is the first at or below it). */
  maxCoarsestNodes: number;
  /** Line-smoothing sweeps on the coarsest level. */
  coarseSweeps: number;
  /** Line colors per family: 3 (no same-color coupling) or 2 (+-2 cross-line couplings majorized). */
  lineColors: 2 | 3;
  /**
   * Reuse the coarse Galerkin levels built at a substep's first Newton iteration for its later
   * iterations (only the fine operator is reassembled). The preconditioner stays SPD.
   */
  lagCoarse?: boolean;
  /** Choose the step length among {1, 1/2, 1/4} by the incremental potential (descent safeguard). */
  lineSearch?: boolean;
}

export const MG_DEFAULTS: MGConfig = { maxCoarsestNodes: 48, coarseSweeps: 6, lineColors: 3 };

export const STENCIL = 25;
export const CENTER = 12;
export function offsetIndex(di: number, dj: number): number {
  return (dj + 2) * 5 + (di + 2);
}

export class StencilOperator {
  readonly N: number;
  readonly layers: LatticeLayer[];
  /** N * 25 * 9 row-major blocks. */
  readonly A: Float64Array;
  /** Node is eliminated (pinned / unsupported): identity row, zero correction. */
  readonly fixed: Uint8Array;
  /** Per node: (i, j, layer). */
  readonly ij: Int32Array;

  constructor(layers: LatticeLayer[]) {
    this.layers = layers;
    let N = 0;
    for (const L of layers) N += L.nx * L.ny;
    this.N = N;
    this.A = new Float64Array(N * STENCIL * 9);
    this.fixed = new Uint8Array(N);
    this.ij = new Int32Array(3 * N);
    for (let l = 0; l < layers.length; l++) {
      const L = layers[l];
      for (let j = 0; j < L.ny; j++) {
        for (let i = 0; i < L.nx; i++) {
          const n = L.offset + j * L.nx + i;
          this.ij[3 * n] = i;
          this.ij[3 * n + 1] = j;
          this.ij[3 * n + 2] = l;
        }
      }
    }
  }

  /** Neighbor of n at (di, dj) or -1. */
  nbr(n: number, di: number, dj: number): number {
    const i = this.ij[3 * n] + di;
    const j = this.ij[3 * n + 1] + dj;
    const L = this.layers[this.ij[3 * n + 2]];
    if (i < 0 || j < 0 || i >= L.nx || j >= L.ny) return -1;
    return L.offset + j * L.nx + i;
  }

  addBlock(row: number, o: number, s: number, B: ArrayLike<number>): void {
    const base = (row * STENCIL + o) * 9;
    for (let q = 0; q < 9; q++) this.A[base + q] += s * B[q];
  }

  /** y = A x (fixed rows give y = x). */
  apply(x: Float64Array, y: Float64Array): void {
    for (let n = 0; n < this.N; n++) {
      if (this.fixed[n]) {
        y[3 * n] = x[3 * n];
        y[3 * n + 1] = x[3 * n + 1];
        y[3 * n + 2] = x[3 * n + 2];
        continue;
      }
      let a0 = 0;
      let a1 = 0;
      let a2 = 0;
      for (let dj = -2; dj <= 2; dj++) {
        for (let di = -2; di <= 2; di++) {
          const m = this.nbr(n, di, dj);
          if (m < 0 || this.fixed[m]) continue;
          const b = (n * STENCIL + offsetIndex(di, dj)) * 9;
          const A = this.A;
          const x0 = x[3 * m];
          const x1 = x[3 * m + 1];
          const x2 = x[3 * m + 2];
          a0 += A[b] * x0 + A[b + 1] * x1 + A[b + 2] * x2;
          a1 += A[b + 3] * x0 + A[b + 4] * x1 + A[b + 5] * x2;
          a2 += A[b + 6] * x0 + A[b + 7] * x1 + A[b + 8] * x2;
        }
      }
      y[3 * n] = a0;
      y[3 * n + 1] = a1;
      y[3 * n + 2] = a2;
    }
  }
}

function axialBlock(kPar: number, kPerp: number, e: ArrayLike<number>, out: Float64Array): void {
  const d = kPar - kPerp;
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r * 3 + c] = d * e[r] * e[c] + (r === c ? kPerp : 0);
}

/** Scatter a two-node term Hessian [[H, -H], [-H, H]] into free rows. */
function scatterPair(A: StencilOperator, a: number, b: number, H: Float64Array): void {
  const ia = A.ij[3 * a];
  const ja = A.ij[3 * a + 1];
  const ib = A.ij[3 * b];
  const jb = A.ij[3 * b + 1];
  if (!A.fixed[a]) {
    A.addBlock(a, CENTER, 1, H);
    if (!A.fixed[b]) A.addBlock(a, offsetIndex(ib - ia, jb - ja), -1, H);
  }
  if (!A.fixed[b]) {
    A.addBlock(b, CENTER, 1, H);
    if (!A.fixed[a]) A.addBlock(b, offsetIndex(ia - ib, ja - jb), -1, H);
  }
}

const I3 = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);

/**
 * Transverse (geometric) stiffness magnitude kept in the solver Hessian for a two-node term with
 * tension f: clamp (max(f, 0)) or absolute-value projection (|f|). Development switch for studies.
 */
export const HESSIAN = { absProjection: false };
function geomStiff(f: number): number {
  return HESSIAN.absProjection ? Math.abs(f) : Math.max(f, 0);
}

/** Assemble the projected Hessian of the reference solver's incremental potential at its current dx. */
export function assembleFineOperator(S: ReferenceSolver): StencilOperator {
  const layers: LatticeLayer[] = S.scene.layers.map((L: LayerInfo) => ({ nx: L.nx, ny: L.ny, offset: L.nodeOffset }));
  const A = new StencilOperator(layers);
  const h = S.params.h;
  for (let n = 0; n < S.N; n++) {
    if (S.scene.pinned[n]) {
      A.fixed[n] = 1;
      A.addBlock(n, CENTER, 1, I3);
    }
  }
  const gTmp = new Float64Array(3);
  for (let n = 0; n < S.N; n++) {
    if (A.fixed[n]) continue;
    A.addBlock(n, CENTER, S.scene.mass[n] / (h * h) + S.gripTerm(n, gTmp), I3);
    const [cd, nn] = S.airDrag(n);
    if (cd > 0) {
      const B = new Float64Array(9);
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) B[3 * r + c] = cd * nn[r] * nn[c];
      A.addBlock(n, CENTER, 1, B);
    }
  }
  const H = new Float64Array(9);
  const e = new Float64Array(3);
  // axial
  for (let s = 0; s < 2 * S.N; s++) {
    if (!S.segActive(s)) continue;
    const a = s >> 1;
    const f = s & 1;
    const b = S.segOther(s);
    let l = 0;
    for (let q = 0; q < 3; q++) {
      e[q] = S.x[3 * b + q] + S.dx[3 * b + q] - S.x[3 * a + q] - S.dx[3 * a + q];
      l += e[q] * e[q];
    }
    l = Math.sqrt(l);
    for (let q = 0; q < 3; q++) e[q] /= l;
    const L = S.restLength[s];
    const yarn = S.layer(a).material.yarns[f];
    const [T, K] = S.response(s, l / L - 1);
    const ln = S.segLength(s);
    const c = yarn.axialViscosity / (L * h);
    const fs = T + c * (l - ln);
    axialBlock(Math.max(K, 0) / L + c, geomStiff(fs) / l, e, H);
    scatterPair(A, a, b, H);
  }
  // bending triples (a, b, c) along each family: block coefficients k * [1,-2,1]^T [1,-2,1]
  for (let b = 0; b < S.N; b++) {
    for (let f = 0; f < 2; f++) {
      const a = S.nbr[4 * b + 2 * f];
      const c = S.nbr[4 * b + 2 * f + 1];
      if (a < 0 || c < 0) continue;
      const s0 = 2 * a + f;
      const s1 = 2 * b + f;
      if (!S.segActive(s0) || !S.segActive(s1)) continue;
      const yarn = S.layer(b).material.yarns[f];
      const Lm = 0.5 * (S.restLength[s0] + S.restLength[s1]);
      const kb = yarn.bendRigidity / (Lm * Lm * Lm);
      // fold-stiffening energy: the exact Hessian kb (Psi' I + 2 Psi'' r r^T) is majorized by the
      // scalar kb (Psi' + 2 Psi'' |r|^2), which keeps the stencil's scalar-identity block form
      let q2 = 0;
      for (let z = 0; z < 3; z++) {
        const rz = S.x[3 * a + z] + S.dx[3 * a + z] - 2 * (S.x[3 * b + z] + S.dx[3 * b + z]) + S.x[3 * c + z] + S.dx[3 * c + z];
        q2 += rz * rz;
      }
      const ps = bendPsi(q2, 4 * Lm * Lm);
      const k = kb * (ps[1] + 2 * ps[2] * q2) + (kb * yarn.bendViscosity) / h;
      const nodes = [a, b, c];
      const w = [1, -2, 1];
      for (let p = 0; p < 3; p++) {
        const r = nodes[p];
        if (A.fixed[r]) continue;
        for (let q = 0; q < 3; q++) {
          const m = nodes[q];
          if (A.fixed[m]) continue;
          const o = offsetIndex(A.ij[3 * m] - A.ij[3 * r], A.ij[3 * m + 1] - A.ij[3 * r + 1]);
          A.addBlock(r, o, k * w[p] * w[q], I3);
        }
      }
    }
  }
  // shear diagonals
  const pa = new Float64Array(3);
  for (let n = 0; n < S.N; n++) {
    const r = S.nbr[4 * n + 1];
    const t = S.nbr[4 * n + 3];
    if (r < 0 || t < 0) continue;
    if (!(S.segActive(2 * n) && S.segActive(2 * n + 1) && S.segActive(2 * r + 1) && S.segActive(2 * t + FAMILY_U))) continue;
    const rt = S.nbr[4 * r + 3];
    const ks = S.layer(n).material.shearStiffness;
    const Ld = Math.hypot(S.restLength[2 * n], S.restLength[2 * n + 1]);
    for (const [p, q] of [
      [n, rt],
      [r, t],
    ]) {
      let l = 0;
      for (let z = 0; z < 3; z++) {
        pa[z] = S.x[3 * q + z] + S.dx[3 * q + z] - S.x[3 * p + z] - S.dx[3 * p + z];
        l += pa[z] * pa[z];
      }
      l = Math.sqrt(l);
      for (let z = 0; z < 3; z++) pa[z] /= l;
      const [fsh, ksh] = shearLaw(ks, Ld, l);
      axialBlock(ksh, geomStiff(fsh) / l, pa, H);
      scatterPair(A, p, q, H);
    }
  }
  return A;
}

export interface Prolongation {
  /** For each fine node, up to 4 (coarse, weight). Fixed fine nodes have none. */
  idx: Int32Array;
  w: Float64Array;
}

/** Coarse lattice layout and bilinear prolongation for one coarsening of `fine`. */
export function buildProlongation(fine: StencilOperator): { coarse: LatticeLayer[]; P: Prolongation } {
  const coarse = coarsenLayers(fine.layers);
  const P: Prolongation = { idx: new Int32Array(4 * fine.N).fill(-1), w: new Float64Array(4 * fine.N) };
  for (let l = 0; l < fine.layers.length; l++) {
    const L = fine.layers[l];
    const C = coarse[l];
    for (let j = 0; j < L.ny; j++) {
      const bj = bracket(j, L.ny);
      for (let i = 0; i < L.nx; i++) {
        const f = L.offset + j * L.nx + i;
        if (fine.fixed[f]) continue;
        const bi = bracket(i, L.nx);
        let k = 0;
        for (const [J, wj] of bj) {
          for (const [I, wi] of bi) {
            P.idx[4 * f + k] = C.offset + J * C.nx + I;
            P.w[4 * f + k] = wi * wj;
            k++;
          }
        }
      }
    }
  }
  return { coarse, P };
}

/** Galerkin coarse operator P^T A P. Coarse rows without free fine support become identity rows. */
export function galerkin(A: StencilOperator, coarseLayers: LatticeLayer[], P: Prolongation): StencilOperator {
  const C = new StencilOperator(coarseLayers);
  const blk = new Float64Array(9);
  for (let a = 0; a < A.N; a++) {
    if (A.fixed[a]) continue;
    for (let dj = -2; dj <= 2; dj++) {
      for (let di = -2; di <= 2; di++) {
        const b = A.nbr(a, di, dj);
        if (b < 0 || A.fixed[b]) continue;
        const base = (a * STENCIL + offsetIndex(di, dj)) * 9;
        let nz = false;
        for (let q = 0; q < 9; q++) {
          blk[q] = A.A[base + q];
          nz ||= blk[q] !== 0;
        }
        if (!nz) continue;
        for (let ka = 0; ka < 4; ka++) {
          const I = P.idx[4 * a + ka];
          if (I < 0) continue;
          const wa = P.w[4 * a + ka];
          for (let kb = 0; kb < 4; kb++) {
            const J = P.idx[4 * b + kb];
            if (J < 0) continue;
            const w = wa * P.w[4 * b + kb];
            const oi = C.ij[3 * J] - C.ij[3 * I];
            const oj = C.ij[3 * J + 1] - C.ij[3 * I + 1];
            if (Math.abs(oi) > 2 || Math.abs(oj) > 2) throw new Error('Galerkin stencil exceeds radius 2');
            C.addBlock(I, offsetIndex(oi, oj), w, blk);
          }
        }
      }
    }
  }
  for (let I = 0; I < C.N; I++) {
    const base = (I * STENCIL + CENTER) * 9;
    if (C.A[base] + C.A[base + 4] + C.A[base + 8] <= 0) {
      C.fixed[I] = 1;
      C.A.fill(0, I * STENCIL * 9, (I + 1) * STENCIL * 9);
      C.A[base] = C.A[base + 4] = C.A[base + 8] = 1;
    }
  }
  return C;
}

// ---- block pentadiagonal line solve ----

function inv3(a: ArrayLike<number>, out: Float64Array): void {
  const [a00, a01, a02, a10, a11, a12, a20, a21, a22] = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]];
  const c00 = a11 * a22 - a12 * a21;
  const c01 = a12 * a20 - a10 * a22;
  const c02 = a10 * a21 - a11 * a20;
  const det = a00 * c00 + a01 * c01 + a02 * c02;
  const id = 1 / det;
  out[0] = c00 * id;
  out[1] = (a02 * a21 - a01 * a22) * id;
  out[2] = (a01 * a12 - a02 * a11) * id;
  out[3] = c01 * id;
  out[4] = (a00 * a22 - a02 * a20) * id;
  out[5] = (a02 * a10 - a00 * a12) * id;
  out[6] = c02 * id;
  out[7] = (a01 * a20 - a00 * a21) * id;
  out[8] = (a00 * a11 - a01 * a10) * id;
}

/**
 * Inverse of a pivot block shifted by a small multiple of its scale. Mirrors pivotInverse in WGSL
 * (common.wgsl): in f32 a Schur pivot can lose its inertia digits entirely next to stiff yarn terms,
 * so the GPU smoother regularizes every pivot; the reference applies the same policy.
 */
export const PIVOT_SHIFT = 2e-6;
const shifted = new Float64Array(9);
function pivotInv3(a: ArrayLike<number>, out: Float64Array, shift: number): void {
  const e = shift * Math.max(Math.abs(a[0]), Math.abs(a[4]), Math.abs(a[8]));
  for (let q = 0; q < 9; q++) shifted[q] = a[q];
  shifted[0] += e;
  shifted[4] += e;
  shifted[8] += e;
  inv3(shifted, out);
}

function mul3(a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array): void {
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
}

/**
 * Solve the symmetric block-pentadiagonal system with diagonal blocks D_k, first upper blocks
 * E_k = A(k, k+1) and second upper blocks F_k = A(k, k+2) (lower blocks are their transposes) by
 * banded block Gaussian elimination without pivoting (valid for SPD systems). A nonzero `shift`
 * regularizes every pivot relative to its scale (smoother use, see PIVOT_SHIFT); 0 solves exactly.
 */
export function solvePenta(n: number, D: Float64Array, E: Float64Array, F: Float64Array, r: Float64Array, x: Float64Array, shift = 0): void {
  const Dk = Float64Array.from(D.subarray(0, 9 * n));
  const U1 = Float64Array.from(E.subarray(0, 9 * n));
  const L1 = new Float64Array(9 * n); // A(k, k-1)
  const rk = Float64Array.from(r.subarray(0, 3 * n));
  for (let k = 1; k < n; k++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) L1[9 * k + a * 3 + b] = E[9 * (k - 1) + b * 3 + a];
  const inv = new Float64Array(9);
  const Lm = new Float64Array(9);
  const t = new Float64Array(9);
  const Amk = new Float64Array(9);
  for (let k = 0; k < n; k++) {
    pivotInv3(Dk.subarray(9 * k, 9 * k + 9), inv, shift);
    for (let m = k + 1; m <= k + 2 && m < n; m++) {
      if (m === k + 1) Amk.set(L1.subarray(9 * m, 9 * m + 9));
      else for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) Amk[a * 3 + b] = F[9 * k + b * 3 + a];
      mul3(Amk, inv, Lm);
      if (k + 1 < n) {
        mul3(Lm, U1.subarray(9 * k, 9 * k + 9), t);
        const dst = m === k + 1 ? Dk : L1; // A(m, k+1)
        for (let q = 0; q < 9; q++) dst[9 * m + q] -= t[q];
      }
      if (k + 2 < n) {
        mul3(Lm, F.subarray(9 * k, 9 * k + 9), t);
        const dst = m === k + 1 ? U1 : Dk; // A(m, k+2)
        for (let q = 0; q < 9; q++) dst[9 * m + q] -= t[q];
      }
      for (let q = 0; q < 3; q++) rk[3 * m + q] -= Lm[q * 3] * rk[3 * k] + Lm[q * 3 + 1] * rk[3 * k + 1] + Lm[q * 3 + 2] * rk[3 * k + 2];
    }
  }
  for (let k = n - 1; k >= 0; k--) {
    const b0 = [rk[3 * k], rk[3 * k + 1], rk[3 * k + 2]];
    for (let q = 0; q < 3; q++) {
      if (k + 1 < n) b0[q] -= U1[9 * k + q * 3] * x[3 * (k + 1)] + U1[9 * k + q * 3 + 1] * x[3 * (k + 1) + 1] + U1[9 * k + q * 3 + 2] * x[3 * (k + 1) + 2];
      if (k + 2 < n) b0[q] -= F[9 * k + q * 3] * x[3 * (k + 2)] + F[9 * k + q * 3 + 1] * x[3 * (k + 2) + 1] + F[9 * k + q * 3 + 2] * x[3 * (k + 2) + 2];
    }
    pivotInv3(Dk.subarray(9 * k, 9 * k + 9), inv, shift);
    for (let q = 0; q < 3; q++) x[3 * k + q] = inv[q * 3] * b0[0] + inv[q * 3 + 1] * b0[1] + inv[q * 3 + 2] * b0[2];
  }
}

export interface MGLevel {
  A: StencilOperator;
  P?: Prolongation;
}

export function buildHierarchy(fine: StencilOperator, cfg: MGConfig = MG_DEFAULTS): MGLevel[] {
  const levels: MGLevel[] = [{ A: fine }];
  let cur = fine;
  while (cur.N > cfg.maxCoarsestNodes && shouldCoarsen(cur.layers)) {
    const { coarse, P } = buildProlongation(cur);
    const C = galerkin(cur, coarse, P);
    levels[levels.length - 1].P = P;
    levels.push({ A: C });
    cur = C;
  }
  return levels;
}

export type Smoother = (A: StencilOperator, x: Float64Array, b: Float64Array, reverse?: boolean, colors?: 2 | 3) => void;

export function vcycle(levels: MGLevel[], k: number, x: Float64Array, b: Float64Array, pre = 1, post = 1, coarseSweeps = 6, sm: Smoother = smoothLines, colors: 2 | 3 = 3): void {
  const { A, P } = levels[k];
  if (!P) {
    for (let s = 0; s < coarseSweeps; s++) sm(A, x, b, (s & 1) === 1, colors);
    return;
  }
  for (let s = 0; s < pre; s++) sm(A, x, b, false, colors);
  const Ax = new Float64Array(3 * A.N);
  A.apply(x, Ax);
  const C = levels[k + 1].A;
  const rc = new Float64Array(3 * C.N);
  for (let f = 0; f < A.N; f++) {
    if (A.fixed[f]) continue;
    for (let kk = 0; kk < 4; kk++) {
      const c = P.idx[4 * f + kk];
      if (c < 0) continue;
      const w = P.w[4 * f + kk];
      for (let q = 0; q < 3; q++) rc[3 * c + q] += w * (b[3 * f + q] - Ax[3 * f + q]);
    }
  }
  for (let c = 0; c < C.N; c++) if (C.fixed[c]) rc[3 * c] = rc[3 * c + 1] = rc[3 * c + 2] = 0;
  const xc = new Float64Array(3 * C.N);
  vcycle(levels, k + 1, xc, rc, pre, post, coarseSweeps, sm, colors);
  for (let f = 0; f < A.N; f++) {
    if (A.fixed[f]) continue;
    for (let kk = 0; kk < 4; kk++) {
      const c = P.idx[4 * f + kk];
      if (c < 0) continue;
      const w = P.w[4 * f + kk];
      for (let q = 0; q < 3; q++) x[3 * f + q] += w * xc[3 * c + q];
    }
  }
  for (let s = 0; s < post; s++) sm(A, x, b, true, colors);
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let q = 0; q < a.length; q++) s += a[q] * b[q];
  return s;
}

/** Preconditioned CG on A x = b with one symmetric V-cycle as preconditioner. */
export function pcg(
  levels: MGLevel[],
  b: Float64Array,
  x: Float64Array,
  iterations: number,
  onIter?: (x: Float64Array) => void,
  sm: Smoother = smoothLines,
  cfg: MGConfig = MG_DEFAULTS,
): void {
  const A = levels[0].A;
  const n = b.length;
  const r = new Float64Array(n);
  const Ap = new Float64Array(n);
  A.apply(x, Ap);
  for (let q = 0; q < n; q++) r[q] = b[q] - Ap[q];
  for (let m = 0; m < A.N; m++) if (A.fixed[m]) r[3 * m] = r[3 * m + 1] = r[3 * m + 2] = 0;
  const z = new Float64Array(n);
  vcycle(levels, 0, z, r, 1, 1, cfg.coarseSweeps, sm, cfg.lineColors);
  const p = Float64Array.from(z);
  let rz = dot(r, z);
  for (let it = 0; it < iterations; it++) {
    A.apply(p, Ap);
    for (let m = 0; m < A.N; m++) if (A.fixed[m]) Ap[3 * m] = Ap[3 * m + 1] = Ap[3 * m + 2] = 0;
    const pAp = dot(p, Ap);
    if (!(pAp > 0)) break;
    const alpha = rz / pAp;
    for (let q = 0; q < n; q++) {
      x[q] += alpha * p[q];
      r[q] -= alpha * Ap[q];
    }
    onIter?.(x);
    if (it + 1 === iterations) break;
    z.fill(0);
    vcycle(levels, 0, z, r, 1, 1, cfg.coarseSweeps, sm, cfg.lineColors);
    const rz2 = dot(r, z);
    const beta = rz2 / rz;
    rz = rz2;
    for (let q = 0; q < n; q++) p[q] = z[q] + beta * p[q];
  }
}

/** One projected-Newton iteration solved by PCG with a V-cycle preconditioner. Returns max |step|. */
const lagged = new WeakMap<ReferenceSolver, MGLevel[]>();

/** Forget lagged coarse levels (call at the start of each substep). */
export function resetLaggedLevels(S: ReferenceSolver): void {
  lagged.delete(S);
}

export function newtonStepPCG(S: ReferenceSolver, pcgIterations: number, cfg: MGConfig = MG_DEFAULTS): number {
  const A = assembleFineOperator(S);
  const b = new Float64Array(3 * S.N);
  const g = new Float64Array(3);
  for (let n = 0; n < S.N; n++) {
    if (S.scene.pinned[n]) continue;
    S.nodeGradient(n, g);
    b[3 * n] = -g[0];
    b[3 * n + 1] = -g[1];
    b[3 * n + 2] = -g[2];
  }
  let levels: MGLevel[];
  const prev = cfg.lagCoarse ? lagged.get(S) : undefined;
  if (prev) {
    levels = [{ A, P: prev[0].P }, ...prev.slice(1)];
  } else {
    levels = buildHierarchy(A, cfg);
    if (cfg.lagCoarse) lagged.set(S, levels);
  }
  const d = new Float64Array(3 * S.N);
  pcg(levels, b, d, pcgIterations, undefined, smoothLines, cfg);
  let alpha = 1;
  if (cfg.lineSearch) {
    // step length with the lowest potential among {1, 1/2, 1/4}; 1/8 if none decreases it (energy.wgsl)
    const e = S.energyChange(d, [1, 0.5, 0.25]);
    let best = e[0];
    if (e[1] < best) {
      alpha = 0.5;
      best = e[1];
    }
    if (e[2] < best) {
      alpha = 0.25;
      best = e[2];
    }
    if (best > 0) alpha = 0.125;
  }
  let m = 0;
  for (let q = 0; q < d.length; q++) {
    d[q] *= alpha;
    S.dx[q] += d[q];
    m = Math.max(m, Math.abs(d[q]));
  }
  for (let n = 0; n < S.N; n++) S.lastUpdate[n] = Math.hypot(d[3 * n], d[3 * n + 1], d[3 * n + 2]);
  return m;
}

function frob(B: ArrayLike<number>, o: number): number {
  let s = 0;
  for (let q = 0; q < 9; q++) s += B[o + q] * B[o + q];
  return Math.sqrt(s);
}

/**
 * Smoother mirrored by the GPU: yarn lines colored modulo 3 per family (lines updated together never
 * share a stencil coupling), each line solved exactly as a block-tridiagonal system. The +-2 coupling
 * along the line is majorized onto the diagonal with ||B||_F I (lagged at the old iterate), which keeps
 * the pass a descent step for the SPD system. Forward/reverse color order gives a symmetric V-cycle.
 */
export function smoothLines(A: StencilOperator, x: Float64Array, b: Float64Array, reverse = false, colors: 2 | 3 = 3): void {
  const maxLen = Math.max(...A.layers.map((L) => Math.max(L.nx, L.ny)));
  const D = new Float64Array(9 * maxLen);
  const E = new Float64Array(9 * maxLen);
  const F = new Float64Array(9 * maxLen);
  const r = new Float64Array(3 * maxLen);
  const sol = new Float64Array(3 * maxLen);
  const order: Array<[number, number]> = [];
  for (const family of [1, 0]) for (let c = 0; c < colors; c++) order.push([family, c]);
  if (reverse) order.reverse();
  for (const [family, color] of order) {
    const xOld = Float64Array.from(x);
    for (const L of A.layers) {
      const lines = family === FAMILY_U ? L.ny : L.nx;
      const len = family === FAMILY_U ? L.nx : L.ny;
      for (let line = color; line < lines; line += colors) {
        const node = (k: number) => (family === FAMILY_U ? L.offset + line * L.nx + k : L.offset + k * L.nx + line);
        D.fill(0, 0, 9 * len);
        E.fill(0, 0, 9 * len);
        for (let k = 0; k < len; k++) {
          const n = node(k);
          r[3 * k] = b[3 * n];
          r[3 * k + 1] = b[3 * n + 1];
          r[3 * k + 2] = b[3 * n + 2];
          if (A.fixed[n]) {
            D[9 * k] = D[9 * k + 4] = D[9 * k + 8] = 1;
            continue;
          }
          let maj = 0;
          for (let dj = -2; dj <= 2; dj++) {
            for (let di = -2; di <= 2; di++) {
              const m = A.nbr(n, di, dj);
              if (m < 0 || (A.fixed[m] && !(di === 0 && dj === 0))) continue;
              const base = (n * STENCIL + offsetIndex(di, dj)) * 9;
              const along = family === FAMILY_U ? di : dj;
              const across = family === FAMILY_U ? dj : di;
              if (across === 0 && along === 0) {
                for (let q = 0; q < 9; q++) D[9 * k + q] += A.A[base + q];
                continue;
              }
              if (across === 0 && along === 1) {
                for (let q = 0; q < 9; q++) E[9 * k + q] = A.A[base + q];
                continue;
              }
              if (across === 0 && along === -1) continue;
              // +-2 along the line, and +-2 across it when those lines share a color, are majorized
              if (across === 0 || Math.abs(across) === colors) maj += frob(A.A, base);
              const x0 = xOld[3 * m];
              const x1 = xOld[3 * m + 1];
              const x2 = xOld[3 * m + 2];
              r[3 * k] -= A.A[base] * x0 + A.A[base + 1] * x1 + A.A[base + 2] * x2;
              r[3 * k + 1] -= A.A[base + 3] * x0 + A.A[base + 4] * x1 + A.A[base + 5] * x2;
              r[3 * k + 2] -= A.A[base + 6] * x0 + A.A[base + 7] * x1 + A.A[base + 8] * x2;
            }
          }
          D[9 * k] += maj;
          D[9 * k + 4] += maj;
          D[9 * k + 8] += maj;
          r[3 * k] += maj * xOld[3 * n];
          r[3 * k + 1] += maj * xOld[3 * n + 1];
          r[3 * k + 2] += maj * xOld[3 * n + 2];
        }
        F.fill(0, 0, 9 * len);
        solvePenta(len, D, E, F, r, sol, PIVOT_SHIFT);
        for (let k = 0; k < len; k++) {
          const n = node(k);
          if (A.fixed[n]) continue;
          x[3 * n] = sol[3 * k];
          x[3 * n + 1] = sol[3 * k + 1];
          x[3 * n + 2] = sol[3 * k + 2];
        }
      }
    }
  }
}
