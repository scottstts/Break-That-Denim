import { axialEnergyDensity, damagedResponse, isSeparated } from '../axial';
import { FAMILY_U } from '../material';
import { SEG_BROKEN, SEG_EXISTS, type CompiledScene, type LayerInfo } from '../scene';
import { newtonStepPCG } from './newtonMultigrid';

/**
 * Double-precision reference mechanics: the incremental potential, its gradient, the projected
 * Newton / PCG / Galerkin-multigrid substep (see newtonMultigrid.ts), and the post-step history
 * updates (cohesive damage, rupture, crossing slip). The GPU kernels mirror this file.
 *
 *   Phi(dx) = sum m/(2h^2) |dx - pred|^2 + E_axial + E_bend + E_shear (+ viscous terms)
 */

export interface SolverParams {
  h: number;
  gravity: [number, number, number];
  newtonIterations: number;
  pcgIterations: number;
}

export const DEFAULT_PARAMS: SolverParams = { h: 1 / 240, gravity: [0, 0, -9.81], newtonIterations: 2, pcgIterations: 4 };

export class ReferenceSolver {
  readonly scene: CompiledScene;
  readonly params: SolverParams;
  readonly N: number;
  readonly x: Float64Array;
  readonly v: Float64Array;
  readonly dx: Float64Array;
  readonly pred: Float64Array;
  /** Per node lattice neighbors [u-, u+, v-, v+] (global index or -1). */
  readonly nbr: Int32Array;
  readonly layerOf: Uint8Array;
  readonly restLength: Float64Array;
  readonly maxExt: Float64Array;
  readonly segFlags: Uint8Array;
  readonly tails: Float64Array;
  /** Spans separated by cohesive failure since construction. */
  brokenCount = 0;
  /** Spans released because a yarn slid out of a crossing. */
  pulledOutCount = 0;

  private readonly e = new Float64Array(3);
  private readonly r = new Float64Array(3);
  private readonly rn = new Float64Array(3);
  private readonly pa = new Float64Array(3);
  private readonly pb = new Float64Array(3);

  constructor(scene: CompiledScene, params: SolverParams = DEFAULT_PARAMS) {
    this.scene = scene;
    this.params = params;
    const N = scene.nodeCount;
    this.N = N;
    this.x = Float64Array.from(scene.position);
    this.v = new Float64Array(3 * N);
    this.dx = new Float64Array(3 * N);
    this.pred = new Float64Array(3 * N);
    this.restLength = Float64Array.from(scene.restLength);
    this.maxExt = Float64Array.from(scene.maxExt);
    this.segFlags = Uint8Array.from(scene.segFlags);
    this.tails = Float64Array.from(scene.tails);
    this.nbr = new Int32Array(4 * N).fill(-1);
    this.layerOf = new Uint8Array(N);
    for (const L of scene.layers) {
      for (let j = 0; j < L.ny; j++) {
        for (let i = 0; i < L.nx; i++) {
          const n = L.nodeOffset + j * L.nx + i;
          this.layerOf[n] = L.index;
          if (i > 0) this.nbr[4 * n + 0] = n - 1;
          if (i < L.nx - 1) this.nbr[4 * n + 1] = n + 1;
          if (j > 0) this.nbr[4 * n + 2] = n - L.nx;
          if (j < L.ny - 1) this.nbr[4 * n + 3] = n + L.nx;
        }
      }
    }
  }

  layer(n: number): LayerInfo {
    return this.scene.layers[this.layerOf[n]];
  }

  /** Segment of family f on side dir (-1 or +1) of node n, or -1. */
  seg(n: number, f: number, dir: -1 | 1): number {
    if (dir > 0) return this.nbr[4 * n + 2 * f + 1] >= 0 ? 2 * n + f : -1;
    const p = this.nbr[4 * n + 2 * f];
    return p >= 0 ? 2 * p + f : -1;
  }

  segActive(s: number): boolean {
    return s >= 0 && (this.segFlags[s] & (SEG_EXISTS | SEG_BROKEN)) === SEG_EXISTS;
  }

  /** Other endpoint of segment s (owned by node s>>1 in family s&1). */
  segOther(s: number): number {
    return this.nbr[4 * (s >> 1) + 2 * (s & 1) + 1];
  }

  /** Whether the cell with lower-left node c has all four spans active. */
  cellActive(c: number): boolean {
    const r = this.nbr[4 * c + 1];
    const t = this.nbr[4 * c + 3];
    if (r < 0 || t < 0) return false;
    return this.segActive(2 * c) && this.segActive(2 * c + 1) && this.segActive(2 * r + 1) && this.segActive(2 * t + FAMILY_U);
  }

  private pos(n: number, out: Float64Array): void {
    out[0] = this.x[3 * n] + this.dx[3 * n];
    out[1] = this.x[3 * n + 1] + this.dx[3 * n + 1];
    out[2] = this.x[3 * n + 2] + this.dx[3 * n + 2];
  }

  /**
   * Axial response of segment s at the current iterate (displacement form e = e_n + (dx_b - dx_a)).
   * Returns [fs, kPar, kPerp, l]: dE/dl including viscosity, projected axial/transverse stiffness,
   * current length. Fills this.e with the unit vector from owner to far endpoint.
   */
  segForce(s: number): [number, number, number, number] {
    const a = s >> 1;
    const f = s & 1;
    const b = this.segOther(s);
    const x = this.x;
    const dx = this.dx;
    const en0 = x[3 * b] - x[3 * a];
    const en1 = x[3 * b + 1] - x[3 * a + 1];
    const en2 = x[3 * b + 2] - x[3 * a + 2];
    const e0 = en0 + (dx[3 * b] - dx[3 * a]);
    const e1 = en1 + (dx[3 * b + 1] - dx[3 * a + 1]);
    const e2 = en2 + (dx[3 * b + 2] - dx[3 * a + 2]);
    const l = Math.hypot(e0, e1, e2);
    const ln = Math.hypot(en0, en1, en2);
    this.e[0] = e0 / l;
    this.e[1] = e1 / l;
    this.e[2] = e2 / l;
    const L = this.restLength[s];
    const yarn = this.layer(a).material.yarns[f];
    const [T, K] = damagedResponse(yarn, L, this.scene.peakTension[s], this.scene.softening[s], this.maxExt[s], l / L - 1);
    const c = yarn.axialViscosity / (L * this.params.h);
    const fs = T + c * (l - ln);
    return [fs, Math.max(K, 0) / L + c, Math.max(fs, 0) / l, l];
  }

  /** Elastic and viscous bending coefficients of the triple centered at node b. */
  bendCoeffs(b: number, f: number, s0: number, s1: number): [number, number] {
    const yarn = this.layer(b).material.yarns[f];
    const Lm = 0.5 * (this.restLength[s0] + this.restLength[s1]);
    const kb = yarn.bendRigidity / (Lm * Lm * Lm);
    return [kb, (kb * yarn.bendViscosity) / this.params.h];
  }

  private triple(a: number, b: number, c: number): void {
    for (let k = 0; k < 3; k++) {
      this.rn[k] = this.x[3 * a + k] - 2 * this.x[3 * b + k] + this.x[3 * c + k];
      this.r[k] = this.rn[k] + this.dx[3 * a + k] - 2 * this.dx[3 * b + k] + this.dx[3 * c + k];
    }
  }

  /** Bending gradient contributions of family f triples containing node n. */
  private addBendingGradient(n: number, f: number, g: Float64Array): void {
    const prev = this.nbr[4 * n + 2 * f];
    const nxt = this.nbr[4 * n + 2 * f + 1];
    const sPrev = prev >= 0 ? 2 * prev + f : -1;
    const sNext = nxt >= 0 ? 2 * n + f : -1;
    const { r, rn } = this;
    if (this.segActive(sPrev) && this.segActive(sNext)) {
      const [ke, kd] = this.bendCoeffs(n, f, sPrev, sNext);
      this.triple(prev, n, nxt);
      for (let k = 0; k < 3; k++) g[k] += -2 * (ke * r[k] + kd * (r[k] - rn[k]));
    }
    if (this.segActive(sNext)) {
      const nxt2 = this.nbr[4 * nxt + 2 * f + 1];
      const sNext2 = nxt2 >= 0 ? 2 * nxt + f : -1;
      if (this.segActive(sNext2)) {
        const [ke, kd] = this.bendCoeffs(nxt, f, sNext, sNext2);
        this.triple(n, nxt, nxt2);
        for (let k = 0; k < 3; k++) g[k] += ke * r[k] + kd * (r[k] - rn[k]);
      }
    }
    if (this.segActive(sPrev)) {
      const prev2 = this.nbr[4 * prev + 2 * f];
      const sPrev2 = prev2 >= 0 ? 2 * prev2 + f : -1;
      if (this.segActive(sPrev2)) {
        const [ke, kd] = this.bendCoeffs(prev, f, sPrev2, sPrev);
        this.triple(prev2, prev, n);
        for (let k = 0; k < 3; k++) g[k] += ke * r[k] + kd * (r[k] - rn[k]);
      }
    }
  }

  /** Shear diagonal partners of n as [cell lower-left, partner] pairs (cells must be active). */
  shearPartners(n: number): Array<[number, number]> {
    const nb = this.nbr;
    const left = nb[4 * n];
    const right = nb[4 * n + 1];
    const down = nb[4 * n + 2];
    const up = nb[4 * n + 3];
    const out: Array<[number, number]> = [];
    if (right >= 0 && up >= 0) out.push([n, nb[4 * up + 1]]);
    if (left >= 0 && down >= 0) out.push([nb[4 * down], nb[4 * down]]);
    if (left >= 0 && up >= 0) out.push([left, nb[4 * up]]);
    if (right >= 0 && down >= 0) out.push([down, nb[4 * down + 1]]);
    return out.filter(([cell, p]) => p >= 0 && this.cellActive(cell));
  }

  shearRest(cell: number): number {
    return Math.hypot(this.restLength[2 * cell], this.restLength[2 * cell + 1]);
  }

  /** Gradient of the incremental potential w.r.t. dx at node n (zero for pinned nodes). */
  nodeGradient(n: number, g: Float64Array): void {
    g.fill(0);
    if (this.scene.pinned[n]) return;
    const h = this.params.h;
    const w = this.scene.mass[n] / (h * h);
    for (let q = 0; q < 3; q++) g[q] += w * (this.dx[3 * n + q] - this.pred[3 * n + q]);
    for (let f = 0; f < 2; f++) {
      const sPrev = this.seg(n, f, -1);
      const sNext = this.seg(n, f, 1);
      if (this.segActive(sPrev)) {
        const [fs] = this.segForce(sPrev);
        for (let q = 0; q < 3; q++) g[q] += fs * this.e[q];
      }
      if (this.segActive(sNext)) {
        const [fs] = this.segForce(sNext);
        for (let q = 0; q < 3; q++) g[q] -= fs * this.e[q];
      }
      this.addBendingGradient(n, f, g);
    }
    const ks = this.layer(n).material.shearStiffness;
    for (const [cell, partner] of this.shearPartners(n)) {
      this.pos(n, this.pa);
      this.pos(partner, this.pb);
      const d0 = this.pb[0] - this.pa[0];
      const d1 = this.pb[1] - this.pa[1];
      const d2 = this.pb[2] - this.pa[2];
      const l = Math.hypot(d0, d1, d2);
      const fs = ks * (l - this.shearRest(cell));
      g[0] -= (fs * d0) / l;
      g[1] -= (fs * d1) / l;
      g[2] -= (fs * d2) / l;
    }
  }

  /** Total incremental potential at the current dx (history-free material assumed), for verification. */
  potential(): number {
    const h = this.params.h;
    let E = 0;
    for (let n = 0; n < this.N; n++) {
      if (this.scene.pinned[n]) continue;
      const w = this.scene.mass[n] / (2 * h * h);
      for (let q = 0; q < 3; q++) {
        const d = this.dx[3 * n + q] - this.pred[3 * n + q];
        E += w * d * d;
      }
    }
    for (let s = 0; s < 2 * this.N; s++) {
      if (!this.segActive(s)) continue;
      const a = s >> 1;
      const f = s & 1;
      const b = this.segOther(s);
      this.pos(a, this.pa);
      this.pos(b, this.pb);
      const l = Math.hypot(this.pb[0] - this.pa[0], this.pb[1] - this.pa[1], this.pb[2] - this.pa[2]);
      const ln = this.segLength(s);
      const L = this.restLength[s];
      const yarn = this.layer(a).material.yarns[f];
      E += L * axialEnergyDensity(yarn, l / L - 1);
      const c = yarn.axialViscosity / (L * h);
      E += 0.5 * c * (l - ln) * (l - ln);
      const nb = this.nbr[4 * b + 2 * f + 1];
      if (nb >= 0 && this.segActive(2 * b + f)) {
        const [ke, kd] = this.bendCoeffs(b, f, s, 2 * b + f);
        this.triple(a, b, nb);
        for (let q = 0; q < 3; q++) {
          E += 0.5 * ke * this.r[q] * this.r[q];
          const d = this.r[q] - this.rn[q];
          E += 0.5 * kd * d * d;
        }
      }
    }
    for (let n = 0; n < this.N; n++) {
      if (!this.cellActive(n)) continue;
      const r = this.nbr[4 * n + 1];
      const t = this.nbr[4 * n + 3];
      const rt = this.nbr[4 * r + 3];
      const ks = this.layer(n).material.shearStiffness;
      const Ld = this.shearRest(n);
      for (const [p, q] of [
        [n, rt],
        [r, t],
      ]) {
        this.pos(p, this.pa);
        this.pos(q, this.pb);
        const l = Math.hypot(this.pb[0] - this.pa[0], this.pb[1] - this.pa[1], this.pb[2] - this.pa[2]);
        E += 0.5 * ks * (l - Ld) * (l - Ld);
      }
    }
    return E;
  }

  /**
   * Inertial predictor pred = h v + h^2 g. The Newton iterate starts from h v: extrapolating gravity
   * into the initial iterate drops the interior against the pinned collar and creates a kinked
   * boundary layer whose geometric stiffness Newton recovers only slowly.
   */
  predict(): void {
    const { h, gravity } = this.params;
    for (let n = 0; n < this.N; n++) {
      const pinned = this.scene.pinned[n];
      for (let q = 0; q < 3; q++) {
        const hv = pinned ? 0 : h * this.v[3 * n + q];
        this.pred[3 * n + q] = pinned ? 0 : hv + h * h * gravity[q];
        this.dx[3 * n + q] = hv;
      }
    }
  }

  substep(): void {
    this.predict();
    for (let it = 0; it < this.params.newtonIterations; it++) newtonStepPCG(this, this.params.pcgIterations);
    this.finalize();
  }

  /** Commit positions/velocities, update cohesive history, rupture, and crossing slip. */
  finalize(): void {
    const h = this.params.h;
    for (let q = 0; q < 3 * this.N; q++) {
      this.v[q] = this.dx[q] / h;
      this.x[q] += this.dx[q];
      this.dx[q] = 0;
    }
    this.updateDamage();
    this.updateSlip();
  }

  segLength(s: number): number {
    const a = s >> 1;
    const b = this.segOther(s);
    return Math.hypot(this.x[3 * b] - this.x[3 * a], this.x[3 * b + 1] - this.x[3 * a + 1], this.x[3 * b + 2] - this.x[3 * a + 2]);
  }

  /** Tension of an active segment at committed positions (no viscosity) and dT/dL at fixed length. */
  segTension(s: number): [number, number] {
    const a = s >> 1;
    const f = s & 1;
    const L = this.restLength[s];
    const l = this.segLength(s);
    const yarn = this.layer(a).material.yarns[f];
    const [T, K] = damagedResponse(yarn, L, this.scene.peakTension[s], this.scene.softening[s], this.maxExt[s], l / L - 1);
    return [T, Math.max(K, 0) * (l / (L * L))];
  }

  updateDamage(): void {
    for (let s = 0; s < 2 * this.N; s++) {
      if (!this.segActive(s)) continue;
      const ext = this.segLength(s) - this.restLength[s];
      if (ext > this.maxExt[s]) this.maxExt[s] = ext;
      const yarn = this.layer(s >> 1).material.yarns[s & 1];
      if (isSeparated(yarn, this.restLength[s], this.scene.peakTension[s], this.scene.softening[s], this.maxExt[s])) {
        this.rupture(s);
        this.brokenCount++;
      }
    }
  }

  /** Separate span s at its failure site; its rest length becomes two tails (ledger-preserving). */
  rupture(s: number): void {
    const a = s >> 1;
    const f = s & 1;
    const b = this.segOther(s);
    const L = this.restLength[s];
    const phi = this.scene.failSite[s];
    this.segFlags[s] |= SEG_BROKEN;
    this.tails[4 * a + 2 * f + 1] += L * phi;
    this.tails[4 * b + 2 * f] += L * (1 - phi);
  }

  /** Friction-limited crossing slip in two parity passes per family. */
  updateSlip(): void {
    for (let f = 0; f < 2; f++) {
      for (let parity = 0; parity < 2; parity++) {
        for (const L of this.scene.layers) {
          for (let j = 0; j < L.ny; j++) {
            for (let i = 0; i < L.nx; i++) {
              const along = f === FAMILY_U ? i : j;
              if ((along & 1) !== parity) continue;
              this.slipNode(L.nodeOffset + j * L.nx + i, f);
            }
          }
        }
      }
    }
  }

  slipNode(n: number, f: number): void {
    if (this.scene.pinned[n]) return;
    const sL = this.seg(n, f, -1);
    const sR = this.seg(n, f, 1);
    if (sL < 0 || sR < 0) return;
    const actL = this.segActive(sL);
    const actR = this.segActive(sR);
    if (!actL && !actR) return;
    const tailL = 4 * n + 2 * f;
    const tailR = 4 * n + 2 * f + 1;
    const [TL, kL] = actL ? this.segTension(sL) : [0, 0];
    const [TR, kR] = actR ? this.segTension(sR) : [0, 0];
    let tc = 0;
    let cnt = 0;
    for (const dir of [-1, 1] as const) {
      const s = this.seg(n, 1 - f, dir);
      if (this.segActive(s)) {
        tc += Math.max(0, this.segTension(s)[0]);
        cnt++;
      }
    }
    if (cnt === 0) return; // no crossing yarn holds this one here
    tc /= cnt;
    const mat = this.layer(n).material;
    const Fc = mat.crossingFriction * (this.scene.crossingNormal0[n] + mat.crossingWrap * tc);
    const dT = Math.max(TR, 0) - Math.max(TL, 0);
    if (Math.abs(dT) <= Fc) return;
    const highR = dT > 0;
    const kHigh = highR ? kR : kL;
    const lowActive = highR ? actL : actR;
    const kLow = lowActive ? (highR ? kL : kR) : 0;
    let dl = (Math.abs(dT) - Fc) / Math.max(kHigh + kLow, 1e-9);
    const pitch = f === FAMILY_U ? mat.pitchU : mat.pitchV;
    const sHigh = highR ? sR : sL;
    const sLow = highR ? sL : sR;
    if (lowActive) {
      dl = Math.min(dl, Math.max(0, this.restLength[sLow] - 0.55 * pitch));
      this.restLength[sLow] -= dl;
      this.restLength[sHigh] += dl;
    } else {
      const tIdx = highR ? tailL : tailR;
      dl = Math.min(dl, this.tails[tIdx]);
      this.tails[tIdx] -= dl;
      this.restLength[sHigh] += dl;
      if (this.tails[tIdx] <= 1e-9) {
        // The yarn leaves this crossing: the high-side span becomes a free tail of its far node.
        this.tails[tIdx] = 0;
        const far = highR ? this.segOther(sHigh) : sHigh >> 1;
        const farTail = highR ? 4 * far + 2 * f : 4 * far + 2 * f + 1;
        this.tails[farTail] += this.restLength[sHigh];
        this.segFlags[sHigh] |= SEG_BROKEN;
        this.pulledOutCount++;
      }
    }
  }

  /** Sum of rest lengths + tails along every yarn (ledger check). */
  yarnMaterialLength(): number {
    let total = 0;
    for (let s = 0; s < 2 * this.N; s++) if (this.segActive(s)) total += this.restLength[s];
    for (let q = 0; q < 4 * this.N; q++) total += this.tails[q];
    return total;
  }
}
