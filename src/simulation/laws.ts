/**
 * Bending and in-plane shear laws of the woven lattice. Mirrored exactly in WGSL
 * (src/simulation/gpu/wgsl/common.wgsl: bendPsi, bendPsiDelta, shearLaw).
 */

/**
 * Fold-stiffening bending of a yarn triple (a, b, c): energy 0.5 kb Psi(q) with
 * q = |x_a - 2 x_b + x_c|^2 and a = 4 Lm^2. For equal spans q / a = sin^2(theta / 2), and
 * Psi = q / (1 - q / a) = a tan^2(theta / 2), i.e. the discrete-elastic-rod curvature (2 tan(theta/2))^2.
 * Gentle bends match the quadratic energy; a fold back onto itself costs without bound, so a crease
 * cannot collapse into a single lattice node. Beyond s = BEND_S0, Psi continues as its second-order
 * Taylor expansion (stretched yarns can exceed s = 1). Psi is convex and increasing in q, so the
 * energy is convex in the positions.
 */
export const BEND_S0 = 0.85;

/** [Psi, Psi', Psi''] at q (a = 4 Lm^2). */
export function bendPsi(q: number, a: number): [number, number, number] {
  const s = q / a;
  if (s < BEND_S0) {
    const d = 1 - s;
    return [q / d, 1 / (d * d), 2 / (a * d * d * d)];
  }
  const d0 = 1 - BEND_S0;
  const q0 = BEND_S0 * a;
  const p1 = 1 / (d0 * d0);
  const p2 = 2 / (a * d0 * d0 * d0);
  const t = q - q0;
  return [q0 / d0 + p1 * t + 0.5 * p2 * t * t, p1 + p2 * t, p2];
}

/**
 * Discrete bending of a yarn triple (a, b, c) whose spans have rest lengths La (a-b) and Lc (b-c).
 * Crossing slip moves material between neighbouring spans, so they are generally unequal. The
 * curvature vector r = wa x_a - 2 x_b + wc x_c with wa = 2 Lc / (La + Lc), wc = 2 La / (La + Lc)
 * vanishes for every straight chain whose spans are strained alike (positions in proportion to the
 * rest lengths), and |r| = Lh |t_c - t_a| = 2 Lh sin(theta / 2) with the harmonic mean
 * Lh = 2 La Lc / (La + Lc). The rod energy B kappa^2 l / 2 over the Voronoi length l = (La + Lc) / 2,
 * with kappa = theta / l, is then 0.5 kb |r|^2 with kb = B / (Lh^2 l); the fold scale is a = 4 Lh^2
 * (bendPsi). For La = Lc these reduce to the second difference [1, -2, 1] and kb = B / L^3. The
 * weights depend on rest lengths only (frozen during a solve), so r stays linear in the positions
 * and the energy convex.
 */
export interface BendGeometry {
  /** Weights of x_a and x_c (x_b has -2). */
  wa: number;
  wc: number;
  /** Stiffness per unit bending rigidity, 1 / (Lh^2 l), 1/m^3. */
  kPerB: number;
  /** Fold scale a = 4 Lh^2. */
  fold: number;
}

export function bendGeometry(La: number, Lc: number): BendGeometry {
  const s = La + Lc;
  const lh = (2 * La * Lc) / s;
  return { wa: (2 * Lc) / s, wc: (2 * La) / s, kPerB: 1 / (lh * lh * 0.5 * s), fold: 4 * lh * lh };
}

/**
 * Fraction of a bundle's bending rigidity carried across the middle node of a triple. Only yarns
 * continuous through it bend: a yarn separated in either span (bit q of a span's cut mask) contributes
 * nothing, even if both spans keep other yarns. With intact fractions ia, ic of the spans and B yarns,
 * the continuous yarns' share is their count over B, times the weaker span's mean intactness per
 * surviving yarn (the damage of individual yarns is not resolved along the bundle).
 */
export function bendScale(ia: number, ic: number, maskA: number, maskC: number, B: number): number {
  const full = B >= 32 ? 0xffffffff : (1 << B) - 1;
  const cont = popcount(~(maskA | maskC) & full);
  if (cont === 0) return 0;
  const ua = popcount(~maskA & full);
  const uc = popcount(~maskC & full);
  return (cont / B) * Math.min(ua > 0 ? (ia * B) / ua : 0, uc > 0 ? (ic * B) / uc : 0);
}

function popcount(v: number): number {
  let x = v >>> 0;
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24) >>> 0;
}

/**
 * Psi(q1) - Psi(q0) without cancellation (dq = q1 - q0 computed stably): on both branches the change
 * is a product with dq, so it vanishes exactly for dq = 0 and its rounding scales with |dq|, not with
 * Psi. A difference of Psi values lost ~1e-7 Psi on sharply folded triples, which the line search read
 * as a conclusive increase (docs/decisions/2026-10-10-interaction-round.md).
 */
export function bendPsiDelta(q0: number, q1: number, dq: number, a: number): number {
  const s0 = q0 / a;
  const s1 = q1 / a;
  if (s0 < BEND_S0 && s1 < BEND_S0) return dq / ((1 - s0) * (1 - s1));
  const qb = BEND_S0 * a;
  const d0 = 1 - BEND_S0;
  const p1 = 1 / (d0 * d0);
  const p2 = 2 / (a * d0 * d0 * d0);
  if (s0 >= BEND_S0 && s1 >= BEND_S0) return dq * (p1 + 0.5 * p2 * (q0 - qb + (q1 - qb)));
  // across the joint: the lower branch up to qb, the quadratic continuation beyond it
  const lo = Math.min(q0, q1);
  const t = Math.max(q0, q1) - qb;
  const up = (qb - lo) / ((1 - lo / a) * d0) + t * (p1 + 0.5 * p2 * t);
  return q1 >= q0 ? up : -up;
}

/**
 * Shear locking. Each cell carries a diagonal pair of springs (stiffness ks, rest length Ld). Woven
 * cloth shears freely until its yarns jam (sound denim: about 17 degrees, a compressive diagonal strain
 * SHEAR_JAM), and it cannot shrink its area by squeezing yarns together; beyond that it buckles out of
 * plane instead. Both show up as a compressed diagonal, so past the jam strain the diagonal stiffness
 * ramps (over SHEAR_RAMP of strain) up to SHEAR_LOCK times ks.
 */
export const SHEAR_JAM = 0.16;
export const SHEAR_RAMP = 0.08;
export const SHEAR_LOCK = 40;

/**
 * Yarns jam when shear closes the spacing between them, p cos(gamma), down to their width d: the lock
 * angle is acos(d / p). Abrasion thins the yarns (d ~ sqrt of the remaining cross-section), so worn
 * weave turns much further before it locks, which is how a worn region distorts and lets a hole open
 * around it. SHEAR_COS0 is d / p of sound cloth (the lock angle of SHEAR_JAM); for near-square cells
 * the compressed diagonal's strain at shear angle gamma is sqrt(1 - sin gamma) - 1.
 */
export const SHEAR_COS0 = Math.sqrt(1 - Math.pow(1 - (1 - SHEAR_JAM) * (1 - SHEAR_JAM), 2));

/** Jam strain (compressive diagonal strain) of a cell whose yarns keep the cross-section `area`. */
export function shearJam(area: number): number {
  const c = SHEAR_COS0 * Math.sqrt(Math.min(1, Math.max(0.02, area)));
  return 1 - Math.sqrt(1 - Math.sqrt(1 - c * c));
}

/**
 * Cross-section governing a cell's jam, from the intact fractions of its spans (bottom and top u
 * spans, left and right v spans; a separated span counts 0): the denser family jams first.
 */
export function cellArea(cu: number, tu: number, cv: number, rv: number): number {
  return Math.max(0.5 * (cu + tu), 0.5 * (cv + rv));
}

/** [force (tension positive), tangent stiffness, energy] of one diagonal at length l. */
export function shearLaw(ks: number, Ld: number, l: number, jam = SHEAR_JAM): [number, number, number] {
  const d = l - Ld;
  let f = ks * d;
  let k = ks;
  let e = 0.5 * ks * d * d;
  const c = -d / Ld - jam;
  if (c > 0) {
    const extra = ks * (SHEAR_LOCK - 1);
    if (c < SHEAR_RAMP) {
      f -= (extra * Ld * c * c) / (2 * SHEAR_RAMP);
      k += (extra * c) / SHEAR_RAMP;
      e += (extra * Ld * Ld * c * c * c) / (6 * SHEAR_RAMP);
    } else {
      f -= extra * Ld * (c - 0.5 * SHEAR_RAMP);
      k += extra;
      e += extra * Ld * Ld * ((SHEAR_RAMP * SHEAR_RAMP) / 6 + 0.5 * (c - SHEAR_RAMP) * (c - SHEAR_RAMP) + 0.5 * SHEAR_RAMP * (c - SHEAR_RAMP));
    }
  }
  return [f, k, e];
}
