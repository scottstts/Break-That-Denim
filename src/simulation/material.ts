/**
 * Versioned material assets in SI units (m, kg, s, N, J).
 *
 * Values are provisional G0/G1 calibration points, not measured textile constants. They are chosen
 * to be physically plausible for a ~12 oz warp-faced 3/1 denim and a brushed woven backing, and to
 * match the projected weave scale measured on native reference frames (see docs/decisions).
 */

export const MATERIAL_VERSION = 1;

/** Family index: 0 = yarns running along material u (weft / picks), 1 = along v (warp / ends). */
export const FAMILY_U = 0;
export const FAMILY_V = 1;

export interface YarnFamily {
  /** Linear density, kg/m. */
  linearDensity: number;
  /** Axial tangent stiffness EA while decrimping, N. */
  eaLow: number;
  /** Axial tangent stiffness after decrimping, N. */
  eaHigh: number;
  /** Strain over which tangent stiffness ramps from eaLow to eaHigh. */
  crimpStrain: number;
  /**
   * Compressive stiffness, N. In-plane the fabric cannot shorten much by squeezing yarns (it buckles
   * out of plane instead); half the decrimping stiffness keeps it from gathering like rubber.
   */
  eaCompress: number;
  /** Viscous axial coefficient, N·s (tension = c * strain rate). */
  axialViscosity: number;
  /** Bending rigidity per yarn, N·m² (denim: Peirce bending length ~3 cm, i.e. ~1e-4 N m per unit width). */
  bendRigidity: number;
  /** Fraction of bending stiffness acting viscously over one step (dimensionless rate factor, s). */
  bendViscosity: number;
  /** Intact peak tension, N. */
  strength: number;
  /** Physical post-peak opening to complete separation (softening length), m. 2*Gamma/T_peak. */
  softeningOpening: number;
  /** Visual in-plane yarn width, m. */
  width: number;
  /** Visual yarn height, m. */
  height: number;
  /** Crimp amplitude: centerline offset from the crossing midplane, m. */
  crimpAmplitude: number;
  /** Parallel yarns carried by one lattice span of this family (1 for a yarn-level lattice). */
  yarnsPerSpan?: number;
  /**
   * Independent failure units ("strands") of a span: its yarns are split into this many equal groups
   * (at most MAX_STRANDS), each with its own strength and failure site (axial.ts bundleResponse).
   */
  strandsPerSpan?: number;
}

export interface WeaveMaterial {
  id: number;
  name: string;
  /** Spacing between adjacent v-yarns measured along u (= rest length of u-segments), m. */
  pitchU: number;
  /** Spacing between adjacent u-yarns measured along v (= rest length of v-segments), m. */
  pitchV: number;
  /** [family U (weft), family V (warp)]. */
  yarns: [YarnFamily, YarnFamily];
  /** Twill repeat and number of crossings per repeat where the v-yarn (warp) floats on the face. */
  twillRepeat: number;
  twillWarpOver: number;
  /** In-plane shear stiffness of a cell's diagonal pair, N/m. */
  shearStiffness: number;
  /** Yarn/yarn friction coefficient at crossings. */
  crossingFriction: number;
  /** Crossing normal force with no yarn tension (weave compaction), N. */
  crossingNormal0: number;
  /** Crossing normal force gained per newton of crossing-yarn tension (crimp wrap factor). */
  crossingWrap: number;
  /** Apparent cloth thickness for contact (midsurface separation of two plies), m. */
  thickness: number;
  /** Cloth/cloth kinetic friction coefficient. */
  clothFriction: number;
  /**
   * Aerodynamic drag on motion normal to the cloth, N s / m^3 (force per area per normal speed).
   * Calibrated so a released, sagging coupon settles within about a second.
   */
  airDrag: number;
}

/** Warp-faced 3/1 twill denim. Warp: ring-dyed indigo (optics in the renderer); weft: undyed. */
export const DENIM: WeaveMaterial = {
  id: 0,
  name: 'denim-3x1',
  pitchU: 0.38e-3, // ~67 ends/inch
  pitchV: 0.58e-3, // ~44 picks/inch
  yarns: [
    {
      // weft (u-family), ~Ne 10 (60 tex)
      linearDensity: 60e-6,
      eaLow: 6,
      eaHigh: 140,
      crimpStrain: 0.03,
      eaCompress: 3,
      axialViscosity: 2e-4,
      bendRigidity: 4e-8, // ~7e-5 N m per unit width (1724 picks/m)
      bendViscosity: 2e-3,
      strength: 8.5,
      softeningOpening: 0.12e-3,
      // in-fabric (flattened) cross-section; the weft lies low under the warp floats
      width: 0.4e-3,
      height: 0.16e-3,
      crimpAmplitude: 0.0,
    },
    {
      // warp (v-family), ~Ne 7 (84 tex)
      linearDensity: 84e-6,
      eaLow: 10,
      eaHigh: 180,
      crimpStrain: 0.025,
      eaCompress: 5,
      axialViscosity: 2e-4,
      bendRigidity: 4.5e-8, // ~1.2e-4 N m per unit width (2632 ends/m)
      bendViscosity: 2e-3,
      strength: 11,
      softeningOpening: 0.12e-3,
      // flattened in the fabric: warps overlap their neighbours (cover factor above one)
      width: 0.7e-3,
      height: 0.26e-3,
      crimpAmplitude: 0.13e-3,
    },
  ],
  twillRepeat: 4,
  twillWarpOver: 3,
  shearStiffness: 60,
  crossingFriction: 0.35,
  crossingNormal0: 0.25,
  crossingWrap: 0.3,
  thickness: 0.7e-3,
  clothFriction: 0.45,
  airDrag: 1.5,
};

/** Brushed woven backing: denser, thicker, softer, with heavy fibrillation (optics in the renderer). */
export const BACKING: WeaveMaterial = {
  id: 1,
  name: 'brushed-backing',
  pitchU: 0.5e-3,
  pitchV: 0.5e-3,
  yarns: [
    {
      linearDensity: 55e-6,
      eaLow: 5,
      eaHigh: 90,
      crimpStrain: 0.04,
      eaCompress: 0.3,
      axialViscosity: 2e-4,
      bendRigidity: 6e-9,
      bendViscosity: 2e-3,
      strength: 6,
      softeningOpening: 0.25e-3,
      width: 0.52e-3,
      height: 0.34e-3,
      crimpAmplitude: 0.08e-3,
    },
    {
      linearDensity: 55e-6,
      eaLow: 5,
      eaHigh: 90,
      crimpStrain: 0.04,
      eaCompress: 0.3,
      axialViscosity: 2e-4,
      bendRigidity: 6e-9,
      bendViscosity: 2e-3,
      strength: 6,
      softeningOpening: 0.25e-3,
      width: 0.52e-3,
      height: 0.34e-3,
      crimpAmplitude: 0.08e-3,
    },
  ],
  twillRepeat: 3,
  twillWarpOver: 2,
  shearStiffness: 45,
  crossingFriction: 0.45,
  crossingNormal0: 0.2,
  crossingWrap: 0.3,
  thickness: 1.0e-3,
  clothFriction: 0.55,
  airDrag: 1.5,
};

export const MATERIALS: readonly WeaveMaterial[] = [DENIM, BACKING];

/**
 * Mechanical material of a lattice whose spans are bundles of parallel yarns: `bu` warp yarns per
 * lattice column (along u) and `bv` picks per lattice row (along v), i.e. bu x bv real crossings
 * per node. Per-span quantities sum over the bundled yarns; per-yarn intensive quantities (strain
 * law shape, softening opening, crimp, friction coefficients) are unchanged. Optical/visual fields
 * (width, height, crimpAmplitude, twill) keep describing the individual yarns.
 */
/** Most failure strands per span (4-bit masks and packed u8 strand records). */
export const MAX_STRANDS = 4;

/** Strands for a bundle of b yarns: the largest divisor of b not above MAX_STRANDS. */
export function strandsFor(b: number): number {
  for (let s = Math.min(MAX_STRANDS, b); s > 1; s--) if (b % s === 0) return s;
  return 1;
}

export function bundleWeave(m: WeaveMaterial, bu: number, bv: number = bu): WeaveMaterial {
  for (const b of [bu, bv]) if (!Number.isInteger(b) || b < 1) throw new Error(`bundle factors must be positive integers, got ${bu}, ${bv}`);
  if (bu === 1 && bv === 1) return m;
  // u-family spans (picks) carry bv yarns; v-family spans (warps) carry bu yarns
  const yarn = (y: YarnFamily, b: number): YarnFamily => ({
    ...y,
    linearDensity: b * y.linearDensity,
    eaLow: b * y.eaLow,
    eaHigh: b * y.eaHigh,
    eaCompress: b * y.eaCompress,
    axialViscosity: b * y.axialViscosity,
    bendRigidity: b * y.bendRigidity,
    strength: b * y.strength,
    yarnsPerSpan: b,
    strandsPerSpan: strandsFor(b),
  });
  // diagonal spring giving the same in-plane shear modulus scales with (a^2 + c^2) / (a c)
  const shape = (a: number, c: number) => (a * a + c * c) / (a * c);
  return {
    ...m,
    name: `${m.name}/b${bu}x${bv}`,
    pitchU: bu * m.pitchU,
    pitchV: bv * m.pitchV,
    yarns: [yarn(m.yarns[0], bv), yarn(m.yarns[1], bu)],
    shearStiffness: (m.shearStiffness * shape(bu * m.pitchU, bv * m.pitchV)) / shape(m.pitchU, m.pitchV),
    // a sliding bundle passes bu x bv real crossings; the wrap force from the crossing bundle's
    // tension is scaled by the sliding family's yarn count (Yarn.yarnsPerSpan) where it is applied
    crossingNormal0: bu * bv * m.crossingNormal0,
  };
}

/** Mass lumped at one crossing node: one u-span of weft plus one v-span of warp. */
export function crossingMass(m: WeaveMaterial): number {
  return m.yarns[0].linearDensity * m.pitchU + m.yarns[1].linearDensity * m.pitchV;
}

/** Whether the v-yarn (warp) is on the face at crossing (i, j). Step-1 right-hand twill. */
export function warpOnFace(m: WeaveMaterial, i: number, j: number): boolean {
  const r = m.twillRepeat;
  return (((i - j) % r) + r) % r < m.twillWarpOver;
}
