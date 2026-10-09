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
  /** Small compressive stiffness, N (yarns buckle; kept nonzero for conditioning). */
  eaCompress: number;
  /** Viscous axial coefficient, N·s (tension = c * strain rate). */
  axialViscosity: number;
  /** Bending rigidity, N·m². */
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
      eaCompress: 0.4,
      axialViscosity: 2e-4,
      bendRigidity: 5e-9,
      bendViscosity: 2e-3,
      strength: 8.5,
      softeningOpening: 0.12e-3,
      width: 0.5e-3,
      height: 0.26e-3,
      crimpAmplitude: 0.075e-3,
    },
    {
      // warp (v-family), ~Ne 7 (84 tex)
      linearDensity: 84e-6,
      eaLow: 10,
      eaHigh: 180,
      crimpStrain: 0.025,
      eaCompress: 0.5,
      axialViscosity: 2e-4,
      bendRigidity: 9e-9,
      bendViscosity: 2e-3,
      strength: 11,
      softeningOpening: 0.12e-3,
      width: 0.4e-3,
      height: 0.3e-3,
      crimpAmplitude: 0.07e-3,
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
};

export const MATERIALS: readonly WeaveMaterial[] = [DENIM, BACKING];

/** Mass lumped at one crossing node: one u-span of weft plus one v-span of warp. */
export function crossingMass(m: WeaveMaterial): number {
  return m.yarns[0].linearDensity * m.pitchU + m.yarns[1].linearDensity * m.pitchV;
}

/** Whether the v-yarn (warp) is on the face at crossing (i, j). Step-1 right-hand twill. */
export function warpOnFace(m: WeaveMaterial, i: number, j: number): boolean {
  const r = m.twillRepeat;
  return (((i - j) % r) + r) % r < m.twillWarpOver;
}
