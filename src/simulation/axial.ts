/**
 * Axial yarn response: a C1 "J-curve" (decrimping then yarn extension), and the per-yarn cohesive
 * damage law of the yarn damage field (yarnField.ts).
 *
 * Mirrored exactly by `axialTension` / `axialStiffness` in WGSL (src/simulation/gpu/common.wgsl).
 */

export interface AxialParams {
  eaLow: number;
  eaHigh: number;
  crimpStrain: number;
  eaCompress: number;
}

/** Undamaged tension T(eps), N. */
export function axialTension(p: AxialParams, eps: number): number {
  if (eps <= 0) return p.eaCompress * eps;
  const ec = p.crimpStrain;
  const dk = p.eaHigh - p.eaLow;
  if (eps <= ec) return p.eaLow * eps + (dk * eps * eps) / (2 * ec);
  return p.eaLow * ec + (dk * ec) / 2 + p.eaHigh * (eps - ec);
}

/** Undamaged tangent dT/deps, N. */
export function axialStiffness(p: AxialParams, eps: number): number {
  if (eps <= 0) return p.eaCompress;
  const ec = p.crimpStrain;
  if (eps <= ec) return p.eaLow + ((p.eaHigh - p.eaLow) * eps) / ec;
  return p.eaHigh;
}

/** Undamaged strain energy per unit rest length, J/m: integral of T d(eps). */
export function axialEnergyDensity(p: AxialParams, eps: number): number {
  if (eps <= 0) return 0.5 * p.eaCompress * eps * eps;
  const ec = p.crimpStrain;
  const dk = p.eaHigh - p.eaLow;
  if (eps <= ec) return 0.5 * p.eaLow * eps * eps + (dk * eps * eps * eps) / (6 * ec);
  const tc = p.eaLow * ec + (dk * ec) / 2;
  const ecEnergy = 0.5 * p.eaLow * ec * ec + (dk * ec * ec) / 6;
  const d = eps - ec;
  return ecEnergy + tc * d + 0.5 * p.eaHigh * d * d;
}

/** Strain at which the undamaged curve reaches tension t (t > 0). */
export function strainAtTension(p: AxialParams, t: number): number {
  const ec = p.crimpStrain;
  const dk = p.eaHigh - p.eaLow;
  const tc = p.eaLow * ec + (dk * ec) / 2;
  if (t >= tc) return ec + (t - tc) / p.eaHigh;
  // Solve (dk/(2ec)) e^2 + eaLow e - t = 0 (cancellation-free root).
  const a = dk / (2 * ec);
  return (2 * t) / (p.eaLow + Math.sqrt(p.eaLow * p.eaLow + 4 * a * t));
}

/**
 * Irreversible damage of one yarn of a bundled span at strain `eps`, from a triangular cohesive law on
 * the yarn's own tension: elastic up to its peak tension `peakYarn` (N), then softening linearly to
 * zero over `softStrain`. `p` is the bundle law of `yarns` parallel yarns; an abraded yarn keeps the
 * fraction `area` of its fibres, so its elastic law is area / yarns of the bundle's. Damage is the
 * secant loss d = 1 - T_env / T_el, 0 below the peak, 1 once the yarn has separated; the span then
 * carries (1 - d) of the yarn's elastic tension (yarnField.ts). Mirrored in WGSL (common.wgsl).
 */
export function yarnDamage(p: AxialParams, yarns: number, area: number, peakYarn: number, softStrain: number, eps: number): number {
  const ep = strainAtTension(p, (peakYarn * yarns) / area);
  if (eps <= ep) return 0;
  if (eps >= ep + softStrain) return 1;
  const tEnv = peakYarn * (1 - (eps - ep) / softStrain);
  const tEl = (area * axialTension(p, eps)) / yarns;
  return Math.min(1, Math.max(0, 1 - tEnv / Math.max(tEl, 1e-12)));
}

/**
 * Tension and tangent of a span whose yarns keep the intact fraction `intact` of the bundle's load
 * capacity (abraded cross-section and damage, yarnField.ts): the elastic bundle law scaled by it. Damage is frozen during a substep and updated after it, so the
 * law is elastic with the damaged secant stiffness (unloading never heals). Mirrored in WGSL
 * (common.wgsl spanResponse).
 */
export function spanResponse(p: AxialParams, intact: number, eps: number): [number, number] {
  return [intact * axialTension(p, eps), intact * axialStiffness(p, eps)];
}

/** Strain energy per unit rest length of a span with intact fraction `intact`, J/m. */
export function spanEnergyDensity(p: AxialParams, intact: number, eps: number): number {
  return intact * axialEnergyDensity(p, eps);
}
