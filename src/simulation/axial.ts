/**
 * Axial yarn response: a C1 "J-curve" (decrimping then yarn extension) with irreversible
 * triangular cohesive softening defined on physical extension.
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

/** Remaining tension envelope at extension `ext` (only meaningful at/after peak). */
export function envelopeTension(peakTension: number, peakExt: number, softening: number, ext: number): number {
  if (ext <= peakExt) return peakTension;
  return Math.max(0, peakTension * (1 - (ext - peakExt) / softening));
}

/**
 * Damaged tension and a positive semidefinite solver tangent for the frozen committed history
 * `maxExt` and current extension. Returns [T, K].
 *
 * On the softening branch the true tangent is negative (concave energy). Zeroing it would leave a
 * Newton step unbounded along the yarn, so K is the tangent of the damaged secant curve through the
 * current point, (T_env / T_el) * dT_el/deps, which majorizes the concave branch and keeps steps
 * conservative. The gradient (T) is exact.
 */
export function damagedResponse(
  p: AxialParams,
  restLength: number,
  peakTension: number,
  softening: number,
  maxExt: number,
  eps: number,
): [number, number] {
  const peakExt = strainAtTension(p, peakTension) * restLength;
  const ext = eps * restLength;
  const tEl = axialTension(p, eps);
  const kEl = axialStiffness(p, eps);
  let s = 1;
  if (maxExt > peakExt) {
    const tAtMax = axialTension(p, maxExt / restLength);
    s = tAtMax > 0 ? envelopeTension(peakTension, peakExt, softening, maxExt) / tAtMax : 0;
  }
  const tSecant = s * tEl;
  if (ext > peakExt) {
    const tEnv = envelopeTension(peakTension, peakExt, softening, ext);
    if (tEnv < tSecant) return [tEnv, tEl > 0 ? (tEnv / tEl) * kEl : 0];
  }
  return [tSecant, s * kEl];
}

/** Separation is complete once the committed history passes the end of the softening branch. */
export function isSeparated(p: AxialParams, restLength: number, peakTension: number, softening: number, maxExt: number): boolean {
  return maxExt >= strainAtTension(p, peakTension) * restLength + softening;
}

/**
 * Strand-level damage inside a bundled span. The span's yarns form S equal strands (material.ts
 * strandsPerSpan); each strand stores one strength multiplier (packed u8, see unpackYarnPeak):
 * strand k peaks at the bundle tension peak * m_k, i.e. at its own extension, and softens on its
 * own; all share the span's extension history. The span force is the mean of the per-strand damaged
 * responses evaluated at bundle scale, so a bundle fails strand by strand and keeps load in the
 * survivors. A zero byte marks a strand that is absent (worn through before interaction). Mirrored
 * in WGSL (common.wgsl: bundleResponse).
 */
/** Strength multiplier of packed byte u (0 = absent yarn). */
export function unpackYarnPeak(u: number): number {
  return u === 0 ? 0 : 0.4 + (1.2 * (u - 1)) / 254;
}

/** Byte for a strength multiplier m in [0.4, 1.6] (never 0: absence is explicit). */
export function packYarnPeak(m: number): number {
  return 1 + Math.round((Math.min(1.6, Math.max(0.4, m)) - 0.4) * (254 / 1.2));
}

/** Damaged (T, K) of a bundle of `b` strands with packed multipliers `peaks` (byte k = strand k). */
export function bundleResponse(p: AxialParams, b: number, restLength: number, peakTension: number, softening: number, maxExt: number, eps: number, peaks: number): [number, number] {
  let T = 0;
  let K = 0;
  for (let k = 0; k < b; k++) {
    const m = unpackYarnPeak((peaks >>> (8 * k)) & 0xff);
    if (m === 0) continue;
    const [t, kk] = damagedResponse(p, restLength, peakTension * m, softening, maxExt, eps);
    T += t;
    K += kk;
  }
  return [T / b, K / b];
}

/** Bit k set when strand k of the bundle is absent or fully separated. */
export function bundleBrokenMask(p: AxialParams, b: number, restLength: number, peakTension: number, softening: number, maxExt: number, peaks: number): number {
  let mask = 0;
  for (let k = 0; k < b; k++) {
    const m = unpackYarnPeak((peaks >>> (8 * k)) & 0xff);
    if (m === 0 || isSeparated(p, restLength, peakTension * m, softening, maxExt)) mask |= 1 << k;
  }
  return mask;
}
