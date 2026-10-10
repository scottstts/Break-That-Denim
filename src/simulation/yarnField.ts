/**
 * Yarn damage field: where the cloth fails, at the resolution of its yarns.
 *
 * The mechanical lattice bundles bu warps x bv picks per span (about 3.4 mm cells). Deciding failure
 * per span would make every opening a union of those cells and every crack a lattice line. Instead
 * failure is tracked per yarn at every crossing: crossing (k, m) of warp k and pick m holds the
 * damage of the warp near it and of the pick near it (u8, 255 = cut, 254 = pulled out of the weave,
 * see pullOut), and each yarn segment
 * has its own peak tension from the continuous wear field (rounded worn cells), a correlated scatter
 * and the yarn's own slub variation. After each substep a segment's strain is interpolated from the
 * active parallel spans around it and drives its cohesive law (axial.ts yarnDamage). The failure
 * front is then a level set of smooth fields, free to run in any direction, quantised only at the
 * real yarn pitch.
 *
 * Abrasion removes fibres: a worn yarn segment keeps the fraction `area` of its cross-section (u8 per
 * crossing and family), and so of its stiffness as well as its strength. A worn region is therefore a
 * compliant inclusion that takes up the strain of a pull and gives way as a region, instead of only a
 * weak path a crack runs along.
 *
 * The lattice stays the force carrier and follows the field: a span carries the intact fraction of
 * its yarns' load capacity, the mean over its yarns of (1 - d) at the yarn's most damaged crossing
 * times its cross-section (harmonic mean along the span: segments in series), and separates only when
 * every yarn crossing it has separated somewhere within its length. Its rest length then splits into
 * tails at the mean position of those cuts.
 *
 * Indexing per layer: K = nx bu warps, M = ny bv picks, crossing c = offset + m K + k. Node (i, j) sits
 * at fine coordinates U = i bu + (bu - 1) / 2, V = j bv + (bv - 1) / 2. Warp k belongs to lattice
 * column floor(k / bu), pick m to row floor(m / bv). The v-span (i, j) -> (i, j + 1) covers the
 * crossings m in [j bv + floor(bv / 2), + bv); the u-span (i, j) -> (i + 1, j) covers k in
 * [i bu + floor(bu / 2), + bu). Damage and strength are stored per crossing and family f (0 = pick,
 * running along u; 1 = warp, along v) at index 2 c + f. Mirrored in WGSL (yarnfield.wgsl).
 */
import { axialEnergyDensity, axialTension, strainAtTension, yarnDamage } from './axial';
import type { LayerInfo } from './scene';

export const DAMAGE_CUT = 255;
/**
 * The yarn has pulled out of the weave at this crossing: still there (hanging from where it is held),
 * no longer interlaced with the crossing yarn and carrying no load across it.
 */
export const DAMAGE_PULLED = 254;
/** Damage codes at or above this separate the yarn at the crossing (cut or pulled out). */
export const DAMAGE_SEPARATED = DAMAGE_PULLED;
/** Most yarns per span (bit masks of cut yarns in one u32). */
export const MAX_BUNDLE = 32;

export interface FieldLayer {
  /** First crossing of the layer. */
  offset: number;
  /** Warps (crossings along u) and picks (along v). */
  K: number;
  M: number;
  /** Warps per lattice column, picks per lattice row. */
  bu: number;
  bv: number;
}

export function fieldLayers(layers: readonly LayerInfo[]): { layers: FieldLayer[]; count: number } {
  let offset = 0;
  const out = layers.map((L) => {
    const fl: FieldLayer = { offset, K: L.nx * L.bundle[0], M: L.ny * L.bundle[1], bu: L.bundle[0], bv: L.bundle[1] };
    offset += fl.K * fl.M;
    return fl;
  });
  return { layers: out, count: offset };
}

/** Yarns carried by a span of family f. */
export function bundleOf(fl: FieldLayer, f: number): number {
  return f === 1 ? fl.bu : fl.bv;
}

/** Smallest represented cross-section fraction of a worn yarn segment. */
export const AREA_MIN = 0.02;

/** Cross-section code (u8) of fraction a, and back. */
export function areaCode(a: number): number {
  return Math.max(1, Math.min(255, Math.round(a * 255)));
}
export function areaOf(code: number): number {
  return Math.max(AREA_MIN, code / 255);
}

/** Quantised damage code of d in [0, 1]: 255 only for complete separation, never DAMAGE_PULLED. */
export function damageCode(d: number): number {
  return d >= 1 ? DAMAGE_CUT : Math.min(DAMAGE_PULLED - 1, Math.floor(d * DAMAGE_CUT));
}

/** Longest pull-out (crossings from the break) evaluated along a yarn. */
export const RECOIL_MAX = 12;

/**
 * Pull-out of a free yarn end (shear lag). A cut yarn carries no tension at its end; along it the
 * crossings pass at most the friction f per unit length, so where the cloth around it is strained the
 * yarn's tension rises from zero over T / f, and the yarn slides relative to the weave there. Its end
 * moves back through the crossings by
 *
 *   delta = integral of eps(T(s)) ds = (1 / f) integral_0^T eps dT = (T eps - W(eps)) / f,
 *
 * the yarn's complementary energy per unit length over the friction per unit length (exact for the
 * J-curve). T and W are the yarn's share of the bundle law at the local strain (area / B of it), and
 * f = mu (n0 + wrap T_cross) / pitch is the crossing friction along the yarn: compaction plus the wrap
 * force of the crossing yarn's tension, one crossing per pitch. A yarn end in slack or unloaded cloth
 * does not move; an end in strained, loosened (worn) weave slides out of several crossings.
 */
export function recoilLength(y: YarnMaterialLaw, B: number, area: number, eps: number, mu: number, n0: number, wrap: number, tCross: number, pitch: number): number {
  if (!(eps > 0)) return 0;
  const share = area / B;
  const comp = share * (axialTension(y, eps) * eps - axialEnergyDensity(y, eps));
  return (comp * pitch) / (mu * (n0 + wrap * Math.max(0, tCross)));
}

/**
 * Pull-out of yarn ends (DAMAGE_PULLED): a crossing whose yarn is still interlaced is marked pulled out
 * once the nearest break along the yarn (a cut reached through crossings already pulled out) lies
 * within the end's retraction recoilLength, measured from half a pitch inside the cut. Marks are never
 * undone: the yarn does not re-enter a crossing it left. Evaluated after each substep from the strain
 * snapshot `strain` (per span, NaN where none), so ends move out progressively as the cloth around them
 * is loaded. Mirrored by pullMark / pullApply in yarnfield.wgsl. Returns the number of crossings pulled out.
 */
export function pullOut(layers: readonly LayerInfo[], fls: readonly FieldLayer[], area: Uint8Array, damage: Uint8Array, strain: ArrayLike<number>, crossingNormal0: ArrayLike<number>): number {
  const marked: number[] = [];
  layers.forEach((L, l) => {
    const fl = fls[l];
    const mat = L.material;
    for (let m = 0; m < fl.M; m++) {
      for (let k = 0; k < fl.K; k++) {
        const c = fl.offset + m * fl.K + k;
        for (let f = 0; f < 2; f++) {
          if (damage[2 * c + f] >= DAMAGE_SEPARATED) continue;
          const d = breakDistance(fl, damage, f, k, m);
          if (d < 0) continue;
          const eps = interpolatedStrain(L, fl, f, k, m, strain);
          if (!Number.isFinite(eps)) continue;
          const g = 1 - f;
          const ec = interpolatedStrain(L, fl, g, k, m, strain);
          const tCross = Number.isFinite(ec) ? (Math.max(0, axialTension(mat.yarns[g], ec)) * areaOf(area[2 * c + g])) / bundleOf(fl, g) : 0;
          const n = nodeOfCrossing(L, fl, k, m);
          const pitch = f === 1 ? L.weave.pitchV : L.weave.pitchU;
          const delta = recoilLength(mat.yarns[f], bundleOf(fl, f), areaOf(area[2 * c + f]), eps, mat.crossingFriction, crossingNormal0[n] / (fl.bu * fl.bv), mat.crossingWrap, tCross, pitch);
          if (delta >= (d - 0.5) * pitch) marked.push(2 * c + f);
        }
      }
    }
  });
  for (const q of marked) damage[q] = DAMAGE_PULLED;
  return marked.length;
}

/**
 * Crossings from (k, m) to the nearest cut along its family-f yarn reached through pulled-out
 * crossings only (1 = adjacent), within RECOIL_MAX; -1 if there is none.
 */
export function breakDistance(fl: FieldLayer, damage: Uint8Array, f: number, k: number, m: number): number {
  const pos = f === 0 ? k : m;
  const len = f === 0 ? fl.K : fl.M;
  const stride = f === 0 ? 1 : fl.K;
  const c = fl.offset + m * fl.K + k;
  let best = -1;
  for (const dir of [-1, 1]) {
    for (let d = 1; d <= RECOIL_MAX; d++) {
      const p = pos + dir * d;
      if (p < 0 || p >= len) break;
      const code = damage[2 * (c + dir * d * stride) + f];
      if (code === DAMAGE_CUT) {
        if (best < 0 || d < best) best = d;
        break;
      }
      if (code !== DAMAGE_PULLED) break;
    }
  }
  return best;
}

/** Lattice node nearest to crossing (k, m): the bundle crossing it belongs to. */
export function nodeOfCrossing(L: LayerInfo, fl: FieldLayer, k: number, m: number): number {
  const i = Math.min(L.nx - 1, Math.floor(k / fl.bu));
  const j = Math.min(L.ny - 1, Math.floor(m / fl.bv));
  return L.nodeOffset + j * L.nx + i;
}

/** Lattice span (owner i, j) of the yarn of family f at crossing (k, m), or null outside every span. */
export function ownSpan(L: LayerInfo, fl: FieldLayer, k: number, m: number, f: number): [number, number] | null {
  if (f === 1) {
    const i = Math.floor(k / fl.bu);
    const j = Math.floor((m - Math.floor(fl.bv / 2)) / fl.bv);
    return i < L.nx && j >= 0 && j <= L.ny - 2 ? [i, j] : null;
  }
  const j = Math.floor(m / fl.bv);
  const i = Math.floor((k - Math.floor(fl.bu / 2)) / fl.bu);
  return j < L.ny && i >= 0 && i <= L.nx - 2 ? [i, j] : null;
}

/**
 * Crossing index of yarn q of span (i, j, f) at its p-th crossing along the span (from the owner).
 */
export function spanCrossing(fl: FieldLayer, i: number, j: number, f: number, q: number, p: number): number {
  if (f === 1) return fl.offset + (j * fl.bv + Math.floor(fl.bv / 2) + p) * fl.K + i * fl.bu + q;
  return fl.offset + (j * fl.bv + q) * fl.K + i * fl.bu + Math.floor(fl.bu / 2) + p;
}

/** Crossings along a span of family f (the other family's yarn count per span). */
export function spanLength(fl: FieldLayer, f: number): number {
  return f === 1 ? fl.bv : fl.bu;
}

/**
 * Strain of family f interpolated at crossing (k, m): bilinear over the four nearest parallel spans
 * (at their midpoints), active ones only, renormalised. `strain` holds per span 2 n + f its strain or
 * NaN when it carries none.
 */
export function interpolatedStrain(L: LayerInfo, fl: FieldLayer, f: number, k: number, m: number, strain: ArrayLike<number>): number {
  const s = f === 1 ? (k - (fl.bu - 1) / 2) / fl.bu : (k - (fl.bu - 0.5)) / fl.bu;
  const t = f === 1 ? (m - (fl.bv - 0.5)) / fl.bv : (m - (fl.bv - 1) / 2) / fl.bv;
  const i0 = Math.floor(s);
  const j0 = Math.floor(t);
  const fu = s - i0;
  const fv = t - j0;
  const imax = f === 1 ? L.nx - 1 : L.nx - 2;
  const jmax = f === 1 ? L.ny - 2 : L.ny - 1;
  let num = 0;
  let den = 0;
  for (let dj = 0; dj < 2; dj++) {
    for (let di = 0; di < 2; di++) {
      const i = i0 + di;
      const j = j0 + dj;
      if (i < 0 || j < 0 || i > imax || j > jmax) continue;
      const e = strain[2 * (L.nodeOffset + j * L.nx + i) + f];
      if (!Number.isFinite(e)) continue;
      const w = (di ? fu : 1 - fu) * (dj ? fv : 1 - fv);
      num += w * e;
      den += w;
    }
  }
  return den > 1e-6 ? num / den : NaN;
}

/** Per-span mechanical state the field reads and writes. */
export interface FieldSpans {
  restLength: ArrayLike<number>;
  /** Softening opening per span (m): a yarn separates over this extension past its peak. */
  softening: ArrayLike<number>;
  segFlags: { [k: number]: number; length: number };
  intact: { [k: number]: number; length: number };
  cutMask: Uint32Array;
  failSite: { [k: number]: number; length: number };
  tails: { [k: number]: number; length: number };
}

export interface YarnMaterialLaw {
  eaLow: number;
  eaHigh: number;
  crimpStrain: number;
  eaCompress: number;
}

/**
 * Local load sharing. Where a yarn has separated, the load it carried is passed through the crossing
 * yarns to its immediate neighbours over a short length around the break (the shear-lag length), so a
 * yarn next to a run of separated neighbours is overloaded there, by half the run on each side (the
 * load of a broken cluster goes to the two sound yarns bounding it). The mechanical lattice only sees
 * each bundle's mean; this term puts the overload where it is, so a crack front runs on from yarn to
 * yarn, at the place along them where the last ones broke, independently of bundle boundaries.
 *
 * The crossings pass that load on by friction only (shareCap): no more than mu (n0 + wrap T_cross)
 * each over the transfer length. Beyond it the yarns at the tip slip together and share it (the del
 * zone of a woven tear) and the broken ends slide, so the load spreads further instead of
 * concentrating on one neighbour. Compacted, tensioned weave concentrates and tears in cracks; worn,
 * loosened weave blunts them.
 */
export const SHARE = {
  /** Fraction of the cluster's load taken by each bounding yarn beyond the bundle mean. */
  fraction: 0.6,
  /** Neighbours counted on each side (the cluster size that still concentrates load). */
  maxRun: 6,
  /** Half-width along the yarn (m) within which a neighbour's separation counts as beside it. */
  reach: 1.2e-3,
};

/** Crossings along a yarn of family f within SHARE.reach (warps run over picks, picks over warps). */
export function shareReach(weave: { pitchU: number; pitchV: number }, f: number): number {
  return Math.max(1, Math.round(SHARE.reach / (f === 1 ? weave.pitchV : weave.pitchU)));
}

/**
 * Largest overload (relative to its peak tension `peak`) the crossings within `reach` of a yarn
 * can pass to it by friction: 2 reach + 1 crossings of friction mu (n0 + wrap tCross) each, with n0 the
 * compaction normal force of one crossing and tCross the tension of one crossing yarn.
 */
export function shareCap(reach: number, mu: number, n0: number, wrap: number, tCross: number, peak: number): number {
  return ((2 * reach + 1) * mu * (n0 + wrap * tCross)) / peak;
}

/**
 * Overload factor of the yarn of family f at crossing (k, m) from the separated-state snapshot
 * `sep` (1 per crossing and family, 2 c + f): 1 + fraction (run left + run right) / 2, at most
 * 1 + cap (shareCap).
 */
export function shareFactor(fl: FieldLayer, sep: Uint8Array, f: number, k: number, m: number, reach: number, cap = Infinity): number {
  const along = f === 1 ? m : k;
  const lenAlong = f === 1 ? fl.M : fl.K;
  const across = f === 1 ? k : m;
  const lenAcross = f === 1 ? fl.K : fl.M;
  const at = (y: number, a: number) => fl.offset + (f === 1 ? a * fl.K + y : y * fl.K + a);
  let runs = 0;
  for (const dir of [-1, 1]) {
    for (let d = 1; d <= SHARE.maxRun; d++) {
      const y = across + dir * d;
      if (y < 0 || y >= lenAcross) break;
      let broken = false;
      for (let a = Math.max(0, along - reach); a <= Math.min(lenAlong - 1, along + reach) && !broken; a++) broken = sep[2 * at(y, a) + f] === 1;
      if (!broken) break;
      runs++;
    }
  }
  return 1 + Math.min((SHARE.fraction * runs) / 2, cap);
}

/**
 * Advance the damage of every yarn whose span is active and that has not yet separated within it (a
 * separated yarn carries no load there). A yarn's crossings within one span are in series and carry
 * the same tension, so it softens at one place only: once it has started to soften, there; otherwise
 * at the crossing where its overload (shareFactor) relative to its peak strain is largest. Its strain
 * is interpolated across the bundle at its own position, at the middle of the span, and amplified by
 * that overload. Without the single site, a span whose strain jumps past the softening range in one
 * substep would cut its yarns at every crossing at once (an opening growing in whole lattice spans);
 * without the overload, cuts would scatter over the span independently of where the neighbours
 * broke, and a crack could only run along the bands of lattice spans. The overload is limited by the
 * friction the crossings can pass (shareCap), from the compaction normal force `crossingNormal0` (per
 * node, of the bundled crossing) and the crossing yarns' tension at the middle of the span. Neighbour
 * states are read from the snapshot before the step. Reference implementation of fieldUpdate in
 * yarnfield.wgsl.
 */
export function updateField(
  layers: readonly LayerInfo[],
  fls: readonly FieldLayer[],
  strength: Float32Array,
  area: Uint8Array,
  damage: Uint8Array,
  spans: FieldSpans,
  strain: ArrayLike<number>,
  isActive: (s: number) => boolean,
  crossingNormal0: ArrayLike<number>,
): void {
  const sep = new Uint8Array(damage.length);
  for (let q = 0; q < damage.length; q++) sep[q] = damage[q] >= DAMAGE_SEPARATED ? 1 : 0;
  layers.forEach((L, l) => {
    const fl = fls[l];
    for (let j = 0; j < L.ny; j++) {
      for (let i = 0; i < L.nx; i++) {
        for (let f = 0; f < 2; f++) {
          if ((f === 0 && i >= L.nx - 1) || (f === 1 && j >= L.ny - 1)) continue;
          const s = 2 * (L.nodeOffset + j * L.nx + i) + f;
          if (!isActive(s)) continue;
          const y = L.material.yarns[f];
          const B = bundleOf(fl, f);
          const P = spanLength(fl, f);
          const soft = spans.softening[s] / spans.restLength[s];
          const reach = shareReach(L.weave, f);
          for (let q = 0; q < B; q++) {
            if ((spans.cutMask[s] >>> q) & 1) continue;
            const mid = spanCrossing(fl, i, j, f, q, P >> 1) - fl.offset;
            const eps = interpolatedStrain(L, fl, f, mid % fl.K, Math.floor(mid / fl.K), strain);
            if (!Number.isFinite(eps)) continue;
            // exact early out: no crossing can soften even at the largest overload
            let minPeak = Infinity;
            for (let p = 0; p < P; p++) {
              const c = spanCrossing(fl, i, j, f, q, p);
              minPeak = Math.min(minPeak, strainAtTension(y, (strength[2 * c + f] * B) / areaOf(area[2 * c + f])));
            }
            if (eps * (1 + SHARE.fraction * SHARE.maxRun) <= minPeak) continue;
            // friction the crossings can pass to this yarn: compaction of one crossing plus the wrap
            // force of the crossing yarn's tension at the middle of the span
            const g = 1 - f;
            const kM = mid % fl.K;
            const mM = Math.floor(mid / fl.K);
            const ec = interpolatedStrain(L, fl, g, kM, mM, strain);
            const tCross = Number.isFinite(ec) ? (Math.max(0, axialTension(L.material.yarns[g], ec)) * areaOf(area[2 * (mid + fl.offset) + g])) / bundleOf(fl, g) : 0;
            const transfer = shareCap(reach, L.material.crossingFriction, crossingNormal0[s >> 1] / (fl.bu * fl.bv), L.material.crossingWrap, tCross, 1);
            let best = -1;
            let bestScore = -Infinity;
            let bestOmega = 1;
            let soft0 = -1;
            let softCode = 0;
            for (let p = 0; p < P; p++) {
              const c = spanCrossing(fl, i, j, f, q, p);
              const local = c - fl.offset;
              const omega = shareFactor(fl, sep, f, local % fl.K, Math.floor(local / fl.K), reach, transfer / strength[2 * c + f]);
              const peak = strainAtTension(y, (strength[2 * c + f] * B) / areaOf(area[2 * c + f]));
              const score = omega / peak;
              if (score > bestScore) [best, bestScore, bestOmega] = [c, score, omega];
              if (damage[2 * c + f] > softCode) [soft0, softCode] = [c, damage[2 * c + f]];
            }
            if (soft0 >= 0) {
              best = soft0;
              const local = best - fl.offset;
              bestOmega = shareFactor(fl, sep, f, local % fl.K, Math.floor(local / fl.K), reach, transfer / strength[2 * best + f]);
            }
            const code = damageCode(yarnDamage(y, B, areaOf(area[2 * best + f]), strength[2 * best + f], soft, eps * bestOmega));
            if (code > damage[2 * best + f]) damage[2 * best + f] = code;
          }
        }
      }
    }
  });
}

/**
 * Span state from the field: intact fraction and cut-yarn mask of every active span; a span whose
 * yarns have all separated ruptures (its rest length splits into tails at the mean cut position).
 * Returns the number of ruptures. Reference implementation of couple in yarnfield.wgsl.
 */
export function coupleSpans(layers: readonly LayerInfo[], fls: readonly FieldLayer[], area: Uint8Array, damage: Uint8Array, spans: FieldSpans, isActive: (s: number) => boolean, brokenFlag: number): number {
  let ruptures = 0;
  layers.forEach((L, l) => {
    const fl = fls[l];
    for (let j = 0; j < L.ny; j++) {
      for (let i = 0; i < L.nx; i++) {
        for (let f = 0; f < 2; f++) {
          if ((f === 0 && i >= L.nx - 1) || (f === 1 && j >= L.ny - 1)) continue;
          const n = L.nodeOffset + j * L.nx + i;
          const s = 2 * n + f;
          if (!isActive(s)) continue;
          const B = bundleOf(fl, f);
          const P = spanLength(fl, f);
          let sum = 0;
          let mask = 0;
          let site = 0;
          for (let q = 0; q < B; q++) {
            let dmax = 0;
            let first = -1;
            let compliance = 0;
            for (let p = 0; p < P; p++) {
              const c = spanCrossing(fl, i, j, f, q, p);
              const d = damage[2 * c + f];
              if (d > dmax) dmax = d;
              if (d >= DAMAGE_SEPARATED && first < 0) first = p;
              compliance += 1 / areaOf(area[2 * c + f]);
            }
            sum += (dmax >= DAMAGE_SEPARATED ? 0 : 1 - dmax / DAMAGE_CUT) * (P / compliance);
            if (dmax >= DAMAGE_SEPARATED) {
              mask |= 1 << q;
              site += (first + 0.5) / P;
            }
          }
          spans.intact[s] = sum / B;
          spans.cutMask[s] = mask >>> 0;
          if (mask === (B === 32 ? 0xffffffff : (1 << B) - 1) >>> 0) {
            const phi = site / B;
            const rest = spans.restLength[s];
            const b = f === 0 ? n + 1 : n + L.nx;
            spans.segFlags[s] |= brokenFlag;
            spans.failSite[s] = phi;
            spans.tails[4 * n + 2 * f + 1] += rest * phi;
            spans.tails[4 * b + 2 * f] += rest * (1 - phi);
            ruptures++;
          }
        }
      }
    }
  });
  return ruptures;
}
