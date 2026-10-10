/**
 * Yarn damage field: where the cloth fails, at the resolution of its yarns.
 *
 * The mechanical lattice bundles bu warps x bv picks per span (about 3.4 mm cells). Deciding failure
 * per span would make every opening a union of those cells and every crack a lattice line. Instead
 * failure is tracked per yarn at every crossing: crossing (k, m) of warp k and pick m holds the
 * damage of the warp near it and of the pick near it (u8, 255 = separated), and each yarn segment
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
import { yarnDamage } from './axial';
import type { LayerInfo } from './scene';

export const DAMAGE_CUT = 255;
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

/** Quantised damage code of d in [0, 1]: 255 only for complete separation. */
export function damageCode(d: number): number {
  return d >= 1 ? DAMAGE_CUT : Math.min(DAMAGE_CUT - 1, Math.floor(d * DAMAGE_CUT));
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
 * Advance the damage of every yarn segment whose span is active and whose yarn has not yet separated
 * within it (a separated yarn carries no load there). Reference implementation of fieldUpdate in
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
): void {
  layers.forEach((L, l) => {
    const fl = fls[l];
    for (let m = 0; m < fl.M; m++) {
      for (let k = 0; k < fl.K; k++) {
        const c = fl.offset + m * fl.K + k;
        for (let f = 0; f < 2; f++) {
          if (damage[2 * c + f] === DAMAGE_CUT) continue;
          const own = ownSpan(L, fl, k, m, f);
          if (!own) continue;
          const s = 2 * (L.nodeOffset + own[1] * L.nx + own[0]) + f;
          if (!isActive(s)) continue;
          const q = f === 1 ? k - own[0] * fl.bu : m - own[1] * fl.bv;
          if ((spans.cutMask[s] >>> q) & 1) continue;
          const eps = interpolatedStrain(L, fl, f, k, m, strain);
          if (!Number.isFinite(eps)) continue;
          const y = L.material.yarns[f];
          const B = bundleOf(fl, f);
          const code = damageCode(yarnDamage(y, B, areaOf(area[2 * c + f]), strength[2 * c + f], spans.softening[s] / spans.restLength[s], eps));
          if (code > damage[2 * c + f]) damage[2 * c + f] = code;
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
              if (d === DAMAGE_CUT && first < 0) first = p;
              compliance += 1 / areaOf(area[2 * c + f]);
            }
            sum += (1 - dmax / DAMAGE_CUT) * (P / compliance);
            if (dmax === DAMAGE_CUT) {
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
