import { describe, expect, it } from 'vitest';
import { DENIM } from '../../src/simulation/material';
import { compileScene, SEG_BROKEN, SEG_EXISTS, type LayerSpec } from '../../src/simulation/scene';
import { axialEnergyDensity, axialTension } from '../../src/simulation/axial';
import { coupleSpans, DAMAGE_CUT, DAMAGE_PULLED, interpolatedStrain, ownSpan, pullOut, recoilLength, SHARE, shareCap, shareFactor, shareReach, spanCrossing, spanLength, updateField } from '../../src/simulation/yarnField';

function coupon(extra: Partial<LayerSpec> = {}): LayerSpec {
  return {
    material: DENIM,
    bundle: [9, 6],
    sizeU: 12 * 9 * DENIM.pitchU,
    sizeV: 10 * 6 * DENIM.pitchV,
    z: 0,
    angle: 0,
    compressU: 0,
    compressV: 0,
    shear: 0,
    collarRows: 1,
    wear: { center: [0, 0], axis: [1, 0], radii: [1, 1], amount: 0, cellSize: 0, featureScale: 0.01, defectThreshold: 2, defectDensity: 0, sparseFrom: 2 },
    wearStrengthLoss: [0, 0],
    seed: 3,
    perturbation: 0,
    ...extra,
  };
}

describe('yarn damage field', () => {
  const scene = compileScene([coupon()]);
  const L = scene.layers[0];
  const fl = scene.field[0];

  it('assigns every crossing of a span back to that span', () => {
    for (const f of [0, 1]) {
      for (const [i, j] of [
        [2, 3],
        [5, 1],
      ]) {
        const B = f === 1 ? fl.bu : fl.bv;
        for (let q = 0; q < B; q++) {
          for (let p = 0; p < spanLength(fl, f); p++) {
            const c = spanCrossing(fl, i, j, f, q, p) - fl.offset;
            expect(ownSpan(L, fl, c % fl.K, Math.floor(c / fl.K), f)).toEqual([i, j]);
          }
        }
      }
    }
  });

  it('interpolates a uniform strain exactly and ignores spans that carry none', () => {
    const strain = new Float64Array(2 * scene.nodeCount).fill(0.02);
    expect(interpolatedStrain(L, fl, 1, 40, 30, strain)).toBeCloseTo(0.02, 12);
    strain.fill(NaN);
    strain[2 * (3 * L.nx + 4) + 1] = 0.05;
    // a crossing of that span sees it alone
    const c = spanCrossing(fl, 4, 3, 1, 4, 2) - fl.offset;
    expect(interpolatedStrain(L, fl, 1, c % fl.K, Math.floor(c / fl.K), strain)).toBeCloseTo(0.05, 12);
  });

  it('a span keeps the intact fraction of its yarns and separates once all have, conserving length', () => {
    const s = compileScene([coupon()]);
    const damage = Uint8Array.from(s.fieldDamage);
    const i = 4;
    const j = 3;
    const f = 1;
    const sIdx = 2 * (j * L.nx + i) + f;
    const active = (q: number) => (s.segFlags[q] & (SEG_EXISTS | SEG_BROKEN)) === SEG_EXISTS;
    // cut 3 of the 9 warps (at different crossings), half-damage a fourth
    for (let q = 0; q < 3; q++) damage[2 * spanCrossing(fl, i, j, f, q, q + 1) + f] = DAMAGE_CUT;
    damage[2 * spanCrossing(fl, i, j, f, 3, 0) + f] = 127;
    expect(coupleSpans(s.layers, s.field, s.fieldArea, damage, s, active, SEG_BROKEN)).toBe(0);
    expect(s.intact[sIdx]).toBeCloseTo((5 + (1 - 127 / 255)) / 9, 9);
    expect(s.cutMask[sIdx]).toBe(0b111);
    const total = s.tails.reduce((a, b) => a + b, 0);
    for (let q = 0; q < 9; q++) damage[2 * spanCrossing(fl, i, j, f, q, 2) + f] = DAMAGE_CUT;
    expect(coupleSpans(s.layers, s.field, s.fieldArea, damage, s, active, SEG_BROKEN)).toBe(1);
    expect(s.segFlags[sIdx] & SEG_BROKEN).toBe(SEG_BROKEN);
    expect(s.intact[sIdx]).toBe(0);
    // the span's rest length moved into its two tails
    expect(s.tails.reduce((a, b) => a + b, 0) - total).toBeCloseTo(s.restLength[sIdx], 12);
  });

  it('a yarn overstretched in one step separates at its weakest crossing only', () => {
    const s = compileScene([coupon()]);
    const damage = Uint8Array.from(s.fieldDamage);
    const active = (q: number) => (s.segFlags[q] & (SEG_EXISTS | SEG_BROKEN)) === SEG_EXISTS;
    // far past every yarn's softening range everywhere
    const strain = new Float64Array(2 * s.nodeCount).fill(0.5);
    updateField(s.layers, s.field, s.fieldStrength, s.fieldArea, damage, s, strain, active, s.crossingNormal0);
    const i = 4;
    const j = 3;
    for (const f of [0, 1]) {
      const B = f === 1 ? fl.bu : fl.bv;
      for (let q = 0; q < B; q++) {
        let cuts = 0;
        let weakest = -1;
        let low = Infinity;
        for (let p = 0; p < spanLength(fl, f); p++) {
          const c = spanCrossing(fl, i, j, f, q, p);
          if (damage[2 * c + f] === DAMAGE_CUT) cuts++;
          const peak = s.fieldStrength[2 * c + f] / (s.fieldArea[2 * c + f] / 255);
          if (peak < low) [low, weakest] = [peak, c];
        }
        expect(cuts).toBe(1);
        expect(damage[2 * weakest + f]).toBe(DAMAGE_CUT);
      }
    }
  });

  it('a cut end pulls out by its shear-lag retraction under load, and never in unloaded cloth', () => {
    const s = compileScene([coupon()]);
    const damage = Uint8Array.from(s.fieldDamage);
    const c0 = fl.offset + 30 * fl.K + 40;
    damage[2 * c0] = DAMAGE_CUT;
    const pullAll = (strain: Float64Array) => {
      let total = 0;
      for (let n = pullOut(s.layers, s.field, s.fieldStrength, s.fieldArea, damage, strain, s.crossingNormal0); n > 0; n = pullOut(s.layers, s.field, s.fieldStrength, s.fieldArea, damage, strain, s.crossingNormal0)) total += n;
      return total;
    };
    // slack cloth: the end carries nothing and stays where it broke
    expect(pullAll(new Float64Array(2 * s.nodeCount))).toBe(0);
    // strained cloth (picks at 6 %, warps unloaded): the picks' end slides out over (T eps - W) / f
    const strain = new Float64Array(2 * s.nodeCount);
    for (let q = 0; q < strain.length; q += 2) strain[q] = 0.06;
    pullAll(strain);
    const y = L.material.yarns[0];
    const B = fl.bv;
    const comp = (axialTension(y, 0.06) * 0.06 - axialEnergyDensity(y, 0.06)) / B;
    const p = L.weave.pitchU;
    const delta = (comp * p) / (L.material.crossingFriction * (s.crossingNormal0[0] / (fl.bu * fl.bv)));
    expect(recoilLength(y, B, 1, 1e3, 0.06, L.material.crossingFriction, s.crossingNormal0[0] / (fl.bu * fl.bv), L.material.crossingWrap, 0, p)).toBeCloseTo(delta, 12);
    // the far tension never exceeds the yarn's peak: an opening crack's apparent strain changes nothing
    const peak = s.fieldStrength[2 * c0];
    const capped = recoilLength(y, B, 1, peak, 0.6, L.material.crossingFriction, s.crossingNormal0[0] / (fl.bu * fl.bv), L.material.crossingWrap, 0, p);
    expect(capped).toBeCloseTo(recoilLength(y, B, 1, peak, 0.2, L.material.crossingFriction, s.crossingNormal0[0] / (fl.bu * fl.bv), L.material.crossingWrap, 0, p), 12);
    const run = Math.min(12, Math.floor(delta / p + 0.5));
    expect(run).toBeGreaterThan(0);
    for (const dir of [-1, 1]) {
      for (let k = 1; k <= run; k++) expect(damage[2 * (c0 + dir * k)]).toBe(DAMAGE_PULLED);
      expect(damage[2 * (c0 + dir * (run + 1))]).toBeLessThan(DAMAGE_PULLED);
    }
    // unloading does not draw the yarn back in
    expect(pullAll(new Float64Array(2 * s.nodeCount))).toBe(0);
    expect(damage[2 * (c0 + 1)]).toBe(DAMAGE_PULLED);
  });

  it('overloads the yarns beside a run of separated neighbours, near the breaks only', () => {
    const s = compileScene([coupon()]);
    const sep = new Uint8Array(s.fieldDamage.length);
    const reach = shareReach(s.layers[0].weave, 1);
    // warps 40..42 separated at pick 30
    for (let k = 40; k <= 42; k++) sep[2 * (fl.offset + 30 * fl.K + k) + 1] = 1;
    expect(shareFactor(fl, sep, 1, 43, 30, reach)).toBeCloseTo(1 + (SHARE.fraction * 3) / 2, 12);
    expect(shareFactor(fl, sep, 1, 39, 30 + reach, reach)).toBeCloseTo(1 + (SHARE.fraction * 3) / 2, 12);
    expect(shareFactor(fl, sep, 1, 43, 30 + reach + 1, reach)).toBe(1);
    expect(shareFactor(fl, sep, 1, 45, 30, reach)).toBe(1);
    // the crossings pass no more than their friction: the overload is capped
    expect(shareFactor(fl, sep, 1, 43, 30, reach, 0.1)).toBeCloseTo(1.1, 12);
    expect(shareCap(2, 0.35, 0.25, 0.3, 0, 11)).toBeCloseTo((5 * 0.35 * 0.25) / 11, 12);
  });

  it('a crack runs on beside the last break, across bundle boundaries', () => {
    const s = compileScene([coupon()]);
    const damage = Uint8Array.from(s.fieldDamage);
    const active = (q: number) => (s.segFlags[q] & (SEG_EXISTS | SEG_BROKEN)) === SEG_EXISTS;
    // the last warp of bundle column 3 cut at a pick of row band 3; the first warp of column 4 is next
    const kEdge = 4 * fl.bu - 1;
    const mCut = 3 * fl.bv + Math.floor(fl.bv / 2) + 1;
    damage[2 * (fl.offset + mCut * fl.K + kEdge) + 1] = DAMAGE_CUT;
    coupleSpans(s.layers, s.field, s.fieldArea, damage, s, active, SEG_BROKEN);
    const strain = new Float64Array(2 * s.nodeCount).fill(0.2);
    updateField(s.layers, s.field, s.fieldStrength, s.fieldArea, damage, s, strain, active, s.crossingNormal0);
    // warp kEdge + 1 (next bundle) separated within reach of mCut
    const reach = shareReach(s.layers[0].weave, 1);
    let at = -1;
    for (let m = 0; m < fl.M; m++) if (damage[2 * (fl.offset + m * fl.K + kEdge + 1) + 1] === DAMAGE_CUT && Math.abs(m - mCut) <= reach) at = m;
    expect(at).toBeGreaterThanOrEqual(0);
  });
});
