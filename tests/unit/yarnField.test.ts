import { describe, expect, it } from 'vitest';
import { DENIM } from '../../src/simulation/material';
import { compileScene, SEG_BROKEN, SEG_EXISTS, type LayerSpec } from '../../src/simulation/scene';
import { coupleSpans, DAMAGE_CUT, interpolatedStrain, ownSpan, spanCrossing, spanLength } from '../../src/simulation/yarnField';

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
});
