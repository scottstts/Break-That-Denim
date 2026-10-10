import { describe, expect, it } from 'vitest';
import { GRIP, gripPatch } from '../../src/simulation/grip';
import { DENIM } from '../../src/simulation/material';
import { compileScene } from '../../src/simulation/scene';

// a 40 x 40 crossing coupon at single-yarn pitch: a 16 mm pinch spans many nodes
const scene = compileScene([
  {
    material: DENIM,
    sizeU: 39 * DENIM.pitchU,
    sizeV: 39 * DENIM.pitchV,
    z: 0,
    angle: 0,
    compressU: 0,
    compressV: 0,
    shear: 0,
    collarRows: 1,
    wear: { center: [0, 0], axis: [1, 0], radii: [1, 1], amount: 0, cellSize: 0, featureScale: 0.01, defectThreshold: 2, defectDensity: 0, sparseFrom: 2 },
    wearStrengthLoss: [0, 0],
    seed: 7,
    perturbation: 0,
  },
]);
const L = scene.layers[0];
const at = (i: number, j: number) => L.nodeOffset + j * L.nx + i;

describe('grip patches', () => {
  it('a second finger keeps only the material the first leaves, with the same force bound', () => {
    const a = gripPatch(scene, at(16, 20))!;
    const held = new Set(a.nodes);
    const b = gripPatch(scene, at(26, 20), (n) => !held.has(n))!;
    const bAlone = gripPatch(scene, at(26, 20))!;
    expect(b.nodes.length).toBeGreaterThan(0);
    expect(b.nodes.length).toBeLessThan(bAlone.nodes.length);
    for (const n of b.nodes) expect(held.has(n)).toBe(false);
    // stiffness per unit weight scales so the summed force at the largest lead is GRIP.maxForce
    for (const p of [a, b]) {
      const W = p.weights.reduce((s, w) => s + w, 0);
      expect(p.stiffness * GRIP.maxLead * W).toBeCloseTo(GRIP.maxForce, 6);
    }
  });

  it('a finger on fully held material holds nothing', () => {
    const a = gripPatch(scene, at(20, 20))!;
    const held = new Set(a.nodes);
    expect(gripPatch(scene, at(20, 20), (n) => !held.has(n))).toBeNull();
  });
});
