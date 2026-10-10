import { describe, expect, it } from 'vitest';
import { GRIP, gripPatch, patchWindowRadius, type PatchWindow } from '../../src/simulation/grip';
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
    const held = new Set([...a.nodes, ...a.between]);
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
    const held = new Set([...a.nodes, ...a.between]);
    expect(gripPatch(scene, at(20, 20), (n) => !held.has(n))).toBeNull();
  });
});

/** An intact connectivity window around `node` (every own span survives, all cloth). */
function intactWindow(node: number, radius: number): PatchWindow {
  const side = 2 * radius + 1;
  const bits = new Uint32Array(side * side);
  const i0 = (node - L.nodeOffset) % L.nx;
  const j0 = Math.floor((node - L.nodeOffset) / L.nx);
  for (let dj = -radius; dj <= radius; dj++) {
    for (let di = -radius; di <= radius; di++) {
      const i = i0 + di;
      const j = j0 + dj;
      const k = (dj + radius) * side + (di + radius);
      bits[k] = i < 0 || j < 0 || i >= L.nx || j >= L.ny ? 8 : (i + 1 < L.nx ? 1 : 0) | (j + 1 < L.ny ? 2 : 0);
    }
  }
  return { radius, bits };
}

describe('grip patches through current connectivity', () => {
  const node = at(20, 20);
  const R = patchWindowRadius(scene);
  const side = 2 * R + 1;
  const cell = (di: number, dj: number) => (dj + R) * side + (di + R);

  it('intact cloth gives the rest-space disc', () => {
    const disc = gripPatch(scene, node)!;
    const grown = gripPatch(scene, node, () => true, intactWindow(node, R))!;
    expect(grown.nodes).toEqual(disc.nodes);
    expect(grown.weights).toEqual(disc.weights);
  });

  it('the far lip of a slit beside the pinch is not held', () => {
    const win = intactWindow(node, R);
    // cut every u-span between columns +1 and +2 over the whole window: a slit longer than the patch
    for (let dj = -R; dj <= R; dj++) win.bits[cell(1, dj)] &= ~1;
    const p = gripPatch(scene, node, () => true, win)!;
    const disc = gripPatch(scene, node)!;
    const col = (n: number) => ((n - L.nodeOffset) % L.nx) - 20;
    expect(p.nodes.some((n) => col(n) >= 2)).toBe(false);
    expect(disc.nodes.some((n) => col(n) >= 2)).toBe(true);
    // the near side is still held, and the force bound is renormalized over what is held
    expect(p.nodes.some((n) => col(n) === 1)).toBe(true);
    const W = p.weights.reduce((s, w) => s + w, 0);
    expect(p.stiffness * GRIP.maxLead * W).toBeCloseTo(GRIP.maxForce, 6);
  });

  it('a slit that ends inside the patch still lets the pinch reach around it', () => {
    const win = intactWindow(node, R);
    // a short cut between columns +1 and +2, rows -1..+1: material beyond it is reached around its ends
    for (let dj = -1; dj <= 1; dj++) win.bits[cell(1, dj)] &= ~1;
    const p = gripPatch(scene, node, () => true, win)!;
    expect(p.nodes).toContain(at(22, 20));
  });

  it('debris and a picked node that is no longer cloth hold nothing', () => {
    const win = intactWindow(node, R);
    win.bits[cell(1, 0)] |= 4;
    const p = gripPatch(scene, node, () => true, win)!;
    expect(p.nodes).not.toContain(at(21, 20));
    win.bits[cell(0, 0)] |= 4;
    expect(gripPatch(scene, node, () => true, win)).toBeNull();
  });
});

describe('pad pinch', () => {
  const node = at(20, 20);
  const R = patchWindowRadius(scene);
  const side = 2 * R + 1;
  const cell = (di: number, dj: number) => (dj + R) * side + (di + R);
  const col = (n: number) => ((n - L.nodeOffset) % L.nx) - 20;
  const row = (n: number) => Math.floor((n - L.nodeOffset) / L.nx) - 20;

  it('holds two pads either side of the grab line along u; the line between is only covered', () => {
    expect(GRIP.pinchMode).toBe('pads');
    const p = gripPatch(scene, node)!;
    expect(p.padPitch).toBeCloseTo(L.material.pitchU, 12);
    expect(p.between.every((n) => col(n) === 0)).toBe(true);
    expect(p.between).toContain(node);
    p.nodes.forEach((n, k) => expect(p.sides[k]).toBe(Math.sign(col(n))));
    expect(p.sides).toContain(1);
    expect(p.sides).toContain(-1);
    const W = p.weights.reduce((s, w) => s + w, 0);
    expect(p.stiffness * GRIP.maxLead * W).toBeCloseTo(GRIP.maxForce, 6);
  });

  it('closes along v on a strand that runs along v', () => {
    const win = intactWindow(node, R);
    // only the grab column survives, as a strand along v
    for (let dj = -R; dj <= R; dj++) for (let di = -R; di <= R; di++) win.bits[cell(di, dj)] = di === 0 ? win.bits[cell(di, dj)] & 2 : 4;
    const p = gripPatch(scene, node, () => true, win)!;
    expect(p.nodes.every((n) => col(n) === 0)).toBe(true);
    p.nodes.forEach((n, k) => expect(p.sides[k]).toBe(Math.sign(row(n))));
    expect(p.padPitch).toBeCloseTo(L.material.pitchV, 12);
  });

  it('falls back to the radial pinch with cloth on one side only along both axes', () => {
    const win = intactWindow(node, R);
    // a corner of cloth: everything left of or below the grab point is gone
    for (let dj = -R; dj <= R; dj++) for (let di = -R; di <= R; di++) if (di < 0 || dj < 0) win.bits[cell(di, dj)] = 4;
    const p = gripPatch(scene, node, () => true, win)!;
    expect(p.sides.every((q) => q === 0)).toBe(true);
    expect(p.between).toEqual([]);
  });
});
