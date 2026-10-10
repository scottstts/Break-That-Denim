import { describe, expect, it } from 'vitest';
import { BEND_S0, bendGeometry, bendPsi, bendPsiDelta, bendScale, cellArea, SHEAR_JAM, SHEAR_RAMP, shearJam, shearLaw } from '../../src/simulation/laws';

const fd = (f: (x: number) => number, x: number, h: number) => (f(x + h) - f(x - h)) / (2 * h);

describe('fold-stiffening bending', () => {
  const a = 4 * 1.2e-3 * 1.2e-3;
  it('matches the quadratic energy for gentle bends and grows like tan^2 toward a fold', () => {
    const q = 1e-4 * a;
    expect(bendPsi(q, a)[0] / q).toBeCloseTo(1, 3);
    // equal spans: q / a = sin^2(theta/2), Psi = a tan^2(theta/2)
    const th = 2.0;
    const s = Math.sin(th / 2) ** 2;
    expect(bendPsi(s * a, a)[0]).toBeCloseTo(a * Math.tan(th / 2) ** 2, 12);
  });
  it('has consistent derivatives on both branches and is C2 at the joint', () => {
    for (const s of [0.1, 0.5, 0.8, BEND_S0, 0.95, 1.3]) {
      const q = s * a;
      const [, d1, d2] = bendPsi(q, a);
      expect(fd((x) => bendPsi(x, a)[0], q, 1e-7 * a) / d1).toBeCloseTo(1, 5);
      expect(fd((x) => bendPsi(x, a)[1], q, 1e-7 * a) / d2).toBeCloseTo(1, 4);
    }
    const q0 = BEND_S0 * a;
    const lo = bendPsi(q0 * (1 - 1e-12), a);
    const hi = bendPsi(q0 * (1 + 1e-12), a);
    for (let k = 0; k < 3; k++) expect(hi[k] / lo[k]).toBeCloseTo(1, 6);
  });
  it('computes energy changes without cancellation', () => {
    const q0 = 0.3 * a;
    const dq = 1e-9 * a;
    expect(bendPsiDelta(q0, q0 + dq, dq, a) / (bendPsi(q0, a)[1] * dq)).toBeCloseTo(1, 6);
  });
});

describe('shear locking', () => {
  const ks = 60;
  const Ld = 1.6e-3;
  it('is linear until the jam strain and stiffens over the ramp', () => {
    expect(shearLaw(ks, Ld, Ld * 1.1)[1]).toBe(ks);
    expect(shearLaw(ks, Ld, Ld * (1 - SHEAR_JAM + 0.01))[1]).toBe(ks);
    expect(shearLaw(ks, Ld, Ld * (1 - SHEAR_JAM - SHEAR_RAMP - 0.05))[1]).toBeCloseTo(40 * ks, 9);
  });
  it('force and stiffness are derivatives of the energy on every branch', () => {
    for (const r of [1.2, 0.9, 1 - SHEAR_JAM - 0.5 * SHEAR_RAMP, 1 - SHEAR_JAM - 2 * SHEAR_RAMP]) {
      const l = r * Ld;
      const [f, k] = shearLaw(ks, Ld, l);
      expect(fd((x) => shearLaw(ks, Ld, x)[2], l, 1e-9) / f).toBeCloseTo(1, 5);
      expect(fd((x) => shearLaw(ks, Ld, x)[0], l, 1e-9) / k).toBeCloseTo(1, 5);
    }
  });
});

describe('wear-dependent shear jam', () => {
  it('keeps the sound jam and turns further as abrasion thins the yarns', () => {
    expect(shearJam(1)).toBeCloseTo(SHEAR_JAM, 12);
    // lock angle acos(cos0 sqrt(a)): area 0.5 locks near 47 degrees
    const g = Math.asin(1 - (1 - shearJam(0.5)) ** 2);
    expect((g * 180) / Math.PI).toBeGreaterThan(44);
    expect((g * 180) / Math.PI).toBeLessThan(50);
    for (let a = 0.05; a < 1; a += 0.05) expect(shearJam(a)).toBeGreaterThan(shearJam(a + 0.05));
    // the denser family governs
    expect(cellArea(1, 1, 0.2, 0.4)).toBe(1);
    expect(cellArea(0, 0.4, 0.3, 0.3)).toBeCloseTo(0.3, 12);
  });
});

describe('bending of unequal spans', () => {
  const curvature = (La: number, Lc: number, pa: number[], pb: number[], pc: number[]) => {
    const g = bendGeometry(La, Lc);
    return [0, 1, 2].map((k) => g.wa * pa[k] - 2 * pb[k] + g.wc * pc[k]);
  };
  it('reduces to the second difference for equal spans', () => {
    const L = 3.4e-3;
    const g = bendGeometry(L, L);
    expect(g.wa).toBeCloseTo(1, 14);
    expect(g.wc).toBeCloseTo(1, 14);
    expect(g.kPerB * L ** 3).toBeCloseTo(1, 12);
    expect(g.fold / (4 * L * L)).toBeCloseTo(1, 12);
  });
  it('has no bending energy in a straight chain of unequal spans strained alike', () => {
    // the assessment's case: rest 3 mm and 4 mm (after crossing slip), straight and unstretched
    for (const strain of [0, 0.05]) {
      const s = 1 + strain;
      const r = curvature(3e-3, 4e-3, [0, 0, 0], [3e-3 * s, 0, 0], [7e-3 * s, 0, 0]);
      expect(Math.hypot(r[0], r[1], r[2])).toBeLessThan(1e-18);
    }
  });
  it('gives the rod energy B kappa^2 l / 2 on a circular arc', () => {
    const B = 2.4e-7;
    const R = 0.02;
    const La = 2.6e-3;
    const Lc = 4.1e-3;
    // points on the circle at arc lengths -La, 0, Lc (chords differ from arcs at O((L/R)^2))
    const at = (s: number) => [R * Math.sin(s / R), R * (1 - Math.cos(s / R)), 0];
    const r = curvature(La, Lc, at(-La), at(0), at(Lc));
    const g = bendGeometry(La, Lc);
    const e = 0.5 * B * g.kPerB * (r[0] ** 2 + r[1] ** 2 + r[2] ** 2);
    const exact = (0.5 * B * (0.5 * (La + Lc))) / (R * R);
    expect(e / exact).toBeCloseTo(1, 2);
  });
  it('bends only through yarns continuous across the middle node', () => {
    const B = 9;
    expect(bendScale(1, 1, 0, 0, B)).toBeCloseTo(1, 12);
    // the same three yarns cut in both spans: the rest bend, at the weaker span's intactness
    expect(bendScale(6 / 9, 0.5, 0b111, 0b111, B)).toBeCloseTo(0.5, 12);
    // disjoint cuts leave no yarn continuous through the node although both spans carry load
    expect(bendScale(4 / 9, 4 / 9, 0b000011111, 0b111110000, B)).toBe(0);
  });
});
