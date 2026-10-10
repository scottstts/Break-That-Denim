import { describe, expect, it } from 'vitest';
import { BEND_S0, bendPsi, bendPsiDelta, SHEAR_JAM, SHEAR_RAMP, shearLaw } from '../../src/simulation/laws';

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
