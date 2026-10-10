import { describe, expect, it } from 'vitest';
import { axialEnergyDensity, axialStiffness, axialTension, spanResponse, strainAtTension, yarnDamage } from '../../src/simulation/axial';
import { DENIM } from '../../src/simulation/material';

const p = DENIM.yarns[1];

describe('axial J-curve', () => {
  it('tension is the derivative of energy and stiffness the derivative of tension', () => {
    for (const eps of [-0.01, 0.001, 0.01, 0.02, 0.03, 0.05, 0.08]) {
      const h = 1e-6;
      const dE = (axialEnergyDensity(p, eps + h) - axialEnergyDensity(p, eps - h)) / (2 * h);
      expect(dE).toBeCloseTo(axialTension(p, eps), 5);
      const dT = (axialTension(p, eps + h) - axialTension(p, eps - h)) / (2 * h);
      expect(dT).toBeCloseTo(axialStiffness(p, eps), 3);
    }
  });

  it('strainAtTension inverts the curve', () => {
    for (const t of [0.01, 0.1, 0.5, 2, 11]) expect(axialTension(p, strainAtTension(p, t))).toBeCloseTo(t, 9);
  });

  it('a yarn\'s cohesive softening dissipates exactly T_peak * softening / 2 beyond its peak', () => {
    // one warp of a 9-warp bundle law
    const B = 9;
    const y = { ...p, eaLow: B * p.eaLow, eaHigh: B * p.eaHigh, eaCompress: B * p.eaCompress };
    const peak = 3;
    const es = 0.05;
    const ep = strainAtTension(y, peak * B);
    const steps = 20000;
    let work = 0;
    for (let k = 0; k < steps; k++) {
      const eps = ep + ((k + 0.5) / steps) * es;
      work += ((1 - yarnDamage(y, B, 1, peak, es, eps)) * axialTension(y, eps) * es) / B / steps;
    }
    expect(work).toBeCloseTo((peak * es) / 2, 6);
    expect(yarnDamage(y, B, 1, peak, es, 0.99 * ep)).toBe(0);
    expect(yarnDamage(y, B, 1, peak, es, ep + es)).toBe(1);
    // an abraded yarn with strength in proportion to its cross-section fails at the same strain
    expect(yarnDamage(y, B, 0.3, 0.3 * peak, es, 0.99 * ep)).toBe(0);
    expect(yarnDamage(y, B, 0.3, 0.3 * peak, es, ep + es)).toBe(1);
  });

  it('damage grows monotonically with strain and the span law scales with the intact fraction', () => {
    const peak = 3;
    const es = 0.05;
    const ep = strainAtTension(p, peak);
    let prev = 0;
    for (let k = 0; k <= 50; k++) {
      const d = yarnDamage(p, 1, 1, peak, es, ep + (k / 50) * es);
      expect(d).toBeGreaterThanOrEqual(prev);
      prev = d;
    }
    const [t1, k1] = spanResponse(p, 1, 0.02);
    const [t3, k3] = spanResponse(p, 0.3, 0.02);
    expect(t3).toBeCloseTo(0.3 * t1, 12);
    expect(k3).toBeCloseTo(0.3 * k1, 12);
  });
});
