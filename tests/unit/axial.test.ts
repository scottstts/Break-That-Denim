import { describe, expect, it } from 'vitest';
import { axialEnergyDensity, axialStiffness, axialTension, damagedResponse, envelopeTension, strainAtTension } from '../../src/simulation/axial';
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

  it('cohesive softening dissipates exactly T_peak * opening / 2 beyond the peak', () => {
    const L = DENIM.pitchV;
    const peak = 3;
    const soft = 0.2e-3;
    const peakExt = strainAtTension(p, peak) * L;
    // integrate the monotonic-loading tension over extension from peak to separation
    const steps = 20000;
    let work = 0;
    for (let k = 0; k < steps; k++) {
      const ext = peakExt + ((k + 0.5) / steps) * soft;
      const [T] = damagedResponse(p, L, peak, soft, ext, ext / L);
      work += (T * soft) / steps;
    }
    expect(work).toBeCloseTo((peak * soft) / 2, 6);
    expect(envelopeTension(peak, peakExt, soft, peakExt + soft)).toBe(0);
  });

  it('unloading follows the damaged secant and never heals', () => {
    const L = DENIM.pitchV;
    const peak = 3;
    const soft = 0.2e-3;
    const peakExt = strainAtTension(p, peak) * L;
    const maxExt = peakExt + soft / 2;
    const [tUnload] = damagedResponse(p, L, peak, soft, maxExt, (0.5 * maxExt) / L);
    const [tIntact] = damagedResponse(p, L, peak, soft, 0, (0.5 * maxExt) / L);
    expect(tUnload).toBeLessThan(tIntact);
    expect(tUnload).toBeGreaterThan(0);
  });
});
