import { describe, expect, it } from 'vitest';
import { DENIM } from '../../src/simulation/material';
import { newtonStepPCG } from '../../src/simulation/reference/newtonMultigrid';
import { ReferenceSolver } from '../../src/simulation/reference/solver';
import { compileScene, SEG_BROKEN, type LayerSpec } from '../../src/simulation/scene';

function coupon(nu: number, nv: number, overrides: Partial<LayerSpec> = {}): LayerSpec {
  return {
    material: DENIM,
    sizeU: (nu - 1) * DENIM.pitchU,
    sizeV: (nv - 1) * DENIM.pitchV,
    z: 0,
    angle: 0.3,
    compressU: 0.01,
    compressV: 0.005,
    shear: 0.01,
    collarRows: 1,
    wear: { center: [0, 0], axis: [1, 0], radii: [1, 1], amount: 0, cellSize: 0, defectThreshold: 2, defectDensity: 0 },
    wearStrengthLoss: [0, 0],
    seed: 7,
    perturbation: 0.2e-3,
    ...overrides,
  };
}

function randomize(s: ReferenceSolver, amp: number, seed = 1): void {
  let r = seed;
  const rnd = () => {
    r = (r * 1664525 + 1013904223) >>> 0;
    return r / 4294967296 - 0.5;
  };
  for (let n = 0; n < s.N; n++) {
    if (s.scene.pinned[n]) continue;
    for (let q = 0; q < 3; q++) s.dx[3 * n + q] = amp * rnd();
  }
}

describe('reference solver energies', () => {
  it('assembled gradient matches finite differences of the potential', () => {
    const scene = compileScene([coupon(9, 8)]);
    const s = new ReferenceSolver(scene);
    for (let q = 0; q < s.v.length; q++) s.v[q] = 0.01 * Math.sin(q * 1.7);
    s.predict();
    randomize(s, 2e-5);
    const g = new Float64Array(3);
    const free: number[] = [];
    for (let n = 0; n < s.N; n++) if (!scene.pinned[n]) free.push(n);
    for (const n of free.slice(0, 20)) {
      s.nodeGradient(n, g);
      for (let q = 0; q < 3; q++) {
        const h = 1e-9;
        const k = 3 * n + q;
        const x0 = s.dx[k];
        s.dx[k] = x0 + h;
        const ep = s.potential();
        s.dx[k] = x0 - h;
        const em = s.potential();
        s.dx[k] = x0;
        const fd = (ep - em) / (2 * h);
        const scale = Math.max(1e-6, Math.abs(fd), Math.abs(g[q]));
        expect(Math.abs(fd - g[q]) / scale).toBeLessThan(2e-4);
      }
    }
  });

  it('elastic energy is invariant under rigid motion', () => {
    const scene = compileScene([coupon(8, 7, { compressU: 0.0, compressV: 0.0, shear: 0 })]);
    const s = new ReferenceSolver(scene);
    randomize(s, 3e-5, 5);
    // remove inertia contribution by setting pred = dx
    s.pred.set(s.dx);
    const e0 = s.potential();
    // rotate about an arbitrary axis and translate (viscous terms compare against accepted x: move both)
    const ax = [0.3, -0.5, 0.81];
    const al = Math.hypot(ax[0], ax[1], ax[2]);
    const [kx, ky, kz] = ax.map((a) => a / al);
    const th = 0.7;
    const c = Math.cos(th);
    const sn = Math.sin(th);
    const R = [
      c + kx * kx * (1 - c), kx * ky * (1 - c) - kz * sn, kx * kz * (1 - c) + ky * sn,
      ky * kx * (1 - c) + kz * sn, c + ky * ky * (1 - c), ky * kz * (1 - c) - kx * sn,
      kz * kx * (1 - c) - ky * sn, kz * ky * (1 - c) + kx * sn, c + kz * kz * (1 - c),
    ];
    const t = [0.01, -0.02, 0.005];
    for (let n = 0; n < s.N; n++) {
      const y = [0, 1, 2].map((q) => s.x[3 * n + q] + s.dx[3 * n + q]);
      const x = [0, 1, 2].map((q) => s.x[3 * n + q]);
      for (let q = 0; q < 3; q++) {
        const ry = R[3 * q] * y[0] + R[3 * q + 1] * y[1] + R[3 * q + 2] * y[2] + t[q];
        const rx = R[3 * q] * x[0] + R[3 * q + 1] * x[1] + R[3 * q + 2] * x[2] + t[q];
        s.x[3 * n + q] = rx;
        s.dx[3 * n + q] = ry - rx;
        s.pred[3 * n + q] = s.dx[3 * n + q];
      }
    }
    const e1 = s.potential();
    expect(Math.abs(e1 - e0) / Math.max(e0, 1e-12)).toBeLessThan(1e-7);
  });
});

describe('reference solver dynamics', () => {
  it('projected Newton steps with PCG-multigrid decrease the incremental potential', () => {
    const scene = compileScene([coupon(30, 24)]);
    const s = new ReferenceSolver(scene, { h: 1 / 240, gravity: [0, 0, -9.81], newtonIterations: 2, pcgIterations: 4 });
    for (let k = 0; k < 10; k++) s.substep();
    s.predict();
    const e0 = s.potential();
    let prev = e0;
    for (let it = 0; it < 4; it++) {
      newtonStepPCG(s, 4);
      const e = s.potential();
      // No line search: the stiffening J-curve can make a near-converged step overshoot slightly.
      expect(e).toBeLessThanOrEqual(prev * 1.02);
      prev = e;
    }
    expect(prev).toBeLessThan(e0);
  });

  it('settles under gravity without blowing up and stays within the mounts', () => {
    const scene = compileScene([coupon(20, 16)]);
    const s = new ReferenceSolver(scene, { h: 1 / 240, gravity: [0, 0, -9.81], newtonIterations: 2, pcgIterations: 4 });
    for (let k = 0; k < 240; k++) s.substep();
    let maxV = 0;
    let minZ = 0;
    for (let n = 0; n < s.N; n++) {
      maxV = Math.max(maxV, Math.hypot(s.v[3 * n], s.v[3 * n + 1], s.v[3 * n + 2]));
      minZ = Math.min(minZ, s.x[3 * n + 2]);
      expect(Number.isFinite(s.x[3 * n + 2])).toBe(true);
    }
    expect(maxV).toBeLessThan(0.2);
    // A 20x16 denim coupon sags by at most a few mm, and does fall below the mount plane.
    expect(minZ).toBeLessThan(0);
    expect(minZ).toBeGreaterThan(-0.01);
  });

  it('pulling a coupon apart ruptures spans and conserves yarn material length', () => {
    const scene = compileScene([coupon(14, 10, { compressU: 0, compressV: 0, shear: 0, perturbation: 0 })]);
    const s = new ReferenceSolver(scene, { h: 1 / 240, gravity: [0, 0, 0], newtonIterations: 2, pcgIterations: 4 });
    const L0 = s.yarnMaterialLength();
    const L = scene.layers[0];
    // Move the top collar row upward along v steadily: tension builds in v-yarns until they fail.
    const top: number[] = [];
    for (let i = 0; i < L.nx; i++) top.push(L.nodeOffset + (L.ny - 1) * L.nx + i);
    const vDir = [L.mapA[1], L.mapA[3]];
    const vl = Math.hypot(vDir[0], vDir[1]);
    for (let k = 0; k < 400 && s.brokenCount < 6; k++) {
      for (const n of top) {
        s.x[3 * n] += (0.02 / 240) * (vDir[0] / vl);
        s.x[3 * n + 1] += (0.02 / 240) * (vDir[1] / vl);
      }
      s.substep();
    }
    expect(s.brokenCount).toBeGreaterThan(0);
    let broken = 0;
    for (let q = 0; q < s.segFlags.length; q++) if (s.segFlags[q] & SEG_BROKEN) broken++;
    expect(broken).toBeGreaterThanOrEqual(s.brokenCount);
    expect(Math.abs(s.yarnMaterialLength() - L0) / L0).toBeLessThan(1e-9);
  });
});
