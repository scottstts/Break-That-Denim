/**
 * Convergence diagnosis (CPU f64): during the grab study's pull phase, record the Newton update
 * sequence of one substep for several inner (PCG) solve strengths.
 *
 *   node tools/gpu/run.mjs tools/studies/convergence.ts
 */
import { DENIM } from '../../src/simulation/material';
import { HESSIAN, newtonStepPCG, type MGConfig } from '../../src/simulation/reference/newtonMultigrid';
import { ReferenceSolver } from '../../src/simulation/reference/solver';
import { compileScene, type LayerSpec } from '../../src/simulation/scene';
import { gripPatch } from '../../src/simulation/grip';

const LEAN3: MGConfig = { maxCoarsestNodes: 400, coarseSweeps: 2, lineColors: 3 };

const EAC = Number(process.env.EAC ?? 1);
function spec(): LayerSpec {
  return {
    material: { ...DENIM, yarns: [{ ...DENIM.yarns[0], eaCompress: DENIM.yarns[0].eaCompress * EAC }, { ...DENIM.yarns[1], eaCompress: DENIM.yarns[1].eaCompress * EAC }] },
    bundle: [3, 2],
    sizeU: 45 * 3 * DENIM.pitchU,
    sizeV: 45 * 2 * DENIM.pitchV,
    z: 0,
    angle: -0.4,
    compressU: 0.02,
    compressV: 0.01,
    shear: 0.02,
    collarRows: 2,
    wear: { center: [0, 0], axis: [1, -1], radii: [0.012, 0.009], amount: 0.9, cellSize: 0.009, featureScale: 0.01, defectThreshold: 0.85, defectDensity: Number(process.env.DEFECTS ?? 0.4) },
    wearStrengthLoss: [0.55, 0.93],
    seed: 11,
    perturbation: 0.5e-3,
  };
}

export default async function main(): Promise<void> {
  const scene = compileScene([spec()]);
  const L = scene.layers[0];
  // settle this material (converged, damage-free, kinetic damping)
  const R = new ReferenceSolver(scene, { h: 1 / 120, gravity: [0, 0, -9.81], newtonIterations: 6, pcgIterations: 6 });
  R.historyEnabled = false;
  for (let k = 0; k < 240; k++) {
    R.substep();
    if (k % 10 === 9 && k < 200) R.v.fill(0);
  }
  const settled = Array.from(R.x);
  HESSIAN.absProjection = process.env.ABS === '1';
  const picked = Math.floor(L.ny / 2) * L.nx + Math.floor(L.nx / 2);
  for (const [label, pcg, mg] of [
    ['PCG-1 lean', 1, LEAN3],
    ['PCG-3 lean', 3, LEAN3],
    ['PCG-20 lean', 20, LEAN3],
  ] as Array<[string, number, MGConfig]>) {
    const S = new ReferenceSolver(scene, { h: 1 / 120, gravity: [0, 0, -9.81], newtonIterations: 1, pcgIterations: pcg, mg });
    S.x.set(settled);
    S.historyEnabled = false;
    const patch = gripPatch(scene, picked)!;
    const p0 = [S.x[3 * picked], S.x[3 * picked + 1], S.x[3 * picked + 2]];
    const offset = new Float64Array(3 * patch.nodes.length);
    patch.nodes.forEach((n, k) => {
      for (let q = 0; q < 3; q++) offset[3 * k + q] = S.x[3 * n + q] - p0[q];
    });
    const handle: [number, number, number] = [p0[0], p0[1], p0[2]];
    S.setGrip({ nodes: Int32Array.from(patch.nodes), w: Float64Array.from(patch.weights), offset, goal: handle, k: patch.stiffness * Number(process.env.KS ?? 1) });
    // advance 6 substeps of the pull with a converged solve, then examine substep 7
    const conv: MGConfig = { maxCoarsestNodes: 48, coarseSweeps: 6, lineColors: 3 };
    for (let k = 0; k < 7; k++) {
      handle[2] += 0.15 / 120;
      S.predict();
      const seq: number[] = [];
      const iters = k < 6 ? 8 : 10;
      const where: string[] = [];
      for (let it = 0; it < iters; it++) {
        const before = Float64Array.from(S.dx);
        seq.push(newtonStepPCG(S, k < 6 ? 10 : pcg, k < 6 ? conv : mg));
        if (k === 6 && it >= 4) {
          let bm = 0;
          let bn = 0;
          for (let n = 0; n < S.N; n++) {
            const d = Math.hypot(S.dx[3 * n] - before[3 * n], S.dx[3 * n + 1] - before[3 * n + 1], S.dx[3 * n + 2] - before[3 * n + 2]);
            if (d > bm) {
              bm = d;
              bn = n;
            }
          }
          const gi = patch.nodes.indexOf(bn);
          where.push(`n${bn}(${bn % L.nx},${Math.floor(bn / L.nx)})${gi >= 0 ? 'grip' : ''}`);
        }
      }
      if (k === 6) console.log(label.padEnd(12), seq.map((m) => (m * 1e6).toFixed(3)).join(' '), 'um', where.join(' '), 'picked', picked % L.nx, Math.floor(picked / L.nx));
      S.finalize();
    }
  }
}
