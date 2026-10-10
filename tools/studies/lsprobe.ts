/**
 * Line-search diagnostics on the GPU-test coupon (f64 reference): per Newton iteration, the directional
 * derivative from the gradient and from the evaluated energy, the energy changes and the chosen step.
 *   npx vite-node tools/studies/lsprobe.ts
 */
import { DENIM } from '../../src/simulation/material';
import { ReferenceSolver } from '../../src/simulation/reference/solver';
import { chooseStep } from '../../src/simulation/reference/newtonMultigrid';
import { compileScene } from '../../src/simulation/scene';

const spec = {
  material: DENIM, bundle: [3, 2] as [number, number], sizeU: 47 * 3 * DENIM.pitchU, sizeV: 39 * 2 * DENIM.pitchV, z: 0, angle: 0.3,
  compressU: 0.03, compressV: 0.02, shear: 0.02, collarRows: 2,
  wear: { center: [0, 0] as [number, number], axis: [1, 0] as [number, number], radii: [0.03, 0.03] as [number, number], amount: 1, cellSize: 0, featureScale: 0.01, defectThreshold: 2, defectDensity: 0, sparseFrom: 2 },
  wearStrengthLoss: [0.45, 0.9] as [number, number], seed: 7, perturbation: 0.4e-3,
};
const scene = compileScene([spec]);
const cpu = new ReferenceSolver(scene, { h: 1 / 120, gravity: [0, 0, -9.81], newtonIterations: 5, pcgIterations: 3, mg: { maxCoarsestNodes: 400, coarseSweeps: 2, lineColors: 3, lagCoarse: true, lineSearch: true } });
const orig = cpu.energyChange.bind(cpu);
const counts = new Map<number, number>();
cpu.energyChange = (d, alphas, mag = alphas.map(() => 0)) => {
  const e = orig(d, alphas, mag);
  if (alphas.length === 4) {
    const a = chooseStep(e, mag);
    counts.set(a, (counts.get(a) ?? 0) + 1);
    if (process.env.VERBOSE) {
      const g = new Float64Array(3);
      let gd = 0;
      for (let n = 0; n < cpu.N; n++) {
        cpu.nodeGradient(n, g);
        gd += g[0] * d[3 * n] + g[1] * d[3 * n + 1] + g[2] * d[3 * n + 2];
      }
      const [ep, em] = orig(d, [1e-5, -1e-5]);
      console.log(`g.d=${gd.toExponential(3)} dE/da=${((ep - em) / 2e-5).toExponential(3)} e=[${e.map((v) => v.toExponential(2)).join(' ')}] -> ${a}`);
    }
  }
  return e;
};
for (let k = 0; k < Number(process.env.STEPS ?? 24); k++) cpu.substep();
console.log('chosen step lengths:', [...counts.entries()].sort((a, b) => b[0] - a[0]).map(([a, c]) => `${a}: ${c}`).join(', '));
