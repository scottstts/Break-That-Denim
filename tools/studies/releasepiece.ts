/**
 * The interaction test's crushed-piece release (tests/gpu/interaction.ts topologyRace), traced: a 3 x 3
 * piece cut free inside the pinch, held 0.3 s, let go as the worker does; per frame the piece's span
 * strains (min, max), its largest speed relative to its mean and its mean fall speed.
 *   node tools/gpu/run.mjs tools/studies/releasepiece.ts
 */
import { readBuffer } from '../../src/gpu/util';
import { Interaction } from '../../src/simulation/gpu/interaction';
import { Topology } from '../../src/simulation/gpu/topology';
import { Hand } from '../../src/simulation/hand';
import { drain, gpuDevice } from '../gpu/env';
import { settledSolver } from '../gpu/scripts/render';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (p) => interaction.encodeSubstep(p);
  solver.preAssemble = (d) => interaction.encode(d);
  const topology = new Topology(device, solver, scene, () => interaction.coreNodes);
  const nx = scene.layers[0].nx;
  const x0 = new Float32Array(await readBuffer(device, solver.x));
  const n = Math.floor(solver.N / 2);
  const ci = n % nx;
  const cj = Math.floor(n / nx);
  const flags = scene.segFlags.slice();
  for (let j = cj - 1; j <= cj + 1; j++) {
    flags[2 * (j * nx + ci - 2)] |= 2;
    flags[2 * (j * nx + ci + 1)] |= 2;
  }
  for (let i = ci - 1; i <= ci + 1; i++) {
    flags[2 * ((cj - 2) * nx + i) + 1] |= 2;
    flags[2 * ((cj + 1) * nx + i) + 1] |= 2;
  }
  solver.setSegments({ ...scene, segFlags: flags }, scene.tails);
  const piece: number[] = [];
  const spans: Array<[number, number]> = [];
  for (let j = cj - 1; j <= cj + 1; j++) {
    for (let i = ci - 1; i <= ci + 1; i++) {
      const k = j * nx + i;
      piece.push(k);
      if (i < ci + 1) spans.push([k, 0]);
      if (j < cj + 1) spans.push([k, 1]);
    }
  }
  const pending = topology.request();
  interaction.attach(n, [x0[4 * n], x0[4 * n + 1], x0[4 * n + 2]]);
  await pending;
  const hand = new Hand([x0[4 * n], x0[4 * n + 1], x0[4 * n + 2]], [0, 0, 1]);
  const step = (k: number, it = 0) => {
    for (let q = 0; q < k; q++) {
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc, it);
      device.queue.submit([enc.finish()]);
    }
  };
  const report = async (label: string) => {
    await drain(device);
    const x = new Float32Array(await readBuffer(device, solver.x));
    const v = new Float32Array(await readBuffer(device, solver.v));
    const strains = spans.map(([a, f]) => {
      const b = a + (f === 0 ? 1 : nx);
      const l = Math.hypot(x[4 * b] - x[4 * a], x[4 * b + 1] - x[4 * a + 1], x[4 * b + 2] - x[4 * a + 2]);
      return l / scene.restLength[2 * a + f] - 1;
    });
    const mean = [0, 1, 2].map((q) => piece.reduce((s, k) => s + v[4 * k + q], 0) / piece.length);
    const rel = Math.max(...piece.map((k) => Math.hypot(v[4 * k] - mean[0], v[4 * k + 1] - mean[1], v[4 * k + 2] - mean[2])));
    console.log(`${label} state ${x[4 * n + 3]} strain ${(100 * Math.min(...strains)).toFixed(1)}..${(100 * Math.max(...strains)).toFixed(1)} %, relative ${rel.toFixed(3)} m/s, mean ${Math.hypot(...mean).toFixed(3)} m/s`);
  };
  for (let k = 0; k < 36; k++) {
    hand.step(1 / 120, [0, 0, 0]);
    interaction.setTarget(hand.target);
    interaction.setPinch(hand.pinch);
    step(1);
  }
  await report('held');
  interaction.letGo(0);
  const opened = topology.request().then(() => interaction.release(0));
  step(4);
  await opened;
  await report('t=0.033');
  for (let k = 1; k <= 12; k++) {
    step(2, k <= 6 ? 4 : 0);
    await report(`t=${(0.033 + (2 * k) / 120).toFixed(3)}`);
  }
  device.destroy();
}
