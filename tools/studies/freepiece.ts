/**
 * How the cloth solve moves a piece that nothing anchors (cut free on all sides, not yet made ballistic
 * by the connectivity analysis) or that hangs by a single span, from rest: an exact solve lets it fall
 * (about g t), so speeds far above that are the solve flinging it.
 * With GRIP=1 the piece is pinched at its centre by the product's grip and hand, held HOLD seconds and
 * released (mode intact: no cuts), and the piece is followed after the release.
 *   node tools/gpu/run.mjs tools/studies/freepiece.ts [free|hinge|intact] [half]
 */
import { readBuffer } from '../../src/gpu/util';
import { settledSolver } from '../gpu/scripts/render';
import { drain, gpuDevice } from '../gpu/env';
import { Interaction } from '../../src/simulation/gpu/interaction';
import { GRIP } from '../../src/simulation/grip';
import { Hand } from '../../src/simulation/hand';

export default async function main(args: string[]): Promise<void> {
  const mode = args[0] ?? 'free';
  const half = Number(args[1] ?? 3);
  const device = await gpuDevice();
  Object.assign(GRIP, JSON.parse(process.env.GRIPP ?? '{}') as object);
  const { solver, scene } = await settledSolver(device, JSON.parse(process.env.SOLVER ?? '{}') as object);
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (p) => interaction.encodeSubstep(p);
  solver.preAssemble = (d) => interaction.encode(d);
  const L = scene.layers[0];
  // a square of (2 half + 1)^2 crossings in intact cloth (x = -0.18 m)
  let at = 0;
  let best = Infinity;
  for (let n = 0; n < scene.nodeCount; n++) {
    const d = Math.hypot(scene.position[3 * n] + 0.18, scene.position[3 * n + 1]);
    if (d < best) [best, at] = [d, n];
  }
  const i0 = at % L.nx;
  const j0 = Math.floor(at / L.nx);
  const segs = new Uint32Array(await readBuffer(device, solver.segs));
  const cut = (n: number, f: number) => (segs[8 * (2 * n + f) + 5] |= 2);
  for (let j = j0 - half; j <= j0 + half && mode !== 'intact'; j++) {
    cut(j * L.nx + i0 - half - 1, 0);
    if (!(mode === 'hinge' && j === j0)) cut(j * L.nx + i0 + half, 0);
  }
  for (let i = i0 - half; i <= i0 + half && mode !== 'intact'; i++) {
    cut((j0 - half - 1) * L.nx + i, 1);
    cut((j0 + half) * L.nx + i, 1);
  }
  device.queue.writeBuffer(solver.segs, 0, segs);
  const piece: number[] = [];
  for (let j = j0 - half; j <= j0 + half; j++) for (let i = i0 - half; i <= i0 + half; i++) piece.push(j * L.nx + i);
  const h = solver.params.h;
  if (process.env.GRIP) {
    const x = new Float32Array(await readBuffer(device, solver.x));
    const p: [number, number, number] = [x[4 * at], x[4 * at + 1], x[4 * at + 2]];
    interaction.attach(at, p);
    const hand = new Hand(p, [0, 0, 1]);
    const steps = Math.round(Number(process.env.HOLD ?? 0.4) / h);
    for (let k = 0; k < steps; k++) {
      hand.step(h, [0, 0, 0]);
      interaction.setTarget(hand.target);
      interaction.setPinch(hand.pinch);
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
    }
    await drain(device);
    interaction.release();
  }
  for (let k = 1; k <= 24; k++) {
    const enc = device.createCommandEncoder();
    solver.encodeSubstep(enc, process.env.GRIP && k <= 12 ? 4 : 0);
    device.queue.submit([enc.finish()]);
    if (k % 2) continue;
    await drain(device);
    const v = new Float32Array(await readBuffer(device, solver.v));
    const sp = piece.map((n) => Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]));
    const com = [0, 1, 2].map((q) => piece.reduce((a, n) => a + v[4 * n + q], 0) / piece.length);
    console.log(`t=${(k * h).toFixed(3)} piece speed max ${Math.max(...sp).toFixed(3)} m/s, centre-of-mass velocity ${com.map((q) => q.toFixed(3)).join(',')} (free fall ${(9.81 * k * h).toFixed(3)})`);
  }
  device.destroy();
}
