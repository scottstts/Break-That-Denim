import type { GpuSolver } from './solver';
import { sceneHash } from '../scene';

/** Precomputed settled drape: [magic, version, node count, scene hash] then 4 f32 per node. */
const DRAPE_MAGIC = 0x50415244;
const DRAPE_VERSION = 1;

export function encodeDrape(solver: GpuSolver, positions: Float32Array): Uint8Array {
  const out = new Uint8Array(16 + positions.byteLength);
  new Uint32Array(out.buffer, 0, 4).set([DRAPE_MAGIC, DRAPE_VERSION, solver.N, sceneHash(solver.scene)]);
  out.set(new Uint8Array(positions.buffer, positions.byteOffset, positions.byteLength), 16);
  return out;
}

/** Upload a precomputed drape if it matches this scene exactly; returns false otherwise. */
export function loadDrape(solver: GpuSolver, data: ArrayBuffer): boolean {
  if (data.byteLength !== 16 + 16 * solver.N) return false;
  const h = new Uint32Array(data, 0, 4);
  if (h[0] !== DRAPE_MAGIC || h[1] !== DRAPE_VERSION || h[2] !== solver.N || h[3] !== sceneHash(solver.scene)) return false;
  solver.device.queue.writeBuffer(solver.x, 0, new Float32Array(data, 16));
  return true;
}

/** Drape settling: converged, damage-free steps with kinetic damping, then free steps. */
const SETTLE_STEPS = 420;
const SETTLE_DAMPED = 360;

/**
 * Settle the authored drape before interaction. Converged Newton solves (an unconverged solve leaves
 * a residual that keeps the cloth trembling), no history updates (pre-tension must never tear), and
 * periodic velocity removal (kinetic damping) until the last free steps confirm rest. Work is
 * submitted two substeps at a time with an idle gap so settling never monopolizes the GPU.
 */
export async function settleDrape(device: GPUDevice, solver: GpuSolver): Promise<void> {
  const p = solver.params;
  const saved = { newtonIterations: p.newtonIterations, newtonMax: p.newtonMax, pcgIterations: p.pcgIterations };
  p.newtonIterations = 5;
  p.newtonMax = 5;
  p.pcgIterations = 2;
  solver.setHistoryEnabled(false);
  try {
    for (let k = 0; k < SETTLE_STEPS; k++) {
      const enc = device.createCommandEncoder({ label: 'settle' });
      solver.encodeSubstep(enc);
      if (k % 10 === 9 && k < SETTLE_DAMPED) enc.clearBuffer(solver.v);
      device.queue.submit([enc.finish()]);
      if (k % 2 === 1) {
        const t0 = performance.now();
        await device.queue.onSubmittedWorkDone();
        await new Promise((r) => setTimeout(r, Math.min(30, performance.now() - t0)));
      }
    }
  } finally {
    Object.assign(p, saved);
    solver.setHistoryEnabled(true);
  }
  const enc = device.createCommandEncoder({ label: 'settle.end' });
  enc.clearBuffer(solver.v);
  enc.clearBuffer(solver.counters);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
}
