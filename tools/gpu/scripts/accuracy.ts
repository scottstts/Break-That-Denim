/**
 * Solver accuracy vs cost on the default scene: grab, lift, release, ring down; positions every 0.05 s
 * compared with a converged run. Configs: "<newton>x<pcg>[-lean]" or "conv".
 *   node tools/gpu/run.mjs tools/gpu/scripts/accuracy.ts 2x1 3x1 ...
 */
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../../src/gpu/util';
import { Interaction } from '../../../src/simulation/gpu/interaction';
import { GpuSolver, GPU_DEFAULTS, type GpuSolverParams } from '../../../src/simulation/gpu/solver';
import { drain, gpuDevice, ROOT } from '../env';
import { settledSolver } from './render';

function config(name: string): Partial<GpuSolverParams> {
  if (name === 'conv') return { newtonIterations: 6, newtonMax: 6, pcgIterations: 4, maxCoarsestNodes: 48, coarseSweeps: 6 };
  const m = /^(\d)x(\d)(-lean)?(@240|@60|@80)?$/.exec(name);
  if (!m) throw new Error(`bad config ${name}`);
  const n = Number(m[1]);
  return { newtonIterations: n, newtonMax: n, pcgIterations: Number(m[2]), ...(m[3] ? {} : { maxCoarsestNodes: 48, coarseSweeps: 6 }), ...(m[4] ? { h: 1 / Number(m[4].slice(1)) } : {}) };
}

export default async function main(args: string[]): Promise<void> {
  const device = await gpuDevice();
  const { solver: base, scene } = await settledSolver(device);
  const x0 = new Float32Array(await readBuffer(device, base.x));
  let picked = 0;
  let best = 1e9;
  for (let n = 0; n < scene.nodeCount; n++) {
    const d = Math.hypot(x0[4 * n] + 0.03, x0[4 * n + 1] - 0.02);
    if (d < best) {
      best = d;
      picked = n;
    }
  }
  const p0: [number, number, number] = [x0[4 * picked], x0[4 * picked + 1], x0[4 * picked + 2]];
  for (const name of args) {
    const cfg = { ...GPU_DEFAULTS, ...config(name) };
    const h = cfg.h;
    const per = Math.round(0.05 / h);
    const solver = new GpuSolver(device, scene, cfg);
    device.queue.writeBuffer(solver.x, 0, x0);
    const interaction = new Interaction(device, solver, scene);
    solver.preSubstep = (pass) => interaction.encodeSubstep(pass);
    solver.preAssemble = (dispatch) => interaction.encode(dispatch);
    interaction.attach(picked, p0);
    const samples: Float32Array[] = [];
    let gpuMs = 0;
    const steps = Math.round(2.5 / h);
    for (let k = 0; k < steps; k++) {
      const t = k * h;
      if (t < 0.25) interaction.setTarget([p0[0], p0[1], p0[2] + 0.04 * t]);
      if (Math.abs(t - 0.5) < h / 2) interaction.release();
      if (k % per === 0) {
        await drain(device);
        samples.push(new Float32Array(await readBuffer(device, solver.x)));
      }
      const t0 = performance.now();
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      gpuMs += performance.now() - t0;
    }
    const file = resolve(ROOT, 'artifacts/studies/accuracy-conv.bin');
    if (name === 'conv') {
      const flat = new Float32Array(samples.length * samples[0].length);
      samples.forEach((s, q) => flat.set(s, q * s.length));
      writeFileSync(file, new Uint8Array(flat.buffer));
    }
    let report = '';
    if (name !== 'conv' && existsSync(file)) {
      const truth = new Float32Array(readFileSync(file).buffer.slice(0));
      const N4 = samples[0].length;
      const errs: number[] = [];
      for (let q = 0; q < samples.length; q++) {
        let e = 0;
        for (let n = 0; n < N4 / 4; n++) {
          const o = q * N4 + 4 * n;
          e = Math.max(e, Math.hypot(samples[q][4 * n] - truth[o], samples[q][4 * n + 1] - truth[o + 1], samples[q][4 * n + 2] - truth[o + 2]));
        }
        errs.push(e * 1e3);
      }
      report = `max err ${Math.max(...errs).toFixed(2)} mm; at 0.6/1.0/1.5/2.0/2.5 s: ${[12, 20, 30, 40, 49].map((q) => errs[Math.min(q, errs.length - 1)].toFixed(2)).join(' / ')}`;
    }
    const rc = new Uint32Array(await readBuffer(device, solver.counters, 8));
    let ke = 0;
    const vv = new Float32Array(await readBuffer(device, solver.v));
    for (let n = 0; n < scene.nodeCount; n++) ke += 0.5 * scene.mass[n] * (vv[4 * n] ** 2 + vv[4 * n + 1] ** 2 + vv[4 * n + 2] ** 2);
    report += ` rupt ${rc[0]} KE(2.5s) ${(ke * 1e6).toFixed(2)} uJ`;
    console.log(`${name.padEnd(12)} ${(gpuMs / steps).toFixed(2)} ms/substep, ${((gpuMs / steps) / h / 1000).toFixed(2)} s GPU per sim s (wall)  ${report}`);
  }
  device.destroy();
}
