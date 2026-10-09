/**
 * GPU grab study on the full default scene: a settled drape is grabbed with the app's grip (worn and
 * intact spots), pulled at the app's handle speed, held, released and left to ring down. Ruptures and
 * positions are compared with an over-resolved GPU run ("truth").
 *
 *   node tools/gpu/run.mjs tools/gpu/scripts/grabstudy.ts <config>
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../../src/gpu/util';
import { Interaction } from '../../../src/simulation/gpu/interaction';
import { GpuSolver, GPU_DEFAULTS, type GpuSolverParams } from '../../../src/simulation/gpu/solver';
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { drain, gpuDevice, ROOT } from '../env';

const OUT = resolve(ROOT, 'artifacts/studies');
const CONFIGS: Record<string, Partial<GpuSolverParams>> = {
  truth: { h: 1 / 480, newtonIterations: 8, newtonMax: 8, pcgIterations: 4, maxCoarsestNodes: 48, coarseSweeps: 6 },
  prod: {},
  pcg2: { pcgIterations: 2 },
  c1500: { maxCoarsestNodes: 1500 },
  c1500cs4: { maxCoarsestNodes: 1500, coarseSweeps: 4 },
  cap3: { newtonMax: 3 },
  fixed2: { newtonMax: 2 },
};
const T_PULL = 0.3;
const T_END = 0.8;
const SAMPLE = 1 / 30;
const GOAL: [number, number, number] = [0.006, 0.003, 0.009];
const HANDLE_SPEED = 0.15;

async function settle(device: GPUDevice, solver: GpuSolver): Promise<void> {
  const file = resolve(OUT, 'gpu-settled.bin');
  if (existsSync(file)) {
    device.queue.writeBuffer(solver.x, 0, new Float32Array(readFileSync(file).buffer.slice(0)));
    return;
  }
  solver.setHistoryEnabled(false);
  for (let k = 0; k < 480; k++) {
    const enc = device.createCommandEncoder();
    solver.encodeSubstep(enc);
    // kinetic damping while settling: periodically remove all velocity
    if (k % 12 === 11 && k < 400) enc.clearBuffer(solver.v);
    device.queue.submit([enc.finish()]);
    if (k % 4 === 3) await drain(device);
  }
  solver.setHistoryEnabled(true);
  const c = new Uint32Array(await readBuffer(device, solver.counters, 16));
  console.log('settled; max speed over the last steps', new Float32Array(c.buffer)[2].toExponential(2), 'm/s');
  mkdirSync(OUT, { recursive: true });
  writeFileSync(file, new Uint8Array(await readBuffer(device, solver.x)));
}

export default async function main(args: string[]): Promise<void> {
  const name = args[0] ?? 'prod';
  const cfg = { ...GPU_DEFAULTS, ...CONFIGS[name] };
  const device = await gpuDevice();
  const scene = compileScene(defaultSceneSpecs());
  const L = scene.layers[0];
  const report: Record<string, unknown> = {};
  for (const spot of ['worn', 'intact'] as const) {
    const settler = new GpuSolver(device, scene, { ...GPU_DEFAULTS, newtonMax: 6, newtonTolerance: 1e-6 });
    await settle(device, settler);
    const xs0 = new Float32Array(await readBuffer(device, settler.x));
    const solver = new GpuSolver(device, scene, cfg);
    device.queue.writeBuffer(solver.x, 0, xs0);
    const interaction = new Interaction(device, solver, scene);
    solver.preAssemble = (dispatch) => interaction.encode(dispatch);
    solver.preSubstep = (pass) => interaction.encodeSubstep(pass);
    // grab: nearest node to the chosen world point
    const target = spot === 'worn' ? [0.003, -0.002] : [-0.035, 0.03];
    let picked = 0;
    let best = 1e9;
    for (let n = 0; n < scene.nodeCount; n++) {
      const d = Math.hypot(xs0[4 * n] - target[0], xs0[4 * n + 1] - target[1]);
      if (d < best && !scene.pinned[n]) {
        best = d;
        picked = n;
      }
    }
    const p0: [number, number, number] = [xs0[4 * picked], xs0[4 * picked + 1], xs0[4 * picked + 2]];
    interaction.attach(picked, p0);
    const handle: [number, number, number] = [...p0];
    const steps = Math.round(T_END / cfg.h);
    const per = Math.round(SAMPLE / cfg.h);
    const samples: Float32Array[] = [];
    const ruptures: number[] = [];
    let released = false;
    const t0 = performance.now();
    for (let k = 0; k <= steps; k++) {
      if (k % per === 0) {
        await drain(device);
        samples.push(new Float32Array(await readBuffer(device, solver.x)));
        ruptures.push(new Uint32Array(await readBuffer(device, solver.counters, 16))[0]);
      }
      if (k === steps) break;
      const tn = (k + 1) * cfg.h;
      if (!released && tn > T_PULL) {
        released = true;
        interaction.release();
      }
      if (!released) {
        const g = [p0[0] + GOAL[0], p0[1] + GOAL[1], p0[2] + GOAL[2]];
        const d = [g[0] - handle[0], g[1] - handle[1], g[2] - handle[2]];
        const len = Math.hypot(d[0], d[1], d[2]);
        const step = HANDLE_SPEED * cfg.h;
        for (let q = 0; q < 3; q++) handle[q] = len <= step ? g[q] : handle[q] + (d[q] * step) / len;
        interaction.setTarget(handle);
      }
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
      if (k % 4 === 3) await drain(device);
    }
    const wall = (performance.now() - t0) / 1000;
    const file = resolve(OUT, `gpu-${spot}-${name}.bin`);
    const flat = new Float32Array(samples.length * samples[0].length);
    samples.forEach((s, q) => flat.set(s, q * s.length));
    writeFileSync(file, new Uint8Array(flat.buffer));
    writeFileSync(file + '.json', JSON.stringify({ ruptures }));
    const line: string[] = [`${spot}: ruptures ${ruptures[ruptures.length - 1]} (${wall.toFixed(1)} s)`];
    const tf = resolve(OUT, `gpu-${spot}-truth.bin`);
    if (name !== 'truth' && existsSync(tf)) {
      const truth = new Float32Array(readFileSync(tf).buffer.slice(0));
      const tr = JSON.parse(readFileSync(tf + '.json', 'utf8')) as { ruptures: number[] };
      const N4 = samples[0].length;
      let maxErr = 0;
      for (let q = 0; q < samples.length; q++) {
        for (let n = 0; n < N4 / 4; n++) {
          const o = q * N4 + 4 * n;
          maxErr = Math.max(maxErr, Math.hypot(flat[o] - truth[o], flat[o + 1] - truth[o + 1], flat[o + 2] - truth[o + 2]));
        }
      }
      line.push(`truth ruptures ${tr.ruptures[tr.ruptures.length - 1]}, max position error ${(maxErr * 1e3).toFixed(2)} mm`);
    }
    report[spot] = line.join('; ');
    console.log(name, line.join('; '));
    void L;
  }
  device.destroy();
}
