/**
 * Headless tear sequence: grab the settled drape at a world point, pull the handle along a path at the
 * app's handle speed, and render frames along the way (artifacts/captures/<name>-NN.png).
 *   node tools/gpu/run.mjs tools/gpu/scripts/tear.ts <name> [x y] [dx dy dz (m)] [seconds]
 */
import { readBuffer } from '../../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { Interaction } from '../../../src/simulation/gpu/interaction';
import { drain, gpuDevice, saveTexture } from '../env';
import { settledSolver } from './render';

const HANDLE_SPEED = 0.2;

export default async function main(args: string[]): Promise<void> {
  const name = args[0] ?? 'tear';
  const at = [Number(args[1] ?? 0.003), Number(args[2] ?? -0.002)];
  const pull = [Number(args[3] ?? 0.012), Number(args[4] ?? 0.006), Number(args[5] ?? 0.02)];
  const seconds = Number(args[6] ?? 1.2);
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (pass) => interaction.encodeSubstep(pass);
  solver.preAssemble = (dispatch) => interaction.encode(dispatch);
  const format: GPUTextureFormat = 'rgba8unorm';
  const width = 1600;
  const height = 1000;
  const renderer = new Renderer(device, format, solver, scene);
  renderer.resize(width, height);
  const tex = device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const xs = new Float32Array(await readBuffer(device, solver.x));
  let picked = 0;
  let best = 1e9;
  for (let n = 0; n < scene.nodeCount; n++) {
    const d = Math.hypot(xs[4 * n] - at[0], xs[4 * n + 1] - at[1]);
    if (d < best && !scene.pinned[n]) {
      best = d;
      picked = n;
    }
  }
  const p0: [number, number, number] = [xs[4 * picked], xs[4 * picked + 1], xs[4 * picked + 2]];
  interaction.attach(picked, p0);
  const handle: [number, number, number] = [...p0];
  const goal = [p0[0] + pull[0], p0[1] + pull[1], p0[2] + pull[2]];
  const h = solver.params.h;
  const steps = Math.round(seconds / h);
  const frames = 6;
  let frame = 0;
  const t0 = performance.now();
  for (let k = 0; k <= steps; k++) {
    if (k % Math.round(steps / (frames - 1)) === 0 || k === steps) {
      await drain(device);
      const enc = device.createCommandEncoder();
      renderer.encode(enc, DEFAULT_CAMERA, k * h, tex.createView());
      device.queue.submit([enc.finish()]);
      await saveTexture(device, tex, `${name}-${String(frame++).padStart(2, '0')}`);
      const c = new Uint32Array(await readBuffer(device, solver.counters, 8));
      console.log(`t=${(k * h).toFixed(2)} s ruptures ${c[0]} pulled out ${c[1]}`);
    }
    if (k === steps) break;
    const d = [goal[0] - handle[0], goal[1] - handle[1], goal[2] - handle[2]];
    const len = Math.hypot(d[0], d[1], d[2]);
    const st = HANDLE_SPEED * h;
    for (let q = 0; q < 3; q++) handle[q] = len <= st ? goal[q] : handle[q] + (d[q] * st) / len;
    interaction.setTarget(handle);
    const enc = device.createCommandEncoder();
    solver.encodeSubstep(enc);
    device.queue.submit([enc.finish()]);
    if (k % 3 === 2) await drain(device);
  }
  console.log(`simulated ${seconds} s in ${((performance.now() - t0) / 1000).toFixed(1)} s (with throttling)`);
  device.destroy();
}
