/**
 * Headless frame capture of the product renderer (Node + Dawn): settles the default drape (cached
 * under artifacts/studies) and renders still frames to artifacts/captures.
 *   node tools/gpu/run.mjs tools/gpu/scripts/render.ts <name> [width height] [distance pitch]
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { settleDrape } from '../../../src/simulation/gpu/settle';
import { GpuSolver, GPU_DEFAULTS } from '../../../src/simulation/gpu/solver';
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { gpuDevice, ROOT, saveTexture } from '../env';

export async function settledSolver(device: GPUDevice): Promise<{ solver: GpuSolver; scene: ReturnType<typeof compileScene> }> {
  const scene = compileScene(defaultSceneSpecs());
  const solver = new GpuSolver(device, scene, GPU_DEFAULTS);
  const dir = resolve(ROOT, 'artifacts/studies');
  const file = resolve(dir, `drape-${scene.nodeCount}.bin`);
  if (existsSync(file)) {
    device.queue.writeBuffer(solver.x, 0, new Float32Array(readFileSync(file).buffer.slice(0)));
  } else {
    const t0 = performance.now();
    await settleDrape(device, solver);
    console.log(`settled in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    mkdirSync(dir, { recursive: true });
    writeFileSync(file, new Uint8Array(await readBuffer(device, solver.x)));
  }
  return { solver, scene };
}

export default async function main(args: string[]): Promise<void> {
  const name = args[0] ?? 'frame';
  const width = Number(args[1] ?? 1600);
  const height = Number(args[2] ?? 1000);
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const format: GPUTextureFormat = 'rgba8unorm';
  const renderer = new Renderer(device, format, solver, scene);
  renderer.resize(width, height);
  const camera = { ...DEFAULT_CAMERA };
  if (args[3]) camera.distance = Number(args[3]);
  if (args[4]) camera.pitch = Number(args[4]);
  const tex = device.createTexture({ size: [width, height], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const enc = device.createCommandEncoder();
  renderer.encode(enc, camera, 0, tex.createView());
  device.queue.submit([enc.finish()]);
  console.log(await saveTexture(device, tex, name));
  device.destroy();
}
