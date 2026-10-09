/**
 * Headless frame capture of the product renderer (Node + Dawn) from the app's settled drape; still
 * frames go to artifacts/captures.
 *   node tools/gpu/run.mjs tools/gpu/scripts/render.ts <name> [width height] [distance pitch]
 */
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { loadDrape } from '../../../src/simulation/gpu/settle';
import { GpuSolver, GPU_DEFAULTS, type GpuSolverParams } from '../../../src/simulation/gpu/solver';
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { gpuDevice, ROOT, saveTexture } from '../env';

/** Solver on the app's validated settled drape (public/drape.bin; regenerate with npm run gpu:drape). */
export async function settledSolver(device: GPUDevice, params: Partial<GpuSolverParams> = {}): Promise<{ solver: GpuSolver; scene: ReturnType<typeof compileScene> }> {
  const scene = compileScene(defaultSceneSpecs());
  const solver = new GpuSolver(device, scene, { ...GPU_DEFAULTS, ...params });
  const file = resolve(ROOT, 'public/drape.bin');
  const data = existsSync(file) ? readFileSync(file) : null;
  if (!data || !loadDrape(solver, data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))) {
    throw new Error('public/drape.bin is missing or stale: run npm run gpu:drape');
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
