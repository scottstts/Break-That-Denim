/**
 * Loose-thread extraction check on a dumped session state (session.ts DUMP=1): renders one frame and
 * compares the GPU thread bits and run count with the runs found on the CPU from the same field.
 *   node tools/gpu/run.mjs tools/gpu/scripts/threadcheck.ts <state name>
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { CAPTURES, gpuDevice } from '../env';
import { settledSolver } from './render';

export default async function main(args: string[]): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const N = scene.nodeCount;
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  let o = 0;
  for (const b of [solver.x, solver.segs, solver.tails, solver.field]) {
    const size = b === solver.x ? 16 * N : b.size;
    device.queue.writeBuffer(b, 0, buf, o, size);
    o += size;
  }
  const renderer = new Renderer(device, 'rgba8unorm', solver, scene);
  renderer.resize(64, 64);
  const tex = device.createTexture({ size: [64, 64], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const enc = device.createCommandEncoder();
  renderer.encode(enc, DEFAULT_CAMERA, 0, tex.createView());
  device.queue.submit([enc.finish()]);
  const r = renderer as unknown as { threadBits: GPUBuffer; threadCount: GPUBuffer; runs: GPUBuffer; runGeo: GPUBuffer };
  const bits = new Uint32Array(await readBuffer(device, r.threadBits));
  const count = new Uint32Array(await readBuffer(device, r.threadCount));
  const field = new Uint32Array(await readBuffer(device, solver.field));
  const F = scene.field[0];
  const cut = (w: number, f: number) => ((w >> (8 * f)) & 0xff) === 255;
  let lone = 0;
  let inRuns = 0;
  let marked = 0;
  let runs = 0;
  const hist = new Map<number, number>();
  for (let f = 0; f < 2; f++) {
    const len = f === 0 ? F.K : F.M;
    const lines = f === 0 ? F.M : F.K;
    for (let y = 0; y < lines; y++) {
      let run = 0;
      for (let p = 0; p <= len; p++) {
        const c = p < len ? (f === 0 ? y * F.K + p : p * F.K + y) : -1;
        const isLone = c >= 0 && !cut(field[c], f) && cut(field[c], 1 - f);
        if (isLone) {
          lone++;
          run++;
          if ((bits[c >> 4] >> (2 * (c & 15) + f)) & 1) marked++;
        } else {
          if (run >= 4) {
            inRuns += run;
            runs++;
            const b = Math.min(64, 2 ** Math.ceil(Math.log2(run)));
            hist.set(b, (hist.get(b) ?? 0) + 1);
          }
          run = 0;
        }
      }
    }
  }
  const runRecs = new Uint32Array(await readBuffer(device, r.runs, 16 * Math.max(1, count[0])));
  const gh = new Map<string, number>();
  for (let q = 0; q < 4 * count[0]; q += 4) {
    const m = runRecs[q];
    const n = runRecs[q + 2];
    const key = `fam${(m >> 4) & 1} ends${(m >> 5) & 3}/${(m >> 7) & 3} n<=${Math.min(64, 2 ** Math.ceil(Math.log2(n)))}`;
    gh.set(key, (gh.get(key) ?? 0) + 1);
  }
  console.log('GPU runs by family, ends (0 held, 1 free, 2 border) and length:', [...gh].sort());
  // runs drawn much longer than their rest length (stretched across an opening)
  const geo = new Float32Array(await readBuffer(device, r.runGeo, 112 * Math.max(1, count[0])));
  const w = scene.layers[0].weave;
  const long: string[] = [];
  for (let q = 0; q < count[0]; q++) {
    const m = runRecs[4 * q];
    const f = (m >> 4) & 1;
    const rest = (runRecs[4 * q + 2] + 1) * (f === 0 ? w.pitchU : w.pitchV);
    const g = 28 * q;
    const arc = geo[g];
    const dA = Math.hypot(geo[g + 4] - geo[g + 20], geo[g + 5] - geo[g + 21], geo[g + 6] - geo[g + 22]);
    const dB = Math.hypot(geo[g + 8] - geo[g + 24], geo[g + 9] - geo[g + 25], geo[g + 10] - geo[g + 26]);
    if (arc > 3 * rest || dA > 0.003 || dB > 0.003) long.push(`run ${q} fam ${f} ends ${(m >> 5) & 3}/${(m >> 7) & 3} yarn ${runRecs[4 * q + 1] & 0xffff} pos ${runRecs[4 * q + 1] >>> 16} n ${runRecs[4 * q + 2]} chain ${geo[g + 1]} rest ${(rest * 1e3).toFixed(1)}mm arc ${(arc * 1e3).toFixed(1)}mm dA ${(dA * 1e3).toFixed(1)} dB ${(dB * 1e3).toFixed(1)}`);
  }
  console.log(`${long.length} long or displaced runs`, long.slice(0, 15));
  console.log('run lengths (<= bin):', [...hist].sort((a, b) => a[0] - b[0]));
  console.log({ gpuRuns: count[0], gpuChunks: count[1], cpuRuns: runs, lone, inRuns, marked });
  device.destroy();
}
