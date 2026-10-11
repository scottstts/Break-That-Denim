/**
 * Held thread ends drawn away from their yarn's chain (a dumped session state, session.ts DUMP=1). The
 * renderer places a run's held end at its crossing's material point (material_map.wgsl crossingPoint)
 * and shifts the chain to meet it (thread_geom.wgsl endOffset); for each chain-carried run whose shift
 * exceeds MINOFF (m, default 0.005) this prints the shift, the chain chord, and the end crossing's cell:
 * its corner spans, which of its spans hold mechanically, and how far the drawn end and the chain end
 * are from each corner. A cell stretched across a gap with its spans across the yarn broken, and a
 * drawn end between its corners, is material placed part of the way across the gap.
 *   node tools/gpu/run.mjs tools/studies/runends.ts <state name>   (SHOW: runs printed, default 6)
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../src/rendering/renderer';
import { CAPTURES, drain, gpuDevice } from '../gpu/env';
import { settledSolver } from '../gpu/scripts/render';

type V = number[];
const dist = (a: V, b: V) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const mm = (m: number) => (1e3 * m).toFixed(1);

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
  const refresh = device.createCommandEncoder();
  solver.encodeContactRefresh(refresh);
  device.queue.submit([refresh.finish()]);
  const renderer = new Renderer(device, 'rgba8unorm', solver, scene);
  renderer.resize(640, 400);
  const tex = device.createTexture({ size: [640, 400], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  for (let k = 0; k < 4; k++) {
    const enc = device.createCommandEncoder();
    renderer.encode(enc, DEFAULT_CAMERA, k / 30, tex.createView());
    device.queue.submit([enc.finish()]);
    await drain(device);
  }
  // development read of the renderer's run records (thread.wgsl RunInfo / runGeoAt layout)
  const rr = renderer as unknown as { runs: GPUBuffer; runGeo: GPUBuffer; threadCount: GPUBuffer };
  const count = new Uint32Array(await readBuffer(device, rr.threadCount))[0];
  const runs = new Uint32Array(await readBuffer(device, rr.runs));
  const geo = new Float32Array(await readBuffer(device, rr.runGeo));
  const x = new Float32Array(await readBuffer(device, solver.x));
  const segs = new Uint32Array(await readBuffer(device, solver.segs));
  const L = scene.layers[0];
  const F = scene.field[0];
  const P = (n: number): V => [x[4 * n], x[4 * n + 1], x[4 * n + 2]];
  const holds = (n: number, f: number) => (segs[8 * (2 * n + f) + 5] & 3) === 1;
  const minOff = Number(process.env.MINOFF ?? 0.005);
  let shown = 0;
  for (let r = 0; r < count && shown < Number(process.env.SHOW ?? 6); r++) {
    const g = 36 * r;
    if (geo[g] !== 1) continue;
    const rec = runs[4 * r];
    const f = (rec >>> 4) & 1;
    const heldA = ((rec >>> 5) & 3) !== 1;
    const heldB = ((rec >>> 7) & 3) !== 1;
    const inA = ((rec >>> 5) & 3) === 0;
    const inB = ((rec >>> 7) & 3) === 0;
    const yarn = runs[4 * r + 1] & 65535;
    const pos = runs[4 * r + 1] >>> 16;
    const drawA = pos - (heldA ? 1 : 0.5);
    const drawB = pos + runs[4 * r + 2] - 1 + (heldB ? 1 : 0.5);
    const pA = Array.from(geo.slice(g + 4, g + 7));
    const pB = Array.from(geo.slice(g + 8, g + 11));
    const cA = Array.from(geo.slice(g + 20, g + 23));
    const cB = Array.from(geo.slice(g + 24, g + 27));
    if (Math.max(dist(pA, cA), dist(pB, cB)) < minOff) continue;
    shown++;
    console.log(`run ${r} family ${f} yarn ${yarn} drawn ${drawA}..${drawB}: end shifts ${mm(dist(pA, cA))} / ${mm(dist(pB, cB))} mm, chain chord ${mm(dist(cA, cB))} mm`);
    const Bal = f === 0 ? F.bu : F.bv;
    const Bcr = f === 0 ? F.bv : F.bu;
    for (const [end, p, held, drawn, chain] of [['A', drawA, inA, pA, cA], ['B', drawB, inB, pB, cB]] as const) {
      if (!held) continue;
      const a = Math.round(p);
      const k = f === 0 ? a : yarn;
      const m = f === 0 ? yarn : a;
      const su = (k - 0.5 * (F.bu - 1)) / F.bu;
      const sv = (m - 0.5 * (F.bv - 1)) / F.bv;
      const ci = Math.min(Math.max(Math.floor(su), 0), L.nx - 2);
      const cj = Math.min(Math.max(Math.floor(sv), 0), L.ny - 2);
      const n00 = L.nodeOffset + cj * L.nx + ci;
      const cs = [n00, n00 + 1, n00 + L.nx, n00 + L.nx + 1];
      const along = Math.floor((p - 0.5 * (Bal - 1)) / Bal);
      const across = Math.floor(yarn / Bcr);
      const own = f === 0 ? L.nodeOffset + across * L.nx + along : L.nodeOffset + along * L.nx + across;
      console.log(
        `  end ${end}: crossing (${k}, ${m}) in cell (${ci}, ${cj}) at s ${(su - ci).toFixed(2)} t ${(sv - cj).toFixed(2)}; ` +
          `corner spans u ${mm(dist(P(cs[0]), P(cs[1])))} / ${mm(dist(P(cs[2]), P(cs[3])))} v ${mm(dist(P(cs[0]), P(cs[2])))} / ${mm(dist(P(cs[1]), P(cs[3])))} mm, ` +
          `holding u ${holds(cs[0], 0)} / ${holds(cs[2], 0)} v ${holds(cs[0], 1)} / ${holds(cs[1], 1)}; chain node is corner ${cs.indexOf(own)}; ` +
          `drawn end to corners ${cs.map((c) => mm(dist(drawn, P(c)))).join(', ')} mm, chain end ${cs.map((c) => mm(dist(chain, P(c)))).join(', ')} mm`,
      );
    }
  }
  device.destroy();
}
