/**
 * Does a dumped state come to rest? (session.ts DUMP=1: positions, spans, tails and yarn field; the
 * velocities start at zero.) Runs SECONDS (default 2) of substeps without any hand and prints, per
 * 0.1 s, the largest speed and kinetic energy of attached cloth and of detached pieces, and the
 * fastest node. Motion that keeps going from rest is energy the solve injects (a detached thread lying
 * on a flap chattering against it, say). SOLVER (JSON) overrides solver parameters.
 *   node tools/gpu/run.mjs tools/studies/restcontact.ts <state name>
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../src/gpu/util';
import { settledSolver } from '../gpu/scripts/render';
import { CAPTURES, drain, gpuDevice } from '../gpu/env';

export default async function main(args: string[]): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device, JSON.parse(process.env.SOLVER ?? '{}') as object);
  const N = solver.N;
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  let o = 0;
  const take = (bytes: number) => {
    const out = buf.slice(o, o + bytes);
    o += bytes;
    return out;
  };
  const x = new Float32Array(take(16 * N));
  device.queue.writeBuffer(solver.segs, 0, take(64 * N));
  device.queue.writeBuffer(solver.tails, 0, take(16 * N));
  device.queue.writeBuffer(solver.field, 0, take(solver.field.size));
  solver.setState(x);
  const h = solver.params.h;
  const steps = Math.round(0.1 / h);
  const seconds = Number(process.env.SECONDS ?? 2);
  for (let k = 1; k <= Math.round(seconds / 0.1); k++) {
    for (let q = 0; q < steps; q++) {
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
    }
    await drain(device);
    const xs = new Float32Array(await readBuffer(device, solver.x));
    const v = new Float32Array(await readBuffer(device, solver.v));
    const c = new Uint32Array(await readBuffer(device, solver.counters, 64));
    const ke = [0, 0];
    const vmax = [0, 0];
    let top = -1;
    for (let n = 0; n < N; n++) {
      const w = xs[4 * n + 3];
      if (w > 1.5 || scene.pinned[n]) continue;
      const s = w > 0.5 ? 1 : 0;
      const sp = Math.hypot(v[4 * n], v[4 * n + 1], v[4 * n + 2]);
      ke[s] += 0.5 * scene.mass[n] * sp * sp;
      if (sp > vmax[s]) vmax[s] = sp;
      if (sp >= Math.max(...vmax)) top = n;
    }
    console.log(`t=${(0.1 * k).toFixed(1)} cloth vmax ${vmax[0].toFixed(4)} KE ${(1e9 * ke[0]).toFixed(2)} nJ, detached vmax ${vmax[1].toFixed(4)} KE ${(1e9 * ke[1]).toFixed(2)} nJ, fastest ${top} (state ${xs[4 * top + 3]}), ruptures ${c[0]}, rejected ${c[12]}`);
  }
  device.destroy();
}
