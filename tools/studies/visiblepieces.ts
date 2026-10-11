/** Compare visible yarn islands with mechanical connectivity in a captured staged state.
 * node tools/gpu/run.mjs tools/studies/visiblepieces.ts <state> [clip time]
 * Reports small surface islands, their screen positions, and the nodes carrying them.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ComputeModule, createBuffer, readBuffer } from '../../src/gpu/util';
import { cameraMatrices, DEFAULT_CAMERA, Renderer } from '../../src/rendering/renderer';
import { transformPoint } from '../../src/rendering/math';
import materialMap from '../../src/rendering/wgsl/material_map.wgsl?raw';
import { CAPTURES, gpuDevice } from '../gpu/env';
import { settledSolver } from '../gpu/scripts/render';
import { loadClip } from '../stage/env';

export default async function main(args: string[]): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  let offset = 0;
  for (const b of [solver.x, solver.segs, solver.tails, solver.field]) {
    device.queue.writeBuffer(b, 0, data, offset, b.size);
    offset += b.size;
  }
  const renderer = new Renderer(device, 'rgba8unorm', solver, scene);
  renderer.resize(64, 64);
  const tex = device.createTexture({ size: [64, 64], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const enc = device.createCommandEncoder();
  renderer.encode(enc, DEFAULT_CAMERA, 0, tex.createView());
  device.queue.submit([enc.finish()]);
  const r = renderer as unknown as { common: string; renderU: GPUBuffer; rsurf: GPUBuffer; tframe: GPUBuffer; kappa: GPUBuffer; threadBits: GPUBuffer };
  const F = scene.field[0];
  const L = scene.layers[0];
  const count = F.K * F.M;
  const out = createBuffer(device, 'visiblepieces', count * 32);
  const mod = new ComputeModule(device, 'visiblepieces', r.common + `
@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> rsurf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> kappaM: array<u32>;
@group(0) @binding(5) var<storage, read_write> result: array<vec4<f32>>;
` + materialMap + `
@compute @workgroup_size(128)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  let F = R.layers[0].field;
  if (c >= F.y * F.z) { return; }
  let k = i32(c % F.y);
  let m = i32(c / F.y);
  let cc = cellOfUV(0u, f32(k), f32(m));
  let n = u32(cc.y) * R.layers[0].lattice.y + u32(cc.x);
  let nx = R.layers[0].lattice.y;
  let kap = vec4<f32>(kappaAt(0u,n,k,m), kappaAt(0u,n+1u,k,m), kappaAt(0u,n+nx,k,m), kappaAt(0u,n+nx+1u,k,m));
  let p = crossingPoint(0u,k,m);
  result[2u*c] = vec4<f32>(p.p, select(-1.0, f32(n), p.valid));
  result[2u*c+1u] = kap;
}`);
  const p = device.createCommandEncoder();
  const pass = p.beginComputePass();
  pass.setPipeline(mod.pipeline('probe'));
  pass.setBindGroup(0, mod.bindGroup('probe', { 0: r.renderU, 1: r.rsurf, 2: solver.x, 3: r.tframe, 4: r.kappa, 5: out }));
  pass.dispatchWorkgroups(Math.ceil(count / 128));
  pass.end();
  device.queue.submit([p.finish()]);
  const [pb, bb, fb, xb, sb] = await Promise.all([out, r.threadBits, solver.field, solver.x, solver.segs].map((b) => readBuffer(device, b)));
  const pts = new Float32Array(pb), bits = new Uint32Array(bb), field = new Uint32Array(fb), x = new Float32Array(xb), seg = new Uint32Array(sb);
  const surf = (c: number, f: number) => ((field[c] >>> (8 * f)) & 255) < 254 && ((bits[c >> 4] >>> (2 * (c & 15) + f)) & 1) === 0;
  const debris = (c: number) => (bits[Math.ceil(count / 16) + (c >> 5)] >>> (c & 31)) & 1;
  const active = (s: number) => (seg[8 * s + 5] & 3) === 1;
  const labels = new Int32Array(solver.N).fill(-1);
  const components: number[][] = [];
  for (let seed = 0; seed < solver.N; seed++) {
    if (labels[seed] >= 0 || x[4 * seed + 3] > 1.5) continue;
    const id = components.length, ns = [seed];
    labels[seed] = id;
    for (let at = 0; at < ns.length; at++) {
      const n = ns[at], i = n % L.nx, j = Math.floor(n / L.nx);
      for (const [m, s] of [[i + 1 < L.nx ? n + 1 : -1, 2 * n], [i > 0 ? n - 1 : -1, 2 * (n - 1)], [j + 1 < L.ny ? n + L.nx : -1, 2 * n + 1], [j > 0 ? n - L.nx : -1, 2 * (n - L.nx) + 1]]) {
        if (m < 0 || labels[m] >= 0 || !active(s) || x[4 * m + 3] > 1.5) continue;
        labels[m] = id;
        ns.push(m);
      }
    }
    components.push(ns);
  }
  const clip = loadClip().sample(Number(args[1] ?? 6));
  const vp = cameraMatrices({ ...DEFAULT_CAMERA, eye: [...clip.eye], target: [...clip.target], up: [...clip.up], fovY: clip.fovY }, 1200 / 750).viewProj;
  const seen = new Uint8Array(count);
  const islands: { cs: number[]; owners: Set<number>; mean: number[] }[] = [];
  for (let seed = 0; seed < count; seed++) {
    if (seen[seed] || (!surf(seed, 0) && !surf(seed, 1))) continue;
    const cs = [seed];
    seen[seed] = 1;
    for (let at = 0; at < cs.length; at++) {
      const c = cs[at], k = c % F.K, m = Math.floor(c / F.K);
      for (const [q, f] of [[k > 0 ? c - 1 : -1, 0], [k + 1 < F.K ? c + 1 : -1, 0], [m > 0 ? c - F.K : -1, 1], [m + 1 < F.M ? c + F.K : -1, 1]]) {
        if (q < 0 || seen[q] || !surf(c, f) || !surf(q, f) || debris(c) !== debris(q)) continue;
        seen[q] = 1;
        cs.push(q);
      }
    }
    if (cs.length < 8 || cs.length > 10000) continue;
    const owners = new Set<number>(), mean = [0, 0, 0];
    let valid = 0;
    for (const c of cs) {
      const n = pts[8 * c + 3];
      if (n < 0) continue;
      valid++;
      for (let axis = 0; axis < 3; axis++) mean[axis] += pts[8 * c + axis];
      [n, n + 1, n + L.nx, n + L.nx + 1].forEach((q, a) => { if (pts[8 * c + 4 + a] > 0.02) owners.add(q); });
    }
    if (valid === 0) continue;
    islands.push({ cs, owners, mean: mean.map((v) => v / valid) });
  }
  islands.sort((a, b) => b.cs.length - a.cs.length);
  for (const { cs, owners, mean } of islands.slice(0, 35)) {
    const projected = transformPoint(vp, mean as [number, number, number]);
    const pixel = [(0.5 + projected[0] / projected[3] / 2) * 1200, (0.5 - projected[1] / projected[3] / 2) * 750];
    const comps = [...new Set([...owners].map((n) => labels[n]))];
    console.log(JSON.stringify({ crossings: cs.length, seed: cs[0], pixel: pixel.map(Math.round), xyzMM: mean.map((v) => Math.round(v * 1000)), owners: [...owners], states: [...new Set([...owners].map((n) => x[4 * n + 3]))], components: comps.map((id) => ({ id, count: components[id]?.length, mounted: components[id]?.some((n) => scene.pinned[n]) })) }));
  }
  device.destroy();
}
