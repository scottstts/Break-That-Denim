/**
 * Replay one detached component of a staged dump from rest, with no creature or grips. Compare the
 * production solve with ablations of contact between nodes of that same component. Contacts with
 * mounted cloth and other pieces are retained. These ablations are diagnostics, not contact fixes.
 *   node tools/gpu/run.mjs tools/studies/foldedfall.ts <state name> <node in piece>
 * FALL_TIME (default 0.25 s) sets the replay duration; FALL_VARIANTS selects comma-separated cases.
 * The isolated cases retire everything outside the piece, then compare its captured folded shape
 * with the same nodes and tear graph laid flat at their rest coordinates.
 * The "with body" cases add the creature frozen at BODY_TIME (default 6 s), without holds or
 * piercing; they also measure support against the actual blend-skinned mesh triangles.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { readBuffer } from '../../src/gpu/util';
import { compileScene } from '../../src/simulation/scene';
import { GpuSolver, GPU_DEFAULTS } from '../../src/simulation/gpu/solver';
import { Topology } from '../../src/simulation/gpu/topology';
import { BodyContact } from '../../src/simulation/gpu/body';
import { Puppet } from '../../src/stage/puppet';
import { apply4, Rig, v3, type V } from '../../src/stage/rig';
import { loadClip, loadCreature } from '../stage/env';
import { CAPTURES, drain, experimentSpecs, gpuDevice } from '../gpu/env';

type Variant = 'production' | 'no internal bounds' | 'no internal terms' | 'no internal contact' | 'isolated folded' | 'isolated flat' | 'with body' | 'with body no internal' | 'moving body' | 'moving body no internal';

/** Distance to the actual blend-skinned triangles, not dominant-joint vertices. */
function skinDistance(puppet: Puppet, model: ReturnType<typeof loadCreature>['model']): (p: V) => number {
  const skin = puppet.rig.skinMatrices();
  const vertices: V[] = [];
  for (let i = 0; i < model.position.length / 3; i++) {
    const p: V = [0, 0, 0];
    for (let q = 0; q < 4; q++) {
      const w = model.weight[4 * i + q];
      if (!w) continue;
      const j = model.jointIndex[4 * i + q];
      const v = apply4(skin.subarray(16 * j, 16 * j + 16), model.position.subarray(3 * i, 3 * i + 3));
      for (let a = 0; a < 3; a++) p[a] += w * v[a];
    }
    vertices.push(p);
  }
  const triangles: { a: V; b: V; c: V; lo: number[]; hi: number[] }[] = [];
  for (let i = 0; i < model.index.length; i += 3) {
    const a = vertices[model.index[i]], b = vertices[model.index[i + 1]], c = vertices[model.index[i + 2]];
    triangles.push({ a, b, c, lo: a.map((v, k) => Math.min(v, b[k], c[k])), hi: a.map((v, k) => Math.max(v, b[k], c[k])) });
  }
  const edge = (p: V, a: V, b: V) => {
    const e = v3.sub(b, a), d = v3.sub(p, a);
    const t = Math.min(1, Math.max(0, v3.dot(d, e) / Math.max(1e-20, v3.dot(e, e))));
    const r = v3.sub(d, v3.scale(e, t));
    return v3.dot(r, r);
  };
  return (p: V) => {
    let best = Infinity;
    for (const { a, b, c, lo, hi } of triangles) {
      let lower = 0;
      for (let k = 0; k < 3; k++) lower += Math.max(lo[k] - p[k], p[k] - hi[k], 0) ** 2;
      if (lower >= best) continue;
      const ab = v3.sub(b, a), ac = v3.sub(c, a), ap = v3.sub(p, a);
      const aa = v3.dot(ab, ab), bb = v3.dot(ab, ac), cc = v3.dot(ac, ac);
      const d = v3.dot(ap, ab), e = v3.dot(ap, ac), det = aa * cc - bb * bb;
      const u = (cc * d - bb * e) / det, v = (aa * e - bb * d) / det;
      if (det > 1e-20 && u >= 0 && v >= 0 && u + v <= 1) {
        const r = v3.sub(ap, v3.add(v3.scale(ab, u), v3.scale(ac, v)));
        best = Math.min(best, v3.dot(r, r));
      } else best = Math.min(best, edge(p, a, b), edge(p, b, c), edge(p, c, a));
    }
    return Math.sqrt(best);
  };
}

export default async function main(args: string[]): Promise<void> {
  const scene = compileScene(experimentSpecs());
  const N = scene.nodeCount;
  const data = readFileSync(resolve(CAPTURES, `${args[0]}.state`));
  const buf = data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  const x = new Float32Array(buf.slice(0, 16 * N));
  const segs = new Uint32Array(buf.slice(16 * N, 80 * N));
  const seed = Number(args[1]);
  if (!(x[4 * seed + 3] > 0.5 && x[4 * seed + 3] < 1.5)) throw new Error('Seed must be a live detached node');
  const L = scene.layers.find((l) => seed >= l.nodeOffset && seed < l.nodeOffset + l.nodeCount)!;
  const active = (q: number) => q >= 0 && (segs[8 * q + 5] & 3) === 1;
  const piece = [seed];
  const members = new Set(piece);
  let bridges = 0;
  for (let k = 0; k < piece.length; k++) {
    const n = piece[k];
    const i = (n - L.nodeOffset) % L.nx;
    const j = Math.floor((n - L.nodeOffset) / L.nx);
    const neighbors = [
      [2 * n, i + 1 < L.nx ? n + 1 : -1],
      [2 * n + 1, j + 1 < L.ny ? n + L.nx : -1],
      [2 * (n - 1), i > 0 ? n - 1 : -1],
      [2 * (n - L.nx) + 1, j > 0 ? n - L.nx : -1],
    ];
    for (const [s, m] of neighbors) {
      if (m < 0 || !active(s)) continue;
      if (x[4 * m + 3] < 0.5) bridges++;
      if (members.has(m) || !(x[4 * m + 3] > 0.5 && x[4 * m + 3] < 1.5)) continue;
      members.add(m);
      piece.push(m);
    }
  }
  piece.sort((a, b) => a - b);
  const mass = piece.reduce((s, n) => s + scene.mass[n], 0);
  const mean = (a: Float32Array, axis: number) => piece.reduce((s, n) => s + scene.mass[n] * a[4 * n + axis], 0) / mass;
  console.log(`piece ${piece[0]}: ${piece.length} nodes, ${bridges} span links to mounted cloth; mass ${(mass * 1e3).toFixed(3)} g, weight ${(mass * 9.81 * 1e3).toFixed(3)} mN`);
  // Exact membership for this captured component, compiled only into this development probe.
  const ranges: string[] = [];
  for (let k = 0; k < piece.length; ) {
    let e = k + 1;
    while (e < piece.length && piece[e] === piece[e - 1] + 1) e++;
    ranges.push(e === k + 1 ? `n == ${piece[k]}u` : `(n >= ${piece[k]}u && n <= ${piece[e - 1]}u)`);
    k = e;
  }
  const membership = `fn probePiece(n: u32) -> bool { return ${ranges.join(' || ')}; }\n`;
  const samePiece = (c: string) => `probePiece(n) && probePiece(${c}.x) && probePiece(${c}.y) && probePiece(${c}.z) && probePiece(${c}.w)`;
  const variants: Variant[] = ['production', 'no internal bounds', 'no internal terms', 'no internal contact', 'isolated folded', 'isolated flat', 'with body', 'with body no internal', 'moving body', 'moving body no internal'];
  for (const variant of variants) {
    if (process.env.FALL_VARIANTS && !process.env.FALL_VARIANTS.split(',').includes(variant)) continue;
    const device = await gpuDevice();
    const solver = new GpuSolver(device, scene, {
      ...GPU_DEFAULTS,
      patch: (module, code) => {
        if (module !== 'selfcontact') return code;
        let out = membership + code;
        if (variant === 'no internal contact' || variant.endsWith('no internal')) {
          out = out.replace('let cf = corners(f);', `let cf = corners(f);\n          if (${samePiece('cf')}) { continue; }`);
        }
        if (variant === 'no internal bounds') {
          out = out.replace('  let share = SU.law.w * max(db - HARD_CORE, 0.0);', `  let probeC = corners(fb);\n  if (${samePiece('probeC')}) {\n    cmeta[2u * n] = vec4<f32>(0.0, 0.0, 0.0, 1e9);\n    return;\n  }\n  let share = SU.law.w * max(db - HARD_CORE, 0.0);`);
        }
        if (variant === 'no internal terms') {
          const at = out.indexOf('fn terms(');
          out = out.slice(0, at) + out.slice(at).replace('let c = corners(f);', `let c = corners(f);\n    if (${samePiece('c')}) { continue; }`);
        }
        // The gradient's unused w lane reports the closest candidate face. This does not enter the
        // Newton system and lets the probe distinguish internal contacts from external supports.
        out = out.replace('vec4<f32>(v[0], v[1], v[2], 0.0)', 'vec4<f32>(v[0], v[1], v[2], f32(atomicLoad(&pairs[K * n])))');
        return out;
      },
    });
    device.queue.writeBuffer(solver.segs, 0, buf.slice(16 * N, 80 * N));
    device.queue.writeBuffer(solver.tails, 0, buf.slice(80 * N, 96 * N));
    device.queue.writeBuffer(solver.field, 0, buf.slice(96 * N));
    const initial = x.slice();
    if (variant.startsWith('isolated')) {
      for (let n = 0; n < N; n++) if (!members.has(n)) initial[4 * n + 3] = 2;
      if (variant === 'isolated flat') {
        const height = mean(x, 2);
        for (const n of piece) initial.set([scene.position[3 * n], scene.position[3 * n + 1], height, 1], 4 * n);
      }
    }
    const z0 = mean(initial, 2);
    solver.setState(initial);
    await new Topology(device, solver, scene).request();
    let support: ((xs: Float32Array) => void) | null = null;
    let moveBody: ((t: number) => void) | null = null;
    if (variant.includes('body')) {
      const { model } = loadCreature();
      const puppet = new Puppet(new Rig(model));
      const clip = loadClip();
      puppet.bind(clip);
      puppet.play(clip.sample(Number(process.env.BODY_TIME ?? 6)));
      const body = new BodyContact(device, solver, scene);
      body.setPose(puppet.capsules(), solver.params.h);
      solver.preAssemble = (dispatch) => body.encode(dispatch);
      const joints = puppet.capsuleJoints();
      if (variant.startsWith('moving')) moveBody = (t) => {
        puppet.play(clip.sample(Number(process.env.BODY_TIME ?? 6) + t));
        body.setPose(puppet.capsules(), solver.params.h);
      };
      support = (xs) => {
        const distance = skinDistance(puppet, model);
        const caps = puppet.capsules();
        const near: { n: number; capsule: string; gapMM: number; skinMM: number }[] = [];
        for (const n of piece) {
          const p = Array.from(xs.subarray(4 * n, 4 * n + 3)) as V;
          let best = Infinity, closest = -1;
          caps.forEach((c, i) => {
            if (!c.active) return;
            const e = v3.sub(c.b as V, c.a as V), d = v3.sub(p, c.a as V);
            const t = Math.min(1, Math.max(0, v3.dot(d, e) / Math.max(1e-20, v3.dot(e, e))));
            const gap = v3.len(v3.sub(d, v3.scale(e, t))) - c.radius;
            if (gap < best) { best = gap; closest = i; }
          });
          if (best < 0.002) near.push({ n, capsule: puppet.rig.names[joints[closest]], gapMM: best * 1000, skinMM: distance(p) * 1000 });
        }
        near.sort((a, b) => b.skinMM - a.skinMM);
        console.log('capsule support', JSON.stringify({ nodes: near.length, examples: near.slice(0, 5) }));
      };
      support(initial);
    }
    const h = solver.params.h;
    const steps = Math.round(Number(process.env.FALL_TIME ?? 0.25) / h);
    for (let k = 1; k <= steps; k++) {
      moveBody?.(k * h);
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
      if (k !== 1 && ![0.1, 0.25, 0.5, 1].some((t) => k === Math.round(t / h)) && k !== steps) continue;
      await drain(device);
      const [xb, vb, eb, mb] = await Promise.all([solver.x, solver.v, solver.ext, solver.contact.meta].map((b) => readBuffer(device, b)));
      const xs = new Float32Array(xb);
      const v = new Float32Array(vb);
      const ext = new Float32Array(eb);
      const meta = new Float32Array(mb);
      let internal = 0;
      let external = 0;
      let bounded = 0;
      let gradZ = 0;
      let live = 0;
      for (const n of piece) {
        if (xs[4 * n + 3] < 1.5) live++;
        gradZ += ext[12 * n + 2];
        if (Math.hypot(...meta.slice(8 * n, 8 * n + 3)) > 0.1 || Math.hypot(...meta.slice(8 * n + 4, 8 * n + 7)) > 0.1) bounded++;
        const f = ext[12 * n + 3];
        if (f >= N || f < 0) continue;
        if ([f, f + 1, f + L.nx, f + L.nx + 1].every((q) => members.has(q))) internal++;
        else external++;
      }
      console.log(`${variant.padEnd(20)} t=${(k * h).toFixed(3)} dz=${((mean(xs, 2) - z0) * 1e3).toFixed(2)} mm vz=${mean(v, 2).toFixed(4)} m/s; contact force z=${(-gradZ * 1e3).toFixed(4)} mN; candidates ${internal} internal/${external} external; bounds ${bounded}; live ${live}`);
      if (k === steps) support?.(xs);
    }
    device.destroy();
  }
}
