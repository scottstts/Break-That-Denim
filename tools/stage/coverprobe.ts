/**
 * Skin outside the collision capsules: at clip time T (default 3.4 s), every vertex farther than
 * MARGIN (default 1.5 mm) outside all active capsules, counted by the joint carrying it.
 *   node tools/gpu/run.mjs tools/stage/coverprobe.ts [T]
 */
import { Puppet } from '../../src/stage/puppet';
import { Rig, v3, type V } from '../../src/stage/rig';
import { loadClip, loadCreature } from './env';

export default async function main(args: string[]): Promise<void> {
  const t = Number(args[0] ?? 3.4);
  const margin = Number(process.env.MARGIN ?? 1.5e-3);
  const { model } = loadCreature();
  const rig = new Rig(model);
  const puppet = new Puppet(rig);
  const clip = loadClip();
  puppet.bind(clip);
  puppet.play(clip.sample(t));
  const caps = puppet.capsules().filter((c) => c.active);
  const skin = rig.skinMatrices();
  const out = new Map<string, { n: number; worst: number; z: number }>();
  for (let v = 0; v < model.position.length / 3; v++) {
    const p: V = [0, 0, 0];
    let best = 0;
    for (let q = 0; q < 4; q++) {
      const w = model.weight[4 * v + q];
      if (w > model.weight[4 * v + best]) best = q;
      if (w <= 0) continue;
      const m = skin.subarray(16 * model.jointIndex[4 * v + q], 16 * model.jointIndex[4 * v + q] + 16);
      const x = model.position[3 * v], y = model.position[3 * v + 1], z = model.position[3 * v + 2];
      for (let r = 0; r < 3; r++) p[r] += w * (m[r] * x + m[4 + r] * y + m[8 + r] * z + m[12 + r]);
    }
    if (p[2] < -0.01) continue;
    let d = Infinity;
    for (const c of caps) {
      const a = c.a as V;
      const e = v3.sub(c.b as V, a);
      const ee = v3.dot(e, e);
      const s = ee > 0 ? Math.min(1, Math.max(0, v3.dot(v3.sub(p, a), e) / ee)) : 0;
      d = Math.min(d, v3.len(v3.sub(p, v3.add(a, v3.scale(e, s)))) - c.radius);
    }
    if (d <= margin) continue;
    const name = rig.names[model.jointIndex[4 * v + best]];
    const o = out.get(name) ?? { n: 0, worst: 0, z: -1 };
    o.n++;
    o.worst = Math.max(o.worst, d);
    o.z = Math.max(o.z, p[2]);
    out.set(name, o);
  }
  for (const [name, o] of [...out].sort((a, b) => b[1].n - a[1].n)) console.log(name.padEnd(22), 'verts', String(o.n).padStart(4), 'worst', (o.worst * 1000).toFixed(1), 'mm', 'top z', (o.z * 1000).toFixed(1), 'mm');
}
