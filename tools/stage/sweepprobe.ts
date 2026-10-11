/**
 * How far each body capsule moves in one substep, against its radius (a capsule that moves more than its
 * radius per substep can carry cloth through itself), from T0 to T1 (s): the worst ratio per capsule.
 *   node tools/gpu/run.mjs tools/stage/sweepprobe.ts [T0 T1]
 */
import { GPU_DEFAULTS } from '../../src/simulation/gpu/solver';
import { Puppet } from '../../src/stage/puppet';
import { Rig, v3, type V } from '../../src/stage/rig';
import { loadClip, loadCreature } from './env';

export default async function main(args: string[]): Promise<void> {
  const [t0, t1] = [Number(args[0] ?? 0), Number(args[1] ?? 13.8)];
  const h = GPU_DEFAULTS.h;
  const { model } = loadCreature();
  const rig = new Rig(model);
  const puppet = new Puppet(rig);
  const clip = loadClip();
  puppet.bind(clip);
  const worst = new Map<number, { ratio: number; t: number; step: number; r: number; z: number }>();
  puppet.play(clip.sample(t0));
  let prev = puppet.capsules();
  for (let t = t0 + h; t <= t1; t += h) {
    puppet.play(clip.sample(t));
    const caps = puppet.capsules();
    caps.forEach((c, k) => {
      const step = Math.max(v3.len(v3.sub(c.a as V, prev[k].a as V)), v3.len(v3.sub(c.b as V, prev[k].b as V)));
      const ratio = step / c.radius;
      const w = worst.get(k);
      if (!w || ratio > w.ratio) worst.set(k, { ratio, t, step, r: c.radius, z: Math.max(c.a[2], c.b[2]) });
    });
    prev = caps;
  }
  const names = puppet.capsuleJoints().map((j) => rig.names[j]);
  for (const [k, w] of [...worst].sort((a, b) => b[1].ratio - a[1].ratio).slice(0, 60)) {
    console.log(`${names[k].padEnd(20)} step ${(w.step * 1000).toFixed(1).padStart(5)} mm  radius ${(w.r * 1000).toFixed(1).padStart(5)} mm  ratio ${w.ratio.toFixed(2)}  at t=${w.t.toFixed(3)}  top z ${(w.z * 1000).toFixed(0)} mm`);
  }
}
