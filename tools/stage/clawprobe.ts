/**
 * Claw tips over clip time: every hand's four tips (height above the cloth's plane, mm) and the highest
 * point of each sharp capsule's cap, from T0 to T1 (s) in steps of DT.
 *   node tools/gpu/run.mjs tools/stage/clawprobe.ts [T0 T1 DT]
 */
import { Puppet } from '../../src/stage/puppet';
import { Rig } from '../../src/stage/rig';
import { loadClip, loadCreature } from './env';

export default async function main(args: string[]): Promise<void> {
  const [t0, t1, dt] = [Number(args[0] ?? 0), Number(args[1] ?? 1), Number(args[2] ?? 0.05)];
  const { model } = loadCreature();
  const puppet = new Puppet(new Rig(model));
  const clip = loadClip();
  puppet.bind(clip);
  for (let t = t0; t <= t1 + 1e-9; t += dt) {
    puppet.play(clip.sample(t));
    const row = (['R', 'L'] as const).map((s) => `${s} ${puppet.clawTips(s).map((p) => (p[2] * 1000).toFixed(0).padStart(4)).join(' ')}`);
    const caps = puppet.capsules().filter((c) => c.sharp).map((c) => ((c.b[2] + c.radius) * 1000).toFixed(0).padStart(4));
    console.log(`t=${t.toFixed(2)}  tips z(mm) ${row.join(' | ')}  cap tops ${caps.join(' ')}`);
  }
}
