/** Cost of posing the creature from its clip (Stage.poseAt per substep), CPU only. */
import { Puppet } from '../../src/stage/puppet';
import { Rig } from '../../src/stage/rig';
import { loadClip, loadCreature } from './env';

export default async function main(): Promise<void> {
  const { model } = loadCreature();
  const puppet = new Puppet(new Rig(model));
  const clip = loadClip();
  puppet.bind(clip);
  const n = 2000;
  for (let pass = 0; pass < 2; pass++) {
    const t0 = performance.now();
    for (let k = 0; k < n; k++) {
      puppet.play(clip.sample((k / n) * clip.duration));
      puppet.reach('R', [0.002, -0.001, 0.003]);
      puppet.capsules();
    }
    const per = (performance.now() - t0) / n;
    if (pass === 1) console.log(`pose: ${(per * 1000).toFixed(0)} us each`);
  }
}
