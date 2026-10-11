/**
 * How far the creature's collision capsules stand off its skin, at clip time T (s, default 6): per
 * capsule, points sampled over its surface against the nearest skin vertex (every vertex, each carried
 * rigidly by its dominant joint, as the capsules are). Cloth resting on a capsule floats this far off
 * the visible body.
 *   node tools/gpu/run.mjs tools/stage/colliderfit.ts [T]
 */
import { Puppet } from '../../src/stage/puppet';
import { Rig, v3, type V } from '../../src/stage/rig';
import { loadClip, loadCreature } from './env';

export default async function main(args: string[]): Promise<void> {
  const T = Number(args[0] ?? 6);
  const { model } = loadCreature();
  const rig = new Rig(model);
  const puppet = new Puppet(rig);
  const clip = loadClip();
  puppet.bind(clip);
  puppet.play(clip.sample(T));
  const carried = rig.carriedVertices();
  const skin: V[] = [];
  carried.forEach((pts, k) => pts.forEach((p) => skin.push(rig.point(k, p))));
  // a uniform grid over the skin for nearest-vertex queries
  const cell = 0.01;
  const grid = new Map<string, V[]>();
  const key = (p: V) => p.map((x) => Math.floor(x / cell)).join(',');
  for (const p of skin) {
    const k = key(p);
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k)!.push(p);
  }
  const nearest = (p: V): number => {
    const c = p.map((x) => Math.floor(x / cell));
    for (let r = 1; r < 30; r++) {
      let best = Infinity;
      for (let i = -r; i <= r; i++) for (let j = -r; j <= r; j++) for (let k = -r; k <= r; k++) {
        for (const q of grid.get(`${c[0] + i},${c[1] + j},${c[2] + k}`) ?? []) best = Math.min(best, v3.len(v3.sub(p, q)));
      }
      if (best <= r * cell) return best;
    }
    return Infinity;
  };
  const caps = puppet.capsules();
  const names = puppet.capsuleJoints().map((j) => rig.names[j]);
  const rows: Array<[string, number, number, number, number]> = [];
  caps.forEach((c, k) => {
    const e = v3.sub(c.b as V, c.a as V);
    const L = v3.len(e);
    const ax = L > 1e-9 ? v3.scale(e, 1 / L) : ([0, 0, 1] as V);
    const u = v3.norm(v3.cross(ax, Math.abs(ax[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
    const w = v3.cross(ax, u);
    const gaps: number[] = [];
    // the cylinder and both caps, evenly in angle
    for (let s = 0; s <= 8; s++) {
      for (let q = 0; q < 16; q++) {
        const th = (2 * Math.PI * q) / 16;
        const dir = v3.add(v3.scale(u, Math.cos(th)), v3.scale(w, Math.sin(th)));
        gaps.push(nearest(v3.add(v3.add(c.a as V, v3.scale(ax, (L * s) / 8)), v3.scale(dir, c.radius))));
      }
    }
    for (const [end, sign] of [[c.a, -1], [c.b, 1]] as const) {
      for (let p = 1; p <= 3; p++) {
        for (let q = 0; q < 16; q++) {
          const th = (2 * Math.PI * q) / 16;
          const ph = (Math.PI / 2) * (p / 3);
          const dir = v3.add(v3.scale(ax, sign * Math.sin(ph)), v3.scale(v3.add(v3.scale(u, Math.cos(th)), v3.scale(w, Math.sin(th))), Math.cos(ph)));
          gaps.push(nearest(v3.add(end as V, v3.scale(dir, c.radius))));
        }
      }
    }
    gaps.sort((a, b) => a - b);
    const at = (f: number) => gaps[Math.round(f * (gaps.length - 1))];
    rows.push([names[k], c.radius, at(0.5), at(0.9), gaps.filter((g) => g > 0.01).length / gaps.length]);
  });
  rows.sort((a, b) => b[3] - a[3]);
  for (const [n, r, med, p90, far] of rows) console.log(`${n.padEnd(20)} radius ${(r * 1e3).toFixed(1).padStart(5)} mm  skin gap median ${(med * 1e3).toFixed(1).padStart(5)} p90 ${(p90 * 1e3).toFixed(1).padStart(5)} mm  over 10 mm ${(100 * far).toFixed(0).padStart(3)} %`);
}
