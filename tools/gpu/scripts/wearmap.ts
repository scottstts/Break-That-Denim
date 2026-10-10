/**
 * Maps of the authored material (no GPU): per crossing the warp and pick strength relative to intact
 * and the initial damage, over the worn zone (rest chart, rows stretched to the pick / warp pitch).
 *   node tools/gpu/run.mjs tools/gpu/scripts/wearmap.ts [out name]
 * Red = warp strength, green = pick strength (bright = intact), blue = initial cut.
 */
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { CAPTURES, encodePng } from '../env';

export default async function main(args: string[]): Promise<void> {
  const specs = defaultSceneSpecs();
  const scene = compileScene(specs);
  const F = scene.field[0];
  const w = specs[0].material;
  const k0 = 300, k1 = 1340, m0 = 60, m1 = 640;
  const W = k1 - k0, H = Math.round((m1 - m0) * 1.53);
  const rgba = new Uint8Array(4 * W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const k = k0 + x, m = m1 - Math.floor(y / 1.53);
      const c = F.offset + m * F.K + k;
      const sw = scene.fieldStrength[2 * c + 1] / w.yarns[1].strength;
      const sp = scene.fieldStrength[2 * c] / w.yarns[0].strength;
      const cut = scene.fieldDamage[2 * c + 1] === 255;
      const o = 4 * (y * W + x);
      rgba[o] = Math.min(255, 255 * sw);
      rgba[o + 1] = Math.min(255, 255 * sp);
      rgba[o + 2] = cut ? 255 : 0;
      rgba[o + 3] = 255;
    }
  }
  // weakest spots on a 2 cm grid (world position in session.ts gesture units, 10 cm framing)
  const L = scene.layers[0];
  const bu = L.bundle[0];
  const bv = L.bundle[1];
  const spots: string[] = [];
  for (let gm = m0; gm < m1; gm += 34) {
    for (let gk = k0; gk < k1; gk += 52) {
      let best = 1e9;
      let bk = 0;
      let bm = 0;
      for (let m = gm; m < gm + 34; m += 2) {
        for (let k = gk; k < gk + 52; k += 2) {
          const c = F.offset + m * F.K + k;
          const st = scene.fieldStrength[2 * c + 1] / w.yarns[1].strength + scene.fieldStrength[2 * c] / w.yarns[0].strength;
          if (st < best) [best, bk, bm] = [st, k, m];
        }
      }
      if (best > 0.6) continue;
      const n = L.nodeOffset + Math.round((bm - (bv - 1) / 2) / bv) * L.nx + Math.round((bk - (bu - 1) / 2) / bu);
      const px = scene.position[3 * n] / 3.2;
      const py = scene.position[3 * n + 1] / 3.2;
      spots.push(`(${(bk - k0)},${Math.round((m1 - bm) * 1.53)}) strength ${best.toFixed(2)} grab ${px.toFixed(4)},${py.toFixed(4)}`);
    }
  }
  console.log(spots.join('\n'));
  const path = resolve(CAPTURES, `${args[0] ?? 'wearmap'}.png`);
  writeFileSync(path, encodePng(W, H, rgba));
  console.log(path);
}
