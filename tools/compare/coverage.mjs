// Coverage of everything drawn over the openings (fringe, fibre, threads), from two stills of the same
// state over different backgrounds:
//   LIGHT='{"background":[0,0,0]}'       node tools/gpu/run.mjs tools/gpu/scripts/still.ts <state> a 1600 1000
//   LIGHT='{"background":[0.18,0.18,0.18]}' node tools/gpu/run.mjs tools/gpu/scripts/still.ts <state> b 1600 1000
//   node tools/compare/coverage.mjs artifacts/captures/a.png artifacts/captures/b.png <WxH+X+Y> [0.18] [--px mm]
// Undoes the output encoding and the tone map (post.wgsl, exposure 1), so alpha = 1 - (b - a) / B per
// pixel, and prints its mean by distance from opaque material: the fringe's coverage profile.
import { execFileSync } from 'node:child_process';

const args = process.argv.slice(2);
const pxAt = args.indexOf('--px');
const mmPerPx = pxAt >= 0 ? Number(args[pxAt + 1]) : 0.215;
const [a0, a1, crop, B = '0.18'] = args.filter((_, i) => pxAt < 0 || (i !== pxAt && i !== pxAt + 1));
const [wh] = crop.split('+');
const [W, H] = wh.split('x').map(Number);
const get = (f) => execFileSync('magick', [f, '-crop', crop, '+repage', '-depth', '8', 'rgb:-'], { maxBuffer: 1 << 28 });
const aces = (x) => Math.min(1, Math.max(0, (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)));
const inv = (y) => {
  let lo = 0, hi = 50;
  for (let i = 0; i < 50; i++) {
    const m = (lo + hi) / 2;
    if (aces(m) < y) lo = m;
    else hi = m;
  }
  return lo;
};
const lut = new Float64Array(256);
for (let c = 0; c < 256; c++) {
  const v = c / 255;
  lut[c] = inv(v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
}
const r0 = get(a0), r1 = get(a1);
const alpha = new Float32Array(W * H);
for (let i = 0; i < W * H; i++) {
  let d = 0;
  for (let k = 0; k < 3; k++) d += lut[r1[3 * i + k]] - lut[r0[3 * i + k]];
  alpha[i] = Math.min(1, Math.max(0, 1 - d / 3 / Number(B)));
}
// distance (px, chamfer) from opaque pixels
const dist = new Float32Array(W * H).fill(1e9);
for (let i = 0; i < W * H; i++) if (alpha[i] > 0.97) dist[i] = 0;
const steps = [[-1, 0, 1], [0, -1, 1], [-1, -1, Math.SQRT2], [1, -1, Math.SQRT2]];
for (let pass = 0; pass < 2; pass++) {
  const s = pass ? -1 : 1;
  for (let yy = 0; yy < H; yy++) {
    const y = pass ? H - 1 - yy : yy;
    for (let xx = 0; xx < W; xx++) {
      const x = pass ? W - 1 - xx : xx;
      const i = y * W + x;
      let d = dist[i];
      for (const [dx, dy, c] of steps) {
        const nx = x + s * dx, ny = y + s * dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        d = Math.min(d, dist[ny * W + nx] + c);
      }
      dist[i] = d;
    }
  }
}
const bins = [1, 2, 4, 6, 9, 13, 18, 25, 35];
const sum = bins.map(() => 0), cnt = bins.map(() => 0);
for (let i = 0; i < W * H; i++) {
  const d = dist[i];
  if (d === 0) continue;
  const b = bins.findIndex((v) => d <= v);
  if (b >= 0) {
    sum[b] += alpha[i];
    cnt[b]++;
  }
}
console.log(`mean coverage by distance from opaque material (mm; ${mmPerPx} mm/px):`);
console.log(bins.map((b, k) => `<=${(b * mmPerPx).toFixed(2)}: ${(sum[k] / Math.max(cnt[k], 1)).toFixed(3)}`).join('  '));
