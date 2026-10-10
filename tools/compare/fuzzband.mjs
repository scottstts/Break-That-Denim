// Loose-material band of torn cloth in an image (ours or a reference frame at the same pixel scale):
// pixels are classed as intact denim (blue, saturated), background (behind the openings: our dark
// void, or the reference's tan backing and its dark voids) or loose material (everything else: pale
// fibre, threads, fuzz), majority-filtered, then
//   width = loose area / length of the boundary between denim and the rest
// is the mean depth of the loose band per unit edge, with its mean colour and how much of it is
// bright (dense fibre) vs faint. An optional mask image (white = keep) restricts the census.
//   node tools/compare/fuzzband.mjs <image.png> <WxH+X+Y> [--px mm-per-pixel] [--sat 0.24] [--mask mask.png] [--out classes.png]
import { execFileSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';

const args = process.argv.slice(2);
const opt = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
};
const [img, crop] = args;
const mmPerPx = Number(opt('--px', '0.215'));
// saturation separating intact denim from pale loose fibre
const SAT = Number(opt('--sat', '0.24'));
const [wh] = crop.split('+');
const [W, H] = wh.split('x').map(Number);
const raw = execFileSync('magick', [img, '-crop', crop, '+repage', '-depth', '8', 'rgb:-'], { maxBuffer: 1 << 28 });
const maskPath = opt('--mask');
const mask = maskPath ? execFileSync('magick', [maskPath, '-crop', crop, '+repage', '-colorspace', 'Gray', '-depth', '8', 'gray:-'], { maxBuffer: 1 << 28 }) : null;

const toLin = (c) => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
};
const DENIM = 0, BACK = 1, LOOSE = 2;
const cls = new Uint8Array(W * H);
const lum = new Float32Array(W * H);
for (let i = 0; i < W * H; i++) {
  const r = raw[3 * i], g = raw[3 * i + 1], b = raw[3 * i + 2];
  const L = 0.2126 * toLin(r) + 0.7152 * toLin(g) + 0.0722 * toLin(b);
  lum[i] = L;
  const mx = Math.max(r, g, b);
  const sat = mx > 0 ? (mx - Math.min(r, g, b)) / mx : 0;
  if (r > b + 6 || (L < 0.004 && sat < 0.35) || mx < 6) cls[i] = BACK;
  else if (b > r && sat >= SAT) cls[i] = DENIM;
  else cls[i] = LOOSE;
}
// 5x5 majority filter: classification speckle must not count as boundary
const f = new Uint8Array(W * H);
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const n = [0, 0, 0];
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        const xx = Math.min(W - 1, Math.max(0, x + dx)), yy = Math.min(H - 1, Math.max(0, y + dy));
        n[cls[yy * W + xx]]++;
      }
    }
    f[y * W + x] = n[0] >= n[1] && n[0] >= n[2] ? 0 : n[1] >= n[2] ? 1 : 2;
  }
}
const keep = (i) => !mask || mask[i] > 127;
let loose = 0, edge = 0, back = 0, denim = 0, sumL = 0, sumR = 0, sumG = 0, sumB = 0, bright = 0;
for (let y = 0; y < H; y++) {
  for (let x = 0; x < W; x++) {
    const i = y * W + x;
    if (!keep(i)) continue;
    const c = f[i];
    if (c === LOOSE) {
      loose++;
      sumL += lum[i];
      sumR += raw[3 * i];
      sumG += raw[3 * i + 1];
      sumB += raw[3 * i + 2];
      if (lum[i] > 0.15) bright++;
    } else if (c === BACK) back++;
    else denim++;
    // denim / non-denim transitions (4-neighbour, right and down), counted once
    if (x + 1 < W && keep(i + 1) && (c === DENIM) !== (f[i + 1] === DENIM)) edge++;
    if (y + 1 < H && keep(i + W) && (c === DENIM) !== (f[i + W] === DENIM)) edge++;
  }
}
// a 4-neighbour transition count overestimates a diagonal boundary's length by up to sqrt(2): take
// the mean over orientations, 4/pi
const edgeLen = edge / (4 / Math.PI);
const widthPx = loose / Math.max(edgeLen, 1);
console.log(
  `${img.split('/').pop()} ${crop}: denim ${denim} back ${back} loose ${loose} px; edge ${edgeLen.toFixed(0)} px; ` +
    `band ${widthPx.toFixed(2)} px = ${(widthPx * mmPerPx).toFixed(2)} mm; loose mean L ${(sumL / Math.max(loose, 1)).toFixed(3)} ` +
    `sRGB ${(sumR / loose).toFixed(0)},${(sumG / loose).toFixed(0)},${(sumB / loose).toFixed(0)}; bright (L>0.15) ${((100 * bright) / Math.max(loose, 1)).toFixed(0)}%`,
);

const out = opt('--out');
if (out) {
  // class map: denim blue, background black, loose white (PNG, no dependencies)
  const pal = [
    [40, 70, 160],
    [0, 0, 0],
    [255, 255, 255],
  ];
  const rows = [];
  for (let y = 0; y < H; y++) {
    const row = Buffer.alloc(1 + 3 * W);
    for (let x = 0; x < W; x++) {
      const p = keep(y * W + x) ? pal[f[y * W + x]] : [128, 0, 0];
      row.set(p, 1 + 3 * x);
    }
    rows.push(row);
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const td = Buffer.concat([Buffer.from(type), data]);
    let c = ~0;
    for (const byte of td) {
      c ^= byte;
      for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
    }
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(~c >>> 0);
    return Buffer.concat([len, td, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(W, 0);
  ihdr.writeUInt32BE(H, 4);
  ihdr.set([8, 2, 0, 0, 0], 8);
  writeFileSync(out, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(Buffer.concat(rows))), chunk('IEND', Buffer.alloc(0))]));
}
