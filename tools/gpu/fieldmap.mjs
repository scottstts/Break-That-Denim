// Yarn damage field map of a dumped session state (session.ts DUMP=1), cropped to the damage, rest
// chart (rows stretched to the pick / warp pitch ratio): black = both yarns cut, cream = warp cut
// (pick shows), cyan = pick cut, blue shades = warp damage. Lattice size is the default scene's.
//   node tools/gpu/fieldmap.mjs artifacts/captures/<name>.state out.png [pixels per crossing]
import { readFileSync, writeFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
const [file, out, sc] = process.argv.slice(2);
const S = Number(sc ?? 2);
const nx = Math.floor(0.62 / (9 * 0.38e-3)) + 1, ny = Math.floor(0.4 / (6 * 0.58e-3)) + 1;
const N = nx * ny, K = nx * 9, M = ny * 6;
const b = readFileSync(file);
const off = 16 * N + 64 * N + 16 * N;
const f = new Uint32Array(b.buffer.slice(b.byteOffset + off, b.byteOffset + off + 4 * K * M));
let k0 = 1e9, k1 = -1, m0 = 1e9, m1 = -1;
for (let m = 0; m < M; m++) for (let k = 0; k < K; k++) { const w = f[m * K + k]; if ((w & 0xff) > 40 || ((w >> 8) & 0xff) > 40) { k0 = Math.min(k0, k); k1 = Math.max(k1, k); m0 = Math.min(m0, m); m1 = Math.max(m1, m); } }
if (k1 < 0) { console.log('no damage'); process.exit(0); }
k0 = Math.max(0, k0 - 30); k1 = Math.min(K - 1, k1 + 30); m0 = Math.max(0, m0 - 20); m1 = Math.min(M - 1, m1 + 20);
// physical aspect: pick pitch 0.58 vs warp 0.38 -> stretch rows by 1.5
const W = (k1 - k0 + 1) * S, H = Math.round((m1 - m0 + 1) * S * 1.53);
const img = new Uint8Array(W * H * 3);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const k = k0 + Math.floor(x / S), m = m1 - Math.floor(y / (S * 1.53));
  const w = f[m * K + k]; const dp = w & 0xff, dw = (w >> 8) & 0xff;
  let c;
  if (dp === 255 && dw === 255) c = [0, 0, 0];
  else if (dw === 255) c = [235, 225, 200];
  else if (dp === 255) c = [90, 200, 230];
  else { const t = dw / 255; c = [Math.round(40 + 120 * t), Math.round(60 + 60 * t), Math.round(140 - 40 * t)]; }
  img.set(c, 3 * (y * W + x));
}
const raw = Buffer.alloc((W * 3 + 1) * H);
for (let y = 0; y < H; y++) { raw[y * (W * 3 + 1)] = 0; Buffer.from(img.buffer, y * W * 3, W * 3).copy(raw, y * (W * 3 + 1) + 1); }
const crcT = []; for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; crcT[n] = c >>> 0; }
const crc = (buf) => { let r = 0xffffffff; for (const v of buf) r = crcT[(r ^ v) & 255] ^ (r >>> 8); return (r ^ 0xffffffff) >>> 0; };
const chunk = (t, d) => { const len = Buffer.alloc(4); len.writeUInt32BE(d.length); const td = Buffer.concat([Buffer.from(t), d]); const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 2;
writeFileSync(out, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]));
console.log(out, { k0, k1, m0, m1, W, H });
