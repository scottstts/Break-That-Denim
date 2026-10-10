// Side-by-side waveforms of two recordings (the tear voice against refs/fabric_tear_sound_sample.mp3):
// for each time scale (s), a strip of A (cyan) above a strip of B (orange), starting at tA and tB,
// each normalised to its peak over the following second. Grid lines every tenth of the strip.
//   node tools/audio/compare.mjs A tA B tB out.png [scales, default 1,0.2,0.04,0.01]
import { execFileSync } from 'node:child_process';
import { unlinkSync, writeFileSync } from 'node:fs';
const [fa, ta, fb, tb, out, sc] = process.argv.slice(2);
const SR = 48000;
const load = (f) => { const raw = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', f, '-ar', String(SR), '-ac', '1', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], { maxBuffer: 1 << 28 }); return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4); };
const A = load(fa), B = load(fb);
const scales = (sc ?? '1,0.2,0.04,0.01').split(',').map(Number);
const W = 1600, Hs = 170;
const rows = [];
const norm = (x, t0) => { let p = 0; for (let i = Math.round(t0 * SR); i < Math.round((t0 + 1) * SR) && i < x.length; i++) p = Math.max(p, Math.abs(x[i])); return p || 1; };
const na = norm(A, Number(ta)), nb = norm(B, Number(tb));
for (const s of scales) for (const [x, t0, n, col] of [[A, Number(ta), na, [120, 220, 255]], [B, Number(tb), nb, [255, 170, 80]]]) {
  const img = new Uint8Array(W * Hs * 3); const s0 = Math.round(t0 * SR), win = Math.round(s * SR);
  for (let px = 0; px < W; px++) {
    const a = s0 + Math.floor((px * win) / W), b = Math.max(a, s0 + Math.floor(((px + 1) * win) / W) - 1);
    let lo = 1, hi = -1; for (let i = a; i <= b; i++) { const v = (x[i] ?? 0) / n; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const y0 = Math.max(0, Math.round((0.5 - hi * 0.48) * Hs)), y1 = Math.min(Hs - 1, Math.round((0.5 - lo * 0.48) * Hs));
    for (let y = y0; y <= y1; y++) { const o = (y * W + px) * 3; img[o] = col[0]; img[o + 1] = col[1]; img[o + 2] = col[2]; }
  }
  for (let px = 0; px < W; px += 160) for (let y = 0; y < Hs; y += 5) { const o = (y * W + px) * 3; if (!img[o]) img[o] = img[o + 1] = img[o + 2] = 60; }
  for (let px = 0; px < W; px++) { const o = ((Hs - 1) * W + px) * 3; img[o] = img[o + 1] = img[o + 2] = 110; }
  rows.push(img);
}
const all = new Uint8Array(W * Hs * 3 * rows.length); rows.forEach((r, k) => all.set(r, k * W * Hs * 3));
writeFileSync(`${out}.rgb`, all);
execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${Hs * rows.length}`, '-i', `${out}.rgb`, out]);
unlinkSync(`${out}.rgb`);
