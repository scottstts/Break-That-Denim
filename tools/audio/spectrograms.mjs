// Side-by-side: for each (file, t0, dur): waveform strip (normalised to its own peak) and a
// spectrogram (log frequency 60 Hz - 20 kHz, dB re the panel's peak, 60 dB range), stacked.
//   node tools/audio/spectrograms.mjs out.png file@t0@dur ...
import { execFileSync } from 'node:child_process';
import { writeFileSync, unlinkSync } from 'node:fs';
import { load } from './metrics.mjs';
const [out, ...items] = process.argv.slice(2);
const SR = 48000, W = 1400, HW = 120, HS = 260, TOP = 20000;
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) { const a = (-2 * Math.PI) / len; for (let i = 0; i < n; i += len) for (let k = 0; k < len / 2; k++) { const c = Math.cos(a * k), s = Math.sin(a * k); const h = i + k + len / 2; const vr = re[h] * c - im[h] * s, vi = re[h] * s + im[h] * c; re[h] = re[i + k] - vr; im[h] = im[i + k] - vi; re[i + k] += vr; im[i + k] += vi; } }
}
const panels = [];
for (const it of items) {
  const [f, t0s, ds] = it.split('@');
  const x = load(f); const t0 = Number(t0s), dur = Number(ds);
  const s0 = Math.round(t0 * SR), N = Math.round(dur * SR);
  const img = new Uint8Array(W * (HW + HS) * 3);
  let pk = 0; for (let i = s0; i < s0 + N; i++) pk = Math.max(pk, Math.abs(x[i] ?? 0));
  for (let px = 0; px < W; px++) {
    const a = s0 + Math.floor(px * N / W), b = s0 + Math.floor((px + 1) * N / W);
    let lo = 0, hi = 0; for (let i = a; i < b; i++) { const v = (x[i] ?? 0) / pk; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (let y = Math.round((0.5 - hi * 0.48) * HW); y <= Math.round((0.5 - lo * 0.48) * HW); y++) { const o = (y * W + px) * 3; img[o] = 230; img[o + 1] = 230; img[o + 2] = 230; }
  }
  const FW = 1024; const cols = [];
  for (let px = 0; px < W; px++) {
    const c = s0 + Math.floor((px + 0.5) * N / W) - FW / 2;
    const re = new Float64Array(FW), im = new Float64Array(FW);
    for (let i = 0; i < FW; i++) re[i] = (x[c + i] ?? 0) * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FW - 1)));
    fft(re, im); const p = new Float64Array(FW / 2); for (let k = 0; k < FW / 2; k++) p[k] = re[k] ** 2 + im[k] ** 2; cols.push(p);
  }
  let mx = 0; for (const p of cols) for (const v of p) mx = Math.max(mx, v);
  for (let px = 0; px < W; px++) for (let y = 0; y < HS; y++) {
    const fq = 60 * (TOP / 60) ** (1 - y / (HS - 1)); const k = Math.min(FW / 2 - 1, Math.round(fq * FW / SR));
    const d = 10 * Math.log10(cols[px][k] / mx + 1e-12); const v = Math.max(0, Math.min(1, (d + 60) / 60));
    const o = ((HW + y) * W + px) * 3; img[o] = Math.round(255 * Math.min(1, v * 1.6)); img[o + 1] = Math.round(255 * Math.max(0, v * 1.6 - 0.6)); img[o + 2] = Math.round(255 * (v < 0.4 ? v * 1.5 : Math.max(0, 1.2 - v * 1.5)));
  }
  // frequency gridlines at 100, 1k, 10k
  for (const fq of [100, 1000, 10000]) { const y = Math.round((1 - Math.log(fq / 60) / Math.log(TOP / 60)) * (HS - 1)); for (let px = 0; px < W; px += 4) { const o = ((HW + y) * W + px) * 3; img[o] = img[o + 1] = img[o + 2] = 140; } }
  panels.push(img);
}
const H = HW + HS, all = new Uint8Array(W * (H + 6) * 3 * panels.length);
panels.forEach((p, i) => all.set(p, i * W * (H + 6) * 3));
writeFileSync(out + '.rgb', all);
execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', `${W}x${(H + 6) * panels.length}`, '-i', out + '.rgb', out]);
unlinkSync(out + '.rgb');
