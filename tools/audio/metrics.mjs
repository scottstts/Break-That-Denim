// Fine-structure metrics of a tear sound over spans, and their distance to the recording's
// (docs/decisions/2026-10-10-tear-sound-structure.md): 50 ms kurtosis / crest, envelope modulation
// spectrum and autocorrelation of the 0.8-8 kHz band, per-octave band kurtosis, cross-band coherence
// of the fine envelopes. The recording's own tears sit 1.8-2.8 from their pooled metrics.
//   node tools/audio/metrics.mjs file t0:t1[,t0:t1...]
// prints the metrics and the distance to refs/fabric_tear_sound_sample.mp3 (its three tears).
import { execFileSync } from 'node:child_process';
export function load(file) {
  const SR = 48000;
  const raw = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', file, '-ar', String(SR), '-ac', '1', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], { maxBuffer: 1 << 28 });
  return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
}
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) { const a = (-2 * Math.PI) / len; const wr = Math.cos(a), wi = Math.sin(a); for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) { const h = i + k + len / 2; const vr = re[h] * cr - im[h] * ci; const vi = re[h] * ci + im[h] * cr; re[h] = re[i + k] - vr; im[h] = im[i + k] - vi; re[i + k] += vr; im[i + k] += vi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } } }
}
export function metrics(x, spans) {
  const SR = 48000;
  const sel = spans.split(',').map((s) => s.split(':').map(Number));
  let n = 1; while (n < x.length) n <<= 1;
  const RE = new Float64Array(n), IM = new Float64Array(n);
  RE.set(x); fft(RE, IM);
  const bandSig = (lo, hi) => { const re = new Float64Array(n), im = new Float64Array(n); for (let k = 0; k < n; k++) { const f = Math.min(k, n - k) * SR / n; if (f >= lo && f < hi) { re[k] = RE[k]; im[k] = -IM[k]; } } fft(re, im); for (let i = 0; i < n; i++) re[i] /= n; return re; };
  // 50 ms kurtosis / crest of the full signal (loud windows: within 12 dB of the loudest)
  const W = 2400; const win = [];
  for (const [a, b] of sel) for (let s = Math.round(a * SR); s + W <= Math.round(b * SR); s += W) { let m2 = 0, m4 = 0, pk = 0; for (let i = s; i < s + W; i++) { const v = x[i]; m2 += v * v; m4 += v ** 4; pk = Math.max(pk, Math.abs(v)); } m2 /= W; m4 /= W; win.push([m2, m4 / (m2 * m2), 10 * Math.log10(pk * pk / m2)]); }
  const mx = Math.max(...win.map((w) => w[0]));
  const loud = win.filter((w) => w[0] > mx / 16);
  const med = (a) => { const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
  const kurt = med(loud.map((w) => w[1])), crest = med(loud.map((w) => w[2]));
  // modulation spectrum and autocorrelation of the 0.8-8 kHz envelope (0.25 ms hops)
  const y = bandSig(800, 8000); const H = 12, ER = SR / H;
  const env = new Float64Array(Math.floor(x.length / H));
  for (let f = 0; f < env.length; f++) { let s = 0; for (let i = f * H; i < (f + 1) * H; i++) s += y[i] * y[i]; env[f] = s; }
  const MW = 512; const acc = new Float64Array(MW / 2), ac = new Float64Array(41); let c = 0;
  for (const [a, b] of sel) for (let s = Math.round(a * ER); s + MW <= Math.round(b * ER); s += MW / 4) {
    let m = 0; for (let k = 0; k < MW; k++) m += env[s + k]; m /= MW;
    const r = new Float64Array(MW), i2 = new Float64Array(MW);
    for (let k = 0; k < MW; k++) r[k] = (env[s + k] / m - 1) * (0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (MW - 1)));
    for (let L = 0; L < ac.length; L++) { let q = 0; for (let k = 0; k + L < MW; k++) q += (env[s + k] / m - 1) * (env[s + k + L] / m - 1); ac[L] += q / (MW - L); }
    fft(r, i2); for (let k = 0; k < MW / 2; k++) acc[k] += (r[k] ** 2 + i2[k] ** 2) / MW; c++;
  }
  const mod = [[4, 16], [16, 32], [32, 64], [64, 128], [128, 256], [256, 512], [512, 1024], [1024, 2000]].map(([lo, hi]) => { let s = 0, k0 = 0; for (let k = 1; k < MW / 2; k++) { const f = k * ER / MW; if (f >= lo && f < hi) { s += acc[k]; k0++; } } return 10 * Math.log10(s / k0 / c); });
  const acf = [1, 2, 4, 8, 16, 20, 30].map((L) => ac[L] / ac[0]);
  // band kurtosis and cross-band envelope coherence (0.5 ms log envelopes, 20 ms detrend)
  const bands = [[200, 400], [400, 800], [800, 1600], [1600, 3200], [3200, 6400], [6400, 11000]];
  const HE = 24; const envs = []; const bk = [];
  for (const [lo, hi] of bands) {
    const z = bandSig(lo, hi);
    const ks = [];
    for (const [a, b] of sel) for (let s = Math.round(a * SR); s + W <= Math.round(b * SR); s += W) { let m2 = 0, m4 = 0; for (let i = s; i < s + W; i++) { m2 += z[i] ** 2; m4 += z[i] ** 4; } m2 /= W; m4 /= W; if (m2 > 0) ks.push(m4 / (m2 * m2)); }
    bk.push(med(ks));
    const e = new Float64Array(Math.floor(x.length / HE));
    for (let f = 0; f < e.length; f++) { let s = 0; for (let i = f * HE; i < (f + 1) * HE; i++) s += z[i] * z[i]; e[f] = Math.log(s + 1e-16); }
    const d = new Float64Array(e.length); const M = 20;
    const pre = new Float64Array(e.length + 1); for (let i = 0; i < e.length; i++) pre[i + 1] = pre[i] + e[i];
    for (let i = 0; i < e.length; i++) { const a = Math.max(0, i - M), b = Math.min(e.length, i + M); d[i] = e[i] - (pre[b] - pre[a]) / (b - a); }
    envs.push(d);
  }
  const corr = (A, B) => { let sa = 0, sb = 0, cc = 0; for (const [a, b] of sel) for (let i = Math.round(a * SR / HE); i < Math.round(b * SR / HE); i++) { sa += A[i]; sb += B[i]; cc++; } sa /= cc; sb /= cc; let ab = 0, aa = 0, bb = 0; for (const [a, b] of sel) for (let i = Math.round(a * SR / HE); i < Math.round(b * SR / HE); i++) { ab += (A[i] - sa) * (B[i] - sb); aa += (A[i] - sa) ** 2; bb += (B[i] - sb) ** 2; } return ab / Math.sqrt(aa * bb); };
  const coh = [[1, 2], [2, 3], [3, 4], [4, 5], [2, 4], [1, 4]].map(([i, j]) => corr(envs[i], envs[j]));
  return { kurt, crest, mod, acf, bk, coh };
}
export function distance(a, r) {
  let d = 0; const parts = {};
  const add = (k, v) => { parts[k] = v; d += v; };
  add('kc', Math.abs(Math.log(a.kurt / r.kurt)) * 2 + Math.abs(a.crest - r.crest) / 3);
  add('mod', a.mod.slice(1, 7).reduce((s, v, i) => s + Math.abs(v - r.mod[i + 1]), 0) / 6 / 2);
  add('acf', a.acf.reduce((s, v, i) => s + Math.abs(v - r.acf[i]), 0) / a.acf.length * 10);
  add('bk', a.bk.reduce((s, v, i) => s + Math.abs(Math.log(v / r.bk[i])), 0) / a.bk.length * 4);
  add('coh', a.coh.reduce((s, v, i) => s + Math.abs(v - r.coh[i]), 0) / a.coh.length * 10);
  return { d, parts };
}
if (process.argv[1].endsWith('metrics.mjs')) {
  const f = (a, d = 2) => a.map((v) => v.toFixed(d)).join(' ');
  const show = (m) => `K ${m.kurt.toFixed(1)} crest ${m.crest.toFixed(1)} | mod ${f(m.mod, 1)} | acf ${f(m.acf)} | band K ${f(m.bk, 1)} | coh ${f(m.coh)}`;
  const ref = metrics(load(new URL('../../refs/fabric_tear_sound_sample.mp3', import.meta.url).pathname), '0.6:1.25,2.3:2.75,3.85:5.0');
  const m = metrics(load(process.argv[2]), process.argv[3]);
  const r = distance(m, ref);
  console.log(`recording  ${show(ref)}\nthis       ${show(m)}\ndistance ${r.d.toFixed(2)} (${Object.entries(r.parts).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')})`);
}
