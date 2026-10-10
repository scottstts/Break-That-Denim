// Sound-texture statistics of the auditory periphery (McDermott & Simoncelli 2011, "Sound texture
// perception via statistics of the auditory periphery"): sounds that share them are heard as the same
// texture. 30 ERB-spaced cochlear bands (half-cosine, 50 Hz - 11 kHz: the recording's range), compressed
// envelopes (^0.3) at 2.4 kHz; per band: mean (spectrum), std/mean, skewness, kurtosis; correlation of
// envelopes across bands; envelope modulation power per octave band (2-512 Hz, re the band's variance);
// correlation of each modulation band between neighbouring cochlear bands (C1).
// Fine texture (`fine`): the same moments of each envelope over its own 50 ms moving mean, so the
// hand's slow loudness swings (2-8 Hz) do not count; modulation 16-512 Hz; C1 16-256 Hz.
//   node tools/audio/texture.mjs a.wav t0:t1[,..] [b.wav t0:t1[,..]] ...   (first = reference)
import { load } from './metrics.mjs';

const SR = 48000;
const NB = 30;
const LO = 50;
const HI = 11000;
const DS = 20; // envelope at 2.4 kHz
const ER = SR / DS;
const MODS = [2, 4, 8, 16, 32, 64, 128, 256, 512];

function fft(re, im, inv = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const a = ((inv ? 2 : -2) * Math.PI) / len;
    const wr = Math.cos(a), wi = Math.sin(a);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const h = i + k + len / 2;
        const vr = re[h] * cr - im[h] * ci, vi = re[h] * ci + im[h] * cr;
        re[h] = re[i + k] - vr; im[h] = im[i + k] - vi; re[i + k] += vr; im[i + k] += vi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}
const erb = (f) => 9.265 * Math.log(1 + f / (24.7 * 9.265));
const erbInv = (e) => 24.7 * 9.265 * (Math.exp(e / 9.265) - 1);

/** Envelopes (compressed, downsampled) of the cochlear bands, and the band centres. */
function cochlea(x) {
  let n = 1; while (n < x.length) n <<= 1;
  const RE = new Float64Array(n), IM = new Float64Array(n);
  RE.set(x); fft(RE, IM);
  const e0 = erb(LO), e1 = erb(HI), step = (e1 - e0) / (NB + 1);
  const envs = [], centres = [];
  for (let b = 1; b <= NB; b++) {
    const lo = erbInv(e0 + (b - 1) * step), hi = erbInv(e0 + (b + 1) * step), c = erbInv(e0 + b * step);
    centres.push(c);
    const re = new Float64Array(n), im = new Float64Array(n);
    // analytic signal of the band: positive frequencies only, half-cosine on the ERB scale
    for (let k = 1; k < n / 2; k++) {
      const f = (k * SR) / n;
      if (f <= lo || f >= hi) continue;
      const w = Math.cos((Math.PI * (erb(f) - (e0 + b * step))) / (2 * step));
      re[k] = 2 * RE[k] * w; im[k] = 2 * IM[k] * w;
    }
    fft(re, im, true);
    const env = new Float64Array(Math.floor(x.length / DS));
    for (let i = 0; i < env.length; i++) { let s = 0; for (let j = i * DS; j < (i + 1) * DS; j++) s += Math.sqrt(re[j] * re[j] + im[j] * im[j]); env[i] = (s / DS) ** 0.3; }
    envs.push(env);
  }
  return { envs, centres };
}

export function stats(x, spans) {
  const { envs, centres } = cochlea(x);
  const sel = spans.split(',').map((s) => s.split(':').map((v) => Math.round(Number(v) * ER)));
  const pick = (e) => { const out = []; for (const [a, b] of sel) for (let i = a; i < b && i < e.length; i++) out.push(e[i]); return Float64Array.from(out); };
  const E = envs.map(pick);
  const mom = E.map((e) => {
    const n = e.length; let m = 0; for (const v of e) m += v; m /= n;
    let m2 = 0, m3 = 0, m4 = 0; for (const v of e) { const d = v - m; m2 += d * d; m3 += d * d * d; m4 += d ** 4; }
    m2 /= n; m3 /= n; m4 /= n;
    return { mean: m, cv: Math.sqrt(m2) / m, skew: m3 / m2 ** 1.5, kurt: m4 / (m2 * m2) };
  });
  const corr = (a, b) => { let sa = 0, sb = 0; for (let i = 0; i < a.length; i++) { sa += a[i]; sb += b[i]; } sa /= a.length; sb /= b.length; let ab = 0, aa = 0, bb = 0; for (let i = 0; i < a.length; i++) { ab += (a[i] - sa) * (b[i] - sb); aa += (a[i] - sa) ** 2; bb += (b[i] - sb) ** 2; } return ab / Math.sqrt(aa * bb); };
  // modulation bands per cochlear band: per span, FFT of the (mean-removed) envelope, octave bands
  const modPow = [], modSig = [];
  for (let b = 0; b < NB; b++) {
    const P = new Float64Array(MODS.length); let v = 0;
    const sigs = MODS.map(() => []);
    for (const [a, c] of sel) {
      const seg = envs[b].subarray(a, Math.min(c, envs[b].length));
      let n = 1; while (n < seg.length) n <<= 1;
      const re = new Float64Array(n), im = new Float64Array(n);
      let m = 0; for (const s of seg) m += s; m /= seg.length;
      for (let i = 0; i < seg.length; i++) { re[i] = seg[i] - m; v += re[i] ** 2; }
      fft(re, im);
      MODS.forEach((fc, q) => {
        const r2 = new Float64Array(n), i2 = new Float64Array(n);
        for (let k = 1; k < n / 2; k++) {
          const f = (k * ER) / n;
          const w = f > fc / Math.SQRT2 && f < fc * Math.SQRT2 ? Math.cos(Math.PI * Math.log2(f / fc)) ** 2 : 0;
          P[q] += w * (re[k] ** 2 + im[k] ** 2) * 2 / n;
          r2[k] = re[k] * w; i2[k] = im[k] * w; r2[n - k] = re[k] * w; i2[n - k] = -im[k] * w;
        }
        fft(r2, i2, true);
        for (let i = 0; i < seg.length; i++) sigs[q].push(r2[i]);
      });
    }
    modPow.push(Array.from(P, (p) => p / v));
    modSig.push(sigs.map((s) => Float64Array.from(s)));
  }
  // fine moments: each envelope over its own 50 ms moving mean
  const M50 = Math.round(0.025 * ER);
  const fine = envs.map((env) => {
    const pre = new Float64Array(env.length + 1);
    for (let i = 0; i < env.length; i++) pre[i + 1] = pre[i] + env[i];
    const r = new Float64Array(env.length);
    for (let i = 0; i < env.length; i++) { const a = Math.max(0, i - M50), b = Math.min(env.length, i + M50); r[i] = env[i] / ((pre[b] - pre[a]) / (b - a)); }
    const e = pick(r);
    const n = e.length; let m = 0; for (const v of e) m += v; m /= n;
    let m2 = 0, m3 = 0, m4 = 0; for (const v of e) { const d = v - m; m2 += d * d; m3 += d * d * d; m4 += d ** 4; }
    m2 /= n; m3 /= n; m4 /= n;
    return { cv: Math.sqrt(m2) / m, skew: m3 / m2 ** 1.5, kurt: m4 / (m2 * m2) };
  });
  const C = [1, 2, 4, 8].map((d) => { const r = []; for (let b = 0; b + d < NB; b++) r.push(corr(E[b], E[b + d])); return r; });
  const C1 = MODS.map((_, q) => { const r = []; for (let b = 0; b + 1 < NB; b++) r.push(corr(modSig[b][q], modSig[b + 1][q])); return r; });
  return { centres, mom, fine, C, modPow, C1 };
}

const REGIONS = [[50, 400], [400, 1500], [1500, 4000], [4000, 11000]];
const regionAvg = (centres, vals, [lo, hi]) => { let s = 0, n = 0; vals.forEach((v, b) => { if (centres[b] >= lo && centres[b] < hi && Number.isFinite(v)) { s += v; n++; } }); return s / n; };

/** Distance of fine texture statistics (b against reference a), per class. */
export function fineDistance(b, a) {
  const parts = { cv: 0, skew: 0, kurt: 0, mod: 0, c1: 0 };
  for (const r of REGIONS) {
    const d = (fn) => Math.abs(regionAvg(b.centres, fn(b), r) - regionAvg(a.centres, fn(a), r));
    parts.cv += 10 * d((s) => s.fine.map((q) => q.cv));
    parts.skew += 2 * d((s) => s.fine.map((q) => q.skew));
    parts.kurt += d((s) => s.fine.map((q) => q.kurt));
    // the shape of the fine modulation spectrum (16-512 Hz, re its own total: slow swings excluded)
    const fineMod = (st, q) => st.modPow.map((p) => { let t = 0; MODS.forEach((fc, j) => { if (fc >= 16) t += p[j]; }); return 10 * Math.log10(p[q] / t); });
    MODS.forEach((fc, q) => { if (fc >= 16) parts.mod += d((st) => fineMod(st, q)) / 6; });
    MODS.forEach((fc, q) => { if (fc >= 16 && fc <= 256) parts.c1 += 2 * d((s) => s.C1[q]); });
  }
  let total = 0;
  for (const k of Object.keys(parts)) { parts[k] /= REGIONS.length; total += parts[k]; }
  return { total, parts };
}

const regions = [[50, 400], [400, 1500], [1500, 4000], [4000, 11000]];
const avg = (centres, vals, [lo, hi]) => { let s = 0, n = 0; vals.forEach((v, b) => { if (centres[b] >= lo && centres[b] < hi && Number.isFinite(v)) { s += v; n++; } }); return s / n; };
if (!process.argv[1].endsWith('texture.mjs')) { /* imported */ } else main();
function main() {
const args = process.argv.slice(2);
const runs = [];
for (let i = 0; i < args.length; i += 2) runs.push([args[i].split('/').pop(), stats(load(args[i]), args[i + 1])]);
const row = (label, fn, d = 2) => console.log(label.padEnd(26) + runs.map(([, s]) => regions.map((r) => fn(s, r).toFixed(d).padStart(6)).join('') + '  |').join(''));
console.log(''.padEnd(26) + runs.map(([n]) => n.slice(0, 22).padEnd(24) + '  |').join(''));
console.log('region (Hz)'.padEnd(26) + runs.map(() => regions.map((r) => String(r[0]).padStart(6)).join('') + '  |').join(''));
row('spectrum dB re max', (s, r) => { const m = Math.max(...s.mom.map((q) => q.mean)); return avg(s.centres, s.mom.map((q) => 20 / 0.3 * Math.log10(q.mean / m)), r); }, 1);
row('env std/mean', (s, r) => avg(s.centres, s.mom.map((q) => q.cv), r));
row('env skewness', (s, r) => avg(s.centres, s.mom.map((q) => q.skew), r));
row('env kurtosis', (s, r) => avg(s.centres, s.mom.map((q) => q.kurt), r));
row('fine env std/mean', (s, r) => avg(s.centres, s.fine.map((q) => q.cv), r));
row('fine env skewness', (s, r) => avg(s.centres, s.fine.map((q) => q.skew), r));
row('fine env kurtosis', (s, r) => avg(s.centres, s.fine.map((q) => q.kurt), r));
[1, 2, 4, 8].forEach((d, i) => row(`env corr, ${d} band${d > 1 ? 's' : ''} apart`, (s, r) => avg(s.centres, s.C[i], r)));
MODS.forEach((fc, q) => row(`mod power ${fc} Hz (dB)`, (s, r) => avg(s.centres, s.modPow.map((p) => 10 * Math.log10(p[q])), r), 1));
MODS.forEach((fc, q) => { if (fc >= 8) row(`mod corr C1 ${fc} Hz`, (s, r) => avg(s.centres, s.C1[q], r)); });
for (const [name, st] of runs.slice(1)) { const d = fineDistance(st, runs[0][1]); console.log(`fine-texture distance ${name}: ${d.total.toFixed(2)} (${Object.entries(d.parts).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(', ')})`); }
}
