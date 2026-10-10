// Acoustic analysis of a recorded fabric tear (refs/fabric_tear_sound_sample.mp3): the measurements
// the procedural tear voice (src/audio/) is grounded in. Prints envelope segments, grain (crackle)
// statistics, the long-term spectrum of active vs quiet passages, loudness/brightness coupling and the
// mean grain shape.
//
//   node tools/audio/analyze.mjs [file]
import { execFileSync } from 'node:child_process';

const file = process.argv[2] ?? 'refs/fabric_tear_sound_sample.mp3';
const SR = 48000;
const raw = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', file, '-ar', String(SR), '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], {
  maxBuffer: 1 << 28,
});
const st = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
const N = st.length >> 1;
const x = new Float32Array(N);
let sLL = 0;
let sRR = 0;
let sLR = 0;
for (let i = 0; i < N; i++) {
  const l = st[2 * i];
  const r = st[2 * i + 1];
  x[i] = 0.5 * (l + r);
  sLL += l * l;
  sRR += r * r;
  sLR += l * r;
}
const db = (v) => 10 * Math.log10(Math.max(v, 1e-20));
const fmt = (v, d = 1) => v.toFixed(d);
console.log(`${file}: ${fmt(N / SR, 2)} s, stereo correlation ${fmt(sLR / Math.sqrt(sLL * sRR), 3)}`);

// ---------------------------------------------------------------- FFT
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const a = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const c = Math.cos(a * k);
        const s = Math.sin(a * k);
        const ur = re[i + k];
        const ui = im[i + k];
        const vr = re[i + k + len / 2] * c - im[i + k + len / 2] * s;
        const vi = re[i + k + len / 2] * s + im[i + k + len / 2] * c;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr;
        im[i + k + len / 2] = ui - vi;
      }
    }
  }
}
function powerSpectrum(sig, start, n) {
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = (sig[start + i] ?? 0) * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
  fft(re, im);
  const p = new Float64Array(n / 2);
  for (let k = 0; k < n / 2; k++) p[k] = re[k] * re[k] + im[k] * im[k];
  return p;
}

// ---------------------------------------------------------------- envelope and segments
const HOP = Math.round(0.005 * SR);
const frames = Math.floor(N / HOP);
const env = new Float64Array(frames);
for (let f = 0; f < frames; f++) {
  let s = 0;
  for (let i = f * HOP; i < (f + 1) * HOP; i++) s += x[i] * x[i];
  env[f] = s / HOP;
}
const peak = Math.max(...env);
const envDb = Array.from(env, (e) => db(e / peak));
const segs = [];
const ACTIVE = -30;
for (let f = 0; f < frames; ) {
  if (envDb[f] > ACTIVE) {
    let g = f;
    // join gaps shorter than 60 ms
    while (g < frames && (envDb[g] > ACTIVE || envDb.slice(g, g + 12).some((v) => v > ACTIVE))) g++;
    segs.push([f, g]);
    f = g;
  } else f++;
}
console.log(`\nActive segments (> ${ACTIVE} dB re peak 5 ms RMS):`);
for (const [a, b] of segs) {
  const e = envDb.slice(a, b);
  const pk = Math.max(...e);
  const ipk = e.indexOf(pk);
  // attack: from -20 dB below the segment peak to within 3 dB of it
  const smooth = e.map((_, i) => db(e.slice(Math.max(0, i - 4), i + 5).reduce((s, v) => s + 10 ** (v / 10), 0) / 9));
  const spk = Math.max(...smooth);
  const firstNear = smooth.findIndex((v) => v > spk - 3);
  console.log(
    `  ${fmt((a * HOP) / SR, 3)}-${fmt((b * HOP) / SR, 3)} s (${fmt(((b - a) * HOP) / SR, 2)} s): peak ${fmt(pk)} dB at +${fmt((ipk * HOP) / SR, 3)} s, reaches -3 dB of smoothed peak at +${fmt((firstNear * HOP) / SR, 3)} s, median ${fmt([...e].sort((p, q) => p - q)[e.length >> 1])} dB`,
  );
}
// the quiet passages leading into each loud tear (onset ramp)
console.log('\nEnvelope 20 ms means, dB re peak (one row per segment):');
for (const [a, b] of segs) {
  const row = [];
  for (let f = Math.max(0, a - 20); f < Math.min(frames, b + 20); f += 4) row.push(fmt(db(env.slice(f, f + 4).reduce((s, v) => s + v, 0) / 4 / peak), 0));
  console.log('  ' + row.join(' '));
}

// ---------------------------------------------------------------- long-term spectra
function ltas(ranges) {
  const n = 2048;
  const acc = new Float64Array(n / 2);
  let count = 0;
  for (const [a, b] of ranges) {
    for (let s = a; s + n <= b; s += n / 2) {
      const p = powerSpectrum(x, s, n);
      for (let k = 0; k < n / 2; k++) acc[k] += p[k];
      count++;
    }
  }
  for (let k = 0; k < n / 2; k++) acc[k] /= Math.max(1, count);
  return { acc, n };
}
const loud = [];
const quiet = [];
for (let f = 0; f < frames; f++) {
  const r = [f * HOP, (f + 1) * HOP];
  if (envDb[f] > -12) loud.push(r);
  else if (envDb[f] > -45 && envDb[f] < -25) quiet.push(r);
}
const merge = (rs) => {
  const out = [];
  for (const r of rs) {
    if (out.length && out[out.length - 1][1] === r[0]) out[out.length - 1][1] = r[1];
    else out.push([...r]);
  }
  return out;
};
const L = ltas(merge(loud));
const Q = ltas(merge(quiet));
const bands = [];
for (let fc = 63; fc <= 16000; fc *= 2 ** (1 / 3)) bands.push(fc);
const band = (S, fc) => {
  let s = 0;
  const lo = fc / 2 ** (1 / 6);
  const hi = fc * 2 ** (1 / 6);
  for (let k = 1; k < S.n / 2; k++) {
    const f = (k * SR) / S.n;
    if (f >= lo && f < hi) s += S.acc[k];
  }
  return s;
};
const lRef = Math.max(...bands.map((fc) => band(L, fc)));
console.log('\nThird-octave band power, dB re loudest band (loud passages | quiet passages, each re its own max):');
const qRef = Math.max(...bands.map((fc) => band(Q, fc)));
for (const fc of bands) console.log(`  ${fmt(fc, 0).padStart(6)} Hz  ${fmt(db(band(L, fc) / lRef)).padStart(6)}  ${fmt(db(band(Q, fc) / qRef)).padStart(6)}`);
const centroid = (S, lo = 100, hi = 11000) => {
  let a = 0;
  let w = 0;
  for (let k = 1; k < S.n / 2; k++) {
    const f = (k * SR) / S.n;
    if (f < lo || f > hi) continue;
    a += f * S.acc[k];
    w += S.acc[k];
  }
  return a / w;
};
console.log(`  centroid: loud ${fmt(centroid(L), 0)} Hz, quiet ${fmt(centroid(Q), 0)} Hz`);

// ---------------------------------------------------------------- loudness vs brightness
{
  const n = 512;
  const pts = [];
  for (let s = 0; s + n <= N; s += n) {
    const p = powerSpectrum(x, s, n);
    let a = 0;
    let w = 0;
    for (let k = 2; k < n / 2; k++) {
      const f = (k * SR) / n;
      if (f > 11000) break;
      a += f * p[k];
      w += p[k];
    }
    const e = db(env[Math.floor(s / HOP)] / peak);
    if (e > -40) pts.push([e, a / w]);
  }
  const bins = [-40, -30, -20, -12, -6, 0];
  console.log('\nSpectral centroid (10.7 ms frames) by level:');
  for (let b = 0; b < bins.length - 1; b++) {
    const sel = pts.filter((p) => p[0] >= bins[b] && p[0] < bins[b + 1]).map((p) => p[1]);
    if (sel.length) console.log(`  ${bins[b]}..${bins[b + 1]} dB: ${fmt(sel.reduce((s, v) => s + v, 0) / sel.length, 0)} Hz (${sel.length} frames)`);
  }
}

// ---------------------------------------------------------------- grains (crackle events)
{
  // pre-emphasis isolates the impulsive component; grains are local maxima of the 0.25 ms energy that
  // stand well above the 20 ms running median of the same signal
  const y = new Float32Array(N);
  for (let i = 1; i < N; i++) y[i] = x[i] - 0.95 * x[i - 1];
  const W = Math.round(0.00025 * SR);
  const e = new Float64Array(Math.floor(N / W));
  for (let k = 0; k < e.length; k++) {
    let s = 0;
    for (let i = k * W; i < (k + 1) * W; i++) s += y[i] * y[i];
    e[k] = s / W;
  }
  const M = Math.round(0.02 / 0.00025);
  const onsets = [];
  for (let k = M; k < e.length - M; k++) {
    if (e[k] < e[k - 1] || e[k] < e[k + 1]) continue;
    const win = Array.from(e.slice(k - M, k + M)).sort((a, b) => a - b);
    const med = win[win.length >> 1];
    if (e[k] > 6 * med && db(env[Math.floor((k * W) / HOP)] / peak) > -30) onsets.push({ k, t: (k * W) / SR, a: e[k], r: e[k] / med });
  }
  // refractory 1 ms
  const g = [];
  for (const o of onsets) {
    if (g.length && o.t - g[g.length - 1].t < 0.001) {
      if (o.a > g[g.length - 1].a) g[g.length - 1] = o;
    } else g.push(o);
  }
  console.log(`\nGrains (pre-emphasised 0.25 ms energy > 6x 40 ms median): ${g.length}`);
  for (const [a, b] of segs) {
    const t0 = (a * HOP) / SR;
    const t1 = (b * HOP) / SR;
    const sel = g.filter((o) => o.t >= t0 && o.t < t1);
    if (sel.length < 3) continue;
    const ioi = sel.slice(1).map((o, i) => o.t - sel[i].t);
    const mean = ioi.reduce((s, v) => s + v, 0) / ioi.length;
    const sd = Math.sqrt(ioi.reduce((s, v) => s + (v - mean) ** 2, 0) / ioi.length);
    console.log(`  ${fmt(t0, 2)}-${fmt(t1, 2)} s: ${sel.length} grains, ${fmt(sel.length / (t1 - t0), 0)}/s, IOI mean ${fmt(mean * 1000, 2)} ms, CV ${fmt(sd / mean, 2)}`);
  }
  // amplitude distribution: CCDF slope of grain peak amplitude (log-log), above the detection floor
  const amps = g.map((o) => Math.sqrt(o.a)).sort((a, b) => b - a);
  const pts = [];
  for (let i = 0; i < amps.length; i += Math.max(1, Math.floor(amps.length / 40))) pts.push([Math.log(amps[i]), Math.log((i + 1) / amps.length)]);
  const top = pts.filter((_, i) => i < pts.length * 0.8);
  const mx = top.reduce((s, p) => s + p[0], 0) / top.length;
  const my = top.reduce((s, p) => s + p[1], 0) / top.length;
  const slope = top.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0) / top.reduce((s, p) => s + (p[0] - mx) ** 2, 0);
  console.log(`  amplitude CCDF slope (power-law exponent of P(A > a)): ${fmt(slope, 2)}; dynamic range p95/p50 ${fmt(20 * Math.log10(amps[Math.floor(amps.length * 0.05)] / amps[amps.length >> 1]))} dB`);

  // mean grain shape: energy envelope aligned on strong grains, and the grain spectrum
  const strong = g.filter((o) => o.r > 20);
  const pre = Math.round(0.001 * SR);
  const post = Math.round(0.008 * SR);
  const shape = new Float64Array(pre + post);
  for (const o of strong) {
    const c = o.k * W;
    let m = 0;
    for (let i = c - W; i < c + 2 * W; i++) if (Math.abs(y[i]) > Math.abs(y[c + m - W] ?? 0)) m = i - c + W;
    const c0 = c + m - W;
    for (let i = -pre; i < post; i++) shape[i + pre] += y[c0 + i] ** 2;
  }
  const sm = (i) => shape.slice(Math.max(0, i - 4), i + 5).reduce((s, v) => s + v, 0) / 9;
  const s0 = sm(pre);
  const t20 = Array.from(shape.keys()).find((i) => i > pre && sm(i) < s0 * 0.01);
  const t10 = Array.from(shape.keys()).find((i) => i > pre && sm(i) < s0 * 0.1);
  console.log(`  mean strong-grain envelope (${strong.length} grains): -10 dB after ${fmt((((t10 ?? pre) - pre) / SR) * 1000, 2)} ms, -20 dB after ${fmt((((t20 ?? pre) - pre) / SR) * 1000, 2)} ms`);
  const n = 256;
  const acc = new Float64Array(n / 2);
  for (const o of strong) {
    const p = powerSpectrum(x, o.k * W - 32, n);
    for (let k = 0; k < n / 2; k++) acc[k] += p[k];
  }
  let a = 0;
  let w = 0;
  let best = 0;
  for (let k = 1; k < n / 2; k++) {
    const f = (k * SR) / n;
    if (f > 11000) break;
    a += f * acc[k];
    w += acc[k];
    if (acc[k] > acc[best]) best = k;
  }
  console.log(`  grain spectrum (5.3 ms windows on strong grains): centroid ${fmt(a / w, 0)} Hz, peak ${fmt((best * SR) / n, 0)} Hz`);
}

// ---------------------------------------------------------------- impulsiveness
{
  // per 50 ms window over the loud passages (within 10 dB of the loudest window): median kurtosis and
  // crest, which separate a dense crackle (recording: 5.1, 14.3 dB) from Gaussian hiss (3, ~11 dB) and
  // from sparse clicks (far higher)
  const W50 = Math.round(0.05 * SR);
  const rows = [];
  for (let s = 0; s + W50 <= N; s += W50) {
    let m2 = 0, m4 = 0, pk = 0;
    for (let i = s; i < s + W50; i++) { m2 += x[i] ** 2; m4 += x[i] ** 4; pk = Math.max(pk, Math.abs(x[i])); }
    rows.push({ rms: Math.sqrt(m2 / W50), k: m4 / W50 / (m2 / W50) ** 2, c: pk / Math.sqrt(m2 / W50) });
  }
  const top = rows.reduce((m, r) => Math.max(m, r.rms), 0);
  const loudW = rows.filter((r) => r.rms > top * 0.3);
  const med = (a) => a.sort((p, q) => p - q)[a.length >> 1];
  console.log(`\nLoud 50 ms windows (${loudW.length}): median kurtosis ${fmt(med(loudW.map((r) => r.k)))}, crest ${fmt(20 * Math.log10(med(loudW.map((r) => r.c))))} dB`);
  for (const [a, b] of segs) {
    let s2 = 0;
    let s4 = 0;
    let mx = 0;
    const n = (b - a) * HOP;
    for (let i = a * HOP; i < b * HOP; i++) {
      s2 += x[i] * x[i];
      s4 += x[i] ** 4;
      mx = Math.max(mx, Math.abs(x[i]));
    }
    const rms = Math.sqrt(s2 / n);
    console.log(`  segment ${fmt((a * HOP) / SR, 2)} s: kurtosis ${fmt(s4 / n / (s2 / n) ** 2, 1)}, crest ${fmt(20 * Math.log10(mx / rms))} dB, RMS ${fmt(20 * Math.log10(rms))} dBFS`);
  }
}

// ---------------------------------------------------------------- amplitude modulation spectrum
{
  // fluctuation of the 1 ms envelope inside loud passages: which rates carry the "rip" texture
  const H1 = Math.round(0.001 * SR);
  const e1 = [];
  for (let i = 0; i + H1 <= N; i += H1) {
    let s = 0;
    for (let j = i; j < i + H1; j++) s += x[j] * x[j];
    e1.push(Math.sqrt(s / H1));
  }
  const n = 256;
  const acc = new Float64Array(n / 2);
  let c = 0;
  for (const [a, b] of merge(loud)) {
    const fa = Math.floor(a / H1);
    const fb = Math.floor(b / H1);
    for (let s = fa; s + n <= fb; s += n / 2) {
      const seg = e1.slice(s, s + n);
      const m = seg.reduce((p, v) => p + v, 0) / n;
      const p = powerSpectrum(seg.map((v) => v - m), 0, n);
      for (let k = 0; k < n / 2; k++) acc[k] += p[k];
      c++;
    }
  }
  if (c) {
    const out = [];
    for (const r of [[2, 8], [8, 16], [16, 32], [32, 64], [64, 128], [128, 256], [256, 500]]) {
      let s = 0;
      for (let k = 1; k < n / 2; k++) {
        const f = (k * 1000) / n;
        if (f >= r[0] && f < r[1]) s += acc[k];
      }
      out.push(`${r[0]}-${r[1]} Hz ${fmt(db(s))}`);
    }
    console.log(`\nEnvelope modulation power in loud passages (${c} windows): ${out.join(', ')}`);
  }
}
