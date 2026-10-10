// Event-level comparison of tear sounds: detects the events of each recording (300 Hz - 11 kHz, the
// recording's range) and prints their distributions: prominence over the local level, amplitude
// spread, spacing, duration, and how tonal each event is (spectral flatness and peak share).
//   node tools/audio/events.mjs a.wav t0:t1[,..] [b.wav t0:t1[,..]] ...
import { load } from './metrics.mjs';

const SR = 48000;
function fft(re, im, inv = false) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let b = n >> 1; for (; j & b; b >>= 1) j ^= b; j ^= b; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const a = ((inv ? 2 : -2) * Math.PI) / len, wr = Math.cos(a), wi = Math.sin(a);
    for (let i = 0; i < n; i += len) { let cr = 1, ci = 0; for (let k = 0; k < len / 2; k++) { const h = i + k + len / 2; const vr = re[h] * cr - im[h] * ci, vi = re[h] * ci + im[h] * cr; re[h] = re[i + k] - vr; im[h] = im[i + k] - vi; re[i + k] += vr; im[i + k] += vi; const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t; } }
  }
  if (inv) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

export function events(x0, spans) {
  let n = 1; while (n < x0.length) n <<= 1;
  const re = new Float64Array(n), im = new Float64Array(n);
  re.set(x0); fft(re, im);
  for (let k = 0; k < n; k++) { const f = (Math.min(k, n - k) * SR) / n; if (f < 300 || f > 11000) { re[k] = 0; im[k] = 0; } }
  // analytic band signal: its magnitude is the envelope
  for (let k = n / 2 + 1; k < n; k++) { re[k] = 0; im[k] = 0; }
  for (let k = 1; k < n / 2; k++) { re[k] *= 2; im[k] *= 2; }
  fft(re, im, true);
  const x = Float64Array.from({ length: x0.length }, (_, i) => re[i]);
  const env = Float64Array.from({ length: x0.length }, (_, i) => Math.hypot(re[i], im[i]));
  // smooth 0.25 ms
  const H = 6; const sm = new Float64Array(env.length);
  { let s = 0; for (let i = 0; i < env.length; i++) { s += env[i] - (i >= 2 * H ? env[i - 2 * H] : 0); sm[i] = s / (2 * H); } }
  const out = { prom: [], amp: [], ioi: [], dur: [], flat: [], peakShare: [] };
  for (const sp of spans.split(',')) {
    const [a, b] = sp.split(':').map((v) => Math.round(Number(v) * SR));
    // local level: rms of the band signal over 20 ms
    const W = 480;
    let last = -1;
    const peaks = [];
    for (let i = a + W; i < b - W; i++) {
      if (sm[i] < sm[i - 1] || sm[i] < sm[i + 1]) continue;
      // a peak is an event if it is the largest within +-1 ms
      let ok = true; for (let j = i - 48; j <= i + 48 && ok; j++) if (sm[j] > sm[i]) ok = false;
      if (!ok) continue;
      let e = 0; for (let j = i - W; j < i + W; j += 4) e += x[j] * x[j]; const rms = Math.sqrt(e / (W / 2));
      peaks.push([i, sm[i] / rms]);
    }
    for (const [i, p] of peaks) {
      out.prom.push(20 * Math.log10(p));
      if (p > 2) {
        out.amp.push(sm[i]);
        if (last >= 0) out.ioi.push((i - last) / SR * 1000);
        last = i;
        // duration above half the peak (smoothed envelope)
        let l = i, r = i; while (l > a && sm[l] > sm[i] / 2) l--; while (r < b && sm[r] > sm[i] / 2) r++;
        out.dur.push((r - l) / SR * 1000);
        // tonality of the event: 2.7 ms window spectrum, flatness and the share of its strongest bin pair
        const N = 128; const wr = new Float64Array(N), wi = new Float64Array(N);
        for (let k = 0; k < N; k++) wr[k] = x[i - N / 2 + k] * (0.5 - 0.5 * Math.cos((2 * Math.PI * k) / (N - 1)));
        fft(wr, wi);
        const P = []; for (let k = 2; k < 30; k++) P.push(wr[k] ** 2 + wi[k] ** 2); // 750 Hz - 11 kHz
        const tot = P.reduce((s, v) => s + v, 0);
        const geo = Math.exp(P.reduce((s, v) => s + Math.log(v + 1e-30), 0) / P.length);
        out.flat.push(geo / (tot / P.length));
        let best = 0; for (let k = 0; k + 1 < P.length; k++) best = Math.max(best, P[k] + P[k + 1]);
        out.peakShare.push(best / tot);
      }
    }
  }
  return out;
}

const q = (a, f) => { const s = [...a].sort((u, v) => u - v); return s.length ? s[Math.min(s.length - 1, Math.floor(f * s.length))] : NaN; };
const fmt = (a, d = 2) => [0.1, 0.25, 0.5, 0.75, 0.9].map((f) => q(a, f).toFixed(d)).join(' ');
if (process.argv[1].endsWith('events.mjs')) {
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i += 2) {
    const ev = events(load(args[i]), args[i + 1]);
    const secs = args[i + 1].split(',').reduce((s, sp) => { const [a, b] = sp.split(':').map(Number); return s + b - a; }, 0);
    const amps = [...ev.amp].sort((u, v) => u - v); const med = q(amps, 0.5);
    const cv = (() => { const m = ev.ioi.reduce((s, v) => s + v, 0) / ev.ioi.length; return Math.sqrt(ev.ioi.reduce((s, v) => s + (v - m) ** 2, 0) / ev.ioi.length) / m; })();
    console.log(`\n${args[i].split('/').pop()}  (p10 p25 p50 p75 p90)`);
    console.log(`  peaks per s ${(ev.prom.length / secs).toFixed(0)}, prominence dB ${fmt(ev.prom, 1)}`);
    console.log(`  events (> 2x local rms) per s ${(ev.amp.length / secs).toFixed(0)}; amplitude re median dB ${fmt(ev.amp.map((v) => 20 * Math.log10(v / med)), 1)}`);
    console.log(`  spacing ms ${fmt(ev.ioi, 1)} (CV ${cv.toFixed(2)})`);
    console.log(`  duration ms ${fmt(ev.dur, 2)}`);
    console.log(`  flatness ${fmt(ev.flat)}; strongest 2-bin share ${fmt(ev.peakShare)}`);
  }
}
