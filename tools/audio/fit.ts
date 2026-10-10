/**
 * Fits the tear voice's spectral envelopes (src/audio/tearVoice.ts SPECTRA) to a recorded tear:
 * fracture to the long-term spectrum of its loud passages (within 12 dB of the peak 5 ms level),
 * friction to its quiet passages (25-45 dB below), over the third-octave bands 160 Hz - 10 kHz.
 * Every event of the voice is one band's (unit-energy) response, so the model spectrum is the mixture
 * of the band responses weighted by the share of events per band; the fit minimises the error per
 * third-octave band in dB, with a curvature penalty that keeps each envelope smooth across the bands.
 *
 *   node tools/gpu/run.mjs tools/audio/fit.ts [recording]
 */
import { execFileSync } from 'node:child_process';
import { BANDS, bandFilter, bandImpulse } from '../../src/audio/tearVoice';

const SR = 48000;
const NFFT = 2048;
/** Curvature penalty of the envelopes (weight per squared dB of second difference). */
const SMOOTH = Number(process.env.SMOOTH ?? 0.05);

function fft(re: Float64Array, im: Float64Array): void {
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
        const h = i + k + len / 2;
        const vr = re[h] * c - im[h] * s;
        const vi = re[h] * s + im[h] * c;
        re[h] = re[i + k] - vr;
        im[h] = im[i + k] - vi;
        re[i + k] += vr;
        im[i + k] += vi;
      }
    }
  }
}

function power(sig: ArrayLike<number>, start: number, window: boolean): Float64Array {
  const re = new Float64Array(NFFT);
  const im = new Float64Array(NFFT);
  for (let i = 0; i < NFFT && start + i < sig.length; i++) re[i] = sig[start + i] * (window ? 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (NFFT - 1)) : 1);
  fft(re, im);
  const p = new Float64Array(NFFT / 2);
  for (let k = 0; k < NFFT / 2; k++) p[k] = re[k] * re[k] + im[k] * im[k];
  return p;
}

const thirds: number[] = [];
for (let k = 0; ; k++) {
  const fc = 160 * 2 ** (k / 3);
  if (fc > 10500) break;
  thirds.push(fc);
}
function thirdBands(p: Float64Array): number[] {
  return thirds.map((fc) => {
    let s = 0;
    for (let k = 1; k < NFFT / 2; k++) {
      const f = (k * SR) / NFFT;
      if (f >= fc / 2 ** (1 / 6) && f < fc * 2 ** (1 / 6)) s += p[k];
    }
    return s;
  });
}

export default async function main(args: string[]): Promise<void> {
  const file = args[0] ?? 'refs/fabric_tear_sound_sample.mp3';
  const raw = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', file, '-ar', String(SR), '-ac', '1', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], { maxBuffer: 1 << 28 });
  const x = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
  // passages by 5 ms level
  const HOP = 240;
  const env: number[] = [];
  for (let i = 0; i + HOP <= x.length; i += HOP) {
    let s = 0;
    for (let j = i; j < i + HOP; j++) s += x[j] * x[j];
    env.push(s / HOP);
  }
  const peak = Math.max(...env);
  const ltas = (lo: number, hi: number) => {
    const acc = new Float64Array(NFFT / 2);
    let count = 0;
    for (let s = 0; s + NFFT <= x.length; s += NFFT / 4) {
      // a window counts when all of its 5 ms frames lie in the level range
      let ok = true;
      for (let f = Math.floor(s / HOP); f < Math.floor((s + NFFT) / HOP) && ok; f++) {
        const d = 10 * Math.log10(env[f] / peak);
        ok = d > lo && d <= hi;
      }
      if (!ok) continue;
      const p = power(x, s, true);
      for (let k = 0; k < NFFT / 2; k++) acc[k] += p[k];
      count++;
    }
    return { bands: thirdBands(acc), count };
  };
  const loud = ltas(-12, 1);
  const quiet = ltas(-45, -25);

  // model: every event is one band's (unit-energy) response, so the long-term spectrum is the mixture
  // of the band responses weighted by the share of events in each band
  const M = Array.from({ length: BANDS }, (_, b) => thirdBands(power(bandImpulse(bandFilter(SR, b), NFFT), 0, false)));
  const fit = (name: string, target: number[]) => {
    // log-domain envelope u (dB per band), smooth: squared dB error per third-octave band plus a
    // curvature penalty (dB per band squared), minimised by gradient descent (Adam)
    const tDb = target.map((t) => 10 * Math.log10(t));
    const u = new Float64Array(BANDS);
    const S = new Float64Array(BANDS);
    const pred = () => thirds.map((_, j) => M.reduce((acc, m, b2) => acc + S[b2] * m[j], 0));
    const loss = (v: Float64Array) => {
      for (let i = 0; i < BANDS; i++) S[i] = 10 ** (v[i] / 10);
      const p = pred();
      // the level is free: compare shapes after removing the mean dB difference
      const d = p.map((q, j) => 10 * Math.log10(q) - tDb[j]);
      const mean = d.reduce((acc, q) => acc + q, 0) / d.length;
      let l = d.reduce((acc, q) => acc + (q - mean) ** 2, 0) / d.length;
      for (let i = 1; i + 1 < BANDS; i++) l += SMOOTH * (v[i - 1] - 2 * v[i] + v[i + 1]) ** 2;
      return l;
    };
    const m = new Float64Array(BANDS);
    const vv = new Float64Array(BANDS);
    const g = new Float64Array(BANDS);
    for (let it = 1; it <= 4000; it++) {
      const l0 = loss(u);
      for (let i = 0; i < BANDS; i++) {
        u[i] += 1e-4;
        g[i] = (loss(u) - l0) / 1e-4;
        u[i] -= 1e-4;
      }
      for (let i = 0; i < BANDS; i++) {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        vv[i] = 0.999 * vv[i] + 0.001 * g[i] * g[i];
        u[i] -= (0.05 * (m[i] / (1 - 0.9 ** it))) / (Math.sqrt(vv[i] / (1 - 0.999 ** it)) + 1e-9);
      }
    }
    loss(u);
    const p = pred();
    const tMax = Math.max(...target);
    // fitted levels aligned to the target by their mean difference (the fit compares shapes)
    const off = thirds.reduce((acc, _, j) => acc + 10 * Math.log10(target[j] / p[j]), 0) / thirds.length;
    let err = 0;
    console.log(`\n${name}: third-octave dB re the target maximum (target | fitted)`);
    thirds.forEach((fc, j) => {
      const a = 10 * Math.log10(target[j] / tMax);
      const b = 10 * Math.log10(p[j] / tMax) + off;
      err += (a - b) ** 2;
      console.log(`  ${fc.toFixed(0).padStart(6)} Hz ${a.toFixed(1).padStart(6)} ${b.toFixed(1).padStart(6)}`);
    });
    const sum = S.reduce((s, v) => s + v, 0);
    console.log(`  rms error ${Math.sqrt(err / thirds.length).toFixed(2)} dB`);
    console.log(`  ${name}: [${Array.from(S, (v) => (v / sum).toFixed(4)).join(', ')}],`);
  };
  console.log(`${file}: ${loud.count} loud and ${quiet.count} quiet ${NFFT}-sample windows`);
  fit('fracture', loud.bands);
  fit('friction', quiet.bands);
}
