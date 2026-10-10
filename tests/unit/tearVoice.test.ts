import { describe, expect, it } from 'vitest';
import { RECORD_FLOATS, SOURCES, SOURCE_SLOTS, SRC, decodeRecord, fracturePosition, sourceAt } from '../../src/audio/record';
import { TEAR_SAMPLE, TearVoice, prepareSource, type TearSource } from '../../src/audio/tearVoice';

const FS = 48000;
const H = 1 / 120;

type Fields = Partial<Record<keyof typeof SRC, number>>;

/** Records (one per substep) with the given fields per source, the rest zero. */
function records(n: number, ...sources: Fields[]): Float32Array {
  const out = new Float32Array(n * RECORD_FLOATS);
  for (let k = 0; k < n; k++) {
    out[k * RECORD_FLOATS] = H;
    sources.forEach((fields, s) => {
      for (const [key, v] of Object.entries(fields)) out[k * RECORD_FLOATS + sourceAt(s) + SRC[key as keyof typeof SRC]] = v as number;
    });
  }
  return out;
}

/** Records of a tear: per source, `rate` J/m/s released and a span of 8 yarns parting every fifth substep. */
function tearing(n: number, ...rates: number[]): Float32Array {
  const out = records(n, ...rates.map((r) => ({ fracture: r * H })));
  for (let k = 0; k < n; k += 5) rates.forEach((_, s) => (out[k * RECORD_FLOATS + sourceAt(s) + SRC.cuts] = 8));
  return out;
}

/** A stand-in recording at about the real one's level: noise whose loudness swells 10x, so that flattening shows. */
function recording(): TearSource {
  let seed = 99;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296) * 2 - 1;
  const x = new Float32Array(Math.round(5.5 * FS));
  for (let i = 0; i < x.length; i++) x[i] = rnd() * 0.01 * (1 + 9 * (0.5 + 0.5 * Math.sin((2 * Math.PI * i) / (0.4 * FS))));
  return prepareSource(x, FS);
}
const SOURCE = recording();

const energy = (x: Float32Array) => x.reduce((s, v) => s + v * v, 0);
const rms = (x: Float32Array, t0: number, t1: number) => Math.sqrt(energy(x.subarray(Math.round(t0 * FS), Math.round(t1 * FS))) / ((t1 - t0) * FS));

/** Render `seconds`, feeding `recs` a frame (four substeps) per 1/30 s, with callbacks at given times. */
function render(recs: Float32Array, seconds: number, opts: { seed?: number; held?: boolean; at?: [number, (v: TearVoice) => void][] } = {}): { L: Float32Array; R: Float32Array } {
  const v = new TearVoice(FS, opts.seed ?? 7, opts.held ?? true);
  v.setSource(SOURCE);
  const n = Math.ceil((seconds * FS) / 128) * 128;
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  const frame = 4 * RECORD_FLOATS;
  const events = [...(opts.at ?? [])];
  let next = 0;
  for (let i = 0; i < n; i += 128) {
    while (events.length && events[0][0] * FS <= i) events.shift()![1](v);
    while (next < recs.length && (next / frame) * (FS / 30) <= i) {
      v.push(recs.subarray(next, next + frame));
      next += frame;
    }
    v.process(L.subarray(i, i + 128), R.subarray(i, i + 128));
  }
  return { L, R };
}

describe('tear voice', () => {
  it('prepares the recording: its tearing passages, its slow loudness divided out', () => {
    const grain = Math.round(TEAR_SAMPLE.grain * FS);
    expect(SOURCE.starts).toHaveLength(TEAR_SAMPLE.segments.length);
    SOURCE.starts.forEach(([a, b], k) => {
      expect(a).toBe(Math.round(TEAR_SAMPLE.segments[k][0] * FS));
      expect(b + grain).toBeLessThanOrEqual(Math.round(TEAR_SAMPLE.segments[k][1] * FS));
    });
    // 50 ms windows across a 10x swell come out at the same level
    const w: number[] = [];
    for (let t = 0.6; t < 1.1; t += 0.05) w.push(rms(SOURCE.pcm, t, t + 0.05));
    expect(Math.max(...w) / Math.min(...w)).toBeLessThan(1.3);
  });

  it('is silent without a tear: no records, or fibres giving without yarns parting', () => {
    const quiet = render(new Float32Array(0), 0.5);
    expect(energy(quiet.L)).toBe(0);
    const stretched = render(records(60, { fracture: 60 * H }), 0.8);
    expect(energy(stretched.L) + energy(stretched.R)).toBe(0);
  });

  it('is heard at the recording\'s level at the reference rate, its amplitude the root of the rate', () => {
    const at = (rate: number) => {
      const { L, R } = render(tearing(150, rate), 1.3);
      return Math.sqrt((rms(L, 0.4, 1.2) ** 2 + rms(R, 0.4, 1.2) ** 2) / 2);
    };
    const ref = at(TEAR_SAMPLE.referenceRate);
    expect(ref / SOURCE.level).toBeGreaterThan(0.85);
    expect(ref / SOURCE.level).toBeLessThan(1.15);
    const quarter = at(TEAR_SAMPLE.referenceRate / 4);
    expect(quarter / ref).toBeGreaterThan(0.42);
    expect(quarter / ref).toBeLessThan(0.58);
    // bounded however hard the tear
    expect(at(100 * TEAR_SAMPLE.referenceRate) / ref).toBeLessThan(TEAR_SAMPLE.maxGain * 1.1);
  });

  it('joins its slices without clicks', () => {
    // a smooth stand-in recording (1 kHz): the largest step between samples stays that of the tone
    const x = Float32Array.from({ length: Math.round(5.5 * FS) }, (_, i) => Math.sin((2 * Math.PI * 1000 * i) / FS));
    const v = new TearVoice(FS, 3);
    v.setSource(prepareSource(x, FS));
    const L = new Float32Array(128);
    const R = new Float32Array(128);
    let peak = 0;
    let step = 0;
    let prev = 0;
    for (let k = 0; k < 300; k++) {
      if (k % 12 === 0) v.push(tearing(4, TEAR_SAMPLE.referenceRate));
      v.process(L, R);
      for (let i = 0; i < 128; i++) {
        if (k > 60) {
          peak = Math.max(peak, Math.abs(L[i]));
          step = Math.max(step, Math.abs(L[i] - prev));
        }
        prev = L[i];
      }
    }
    expect(peak).toBeGreaterThan(0);
    expect(step).toBeLessThan(1.5 * peak * 2 * Math.sin((Math.PI * 1000) / FS));
  });

  it('falls off like the room\'s tail when the finger lets go, then is silent', () => {
    const { L } = render(tearing(240, TEAR_SAMPLE.referenceRate), 2.0, { at: [[1.0, (v) => v.grip(0, false)]] });
    const before = rms(L, 0.8, 1.0);
    // not a cut: the first 30 ms after release still carry much of the level
    expect(rms(L, 1.0, 1.03)).toBeGreaterThan(0.3 * before);
    // about 60 dB down after the tail time, silent after
    expect(rms(L, 1.0 + TEAR_SAMPLE.tail, 1.05 + TEAR_SAMPLE.tail)).toBeLessThan(0.003 * before);
    expect(rms(L, 1.6, 2.0)).toBe(0);
  });

  it('falls off the same way when the tear stops under a held finger', () => {
    const { L } = render(tearing(90, TEAR_SAMPLE.referenceRate), 1.8);
    // the last records play at about 0.75 s; the tear window and the tail follow
    const before = rms(L, 0.4, 0.6);
    const end = 0.75 + TEAR_SAMPLE.tearWindow;
    expect(rms(L, end, end + 0.03)).toBeGreaterThan(0.05 * before);
    expect(rms(L, 1.5, 1.8)).toBe(0);
  });

  it('follows each finger: silent until grabbed, then heard', () => {
    const recs = tearing(240, TEAR_SAMPLE.referenceRate);
    const none = render(recs, 1.0, { held: false });
    expect(energy(none.L)).toBe(0);
    const grabbed = render(recs, 1.0, { held: false, at: [[0.2, (v) => v.grip(0, true)]] });
    expect(rms(grabbed.L, 0, 0.2)).toBe(0);
    expect(rms(grabbed.L, 0.5, 0.9)).toBeGreaterThan(0.5 * SOURCE.level);
  });

  it('places each source where its tear is; two hands are two places', () => {
    const tear = (pan: number): Fields => ({ fracture: TEAR_SAMPLE.referenceRate * H, cuts: 2, pan });
    const right = render(records(60, tear(1)), 0.8);
    expect(energy(right.R)).toBeGreaterThan(2 * energy(right.L));
    const both = render(records(60, tear(-1), tear(1)), 0.8);
    const r = energy(both.L) / energy(both.R);
    expect(r).toBeGreaterThan(0.7);
    expect(r).toBeLessThan(1.4);
  });

  it('is deterministic for a seed', () => {
    const r = tearing(30, 40);
    expect(Array.from(render(r, 0.4, { seed: 11 }).L)).toEqual(Array.from(render(r, 0.4, { seed: 11 }).L));
  });
});

describe('acoustic records', () => {
  it('decode the GPU accumulators into physical units per source', () => {
    const raw = new Uint32Array(SOURCES * SOURCE_SLOTS);
    const b = SOURCE_SLOTS; // source 1
    raw[b] = 1500; // 1e-5 J/m
    // fracture-weighted position: (10 mm, -20 mm, 3 mm) at weight 4
    new Int32Array(raw.buffer, 4 * (b + 1), 3).set([40, -80, 12]);
    raw[b + 4] = 4;
    raw[b + 5] = 3; // yarns cut
    const out = new Float32Array(RECORD_FLOATS);
    decodeRecord(raw, 0, out, 0, 0.01, [0, 0.25, 0, 0]);
    const o = sourceAt(1);
    expect(out[0]).toBeCloseTo(0.01, 7);
    expect(out[o + SRC.fracture]).toBeCloseTo(0.015, 7);
    expect(out[o + SRC.pan]).toBeCloseTo(0.25, 6);
    expect(out[o + SRC.cuts]).toBe(3);
    expect(out[sourceAt(0) + SRC.fracture]).toBe(0);
    expect(fracturePosition(raw, 0, 0)).toBeNull();
    const at = fracturePosition(raw, 0, 1)!;
    expect(at[0]).toBeCloseTo(0.01, 9);
    expect(at[1]).toBeCloseTo(-0.02, 9);
    expect(at[2]).toBeCloseTo(0.003, 9);
  });
});
