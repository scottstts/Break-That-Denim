import { describe, expect, it } from 'vitest';
import { RECORD_FLOATS, SOURCES, SOURCE_SLOTS, SRC, decodeRecord, fracturePosition, sourceAt } from '../../src/audio/record';
import { BANDS, SPECTRA, TEAR_VOICE, TearVoice, bandFilter, bandImpulse } from '../../src/audio/tearVoice';
import { axialEnergyDensity, strainAtTension } from '../../src/simulation/axial';
import { COTTON_FIBRE_DENSITY, DENIM } from '../../src/simulation/material';

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

/** Render `seconds` of the voice, fed `recs` as the engine does (four substeps per 1/30 s frame). */
function render(recs: Float32Array, seconds: number, seed = 7): { L: Float32Array; R: Float32Array } {
  const v = new TearVoice(FS, seed);
  const n = Math.ceil((seconds * FS) / 128) * 128;
  const L = new Float32Array(n);
  const R = new Float32Array(n);
  const frame = 4 * RECORD_FLOATS;
  let next = 0;
  for (let i = 0; i < n; i += 128) {
    while (next < recs.length && (next / frame) * (FS / 30) <= i) {
      v.push(recs.subarray(next, next + frame));
      next += frame;
    }
    v.process(L.subarray(i, i + 128), R.subarray(i, i + 128));
  }
  return { L, R };
}

const energy = (x: Float32Array) => x.reduce((s, v) => s + v * v, 0);

/** Power-weighted mean frequency (plain DFT on a coarse grid). */
function centroid(x: Float32Array): number {
  let num = 0;
  let den = 0;
  const n = 4096;
  for (let s = 0; s + n <= x.length; s += n) {
    for (let k = 4; k < n / 2; k += 4) {
      let re = 0;
      let im = 0;
      for (let i = 0; i < n; i++) {
        const a = (-2 * Math.PI * k * i) / n;
        re += x[s + i] * Math.cos(a);
        im += x[s + i] * Math.sin(a);
      }
      const p = re * re + im * im;
      num += ((k * FS) / n) * p;
      den += p;
    }
  }
  return num / den;
}

describe('tear voice', () => {
  it('the energy per fibre follows from the denim yarns', () => {
    // strain energy density at peak tension over the fibre count, averaged over the two families
    const perFibre = DENIM.yarns.map((y) => axialEnergyDensity(y, strainAtTension(y, y.strength)) / (y.linearDensity / COTTON_FIBRE_DENSITY));
    const mean = perFibre.reduce((s, v) => s + v, 0) / perFibre.length;
    expect(Math.abs(TEAR_VOICE.fracture.fibreEnergy / mean - 1)).toBeLessThan(0.05);
    // the smallest event is a strand, well short of a whole yarn
    expect(TEAR_VOICE.fracture.bundle).toBeLessThan(0.2 * Math.min(...DENIM.yarns.map((y) => y.linearDensity / COTTON_FIBRE_DENSITY)));
  });

  it('bands have unit-energy impulse responses and the spectra are shares', () => {
    for (let b = 0; b < BANDS; b++) {
      const ir = bandImpulse(bandFilter(FS, b), FS);
      expect(ir.reduce((s, v) => s + v * v, 0)).toBeCloseTo(1, 6);
    }
    for (const s of [SPECTRA.fracture, SPECTRA.friction]) {
      expect(s).toHaveLength(BANDS);
      expect(s.reduce((a, v) => a + v, 0)).toBeCloseTo(1, 2);
    }
  });

  it('is silent and idle without records', () => {
    const v = new TearVoice(FS);
    const L = new Float32Array(128);
    const R = new Float32Array(128);
    for (let k = 0; k < 50; k++) v.process(L, R);
    expect(energy(L) + energy(R)).toBe(0);
    expect(v.idle).toBe(true);
  });

  it('output energy follows the released energy, stays in range and goes idle afterwards', () => {
    // a steady tear: about 100 J/m per second, thousands of events
    const a = render(records(60, { fracture: 0.8 }), 1.2);
    const b = render(records(60, { fracture: 1.6 }), 1.2);
    const ea = energy(a.L) + energy(a.R);
    const eb = energy(b.L) + energy(b.R);
    expect(ea).toBeGreaterThan(0);
    // twice the fracture energy is twice the sound energy (within the randomness of the events)
    expect(eb / ea).toBeGreaterThan(1.7);
    expect(eb / ea).toBeLessThan(2.3);
    for (const x of [b.L, b.R]) for (const s of x) expect(Math.abs(s)).toBeLessThanOrEqual(1);
    const v = new TearVoice(FS, 3);
    v.push(records(30, { fracture: 0.05 }));
    const L = new Float32Array(128);
    const R = new Float32Array(128);
    for (let k = 0; k < Math.ceil((1.2 * FS) / 128); k++) v.process(L, R);
    expect(v.idle).toBe(true);
  });

  it('sounds a steady release as a continuous texture, not clicks over silence', () => {
    const { L } = render(records(120, { fracture: 0.8 }), 1.0);
    // 5 ms windows over the steady part: none silent, level within a few dB of the median
    const rms: number[] = [];
    for (let s = Math.round(0.25 * FS); s + 240 <= Math.round(0.95 * FS); s += 240) rms.push(Math.sqrt(energy(L.subarray(s, s + 240)) / 240));
    const sorted = [...rms].sort((p, q) => p - q);
    const med = sorted[sorted.length >> 1];
    expect(sorted[0]).toBeGreaterThan(0.25 * med);
    expect(sorted[sorted.length - 1]).toBeLessThan(4 * med);
  });

  it('is deterministic for a seed', () => {
    const r = records(30, { fracture: 0.3, pullouts: 4, pulloutWork: 2e-4, slipRasp: 1e-5 });
    const a = render(r, 0.4, 11);
    const b = render(r, 0.4, 11);
    expect(Array.from(a.L)).toEqual(Array.from(b.L));
  });

  it('friction is darker than fracture', () => {
    const fr = render(records(60, { fracture: 0.8 }), 0.6);
    const sl = render(records(60, { slipRasp: 2e-5 }), 0.6);
    expect(energy(sl.L)).toBeGreaterThan(0);
    expect(centroid(sl.L)).toBeLessThan(0.7 * centroid(fr.L));
  });

  it('creep is quieter than fast slip of the same work', () => {
    // the rasp term is work times slip speed: the same work at a tenth of the speed
    const fast = render(records(60, { slipRasp: 2e-5 }), 0.6);
    const slow = render(records(60, { slipRasp: 2e-6 }), 0.6);
    expect(energy(slow.L) * 5).toBeLessThan(energy(fast.L));
  });

  it('places each source where its fracture is', () => {
    const right = render(records(60, { fracture: 0.8, pan: 0.8 }), 0.6);
    expect(energy(right.R)).toBeGreaterThan(2 * energy(right.L));
    // two hands, two tears: one source on each side, both heard
    const both = render(records(60, { fracture: 0.8, pan: -0.8 }, { fracture: 0.8, pan: 0.8 }), 0.6);
    const l = energy(both.L);
    const r = energy(both.R);
    expect(l / r).toBeGreaterThan(0.6);
    expect(l / r).toBeLessThan(1.6);
    expect(l + r).toBeGreaterThan(1.5 * (energy(right.L) + energy(right.R)));
  });
});

describe('acoustic records', () => {
  it('decode the GPU accumulators into physical units per source', () => {
    const raw = new Uint32Array(SOURCES * SOURCE_SLOTS);
    const b = SOURCE_SLOTS; // source 1
    raw[b] = 1500; // 1e-5 J/m
    raw[b + 1] = 5;
    raw[b + 2] = 4000; // 1e-8 J
    raw[b + 3] = 250; // 1e-8 J mm/s
    // fracture-weighted position: (10 mm, -20 mm, 3 mm) at weight 4
    new Int32Array(raw.buffer, 4 * (b + 4), 3).set([40, -80, 12]);
    raw[b + 7] = 4;
    const out = new Float32Array(RECORD_FLOATS);
    decodeRecord(raw, 0, out, 0, 0.01, [0, 0.25, 0, 0]);
    const o = sourceAt(1);
    expect(out[0]).toBeCloseTo(0.01, 7);
    expect(out[o + SRC.fracture]).toBeCloseTo(0.015, 7);
    expect(out[o + SRC.pullouts]).toBe(5);
    expect(out[o + SRC.pulloutWork]).toBeCloseTo(4e-5, 10);
    expect(out[o + SRC.slipRasp]).toBeCloseTo(2.5e-9, 14);
    expect(out[o + SRC.pan]).toBeCloseTo(0.25, 6);
    expect(out[sourceAt(0) + SRC.fracture]).toBe(0);
    expect(fracturePosition(raw, 0, 0)).toBeNull();
    const at = fracturePosition(raw, 0, 1)!;
    expect(at[0]).toBeCloseTo(0.01, 9);
    expect(at[1]).toBeCloseTo(-0.02, 9);
    expect(at[2]).toBeCloseTo(0.003, 9);
  });
});
