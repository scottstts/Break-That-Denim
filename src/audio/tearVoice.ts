import { RECORD_FLOATS, SOURCES, SRC, sourceAt } from './record';

/**
 * The tear sound: the recorded denim tear (public/fabric_tear_sound_sample.mp3) carries the sound; the
 * mechanics only say, per sound source (one per finger), whether the cloth is tearing there, how hard,
 * and where (docs/decisions/2026-10-10-tear-sound-sample.md).
 *
 * - Content: the recording's tearing passages, played as long overlapping slices (`grain`, power-
 *   complementary windows) from random places, at their own speed and pitch. Their slow loudness (the
 *   recording hand's swells) is divided out so that loudness is the mechanics' alone; everything faster
 *   (the rasp, the events, the treble, the room) is the recording's.
 * - Tearing: a source sounds while it is held and its yarns have been cut within `tearWindow`.
 * - Loudness: acoustic power follows the strain energy the tear releases per second, so the amplitude
 *   is sqrt(rate / referenceRate) (the recording's own level at the reference), averaged over
 *   `rateTime`, bounded by `maxGain`.
 * - Stopping (the tear stops, or the finger lets go): the level falls off like the recording's room
 *   tail, 60 dB in `tail` seconds.
 */
export const TEAR_SAMPLE = {
  /** The recording's tearing passages (s): its handling before each tear is not tearing. */
  segments: [
    [0.58, 1.13],
    [2.4, 2.75],
    [3.87, 5.02],
  ] as [number, number][],
  /** Slice length (s): long enough to carry the recording's rasp and events whole. */
  grain: 0.1,
  /** Time over which the recording's own loudness is measured and divided out (s). */
  levelTime: 0.06,
  /**
   * Strain energy released per second (J/m/s) heard at the recording's level: a hand tear of intact
   * denim at about 200 yarns/s (its rasp) times 0.3 J/m per yarn (400 fibres x 7.6e-4 J/m).
   */
  referenceRate: 60,
  /** Loudest a tear may be heard, re the recording (amplitude). */
  maxGain: 2,
  /** Averaging time of the release rate (s): longer than the lattice's span-by-span parting. */
  rateTime: 0.08,
  /** A source is tearing while a yarn was cut within this time (s). */
  tearWindow: 0.1,
  /** Onset time constant (s). */
  attack: 0.02,
  /** Fall-off: 60 dB in this time (s), the recording's room tail after its tears stop. */
  tail: 0.3,
};

/** Stereo: the screen's width spans this much of the pan range; a tear's place glides over this time (s). */
const PAN_WIDTH = 0.5;
const PAN_GLIDE = 0.06;
/** Limiter: threshold, attack and release (s). */
const LIMIT = 0.7;
const LIMIT_ATTACK = 0.0008;
const LIMIT_RELEASE = 0.25;
/** Below this gain a source has fallen silent. */
const SILENT = 1e-4;

/** The recording prepared for playback: mono, high-passed, its slow loudness divided out. */
export interface TearSource {
  pcm: Float32Array;
  /** Slice start ranges [first, last] (samples): a slice starting there stays inside its passage. */
  starts: [number, number][];
  /** RMS of the recording's tearing passages (the level heard at gain 1). */
  level: number;
}

/**
 * Prepare the decoded recording (mono, at the output sample rate): 2nd-order high-pass at 70 Hz
 * (hand-handling rumble), loudness flattened over `levelTime`, tearing passages only.
 */
export function prepareSource(mono: Float32Array, sampleRate: number): TearSource {
  const S = TEAR_SAMPLE;
  // high-pass (RBJ, Q 0.707)
  const w = (2 * Math.PI * 70) / sampleRate;
  const al = Math.sin(w) / (2 * Math.SQRT1_2);
  const cw = Math.cos(w);
  const a0 = 1 + al;
  const b0 = (1 + cw) / 2 / a0;
  const b1 = -(1 + cw) / a0;
  const a1 = (-2 * cw) / a0;
  const a2 = (1 - al) / a0;
  const x = new Float32Array(mono.length);
  let z1 = 0;
  let z2 = 0;
  for (let i = 0; i < mono.length; i++) {
    const v = mono[i];
    const y = b0 * v + z1;
    z1 = b1 * v - a1 * y + z2;
    z2 = b0 * v - a2 * y;
    x[i] = y;
  }
  // slow loudness: RMS over levelTime (centred)
  const half = Math.round((S.levelTime * sampleRate) / 2);
  const pre = new Float64Array(x.length + 1);
  for (let i = 0; i < x.length; i++) pre[i + 1] = pre[i] + x[i] * x[i];
  const grain = Math.round(S.grain * sampleRate);
  const starts: [number, number][] = [];
  let e = 0;
  let n = 0;
  for (const [t0, t1] of S.segments) {
    const a = Math.round(t0 * sampleRate);
    const b = Math.min(x.length, Math.round(t1 * sampleRate));
    if (b - a > grain) starts.push([a, b - grain]);
    e += pre[b] - pre[a];
    n += Math.max(0, b - a);
  }
  const level = Math.sqrt(e / Math.max(1, n));
  const pcm = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const a = Math.max(0, i - half);
    const b = Math.min(x.length, i + half);
    const rms = Math.sqrt((pre[b] - pre[a]) / Math.max(1, b - a));
    pcm[i] = x[i] / Math.max(rms, 0.1 * level);
  }
  return { pcm, starts, level };
}

/** One finger's tear: its slices, loudness and place. */
interface Voice {
  held: boolean;
  /** Release rate averaged over rateTime (J/m/s); time since the last cut (s, audio time). */
  rate: number;
  idle: number;
  target: number;
  gain: number;
  pan: number;
  panTarget: number;
  panSet: boolean;
  /** Two overlapping slices: their read positions (-1: not playing), and the age of the first (samples). */
  pos: [number, number];
  age: number;
}

const voice = (held: boolean): Voice => ({ held, rate: 0, idle: Infinity, target: 0, gain: 0, pan: 0, panTarget: 0, panSet: false, pos: [-1, -1], age: 0 });

export class TearVoice {
  readonly sampleRate: number;
  private seed: number;
  private source: TearSource | null = null;
  private total = 0;
  private readonly voices: Voice[];
  private readonly grain: number;
  private readonly window: Float32Array;
  private readonly attackK: number;
  private readonly tailK: number;
  private readonly panK: number;
  private env = 0;
  private limGain = 1;
  private readonly limAttack: number;
  private readonly limRelease: number;

  /**
   * `held`: whether the sources start held. The product starts with none and follows each finger's grab
   * and release (`grip`); offline tools hear every record.
   */
  constructor(sampleRate: number, seed = 0x9e3779b9, held = true) {
    this.sampleRate = sampleRate;
    this.seed = seed >>> 0 || 1;
    this.voices = Array.from({ length: SOURCES }, () => voice(held));
    this.grain = 2 * Math.round((TEAR_SAMPLE.grain * sampleRate) / 2);
    // power-complementary across the half overlap (sin^2 + cos^2 = 1; the slices are uncorrelated)
    this.window = Float32Array.from({ length: this.grain }, (_, i) => Math.sin((Math.PI * (i + 0.5)) / this.grain));
    this.attackK = 1 - Math.exp(-1 / (TEAR_SAMPLE.attack * sampleRate));
    this.tailK = 1 - Math.exp(-Math.log(1000) / (TEAR_SAMPLE.tail * sampleRate));
    this.panK = 1 - Math.exp(-1 / (PAN_GLIDE * sampleRate));
    this.limAttack = Math.exp(-1 / (LIMIT_ATTACK * sampleRate));
    this.limRelease = Math.exp(-1 / (LIMIT_RELEASE * sampleRate));
  }

  /** The prepared recording (prepareSource); silent until it is set. */
  setSource(source: TearSource): void {
    this.total = source.starts.reduce((s, [a, b]) => s + b - a + 1, 0);
    this.source = this.total > 0 ? source : null;
  }

  /** Nothing sounding. */
  get idle(): boolean {
    return this.voices.every((v) => v.pos[0] < 0 && v.target === 0);
  }

  /** A finger grabs (held) or lets go of the cloth for source s; letting go falls off like the room's tail. */
  grip(s: number, held: boolean): void {
    const v = this.voices[s];
    if (!v) return;
    v.held = held;
    if (!held) {
      v.target = 0;
      v.rate = 0;
      v.idle = Infinity;
    }
  }

  /** Acoustic records of one frame (RECORD_FLOATS each): what the mechanics released per source. */
  push(records: Float32Array): void {
    const S = TEAR_SAMPLE;
    let dt = 0;
    for (let o = 0; o + RECORD_FLOATS <= records.length; o += RECORD_FLOATS) dt += records[o];
    if (!(dt > 0)) return;
    const a = 1 - Math.exp(-dt / S.rateTime);
    for (let s = 0; s < SOURCES; s++) {
      const v = this.voices[s];
      if (!v.held) continue;
      let energy = 0;
      let cuts = 0;
      let pan = NaN;
      for (let o = 0; o + RECORD_FLOATS <= records.length; o += RECORD_FLOATS) {
        const b = o + sourceAt(s);
        energy += records[b + SRC.fracture];
        cuts += records[b + SRC.cuts];
        if (records[b + SRC.fracture] > 0) pan = records[b + SRC.pan];
      }
      v.rate += a * (energy / dt - v.rate);
      // time since the last cut runs with the audio (process): records stop when the cloth rests
      if (cuts > 0) v.idle = 0;
      if (!Number.isNaN(pan)) {
        v.panTarget = PAN_WIDTH * Math.max(-1, Math.min(1, pan));
        if (!v.panSet) v.pan = v.panTarget;
        v.panSet = true;
      }
      const tearing = v.idle < S.tearWindow;
      v.target = tearing ? Math.min(S.maxGain, Math.sqrt(Math.max(0, v.rate) / S.referenceRate)) : 0;
    }
  }

  private rand(): number {
    let x = this.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.seed = x >>> 0;
    return (this.seed + 0.5) / 4294967296;
  }

  /** A slice start from the tearing passages, chosen at random. */
  private slice(): number {
    const starts = this.source!.starts;
    let r = Math.floor(this.rand() * this.total);
    for (const [a, b] of starts) {
      if (r <= b - a) return a + r;
      r -= b - a + 1;
    }
    return starts[0][0];
  }

  /** Render one block into L and R (equal lengths). */
  process(L: Float32Array, R: Float32Array): void {
    L.fill(0);
    R.fill(0);
    const src = this.source;
    if (!src) return;
    const n = L.length;
    const G = this.grain;
    const H = G / 2;
    const win = this.window;
    const pcm = src.pcm;
    let any = false;
    for (const v of this.voices) {
      if (v.pos[0] < 0) {
        if (v.target === 0) continue;
        // a tear starting: two slices, the second half way through
        v.pos[0] = this.slice();
        v.pos[1] = this.slice();
        v.age = 0;
        v.gain = 0;
      }
      any = true;
      v.idle += n / this.sampleRate;
      if (v.idle >= TEAR_SAMPLE.tearWindow) v.target = 0;
      for (let i = 0; i < n; i++) {
        v.gain += (v.target > v.gain ? this.attackK : this.tailK) * (v.target - v.gain);
        v.pan += this.panK * (v.panTarget - v.pan);
        const a0 = v.age;
        const a1 = a0 < H ? a0 + H : a0 - H;
        const y = (pcm[v.pos[0] + a0] * win[a0] + pcm[v.pos[1] + a1] * win[a1]) * v.gain * src.level;
        const ang = (v.pan + 1) * 0.25 * Math.PI;
        L[i] += y * Math.cos(ang) * Math.SQRT2;
        R[i] += y * Math.sin(ang) * Math.SQRT2;
        // every half slice one of the two ends and a new one starts in its place
        if (++v.age === H) v.pos[1] = this.slice();
        else if (v.age === G) {
          v.age = 0;
          v.pos[0] = this.slice();
        }
      }
      if (v.target === 0 && v.gain < SILENT) {
        v.pos[0] = -1;
        if (!v.held) v.panSet = false;
      }
    }
    if (!any) return;
    // peak limiter (fast attack, slow release) and a soft knee near full scale: several hands at once
    let env = this.env;
    let g = this.limGain;
    for (let i = 0; i < n; i++) {
      const p = Math.max(Math.abs(L[i]), Math.abs(R[i]));
      env = p > env ? p : env * this.limRelease + p * (1 - this.limRelease);
      const target = env > LIMIT ? LIMIT / env : 1;
      g = target < g ? target + (g - target) * this.limAttack : target + (g - target) * this.limRelease;
      L[i] = softClip(L[i] * g);
      R[i] = softClip(R[i] * g);
    }
    this.env = env;
    this.limGain = g;
  }
}

function softClip(x: number): number {
  const a = Math.abs(x);
  if (a <= 0.8) return x;
  // continuous slope at 0.8, saturating at 1
  const y = 0.8 + 0.2 * Math.tanh((a - 0.8) / 0.2);
  return x < 0 ? -y : y;
}
