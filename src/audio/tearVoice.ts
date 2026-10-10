import { RECORD_FLOATS, SOURCES, SRC, sourceAt } from './record';

/**
 * Procedural tear sound driven by the mechanics (record.ts): acoustic emission of what the cloth
 * releases, per sound source (one per finger).
 *
 * A recorded denim tear (refs/fabric_tear_sound_sample.mp3), looked at as a waveform, is a continuous,
 * zero-mean, band-limited oscillation centred around 1.5-3 kHz in which short wavelets of one to two
 * cycles overlap densely, some standing out; there are no steps and no silent gaps between events
 * (tools/audio/compare.mjs; docs/decisions/2026-10-10-tear-sound.md). That is the sound of many small
 * emission events: fibre bundles snapping inside the yarns that fail. So:
 *
 * - fibre fracture: the strain energy the yarns release (J/m) breaks fibres in bundles; each bundle
 *   is one event, its size drawn from the fibre bundle avalanche law (P(s) ~ s^-5/2), thousands per
 *   second while the cloth tears;
 * - crossing slip (yarns rubbing over each other): events from the friction work times slip speed,
 *   so creep under a held load stays silent;
 * - pull-out (a yarn end dragged through a crossing): a short stick-slip chatter.
 *
 * Released energy flows through a two-stage reservoir and events fire from it as a Poisson process:
 * the lattice resolves failure per substep and per bundle of 9 x 6 yarns, while the fibres give one
 * after another. Each event is one wavelet: an impulse into one band of a third-octave bank of
 * resonators (one to two cycles), the band drawn from its mechanism's spectral distribution, fitted
 * to the recording (fracture: its loud tearing passages; friction: the quiet stretching before them;
 * tools/audio/fit.ts). Band responses have unit energy, so an event of acoustic energy e adds e to
 * the output whatever its band.
 */
export const TEAR_VOICE = {
  /** Length of yarn over which a fracture releases its strain energy (m): J/m to J. */
  recoilLength: 1e-3,
  fracture: {
    /**
     * Strain energy one cotton fibre holds when its yarn fails (J/m): the denim yarns' energy density
     * at peak tension over their fibre count (linear density / 1.8 dtex; tests/unit/tearVoice.test.ts).
     */
    fibreEnergy: 7.6e-4,
    /**
     * Fibres in the smallest audible event (a strand of the yarn snapping together, about 8 % of its
     * fibres): with it the tear's spikiness matches the recording (kurtosis 5.0 against 5.1, crest
     * 14.4 dB against 14.3 per 50 ms window; tools/audio/analyze.mjs).
     */
    bundle: 32,
    /** Avalanche size exponent of a fibre bundle under global load sharing, and the largest event (in bundles). */
    exponent: 2.5,
    largest: 30,
  },
  pullout: {
    /** Acoustic share of the friction work of a pull-out. */
    efficiency: 0.03,
    /** Stick-slip chatter of one pull-out: impulses and their period (s). */
    impulses: [3, 6],
    period: [0.25e-3, 0.6e-3],
  },
  slip: {
    /** Acoustic share of slip work at the reference slip speed (m/s); it scales with the speed. */
    efficiency: 0.006,
    refSpeed: 0.01,
    /** Mean acoustic energy per event (J): a dense rasp (some 10^4 events/s at a hard pull's slip). */
    quantum: 1e-6,
  },
  /** Time constant of each of the two reservoir stages (s). */
  smoothing: 0.008,
  /** Quality of the resonator bands. */
  q: 3,
  /** Digital energy per acoustic joule (full-scale squared). */
  gain: 1.6e3,
  /** Start delay after silence (s) and the queue lengths that make playback catch up (s). */
  preroll: 0.03,
  catchUp: 0.08,
  rush: 0.16,
};

/** Band centres (Hz): 125 Hz to 10 kHz in third octaves. */
export const BAND_BASE = 125;
export const BAND_STEP = 2 ** (1 / 3);
export const BANDS = 20;
/**
 * Share of each mechanism's events in each band (sum 1), fitted to the recording by
 * tools/audio/fit.ts: fracture to its loud passages, friction to the quiet ones.
 */
export const SPECTRA = {
  fracture: [0.0362, 0.0326, 0.0298, 0.0289, 0.0294, 0.0316, 0.0364, 0.0462, 0.0629, 0.0887, 0.1184, 0.1346, 0.1163, 0.0786, 0.0484, 0.0311, 0.0212, 0.0143, 0.0090, 0.0053],
  friction: [0.2971, 0.1557, 0.0841, 0.0513, 0.0370, 0.0308, 0.0300, 0.0330, 0.0376, 0.0442, 0.0513, 0.0530, 0.0428, 0.0261, 0.0134, 0.0066, 0.0033, 0.0016, 0.0008, 0.0004],
};

const PENDING = 2048;
const MAX_EVENTS_PER_SAMPLE = 4;
const QUEUE = 512;
/** After this long without excitation (s) the bank has rung out: processing stops. */
const RING_OUT = 0.15;
/** Limiter: threshold, attack and release (s). */
const LIMIT = 0.7;
const LIMIT_ATTACK = 0.0008;
const LIMIT_RELEASE = 0.25;

export interface BandFilter {
  a1: number;
  a2: number;
  a3: number;
  /** Scale giving the band's impulse response unit energy. */
  norm: number;
}

/** TPT state-variable band-pass of band b at sample rate fs, normalised to a unit-energy response. */
export function bandFilter(fs: number, b: number): BandFilter {
  const f = Math.min(BAND_BASE * BAND_STEP ** b, 0.45 * fs);
  const g = Math.tan((Math.PI * f) / fs);
  const a1 = 1 / (1 + g * (g + 1 / TEAR_VOICE.q));
  const a2 = g * a1;
  const a3 = g * a2;
  const ir = bandImpulse({ a1, a2, a3, norm: 1 }, fs);
  let e = 0;
  for (const v of ir) e += v * v;
  return { a1, a2, a3, norm: 1 / Math.sqrt(e) };
}

/** Impulse response of a band filter (until it has decayed by 200 dB, at most n samples). */
export function bandImpulse(F: BandFilter, n: number): Float64Array {
  const out = new Float64Array(n);
  let s1 = 0;
  let s2 = 0;
  let e = 0;
  let len = n;
  for (let i = 0; i < n; i++) {
    const v3 = (i === 0 ? 1 : 0) - s2;
    const v1 = F.a1 * s1 + F.a2 * v3;
    const v2 = s2 + F.a2 * s1 + F.a3 * v3;
    s1 = 2 * v1 - s1;
    s2 = 2 * v2 - s2;
    out[i] = v1 * F.norm;
    e += v1 * v1;
    if (i > 64 && v1 * v1 < 1e-20 * e) {
      len = i + 1;
      break;
    }
  }
  return out.subarray(0, len);
}

/** Cumulative distribution of a spectrum (for drawing an event's band). */
function cumulative(S: readonly number[]): Float64Array {
  const c = new Float64Array(S.length);
  const total = S.reduce((a, v) => a + v, 0);
  let s = 0;
  S.forEach((v, i) => (c[i] = s += v / total));
  c[S.length - 1] = 1;
  return c;
}

/** One emission stream of a source: released energy in two reservoir stages, and its event phase. */
interface Stream {
  inflow: number;
  r1: number;
  r2: number;
  phase: number;
  threshold: number;
}

const stream = (): Stream => ({ inflow: 0, r1: 0, r2: 0, phase: 0, threshold: 1 });

export class TearVoice {
  readonly sampleRate: number;
  private seed: number;
  private spare = NaN;
  private readonly bands: BandFilter[] = [];
  private readonly cdf = { fracture: cumulative(SPECTRA.fracture), friction: cumulative(SPECTRA.friction) };
  // band filter states, two channels
  private readonly s1 = new Float64Array(2 * BANDS);
  private readonly s2 = new Float64Array(2 * BANDS);
  private exc = new Float32Array(2 * BANDS * 128);
  // pending pull-out impulses: absolute sample time, energy, band, pan
  private readonly evT = new Float64Array(PENDING);
  private readonly evE = new Float32Array(PENDING);
  private readonly evB = new Uint8Array(PENDING);
  private readonly evP = new Float32Array(PENDING);
  private pending = 0;
  // record queue (ring)
  private readonly queue = new Float32Array(QUEUE * RECORD_FLOATS);
  private qHead = 0;
  private qCount = 0;
  private queued = 0;
  private waiting = 0;
  private playing = false;
  private recLeft = 0;
  // per source: fracture (J/m) and slip (acoustic J) streams, and pan
  private readonly fracture: Stream[] = Array.from({ length: SOURCES }, stream);
  private readonly slip: Stream[] = Array.from({ length: SOURCES }, stream);
  private readonly pan = new Float64Array(SOURCES);
  private readonly drain: number;
  /** Mean avalanche size (bundles). */
  private readonly meanSize: number;
  private clock = 0;
  private quiet = Infinity;
  private env = 0;
  private limGain = 1;
  private readonly attack: number;
  private readonly release: number;

  constructor(sampleRate: number, seed = 0x9e3779b9) {
    this.sampleRate = sampleRate;
    this.seed = seed >>> 0 || 1;
    for (let b = 0; b < BANDS; b++) this.bands.push(bandFilter(sampleRate, b));
    this.attack = Math.exp(-1 / (LIMIT_ATTACK * sampleRate));
    this.release = Math.exp(-1 / (LIMIT_RELEASE * sampleRate));
    this.drain = 1 - Math.exp(-1 / (TEAR_VOICE.smoothing * sampleRate));
    // mean of the truncated power law avalanche() draws from: a = exponent - 1 on [1, L]
    const F = TEAR_VOICE.fracture;
    const a = F.exponent - 1;
    const L = F.largest;
    this.meanSize = a === 1 ? Math.log(L) / (1 - 1 / L) : ((a / (a - 1)) * (1 - L ** (1 - a))) / (1 - L ** -a);
    for (const st of [...this.fracture, ...this.slip]) st.threshold = this.exponential();
  }

  /** Nothing queued, nothing pending or held back, and the bank has rung out. */
  get idle(): boolean {
    return this.qCount === 0 && !this.active() && this.quiet > RING_OUT * this.sampleRate;
  }

  private active(): boolean {
    if (this.pending > 0 || this.recLeft > 0) return true;
    for (let s = 0; s < SOURCES; s++) {
      if (this.fracture[s].r1 + this.fracture[s].r2 > 0 || this.slip[s].r1 + this.slip[s].r2 > 0) return true;
    }
    return false;
  }

  /** Append records (RECORD_FLOATS each); when the queue is full the oldest are dropped. */
  push(records: Float32Array): void {
    for (let o = 0; o + RECORD_FLOATS <= records.length; o += RECORD_FLOATS) {
      if (this.qCount === QUEUE) this.shift();
      const at = ((this.qHead + this.qCount) % QUEUE) * RECORD_FLOATS;
      this.queue.set(records.subarray(o, o + RECORD_FLOATS), at);
      this.queued += records[o];
      this.qCount++;
    }
  }

  private shift(): number {
    const at = this.qHead * RECORD_FLOATS;
    this.qHead = (this.qHead + 1) % QUEUE;
    this.qCount--;
    this.queued = this.qCount === 0 ? 0 : this.queued - this.queue[at];
    return at;
  }

  // ---------------------------------------------------------------- random numbers

  private bits(): number {
    let x = this.seed;
    x ^= x << 13;
    x ^= x >>> 17;
    x ^= x << 5;
    this.seed = x >>> 0;
    return this.seed;
  }

  private rand(): number {
    return (this.bits() + 0.5) / 4294967296;
  }

  private exponential(): number {
    return -Math.log(this.rand());
  }

  private gauss(): number {
    if (!Number.isNaN(this.spare)) {
      const s = this.spare;
      this.spare = NaN;
      return s;
    }
    const r = Math.sqrt(-2 * Math.log(this.rand()));
    const a = 2 * Math.PI * this.rand();
    this.spare = r * Math.sin(a);
    return r * Math.cos(a);
  }

  /** Avalanche size in bundles: P(s > x) ~ x^(1 - exponent) on [1, largest]. */
  private avalanche(): number {
    const F = TEAR_VOICE.fracture;
    const a = F.exponent - 1;
    return (1 - this.rand() * (1 - F.largest ** -a)) ** (-1 / a);
  }

  private band(cdf: Float64Array): number {
    const u = this.rand();
    let b = 0;
    while (b < BANDS - 1 && cdf[b] < u) b++;
    return b;
  }

  // ---------------------------------------------------------------- events

  /** Add a wavelet of acoustic energy e (J) in band b at block sample i. */
  private wavelet(exc: Float32Array, n: number, i: number, e: number, b: number, pan: number): void {
    const v = (this.rand() < 0.5 ? -1 : 1) * Math.sqrt(TEAR_VOICE.gain * e) * this.bands[b].norm;
    // equal-power pan
    const a = (Math.max(-1, Math.min(1, pan)) + 1) * 0.25 * Math.PI;
    exc[2 * b * n + i] += v * Math.cos(a) * Math.SQRT2;
    exc[(2 * b + 1) * n + i] += v * Math.sin(a) * Math.SQRT2;
  }

  /** Schedule the pull-out chatter of source s (record block at o) over [start, start + length) samples. */
  private pullouts(r: Float32Array, o: number, s: number, start: number, length: number): void {
    const P = TEAR_VOICE.pullout;
    const pulls = r[o + SRC.pullouts];
    if (!(pulls > 0)) return;
    const n = Math.min(32, pulls);
    const each = (P.efficiency * r[o + SRC.pulloutWork]) / n;
    const decay = 0.75;
    for (let q = 0; q < n && this.pending < PENDING - 8; q++) {
      const imp = P.impulses[0] + Math.floor(this.rand() * (P.impulses[1] - P.impulses[0] + 1));
      const period = (P.period[0] + this.rand() * (P.period[1] - P.period[0])) * this.sampleRate;
      const b = this.band(this.cdf.friction);
      const p = this.pan[s] + 0.15 * this.gauss();
      let t = start + this.rand() * length;
      // decaying chatter: weights 1, 0.75, 0.56, ... normalised to the pull-out's energy
      const sum = (1 - decay ** imp) / (1 - decay);
      for (let i = 0; i < imp; i++) {
        const k = this.pending++;
        this.evT[k] = t;
        this.evE[k] = (each * decay ** i) / sum;
        this.evB[k] = b;
        this.evP[k] = p;
        t += period * (0.85 + 0.3 * this.rand());
      }
    }
  }

  /** Start the next queued record at block offset `pos`; false when none may start yet. */
  private nextRecord(pos: number, n: number): boolean {
    const V = TEAR_VOICE;
    const fs = this.sampleRate;
    if (!this.playing) {
      if (this.qCount === 0) {
        this.waiting = 0;
        return false;
      }
      // after silence, start once a cushion is queued or the first record has waited for it
      this.waiting += n - pos;
      if (this.queued < V.preroll && this.waiting < V.preroll * fs) return false;
      this.playing = true;
    }
    if (this.qCount === 0) {
      this.playing = false;
      this.waiting = 0;
      return false;
    }
    const behind = this.queued;
    const at = this.shift();
    // a queue grown beyond the cushion (frame hitches) is played faster until it shrinks
    const speed = behind > V.rush ? 2 : behind > V.catchUp ? 1.25 : 1;
    const length = Math.max(1, (this.queue[at] * fs) / speed);
    for (let s = 0; s < SOURCES; s++) {
      const o = at + sourceAt(s);
      this.pan[s] = this.queue[o + SRC.pan];
      this.fracture[s].inflow = this.queue[o + SRC.fracture] / length;
      this.slip[s].inflow = (V.slip.efficiency * this.queue[o + SRC.slipRasp]) / V.slip.refSpeed / length;
      this.pullouts(this.queue, o, s, this.clock + pos, length);
    }
    this.recLeft = length;
    return true;
  }

  /**
   * Run one stream over block samples [from, to): energy flows through the two reservoir stages and
   * events fire as a Poisson process at the drained rate over `unit` (energy per event on average).
   */
  private run(st: Stream, exc: Float32Array, n: number, from: number, to: number, unit: number, fracture: boolean, pan: number): boolean {
    if (st.r1 + st.r2 <= 0 && st.inflow <= 0) return false;
    const cdf = fracture ? this.cdf.fracture : this.cdf.friction;
    const toJ = fracture ? TEAR_VOICE.recoilLength : 1;
    let fired = false;
    for (let i = from; i < to; i++) {
      st.r1 += st.inflow;
      const d1 = st.r1 * this.drain;
      st.r1 -= d1;
      st.r2 += d1;
      const d = st.r2 * this.drain;
      st.r2 -= d;
      st.phase += d / unit;
      for (let k = 0; st.phase >= st.threshold; k++) {
        // beyond a few events in one sample the rest merge into one (bounded work at any release)
        const merged = k === MAX_EVENTS_PER_SAMPLE - 1 ? st.phase : 1;
        st.phase = merged === 1 ? st.phase - st.threshold : 0;
        st.threshold = this.exponential();
        // fracture: an avalanche of fibre bundles; friction: exponentially distributed energies
        const e = merged * (fracture ? (this.avalanche() / this.meanSize) * unit : this.exponential() * unit);
        this.wavelet(exc, n, i, e * toJ, this.band(cdf), pan + 0.12 * this.gauss());
        fired = true;
      }
    }
    // a negligible remainder (below a millionth of an event) is dropped so the voice can go idle
    if (st.inflow === 0 && st.r1 + st.r2 < 1e-6 * unit) st.r1 = st.r2 = 0;
    return fired;
  }

  /** Render one block into L and R (equal lengths). */
  process(L: Float32Array, R: Float32Array): void {
    const n = L.length;
    if (this.qCount === 0 && !this.active() && this.quiet > RING_OUT * this.sampleRate) {
      L.fill(0);
      R.fill(0);
      this.clock += n;
      this.quiet += n;
      this.waiting = 0;
      return;
    }
    if (this.exc.length < 2 * BANDS * n) this.exc = new Float32Array(2 * BANDS * n);
    const exc = this.exc;
    exc.fill(0, 0, 2 * BANDS * n);
    let excited = false;
    const F = TEAR_VOICE.fracture;
    const fractureUnit = this.meanSize * F.bundle * F.fibreEnergy;
    for (let pos = 0; pos < n; ) {
      if (this.recLeft <= 0 && !this.nextRecord(pos, n)) {
        for (let s = 0; s < SOURCES; s++) this.fracture[s].inflow = this.slip[s].inflow = 0;
      }
      const take = this.recLeft > 0 ? Math.min(n - pos, Math.ceil(this.recLeft)) : n - pos;
      for (let s = 0; s < SOURCES; s++) {
        if (this.run(this.fracture[s], exc, n, pos, pos + take, fractureUnit, true, this.pan[s])) excited = true;
        if (this.run(this.slip[s], exc, n, pos, pos + take, TEAR_VOICE.slip.quantum, false, this.pan[s])) excited = true;
      }
      this.recLeft = Math.max(0, this.recLeft - take);
      pos += take;
    }
    // pull-out impulses due in this block (compacting the pending list)
    const end = this.clock + n;
    let keep = 0;
    for (let k = 0; k < this.pending; k++) {
      const t = this.evT[k];
      if (t >= end) {
        if (keep !== k) {
          this.evT[keep] = t;
          this.evE[keep] = this.evE[k];
          this.evB[keep] = this.evB[k];
          this.evP[keep] = this.evP[k];
        }
        keep++;
        continue;
      }
      this.wavelet(exc, n, Math.max(0, Math.floor(t - this.clock)), this.evE[k], this.evB[k], this.evP[k]);
      excited = true;
    }
    this.pending = keep;
    this.quiet = excited ? 0 : this.quiet + n;
    // band bank
    L.fill(0);
    R.fill(0);
    for (let b = 0; b < BANDS; b++) {
      const { a1, a2, a3 } = this.bands[b];
      for (let ch = 0; ch < 2; ch++) {
        const out = ch === 0 ? L : R;
        const off = (2 * b + ch) * n;
        let s1 = this.s1[2 * b + ch];
        let s2 = this.s2[2 * b + ch];
        for (let i = 0; i < n; i++) {
          const v3 = exc[off + i] - s2;
          const v1 = a1 * s1 + a2 * v3;
          const v2 = s2 + a2 * s1 + a3 * v3;
          s1 = 2 * v1 - s1;
          s2 = 2 * v2 - s2;
          out[i] += v1;
        }
        // flush denormals once the band has rung out
        this.s1[2 * b + ch] = Math.abs(s1) < 1e-15 ? 0 : s1;
        this.s2[2 * b + ch] = Math.abs(s2) < 1e-15 ? 0 : s2;
      }
    }
    // peak limiter (fast attack, slow release) and a soft knee near full scale
    let env = this.env;
    let g = this.limGain;
    for (let i = 0; i < n; i++) {
      const p = Math.max(Math.abs(L[i]), Math.abs(R[i]));
      env = p > env ? p : env * this.release + p * (1 - this.release);
      const target = env > LIMIT ? LIMIT / env : 1;
      g = target < g ? target + (g - target) * this.attack : target + (g - target) * this.release;
      L[i] = softClip(L[i] * g);
      R[i] = softClip(R[i] * g);
    }
    this.env = env;
    this.limGain = g;
    this.clock += n;
  }
}

function softClip(x: number): number {
  const a = Math.abs(x);
  if (a <= 0.8) return x;
  // continuous slope at 0.8, saturating at 1
  const y = 0.8 + 0.2 * Math.tanh((a - 0.8) / 0.2);
  return x < 0 ? -y : y;
}
