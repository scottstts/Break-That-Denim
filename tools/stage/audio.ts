/**
 * Offline sound of a staged run (tools/stage/beneath.ts AUDIO=file.json): the tear voice from the
 * acoustic records with the claws' holds as the product applies them (a source sounds only while held),
 * the laugh (whole, at its own level) and the signal-noise bursts at their cues, mixed at the product's levels into a
 * 16-bit stereo WAV.
 *   node tools/gpu/run.mjs tools/stage/audio.ts <run.json> <out.wav>
 */
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { glitchNoise } from '../../src/audio/glitch';
import { RECORD_FLOATS, SOURCES, SRC, sourceAt } from '../../src/audio/record';
import { TearVoice, prepareSource } from '../../src/audio/tearVoice';
import type { StageEvent } from '../../src/stage/beneath';

const FRAME = 1 / 30;
const READBACK = 0.015;
/** World x (m) to pan: the default view spans about 0.35 m. */
const PAN_SPAN = 0.175;
/** As src/audio/audio.ts. */
const EDGE = 0.015;

function decode(file: string, fs: number, channels: number): Float32Array {
  const raw = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', file, '-ar', String(fs), '-ac', String(channels), '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], { maxBuffer: 1 << 28 });
  return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice();
}

export default async function main(args: string[]): Promise<void> {
  const [input, output] = args;
  const fs = 48000;
  const { h, records, holds, cues } = JSON.parse(readFileSync(input, 'utf8')) as { h: number; records: number[][]; holds: [number, number, boolean][]; cues: StageEvent[] };
  const perFrame = Math.round(FRAME / h);
  const voice = new TearVoice(fs, 1, false);
  voice.setSource(prepareSource(decode('public/fabric_tear_sound_sample.mp3', fs, 1), fs));
  // the laugh is never cut: the mix runs until it has finished
  const laugh = decode('public/exprerience_sounds/creepy_laugh.mp3', fs, 2);
  const laughEnd = Math.max(0, ...cues.filter((c) => c.kind === 'laugh').map((c) => c.t + FRAME + laugh.length / 2 / fs));
  const total = Math.max(records.length * h + 1, laughEnd);
  const N = Math.ceil((total * fs) / 128) * 128;
  const L = new Float32Array(N);
  const R = new Float32Array(N);
  let next = 0;
  let hold = 0;
  for (let at = 0; at < N; at += 128) {
    const t = at / fs;
    while (next < records.length && (next / perFrame + 1) * FRAME + READBACK <= t) {
      // holds reach the voice in order with the records (both arrive over the engine's port)
      while (hold < holds.length && holds[hold][0] <= next) {
        voice.grip(holds[hold][1], holds[hold][2]);
        hold++;
      }
      const batch = new Float32Array(RECORD_FLOATS * perFrame);
      let k = 0;
      for (; k < perFrame && next < records.length; k++, next++) {
        batch.set(records[next].slice(0, RECORD_FLOATS), k * RECORD_FLOATS);
        for (let s = 0; s < SOURCES; s++) {
          const o = k * RECORD_FLOATS + sourceAt(s) + SRC.pan;
          batch[o] = Math.max(-1, Math.min(1, batch[o] / PAN_SPAN));
        }
      }
      voice.push(batch.subarray(0, k * RECORD_FLOATS));
    }
    voice.process(L.subarray(at, at + 128), R.subarray(at, at + 128));
  }
  // the staged sounds at their cues (heard a frame after the substep that cued them)
  const place = (start: number, n: number, sample: (i: number, ch: number) => number, gain: number) => {
    const e = Math.round(EDGE * fs);
    for (let i = 0; i < n && start + i < N; i++) {
      const env = gain * Math.min(1, i / e, (n - 1 - i) / e);
      L[start + i] += env * sample(i, 0);
      R[start + i] += env * sample(i, 1);
    }
  };
  for (const c of cues) {
    const start = Math.round((c.t + FRAME) * fs);
    if (c.kind === 'laugh') {
      for (let i = 0; i < laugh.length / 2 && start + i < N; i++) {
        L[start + i] += laugh[2 * i];
        R[start + i] += laugh[2 * i + 1];
      }
    } else if (c.kind === 'glitch') {
      const n = Math.round(c.duration * fs);
      const gl = new Float32Array(n);
      const gr = new Float32Array(n);
      glitchNoise(fs, c.level, gl, gr);
      place(start, n, (i, ch) => (ch === 0 ? gl[i] : gr[i]), 1);
    }
  }
  const wav = Buffer.alloc(44 + 4 * N);
  wav.write('RIFF', 0);
  wav.writeUInt32LE(36 + 4 * N, 4);
  wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(fs, 24);
  wav.writeUInt32LE(fs * 4, 28);
  wav.writeUInt16LE(4, 32);
  wav.writeUInt16LE(16, 34);
  wav.write('data', 36);
  wav.writeUInt32LE(4 * N, 40);
  let peak = 0;
  for (let i = 0; i < N; i++) {
    peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
    wav.writeInt16LE(Math.round(Math.max(-1, Math.min(1, L[i])) * 32767), 44 + 4 * i);
    wav.writeInt16LE(Math.round(Math.max(-1, Math.min(1, R[i])) * 32767), 46 + 4 * i);
  }
  writeFileSync(output, wav);
  console.log(`${output}: ${(N / fs).toFixed(2)} s, peak ${(20 * Math.log10(peak || 1e-9)).toFixed(1)} dBFS`);
}
