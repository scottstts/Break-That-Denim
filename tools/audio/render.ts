/**
 * Offline rendering of the tear voice from recorded acoustic records (session.ts AUDIO=file.json):
 * records arrive in frame batches as in the product (one batch per 1/30 s, after a readback delay)
 * and the voice renders them in 128-sample blocks into a 16-bit stereo WAV. The recording that carries
 * the sound is decoded with ffmpeg at the output rate, as the browser decodes it.
 *
 *   node tools/gpu/run.mjs tools/audio/render.ts <records.json> <out.wav> [sampleRate]
 *
 * Compare with the recording: node tools/audio/analyze.mjs <out.wav>
 * VOICE='{"tail":0.4}' overrides TEAR_SAMPLE constants.
 */
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { RECORD_FLOATS, SOURCES, SRC, sourceAt } from '../../src/audio/record';
import { TEAR_SAMPLE, TearVoice, prepareSource } from '../../src/audio/tearVoice';

const FRAME = 1 / 30;
const READBACK = 0.015;
/** World x (m) to pan: the default view spans about 0.35 m. */
const PAN_SPAN = 0.175;

function merge(into: Record<string, unknown>, from: Record<string, unknown>): void {
  for (const [k, v] of Object.entries(from)) {
    if (v && typeof v === 'object' && !Array.isArray(v)) merge(into[k] as Record<string, unknown>, v as Record<string, unknown>);
    else into[k] = v;
  }
}

export default async function main(args: string[]): Promise<void> {
  merge(TEAR_SAMPLE as unknown as Record<string, unknown>, JSON.parse(process.env.VOICE ?? '{}') as Record<string, unknown>);
  const [input, output] = args;
  const fs = Number(args[2] ?? 48000);
  const { h, records } = JSON.parse(readFileSync(input, 'utf8')) as { h: number; records: number[][] };
  const perFrame = Math.round(FRAME / h);
  const J = Number(process.env.JITTER ?? 0);
  const jitter = (frame: number) => J * ((Math.sin(frame * 12.9898) * 43758.5453) % 1 + 1) % 1;
  const voice = new TearVoice(fs, Number(process.env.SEED ?? 1));
  const raw = execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', 'public/fabric_tear_sound_sample.mp3', '-ar', String(fs), '-ac', '1', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'], { maxBuffer: 1 << 28 });
  voice.setSource(prepareSource(new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice(), fs));
  const total = records.length * h + 0.5;
  const N = Math.ceil((total * fs) / 128) * 128;
  const L = new Float32Array(N);
  const R = new Float32Array(N);
  let next = 0;
  for (let at = 0; at < N; at += 128) {
    const t = at / fs;
    // JITTER=s delays each batch by up to that much more (uneven frames and readbacks in the browser)
    while (next < records.length && (next / perFrame + 1) * FRAME + READBACK + jitter(next / perFrame) <= t) {
      const batch = new Float32Array(RECORD_FLOATS * perFrame);
      let k = 0;
      for (; k < perFrame && next < records.length; k++, next++) {
        batch.set(records[next].slice(0, RECORD_FLOATS), k * RECORD_FLOATS);
        // the tools record world x as the pan
        for (let s = 0; s < SOURCES; s++) {
          const o = k * RECORD_FLOATS + sourceAt(s) + SRC.pan;
          batch[o] = Math.max(-1, Math.min(1, batch[o] / PAN_SPAN));
        }
      }
      voice.push(batch.subarray(0, k * RECORD_FLOATS));
    }
    voice.process(L.subarray(at, at + 128), R.subarray(at, at + 128));
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
  console.log(`${output}: ${(N / fs).toFixed(2)} s, peak ${(20 * Math.log10(peak + 1e-12)).toFixed(1)} dBFS`);
}
