/** Step of a loudness envelope (s). */
export const ENVELOPE_STEP = 0.05;

/**
 * Loudness of a recording every ENVELOPE_STEP seconds, 0..1: the RMS over all channels in dB, from -60
 * (0) to -30 dBFS and above (1). What moves a jaw with a voice.
 */
export function loudness(channels: readonly Float32Array[], rate: number): Float32Array {
  const step = Math.round(ENVELOPE_STEP * rate);
  const n = Math.ceil(channels[0].length / step);
  const out = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    let sum = 0;
    let count = 0;
    for (const c of channels) {
      const end = Math.min(c.length, (k + 1) * step);
      for (let i = k * step; i < end; i++) sum += c[i] * c[i];
      count += end - k * step;
    }
    const db = 10 * Math.log10(sum / Math.max(1, count) + 1e-12);
    out[k] = Math.min(1, Math.max(0, (db + 60) / 30));
  }
  return out;
}

/** The envelope's value `t` seconds into the recording (0 outside it). */
export function loudnessAt(envelope: Float32Array, t: number): number {
  const k = Math.floor(t / ENVELOPE_STEP);
  return k >= 0 && k < envelope.length ? envelope[k] : 0;
}
