/**
 * Acoustic records: what the mechanics released audibly during one substep, in physical units, as
 * the engine sends them to the tear voice (one run of RECORD_FLOATS per substep): the substep's
 * duration, then one block per sound source (one per finger; common.wgsl nearestSource).
 *
 * GPU accumulation: common.wgsl ACOUSTIC (slots of the solver's counters buffer).
 */

/** Sound sources: one per grip slot (interaction.ts MAX_GRIPS). */
export const SOURCES = 4;
/** u32 slots per source in the counters buffer (common.wgsl ACOUSTIC_STRIDE). */
export const SOURCE_SLOTS = 8;
/** First u32 slot of the acoustic accumulators in the solver's counters buffer. */
export const ACOUSTIC_SLOT = 16;
/** u32 slots reserved for them (copied and cleared as one block per substep). */
export const ACOUSTIC_SLOTS = SOURCES * SOURCE_SLOTS;

/** Offsets within a source's block of a record. */
export const SRC = {
  /** Strain energy released by fibre fracture (J/m). */
  fracture: 0,
  /** Yarn ends pulled out of a crossing (count). */
  pullouts: 1,
  /** Their friction work (J). */
  pulloutWork: 2,
  /** Friction work of crossing slip times its slip speed (J m/s): what drives rubbing noise. */
  slipRasp: 3,
  /** Horizontal screen position of the fracture, -1 (left) .. 1 (right). */
  pan: 4,
} as const;
export const SOURCE_FLOATS = 5;
/** Record: [duration (wall-clock s the substep is presented over), source 0 block, source 1 block, ...]. */
export const RECORD_FLOATS = 1 + SOURCES * SOURCE_FLOATS;

/** Offset of source s's block within a record. */
export function sourceAt(s: number): number {
  return 1 + s * SOURCE_FLOATS;
}

/** Fracture-weighted world position of source s in one substep (m), or null when nothing fractured. */
export function fracturePosition(raw: Uint32Array, base: number, s: number): [number, number, number] | null {
  const b = base + s * SOURCE_SLOTS;
  const w = raw[b + 7];
  if (w === 0) return null;
  const i = new Int32Array(raw.buffer, raw.byteOffset + 4 * (b + 4), 3);
  return [i[0] / (1000 * w), i[1] / (1000 * w), i[2] / (1000 * w)];
}

/**
 * Decode one substep's accumulators (ACOUSTIC_SLOTS u32 at `base`) into a record at `out[o]`, with each
 * source's pan from `pans`.
 */
export function decodeRecord(raw: Uint32Array, base: number, out: Float32Array, o: number, duration: number, pans: ArrayLike<number>): void {
  out[o] = duration;
  for (let s = 0; s < SOURCES; s++) {
    const b = base + s * SOURCE_SLOTS;
    const r = o + sourceAt(s);
    out[r + SRC.fracture] = raw[b] * 1e-5;
    out[r + SRC.pullouts] = raw[b + 1];
    out[r + SRC.pulloutWork] = raw[b + 2] * 1e-8;
    out[r + SRC.slipRasp] = raw[b + 3] * 1e-11;
    out[r + SRC.pan] = pans[s];
  }
}
