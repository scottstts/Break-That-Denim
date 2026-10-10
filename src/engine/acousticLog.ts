import { ACOUSTIC_SLOT, ACOUSTIC_SLOTS } from '../audio/record';

const BLOCK = 4 * ACOUSTIC_SLOTS;
/** Substep records one frame can hold; further substeps of the frame accumulate into the last. */
const CAPACITY = 16;
/** Readbacks in flight; when all are busy, substeps keep accumulating until one is free. */
const STAGING = 3;

/**
 * Per-substep acoustic accumulators (record.ts) gathered on the GPU and read back once per frame
 * without stalling: after each substep its block of the counters buffer is copied into a log slot
 * and cleared; at the end of the frame the log is copied into a free staging buffer and mapped.
 * While no staging buffer is free nothing is copied or cleared, so the counters keep accumulating
 * and the next recorded substep carries them (time-smeared, never lost).
 */
export class AcousticLog {
  private readonly device: GPUDevice;
  private readonly counters: GPUBuffer;
  private readonly log: GPUBuffer;
  private readonly free: GPUBuffer[] = [];
  private staging: GPUBuffer | null = null;
  private count = 0;

  constructor(device: GPUDevice, counters: GPUBuffer) {
    this.device = device;
    this.counters = counters;
    this.log = device.createBuffer({ label: 'acousticLog', size: BLOCK * CAPACITY, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    for (let k = 0; k < STAGING; k++) this.free.push(device.createBuffer({ label: `acousticStaging${k}`, size: BLOCK * CAPACITY, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
  }

  /** Start a frame's records; false when every readback is still in flight (nothing is recorded). */
  begin(): boolean {
    this.count = 0;
    this.staging = this.free.pop() ?? null;
    return this.staging !== null;
  }

  /** After a substep's work in the same encoder: move its accumulators into the next log slot. */
  record(enc: GPUCommandEncoder): void {
    // beyond the capacity the accumulators carry over into the next frame's first record
    if (!this.staging || this.count === CAPACITY) return;
    enc.copyBufferToBuffer(this.counters, 4 * ACOUSTIC_SLOT, this.log, BLOCK * this.count, BLOCK);
    enc.clearBuffer(this.counters, 4 * ACOUSTIC_SLOT, BLOCK);
    this.count++;
  }

  /**
   * Submit the readback of this frame's records. Resolves with `count` blocks of ACOUSTIC_SLOTS u32
   * once the GPU has finished them, or null when nothing was recorded.
   */
  end(): Promise<{ raw: Uint32Array; count: number } | null> {
    const staging = this.staging;
    const count = this.count;
    this.staging = null;
    this.count = 0;
    if (!staging) return Promise.resolve(null);
    if (count === 0) {
      this.free.push(staging);
      return Promise.resolve(null);
    }
    const enc = this.device.createCommandEncoder({ label: 'acousticReadback' });
    enc.copyBufferToBuffer(this.log, 0, staging, 0, BLOCK * count);
    this.device.queue.submit([enc.finish()]);
    return staging.mapAsync(GPUMapMode.READ, 0, BLOCK * count).then(
      () => {
        const raw = new Uint32Array(staging.getMappedRange(0, BLOCK * count).slice(0));
        staging.unmap();
        this.free.push(staging);
        return { raw, count };
      },
      () => {
        this.free.push(staging);
        return null;
      },
    );
  }
}
