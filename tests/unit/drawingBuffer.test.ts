import { describe, expect, it } from 'vitest';
import { computeDrawingBufferSize, MAX_DRAWING_BUFFER_PIXELS } from '../../src/shared/drawingBuffer';

describe('drawing-buffer policy', () => {
  it('caps at 1.7 on small viewports', () => {
    const r = computeDrawingBufferSize(800, 600, 3);
    expect(r.dpr).toBeCloseTo(1.7);
    expect(r.width).toBe(1360);
    expect(r.height).toBe(1020);
  });

  it('never exceeds the pixel budget, including 4K CSS viewports (dpr < 1)', () => {
    for (const [w, h, d] of [
      [3840, 2160, 1],
      [2560, 1440, 2],
      [1054, 1006, 2],
      [5120, 2880, 2],
      [1, 1, 1],
    ]) {
      const r = computeDrawingBufferSize(w, h, d);
      expect(r.width * r.height).toBeLessThanOrEqual(MAX_DRAWING_BUFFER_PIXELS);
      expect(r.width).toBeGreaterThanOrEqual(1);
    }
    expect(computeDrawingBufferSize(3840, 2160, 1).dpr).toBeLessThan(1);
  });

  it('handles invalid inputs', () => {
    const r = computeDrawingBufferSize(Number.NaN, 0, Number.NaN);
    expect(r.width).toBe(1);
    expect(r.height).toBe(1);
  });
});
