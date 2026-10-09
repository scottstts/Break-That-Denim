/** Hard cap on drawing-buffer pixels (project policy). */
export const MAX_DRAWING_BUFFER_PIXELS = 2_000_000;
/** Upper bound on the device pixel ratio actually used. */
export const MAX_DPR = 1.7;

export interface DrawingBufferSize {
  /** Effective DPR; may be below 1 when the CSS viewport alone exceeds the pixel budget. */
  dpr: number;
  width: number;
  height: number;
}

/**
 * dpr = min(devicePixelRatio, 1.7, sqrt(maxPixels / (cssW * cssH))).
 * There is deliberately no final max(1, dpr) floor: that would disable the cap on large viewports.
 * Final integer dimensions are rounded down and then shrunk until the product respects the cap.
 */
export function computeDrawingBufferSize(
  cssWidth: number,
  cssHeight: number,
  devicePixelRatio: number,
  maxPixels = MAX_DRAWING_BUFFER_PIXELS,
): DrawingBufferSize {
  const width = Math.max(1, Number.isFinite(cssWidth) ? cssWidth : 1);
  const height = Math.max(1, Number.isFinite(cssHeight) ? cssHeight : 1);
  const deviceDpr = Number.isFinite(devicePixelRatio) && devicePixelRatio > 0 ? devicePixelRatio : 1;
  let dpr = Math.min(deviceDpr, MAX_DPR, Math.sqrt(maxPixels / (width * height)));
  if (!(Number.isFinite(dpr) && dpr > 0)) dpr = 1;

  let w = Math.max(1, Math.floor(width * dpr));
  let h = Math.max(1, Math.floor(height * dpr));
  // Conservative rounding guard: never exceed the budget because of the 1-pixel minimums.
  while (w * h > maxPixels && (w > 1 || h > 1)) {
    if (w >= h && w > 1) w--;
    else h--;
  }
  return { dpr, width: w, height: h };
}
