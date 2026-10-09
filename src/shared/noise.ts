/** Deterministic integer hashing and smooth noise for asset compilation (seeded, platform-independent). */

/** 32-bit integer hash (lowbias32). Returns an unsigned 32-bit integer. */
export function hash32(x: number): number {
  x = x >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

export function hash3(a: number, b: number, c: number): number {
  return hash32(a ^ hash32(b ^ hash32(c)));
}

/** Uniform in [0, 1). */
export function rand01(a: number, b = 0, c = 0): number {
  return hash3(a, b, c) / 4294967296;
}

/** Uniform in [-1, 1). */
export function randSigned(a: number, b = 0, c = 0): number {
  return rand01(a, b, c) * 2 - 1;
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function grad(seed: number, ix: number, iy: number, fx: number, fy: number): number {
  const h = hash3(seed, ix, iy);
  const ang = (h / 4294967296) * Math.PI * 2;
  return Math.cos(ang) * fx + Math.sin(ang) * fy;
}

/** 2D gradient noise, approximately in [-0.7, 0.7]. */
export function gradientNoise(seed: number, x: number, y: number): number {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;
  const u = fade(fx);
  const v = fade(fy);
  const n00 = grad(seed, ix, iy, fx, fy);
  const n10 = grad(seed, ix + 1, iy, fx - 1, fy);
  const n01 = grad(seed, ix, iy + 1, fx, fy - 1);
  const n11 = grad(seed, ix + 1, iy + 1, fx - 1, fy - 1);
  const a = n00 + (n10 - n00) * u;
  const b = n01 + (n11 - n01) * u;
  return a + (b - a) * v;
}

/** Fractal sum of gradient noise; result roughly in [-1, 1]. */
export function fbm(seed: number, x: number, y: number, octaves: number, lacunarity = 2.03, gain = 0.5): number {
  let sum = 0;
  let amp = 1;
  let norm = 0;
  let fx = x;
  let fy = y;
  for (let o = 0; o < octaves; o++) {
    sum += amp * gradientNoise(seed + o * 1013, fx, fy);
    norm += amp;
    amp *= gain;
    fx *= lacunarity;
    fy *= lacunarity;
  }
  return (sum / norm) * 1.6;
}

/** 1D smooth noise along a yarn (for per-yarn correlated variation). */
export function noise1(seed: number, x: number): number {
  return gradientNoise(seed, x, 0.5 + (seed % 7) * 0.13);
}

export function smoothstep(e0: number, e1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}
