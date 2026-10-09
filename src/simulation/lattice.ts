/**
 * Lattice multigrid hierarchy shared by the CPU reference and the GPU solver: coarse index sets,
 * bilinear transfer weights, the coarsening stop rule, and yarn-line tables with smoothing colors.
 */

export interface LatticeLayer {
  nx: number;
  ny: number;
  offset: number;
}

/** Fine indices kept by one coarsening: every second index plus the last one. */
export function coarseList(n: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < n - 1; i += 2) out.push(i);
  out.push(n - 1);
  return out;
}

export function coarseCount(n: number): number {
  return Math.ceil((n - 1) / 2) + 1;
}

/** Fine index of coarse index I for a fine extent n. */
export function coarseToFine(I: number, n: number): number {
  return Math.min(2 * I, n - 1);
}

/** Bracketing coarse indices and bilinear weights of fine index i (extent n). */
export function bracket(i: number, n: number): Array<[number, number]> {
  const nc = coarseCount(n);
  let lo = Math.min(nc - 1, i >> 1);
  if (coarseToFine(lo, n) > i) lo--;
  const a = coarseToFine(lo, n);
  if (a === i) return [[lo, 1]];
  const b = coarseToFine(lo + 1, n);
  return [
    [lo, (b - i) / (b - a)],
    [lo + 1, (i - a) / (b - a)],
  ];
}

export function coarsenLayers(fine: LatticeLayer[]): LatticeLayer[] {
  let off = 0;
  return fine.map((L) => {
    const C = { nx: coarseCount(L.nx), ny: coarseCount(L.ny), offset: off };
    off += C.nx * C.ny;
    return C;
  });
}

/** Coarsen while every layer stays large enough for a meaningful coarser lattice. */
export function shouldCoarsen(layers: LatticeLayer[], minNodes = 48): boolean {
  return layers.every((L) => L.nx * L.ny > minNodes && L.nx >= 5 && L.ny >= 5);
}

/** Level layouts: coarsen while a level has more than maxCoarsestNodes nodes (and can be coarsened). */
export function levelLayers(fine: LatticeLayer[], maxCoarsestNodes = 48): LatticeLayer[][] {
  const out = [fine];
  for (;;) {
    const cur = out[out.length - 1];
    if (nodeCount(cur) <= maxCoarsestNodes || !shouldCoarsen(cur)) break;
    out.push(coarsenLayers(cur));
  }
  return out;
}

export function nodeCount(layers: LatticeLayer[]): number {
  return layers.reduce((s, L) => s + L.nx * L.ny, 0);
}

/**
 * Smoothing colors: lines of a family are colored modulo `colors` (3: lines updated together never
 * share a radius-2 stencil coupling; 2: their +-2 couplings are majorized by the smoother). Order:
 * v-lines (colors 0..colors-1), then u-lines.
 */

export interface LineTable {
  /** Per line: start node, stride, length, family. */
  lines: Uint32Array;
  /** colorStart[c]..colorStart[c+1] are the lines of color c. */
  colorStart: number[];
  maxLength: number;
}

export function lineTable(layers: LatticeLayer[], colors: 2 | 3 = 3): LineTable {
  const byColor: number[][][] = Array.from({ length: 2 * colors }, () => []);
  let maxLength = 0;
  for (const L of layers) {
    for (let i = 0; i < L.nx; i++) {
      byColor[i % colors].push([L.offset + i, L.nx, L.ny, 1]);
      maxLength = Math.max(maxLength, L.ny);
    }
    for (let j = 0; j < L.ny; j++) {
      byColor[colors + (j % colors)].push([L.offset + j * L.nx, 1, L.nx, 0]);
      maxLength = Math.max(maxLength, L.nx);
    }
  }
  const colorStart = [0];
  const flat: number[] = [];
  for (const c of byColor) {
    for (const l of c) flat.push(...l);
    colorStart.push(colorStart[colorStart.length - 1] + c.length);
  }
  return { lines: new Uint32Array(flat), colorStart, maxLength };
}
