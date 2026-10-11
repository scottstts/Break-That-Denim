import type { CompiledScene } from './scene';

/** Material (rest) coordinates of world point (x, y) on layer `l`'s mount map (inverse of compileLayer's map). */
export function materialAt(scene: CompiledScene, x: number, y: number, l = 0): [number, number] {
  const L = scene.layers[l];
  const [a, b, c, d] = L.mapA;
  const det = a * d - b * c;
  const px = x - L.mapOrigin[0];
  const py = y - L.mapOrigin[1];
  return [L.restCenter[0] + (d * px - b * py) / det, L.restCenter[1] + (-c * px + a * py) / det];
}

/** Lattice node of layer `l` nearest to material point (u, v). */
export function nodeAt(scene: CompiledScene, u: number, v: number, l = 0): number {
  const L = scene.layers[l];
  const i = Math.min(L.nx - 1, Math.max(0, Math.round(u / L.material.pitchU)));
  const j = Math.min(L.ny - 1, Math.max(0, Math.round(v / L.material.pitchV)));
  return L.nodeOffset + j * L.nx + i;
}
