import type { CompiledScene } from './scene';

/**
 * Physical grip parameters (SI). The grip is a quadratic spring from each patch node to the handle
 * (exact for Newton). Its force is bounded by limiting how far the handle may lead the gripped
 * material: |handle - patch centroid| <= maxLead, so the total force stays below maxForce.
 */
export const GRIP = {
  /**
   * Material radius of the gripped patch, m: a fingertip pinch. A wider patch spreads the pull so
   * thinly that weak material far along the load path failed first, away from the hand.
   */
  radius: 0.008,
  /**
   * Maximum total grip force, N: a hard two-finger pull. It opens worn cloth; intact denim (about 50 N
   * per pick bundle, the pull shared over the pinch) holds unless the pinch sits on a torn edge.
   */
  maxForce: 120,
  /** Largest lead of the handle over the patch centroid, m. */
  maxLead: 0.006,
};

export interface GripPatch {
  nodes: number[];
  weights: number[];
  /** Stiffness per unit weight (N/m): the summed force reaches GRIP.maxForce at the largest lead. */
  stiffness: number;
}

/** Smoothly weighted material patch around `node` (same layer, free nodes only). */
export function gripPatch(s: CompiledScene, node: number): GripPatch | null {
  const layer = s.layers.find((L) => node >= L.nodeOffset && node < L.nodeOffset + L.nodeCount);
  if (!layer) return null;
  const u0 = s.restUV[2 * node];
  const v0 = s.restUV[2 * node + 1];
  const R = GRIP.radius;
  const m = layer.material;
  const ri = Math.ceil(R / m.pitchU);
  const rj = Math.ceil(R / m.pitchV);
  const local = node - layer.nodeOffset;
  const i0 = local % layer.nx;
  const j0 = Math.floor(local / layer.nx);
  const nodes: number[] = [];
  const weights: number[] = [];
  let W = 0;
  for (let j = Math.max(0, j0 - rj); j <= Math.min(layer.ny - 1, j0 + rj); j++) {
    for (let i = Math.max(0, i0 - ri); i <= Math.min(layer.nx - 1, i0 + ri); i++) {
      const n = layer.nodeOffset + j * layer.nx + i;
      if (s.pinned[n]) continue;
      const d = Math.hypot(s.restUV[2 * n] - u0, s.restUV[2 * n + 1] - v0) / R;
      if (d >= 1) continue;
      // smooth falloff: full weight in the core, tapering to zero at the rim
      const w = d < 0.5 ? 1 : 1 - (d - 0.5) / 0.5;
      const ws = w * w * (3 - 2 * w);
      if (ws <= 1e-3) continue;
      nodes.push(n);
      weights.push(ws);
      W += ws;
    }
  }
  if (nodes.length === 0) return null;
  return { nodes, weights, stiffness: GRIP.maxForce / (GRIP.maxLead * W) };
}
