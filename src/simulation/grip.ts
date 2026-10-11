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
  /**
   * A pinch gathers the cloth between the fingers. Two fingertips hold a pad of cloth either side of
   * the grab point and slide toward each other along one material axis: over the engagement each pad
   * moves in by (1 - pinch) lattice pitches, so its inner edge ends at this fraction of a pitch from the
   * grab line. The strip between the pads is not held; it folds as the mechanics make it fold.
   * ('radial': every patch target contracts toward the grab point to this fraction of its spread.)
   */
  pinch: 0.3,
  pinchMode: 'pads' as 'pads' | 'radial',
  /** Material axis the fingertips close along when the cloth allows (0: u, the warp; 1: v). */
  pinchAxis: 0 as 0 | 1,
  /** Time over which the fingers close and lift the cloth, s. */
  engageTime: 0.15,
};

/** Fraction of the grab-time patch spread kept by the closing pinch, `t` seconds after the grab. */
export function pinchScale(t: number): number {
  const e = Math.min(1, Math.max(0, t / GRIP.engageTime));
  return 1 - (1 - GRIP.pinch) * e * e * (3 - 2 * e);
}

export interface GripPatch {
  nodes: number[];
  weights: number[];
  /** Pad of each entry along the closing axis (-1, +1), or 0 for a radial pinch. */
  sides: number[];
  /** Lattice pitch along the closing axis (m): how far a pad's inner edge starts from the grab line. */
  padPitch: number;
  /** Crossings between the pads (not held, but inside the pinch: no other finger takes them). */
  between: number[];
  /** Stiffness per unit weight (N/m): the summed force reaches GRIP.maxForce at the largest lead. */
  stiffness: number;
}

/**
 * Current connectivity around a picked node (interact.wgsl pickWindow): per lattice offset (di, dj) in
 * [-radius, radius]^2, row-major from (-radius, -radius): bit0 / bit1 = the node's own span along u / v
 * survives, bit2 = not cloth (debris, retired), bit3 = outside the lattice.
 */
export interface PatchWindow {
  radius: number;
  bits: Uint32Array;
}

/** Longest path through surviving spans, in patch radii, by which a patch member may be reached. */
const PATCH_PATH = 1.5;

/** Lattice cells either side of a picked node that a patch window must cover. */
export function patchWindowRadius(s: CompiledScene): number {
  const pitch = Math.min(...s.layers.map((L) => Math.min(L.material.pitchU, L.material.pitchV)));
  return Math.ceil(GRIP.radius / pitch);
}

/**
 * Smoothly weighted material patch around `node` (same layer, free nodes only), limited to nodes
 * `available` accepts (material another finger already holds stays with that finger).
 *
 * With the current connectivity `window`, the patch is what the fingers can actually hold: crossings
 * still cloth and reached from `node` through surviving spans by a path of at most PATCH_PATH radii, so
 * the far lip of a slit or a scrap lying beside the pinch is not bound to it. Intact cloth gives the
 * rest-space disc (every member lies within sqrt(2) radii along the lattice).
 *
 * A pad pinch (GRIP.pinchMode) holds the disc's two halves either side of the grab line, without the
 * line itself.
 */
export function gripPatch(s: CompiledScene, node: number, available: (n: number) => boolean = () => true, window?: PatchWindow): GripPatch | null {
  const disc = patchDisc(s, node, available, window);
  if (!disc || GRIP.pinchMode === 'radial') return disc;
  // two pads either side of the grab line, along the preferred axis if the cloth reaches both sides
  // (a strand along the other axis, a hole rim: the other axis; neither: the radial pinch)
  for (const axis of [GRIP.pinchAxis, 1 - GRIP.pinchAxis]) {
    const pads = splitPads(s, node, disc, axis);
    if (pads) return pads;
  }
  return disc;
}

/** The disc split into two pads either side of the grab line across `axis`, or null if lopsided. */
function splitPads(s: CompiledScene, node: number, disc: GripPatch, axis: number): GripPatch | null {
  const layer = s.layers.find((L) => node >= L.nodeOffset && node < L.nodeOffset + L.nodeCount)!;
  const pitch = axis === 0 ? layer.material.pitchU : layer.material.pitchV;
  const c = s.restUV[2 * node + axis];
  const out: GripPatch = { nodes: [], weights: [], sides: [], stiffness: 0, padPitch: pitch, between: [] };
  const side = [0, 0];
  disc.nodes.forEach((n, k) => {
    const u = s.restUV[2 * n + axis] - c;
    if (Math.abs(u) < 0.5 * pitch) {
      out.between.push(n);
      return;
    }
    out.nodes.push(n);
    out.weights.push(disc.weights[k]);
    out.sides.push(Math.sign(u));
    side[u > 0 ? 1 : 0] += disc.weights[k];
  });
  // the fingers need cloth on both sides: neither pad under a quarter of the other
  if (Math.min(side[0], side[1]) < 0.25 * Math.max(side[0], side[1]) || side[0] === 0) return null;
  out.stiffness = GRIP.maxForce / (GRIP.maxLead * (side[0] + side[1]));
  return out;
}

/** A hold other than a fingertip pinch: claws hooked into the weave (a staged creature's hand). */
export interface HookProfile {
  /** Material radius each claw holds (m): the yarns caught on its curve. */
  radius: number;
  /** Total force bound (N) and the handle's largest lead (m), as GRIP.maxForce / maxLead. */
  maxForce: number;
  maxLead: number;
}

/**
 * Claws hooked into the cloth at several crossings: the union of a smoothly weighted disc of
 * `profile.radius` around each (a node in two discs takes the larger weight). Claws hold rigidly (no
 * pinch). Null when nothing free is left.
 */
export function hookPatch(s: CompiledScene, claws: readonly number[], profile: HookProfile, available: (n: number) => boolean = () => true, window?: PatchWindow): GripPatch | null {
  const weight = new Map<number, number>();
  for (const c of claws) {
    const disc = patchDisc(s, c, available, window, profile.radius);
    disc?.nodes.forEach((n, k) => weight.set(n, Math.max(weight.get(n) ?? 0, disc.weights[k])));
  }
  if (weight.size === 0) return null;
  const nodes = [...weight.keys()];
  const weights = nodes.map((n) => weight.get(n)!);
  const W = weights.reduce((a, b) => a + b, 0);
  return { nodes, weights, sides: nodes.map(() => 0), padPitch: 0, between: [], stiffness: profile.maxForce / (profile.maxLead * W) };
}

/** The smoothly weighted disc around `node` (see gripPatch), with a radial pinch. */
function patchDisc(s: CompiledScene, node: number, available: (n: number) => boolean, window?: PatchWindow, R = GRIP.radius): GripPatch | null {
  const layer = s.layers.find((L) => node >= L.nodeOffset && node < L.nodeOffset + L.nodeCount);
  if (!layer) return null;
  const u0 = s.restUV[2 * node];
  const v0 = s.restUV[2 * node + 1];
  const m = layer.material;
  const reach = window ? reachable(window, m.pitchU, m.pitchV, PATCH_PATH * R) : null;
  const ri = window ? window.radius : Math.ceil(R / m.pitchU);
  const rj = window ? window.radius : Math.ceil(R / m.pitchV);
  const side = 2 * ri + 1;
  const local = node - layer.nodeOffset;
  const i0 = local % layer.nx;
  const j0 = Math.floor(local / layer.nx);
  const nodes: number[] = [];
  const weights: number[] = [];
  let W = 0;
  for (let j = Math.max(0, j0 - rj); j <= Math.min(layer.ny - 1, j0 + rj); j++) {
    for (let i = Math.max(0, i0 - ri); i <= Math.min(layer.nx - 1, i0 + ri); i++) {
      const n = layer.nodeOffset + j * layer.nx + i;
      if (s.pinned[n] || !available(n)) continue;
      if (reach && !reach[(j - j0 + rj) * side + (i - i0 + ri)]) continue;
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
  return { nodes, weights, sides: nodes.map(() => 0), padPitch: 0, between: [], stiffness: GRIP.maxForce / (GRIP.maxLead * W) };
}

/**
 * Window cells reached from its centre through surviving spans within path length `maxPath` (m),
 * over cloth crossings only (Dijkstra on at most 15 x 15 cells).
 */
export function reachable(win: PatchWindow, pitchU: number, pitchV: number, maxPath: number): Uint8Array {
  const side = 2 * win.radius + 1;
  const count = side * side;
  const dist = new Float64Array(count).fill(Infinity);
  const out = new Uint8Array(count);
  const blocked = (q: number) => (win.bits[q] & 12) !== 0;
  const centre = win.radius * side + win.radius;
  if (blocked(centre)) return out;
  dist[centre] = 0;
  for (;;) {
    let k = -1;
    for (let q = 0; q < count; q++) if (!out[q] && dist[q] <= maxPath && (k < 0 || dist[q] < dist[k])) k = q;
    if (k < 0) return out;
    out[k] = 1;
    const i = k % side;
    const j = (k - i) / side;
    const relax = (q: number, len: number) => {
      if (!blocked(q) && !out[q]) dist[q] = Math.min(dist[q], dist[k] + len);
    };
    if (i + 1 < side && win.bits[k] & 1) relax(k + 1, pitchU);
    if (i > 0 && win.bits[k - 1] & 1) relax(k - 1, pitchU);
    if (j + 1 < side && win.bits[k] & 2) relax(k + side, pitchV);
    if (j > 0 && win.bits[k - side] & 2) relax(k - side, pitchV);
  }
}
