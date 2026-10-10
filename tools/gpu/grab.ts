/**
 * Product grab for headless drivers: a world point is seen through the default camera at a virtual
 * 1600 x 1000 drawing buffer and picked as the worker picks a pointer there (visible material only),
 * then attached with the pick's connectivity window and handed to a Hand with the camera's toward-viewer
 * direction. Scripts then request drags in world metres from the grab point.
 */
import { DEFAULT_CAMERA, cameraMatrices } from '../../src/rendering/renderer';
import { normalize, scale, sub, transformPoint, type Vec3 } from '../../src/rendering/math';
import type { Interaction } from '../../src/simulation/gpu/interaction';
import { Hand } from '../../src/simulation/hand';

const VIEW_W = 1600;
const VIEW_H = 1000;

export interface Grab {
  node: number;
  slot: number;
  anchor: Vec3;
  hand: Hand;
  kind: 'surface' | 'proximity';
}

/** Pixel of world point `p` in the virtual view. */
export function viewPixel(p: Vec3): [number, number] {
  const m = cameraMatrices(DEFAULT_CAMERA, VIEW_W / VIEW_H);
  const c = transformPoint(m.viewProj, p);
  return [(VIEW_W / 2) * (1 + c[0] / c[3]), (VIEW_H / 2) * (1 - c[1] / c[3])];
}

/** Grab what the pointer over world point `p` would grab (null: nothing there, or already held). */
export async function grabAt(interaction: Interaction, p: Vec3, slot = interaction.freeSlot()): Promise<Grab | null> {
  if (slot < 0) return null;
  const m = cameraMatrices(DEFAULT_CAMERA, VIEW_W / VIEW_H);
  const [px, py] = viewPixel(p);
  // the worker's proximity radius at this buffer height
  const hit = await interaction.pickAt(m.viewProj, px, py, VIEW_W, VIEW_H, Math.max(8, 0.016 * VIEW_H));
  if (!hit || interaction.holder(hit.node) >= 0) return null;
  if (!interaction.attach(hit.node, hit.position, slot, hit.window)) return null;
  const toward = scale(normalize(sub(DEFAULT_CAMERA.target, m.eye)), -1);
  return { node: hit.node, slot, anchor: hit.position, hand: new Hand(hit.position, toward), kind: hit.kind };
}
