import { BENEATH_CLIP_URL } from './beneath';
import { Clip } from './clip';
import type { CreatureImages } from './creatureRenderer';
import { parseGlb, type GltfImage, type GltfSkinnedMesh } from './gltf';

/** The staged creature's model (public/models; rigged in blender/smilie). */
export const CREATURE_URL = '/models/smilie.glb';

/** Fetch and decode the creature in a worker or page: the mesh and its material images, as stored. */
export async function fetchCreature(base: string): Promise<{ model: GltfSkinnedMesh; images: CreatureImages }> {
  const r = await fetch(new URL(CREATURE_URL, base));
  if (!r.ok) throw new Error(`creature model unavailable (${r.status})`);
  const model = parseGlb(await r.arrayBuffer());
  const decode = (img: GltfImage | null) => {
    if (!img) throw new Error('creature material image missing');
    return createImageBitmap(new Blob([img.bytes as Uint8Array<ArrayBuffer>], { type: img.mime }), { colorSpaceConversion: 'none', premultiplyAlpha: 'none' });
  };
  const [base_, normal, orm] = await Promise.all([decode(model.baseColor), decode(model.normalMap), decode(model.orm)]);
  return { model, images: { base: base_, normal, orm } };
}

/** Fetch the authored clip of "Beneath" (camera, lights, film, the creature's performance, events). */
export async function fetchBeneathClip(base: string): Promise<Clip> {
  const r = await fetch(new URL(BENEATH_CLIP_URL, base));
  if (!r.ok) throw new Error(`experience clip unavailable (${r.status})`);
  return new Clip(await r.arrayBuffer());
}
