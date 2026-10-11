/** Headless loading of the staged creature (development tooling only). */
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { FloorImage } from '../../src/rendering/renderer';
import { Clip } from '../../src/stage/clip';
import type { CreatureImages } from '../../src/stage/creatureRenderer';
import { parseGlb, type GltfImage, type GltfSkinnedMesh } from '../../src/stage/gltf';
import { ROOT } from '../gpu/env';

/** Decode an embedded image to RGBA8 with ImageMagick (the browser decodes with createImageBitmap). */
function decode(img: GltfImage): FloorImage {
  const [width, height] = execFileSync('magick', ['identify', '-format', '%w %h', '-'], { input: img.bytes, encoding: 'utf8' }).trim().split(' ').map(Number);
  const data = execFileSync('magick', ['-', '-depth', '8', 'rgba:-'], { input: img.bytes, maxBuffer: 4 * width * height + 1024 });
  return { width, height, data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) };
}

function read(path: string): ArrayBuffer {
  const buf = readFileSync(resolve(ROOT, path));
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

/** The experience's clip as baked from Blender (public/stage). */
export function loadClip(): Clip {
  return new Clip(read('public/stage/smilie.clip'));
}

export function loadCreature(): { model: GltfSkinnedMesh; images: CreatureImages } {
  const model = parseGlb(read('public/models/smilie.glb'));
  if (!model.baseColor || !model.normalMap || !model.orm) throw new Error('creature material images missing');
  return { model, images: { base: decode(model.baseColor), normal: decode(model.normalMap), orm: decode(model.orm) } };
}
