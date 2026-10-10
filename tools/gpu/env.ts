/**
 * Headless GPU environment helpers (Node + Dawn, see run.mjs). Development tooling only.
 *
 * GPU work is submitted in short chunks with idle gaps so a long experiment never monopolizes the
 * GPU the desktop compositor also needs.
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { deflateSync } from 'node:zlib';
import { acquireGpu } from '../../src/gpu/device';
import type { FloorImage, FloorImages } from '../../src/rendering/renderer';
import { defaultSceneSpecs, type LayerSpec } from '../../src/simulation/scene';

export const ROOT = resolve(import.meta.dirname, '../..');
export const CAPTURES = resolve(ROOT, 'artifacts/captures');

export async function gpuDevice(): Promise<GPUDevice> {
  const { device } = await acquireGpu(navigator.gpu);
  device.addEventListener('uncapturederror', (e) => console.error('GPU', (e as GPUUncapturedErrorEvent).error.message));
  return device;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Wait for queued work, then idle for as long as the work took (≤50% GPU duty cycle).
 * Returns the measured wall time of the queued work in ms.
 */
export async function drain(device: GPUDevice, since = performance.now()): Promise<number> {
  await device.queue.onSubmittedWorkDone();
  const ms = performance.now() - since;
  await sleep(Math.min(250, Math.max(2, ms)));
  return ms;
}

const crcTable = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** Encode 8-bit RGBA pixels (row-major, top row first) as PNG. */
export function encodePng(width: number, height: number, rgba: Uint8Array): Uint8Array {
  const raw = new Uint8Array(height * (1 + 3 * width));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + 3 * width)] = 0;
    for (let x = 0; x < width; x++) {
      const s = 4 * (y * width + x);
      const d = y * (1 + 3 * width) + 1 + 3 * x;
      raw[d] = rgba[s];
      raw[d + 1] = rgba[s + 1];
      raw[d + 2] = rgba[s + 2];
    }
  }
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr.set([8, 2, 0, 0, 0], 8);
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 6 })), chunk('IEND', new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/** Read an rgba8unorm/bgra8unorm texture and write it as a PNG under artifacts/captures. */
export async function saveTexture(device: GPUDevice, tex: GPUTexture, name: string): Promise<string> {
  const w = tex.width;
  const h = tex.height;
  const bpr = Math.ceil((w * 4) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange());
  const rgba = new Uint8Array(4 * w * h);
  const bgra = tex.format === 'bgra8unorm';
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * bpr + 4 * x;
      const o = 4 * (y * w + x);
      rgba[o] = src[i + (bgra ? 2 : 0)];
      rgba[o + 1] = src[i + 1];
      rgba[o + 2] = src[i + (bgra ? 0 : 2)];
      rgba[o + 3] = 255;
    }
  }
  buf.unmap();
  buf.destroy();
  const path = resolve(CAPTURES, `${name}.png`);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, encodePng(w, h, rgba));
  return path;
}

/**
 * The default scene's specs with development overrides: SCENE (JSON) is merged into the first layer's
 * spec (wear fields merged into its wear spec). Experiments only; the product uses defaultSceneSpecs.
 */
export function experimentSpecs(): LayerSpec[] {
  const specs = defaultSceneSpecs();
  const o = JSON.parse(process.env.SCENE ?? '{}') as Partial<LayerSpec> & { wear?: Partial<LayerSpec['wear']> };
  const { wear, ...rest } = o;
  Object.assign(specs[0], rest);
  if (wear) Object.assign(specs[0].wear, wear);
  return specs;
}

/** Drape file for the experiment scene (DRAPE, default public/drape.bin). */
export function drapeFile(): string {
  return resolve(ROOT, process.env.DRAPE ?? 'public/drape.bin');
}

/** Decode an image to RGBA8 with ImageMagick (the browser decodes with createImageBitmap). */
function decodeImage(file: string): FloorImage {
  const [width, height] = execFileSync('magick', ['identify', '-format', '%w %h', file], { encoding: 'utf8' }).trim().split(' ').map(Number);
  const data = execFileSync('magick', [file, '-depth', '8', 'rgba:-'], { maxBuffer: 4 * width * height + 1024 });
  return { width, height, data: new Uint8Array(data.buffer, data.byteOffset, data.byteLength) };
}

/** The app's floor material set (public/texture), or none with FLOOR=none. */
export function floorImages(): FloorImages | null {
  if (process.env.FLOOR === 'none') return null;
  const at = (name: string) => decodeImage(resolve(ROOT, 'public/texture', name));
  return { base: at('floor_baseColor.jpg'), normal: at('floor_normal.png'), roughness: at('floor_roughness.jpg'), ao: at('floor_AO.jpg') };
}
