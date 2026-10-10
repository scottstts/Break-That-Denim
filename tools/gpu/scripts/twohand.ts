/**
 * Multi-touch session (development diagnostics): several fingers grip the cloth at once through the
 * product's grip slots (interaction.ts) and drag with the worker's handle logic (bounded speed,
 * pinch, drag lift). Checks every frame for non-finite values, reports where the cloth failed
 * relative to the fingers and renders a contact sheet (artifacts/captures/<name>.png).
 *
 *   node tools/gpu/run.mjs tools/gpu/scripts/twohand.ts <script> [name] [every]
 *
 * AUDIO=file.json records the acoustic records of every substep (tools/audio/render.ts).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { ACOUSTIC_SLOTS, RECORD_FLOATS, SOURCES, decodeRecord, fracturePosition } from '../../../src/audio/record';
import { AcousticLog } from '../../../src/engine/acousticLog';
import { readBuffer } from '../../../src/gpu/util';
import { DEFAULT_CAMERA, Renderer } from '../../../src/rendering/renderer';
import { Interaction } from '../../../src/simulation/gpu/interaction';
import { Topology } from '../../../src/simulation/gpu/topology';
import { GRIP, pinchScale } from '../../../src/simulation/grip';
import { CAPTURES, drain, encodePng, gpuDevice } from '../env';
import { settledSolver } from './render';

/** A finger: where it lands (world x, y, m) and its drag (m) over `t` seconds after landing at `start`. */
interface Finger {
  at: [number, number];
  pull: [number, number];
  start: number;
  t: number;
  /** Released this long after it landed (s). */
  hold: number;
}

const SCRIPTS: Record<string, { fingers: Finger[]; length: number }> = {
  // two hands either side of the worn zone pull straight apart, hold, let go
  apart: {
    fingers: [
      { at: [-0.05, 0.0], pull: [-0.06, 0.0], start: 0, t: 0.7, hold: 1.2 },
      { at: [0.05, 0.0], pull: [0.06, 0.0], start: 0.05, t: 0.7, hold: 1.2 },
    ],
    length: 2.4,
  },
  // two hands close together shear the cloth (one up, one down)
  shear: {
    fingers: [
      { at: [-0.02, 0.0], pull: [0.0, 0.07], start: 0, t: 0.7, hold: 1.2 },
      { at: [0.02, 0.0], pull: [0.0, -0.07], start: 0, t: 0.7, hold: 1.2 },
    ],
    length: 2.4,
  },
  // two fingers land 4 mm apart, inside each other's pinch: the second takes only what the first leaves
  overlap: {
    fingers: [
      { at: [0.0, 0.0], pull: [-0.05, 0.0], start: 0, t: 0.6, hold: 1.0 },
      { at: [0.004, 0.0], pull: [0.05, 0.0], start: 0, t: 0.6, hold: 1.0 },
    ],
    length: 1.8,
  },
  // two fingers 12 mm apart: their pinches would overlap, the second keeps the material the first leaves
  adjacent: {
    fingers: [
      { at: [0.0, 0.0], pull: [-0.05, 0.01], start: 0, t: 0.6, hold: 1.0 },
      { at: [0.012, 0.0], pull: [0.05, -0.01], start: 0, t: 0.6, hold: 1.0 },
    ],
    length: 1.8,
  },
  // three fingers: two hold, the third pulls between them
  three: {
    fingers: [
      { at: [-0.06, 0.03], pull: [-0.01, 0.01], start: 0, t: 0.3, hold: 1.6 },
      { at: [0.06, 0.03], pull: [0.01, 0.01], start: 0, t: 0.3, hold: 1.6 },
      { at: [0.0, -0.02], pull: [0.0, -0.08], start: 0.3, t: 0.8, hold: 1.2 },
    ],
    length: 2.4,
  },
};

const H_FRAME = 1 / 30;
/** worker.ts handle logic */
const HANDLE_MAX_SPEED = 2.5;
const ENGAGE_LIFT = 0.002;
const LIFT_PER_DRAG = 0.35;
const MAX_DRAG_LIFT = 0.05;

interface Held {
  finger: Finger;
  slot: number;
  anchor: [number, number, number];
  handle: [number, number, number];
  grabT: number;
}

export default async function main(args: string[]): Promise<void> {
  const scriptName = args[0] ?? 'apart';
  const name = args[1] ?? `twohand-${scriptName}`;
  const every = Number(args[2] ?? 6);
  const script = SCRIPTS[scriptName];
  if (!script) throw new Error(`unknown script ${scriptName}`);
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (pass) => interaction.encodeSubstep(pass);
  solver.preAssemble = (dispatch) => interaction.encode(dispatch);
  const topology = new Topology(device, solver, scene);
  const acoustic = process.env.AUDIO ? new AcousticLog(device, solver.counters) : null;
  const audioRecords: number[][] = [];
  const [W, Hh] = (process.env.SIZE ?? '960x600').split('x').map(Number);
  const format: GPUTextureFormat = 'rgba8unorm';
  const renderer = new Renderer(device, format, solver, scene);
  renderer.resize(W, Hh);
  const tex = device.createTexture({ size: [W, Hh], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const h = solver.params.h;
  const perFrame = Math.round(H_FRAME / h);
  const N = scene.nodeCount;
  let xs = new Float32Array(await readBuffer(device, solver.x));
  const segs0 = new Uint32Array(await readBuffer(device, solver.segs));

  const nearest = (p: [number, number]) => {
    let best = 1e9;
    let at = -1;
    for (let n = 0; n < N; n++) {
      if (scene.pinned[n]) continue;
      const d = Math.hypot(xs[4 * n] - p[0], xs[4 * n + 1] - p[1]);
      if (d < best) [best, at] = [d, n];
    }
    return at;
  };

  const held = new Map<Finger, Held>();
  const done = new Set<Finger>();
  const tiles: Uint8Array[] = [];
  const labels: string[] = [];
  let t = 0;
  let events = 0;
  let bad = 0;
  for (let frame = 0; t < script.length - 1e-9; frame++) {
    // fingers landing and lifting at this frame
    for (const f of script.fingers) {
      if (!held.has(f) && !done.has(f) && t >= f.start) {
        done.add(f);
        const node = nearest(f.at);
        const slot = interaction.freeSlot();
        const taken = interaction.holder(node) >= 0;
        const anchor: [number, number, number] = [xs[4 * node], xs[4 * node + 1], xs[4 * node + 2]];
        const ok = !taken && slot >= 0 && interaction.attach(node, anchor, slot);
        console.log(`  t=${t.toFixed(2)} finger at ${f.at.join(',')}: node ${node}${taken ? ' (already held)' : ''} -> ${ok ? `slot ${slot}, ${interaction.core(slot).length} core nodes` : 'holds nothing'}`);
        if (ok) held.set(f, { finger: f, slot, anchor, handle: [...anchor], grabT: t });
      }
    }
    for (const [f, g] of [...held]) {
      if (t >= f.start + f.hold) {
        interaction.release(g.slot);
        held.delete(f);
        await topology.request(interaction.coreNodes);
        console.log(`  t=${t.toFixed(2)} finger at ${f.at.join(',')} lifts`);
      }
    }
    // sound sources follow the fingers (worker.ts updateAnchors)
    if (held.size > 0) {
      const anchors = new Float32Array(16);
      for (const g of held.values()) anchors.set([...g.handle, 1], 4 * g.slot);
      device.queue.writeBuffer(solver.acousticSources, 0, anchors);
    }
    acoustic?.begin();
    for (let k = 0; k < perFrame; k++) {
      const tk = t + k * h;
      for (const g of held.values()) {
        const f = g.finger;
        const a = Math.min(1, Math.max(0, (tk - f.start) / f.t));
        const target: [number, number, number] = [g.anchor[0] + a * f.pull[0], g.anchor[1] + a * f.pull[1], g.anchor[2]];
        const since = tk - g.grabT;
        const drag = Math.hypot(a * f.pull[0], a * f.pull[1]);
        target[2] += ENGAGE_LIFT * Math.min(1, since / GRIP.engageTime) + Math.min(MAX_DRAG_LIFT, LIFT_PER_DRAG * drag);
        const d = [target[0] - g.handle[0], target[1] - g.handle[1], target[2] - g.handle[2]];
        const len = Math.hypot(d[0], d[1], d[2]);
        const st = HANDLE_MAX_SPEED * h;
        for (let q = 0; q < 3; q++) g.handle[q] = len <= st ? target[q] : g.handle[q] + (d[q] * st) / len;
        interaction.setTarget(g.handle, g.slot);
        interaction.setPinch(pinchScale(since), g.slot);
      }
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc);
      acoustic?.record(enc);
      device.queue.submit([enc.finish()]);
    }
    await drain(device);
    const rec = await acoustic?.end();
    if (rec) {
      const out = new Float32Array(RECORD_FLOATS);
      for (let q = 0; q < rec.count; q++) {
        // pan: the fracture's world x per source (render.ts maps it to the view)
        const pans = Array.from({ length: SOURCES }, (_, src) => fracturePosition(rec.raw, q * ACOUSTIC_SLOTS, src)?.[0] ?? 0);
        decodeRecord(rec.raw, q * ACOUSTIC_SLOTS, out, 0, h, pans);
        audioRecords.push(Array.from(out));
      }
    }
    t += perFrame * h;
    xs = new Float32Array(await readBuffer(device, solver.x));
    const c = new Uint32Array(await readBuffer(device, solver.counters, 64));
    if (c[0] + c[1] !== events) {
      events = c[0] + c[1];
      await topology.request(interaction.coreNodes);
    }
    let nf = 0;
    for (let q = 0; q < 4 * N; q++) if (!Number.isFinite(xs[q])) nf++;
    bad += nf;
    if (frame % every === 0 || nf > 0) {
      console.log(`t=${t.toFixed(2)} ruptures=${c[0]} pulled=${c[1]} rejected=${c[12]} fingers=${held.size} freed=${topology.freedCount} nonfinite=${nf}`);
      const enc = device.createCommandEncoder();
      renderer.encode(enc, { ...DEFAULT_CAMERA, ...(JSON.parse(process.env.CAMERA ?? '{}') as object) }, t, tex.createView());
      device.queue.submit([enc.finish()]);
      tiles.push(await pixels(device, tex));
      labels.push(t.toFixed(2));
    }
  }
  // where the cloth failed: broken spans by their rest position along the line through the first two
  // fingers (between them, or beyond either)
  const segs = new Uint32Array(await readBuffer(device, solver.segs));
  const [A, B] = script.fingers;
  const ax = B.at[0] - A.at[0];
  const ay = B.at[1] - A.at[1];
  const L2 = ax * ax + ay * ay;
  const bins = { between: 0, beyond: 0, far: 0 };
  for (let q = 0; q < 2 * N; q++) {
    if ((segs[8 * q + 5] & 2) === 0 || (segs0[8 * q + 5] & 2) !== 0) continue;
    const n = q >> 1;
    const s = ((scene.position[3 * n] - A.at[0]) * ax + (scene.position[3 * n + 1] - A.at[1]) * ay) / L2;
    const off = Math.abs((scene.position[3 * n] - A.at[0]) * ay - (scene.position[3 * n + 1] - A.at[1]) * ax) / Math.sqrt(L2);
    if (off > 0.04) bins.far++;
    else if (s >= 0 && s <= 1) bins.between++;
    else bins.beyond++;
  }
  console.log(`BROKEN spans: between the first two fingers ${bins.between}, beyond them ${bins.beyond}, > 4 cm off their line ${bins.far}; non-finite values ${bad}`);
  if (acoustic) {
    writeFileSync(resolve(process.env.AUDIO!), JSON.stringify({ h, records: audioRecords }));
    console.log(`acoustic records: ${audioRecords.length} -> ${process.env.AUDIO}`);
  }
  mkdirSync(CAPTURES, { recursive: true });
  const path = resolve(CAPTURES, `${name}.png`);
  writeFileSync(path, sheet(tiles, W, Hh, 3));
  console.log(path, labels.join(' '));
  device.destroy();
}

async function pixels(device: GPUDevice, tex: GPUTexture): Promise<Uint8Array> {
  const w = tex.width;
  const h = tex.height;
  const bpr = Math.ceil((w * 4) / 256) * 256;
  const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
  device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ);
  const src = new Uint8Array(buf.getMappedRange());
  const out = new Uint8Array(4 * w * h);
  for (let y = 0; y < h; y++) out.set(src.subarray(y * bpr, y * bpr + 4 * w), 4 * w * y);
  buf.unmap();
  buf.destroy();
  return out;
}

function sheet(tiles: Uint8Array[], w: number, h: number, cols: number): Uint8Array {
  const rows = Math.ceil(tiles.length / cols);
  const out = new Uint8Array(4 * cols * w * rows * h);
  tiles.forEach((t, k) => {
    const ox = (k % cols) * w;
    const oy = Math.floor(k / cols) * h;
    for (let y = 0; y < h; y++) out.set(t.subarray(4 * w * y, 4 * w * (y + 1)), 4 * ((oy + y) * cols * w + ox));
  });
  return encodePng(cols * w, rows * h, out);
}
