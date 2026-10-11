import type { V } from './rig';

/**
 * A staged experience authored in Blender (blender/smilie: animate.py, export.py) and baked per frame:
 * the creature's joints in glTF terms, the camera, the lights, the film, and timeline events. Sampled
 * on the simulation clock (linear between frames; joint rotations normalised-lerped on the shorter arc).
 *
 * File: u32 magic 'CLIP', u32 header length, header JSON (padded to 4 bytes), then frames x channel
 * floats (f32), then frames x joints x 4 rotation components (i16, unit = 32767).
 */

const MAGIC = 0x50494c43;

export interface ClipEvent {
  t: number;
  /** Marker name before its first space ("hook.R", "laugh", ...) and the numbers after it. */
  name: string;
  args: number[];
}

/** The jaw: its joint, opening axis in that joint's frame, radians per unit of opening. */
export interface ClipJaw {
  joint: string;
  axis: [number, number, number];
  rad: number;
}

interface ClipHeader {
  version: number;
  fps: number;
  frames: number;
  joints: string[];
  bindT: number[][];
  root: number;
  skeleton: number[];
  channels: { name: string; size: number }[];
  events: ClipEvent[];
  jaw: ClipJaw;
}

/** Everything a frame of the clip says, at one time (channel arrays are reused between samples). */
export interface ClipSample {
  /** Joint local rotations [x, y, z, w] (glTF order of the clip's joints) and the root's local translation. */
  rotations: Float64Array;
  rootT: V;
  eye: V;
  target: V;
  up: V;
  fovY: number;
  /** The bulb (a spot): position, unit aim, cosines of its outer and inner cone, radiant intensity, colour. */
  bulbPos: V;
  bulbAim: V;
  bulbCone: [number, number];
  bulbI: number;
  bulbColor: V;
  /** The camera's light: position, unit aim, cosines of its outer and inner cone, radiant intensity, colour. */
  torchPos: V;
  torchAim: V;
  torchCone: [number, number];
  torchI: number;
  torchColor: V;
  /** The pit glow below the cloth: position, radiant intensity per channel. */
  pitPos: V;
  pitI: V;
  /** The room's overhead lamp (its irradiance) and the ambient fill, relative to the product's. */
  over: number;
  ambient: number;
  exposure: number;
  grade: number;
  grain: number;
  vignette: number;
  glitch: number;
  /** The picture holds (a frozen frame of the broken signal). */
  freeze: boolean;
  /** The jaw's keyed opening (0..1) and how far the laugh's loudness (0..1) opens it further. */
  jawOpen: number;
  jawLaugh: number;
}

export class Clip {
  readonly fps: number;
  readonly frames: number;
  readonly duration: number;
  readonly joints: string[];
  /** Bind-pose local translation of every joint (the clip keeps them; the root's moves). */
  readonly bindT: number[][];
  readonly root: number;
  /** World matrix of the skeleton's parent space (the glTF space the joints' locals are in), column-major. */
  readonly skeleton: Float64Array;
  readonly events: ClipEvent[];
  readonly jaw: ClipJaw;
  private readonly data: Float32Array;
  private readonly quats: Int16Array;
  private readonly width: number;
  private readonly offset: Record<string, number> = {};
  private readonly out: ClipSample;

  constructor(buffer: ArrayBuffer) {
    const head = new DataView(buffer);
    if (head.getUint32(0, true) !== MAGIC) throw new Error('not a clip');
    const len = head.getUint32(4, true);
    const h = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, len))) as ClipHeader;
    if (h.version !== 2) throw new Error(`clip version ${h.version}`);
    this.fps = h.fps;
    this.frames = h.frames;
    this.duration = (h.frames - 1) / h.fps;
    this.joints = h.joints;
    this.bindT = h.bindT;
    this.root = h.root;
    this.skeleton = Float64Array.from(h.skeleton);
    this.events = h.events;
    this.jaw = h.jaw;
    let w = 0;
    for (const c of h.channels) {
      this.offset[c.name] = w;
      w += c.size;
    }
    this.width = w;
    const start = 8 + len;
    this.data = new Float32Array(buffer, start, h.frames * w);
    this.quats = new Int16Array(buffer, start + 4 * h.frames * w, h.frames * h.joints.length * 4);
    this.out = {
      rotations: new Float64Array(4 * h.joints.length),
      rootT: [0, 0, 0],
      eye: [0, 0, 0],
      target: [0, 0, 0],
      up: [0, 1, 0],
      fovY: 0,
      bulbPos: [0, 0, 0],
      bulbAim: [0, 0, -1],
      bulbCone: [0, 1],
      bulbI: 0,
      bulbColor: [1, 1, 1],
      torchPos: [0, 0, 0],
      torchAim: [0, 0, -1],
      torchCone: [0, 1],
      torchI: 0,
      torchColor: [1, 1, 1],
      pitPos: [0, 0, 0],
      pitI: [0, 0, 0],
      over: 0,
      ambient: 1,
      exposure: 1,
      grade: 0,
      grain: 0,
      vignette: 0,
      glitch: 0,
      freeze: false,
      jawOpen: 0,
      jawLaugh: 0,
    };
  }

  /** The clip at time t (s), clamped to its range. The returned object is reused by the next call. */
  sample(t: number): ClipSample {
    const x = Math.min(Math.max(t * this.fps, 0), this.frames - 1);
    const f0 = Math.floor(x);
    const f1 = Math.min(f0 + 1, this.frames - 1);
    const u = x - f0;
    const s = this.out;
    const ch = (name: string, k = 0) => {
      const o = this.offset[name] + k;
      const a = this.data[f0 * this.width + o];
      return a + (this.data[f1 * this.width + o] - a) * u;
    };
    const v = (name: string, out: number[]) => {
      for (let k = 0; k < out.length; k++) out[k] = ch(name, k);
    };
    v('rootT', s.rootT);
    v('eye', s.eye);
    v('target', s.target);
    v('up', s.up);
    s.fovY = ch('fovY');
    v('bulbPos', s.bulbPos);
    v('bulbAim', s.bulbAim);
    v('bulbCone', s.bulbCone);
    s.bulbI = ch('bulbI');
    v('bulbColor', s.bulbColor);
    v('torchPos', s.torchPos);
    v('torchAim', s.torchAim);
    v('torchCone', s.torchCone);
    s.torchI = ch('torchI');
    v('torchColor', s.torchColor);
    v('pitPos', s.pitPos);
    v('pitI', s.pitI);
    s.over = ch('over');
    s.ambient = ch('ambient');
    s.exposure = ch('exposure');
    s.grade = ch('grade');
    s.grain = ch('grain');
    s.vignette = ch('vignette');
    s.jawOpen = ch('jawOpen');
    s.jawLaugh = ch('jawLaugh');
    // the glitch's look changes from frame to frame (no blending between corrupted pictures)
    s.glitch = this.data[Math.round(x) * this.width + this.offset.glitch];
    s.freeze = this.data[f0 * this.width + this.offset.freeze] >= 0.5;
    const J = this.joints.length;
    const q = this.quats;
    const r = s.rotations;
    for (let j = 0; j < J; j++) {
      const a = 4 * (f0 * J + j);
      const b = 4 * (f1 * J + j);
      const d = q[a] * q[b] + q[a + 1] * q[b + 1] + q[a + 2] * q[b + 2] + q[a + 3] * q[b + 3] < 0 ? -1 : 1;
      let n = 0;
      for (let k = 0; k < 4; k++) {
        const c = q[a + k] + (d * q[b + k] - q[a + k]) * u;
        r[4 * j + k] = c;
        n += c * c;
      }
      n = 1 / Math.sqrt(n);
      for (let k = 0; k < 4; k++) r[4 * j + k] *= n;
    }
    return s;
  }
}
