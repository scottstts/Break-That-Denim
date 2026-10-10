import { ComputeModule, createBuffer, UNIFORM } from '../../gpu/util';
import { invert, normalize, sub, transformPoint, type Vec3 } from '../../rendering/math';
import type { CompiledScene } from '../scene';
import { GRIP, gripPatch, patchWindowRadius, type PatchWindow } from '../grip';
import type { GpuSolver } from './solver';
import commonWgsl from './wgsl/common.wgsl?raw';
import interactWgsl from './wgsl/interact.wgsl?raw';

/** Grips held at once (one per finger; interact.wgsl GRIPS). */
export const MAX_GRIPS = 4;
/** Patch entries per grip (interact.wgsl SLOT_ENTRIES); all grips fill the solver's 32 x 64 grip dispatch. */
const SLOT_ENTRIES = 512;
const SLOT_BYTES = 48;
/** Proximity candidates (interact.wgsl MAX_CAND). */
const MAX_CAND = 128;
/** Words of the pick output: point, node, kind, then the connectivity window (interact.wgsl pickOut). */
const PICK_OUT_WORDS = 8 + 256;
/** Largest patch window radius one pickWindow workgroup covers ((2 R + 1)^2 <= 256). */
const MAX_WINDOW_RADIUS = 7;

export interface PickResult {
  node: number;
  /** The picked point: where the pointer ray meets the surface, or the crossing itself (proximity). */
  position: [number, number, number];
  /** 'surface': the ray hit visible material; 'proximity': the nearest unoccluded crossing to the pointer. */
  kind: 'surface' | 'proximity';
  /** Current connectivity around the node, for growing the grip patch through it. */
  window: PatchWindow;
}

interface Held {
  nodes: number[];
  core: number[];
  /** Letting go (letGo): still holding, but no longer an anchor of what it holds. */
  releasing: boolean;
  /** Full stiffness per unit weight (N/m). */
  stiffness: number;
}

/**
 * GPU picking and finite-patch grips. Picking returns material identity (node) plus its accepted
 * position via a small asynchronous readback; no full-state readback is ever needed. Up to MAX_GRIPS
 * grips (slots) hold disjoint patches, each with its own handle, bounded force and pinch.
 */
export class Interaction {
  private readonly device: GPUDevice;
  private readonly solver: GpuSolver;
  private readonly scene: CompiledScene;
  private readonly mod: ComputeModule;
  private readonly gripU: GPUBuffer;
  private readonly initU: GPUBuffer;
  private readonly pickU: GPUBuffer;
  private readonly grip: GPUBuffer;
  private readonly pick: GPUBuffer;
  private readonly pickOut: GPUBuffer;
  private readonly gripGoal: GPUBuffer;
  private readonly group: GPUBindGroup;
  private readonly pipes: Record<string, GPUComputePipeline> = {};
  private staging: GPUBuffer[] = [];
  private readonly held: (Held | null)[] = new Array<Held | null>(MAX_GRIPS).fill(null);
  /** Node -> slot holding it. */
  private readonly owner = new Map<number, number>();
  private readonly windowRadius: number;
  private readonly strandRadius: number;

  constructor(device: GPUDevice, solver: GpuSolver, scene: CompiledScene) {
    this.device = device;
    this.solver = solver;
    this.scene = scene;
    // without contact nothing else writes ext, so grip terms overwrite instead of accumulating
    const code = interactWgsl.replace('const GRIP_ADDS: bool = true;', `const GRIP_ADDS: bool = ${solver.contactEnabled};`);
    this.mod = new ComputeModule(device, 'interact', commonWgsl + code);
    for (const e of ['gripTerms', 'gripLead', 'gripInit', 'pickRay', 'pickRayNode', 'pickCandidates', 'pickOcclude', 'pickVisible', 'pickWrite', 'pickWindow']) {
      this.pipes[e] = this.mod.pipeline(e);
    }
    this.gripGoal = createBuffer(device, 'gripGoal', 48 * MAX_GRIPS);
    this.gripU = createBuffer(device, 'gripU', SLOT_BYTES * MAX_GRIPS, UNIFORM);
    this.initU = createBuffer(device, 'gripInitU', 16, UNIFORM);
    this.pickU = createBuffer(device, 'pickU', 128, UNIFORM);
    this.grip = createBuffer(device, 'grip', 32 * SLOT_ENTRIES * MAX_GRIPS);
    this.pick = createBuffer(device, 'pick', 4 * (4 + 2 * MAX_CAND));
    this.pickOut = createBuffer(device, 'pickOut', 4 * PICK_OUT_WORDS);
    this.windowRadius = patchWindowRadius(scene);
    if (this.windowRadius > MAX_WINDOW_RADIUS) throw new Error(`a grip patch spans ${this.windowRadius} lattice cells; the pick window covers ${MAX_WINDOW_RADIUS}`);
    // strands over openings are picked within a quarter of the finer pitch: a bundle is a pitch wide,
    // and a quarter keeps neighbouring strands apart
    this.strandRadius = 0.25 * Math.min(...scene.layers.map((L) => Math.min(L.material.pitchU, L.material.pitchV)));
    this.group = this.mod.bindGroup('interact', {
      0: solver.simParams,
      1: this.gripU,
      2: this.pickU,
      3: solver.nodes,
      4: solver.x,
      5: solver.dx,
      6: solver.segs,
      7: this.grip,
      8: solver.ext,
      9: this.pick,
      10: this.pickOut,
      11: this.gripGoal,
      12: this.initU,
    });
  }

  private run(pass: GPUComputePassEncoder, name: string, groups: number): void {
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(groups);
  }

  /**
   * Pick the visible material under a drawing-buffer pixel: the nearest surface (intact face, or strand
   * over an opening) along the pointer ray; over no surface, the nearest crossing within `radiusPx` that
   * no intact face hides. Material behind a front surface is never picked.
   */
  async pickAt(viewProj: Float32Array, px: number, py: number, width: number, height: number, radiusPx: number): Promise<PickResult | null> {
    const inv = invert(viewProj);
    const nx = (px / width) * 2 - 1;
    const ny = 1 - (py / height) * 2;
    const a = transformPoint(inv, [nx, ny, 0]);
    const b = transformPoint(inv, [nx, ny, 1]);
    const eye: Vec3 = [a[0] / a[3], a[1] / a[3], a[2] / a[3]];
    const dir = normalize(sub([b[0] / b[3], b[1] / b[3], b[2] / b[3]], eye));
    const u = new Float32Array(32);
    u.set(viewProj, 0);
    u.set([px, py, radiusPx, 0], 16);
    u.set([width, height, this.strandRadius, this.windowRadius], 20);
    u.set([...eye, 0], 24);
    u.set([...dir, 0], 28);
    this.device.queue.writeBuffer(this.pickU, 0, u);
    const init = new Uint32Array(4 + 2 * MAX_CAND);
    init.set([0xffffffff, 0xffffffff, 0, 0xffffffff]);
    this.device.queue.writeBuffer(this.pick, 0, init);
    const enc = this.device.createCommandEncoder({ label: 'pick' });
    const pass = enc.beginComputePass();
    const g = Math.ceil(this.solver.N / 128);
    for (const e of ['pickRay', 'pickRayNode', 'pickCandidates', 'pickOcclude']) this.run(pass, e, g);
    this.run(pass, 'pickVisible', Math.ceil(MAX_CAND / 128));
    this.run(pass, 'pickWrite', 1);
    this.run(pass, 'pickWindow', 1);
    pass.end();
    const staging = this.staging.pop() ?? this.device.createBuffer({ size: 4 * PICK_OUT_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(this.pickOut, 0, staging, 0, 4 * PICK_OUT_WORDS);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = staging.getMappedRange().slice(0);
    staging.unmap();
    this.staging.push(staging);
    const w = new Uint32Array(out);
    if (w[4] === 0) return null;
    const f = new Float32Array(out);
    const R = this.windowRadius;
    return {
      node: w[3],
      position: [f[0], f[1], f[2]],
      kind: w[4] === 1 ? 'surface' : 'proximity',
      window: { radius: R, bits: w.slice(8, 8 + (2 * R + 1) ** 2) },
    };
  }

  /** A slot no grip holds, or -1. */
  freeSlot(): number {
    return this.held.indexOf(null);
  }

  /** The slot whose grip holds `node` in its patch, or -1. */
  holder(node: number): number {
    return this.owner.get(node) ?? -1;
  }

  /**
   * Attach grip `slot` to a smoothly weighted material patch around `node` (same layer), grown through
   * the current connectivity `window` when given (PickResult.window). Material another grip already
   * holds stays with it: the patch keeps only free nodes. False when none is left.
   *
   * `anchor` is where the caller takes the picked material to be; targets (setTarget) are in its
   * frame. The GPU applies them from the picked node's position at initialisation, so a stationary
   * target imposes no displacement however old the caller's view of the state is.
   */
  attach(node: number, anchor: [number, number, number], slot = 0, window?: PatchWindow): boolean {
    this.release(slot);
    const patch = gripPatch(this.scene, node, (n) => !this.owner.has(n), window);
    if (!patch) return false;
    const count = Math.min(patch.nodes.length, SLOT_ENTRIES);
    const data = new ArrayBuffer(32 * count);
    const du = new Uint32Array(data);
    const df = new Float32Array(data);
    const nodes: number[] = [];
    const core: number[] = [];
    for (let k = 0; k < count; k++) {
      du[8 * k] = patch.nodes[k];
      df[8 * k + 1] = patch.weights[k];
      df[8 * k + 2] = patch.sides[k];
      nodes.push(patch.nodes[k]);
      if (patch.weights[k] >= 0.5) core.push(patch.nodes[k]);
      this.owner.set(patch.nodes[k], slot);
    }
    // the strip between the pads is inside the pinch: no other finger takes it, but it is not held (not
    // an anchor: cut loose there, it falls)
    for (const n of patch.between) {
      nodes.push(n);
      this.owner.set(n, slot);
    }
    this.held[slot] = { nodes, core, stiffness: patch.stiffness, releasing: false };
    this.device.queue.writeBuffer(this.grip, 32 * SLOT_ENTRIES * slot, data);
    const u = new Float32Array(12);
    u.set([anchor[0], anchor[1], anchor[2], count], 0);
    u.set([patch.stiffness, GRIP.maxLead, 0, 1], 4);
    new Uint32Array(u.buffer)[6] = node;
    u.set([anchor[0], anchor[1], anchor[2], patch.padPitch], 8);
    this.device.queue.writeBuffer(this.gripU, SLOT_BYTES * slot, u);
    this.device.queue.writeBuffer(this.initU, 0, new Uint32Array([slot, 0, 0, 0]));
    const enc = this.device.createCommandEncoder({ label: 'gripInit' });
    const pass = enc.beginComputePass();
    this.run(pass, 'gripInit', Math.ceil(count / 64));
    pass.end();
    this.device.queue.submit([enc.finish()]);
    return true;
  }

  /** Spread of grip `slot`'s patch relative to grab time (grip.ts pinchScale), from the next substep. */
  setPinch(scale: number, slot = 0): void {
    if (!this.held[slot]) return;
    this.device.queue.writeBuffer(this.gripU, SLOT_BYTES * slot + 28, new Float32Array([scale]));
  }

  /**
   * Hold of grip `slot` as a fraction of its full stiffness (fingers opening): the force bound scales
   * with it (the lead bound is fixed).
   */
  setHold(fraction: number, slot = 0): void {
    const h = this.held[slot];
    if (!h) return;
    this.device.queue.writeBuffer(this.gripU, SLOT_BYTES * slot + 16, new Float32Array([h.stiffness * fraction]));
  }

  /** Requested handle of grip `slot`, in the frame of its attach anchor. */
  setTarget(target: [number, number, number], slot = 0): void {
    const h = this.held[slot];
    if (!h) return;
    this.device.queue.writeBuffer(this.gripU, SLOT_BYTES * slot, new Float32Array([target[0], target[1], target[2], h.nodes.length]));
  }

  /**
   * Begin letting go of grip `slot`: it keeps holding at its last target, but its patch no longer
   * anchors detached pieces (coreNodes). The caller ends it (release) once a connectivity analysis
   * requested after this has applied: a piece only the fingers held is then already ballistic, and is
   * never left in the cloth solve without an anchor (its rigid motion is not representable there in
   * f32 next to the yarn stiffness, and the solve flings it).
   */
  letGo(slot = 0): void {
    const h = this.held[slot];
    if (h) h.releasing = true;
  }

  /**
   * Remove grip `slot`. With contact, ext is rewritten every Newton iteration; without it the grips
   * own ext, which is cleared here (the remaining grips rewrite their entries every iteration).
   */
  release(slot = 0): void {
    const h = this.held[slot];
    if (!h) return;
    for (const n of h.nodes) this.owner.delete(n);
    this.held[slot] = null;
    this.device.queue.writeBuffer(this.gripU, SLOT_BYTES * slot + 12, new Float32Array([0]));
    if (!this.solver.contactEnabled) {
      const enc = this.device.createCommandEncoder({ label: 'gripRelease' });
      enc.clearBuffer(this.solver.ext);
      this.device.queue.submit([enc.finish()]);
    }
  }

  releaseAll(): void {
    for (let s = 0; s < MAX_GRIPS; s++) this.release(s);
  }

  /** Any grip held. */
  get active(): boolean {
    return this.held.some((h) => h !== null);
  }

  /** Nodes of the held patches. */
  get patchNodes(): number[] {
    return this.held.flatMap((h) => h?.nodes ?? []);
  }

  /** Cores of the held patches (full weight): what the grips actually hold (detached-piece anchors). */
  get coreNodes(): number[] {
    return this.held.flatMap((h) => (h && !h.releasing ? h.core : []));
  }

  /** Core of one grip's patch. */
  core(slot: number): readonly number[] {
    return this.held[slot]?.core ?? [];
  }

  /** Once per substep before the Newton iterations: clamp each grip's handle lead (bounded force). */
  encodeSubstep(pass: GPUComputePassEncoder): void {
    if (this.active) this.run(pass, 'gripLead', MAX_GRIPS);
  }

  /** Per Newton iteration: refresh grip terms at the current iterate (dispatched by the solver). */
  encode(dispatch: (pipe: GPUComputePipeline, group: GPUBindGroup, groups: number) => void): void {
    if (this.active) dispatch(this.pipes.gripTerms, this.group, (MAX_GRIPS * SLOT_ENTRIES) / 64);
  }
}
