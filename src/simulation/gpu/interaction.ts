import { ComputeModule, createBuffer, UNIFORM } from '../../gpu/util';
import type { CompiledScene } from '../scene';
import { GRIP, gripPatch } from '../grip';
import type { GpuSolver } from './solver';
import commonWgsl from './wgsl/common.wgsl?raw';
import interactWgsl from './wgsl/interact.wgsl?raw';

/** Grips held at once (one per finger; interact.wgsl GRIPS). */
export const MAX_GRIPS = 4;
/** Patch entries per grip (interact.wgsl SLOT_ENTRIES); all grips fill the solver's 32 x 64 grip dispatch. */
const SLOT_ENTRIES = 512;
const SLOT_BYTES = 32;

export interface PickResult {
  node: number;
  position: [number, number, number];
}

interface Held {
  nodes: number[];
  core: number[];
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

  constructor(device: GPUDevice, solver: GpuSolver, scene: CompiledScene) {
    this.device = device;
    this.solver = solver;
    this.scene = scene;
    // without contact nothing else writes ext, so grip terms overwrite instead of accumulating
    const code = interactWgsl.replace('const GRIP_ADDS: bool = true;', `const GRIP_ADDS: bool = ${solver.contactEnabled};`);
    this.mod = new ComputeModule(device, 'interact', commonWgsl + code);
    for (const e of ['gripTerms', 'gripLead', 'gripInit', 'pickDepth', 'pickNearest', 'pickWrite']) this.pipes[e] = this.mod.pipeline(e);
    this.gripGoal = createBuffer(device, 'gripGoal', 16 * MAX_GRIPS);
    this.gripU = createBuffer(device, 'gripU', SLOT_BYTES * MAX_GRIPS, UNIFORM);
    this.initU = createBuffer(device, 'gripInitU', 16, UNIFORM);
    this.pickU = createBuffer(device, 'pickU', 96, UNIFORM);
    this.grip = createBuffer(device, 'grip', 32 * SLOT_ENTRIES * MAX_GRIPS);
    this.pick = createBuffer(device, 'pick', 16);
    this.pickOut = createBuffer(device, 'pickOut', 16);
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

  /** Pick the front-most material crossing near a drawing-buffer pixel. */
  async pickAt(viewProj: Float32Array, px: number, py: number, width: number, height: number, radiusPx: number): Promise<PickResult | null> {
    const u = new Float32Array(24);
    u.set(viewProj, 0);
    u.set([px, py, radiusPx, 0], 16);
    // depth tolerance in NDC: generous enough for one layer thickness at close range
    u.set([width, height, 0.0004, 0], 20);
    this.device.queue.writeBuffer(this.pickU, 0, u);
    this.device.queue.writeBuffer(this.pick, 0, new Uint32Array([0xffffffff, 0xffffffff, 0, 0]));
    const enc = this.device.createCommandEncoder({ label: 'pick' });
    const pass = enc.beginComputePass();
    const g = Math.ceil(this.solver.N / 128);
    this.run(pass, 'pickDepth', g);
    this.run(pass, 'pickNearest', g);
    this.run(pass, 'pickWrite', 1);
    pass.end();
    const staging = this.staging.pop() ?? this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(this.pickOut, 0, staging, 0, 16);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const f = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    this.staging.push(staging);
    if (f[0] === -1 && f[1] === -1) return null;
    const node = new Uint32Array(f.buffer)[3];
    return { node, position: [f[0], f[1], f[2]] };
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
   * Attach grip `slot` to a smoothly weighted material patch around `node` (same layer). Material
   * another grip already holds stays with it: the patch keeps only free nodes. False when none is left.
   */
  attach(node: number, target: [number, number, number], slot = 0): boolean {
    this.release(slot);
    const patch = gripPatch(this.scene, node, (n) => !this.owner.has(n));
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
      nodes.push(patch.nodes[k]);
      if (patch.weights[k] >= 0.5) core.push(patch.nodes[k]);
      this.owner.set(patch.nodes[k], slot);
    }
    this.held[slot] = { nodes, core };
    this.device.queue.writeBuffer(this.grip, 32 * SLOT_ENTRIES * slot, data);
    const u = new Float32Array(8);
    u.set([target[0], target[1], target[2], count], 0);
    u.set([patch.stiffness, GRIP.maxLead, 0, 1], 4);
    new Uint32Array(u.buffer)[6] = node;
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

  setTarget(target: [number, number, number], slot = 0): void {
    const h = this.held[slot];
    if (!h) return;
    this.device.queue.writeBuffer(this.gripU, SLOT_BYTES * slot, new Float32Array([target[0], target[1], target[2], h.nodes.length]));
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
    return this.held.flatMap((h) => h?.core ?? []);
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
