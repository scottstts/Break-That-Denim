import { ComputeModule, createBuffer, UNIFORM } from '../../gpu/util';
import type { CompiledScene } from '../scene';
import { GRIP, gripPatch } from '../grip';
import type { GpuSolver } from './solver';
import commonWgsl from './wgsl/common.wgsl?raw';
import interactWgsl from './wgsl/interact.wgsl?raw';

const MAX_GRIP = 2048;

export interface PickResult {
  node: number;
  position: [number, number, number];
}

/**
 * GPU picking and finite-patch grip. Picking returns material identity (node) plus its accepted
 * position via a small asynchronous readback; no full-state readback is ever needed.
 */
export class Interaction {
  private readonly device: GPUDevice;
  private readonly solver: GpuSolver;
  private readonly scene: CompiledScene;
  private readonly mod: ComputeModule;
  private readonly gripU: GPUBuffer;
  private readonly pickU: GPUBuffer;
  private readonly grip: GPUBuffer;
  private readonly pick: GPUBuffer;
  private readonly pickOut: GPUBuffer;
  private readonly gripGoal: GPUBuffer;
  private readonly group: GPUBindGroup;
  private readonly pipes: Record<string, GPUComputePipeline> = {};
  private staging: GPUBuffer[] = [];
  private readonly excessStaging: GPUBuffer;
  private excessBusy = false;
  /** Number of active grip entries (0 = no grip). */
  count = 0;
  /** Nodes of the active grip patch. */
  patchNodes: readonly number[] = [];
  /** Core of the grip patch (full weight): what the grip actually holds (detached-piece anchors). */
  coreNodes: readonly number[] = [];
  private stiffness = 0;

  constructor(device: GPUDevice, solver: GpuSolver, scene: CompiledScene) {
    this.device = device;
    this.solver = solver;
    this.scene = scene;
    // without contact nothing else writes ext, so grip terms overwrite instead of accumulating
    const code = interactWgsl.replace('const GRIP_ADDS: bool = true;', `const GRIP_ADDS: bool = ${solver.contactEnabled};`);
    this.mod = new ComputeModule(device, 'interact', commonWgsl + code);
    for (const e of ['gripTerms', 'gripLead', 'gripInit', 'pickDepth', 'pickNearest', 'pickWrite']) this.pipes[e] = this.mod.pipeline(e);
    this.gripGoal = createBuffer(device, 'gripGoal', 16);
    this.excessStaging = device.createBuffer({ label: 'gripExcess', size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.gripU = createBuffer(device, 'gripU', 32, UNIFORM);
    this.pickU = createBuffer(device, 'pickU', 96, UNIFORM);
    this.grip = createBuffer(device, 'grip', 32 * MAX_GRIP);
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

  /** Attach the grip to a smoothly weighted material patch around `node` (same layer). */
  attach(node: number, target: [number, number, number]): void {
    this.release();
    const patch = gripPatch(this.scene, node);
    if (!patch) return;
    const entries = patch.nodes.map((n, k) => [n, patch.weights[k]] as const);
    const count = Math.min(entries.length, MAX_GRIP);
    const data = new ArrayBuffer(32 * count);
    const du = new Uint32Array(data);
    const df = new Float32Array(data);
    for (let k = 0; k < count; k++) {
      du[8 * k] = entries[k][0];
      df[8 * k + 1] = entries[k][1];
    }
    this.device.queue.writeBuffer(this.grip, 0, data);
    this.count = count;
    this.patchNodes = entries.slice(0, count).map((e) => e[0]);
    this.coreNodes = entries.slice(0, count).filter((e) => e[1] >= 0.5).map((e) => e[0]);
    this.stiffness = patch.stiffness;
    this.writeUniform(target, node);
    const enc = this.device.createCommandEncoder({ label: 'gripInit' });
    const pass = enc.beginComputePass();
    this.run(pass, 'gripInit', Math.ceil(count / 64));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  private writeUniform(target: [number, number, number], picked = 0): void {
    const u = new Float32Array(8);
    u.set([target[0], target[1], target[2], this.count], 0);
    u.set([this.stiffness, GRIP.maxLead, 0, 1], 4);
    new Uint32Array(u.buffer)[6] = picked;
    this.device.queue.writeBuffer(this.gripU, 0, u);
  }

  /** Spread of the patch relative to grab time (grip.ts pinchScale), applied from the next substep. */
  setPinch(scale: number): void {
    if (this.count === 0) return;
    this.device.queue.writeBuffer(this.gripU, 28, new Float32Array([scale]));
  }

  setTarget(target: [number, number, number]): void {
    if (this.count === 0) return;
    const u = new Float32Array([target[0], target[1], target[2], this.count]);
    this.device.queue.writeBuffer(this.gripU, 0, u);
  }

  /**
   * Remove the grip. With contact, ext is rewritten every Newton iteration; without it the grip owns
   * ext and its entries are cleared here.
   */
  release(): void {
    if (this.count === 0) return;
    this.count = 0;
    this.patchNodes = [];
    this.coreNodes = [];
    this.device.queue.writeBuffer(this.gripU, 12, new Float32Array([0]));
    if (!this.solver.contactEnabled) {
      const enc = this.device.createCommandEncoder({ label: 'gripRelease' });
      enc.clearBuffer(this.solver.ext);
      this.device.queue.submit([enc.finish()]);
    }
  }

  /**
   * How far the requested handle is beyond what the bounded force can follow (m), from the last
   * substep: small asynchronous readback. Null if a read is already in flight or there is no grip.
   */
  async gripExcess(): Promise<number | null> {
    if (this.count === 0 || this.excessBusy) return null;
    this.excessBusy = true;
    try {
      const enc = this.device.createCommandEncoder({ label: 'gripExcess' });
      enc.copyBufferToBuffer(this.gripGoal, 0, this.excessStaging, 0, 16);
      this.device.queue.submit([enc.finish()]);
      await this.excessStaging.mapAsync(GPUMapMode.READ);
      const w = new Float32Array(this.excessStaging.getMappedRange().slice(0))[3];
      this.excessStaging.unmap();
      return w;
    } finally {
      this.excessBusy = false;
    }
  }

  /** Once per substep before the Newton iterations: clamp the handle's lead (bounded force). */
  encodeSubstep(pass: GPUComputePassEncoder): void {
    if (this.count > 0) this.run(pass, 'gripLead', 1);
  }

  /** Per Newton iteration: refresh grip terms at the current iterate (dispatched by the solver). */
  encode(dispatch: (pipe: GPUComputePipeline, group: GPUBindGroup, groups: number) => void): void {
    if (this.count > 0) dispatch(this.pipes.gripTerms, this.group, Math.ceil(this.count / 64));
  }
}
