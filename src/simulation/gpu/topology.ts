import { ComputeModule, createBuffer } from '../../gpu/util';
import type { CompiledScene } from '../scene';
import type { GpuSolver } from './solver';
import commonWgsl from './wgsl/common.wgsl?raw';
import stateWgsl from './wgsl/state.wgsl?raw';

/**
 * Detached-piece detection. After ruptures (or a grip release) the active span graph is read back
 * (one u32 per node) and searched from its anchors: mounted nodes and, while gripping, the grip
 * patch core. Cloth nodes that no longer reach an anchor become ballistic debris (see state.wgsl).
 * Small pieces torn off inside the grip also become debris: a few nodes held by one grip spring
 * are too weakly coupled for the global Newton step (it injects energy into them), and a scrap of
 * that size slipping out of the fingers is plausible.
 * At most one analysis is in flight; requests made meanwhile are coalesced into one rerun.
 */
/** Pieces with fewer nodes than this are debris unless they are connected to a mount. */
const MIN_HELD_PIECE = 24;
export class Topology {
  private readonly device: GPUDevice;
  private readonly solver: GpuSolver;
  private readonly scene: CompiledScene;
  private readonly mod: ComputeModule;
  private readonly group: GPUBindGroup;
  private readonly packPipe: GPUComputePipeline;
  private readonly applyPipe: GPUComputePipeline;
  private readonly packed: GPUBuffer;
  private readonly freedBuf: GPUBuffer;
  private readonly staging: GPUBuffer;
  private readonly freed: Uint32Array;
  private readonly visited: Uint8Array;
  private readonly queue: Int32Array;
  /** Per node: lattice row length (stride between rows), column i and row j. */
  private readonly stride: Int32Array;
  private readonly column: Int32Array;
  private readonly row: Int32Array;
  private tail = 0;
  private busy = false;
  /** Bumped by reset(): an analysis started before it must not apply its result. */
  private epoch = 0;
  private again = false;
  private anchors: readonly number[] = [];
  /** Nodes made ballistic so far (development statistics). */
  freedCount = 0;

  constructor(device: GPUDevice, solver: GpuSolver, scene: CompiledScene) {
    this.device = device;
    this.solver = solver;
    this.scene = scene;
    const N = scene.nodeCount;
    this.mod = new ComputeModule(device, 'state', commonWgsl + stateWgsl);
    this.packPipe = this.mod.pipeline('pack');
    this.applyPipe = this.mod.pipeline('applyFreed');
    this.packed = createBuffer(device, 'state.packed', 4 * N);
    this.freedBuf = createBuffer(device, 'state.freed', 4 * N);
    this.staging = device.createBuffer({ label: 'state.staging', size: Math.max(16, 4 * N), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.group = this.mod.bindGroup('state', { 0: solver.simParams, 1: solver.segs, 2: solver.x, 3: solver.levels[0].fixed, 4: this.packed, 5: this.freedBuf });
    this.freed = new Uint32Array(N);
    this.visited = new Uint8Array(N);
    this.queue = new Int32Array(N);
    this.stride = new Int32Array(N);
    this.column = new Int32Array(N);
    this.row = new Int32Array(N);
    for (const L of scene.layers) {
      for (let k = 0; k < L.nodeCount; k++) {
        const n = L.nodeOffset + k;
        this.stride[n] = L.nx;
        this.column[n] = k % L.nx;
        this.row[n] = Math.floor(k / L.nx);
      }
    }
  }

  /** Request an analysis with the given extra anchors (grip patch nodes; empty when not gripping). */
  request(anchors: readonly number[]): Promise<void> {
    this.anchors = anchors;
    if (this.busy) {
      this.again = true;
      return Promise.resolve();
    }
    return this.run();
  }

  /** Clear all dynamic state (reset). */
  reset(): void {
    this.epoch++;
    this.again = false;
    const enc = this.device.createCommandEncoder({ label: 'state.reset' });
    enc.clearBuffer(this.freedBuf);
    this.device.queue.submit([enc.finish()]);
    const fixed = new Uint32Array(this.scene.nodeCount);
    for (let n = 0; n < fixed.length; n++) fixed[n] = this.scene.pinned[n];
    this.device.queue.writeBuffer(this.solver.levels[0].fixed, 0, fixed);
  }

  private async run(): Promise<void> {
    this.busy = true;
    try {
      do {
        this.again = false;
        const epoch = this.epoch;
        const anchors = this.anchors;
        const enc = this.device.createCommandEncoder({ label: 'state.pack' });
        const pass = enc.beginComputePass();
        pass.setBindGroup(0, this.group);
        pass.setPipeline(this.packPipe);
        pass.dispatchWorkgroups(Math.ceil(this.scene.nodeCount / 128));
        pass.end();
        enc.copyBufferToBuffer(this.packed, 0, this.staging, 0, 4 * this.scene.nodeCount);
        this.device.queue.submit([enc.finish()]);
        await this.staging.mapAsync(GPUMapMode.READ);
        const bits = new Uint32Array(this.staging.getMappedRange()).slice();
        this.staging.unmap();
        if (epoch !== this.epoch) continue;
        if (this.search(bits, anchors) > 0) {
          this.device.queue.writeBuffer(this.freedBuf, 0, this.freed);
          const e2 = this.device.createCommandEncoder({ label: 'state.apply' });
          const p2 = e2.beginComputePass();
          p2.setBindGroup(0, this.group);
          p2.setPipeline(this.applyPipe);
          p2.dispatchWorkgroups(Math.ceil(this.scene.nodeCount / 128));
          p2.end();
          this.device.queue.submit([e2.finish()]);
        }
      } while (this.again);
    } finally {
      this.busy = false;
    }
  }

  /**
   * Flood the active span graph from the mounts, then from each grip anchor not yet reached (pieces
   * smaller than MIN_HELD_PIECE stay unreached). Returns the number of newly freed nodes.
   */
  private search(bits: Uint32Array, anchors: readonly number[]): number {
    const s = this.scene;
    const vis = this.visited;
    vis.fill(0);
    this.tail = 0;
    for (let n = 0; n < s.nodeCount; n++) if (s.pinned[n]) this.push(bits, n);
    this.flood(bits, 0);
    for (const a of anchors) {
      if (vis[a] || bits[a] & 4) continue;
      const start = this.tail;
      this.push(bits, a);
      this.flood(bits, start);
      if (this.tail - start < MIN_HELD_PIECE) for (let k = start; k < this.tail; k++) vis[this.queue[k]] = 2;
    }
    let count = 0;
    for (let n = 0; n < s.nodeCount; n++) {
      const f = vis[n] !== 1 && !(bits[n] & 4) ? 1 : 0;
      this.freed[n] = f;
      count += f;
    }
    this.freedCount += count;
    return count;
  }

  private push(bits: Uint32Array, n: number): void {
    if (this.visited[n] || bits[n] & 4) return;
    this.visited[n] = 1;
    this.queue[this.tail++] = n;
  }

  /** Breadth-first flood over active spans from queue position `head` until the queue is exhausted. */
  private flood(bits: Uint32Array, head: number): void {
    const { stride, column } = this;
    while (head < this.tail) {
      const n = this.queue[head++];
      const nx = stride[n];
      if (bits[n] & 1) this.push(bits, n + 1);
      if (bits[n] & 2) this.push(bits, n + nx);
      if (column[n] > 0 && bits[n - 1] & 1) this.push(bits, n - 1);
      if (this.row[n] > 0 && bits[n - nx] & 2) this.push(bits, n - nx);
    }
  }
}
