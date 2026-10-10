import { ComputeModule, createBuffer, createBufferWith, UNIFORM } from '../../gpu/util';
import type { CompiledScene } from '../scene';
import commonWgsl from './wgsl/common.wgsl?raw';
import contactWgsl from './wgsl/contact.wgsl?raw';
import scanWgsl from './wgsl/scan.wgsl?raw';

/** Contact law (SI). Distances are midsurface separations of two plies in contact. */
export const CONTACT = {
  /** Hash cell size: must be at least the largest contact distance. */
  cell: 1.0e-3,
  tableBits: 20,
  stiffness: 3000,
  friction: 0.45,
  /** Regularization speed below which friction is viscous (m/s). */
  frictionSpeed: 0.002,
  /** Fraction of clearance a node may move per substep. */
  boundFraction: 0.45,
};

const K = 4;
const BLOCK = 512;

interface Stateful {
  device: GPUDevice;
  simParams: GPUBuffer;
  nodes: GPUBuffer;
  x: GPUBuffer;
  dx: GPUBuffer;
  segs: GPUBuffer;
  ext: GPUBuffer;
  N: number;
}

/** Cross-layer node contact: detection on a (cell, layer) hash and per-iteration contact/friction terms. */
export class Contact {
  readonly meta: GPUBuffer;
  private readonly s: Stateful;
  private readonly mod: ComputeModule;
  private readonly scan: ComputeModule;
  private readonly pipes: Record<string, GPUComputePipeline> = {};
  private readonly group: GPUBindGroup;
  private readonly scanGroups: Array<{ scan: GPUBindGroup; add: GPUBindGroup; blocks: number }> = [];
  private readonly cells: number;

  constructor(s: Stateful, scene: CompiledScene) {
    this.s = s;
    const d = s.device;
    this.cells = 1 << CONTACT.tableBits;
    const thick = scene.layers.map((L) => L.material.thickness);
    const pair = (a: number, b: number) => 0.5 * ((thick[a] ?? 0) + (thick[b] ?? 0));
    const u = new Float32Array(12);
    u.set([CONTACT.cell, 0, 0, 0], 0);
    new Uint32Array(u.buffer)[1] = this.cells - 1;
    u.set([CONTACT.stiffness, CONTACT.friction, CONTACT.frictionSpeed, CONTACT.boundFraction], 4);
    u.set([pair(0, 1), pair(0, 2), pair(1, 2), 0], 8);
    const cu = createBufferWith(d, 'contactU', u, UNIFORM);
    this.uniform = cu;
    this.law = u.slice(4, 8);
    const cellCount = createBuffer(d, 'contact.cellCount', 4 * this.cells);
    const cellCursor = createBuffer(d, 'contact.cellCursor', 4 * this.cells);
    const cellItems = createBuffer(d, 'contact.cellItems', 4 * s.N);
    const pairs = createBuffer(d, 'contact.pairs', 4 * K * s.N);
    // 2 per node, as self-contact's (step.wgsl): the node's own bound, no reciprocal bound
    this.meta = createBuffer(d, 'contact.meta', 32 * s.N);
    this.mod = new ComputeModule(d, 'contact', commonWgsl + contactWgsl);
    for (const e of ['hashCount', 'hashFill', 'detect', 'terms']) this.pipes[e] = this.mod.pipeline(e);
    this.group = this.mod.bindGroup('contact', {
      0: s.simParams,
      1: cu,
      2: s.nodes,
      3: s.x,
      4: s.dx,
      5: s.segs,
      6: cellCount,
      7: cellCursor,
      8: cellItems,
      9: pairs,
      10: this.meta,
      11: s.ext,
    });

    // multi-level exclusive scan of the cell counts
    this.scan = new ComputeModule(d, 'scan', scanWgsl);
    this.pipes.scanBlocks = this.scan.pipeline('scanBlocks');
    this.pipes.addBlockSums = this.scan.pipeline('addBlockSums');
    let n = this.cells;
    let data = cellCount;
    while (true) {
      const blocks = Math.ceil(n / BLOCK);
      const sums = createBuffer(d, `scan.sums${this.scanGroups.length}`, 4 * Math.max(blocks, 4));
      const su = createBufferWith(d, 'scanU', new Uint32Array([n, 0, 0, 0]), UNIFORM);
      this.scanGroups.push({
        scan: this.scan.bindGroup('scan', { 0: su, 1: data, 2: sums }),
        add: this.scan.bindGroup('scanAdd', { 0: su, 1: data, 2: sums }),
        blocks,
      });
      if (blocks === 1) break;
      data = sums;
      n = blocks;
    }
    this.cellCount = cellCount;
    this.cellCursor = cellCursor;
  }

  private readonly uniform: GPUBuffer;
  private readonly law: Float32Array;

  /** Development switch: disable contact forces (detection still runs). */
  setEnabled(on: boolean): void {
    const law = on ? this.law : new Float32Array([0, 0, this.law[2], this.law[3]]);
    this.s.device.queue.writeBuffer(this.uniform, 16, law);
  }

  private readonly cellCount: GPUBuffer;
  private readonly cellCursor: GPUBuffer;

  private run(pass: GPUComputePassEncoder, name: string, groups: number, group: GPUBindGroup = this.group): void {
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups);
  }

  /** Encoder-level clears must happen outside compute passes. */
  encodeClear(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.cellCount);
    encoder.clearBuffer(this.cellCursor);
  }

  /** Build the segment hash and detect contact pairs and displacement bounds at the current iterate. */
  encodeDetect(pass: GPUComputePassEncoder): void {
    const nodeGroups = Math.ceil(this.s.N / 128);
    this.run(pass, 'hashCount', nodeGroups);
    for (const g of this.scanGroups) this.run(pass, 'scanBlocks', g.blocks, g.scan);
    for (let k = this.scanGroups.length - 2; k >= 0; k--) this.run(pass, 'addBlockSums', this.scanGroups[k].blocks, this.scanGroups[k].add);
    this.run(pass, 'hashFill', nodeGroups);
    this.run(pass, 'detect', nodeGroups);
  }

  /** Contact and friction terms at the current Newton iterate (overwrites ext). */
  encodeTerms(pass: GPUComputePassEncoder): void {
    this.run(pass, 'terms', Math.ceil(this.s.N / 128));
  }
}
