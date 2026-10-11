import { ComputeModule, createBuffer, createBufferWith, UNIFORM } from '../../gpu/util';
import commonWgsl from './wgsl/common.wgsl?raw';
import selfContactWgsl from './wgsl/selfcontact.wgsl?raw';
import scanWgsl from './wgsl/scan.wgsl?raw';

/** Self-contact law (SI); see selfcontact.wgsl. */
export const SELF_CONTACT = {
  /**
   * Hash cell size: half of it must cover a face's half-diagonal plus the detection reach (each node
   * searches only the 2 x 2 x 2 cells nearest to it).
   */
  cell: 12e-3,
  tableBits: 17,
  /** Detection reach from a node to a face (m). */
  reach: 3e-3,
  /**
   * Contact distance between a node and a face (m): the cloth's apparent thickness plus the bulge of
   * the rendered bicubic surface over the flat collision triangles in tight folds.
   */
  distance: 1.0e-3,
  /**
   * Penalty stiffness k (N/m) of the cubic law (selfcontact.wgsl): the tangent stiffness at half the
   * contact distance. At 0.3 mm of compression it presses ~4 kPa over a lattice node's area, the load of
   * a fabric compression test (5 kPa) over a few tenths of a millimetre; a 1 m/s impact of a node stops
   * 0.23 mm in.
   */
  stiffness: 500,
  friction: 0.45,
  /** Regularization speed below which friction is viscous (m/s). */
  frictionSpeed: 0.002,
  /** Fraction of the clearance a node's Newton correction may cover near a face. */
  boundFraction: 0.45,
};

const K = 4;
const BLOCK = 512;

interface SolverBuffers {
  device: GPUDevice;
  simParams: GPUBuffer;
  nodes: GPUBuffer;
  x: GPUBuffer;
  dx: GPUBuffer;
  segs: GPUBuffer;
  ext: GPUBuffer;
  N: number;
  /** Development hook (GpuSolverParams.patch): rewrites the module's WGSL before compilation. */
  params?: { patch?: (module: string, code: string) => string };
}

/**
 * Self-contact of a cloth layer (node against intact cell): face hash and detection once per
 * substep, penalty/friction terms with corner reactions every Newton iteration (into ext).
 */
export class SelfContact {
  readonly meta: GPUBuffer;
  private readonly s: SolverBuffers;
  private readonly mod: ComputeModule;
  private readonly scan: ComputeModule;
  private readonly pipes: Record<string, GPUComputePipeline> = {};
  private readonly group: GPUBindGroup;
  private readonly scanGroups: Array<{ scan: GPUBindGroup; add: GPUBindGroup; blocks: number }> = [];
  /** Face hash, rebuilt every substep at the predicted iterate (selfcontact.wgsl faceFill): cell counts
   * scanned into start offsets [0, T) and fill cursors [T, 2T); per item, the face centroid and a
   * bounding radius, then its centroid at the substep's start and the face index. A face whose start
   * centroid lies in another cell is listed there too. */
  readonly cells: GPUBuffer;
  readonly items: GPUBuffer;

  constructor(s: SolverBuffers) {
    this.s = s;
    const d = s.device;
    const T = 1 << SELF_CONTACT.tableBits;
    const u = new Float32Array(8);
    u.set([SELF_CONTACT.cell, 0, SELF_CONTACT.reach, SELF_CONTACT.distance], 0);
    new Uint32Array(u.buffer)[1] = T - 1;
    u.set([SELF_CONTACT.stiffness, SELF_CONTACT.friction, SELF_CONTACT.frictionSpeed, SELF_CONTACT.boundFraction], 4);
    const su = createBufferWith(d, 'selfContactU', u, UNIFORM);
    this.cells = createBuffer(d, 'selfContact.cells', 8 * T);
    // a face is hashed at most twice (its start and predicted cells)
    const items = createBuffer(d, 'selfContact.items', 2 * 32 * s.N);
    this.items = items;
    // K candidates per node, then the reciprocal-bound keys (node + 1 in 20 bits) and row damage marks
    if (s.N >= 1 << 20) throw new Error('self-contact bound keys hold at most 2^20 - 1 nodes');
    const pairs = createBuffer(d, 'selfContact.pairs', 4 * (K + 2) * s.N);
    const acc = createBuffer(d, 'selfContact.acc', 4 * 9 * s.N);
    this.meta = createBuffer(d, 'selfContact.meta', 32 * s.N);
    const patch = s.params?.patch ?? ((_: string, c: string) => c);
    this.mod = new ComputeModule(d, 'selfContact', patch('selfcontact', commonWgsl + selfContactWgsl));
    for (const e of ['faceCount', 'faceFill', 'damageRow', 'detect', 'resolveBounds', 'terms', 'gather']) this.pipes[e] = this.mod.pipeline(e);
    this.group = this.mod.bindGroup('selfContact', {
      0: s.simParams,
      1: su,
      2: s.nodes,
      3: s.x,
      4: s.dx,
      5: s.segs,
      6: this.cells,
      7: items,
      8: pairs,
      9: this.meta,
      10: acc,
      11: s.ext,
    });

    // multi-level exclusive scan of the cell counts (the first half of the table)
    this.scan = new ComputeModule(d, 'scan', scanWgsl);
    this.pipes.scanBlocks = this.scan.pipeline('scanBlocks');
    this.pipes.addBlockSums = this.scan.pipeline('addBlockSums');
    let n = T;
    let data: GPUBufferBinding = { buffer: this.cells, offset: 0, size: 4 * T };
    for (;;) {
      const blocks = Math.ceil(n / BLOCK);
      const sums = createBuffer(d, `selfContact.scan${this.scanGroups.length}`, 4 * Math.max(blocks, 4));
      const scanU = createBufferWith(d, 'selfContact.scanU', new Uint32Array([n, 0, 0, 0]), UNIFORM);
      this.scanGroups.push({
        scan: this.scan.bindGroup('scan', { 0: scanU, 1: data, 2: sums }),
        add: this.scan.bindGroup('scanAdd', { 0: scanU, 1: data, 2: sums }),
        blocks,
      });
      if (blocks === 1) break;
      data = { buffer: sums };
      n = blocks;
    }
  }

  private run(pass: GPUComputePassEncoder, name: string, groups: number, group: GPUBindGroup = this.group): void {
    pass.setPipeline(this.pipes[name]);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(groups);
  }

  /** Encoder-level clears must happen outside compute passes. */
  encodeClear(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.cells);
  }

  /**
   * Hash the faces and detect contact candidates and correction bounds at the predicted iterate.
   * `stages` (profiling) selects a subset: 1 count, 2 scan, 4 fill, 8 query.
   */
  encodeDetect(pass: GPUComputePassEncoder, stages = 15): void {
    const groups = Math.ceil(this.s.N / 128);
    if (stages & 1) this.run(pass, 'faceCount', groups);
    if (stages & 2) {
      for (const g of this.scanGroups) this.run(pass, 'scanBlocks', g.blocks, g.scan);
      for (let k = this.scanGroups.length - 2; k >= 0; k--) this.run(pass, 'addBlockSums', this.scanGroups[k].blocks, this.scanGroups[k].add);
    }
    if (stages & 4) this.run(pass, 'faceFill', groups);
    if (stages & 8) {
      this.run(pass, 'damageRow', groups);
      this.run(pass, 'detect', groups);
      this.run(pass, 'resolveBounds', groups);
    }
  }

  /** Contact and friction terms at the current Newton iterate (overwrites ext). */
  encodeTerms(pass: GPUComputePassEncoder): void {
    const groups = Math.ceil(this.s.N / 128);
    this.run(pass, 'terms', groups);
    this.run(pass, 'gather', groups);
  }
}
