import { levelLayers, lineTable, nodeCount, type LatticeLayer, type LineTable } from '../lattice';
import type { CompiledScene } from '../scene';
import { ComputeModule, createBuffer, createBufferWith, UNIFORM, type BindingResource } from '../../gpu/util';
import { Contact } from './contact';
import { SelfContact } from './selfcontact';
import commonWgsl from './wgsl/common.wgsl?raw';
import rowFineWgsl from './wgsl/rowfine.wgsl?raw';
import rowCoarseWgsl from './wgsl/rowcoarse.wgsl?raw';
import assembleWgsl from './wgsl/assemble.wgsl?raw';
import multigridWgsl from './wgsl/multigrid.wgsl?raw';
import smoothWgsl from './wgsl/smooth.wgsl?raw';
import pcgWgsl from './wgsl/pcg.wgsl?raw';
import stepWgsl from './wgsl/step.wgsl?raw';
import yarnFieldWgsl from './wgsl/yarnfield.wgsl?raw';
import { SHARE, shareReach } from '../yarnField';
import gateWgsl from './wgsl/gate.wgsl?raw';
import energyWgsl from './wgsl/energy.wgsl?raw';
import dragWgsl from './wgsl/drag.wgsl?raw';

export interface GpuSolverParams {
  h: number;
  gravity: [number, number, number];
  /** Newton iterations always performed per substep. */
  newtonIterations: number;
  /**
   * Upper bound on Newton iterations. Iterations beyond `newtonIterations` run only while the
   * previous update still moved some node by more than `newtonTolerance` (decided on the GPU).
   */
  newtonMax: number;
  /** Convergence threshold on the largest nodal Newton update, m. */
  newtonTolerance: number;
  /**
   * A span (crossing) advances damage (slip) only if its nodes' last Newton update is below this, m.
   * Off in production (see docs/decisions/2026-10-09-feedback-round1.md): with the production
   * schedule nearly every moving node exceeds any useful tolerance, which suppressed tearing while
   * pulling; ungated history tracks a converged solve closely and adds no spurious ruptures.
   */
  historyTolerance: number;
  pcgIterations: number;
  coarseSweeps: number;
  /** Coarsen while a level has more nodes than this. */
  maxCoarsestNodes: number;
  /** Self-contact of a single-layer scene (selfcontact.wgsl); off where the f64 reference has none. */
  selfContact: boolean;
  /** Development hook: rewrite a module's WGSL before compilation (experiments only). */
  patch?: (module: string, code: string) => string;
}

/**
 * Production schedule (docs/decisions/2026-10-10-feedback-round2.md): 2 Newton x 2 PCG at h = 1/120.
 * With a single PCG iteration the inexact solves inject energy (the sheet never settles after a
 * release and the release itself bounces into spurious tears); the second iteration fits the
 * natural-time budget since the line smoother solves its reduced systems by cyclic reduction.
 */
export const GPU_DEFAULTS: GpuSolverParams = {
  h: 1 / 120,
  gravity: [0, 0, -9.81],
  newtonIterations: 2,
  newtonMax: 2,
  newtonTolerance: 1e-5,
  historyTolerance: 1,
  pcgIterations: 2,
  coarseSweeps: 2,
  maxCoarsestNodes: 400,
  selfContact: true,
};

/** Upper bound on Newton iterations (size of the per-iteration convergence slots). */
const MAX_NEWTON = 8;

const MAX_LINE = 384;
/** Line colors per family (3: lines smoothed together never share a stencil coupling). */
const COLORS_PER_FAMILY = 3;
const LINE_COLORS = 2 * COLORS_PER_FAMILY;
/** Workgroups of the grip kernel in a gated iteration (it early-outs beyond the active entries). */
const GRIP_GROUPS = 32;
/** Floats per operator row (vec4-aligned): fine lattice rows (12 couplings), coarse Galerkin rows (24). */
export const FINE_ROW = 72;
export const COARSE_ROW = 200;
const DOT_GROUPS = 256;
const SLOT = 256;
const SCALARS = 128;

interface Level {
  layers: LatticeLayer[];
  N: number;
  info: GPUBuffer;
  op: GPUBuffer;
  fixed: GPUBuffer;
  X: GPUBuffer;
  B: GPUBuffer;
  R: GPUBuffer;
  lines: GPUBuffer;
  table: LineTable;
  /** Prolongation weights to the next coarser level (4 per node; none on the coarsest level). */
  pw: GPUBuffer | null;
}

/** Packs SimParams (see common.wgsl). */
function simParamsData(scene: CompiledScene, p: GpuSolverParams): ArrayBuffer {
  const buf = new ArrayBuffer(32 + 3 * 112);
  const f = new Float32Array(buf);
  const u = new Uint32Array(buf);
  f[0] = p.h;
  f[1] = p.gravity[0];
  f[2] = p.gravity[1];
  f[3] = p.gravity[2];
  u[4] = scene.nodeCount;
  u[5] = scene.layers.length;
  u[6] = 1;
  f[7] = p.historyTolerance;
  scene.layers.forEach((L, l) => {
    const b = (32 + l * 112) / 4;
    u[b] = L.nodeOffset;
    u[b + 1] = L.nx;
    u[b + 2] = L.ny;
    for (let fam = 0; fam < 2; fam++) {
      const y = L.material.yarns[fam];
      const yb = b + 4 + fam * 8;
      f[yb] = y.eaLow;
      f[yb + 1] = y.eaHigh;
      f[yb + 2] = y.crimpStrain;
      f[yb + 3] = y.eaCompress;
      f[yb + 4] = y.axialViscosity;
      f[yb + 5] = y.bendRigidity;
      f[yb + 6] = y.bendViscosity;
      f[yb + 7] = y.yarnsPerSpan ?? 1;
    }
    const m = L.material;
    f[b + 20] = m.shearStiffness;
    f[b + 21] = m.crossingFriction;
    f[b + 22] = m.crossingWrap;
    f[b + 23] = m.pitchU;
    f[b + 24] = m.pitchV;
    f[b + 25] = m.airDrag;
    f[b + 26] = m.crossingNormal0;
  });
  return buf;
}

/** Per-span state uploaded to the GPU (CompiledScene and ReferenceSolver both provide it). */
export interface SpanState {
  restLength: ArrayLike<number>;
  intact: ArrayLike<number>;
  cutMask: ArrayLike<number>;
  segFlags: ArrayLike<number>;
  failSite: ArrayLike<number>;
}

/** Field damage and cross-section codes packed per crossing: damage pick, warp (bits 0-15), area pick, warp (16-31). */
function packField(damage: Uint8Array, area: Uint8Array): Uint32Array {
  const out = new Uint32Array(damage.length / 2);
  for (let c = 0; c < out.length; c++) out[c] = (damage[2 * c] | (damage[2 * c + 1] << 8) | (area[2 * c] << 16) | (area[2 * c + 1] << 24)) >>> 0;
  return out;
}

function levelInfoData(layers: LatticeLayer[]): Uint32Array {
  const u = new Uint32Array(16);
  u[0] = nodeCount(layers);
  u[1] = layers.length;
  layers.forEach((L, l) => {
    u[4 + 4 * l] = L.offset;
    u[5 + 4 * l] = L.nx;
    u[6 + 4 * l] = L.ny;
  });
  return u;
}

export class GpuSolver {
  readonly device: GPUDevice;
  readonly scene: CompiledScene;
  readonly params: GpuSolverParams;
  readonly N: number;
  // fine state
  readonly simParams: GPUBuffer;
  readonly nodes: GPUBuffer;
  readonly x: GPUBuffer;
  readonly v: GPUBuffer;
  readonly dx: GPUBuffer;
  readonly pred: GPUBuffer;
  readonly segs: GPUBuffer;
  readonly tails: GPUBuffer;
  readonly ext: GPUBuffer;
  /** Yarn damage field (yarnField.ts): per crossing the damage codes of its pick and warp segments. */
  readonly field: GPUBuffer;
  /** Per crossing: peak tension of its pick and warp segment (N). */
  readonly fieldStrength: GPUBuffer;
  private readonly fieldU: GPUBuffer;
  /** [0] = 1 once a damage code changed; cleared by the renderer after rebuilding from the field. */
  readonly fieldEvents: GPUBuffer;
  /** Pull-out marks of one substep (yarnfield.wgsl pullMark / pullApply). */
  private readonly pullBits: GPUBuffer;
  /**
   * [0] ruptures, [1] pulled-out spans, [2] max speed since last cleared (f32 bits), [3] line-search
   * step length of the current Newton iteration (f32 bits),
   * [4 + k] largest nodal update of Newton iteration k in the current substep (f32 bits),
   * [12] Newton directions rejected by the line search (no evaluated step decreased the potential).
   */
  readonly counters: GPUBuffer;
  /** Cross-layer contact for multi-layer scenes, self-contact for a single layer. */
  readonly contact: Contact | SelfContact;
  readonly contactEnabled: boolean;
  readonly gateArgs: GPUBuffer;
  private readonly gateSizes: GPUBuffer;
  /** Distinct workgroup counts dispatched by one Newton iteration (index into gate arguments). */
  private readonly sizeIndex = new Map<number, number>();
  /** While encoding an optional Newton iteration: its index (dispatches become indirect). */
  private gatedIteration = -1;
  // solver workspace
  readonly levels: Level[];
  readonly delta: GPUBuffer;
  readonly p: GPUBuffer;
  readonly Ap: GPUBuffer;
  readonly rec: GPUBuffer;
  readonly partial: GPUBuffer;
  readonly partial4: GPUBuffer;
  readonly scal: GPUBuffer;
  readonly slots: GPUBuffer;
  private slotCount = 0;
  private readonly slotData: Uint32Array;

  private readonly pl: Record<string, GPUComputePipeline> = {};
  private readonly bg: Record<string, GPUBindGroup> = {};
  private readonly mod: Record<string, ComputeModule> = {};
  /** Encodes per-substep external work (grip lead clamp) after the predictor. */
  preSubstep: ((pass: GPUComputePassEncoder) => void) | null = null;
  /** Encodes external terms (grip) into ext before each assembly, using the solver's dispatcher. */
  preAssemble: ((dispatch: (pipe: GPUComputePipeline, group: GPUBindGroup, maxGroups: number) => void) => void) | null = null;

  constructor(device: GPUDevice, scene: CompiledScene, params: GpuSolverParams = GPU_DEFAULTS) {
    this.device = device;
    this.scene = scene;
    this.params = { ...params };
    const N = scene.nodeCount;
    this.N = N;

    // ---- fine state
    this.simParams = createBufferWith(device, 'simParams', new Uint8Array(simParamsData(scene, params)), UNIFORM);
    const nodeData = new ArrayBuffer(16 * N);
    const nu = new Uint32Array(nodeData);
    const nf = new Float32Array(nodeData);
    for (const L of scene.layers) {
      for (let j = 0; j < L.ny; j++) {
        for (let i = 0; i < L.nx; i++) {
          const n = L.nodeOffset + j * L.nx + i;
          nu[4 * n] = i | (j << 12) | (L.index << 24);
          nu[4 * n + 1] = scene.pinned[n];
          nf[4 * n + 2] = scene.mass[n];
          nf[4 * n + 3] = scene.crossingNormal0[n];
        }
      }
    }
    if (scene.layers.some((L) => L.nx >= 4096 || L.ny >= 4096)) throw new Error('lattice too large for ijl packing');
    this.nodes = createBufferWith(device, 'nodes', new Uint8Array(nodeData));
    const xs = new Float32Array(4 * N);
    for (let n = 0; n < N; n++) for (let q = 0; q < 3; q++) xs[4 * n + q] = scene.position[3 * n + q];
    this.x = createBufferWith(device, 'x', xs);
    this.v = createBuffer(device, 'v', 16 * N);
    this.dx = createBuffer(device, 'dx', 16 * N);
    this.pred = createBuffer(device, 'pred', 16 * N);
    this.segs = createBufferWith(device, 'segs', new Uint8Array(this.segData(scene)));
    // the per-yarn field update addresses up to 16 yarns per span (yarnfield.wgsl MAX_BUNDLE_F)
    if (scene.layers.some((L) => L.bundle[0] > 16 || L.bundle[1] > 16)) throw new Error('bundles of more than 16 yarns are not supported by the GPU field update');
    this.field = createBufferWith(device, 'field', packField(scene.fieldDamage, scene.fieldArea));
    this.fieldStrength = createBufferWith(device, 'fieldStrength', scene.fieldStrength);
    const fu = new Uint32Array(12);
    // w: load-sharing reach along picks (bits 0-15) and along warps (16-31), in crossings
    scene.field.forEach((F, l) => fu.set([F.offset, F.K, F.M, shareReach(scene.layers[l].weave, 0) | (shareReach(scene.layers[l].weave, 1) << 16)], 4 * l));
    this.fieldU = createBufferWith(device, 'fieldU', fu, UNIFORM);
    this.fieldEvents = createBufferWith(device, 'fieldEvents', new Uint32Array([1, 0, 0, 0]));
    this.pullBits = createBuffer(device, 'pullBits', 4 * Math.ceil(scene.fieldDamage.length / 2 / 16));
    this.tails = createBufferWith(device, 'tails', Float32Array.from(scene.tails));
    this.ext = createBuffer(device, 'ext', 48 * N);
    this.counters = createBuffer(device, 'counters', 16 + 4 * MAX_NEWTON + 16);
    this.contactEnabled = scene.layers.length > 1 || this.params.selfContact;

    // ---- levels
    const fineLayers: LatticeLayer[] = scene.layers.map((L) => ({ nx: L.nx, ny: L.ny, offset: L.nodeOffset }));
    this.levels = levelLayers(fineLayers, this.params.maxCoarsestNodes).map((layers, k) => {
      const Nl = nodeCount(layers);
      const table = lineTable(layers, COLORS_PER_FAMILY);
      if (table.maxLength > MAX_LINE) throw new Error(`yarn line of ${table.maxLength} exceeds ${MAX_LINE}`);
      const fixedInit = new Uint32Array(Nl);
      if (k === 0) for (let n = 0; n < N; n++) fixedInit[n] = scene.pinned[n];
      return {
        layers,
        N: Nl,
        info: createBufferWith(device, `level${k}.info`, levelInfoData(layers), UNIFORM),
        op: createBuffer(device, `level${k}.op`, Nl * (k === 0 ? FINE_ROW : COARSE_ROW) * 4),
        fixed: createBufferWith(device, `level${k}.fixed`, fixedInit),
        X: createBuffer(device, `level${k}.X`, 16 * Nl),
        B: createBuffer(device, `level${k}.B`, 16 * Nl),
        R: createBuffer(device, `level${k}.R`, 16 * Nl),
        lines: createBufferWith(device, `level${k}.lines`, table.lines),
        table,
        pw: null as GPUBuffer | null,
      };
    });
    this.delta = createBuffer(device, 'pcg.delta', 16 * N);
    this.p = createBuffer(device, 'pcg.p', 16 * N);
    this.Ap = createBuffer(device, 'pcg.Ap', 16 * N);
    this.rec = createBuffer(device, 'smooth.scratch', 10 * 16 * N);
    this.partial = createBuffer(device, 'pcg.partial', 4 * DOT_GROUPS);
    this.partial4 = createBuffer(device, 'lineSearch.partial', 32 * DOT_GROUPS);
    this.scal = createBuffer(device, 'pcg.scal', 4 * SCALARS);
    this.slotData = new Uint32Array((SLOT / 4) * 256);
    this.slots = createBuffer(device, 'slots', this.slotData.byteLength, UNIFORM);

    if (params.newtonMax > MAX_NEWTON || params.newtonIterations < 1 || params.newtonIterations > params.newtonMax) throw new Error('invalid Newton iteration bounds');
    this.contact = scene.layers.length > 1 ? new Contact(this, scene) : new SelfContact(this);
    if (!this.contactEnabled) {
      // no displacement bound: the detection-time displacement is zero and the bound unlimited
      const meta = new Float32Array(4 * N);
      for (let n = 0; n < N; n++) meta[4 * n + 3] = 1e9;
      device.queue.writeBuffer(this.contact.meta, 0, meta);
    }
    // the distinct dispatch sizes of a Newton iteration, recorded from a dry encoding
    const sizes: number[] = [];
    const rec = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups(x: number) { if (!sizes.includes(x)) sizes.push(x); }, end() {} };
    this.createPipelines();
    this.createBindGroups();
    this.encodeNewtonIteration(rec as unknown as GPUComputePassEncoder, 0, true);
    if (!sizes.includes(GRIP_GROUPS)) sizes.push(GRIP_GROUPS);
    sizes.forEach((x, k) => this.sizeIndex.set(x, k));
    this.gateSizes = createBufferWith(device, 'newton.gateSizes', new Uint32Array([sizes.length, new Uint32Array(new Float32Array([params.newtonTolerance]).buffer)[0], ...sizes]));
    this.gateArgs = createBuffer(device, 'newton.gateArgs', 12 * sizes.length * (MAX_NEWTON + 1), GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.createStepBindGroups();
    device.queue.writeBuffer(this.slots, 0, this.slotData);
  }

  private segData(spans: SpanState): ArrayBuffer {
    const s = this.scene;
    const buf = new ArrayBuffer(32 * 2 * s.nodeCount);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    for (let q = 0; q < 2 * s.nodeCount; q++) {
      f[8 * q] = spans.restLength[q];
      f[8 * q + 1] = spans.intact[q];
      f[8 * q + 2] = s.softening[q];
      u[8 * q + 3] = spans.cutMask[q];
      u[8 * q + 5] = spans.segFlags[q];
      f[8 * q + 6] = spans.failSite[q];
    }
    return buf;
  }

  /** Allocate a 256-byte uniform slot holding a vec4<u32>. */
  private slot(a: number, b = 0, c = 0, d = 0): GPUBufferBinding {
    const k = this.slotCount++;
    if (k >= 256) throw new Error('out of uniform slots');
    const o = (k * SLOT) / 4;
    this.slotData[o] = a >>> 0;
    this.slotData[o + 1] = b >>> 0;
    this.slotData[o + 2] = c >>> 0;
    this.slotData[o + 3] = d >>> 0;
    return { buffer: this.slots, offset: k * SLOT, size: 16 };
  }

  private createPipelines(): void {
    const d = this.device;
    const patch = this.params.patch ?? ((_: string, c: string) => c);
    this.mod.assemble = new ComputeModule(d, 'assemble', patch('assemble', commonWgsl + dragWgsl + assembleWgsl));
    this.mod.mgFine = new ComputeModule(d, 'multigrid.fine', patch('multigrid', commonWgsl + multigridWgsl + rowFineWgsl));
    this.mod.mgCoarse = new ComputeModule(d, 'multigrid.coarse', patch('multigrid', commonWgsl + multigridWgsl + rowCoarseWgsl));
    this.mod.smoothFine = new ComputeModule(d, 'smooth.fine', patch('smooth', commonWgsl + smoothWgsl + rowFineWgsl));
    this.mod.smoothCoarse = new ComputeModule(d, 'smooth.coarse', patch('smooth', commonWgsl + smoothWgsl + rowCoarseWgsl));
    this.mod.pcg = new ComputeModule(d, 'pcg', patch('pcg', commonWgsl + pcgWgsl + rowFineWgsl));
    this.mod.step = new ComputeModule(d, 'step', patch('step', commonWgsl + stepWgsl));
    this.pl.assemble = this.mod.assemble.pipeline('main');
    for (const v of ['Fine', 'Coarse']) {
      this.pl[`rap${v}`] = this.mod[`mg${v}`].pipeline('rap');
      this.pl[`residual${v}`] = this.mod[`mg${v}`].pipeline('residual');
      this.pl[`smooth${v}`] = this.mod[`smooth${v}`].pipeline('main');
    }
    this.pl.restrict = this.mod.mgCoarse.pipeline('restrictResidual');
    this.pl.pwFine = this.mod.mgFine.pipeline('prolongWeights');
    this.pl.pwCoarse = this.mod.mgCoarse.pipeline('prolongWeights');
    this.pl.prolong = this.mod.mgCoarse.pipeline('prolongAdd');
    for (const e of ['matvecDot', 'dotPartial', 'reduceFinal', 'updateXR', 'updateP', 'clear2']) this.pl[e] = this.mod.pcg.pipeline(e);
    for (const e of ['predict', 'newtonApply', 'commit', 'slip']) this.pl[e] = this.mod.step.pipeline(e);
    // load-sharing constants from the reference definition (yarnField.ts SHARE)
    const share = yarnFieldWgsl.replace('override SHARE_FRACTION: f32 = 0.6;', `const SHARE_FRACTION: f32 = ${SHARE.fraction.toFixed(6)};`).replace('override SHARE_RUN: u32 = 6u;', `const SHARE_RUN: u32 = ${SHARE.maxRun}u;`);
    this.mod.yarnField = new ComputeModule(d, 'yarnField', commonWgsl + share);
    for (const e of ['spanStrain', 'sepSnapshot', 'fieldUpdatePick', 'fieldUpdateWarp', 'pullMark', 'pullApply', 'couple']) this.pl[e] = this.mod.yarnField.pipeline(e);
    this.mod.gate = new ComputeModule(d, 'gate', gateWgsl);
    this.pl.gate = this.mod.gate.pipeline('gate');
    this.mod.energy = new ComputeModule(d, 'energy', patch('energy', commonWgsl + dragWgsl + energyWgsl));
    this.pl.energyDelta = this.mod.energy.pipeline('energyDelta');
    this.pl.lineChoose = this.mod.energy.pipeline('lineChoose');
  }

  private createBindGroups(): void {
    const L0 = this.levels[0];
    const group = (m: string, key: string, entries: Record<number, BindingResource>) => {
      this.bg[key] = this.mod[m].bindGroup(key, entries);
    };
    group('assemble', 'assemble', { 0: this.simParams, 1: this.nodes, 2: this.x, 3: this.dx, 4: this.pred, 5: this.segs, 6: L0.op, 7: L0.B, 8: this.ext });
    this.levels.forEach((L, k) => {
      const v = k === 0 ? 'Fine' : 'Coarse';
      group(`mg${v}`, `residual${k}`, { 0: L.info, 2: L.op, 3: L.fixed, 6: L.X, 7: L.B, 8: L.R });
      for (let c = 0; c < LINE_COLORS; c++) {
        group(`smooth${v}`, `smooth${k}_${c}`, { 0: L.info, 1: this.slot(L.table.colorStart[c]), 2: L.lines, 3: L.op, 4: L.fixed, 5: L.X, 6: L.B, 7: this.rec });
      }
      const C = this.levels[k + 1];
      if (!C) return;
      L.pw = createBuffer(this.device, `level${k}.pw`, 16 * L.N);
      const pwEntries: Record<number, BindingResource> = { 0: L.info, 2: L.op, 3: L.fixed, 10: L.pw };
      if (k === 0) pwEntries[11] = this.segs;
      group(`mg${v}`, `pw${k}`, pwEntries);
      group(`mg${v}`, `rap${k}`, { 0: L.info, 1: C.info, 2: L.op, 3: L.fixed, 4: C.op, 5: C.fixed, 10: L.pw });
      group('mgCoarse', `restrict${k}`, { 0: L.info, 1: C.info, 3: L.fixed, 5: C.fixed, 6: L.R, 8: C.B, 9: C.X, 10: L.pw });
      group('mgCoarse', `prolong${k}`, { 0: L.info, 1: C.info, 3: L.fixed, 6: C.X, 8: L.X, 10: L.pw });
    });
    const Fi = L0.info;
    group('pcg', 'clear', { 0: Fi, 4: this.delta, 5: L0.X });
    group('pcg', 'dotRZ', { 0: Fi, 4: L0.B, 5: L0.X, 9: this.partial });
    group('pcg', 'matvec', { 0: Fi, 2: L0.op, 3: L0.fixed, 4: this.p, 5: this.Ap, 9: this.partial });
    for (let k = 0; k < SCALARS; k++) group('pcg', `reduce${k}`, { 1: this.slot(0, k, DOT_GROUPS), 9: this.partial, 10: this.scal });
    for (let k = 0; k < 16; k++) {
      group('pcg', `updateXR${k}`, { 0: Fi, 1: this.slot(k), 4: this.delta, 5: this.p, 6: L0.B, 7: this.Ap, 8: L0.X, 10: this.scal });
      group('pcg', `updateP${k}`, { 0: Fi, 1: this.slot(k), 4: this.p, 5: L0.X, 10: this.scal });
    }
    group('pcg', 'initP', { 0: Fi, 1: this.slot(0xffffffff), 4: this.p, 5: L0.X, 10: this.scal });
  }

  private createStepBindGroups(): void {
    const stepEntries = {
      0: this.simParams,
      2: this.nodes,
      3: this.x,
      4: this.v,
      5: this.dx,
      6: this.pred,
      7: this.segs,
      8: this.tails,
      9: this.delta,
      10: this.counters,
      11: this.contact.meta,
    };
    const group = (key: string, entries: Record<number, BindingResource>) => {
      this.bg[key] = this.mod.step.bindGroup(key, entries);
    };
    for (const k of ['predict', 'commit']) group(k, stepEntries);
    this.bg.yarnField = this.mod.yarnField.bindGroup('yarnField', {
      0: this.simParams,
      1: this.fieldU,
      2: this.nodes,
      3: this.x,
      4: this.pred,
      5: this.segs,
      6: this.tails,
      7: this.field,
      8: this.fieldStrength,
      9: this.counters,
      10: this.fieldEvents,
      11: this.pullBits,
    });
    for (const k of ['spanStrain', 'fieldUpdate', 'pull', 'couple']) this.bg[k] = this.bg.yarnField;
    for (let it = 0; it < MAX_NEWTON; it++) {
      group(`newtonApply${it}`, { ...stepEntries, 1: this.slot(it) });
      this.bg[`gate${it}`] = this.mod.gate.bindGroup(`gate${it}`, { 0: this.slot(it), 1: this.counters, 2: this.gateArgs, 3: this.gateSizes });
    }
    for (let f = 0; f < 2; f++) for (let par = 0; par < 2; par++) group(`slip${f}${par}`, { ...stepEntries, 1: this.slot(f, par) });
    this.bg.energy = this.mod.energy.bindGroup('energy', {
      0: this.simParams,
      1: this.nodes,
      2: this.x,
      3: this.dx,
      4: this.pred,
      5: this.segs,
      6: this.delta,
      7: this.ext,
      8: this.partial4,
      9: this.counters,
    });
  }

  /** Dispatch on the current pass; inside an optional Newton iteration the size comes from the gate. */
  private dispatch(pass: GPUComputePassEncoder, pipe: string, group: string, x: number): void {
    this.dispatchRaw(pass, this.pl[pipe], this.bg[group], x);
  }

  private dispatchRaw(pass: GPUComputePassEncoder, pipe: GPUComputePipeline, group: GPUBindGroup, x: number): void {
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    if (this.gatedIteration < 0) {
      pass.dispatchWorkgroups(x);
      return;
    }
    const k = this.sizeIndex.get(x);
    if (k === undefined) throw new Error(`dispatch size ${x} missing from the Newton gate table`);
    pass.dispatchWorkgroupsIndirect(this.gateArgs, 12 * (this.sizeIndex.size * this.gatedIteration + k));
  }

  private nodeGroups(n: number, wg = 128): number {
    return Math.ceil(n / wg);
  }

  /** Smooth all six colors of level k in forward or reverse order. */
  encodeSmooth(pass: GPUComputePassEncoder, k: number, reverse: boolean): void {
    const L = this.levels[k];
    for (let q = 0; q < LINE_COLORS; q++) {
      const c = reverse ? LINE_COLORS - 1 - q : q;
      const count = L.table.colorStart[c + 1] - L.table.colorStart[c];
      if (count > 0) this.dispatch(pass, k === 0 ? 'smoothFine' : 'smoothCoarse', `smooth${k}_${c}`, count);
    }
  }

  /** V-cycle on level k for A x = B with x initially zero on entry at level 0. */
  encodeVCycle(pass: GPUComputePassEncoder, k = 0): void {
    const last = this.levels.length - 1;
    if (k === last) {
      for (let s = 0; s < this.params.coarseSweeps; s++) this.encodeSmooth(pass, k, (s & 1) === 1);
      return;
    }
    const L = this.levels[k];
    const C = this.levels[k + 1];
    this.encodeSmooth(pass, k, false);
    this.dispatch(pass, k === 0 ? 'residualFine' : 'residualCoarse', `residual${k}`, this.nodeGroups(L.N));
    this.dispatch(pass, 'restrict', `restrict${k}`, this.nodeGroups(C.N));
    this.encodeVCycle(pass, k + 1);
    this.dispatch(pass, 'prolong', `prolong${k}`, this.nodeGroups(L.N));
    this.encodeSmooth(pass, k, true);
  }

  /** Single-kernel encoders for profiling. */
  encodeKernel(pass: GPUComputePassEncoder, kind: 'assemble' | 'rap' | 'residual' | 'matvec', level = 0): void {
    if (kind === 'assemble') this.dispatch(pass, 'assemble', 'assemble', this.nodeGroups(this.N));
    if (kind === 'rap') this.dispatch(pass, level === 0 ? 'rapFine' : 'rapCoarse', `rap${level}`, this.nodeGroups(this.levels[level + 1].N, 64));
    if (kind === 'residual') this.dispatch(pass, level === 0 ? 'residualFine' : 'residualCoarse', `residual${level}`, this.nodeGroups(this.levels[level].N));
    if (kind === 'matvec') this.dispatch(pass, 'matvecDot', 'matvec', DOT_GROUPS);
  }

  /**
   * Assemble the Newton system at the current iterate; with `coarse`, also rebuild its Galerkin
   * hierarchy (later Newton iterations of a substep reuse the coarse levels: the preconditioner
   * stays SPD and only the fine operator must be current).
   */
  encodeAssemble(pass: GPUComputePassEncoder, coarse = true): void {
    this.dispatch(pass, 'assemble', 'assemble', this.nodeGroups(this.N));
    if (!coarse) return;
    for (let k = 0; k + 1 < this.levels.length; k++) {
      this.dispatch(pass, k === 0 ? 'pwFine' : 'pwCoarse', `pw${k}`, this.nodeGroups(this.levels[k].N));
      this.dispatch(pass, k === 0 ? 'rapFine' : 'rapCoarse', `rap${k}`, this.nodeGroups(this.levels[k + 1].N, 64));
    }
  }

  /** PCG on the assembled fine system; the solution is left in `delta`. */
  encodePCG(pass: GPUComputePassEncoder): void {
    const K = this.params.pcgIterations;
    const g = this.nodeGroups(this.N, 256);
    this.dispatch(pass, 'clear2', 'clear', g);
    this.encodeVCycle(pass, 0);
    this.dispatch(pass, 'updateP', 'initP', g);
    this.dispatch(pass, 'dotPartial', 'dotRZ', DOT_GROUPS);
    this.dispatch(pass, 'reduceFinal', 'reduce0', 1);
    for (let k = 0; k < K; k++) {
      this.dispatch(pass, 'matvecDot', 'matvec', DOT_GROUPS);
      this.dispatch(pass, 'reduceFinal', `reduce${2 * k + 1}`, 1);
      this.dispatch(pass, 'updateXR', `updateXR${k}`, g);
      if (k + 1 === K) break;
      this.encodeVCycle(pass, 0);
      this.dispatch(pass, 'dotPartial', 'dotRZ', DOT_GROUPS);
      this.dispatch(pass, 'reduceFinal', `reduce${2 * k + 2}`, 1);
      this.dispatch(pass, 'updateP', `updateP${k}`, g);
    }
  }

  /** Line search after a PCG solve: energy changes along delta and the chosen step length. */
  encodeLineSearch(pass: GPUComputePassEncoder): void {
    this.dispatch(pass, 'energyDelta', 'energy', DOT_GROUPS);
    this.dispatch(pass, 'lineChoose', 'energy', 1);
  }

  /** One Newton iteration (index `it` selects its convergence slot). `dry` records sizes only. */
  encodeNewtonIteration(pass: GPUComputePassEncoder, it: number, dry = false): void {
    if (this.contactEnabled) this.contact.encodeTerms(pass);
    if (!dry) this.preAssemble?.((pipe, group) => this.dispatchRaw(pass, pipe, group, GRIP_GROUPS));
    this.encodeAssemble(pass, it === 0);
    this.encodePCG(pass);
    this.encodeLineSearch(pass);
    this.dispatch(pass, 'newtonApply', `newtonApply${it}`, this.nodeGroups(this.N));
  }

  encodePredict(pass: GPUComputePassEncoder): void {
    this.dispatch(pass, 'predict', 'predict', this.nodeGroups(this.N));
  }

  /** Yarn damage field update and span coupling at the committed positions (yarnfield.wgsl). */
  encodeDamage(pass: GPUComputePassEncoder): void {
    this.dispatch(pass, 'spanStrain', 'spanStrain', this.nodeGroups(this.N));
    // per span and yarn, one pass per family (yarnfield.wgsl MAX_BUNDLE_F invocations per node)
    const pullGroups = Math.ceil(Math.ceil(this.scene.fieldDamage.length / 2 / 16) / 128);
    // separated state before the step, read by the load sharing of the update
    this.dispatch(pass, 'sepSnapshot', 'pull', pullGroups);
    this.dispatch(pass, 'fieldUpdatePick', 'fieldUpdate', Math.ceil((16 * this.N) / 128));
    this.dispatch(pass, 'fieldUpdateWarp', 'fieldUpdate', Math.ceil((16 * this.N) / 128));
    // pull-out of loose ends: marked from the codes before the step, then applied
    this.dispatch(pass, 'pullMark', 'pull', pullGroups);
    this.dispatch(pass, 'pullApply', 'pull', pullGroups);
    this.dispatch(pass, 'couple', 'couple', this.nodeGroups(2 * this.N));
  }

  encodeFinalize(pass: GPUComputePassEncoder): void {
    this.dispatch(pass, 'commit', 'commit', this.nodeGroups(this.N));
    this.encodeDamage(pass);
    for (let f = 0; f < 2; f++) for (let par = 0; par < 2; par++) this.dispatch(pass, 'slip', `slip${f}${par}`, this.nodeGroups(this.N));
  }

  /**
   * One accepted substep: predict, contact detection, Newton iterations, commit and history updates.
   * The first `newtonIterations` iterations always run; each further one (up to `newtonMax`) runs
   * with indirect sizes that the gate sets to zero once the previous update fell below tolerance.
   */
  encodeSubstep(encoder: GPUCommandEncoder): void {
    if (this.contactEnabled) this.contact.encodeClear(encoder);
    encoder.clearBuffer(this.counters, 16, 4 * MAX_NEWTON);
    const pass = encoder.beginComputePass({ label: 'substep' });
    this.encodePredict(pass);
    this.preSubstep?.(pass);
    if (this.contactEnabled) this.contact.encodeDetect(pass);
    const { newtonIterations, newtonMax } = this.params;
    for (let it = 0; it < newtonMax; it++) {
      if (it >= newtonIterations) {
        this.dispatch(pass, 'gate', `gate${it - 1}`, Math.ceil(this.sizeIndex.size / 64));
        this.gatedIteration = it;
      }
      try {
        this.encodeNewtonIteration(pass, it);
      } finally {
        this.gatedIteration = -1;
      }
    }
    this.encodeFinalize(pass);
    pass.end();
  }

  /**
   * Contact detection at the current positions without a step, for a state loaded from outside the
   * solve (development stills): the face hash the renderer's thread tails collide with is otherwise
   * built by the substeps. Clears the step increment, which every substep's prediction sets anew.
   */
  encodeContactRefresh(encoder: GPUCommandEncoder): void {
    if (!this.contactEnabled) return;
    this.contact.encodeClear(encoder);
    encoder.clearBuffer(this.dx);
    const pass = encoder.beginComputePass({ label: 'contactRefresh' });
    this.contact.encodeDetect(pass);
    pass.end();
  }

  /** Enable/disable history updates (cohesive damage, crossing slip); off only while settling. */
  setHistoryEnabled(on: boolean): void {
    this.device.queue.writeBuffer(this.simParams, 24, new Uint32Array([on ? 1 : 0]));
  }

  /** Change the timestep (takes effect for subsequently submitted work). */
  setTimestep(h: number): void {
    this.params.h = h;
    this.device.queue.writeBuffer(this.simParams, 0, new Float32Array([h]));
  }

  /** Restore the compiled initial state (positions, zero velocity, spans, tails). */
  resetState(): void {
    const N = this.N;
    const xs = new Float32Array(4 * N);
    for (let n = 0; n < N; n++) for (let q = 0; q < 3; q++) xs[4 * n + q] = this.scene.position[3 * n + q];
    this.setState(xs);
    this.setSegments(this.scene, this.scene.tails);
    this.setField(this.scene.fieldDamage);
    this.device.queue.writeBuffer(this.ext, 0, new Float32Array(12 * N));
  }

  /** Upload per-span state and tails (snapshots, reset, tests). */
  setSegments(spans: SpanState, tails: ArrayLike<number>): void {
    this.device.queue.writeBuffer(this.segs, 0, this.segData(spans));
    this.device.queue.writeBuffer(this.tails, 0, Float32Array.from(tails));
  }

  /** Upload yarn damage codes (2 per crossing, as CompiledScene.fieldDamage). */
  setField(damage: Uint8Array): void {
    this.device.queue.writeBuffer(this.field, 0, packField(damage, this.scene.fieldArea));
    this.device.queue.writeBuffer(this.fieldEvents, 0, new Uint32Array([1]));
  }

  /** Upload a full position set (e.g. a settled snapshot) and zero velocities. */
  setState(position: Float32Array, velocity?: Float32Array): void {
    this.device.queue.writeBuffer(this.x, 0, position);
    if (velocity) this.device.queue.writeBuffer(this.v, 0, velocity);
    else this.device.queue.writeBuffer(this.v, 0, new Float32Array(4 * this.N));
  }
}

