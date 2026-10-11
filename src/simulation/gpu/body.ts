import { ComputeModule, createBuffer, createBufferWith, UNIFORM } from '../../gpu/util';
import type { CompiledScene } from '../scene';
import type { GpuSolver } from './solver';
import commonWgsl from './wgsl/common.wgsl?raw';
import bodyWgsl from './wgsl/body.wgsl?raw';

/** Capsules of one body (body.wgsl BODY_MAX). */
export const BODY_MAX = 48;
const HEADER_FLOATS = 20;
const CAPSULE_FLOATS = 16;

/** Contact law of cloth against a body's skin (SI). */
export const BODY_CONTACT = {
  /**
   * Penalty stiffness (N/m per node): a node pressed with 3 N (cloth stretched over a shoulder while
   * it tears) sinks 1 mm into the skin, within the contact distance.
   */
  stiffness: 3000,
  /** Skin on denim (dry, rubbery). */
  friction: 0.6,
  /** Speed below which friction is viscous (m/s). */
  frictionSpeed: 0.004,
  /**
   * Contact distance from the skin (m): the cloth's apparent thickness plus the bulge of the rendered
   * bicubic surface past the lattice nodes (as SELF_CONTACT.distance), where the skin's small features
   * (a thumb tip, a knuckle) meet it between nodes.
   */
  distance: 1.5e-3,
};

/**
 * A claw's point through denim: the yarns under it part once the point loads the weave there past
 * `force` (N; a pointed nail is blunter than a sewing needle, whose penetration force through denim is
 * of the order of a newton), over the point's half-width `radius` (m).
 */
export const CLAW_PIERCE = { force: 4, radius: 0.0022 };

/** One capsule of a body: segment (a, b), radius, whether it collides now, and whether b ends in a point. */
export interface BodyCapsule {
  a: readonly number[];
  b: readonly number[];
  radius: number;
  active: boolean;
  sharp?: boolean;
}

/**
 * A prescribed (kinematic) body the cloth collides with: capsules whose pose is set once per substep
 * (setPose, at the substep's end time); the step from the previous pose is the body's surface motion
 * that friction acts on. Its terms are added to the solver's ext every Newton iteration (body.wgsl).
 * Sharp capsules pierce the cloth (encodePierce, after each substep's commit). Nodes a claw holds are
 * exempt from both (setExempt).
 */
export class BodyContact {
  private readonly device: GPUDevice;
  private readonly solver: GpuSolver;
  private readonly uniform: GPUBuffer;
  private readonly exemptBuf: GPUBuffer;
  private readonly exempt: Uint32Array;
  private readonly terms: GPUComputePipeline;
  private readonly pierce: GPUComputePipeline;
  private readonly group: GPUBindGroup;
  private readonly data = new Float32Array(HEADER_FLOATS + CAPSULE_FLOATS * BODY_MAX);
  private readonly prev = new Float32Array(6 * BODY_MAX);
  private count = 0;
  private sharp = false;
  private hasPrev = false;

  constructor(device: GPUDevice, solver: GpuSolver, scene: CompiledScene) {
    if (!solver.contactEnabled) throw new Error('body contact adds to the contact terms (self-contact must be on)');
    this.device = device;
    this.solver = solver;
    this.uniform = createBuffer(device, 'bodyU', 4 * this.data.length, UNIFORM);
    this.exempt = new Uint32Array(solver.N);
    this.exemptBuf = createBufferWith(device, 'body.exempt', this.exempt);
    // per layer: first crossing, crossings per row, rows (body.wgsl FU)
    const fu = new Uint32Array(12);
    scene.field.forEach((f, l) => fu.set([f.offset, f.K, f.M, 0], 4 * l));
    const fieldU = createBufferWith(device, 'body.fieldU', fu, UNIFORM);
    const mod = new ComputeModule(device, 'body', commonWgsl + bodyWgsl);
    this.terms = mod.pipeline('bodyTerms');
    this.pierce = mod.pipeline('pierce');
    this.group = mod.bindGroup('body', {
      0: solver.simParams,
      1: this.uniform,
      2: solver.nodes,
      3: solver.x,
      4: solver.dx,
      5: solver.ext,
      6: this.exemptBuf,
      7: solver.field,
      8: fieldU,
      9: solver.fieldEvents,
    });
    this.data.set([CLAW_PIERCE.force, CLAW_PIERCE.radius, 0, 0], 16);
    this.device.queue.writeBuffer(this.uniform, 0, this.data, 0, HEADER_FLOATS);
  }

  /** Any capsule collides. */
  get active(): boolean {
    return this.count > 0;
  }

  /**
   * The body's pose at the end of the next substep (h). Capsule order must stay the same between
   * calls: each capsule's step is taken from its previous pose. An empty list removes the body.
   */
  setPose(capsules: readonly BodyCapsule[], h: number): void {
    const n = Math.min(BODY_MAX, capsules.length);
    const d = this.data;
    const reach = BODY_CONTACT.distance;
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    let live = 0;
    let sharp = false;
    for (let k = 0; k < n; k++) {
      const c = capsules[k];
      const o = HEADER_FLOATS + CAPSULE_FLOATS * k;
      const pv = this.prev.subarray(6 * k, 6 * k + 6);
      d[o] = c.a[0];
      d[o + 1] = c.a[1];
      d[o + 2] = c.a[2];
      d[o + 3] = c.radius;
      d[o + 4] = c.b[0];
      d[o + 5] = c.b[1];
      d[o + 6] = c.b[2];
      d[o + 7] = c.active ? 1 : 0;
      for (let q = 0; q < 3; q++) {
        d[o + 8 + q] = this.hasPrev ? c.a[q] - pv[q] : 0;
        d[o + 12 + q] = this.hasPrev ? c.b[q] - pv[3 + q] : 0;
        pv[q] = c.a[q];
        pv[3 + q] = c.b[q];
      }
      d[o + 11] = c.sharp ? 1 : 0;
      d[o + 15] = 0;
      if (!c.active) continue;
      live++;
      sharp ||= !!c.sharp;
      for (let q = 0; q < 3; q++) {
        lo[q] = Math.min(lo[q], c.a[q] - c.radius - reach, c.b[q] - c.radius - reach);
        hi[q] = Math.max(hi[q], c.a[q] + c.radius + reach, c.b[q] + c.radius + reach);
      }
    }
    this.hasPrev = n > 0;
    this.count = live > 0 ? n : 0;
    this.sharp = live > 0 && sharp;
    new Uint32Array(d.buffer, 0, 4)[0] = this.count;
    d[4] = BODY_CONTACT.stiffness;
    d[5] = BODY_CONTACT.friction;
    d[6] = BODY_CONTACT.frictionSpeed * h;
    d[7] = reach;
    if (live > 0) d.set([lo[0], lo[1], lo[2], 0, hi[0], hi[1], hi[2], 0], 8);
    else d.fill(0, 8, 16);
    this.device.queue.writeBuffer(this.uniform, 0, d, 0, HEADER_FLOATS + CAPSULE_FLOATS * n);
  }

  /**
   * Nodes a claw holds (on: they ride on the claw, which neither pushes nor pierces them) or lets go
   * (off: they slide off it and meet the body again once clear of it, body.wgsl).
   */
  setExempt(nodes: readonly number[], on: boolean): void {
    const sorted = Int32Array.from(nodes).sort();
    // runs of consecutive nodes (a patch is a disc: one run per lattice row)
    for (let k = 0; k < sorted.length; ) {
      let e = k + 1;
      while (e < sorted.length && sorted[e] === sorted[e - 1] + 1) e++;
      this.exempt.fill(on ? 1 : 2, sorted[k], sorted[e - 1] + 1);
      this.device.queue.writeBuffer(this.exemptBuf, 4 * sorted[k], this.exempt, sorted[k], e - k);
      k = e;
    }
  }

  /** Remove the body (no terms from the next substep on) and every exemption. */
  clear(): void {
    this.setPose([], 0);
    this.hasPrev = false;
    this.exempt.fill(0);
    this.device.queue.writeBuffer(this.exemptBuf, 0, this.exempt);
  }

  /** Per Newton iteration, after the contact terms and before the grips (dispatched by the solver). */
  encode(dispatch: (pipe: GPUComputePipeline, group: GPUBindGroup, groups: number) => void): void {
    if (this.count > 0) dispatch(this.terms, this.group, Math.ceil(this.solver.N / 128));
  }

  /** Once per substep after its commit, before the damage update: sharp points part the yarns they load past piercing. */
  encodePierce(pass: GPUComputePassEncoder): void {
    if (!this.sharp) return;
    pass.setPipeline(this.pierce);
    pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(Math.ceil(this.solver.N / 128));
  }
}
