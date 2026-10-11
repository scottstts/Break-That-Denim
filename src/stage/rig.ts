import type { GltfSkinnedMesh } from './gltf';

/** Quaternion [x, y, z, w] and 3-vector helpers (double precision, CPU posing only). */
export type Q = [number, number, number, number];
export type V = [number, number, number];

export const v3 = {
  add: (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  sub: (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  scale: (a: V, s: number): V => [a[0] * s, a[1] * s, a[2] * s],
  dot: (a: V, b: V): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
  cross: (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
  len: (a: V): number => Math.hypot(a[0], a[1], a[2]),
  norm: (a: V): V => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  },
  lerp: (a: V, b: V, t: number): V => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t],
};

export const quat = {
  mul: (a: Q, b: Q): Q => [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ],
  conj: (q: Q): Q => [-q[0], -q[1], -q[2], q[3]],
  axisAngle: (axis: V, angle: number): Q => {
    const a = v3.norm(axis);
    const s = Math.sin(angle / 2);
    return [a[0] * s, a[1] * s, a[2] * s, Math.cos(angle / 2)];
  },
  rotate: (q: Q, v: V): V => {
    const u: V = [q[0], q[1], q[2]];
    const t = v3.scale(v3.cross(u, v), 2);
    return v3.add(v3.add(v, v3.scale(t, q[3])), v3.cross(u, t));
  },
  /** Shortest-arc rotation taking unit direction a to unit direction b. */
  fromTo: (a: V, b: V): Q => {
    const d = v3.dot(a, b);
    if (d < -0.999999) {
      const ax = Math.abs(a[0]) < 0.9 ? v3.cross(a, [1, 0, 0]) : v3.cross(a, [0, 1, 0]);
      return quat.axisAngle(ax, Math.PI);
    }
    const c = v3.cross(a, b);
    return quat.normalize([c[0], c[1], c[2], 1 + d]);
  },
  normalize: (q: Q): Q => {
    const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
    return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
  },
  slerp: (a: Q, b: Q, t: number): Q => {
    let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    const s = d < 0 ? -1 : 1;
    d = Math.abs(d);
    if (d > 0.9995) return quat.normalize([a[0] + (s * b[0] - a[0]) * t, a[1] + (s * b[1] - a[1]) * t, a[2] + (s * b[2] - a[2]) * t, a[3] + (s * b[3] - a[3]) * t]);
    const th = Math.acos(d);
    const wa = Math.sin((1 - t) * th) / Math.sin(th);
    const wb = (s * Math.sin(t * th)) / Math.sin(th);
    return [wa * a[0] + wb * b[0], wa * a[1] + wb * b[1], wa * a[2] + wb * b[2], wa * a[3] + wb * b[3]];
  },
  /** Rotation whose frame maps local axes (f, u) onto world axes (F, U) (each pair orthogonalised). */
  basis: (f: V, u: V, F: V, U: V): Q => {
    const toWorld = quat.fromFrame(F, U);
    const toLocal = quat.fromFrame(f, u);
    return quat.mul(toWorld, quat.conj(toLocal));
  },
  /** Rotation taking (x = f, y = u orthogonalised, z = f x u) from the identity frame. */
  fromFrame: (f: V, u: V): Q => {
    const x = v3.norm(f);
    const z = v3.norm(v3.cross(x, u));
    const y = v3.cross(z, x);
    return quatFromMat3([x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]]);
  },
};

/** Quaternion of a rotation matrix given column-major as 9 numbers (columns may carry a uniform scale). */
export function quatFromMat3(m: ArrayLike<number>): Q {
  const s = Math.hypot(m[0], m[1], m[2]) || 1;
  const a = Array.from({ length: 9 }, (_, k) => m[k] / s);
  const [m00, m10, m20, m01, m11, m21, m02, m12, m22] = a;
  const tr = m00 + m11 + m22;
  let q: Q;
  if (tr > 0) {
    const S = Math.sqrt(tr + 1) * 2;
    q = [(m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S, 0.25 * S];
  } else if (m00 > m11 && m00 > m22) {
    const S = Math.sqrt(1 + m00 - m11 - m22) * 2;
    q = [0.25 * S, (m01 + m10) / S, (m02 + m20) / S, (m21 - m12) / S];
  } else if (m11 > m22) {
    const S = Math.sqrt(1 + m11 - m00 - m22) * 2;
    q = [(m01 + m10) / S, 0.25 * S, (m12 + m21) / S, (m02 - m20) / S];
  } else {
    const S = Math.sqrt(1 + m22 - m00 - m11) * 2;
    q = [(m02 + m20) / S, (m12 + m21) / S, 0.25 * S, (m10 - m01) / S];
  }
  return quat.normalize(q);
}

/** Column-major 4x4 from translation, rotation and uniform-or-not scale. */
export function compose(t: ArrayLike<number>, q: ArrayLike<number>, s: ArrayLike<number>, out: Float64Array<ArrayBufferLike> = new Float64Array(16)): Float64Array<ArrayBufferLike> {
  const [x, y, z, w] = [q[0], q[1], q[2], q[3]];
  out[0] = (1 - 2 * (y * y + z * z)) * s[0];
  out[1] = 2 * (x * y + z * w) * s[0];
  out[2] = 2 * (x * z - y * w) * s[0];
  out[3] = 0;
  out[4] = 2 * (x * y - z * w) * s[1];
  out[5] = (1 - 2 * (x * x + z * z)) * s[1];
  out[6] = 2 * (y * z + x * w) * s[1];
  out[7] = 0;
  out[8] = 2 * (x * z + y * w) * s[2];
  out[9] = 2 * (y * z - x * w) * s[2];
  out[10] = (1 - 2 * (x * x + y * y)) * s[2];
  out[11] = 0;
  out[12] = t[0];
  out[13] = t[1];
  out[14] = t[2];
  out[15] = 1;
  return out;
}

export function mul4(a: ArrayLike<number>, b: ArrayLike<number>, out: Float64Array<ArrayBufferLike> = new Float64Array(16)): Float64Array<ArrayBufferLike> {
  for (let c = 0; c < 4; c++) {
    const b0 = b[4 * c];
    const b1 = b[4 * c + 1];
    const b2 = b[4 * c + 2];
    const b3 = b[4 * c + 3];
    for (let r = 0; r < 4; r++) out[4 * c + r] = a[r] * b0 + a[4 + r] * b1 + a[8 + r] * b2 + a[12 + r] * b3;
  }
  return out;
}

/** Inverse of a rigid transform with uniform scale. */
export function invertRigid(m: ArrayLike<number>, out: Float64Array<ArrayBufferLike> = new Float64Array(16)): Float64Array<ArrayBufferLike> {
  const s2 = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out[4 * c + r] = m[4 * r + c] / s2;
  out[3] = out[7] = out[11] = 0;
  for (let r = 0; r < 3; r++) out[12 + r] = -(out[r] * m[12] + out[4 + r] * m[13] + out[8 + r] * m[14]);
  out[15] = 1;
  return out;
}

export function apply4(m: ArrayLike<number>, p: ArrayLike<number>): V {
  return [m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]];
}

/** A capsule fitted to the skin a joint carries, in that joint's frame. */
export interface BoneCapsule {
  joint: number;
  a: V;
  b: V;
  radius: number;
}

const IDENTITY = compose([0, 0, 0], [0, 0, 0, 1], [1, 1, 1]);

/**
 * The skeleton of a skinned glTF mesh, posed in world space. Joint local rotations are the pose; the
 * scene nodes above the root joint are folded into a fixed pre-transform, and `placement` maps the
 * model's scene space to the world. World-space operations (setWorldRotation, rotateWorld) convert to
 * local rotations so the hierarchy stays the model's own.
 */
export class Rig {
  readonly model: GltfSkinnedMesh;
  readonly count: number;
  readonly names: string[];
  /** Parent joint (skin order), -1 for the root joint. */
  readonly parent: Int32Array;
  /** Joints in hierarchy order (depth-first, parents first: every subtree is a contiguous run). */
  readonly order: number[];
  /** Per joint: its place in `order` and the end (exclusive) of its subtree's run there. */
  private readonly slot: Int32Array;
  private readonly subtreeEnd: Int32Array;
  private static readonly ONE = [1, 1, 1];
  readonly restT: Float64Array;
  readonly restR: Float64Array;
  readonly localT: Float64Array;
  readonly localR: Float64Array;
  /** Model scene space of the root joint's parent. */
  private readonly pre: Float64Array;
  /** Model scene space -> world. */
  placement: Float64Array = Float64Array.from(IDENTITY);
  readonly world: Float64Array;
  /** Skinning matrices (world x inverse bind), f32 for upload: built by skinMatrices. */
  private readonly skin: Float32Array;
  /** Rest world matrices (placement = identity), for fitting. */
  readonly restWorld: Float64Array;
  private readonly rootParent = new Float64Array(16);
  private readonly tmp = new Float64Array(16);

  constructor(model: GltfSkinnedMesh) {
    this.model = model;
    const J = model.joints.length;
    this.count = J;
    const jointOf = new Map<number, number>();
    model.joints.forEach((node, k) => jointOf.set(node, k));
    this.names = model.joints.map((n) => model.nodes[n].name);
    this.parent = new Int32Array(J).fill(-1);
    let root = -1;
    model.joints.forEach((node, k) => {
      let p = model.nodes[node].parent;
      while (p >= 0 && !jointOf.has(p)) p = model.nodes[p].parent;
      if (p >= 0) this.parent[k] = jointOf.get(p)!;
      else root = k;
    });
    if (root < 0) throw new Error('skin has no root joint');
    const children: number[][] = Array.from({ length: J }, () => []);
    for (let k = 0; k < J; k++) if (this.parent[k] >= 0) children[this.parent[k]].push(k);
    this.order = [];
    const stack = [root];
    while (stack.length) {
      const k = stack.pop()!;
      this.order.push(k);
      stack.push(...children[k]);
    }
    if (this.order.length !== J) throw new Error('skin joints are not one hierarchy');
    this.slot = new Int32Array(J);
    this.order.forEach((k, i) => (this.slot[k] = i));
    this.subtreeEnd = new Int32Array(J);
    for (let i = J - 1; i >= 0; i--) {
      const k = this.order[i];
      let end = i + 1;
      for (const c of children[k]) end = Math.max(end, this.subtreeEnd[c]);
      this.subtreeEnd[k] = end;
    }
    // nodes above the root joint, as one fixed transform
    const chain: number[] = [];
    for (let p = model.nodes[model.joints[root]].parent; p >= 0; p = model.nodes[p].parent) chain.unshift(p);
    this.pre = Float64Array.from(IDENTITY);
    for (const n of chain) this.pre = mul4(this.pre, this.nodeLocal(n));
    this.restT = new Float64Array(3 * J);
    this.restR = new Float64Array(4 * J);
    model.joints.forEach((node, k) => {
      const n = model.nodes[node];
      if (n.matrix) throw new Error('joint matrices are not supported');
      if (Math.abs(n.scale[0] - 1) > 1e-3 || Math.abs(n.scale[1] - 1) > 1e-3 || Math.abs(n.scale[2] - 1) > 1e-3) throw new Error(`joint ${n.name} is scaled`);
      this.restT.set(n.translation, 3 * k);
      this.restR.set(quat.normalize([...n.rotation]), 4 * k);
    });
    this.localT = Float64Array.from(this.restT);
    this.localR = Float64Array.from(this.restR);
    this.world = new Float64Array(16 * J);
    this.skin = new Float32Array(16 * J);
    this.update();
    this.restWorld = Float64Array.from(this.world);
  }

  private nodeLocal(n: number): Float64Array {
    const node = this.model.nodes[n];
    return node.matrix ? Float64Array.from(node.matrix) : compose(node.translation, node.rotation, node.scale);
  }

  joint(name: string): number {
    const k = this.names.findIndex((n) => n === name || n.startsWith(`${name}_`));
    if (k < 0) throw new Error(`no joint ${name}`);
    return k;
  }

  /**
   * Place the skeleton by the world matrix of its parent space (the space its root joint's local
   * transform is in): the placement that puts the model's scene nodes above the root there.
   */
  setSkeletonPlacement(m: ArrayLike<number>): void {
    this.placement = Float64Array.from(mul4(m, invertRigid(this.pre)));
  }

  /** Back to the rest pose (local rotations and translations). */
  resetPose(): void {
    this.localT.set(this.restT);
    this.localR.set(this.restR);
  }

  /** World matrices of every joint from the local pose. */
  update(): void {
    mul4(this.placement, this.pre, this.rootParent);
    this.updateRun(0, this.count);
  }

  /** World matrices of joint k and its subtree (only they depend on k's local pose). */
  updateFrom(k: number): void {
    this.updateRun(this.slot[k], this.subtreeEnd[k]);
  }

  private updateRun(from: number, to: number): void {
    for (let i = from; i < to; i++) {
      const k = this.order[i];
      const p = this.parent[k];
      compose(this.localT.subarray(3 * k, 3 * k + 3), this.localR.subarray(4 * k, 4 * k + 4), Rig.ONE, this.tmp);
      mul4(p >= 0 ? this.world.subarray(16 * p, 16 * p + 16) : this.rootParent, this.tmp, this.world.subarray(16 * k, 16 * k + 16));
    }
  }

  /** Skinning matrices (world x inverse bind) of the current pose, f32 (once per rendered frame). */
  skinMatrices(): Float32Array {
    const ibm = this.model.inverseBind;
    for (let k = 0; k < this.count; k++) {
      const m = mul4(this.world.subarray(16 * k, 16 * k + 16), ibm.subarray(16 * k, 16 * k + 16), this.tmp);
      for (let q = 0; q < 16; q++) this.skin[16 * k + q] = m[q];
    }
    return this.skin;
  }

  position(k: number): V {
    const o = 16 * k;
    return [this.world[o + 12], this.world[o + 13], this.world[o + 14]];
  }

  /** A point given in joint k's frame, in the world. */
  point(k: number, local: V): V {
    return apply4(this.world.subarray(16 * k, 16 * k + 16), local);
  }

  worldRotation(k: number): Q {
    const o = 16 * k;
    const w = this.world;
    return quatFromMat3([w[o], w[o + 1], w[o + 2], w[o + 4], w[o + 5], w[o + 6], w[o + 8], w[o + 9], w[o + 10]]);
  }

  private parentRotation(k: number): Q {
    const p = this.parent[k];
    if (p >= 0) return this.worldRotation(p);
    const m = mul4(this.placement, this.pre, this.tmp);
    return quatFromMat3([m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]);
  }

  /** Set joint k's world rotation (its subtree follows at the next update / updateFrom(k)). */
  setWorldRotation(k: number, q: Q): void {
    this.localR.set(quat.normalize(quat.mul(quat.conj(this.parentRotation(k)), q)), 4 * k);
  }

  /** Rotate joint k by `delta` in world space about its own origin; its subtree follows. */
  rotateWorld(k: number, delta: Q): void {
    this.setWorldRotation(k, quat.mul(delta, this.worldRotation(k)));
    this.updateFrom(k);
  }

  /** A direction given in joint k's rest frame, in the world for the current pose. */
  direction(k: number, local: V): V {
    return quat.rotate(this.worldRotation(k), local);
  }

  /** Joint k's frame direction for a rest-pose model-space direction (placement = identity). */
  restLocalDirection(k: number, modelDir: V): V {
    const o = 16 * k;
    const w = this.restWorld;
    const r = quatFromMat3([w[o], w[o + 1], w[o + 2], w[o + 4], w[o + 5], w[o + 6], w[o + 8], w[o + 9], w[o + 10]]);
    return v3.norm(quat.rotate(quat.conj(r), modelDir));
  }

  /**
   * Two-bone IK: joints a (root of the chain), b (middle), c (end) so that c reaches `target`, the
   * middle joint bending toward `pole`.
   */
  twoBone(a: number, b: number, c: number, target: V, pole: V): void {
    const pa = this.position(a);
    const pb = this.position(b);
    const pc = this.position(c);
    const l1 = v3.len(v3.sub(pb, pa));
    const l2 = v3.len(v3.sub(pc, pb));
    const toT = v3.sub(target, pa);
    const d = Math.min(Math.max(v3.len(toT), Math.abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4);
    const u = v3.norm(toT);
    const along = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
    const h = Math.sqrt(Math.max(0, l1 * l1 - along * along));
    const toP = v3.sub(pole, pa);
    const w = v3.norm(v3.sub(toP, v3.scale(u, v3.dot(toP, u))));
    const elbow = v3.add(pa, v3.add(v3.scale(u, along), v3.scale(w, h)));
    this.rotateWorld(a, quat.fromTo(v3.norm(v3.sub(pb, pa)), v3.norm(v3.sub(elbow, pa))));
    const pb2 = this.position(b);
    const pc2 = this.position(c);
    const reach = v3.add(pa, v3.scale(u, d));
    this.rotateWorld(b, quat.fromTo(v3.norm(v3.sub(pc2, pb2)), v3.norm(v3.sub(reach, pb2))));
  }

  /** Skinned rest-pose position of vertex v (model scene space). */
  restVertex(v: number): V {
    const m = this.model;
    const p: V = [0, 0, 0];
    const x = m.position[3 * v];
    const y = m.position[3 * v + 1];
    const z = m.position[3 * v + 2];
    for (let q = 0; q < 4; q++) {
      const w = m.weight[4 * v + q];
      if (w <= 0) continue;
      const k = m.jointIndex[4 * v + q];
      const s = mul4(this.restWorld.subarray(16 * k, 16 * k + 16), m.inverseBind.subarray(16 * k, 16 * k + 16), this.tmp);
      const r = apply4(s, [x, y, z]);
      p[0] += w * r[0];
      p[1] += w * r[1];
      p[2] += w * r[2];
    }
    return p;
  }

  /** Rest positions of the vertices each joint carries most, in that joint's rest frame. */
  carriedVertices(): V[][] {
    const out: V[][] = Array.from({ length: this.count }, () => []);
    const inv = Array.from({ length: this.count }, (_, k) => invertRigid(this.restWorld.subarray(16 * k, 16 * k + 16)));
    const m = this.model;
    for (let v = 0; v < m.position.length / 3; v++) {
      let best = 0;
      for (let q = 1; q < 4; q++) if (m.weight[4 * v + q] > m.weight[4 * v + best]) best = q;
      const k = m.jointIndex[4 * v + best];
      out[k].push(apply4(inv[k], this.restVertex(v)));
    }
    return out;
  }
}

/**
 * Capsules fitted to the skin each joint carries (principal axis of its vertices, the 2nd to 98th
 * percentile along it, radius the 98th percentile of the distance from it plus `blend`), in joint
 * frames: the body the cloth collides with follows the skinned surface without hand-placed shapes.
 * The cloth meets only the capsules, so they enclose the skin: what lies outside shows through cloth
 * draped over it. `blend` (model units) covers the skin between two bones, which blend skinning
 * moves off either bone's rigid frame when the joint bends (knuckles: 3-5 mm at the creature's size).
 */
export function fitCapsules(carried: V[][], blend: number, minVertices = 12): BoneCapsule[] {
  const out: BoneCapsule[] = [];
  carried.forEach((pts, joint) => {
    if (pts.length < minVertices) return;
    const c = pts.reduce((s, p) => v3.add(s, p), [0, 0, 0] as V).map((x) => x / pts.length) as V;
    // principal axis by power iteration on the covariance
    const C = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (const p of pts) {
      const d = v3.sub(p, c);
      for (let r = 0; r < 3; r++) for (let q = 0; q < 3; q++) C[3 * r + q] += d[r] * d[q];
    }
    let axis: V = [0.3, 1, 0.2];
    for (let it = 0; it < 32; it++) axis = v3.norm([C[0] * axis[0] + C[1] * axis[1] + C[2] * axis[2], C[3] * axis[0] + C[4] * axis[1] + C[5] * axis[2], C[6] * axis[0] + C[7] * axis[1] + C[8] * axis[2]]);
    const t = pts.map((p) => v3.dot(v3.sub(p, c), axis)).sort((a, b) => a - b);
    const pct = (arr: number[], f: number) => arr[Math.min(arr.length - 1, Math.max(0, Math.round(f * (arr.length - 1))))];
    const t0 = pct(t, 0.02);
    const t1 = pct(t, 0.98);
    const dist = pts
      .map((p) => {
        const d = v3.sub(p, c);
        const s = Math.min(t1, Math.max(t0, v3.dot(d, axis)));
        return v3.len(v3.sub(d, v3.scale(axis, s)));
      })
      .sort((a, b) => a - b);
    const radius = pct(dist, 0.98) + blend;
    // the rounded caps reach most of a radius past the segment's ends
    const half = Math.max(0, (t1 - t0) / 2 - radius * 0.7);
    const mid = (t0 + t1) / 2;
    out.push({ joint, a: v3.add(c, v3.scale(axis, mid - half)), b: v3.add(c, v3.scale(axis, mid + half)), radius });
  });
  return out;
}
