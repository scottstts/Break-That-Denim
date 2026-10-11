// Picking and grip terms. A grip is a finite material patch attached to a handle by quadratic
// springs (exact for Newton). Its force is bounded by clamping the handle's lead over the patch
// centroid once per substep (gripLead); gradient and stiffness enter the Newton system via ext.
// Up to GRIPS grips (one per finger) hold disjoint patches: entries of grip g occupy
// [g * SLOT_ENTRIES, g * SLOT_ENTRIES + count), and no node is in two patches, so every ext entry
// has a single writer.

const GRIP_ADDS: bool = true;
const GRIPS: u32 = 4u;
const SLOT_ENTRIES: u32 = 512u;
// proximity-pick candidates (cursor disc) tested for occlusion
const MAX_CAND: u32 = 128u;
const NONE: u32 = 0xFFFFFFFFu;
// a candidate counts as hidden only behind a surface at least this far in front of it (m)
const OCCLUDE_MARGIN: f32 = 0.5e-3;

struct GripSlot {
  // requested handle (xyz, in the frame of `anchor`), entry count (w as f32)
  goal: vec4<f32>,
  // stiffness per unit weight (N/m), largest handle lead (m), picked node (bitcast), pinch scale of the
  // grab-time offsets (grip.ts pinchScale)
  params: vec4<f32>,
  // where the host believed the picked material was when it attached (xyz): the handle is
  // origin + goal - anchor, so a stationary pointer imposes no displacement whatever the pick latency;
  // lattice pitch along the pads' closing axis (w; grip.ts GripPatch.padPitch)
  anchor: vec4<f32>,
};

struct GripU {
  slots: array<GripSlot, GRIPS>,
};

struct PickU {
  viewProj: mat4x4<f32>,
  // pointer x, y (drawing-buffer pixels), proximity radius (pixels), unused
  pointer: vec4<f32>,
  // width, height, strand pick radius (m), patch window radius (lattice cells, as f32)
  viewport: vec4<f32>,
  // ray origin (camera eye) and unit direction through the pointer (world)
  eye: vec4<f32>,
  ray: vec4<f32>,
};

struct GripEntry {
  node: u32,
  w: f32,
  // pad along the closing axis (-1, +1), 0 in a radial pinch
  side: f32,
  pad1: f32,
  offset: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<uniform> G: GripU;
@group(0) @binding(2) var<uniform> PK: PickU;
@group(0) @binding(3) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> dx: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> segs: array<Seg>;
@group(0) @binding(7) var<storage, read_write> grip: array<GripEntry>;
@group(0) @binding(8) var<storage, read_write> ext: array<vec4<f32>>; // 3 per node (see assemble.wgsl)
// [0] nearest ray hit (f32 bits of t), [1] its material node, [2] candidate count, [3] nearest visible
// candidate (screen distance << 20 | node), [4, 4 + MAX_CAND) candidates, then their occluded flags
@group(0) @binding(9) var<storage, read_write> pick: array<atomic<u32>>;
// [0..2] picked point (f32 bits), [3] node, [4] kind (0 none, 1 surface hit, 2 proximity),
// [8, 8 + 256) patch window around the node (pickWindow)
@group(0) @binding(10) var<storage, read_write> pickOut: array<u32>;
// [3g] effective handle of grip g (xyz), [3g + 1] its origin: the picked node's position at grab time,
// [3g + 2] the pads' closing axis (unit, world; from the grab-time offsets)
@group(0) @binding(11) var<storage, read_write> gripGoal: array<vec4<f32>>;
// x: the grip whose entries gripInit initialises
@group(0) @binding(12) var<uniform> InitU: vec4<u32>;

var<workgroup> leadSum: array<vec4<f32>, 256>;

// a node of the cloth solve (debris and retired nodes no longer respond to the grip)
fn live(n: u32) -> bool { return x[n].w < 0.5; }

// Target offset of a patch entry from the handle at pinch scale s: a radial pinch contracts it toward
// the grab point; a pad slides rigidly toward the grab line along the closing axis by (1 - s) pitches.
fn gathered(e: GripEntry, S: GripSlot, axis: vec3<f32>) -> vec3<f32> {
  let s = S.params.w;
  if (e.side == 0.0) { return s * e.offset.xyz; }
  return e.offset.xyz - e.side * (1.0 - s) * S.anchor.w * axis;
}

// Once per substep, from committed positions: each grip's effective handle is the requested one,
// moved to within the largest lead of its weighted patch centroid (bounded total force). One
// workgroup per grip.
@compute @workgroup_size(256)
fn gripLead(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.x;
  let S = G.slots[g];
  let count = u32(S.goal.w);
  if (count == 0u) { return; }
  // the closing axis: from the - pad to the + pad, as they lay at grab time
  var dir = vec4<f32>(0.0);
  for (var k = t; k < count; k += 256u) {
    let e = grip[g * SLOT_ENTRIES + k];
    dir += vec4<f32>(e.side * e.w * e.offset.xyz, 0.0);
  }
  leadSum[t] = dir;
  workgroupBarrier();
  var r = 128u;
  loop {
    if (r == 0u) { break; }
    if (t < r) { leadSum[t] += leadSum[t + r]; }
    workgroupBarrier();
    r = r / 2u;
  }
  let ld = length(leadSum[0].xyz);
  let axis = select(vec3<f32>(0.0), leadSum[0].xyz / max(ld, 1e-20), ld > 1e-12);
  workgroupBarrier();
  var acc = vec4<f32>(0.0);
  for (var k = t; k < count; k += 256u) {
    let e = grip[g * SLOT_ENTRIES + k];
    if (!live(e.node)) { continue; }
    acc += vec4<f32>(e.w * (x[e.node].xyz - gathered(e, S, axis)), e.w);
  }
  leadSum[t] = acc;
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) { break; }
    if (t < s) { leadSum[t] += leadSum[t + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (t == 0u) {
    gripGoal[3u * g + 2u] = vec4<f32>(axis, 0.0);
    if (leadSum[0].w <= 0.0) {
      // nothing held is still cloth: the handle exerts nothing
      gripGoal[3u * g] = vec4<f32>(gripGoal[3u * g + 1u].xyz, 0.0);
      return;
    }
    let c = leadSum[0].xyz / leadSum[0].w;
    let requested = gripGoal[3u * g + 1u].xyz + S.goal.xyz - S.anchor.xyz;
    var lead = requested - c;
    let l = length(lead);
    if (l > S.params.y) { lead *= S.params.y / l; }
    gripGoal[3u * g] = vec4<f32>(c + lead, 0.0);
  }
}

@compute @workgroup_size(64)
fn gripTerms(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = gid.x / SLOT_ENTRIES;
  let k = gid.x % SLOT_ENTRIES;
  if (g >= GRIPS) { return; }
  let S = G.slots[g];
  if (k >= u32(S.goal.w)) { return; }
  let e = grip[gid.x];
  if (!live(e.node)) { return; }
  let y = x[e.node].xyz + dx[e.node].xyz;
  let d = y - (gripGoal[3u * g].xyz + gathered(e, S, gripGoal[3u * g + 2u].xyz));
  let kg = S.params.x * e.w;
  let force = kg * d;
  let stiff = kg;
  // with contact, its terms already wrote ext this iteration and the grip adds its own
  let b = 3u * e.node;
  let keep = select(0.0, 1.0, GRIP_ADDS);
  ext[b] = keep * ext[b] + vec4<f32>(force, 0.0);
  ext[b + 1u] = keep * ext[b + 1u] + vec4<f32>(stiff, 0.0, 0.0, stiff);
  ext[b + 2u] = keep * ext[b + 2u] + vec4<f32>(0.0, stiff, 0.0, 0.0);
}

// Offsets of a grip's patch nodes from its picked node at grab time, and the grip's origin (the picked
// node's position now, in the same state the offsets are taken from).
@compute @workgroup_size(64)
fn gripInit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = InitU.x;
  let S = G.slots[g];
  let picked = bitcast<u32>(S.params.z);
  if (gid.x == 0u) {
    gripGoal[3u * g + 1u] = vec4<f32>(x[picked].xyz, 0.0);
    gripGoal[3u * g] = vec4<f32>(x[picked].xyz, 0.0);
  }
  if (gid.x >= u32(S.goal.w)) { return; }
  let k = g * SLOT_ENTRIES + gid.x;
  grip[k].offset = vec4<f32>(x[grip[k].node].xyz - x[picked].xyz, 0.0);
}

// ---------------------------------------------------------------- picking (on demand, at a grab)

fn lat(n: u32) -> vec4<u32> { return P.layers[nodes[n].ijl >> 24u].lattice; }
fn col(n: u32) -> u32 { return nodes[n].ijl & 0xFFFu; }
fn row(n: u32) -> u32 { return (nodes[n].ijl >> 12u) & 0xFFFu; }
fn spanOn(owner: u32, f: u32) -> bool { return segActive(segs[2u * owner + f]); }

// A cloth crossing with at least one surviving incident span, owned or incoming (debris and isolated
// crossings are not grab targets).
fn hasMaterial(n: u32) -> bool {
  if (!live(n)) { return false; }
  if (spanOn(n, 0u) || spanOn(n, 1u)) { return true; }
  let L = lat(n);
  if (col(n) > 0u && spanOn(n - 1u, 0u)) { return true; }
  return row(n) > 0u && spanOn(n - L.y, 1u);
}

// An intact cell with lower-left node f (all four spans hold, all corners cloth): a surface face.
fn faceValid(f: u32) -> bool {
  let L = lat(f);
  if (col(f) + 1u >= L.y || row(f) + 1u >= L.z) { return false; }
  let c10 = f + 1u;
  let c01 = f + L.y;
  if (!live(f) || !live(c10) || !live(c01) || !live(c01 + 1u)) { return false; }
  return spanOn(f, 0u) && spanOn(f, 1u) && spanOn(c10, 1u) && spanOn(c01, 0u);
}

// Ray (o, d) against triangle (a, b, c): (t, u, v) with barycentric weights (1 - u - v, u, v); t < 0
// when missed.
fn rayTri(o: vec3<f32>, d: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
  let e1 = b - a;
  let e2 = c - a;
  let p = cross(d, e2);
  let det = dot(e1, p);
  if (abs(det) < 1e-14) { return vec3<f32>(-1.0, 0.0, 0.0); }
  let inv = 1.0 / det;
  let s = o - a;
  let u = dot(s, p) * inv;
  if (u < 0.0 || u > 1.0) { return vec3<f32>(-1.0, 0.0, 0.0); }
  let q = cross(s, e1);
  let v = dot(d, q) * inv;
  if (v < 0.0 || u + v > 1.0) { return vec3<f32>(-1.0, 0.0, 0.0); }
  return vec3<f32>(dot(e2, q) * inv, u, v);
}

// Ray (o, unit d) against the capsule of radius r around segment (a, b): (t, s) with s the segment
// parameter of the closest approach; t < 0 when missed.
fn rayCapsule(o: vec3<f32>, d: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> vec2<f32> {
  let e = b - a;
  let w = o - a;
  let ee = dot(e, e);
  let de = dot(d, e);
  let den = ee - de * de;
  var s = 0.0;
  if (den > 1e-14) { s = clamp((dot(e, w) * dot(d, d) - de * dot(d, w)) / den, 0.0, 1.0); }
  let q = a + s * e;
  let t = max(dot(q - o, d), 0.0);
  let gap = length(o + t * d - q);
  if (gap > r) { return vec2<f32>(-1.0, 0.0); }
  return vec2<f32>(t - sqrt(r * r - gap * gap), s);
}

struct RayHit {
  t: f32,
  node: u32,
};

// Nearest pointer-ray hit among the material primitives node n owns: the intact face it is the
// lower-left corner of (two triangles), and its surviving spans not covered by an intact face on
// either side (strands over openings, as capsules of the strand pick radius). The material node is the
// hit's nearest corner (face) or nearer end (strand).
fn rayPrims(n: u32) -> RayHit {
  var best = RayHit(3.4e38, NONE);
  let o = PK.eye.xyz;
  let d = PK.ray.xyz;
  let L = lat(n);
  if (faceValid(n)) {
    let c = vec4<u32>(n, n + 1u, n + L.y, n + L.y + 1u);
    let p00 = x[c.x].xyz;
    let p10 = x[c.y].xyz;
    let p01 = x[c.z].xyz;
    let p11 = x[c.w].xyz;
    let h1 = rayTri(o, d, p00, p10, p11);
    if (h1.x > 0.0 && h1.x < best.t) {
      let w = vec3<f32>(1.0 - h1.y - h1.z, h1.y, h1.z);
      best = RayHit(h1.x, select(select(c.w, c.y, w.y >= w.z), c.x, w.x >= max(w.y, w.z)));
    }
    let h2 = rayTri(o, d, p00, p11, p01);
    if (h2.x > 0.0 && h2.x < best.t) {
      let w = vec3<f32>(1.0 - h2.y - h2.z, h2.y, h2.z);
      best = RayHit(h2.x, select(select(c.z, c.w, w.y >= w.z), c.x, w.x >= max(w.y, w.z)));
    }
  }
  if (!live(n)) { return best; }
  for (var f = 0u; f < 2u; f++) {
    if (!spanOn(n, f)) { continue; }
    let other = select(n + L.y, n + 1u, f == 0u);
    if (!live(other)) { continue; }
    // the two cells beside the span
    let side = select(n - 1u, n - L.y, f == 0u);
    let hasSide = select(col(n) > 0u, row(n) > 0u, f == 0u);
    if (faceValid(n) || (hasSide && faceValid(side))) { continue; }
    let h = rayCapsule(o, d, x[n].xyz, x[other].xyz, PK.viewport.z);
    if (h.x > 0.0 && h.x < best.t) { best = RayHit(h.x, select(n, other, h.y > 0.5)); }
  }
  return best;
}

fn project(p: vec3<f32>) -> vec4<f32> {
  let c = PK.viewProj * vec4<f32>(p, 1.0);
  let ndc = c.xyz / c.w;
  return vec4<f32>((ndc.x * 0.5 + 0.5) * PK.viewport.x, (0.5 - ndc.y * 0.5) * PK.viewport.y, ndc.z, c.w);
}

// Pass 1: nearest surface hit along the pointer ray.
@compute @workgroup_size(128)
fn pickRay(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let h = rayPrims(n);
  if (h.node != NONE) { atomicMin(&pick[0], bitcast<u32>(h.t)); }
}

// Pass 2: the material node of that hit (the lowest index on an exact tie).
@compute @workgroup_size(128)
fn pickRayNode(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let front = atomicLoad(&pick[0]);
  if (front == NONE) { return; }
  let h = rayPrims(n);
  if (h.node != NONE && bitcast<u32>(h.t) == front) { atomicMin(&pick[1], h.node); }
}

// Pass 3 (the pointer is over no surface: an opening, a strand, an edge): crossings within the cursor
// disc are candidates.
@compute @workgroup_size(128)
fn pickCandidates(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x || atomicLoad(&pick[0]) != NONE || !hasMaterial(n)) { return; }
  let p = project(x[n].xyz);
  if (p.w <= 0.0 || p.z <= 0.0 || p.z >= 1.0) { return; }
  if (distance(p.xy, PK.pointer.xy) > PK.pointer.z) { return; }
  let k = atomicAdd(&pick[2], 1u);
  if (k < MAX_CAND) { atomicStore(&pick[4u + k], n); }
}

// Pass 4: a candidate is hidden when an intact face lies on the line of sight in front of it. One
// thread per face; faces whose screen footprint misses the cursor disc are skipped at once.
@compute @workgroup_size(128)
fn pickOcclude(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= P.counts.x) { return; }
  let count = min(atomicLoad(&pick[2]), MAX_CAND);
  if (count == 0u || !faceValid(f)) { return; }
  let L = lat(f);
  let c = vec4<u32>(f, f + 1u, f + L.y, f + L.y + 1u);
  let p00 = x[c.x].xyz;
  let p10 = x[c.y].xyz;
  let p01 = x[c.z].xyz;
  let p11 = x[c.w].xyz;
  let s0 = project(p00);
  let s1 = project(p10);
  let s2 = project(p01);
  let s3 = project(p11);
  if (min(min(s0.w, s1.w), min(s2.w, s3.w)) <= 0.0) { return; }
  let lo = min(min(s0.xy, s1.xy), min(s2.xy, s3.xy)) - PK.pointer.z;
  let hi = max(max(s0.xy, s1.xy), max(s2.xy, s3.xy)) + PK.pointer.z;
  if (any(PK.pointer.xy < lo) || any(PK.pointer.xy > hi)) { return; }
  let o = PK.eye.xyz;
  for (var k = 0u; k < count; k++) {
    let m = atomicLoad(&pick[4u + k]);
    if (any(c == vec4<u32>(m))) { continue; }
    let to = x[m].xyz - o;
    let len = length(to);
    let d = to / len;
    let tmax = len - OCCLUDE_MARGIN;
    let h1 = rayTri(o, d, p00, p10, p11);
    let h2 = rayTri(o, d, p00, p11, p01);
    if ((h1.x > 0.0 && h1.x < tmax) || (h2.x > 0.0 && h2.x < tmax)) { atomicOr(&pick[4u + MAX_CAND + k], 1u); }
  }
}

// Pass 5: the visible candidate nearest to the pointer on screen.
@compute @workgroup_size(128)
fn pickVisible(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= min(atomicLoad(&pick[2]), MAX_CAND) || atomicLoad(&pick[4u + MAX_CAND + k]) != 0u) { return; }
  let n = atomicLoad(&pick[4u + k]);
  let d = distance(project(x[n].xyz).xy, PK.pointer.xy);
  let dk = min(4095u, u32(d * 64.0));
  atomicMin(&pick[3], (dk << 20u) | (n & 0xFFFFFu));
}

@compute @workgroup_size(1)
fn pickWrite() {
  let front = atomicLoad(&pick[0]);
  var p = vec3<f32>(0.0);
  var n = NONE;
  var kind = 0u;
  if (front != NONE) {
    n = atomicLoad(&pick[1]);
    p = PK.eye.xyz + bitcast<f32>(front) * PK.ray.xyz;
    kind = 1u;
  } else {
    let key = atomicLoad(&pick[3]);
    if (key != NONE) {
      n = key & 0xFFFFFu;
      p = x[n].xyz;
      kind = 2u;
    }
  }
  pickOut[0] = bitcast<u32>(p.x);
  pickOut[1] = bitcast<u32>(p.y);
  pickOut[2] = bitcast<u32>(p.z);
  pickOut[3] = n;
  pickOut[4] = kind;
}

// The current connectivity around the picked node (grip.ts gripPatch grows the patch through it): per
// lattice offset (di, dj) in [-R, R]^2, bit0 / bit1 = its own span along u / v survives, bit2 = not
// cloth, bit3 = outside the lattice.
@compute @workgroup_size(256)
fn pickWindow(@builtin(local_invocation_index) k: u32) {
  let n = pickOut[3];
  let R = u32(PK.viewport.w);
  let side = 2u * R + 1u;
  if (k >= side * side) { return; }
  if (pickOut[4] == 0u) {
    pickOut[8u + k] = 8u;
    return;
  }
  let L = lat(n);
  let i = i32(col(n)) + i32(k % side) - i32(R);
  let j = i32(row(n)) + i32(k / side) - i32(R);
  if (i < 0 || j < 0 || i >= i32(L.y) || j >= i32(L.z)) {
    pickOut[8u + k] = 8u;
    return;
  }
  let m = L.x + u32(j) * L.y + u32(i);
  var b = 0u;
  if (spanOn(m, 0u)) { b |= 1u; }
  if (spanOn(m, 1u)) { b |= 2u; }
  if (!live(m)) { b |= 4u; }
  pickOut[8u + k] = b;
}

// ---------------------------------------------------------------- nearest material (staged grabs)

// The crossing with material nearest to a point (PK.eye) within PK.viewport.z (m), as the proximity
// candidate key pickWrite reads (pick[3]: quantised distance << 20 | node).
@compute @workgroup_size(128)
fn pickNear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x || !hasMaterial(n)) { return; }
  let d = distance(x[n].xyz, PK.eye.xyz);
  if (d > PK.viewport.z) { return; }
  let dk = min(4095u, u32(d / PK.viewport.z * 4095.0));
  atomicMin(&pick[3], (dk << 20u) | (n & 0xFFFFFu));
}
