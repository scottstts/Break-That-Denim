// Picking and grip terms. A grip is a finite material patch attached to a handle by quadratic
// springs (exact for Newton). Its force is bounded by clamping the handle's lead over the patch
// centroid once per substep (gripLead); gradient and stiffness enter the Newton system via ext.
// Up to GRIPS grips (one per finger) hold disjoint patches: entries of grip g occupy
// [g * SLOT_ENTRIES, g * SLOT_ENTRIES + count), and no node is in two patches, so every ext entry
// has a single writer.

const GRIP_ADDS: bool = true;
const GRIPS: u32 = 4u;
const SLOT_ENTRIES: u32 = 512u;

struct GripSlot {
  // handle target (xyz), entry count (w as f32)
  goal: vec4<f32>,
  // stiffness per unit weight (N/m), largest handle lead (m), picked node (bitcast), pinch scale of the
  // grab-time offsets (grip.ts pinchScale)
  params: vec4<f32>,
};

struct GripU {
  slots: array<GripSlot, GRIPS>,
};

struct PickU {
  viewProj: mat4x4<f32>,
  // pointer x, y (drawing-buffer pixels), radius (pixels), unused
  pointer: vec4<f32>,
  // width, height, layer depth tolerance (NDC depth), unused
  viewport: vec4<f32>,
};

struct GripEntry {
  node: u32,
  w: f32,
  pad0: f32,
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
@group(0) @binding(9) var<storage, read_write> pick: array<atomic<u32>>; // [depthKey, distKey]
@group(0) @binding(10) var<storage, read_write> pickOut: array<vec4<f32>>;
// [g] effective handle of grip g (xyz)
@group(0) @binding(11) var<storage, read_write> gripGoal: array<vec4<f32>>;
// x: the grip whose entries gripInit initialises
@group(0) @binding(12) var<uniform> InitU: vec4<u32>;

var<workgroup> leadSum: array<vec4<f32>, 256>;

// Once per substep, from committed positions: each grip's effective handle is the requested one,
// moved to within the largest lead of its weighted patch centroid (bounded total force). One
// workgroup per grip.
@compute @workgroup_size(256)
fn gripLead(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let g = wg.x;
  let S = G.slots[g];
  let count = u32(S.goal.w);
  if (count == 0u) { return; }
  var acc = vec4<f32>(0.0);
  for (var k = t; k < count; k += 256u) {
    let e = grip[g * SLOT_ENTRIES + k];
    acc += vec4<f32>(e.w * (x[e.node].xyz - S.params.w * e.offset.xyz), e.w);
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
    let c = leadSum[0].xyz / max(leadSum[0].w, 1e-12);
    var lead = S.goal.xyz - c;
    let l = length(lead);
    if (l > S.params.y) { lead *= S.params.y / l; }
    gripGoal[g] = vec4<f32>(c + lead, 0.0);
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
  let y = x[e.node].xyz + dx[e.node].xyz;
  let d = y - (gripGoal[g].xyz + S.params.w * e.offset.xyz);
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

// Offsets of a grip's patch nodes from its picked node at grab time.
@compute @workgroup_size(64)
fn gripInit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let g = InitU.x;
  let S = G.slots[g];
  if (gid.x >= u32(S.goal.w)) { return; }
  let k = g * SLOT_ENTRIES + gid.x;
  let picked = bitcast<u32>(S.params.z);
  grip[k].offset = vec4<f32>(x[grip[k].node].xyz - x[picked].xyz, 0.0);
}

fn hasMaterial(n: u32) -> bool {
  // a cloth crossing with at least one intact span (debris and isolated fragments are not grab targets)
  if (x[n].w > 0.5) { return false; }
  let a = segs[2u * n].flags;
  let b = segs[2u * n + 1u].flags;
  return ((a & 3u) == 1u) || ((b & 3u) == 1u);
}

fn project(n: u32) -> vec3<f32> {
  let c = PK.viewProj * vec4<f32>(x[n].xyz, 1.0);
  let ndc = c.xyz / c.w;
  return vec3<f32>((ndc.x * 0.5 + 0.5) * PK.viewport.x, (0.5 - ndc.y * 0.5) * PK.viewport.y, ndc.z);
}

@compute @workgroup_size(128)
fn pickDepth(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x || !hasMaterial(n)) { return; }
  let p = project(n);
  if (p.z <= 0.0 || p.z >= 1.0) { return; }
  if (distance(p.xy, PK.pointer.xy) > PK.pointer.z) { return; }
  atomicMin(&pick[0], u32(p.z * 16777215.0));
}

@compute @workgroup_size(128)
fn pickNearest(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x || !hasMaterial(n)) { return; }
  let p = project(n);
  if (p.z <= 0.0 || p.z >= 1.0) { return; }
  let d = distance(p.xy, PK.pointer.xy);
  if (d > PK.pointer.z) { return; }
  let front = f32(atomicLoad(&pick[0])) / 16777215.0;
  if (p.z > front + PK.viewport.z) { return; }
  let dk = min(4095u, u32(d * 64.0));
  atomicMin(&pick[1], (dk << 20u) | (n & 0xFFFFFu));
}

@compute @workgroup_size(1)
fn pickWrite() {
  let key = atomicLoad(&pick[1]);
  if (key == 0xFFFFFFFFu) {
    pickOut[0] = vec4<f32>(-1.0);
    return;
  }
  let n = key & 0xFFFFFu;
  pickOut[0] = vec4<f32>(x[n].xyz, bitcast<f32>(n));
}
