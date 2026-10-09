// Per-frame geometry preparation from the accepted mechanical state:
//  - nodeFrame: per-node unit normal from connected yarn neighbours (holes never bridge),
//  - classify: per-cell intact flag, compact lists of explicitly drawn spans and yarn tails,
//  - refine: bicubic (Catmull-Rom) surface samples on an S x S grid per cell with exact derivatives,
//  - args: indirect draw arguments from the list counters.

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> nodes: array<NodeStaticR>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> segw: array<vec4<u32>>; // Seg as 2 x vec4<u32>; flags at [1].y
@group(0) @binding(4) var<storage, read> tails: array<f32>;
@group(0) @binding(5) var<storage, read_write> nrm: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> cellFlags: array<u32>;
@group(0) @binding(7) var<storage, read_write> lists: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> rsurf: array<vec4<f32>>; // 3 per refined vertex
@group(0) @binding(9) var<storage, read_write> drawArgs: array<u32>;
@group(0) @binding(10) var<storage, read> wear: array<f32>;

// lists layout: [0] span count, [1] tail count, [2..3] unused, then spans at SPAN_BASE, tails at TAIL_BASE
const SPAN_BASE: u32 = 4u;
override MAX_SPANS: u32 = 65536u;
override MAX_TAILS: u32 = 65536u;
override YARN_VERTS: u32 = 18u;
override TAIL_VERTS: u32 = 14u;
override FUZZ_VERTS: u32 = 10u;
override FUZZ_PER_YARN: u32 = 10u;
override FUZZ_PER_TAIL: u32 = 14u;

fn segOk(owner: u32, f: u32) -> bool {
  let fl = segw[2u * (2u * owner + f) + 1u].y;
  return (fl & (SEG_EXISTS_R | SEG_BROKEN_R)) == SEG_EXISTS_R;
}

fn layerOf(n: u32) -> u32 { return nodes[n].ijl >> 24u; }
fn nodeI(n: u32) -> i32 { return i32(nodes[n].ijl & 0xFFFu); }
fn nodeJ(n: u32) -> i32 { return i32((nodes[n].ijl >> 12u) & 0xFFFu); }

// Tangent of family f at node n from connected neighbours; zero if isolated in that family.
fn famTangent(n: u32, f: u32) -> vec3<f32> {
  let L = R.layers[layerOf(n)].lattice;
  let stride = select(L.y, 1u, f == 0u);
  let a = select(nodeJ(n), nodeI(n), f == 0u);
  let len = select(i32(L.z), i32(L.y), f == 0u);
  var t = vec3<f32>(0.0);
  if (a + 1 < len && segOk(n, f)) { t += x[n + stride].xyz - x[n].xyz; }
  if (a > 0 && segOk(n - stride, f)) { t += x[n].xyz - x[n - stride].xyz; }
  return t;
}

@compute @workgroup_size(128)
fn nodeFrame(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= R.counts.x) { return; }
  let tu = famTangent(n, 0u);
  let tv = famTangent(n, 1u);
  let prev = nrm[n].xyz;
  var nn = cross(tu, tv);
  let l = length(nn);
  if (l > 1e-14) {
    nn = nn / l;
  } else {
    // one family missing: perpendicular to the remaining tangent, closest to the previous normal
    let t = select(tv, tu, dot(tu, tu) > dot(tv, tv));
    let tl = length(t);
    var p = select(vec3<f32>(0.0, 0.0, 1.0), prev, dot(prev, prev) > 0.5);
    if (tl > 1e-12) {
      let td = t / tl;
      p = p - dot(p, td) * td;
    }
    nn = normalize(p + vec3<f32>(0.0, 0.0, 1e-6));
  }
  nrm[n] = vec4<f32>(nn, 0.0);
}

fn cellOf(layer: u32, ci: u32, cj: u32) -> u32 {
  let L = R.layers[layer];
  return L.refine.z + cj * (L.lattice.y - 1u) + ci;
}

fn cellIntact(n: u32) -> bool {
  let L = R.layers[layerOf(n)].lattice;
  return segOk(n, 0u) && segOk(n, 1u) && segOk(n + 1u, 1u) && segOk(n + L.y, 0u);
}

// Per cell (lower-left node): 1 if all four spans carry material.
@compute @workgroup_size(128)
fn classifyCells(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= R.counts.x) { return; }
  let l = layerOf(n);
  let L = R.layers[l].lattice;
  let i = u32(nodeI(n));
  let j = u32(nodeJ(n));
  if (i + 1u >= L.y || j + 1u >= L.z) { return; }
  cellFlags[cellOf(l, i, j)] = select(0u, 1u, cellIntact(n));
}

fn cellOk(l: u32, ci: i32, cj: i32) -> bool {
  let L = R.layers[l].lattice;
  if (ci < 0 || cj < 0 || ci + 1 >= i32(L.y) || cj + 1 >= i32(L.z)) { return true; }
  return cellFlags[cellOf(l, u32(ci), u32(cj))] != 0u;
}

// Active spans next to a non-intact cell are drawn as explicit yarns; tails of broken or pulled-out
// yarn ends are listed per node side.
@compute @workgroup_size(128)
fn classifySpans(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= R.counts.x) { return; }
  let l = layerOf(n);
  let L = R.layers[l].lattice;
  let i = nodeI(n);
  let j = nodeJ(n);
  for (var f = 0u; f < 2u; f++) {
    let len = select(i32(L.z), i32(L.y), f == 0u);
    let a = select(j, i, f == 0u);
    if (a + 1 >= len || !segOk(n, f)) { continue; }
    // the two cells sharing span (n, f)
    var ok = true;
    if (f == 0u) { ok = cellOk(l, i, j) && cellOk(l, i, j - 1); } else { ok = cellOk(l, i, j) && cellOk(l, i - 1, j); }
    if (ok) { continue; }
    let k = atomicAdd(&lists[0], 1u);
    if (k < MAX_SPANS) { atomicStore(&lists[SPAN_BASE + k], 2u * n + f); }
  }
  for (var side = 0u; side < 4u; side++) {
    if (tails[4u * n + side] > 1e-6) {
      let k = atomicAdd(&lists[1], 1u);
      if (k < MAX_TAILS) { atomicStore(&lists[SPAN_BASE + MAX_SPANS + k], 4u * n + side); }
    }
  }
}

// Indirect draw arguments: [yarns, tails, yarn fuzz, tail fuzz] x (vertexCount, instanceCount, 0, 0).
@compute @workgroup_size(1)
fn args() {
  let spans = min(atomicLoad(&lists[0]), MAX_SPANS);
  let tl = min(atomicLoad(&lists[1]), MAX_TAILS);
  // instances per span / tail: the larger bundle (extra instances of the smaller family are empty)
  let b = bundleMax(R.layers[0]);
  drawArgs[0] = YARN_VERTS;
  drawArgs[1] = spans * b;
  drawArgs[2] = 0u;
  drawArgs[3] = 0u;
  drawArgs[4] = TAIL_VERTS;
  drawArgs[5] = tl * b;
  drawArgs[6] = 0u;
  drawArgs[7] = 0u;
  drawArgs[8] = FUZZ_VERTS;
  drawArgs[9] = spans * b * FUZZ_PER_YARN;
  drawArgs[10] = 0u;
  drawArgs[11] = 0u;
  drawArgs[12] = FUZZ_VERTS;
  drawArgs[13] = tl * b * FUZZ_PER_TAIL;
  drawArgs[14] = 0u;
  drawArgs[15] = 0u;
}

// ---------------------------------------------------------------- bicubic surface samples

fn cr(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
}
fn crd(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  return 0.5 * ((p2 - p0) + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t + 3.0 * (3.0 * p1 - p0 - 3.0 * p2 + p3) * t * t);
}

// Node position at lattice (i, j), linearly extrapolated beyond the lattice border.
fn latticeP(L: vec4<u32>, i: i32, j: i32) -> vec3<f32> {
  let nx = i32(L.y);
  let ny = i32(L.z);
  let ci = clamp(i, 0, nx - 1);
  let cj = clamp(j, 0, ny - 1);
  let base = x[L.x + u32(cj * nx + ci)].xyz;
  var p = base;
  if (i != ci) {
    let inner = x[L.x + u32(cj * nx + clamp(2 * ci - i, 0, nx - 1))].xyz;
    p = 2.0 * base - inner;
  }
  if (j != cj) {
    let ii = clamp(i, 0, nx - 1);
    let inner = x[L.x + u32(clamp(2 * cj - j, 0, ny - 1) * nx + ii)].xyz;
    p = p + (base - inner);
  }
  return p;
}

// One refined vertex: position, d/ds and d/dt over the cell (s along u, t along v), wear.
@compute @workgroup_size(128)
fn refine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= R.counts.w) { return; }
  // locate layer
  var l = 0u;
  for (var k = 1u; k < R.counts.y; k++) {
    if (r >= R.layers[k].refine.x) { l = k; }
  }
  let LR = R.layers[l];
  let L = LR.lattice;
  let S = LR.twill.z;
  let rl = r - LR.refine.x;
  let rnx = LR.refine.y;
  let I = rl % rnx;
  let J = rl / rnx;
  let ci = min(I / S, L.y - 2u);
  let cj = min(J / S, L.z - 2u);
  let s = f32(I - ci * S) / f32(S);
  let t = f32(J - cj * S) / f32(S);
  var rows: array<vec3<f32>, 4>;
  var drows: array<vec3<f32>, 4>;
  for (var q = 0; q < 4; q++) {
    let jj = i32(cj) - 1 + q;
    let p0 = latticeP(L, i32(ci) - 1, jj);
    let p1 = latticeP(L, i32(ci), jj);
    let p2 = latticeP(L, i32(ci) + 1, jj);
    let p3 = latticeP(L, i32(ci) + 2, jj);
    rows[q] = cr(p0, p1, p2, p3, s);
    drows[q] = crd(p0, p1, p2, p3, s);
  }
  let P = cr(rows[0], rows[1], rows[2], rows[3], t);
  let dPs = cr(drows[0], drows[1], drows[2], drows[3], t);
  let dPt = crd(rows[0], rows[1], rows[2], rows[3], t);
  let n00 = L.x + cj * L.y + ci;
  let w = mix(mix(wear[n00], wear[n00 + 1u], s), mix(wear[n00 + L.y], wear[n00 + L.y + 1u], s), t);
  rsurf[3u * r] = vec4<f32>(P, w);
  rsurf[3u * r + 1u] = vec4<f32>(dPs, 0.0);
  rsurf[3u * r + 2u] = vec4<f32>(dPt, 0.0);
}
