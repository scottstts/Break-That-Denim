// Per-frame geometry preparation from the accepted mechanical state:
//  - nodeFrame: per-node unit normal and +u / +v tangents from connected yarn neighbours (holes never
//    bridge; a missing direction completes the frame and an isolated node keeps its last frame),
//  - classify: per-cell intact flag, compact lists of torn cells, explicitly drawn spans and yarn tails,
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
// per node: +u unit tangent with the node's wear in w (static), +v unit tangent
@group(0) @binding(10) var<storage, read_write> tframe: array<vec4<f32>>;

// lists layout: [0] span count, [1] tail count, [2] torn cell count, [3] unused, then spans at
// SPAN_BASE, tails after MAX_SPANS entries, torn cells after MAX_TAILS more
const SPAN_BASE: u32 = 4u;
override MAX_SPANS: u32 = 65536u;
override MAX_TAILS: u32 = 65536u;
override MAX_TORN: u32 = 32768u;
override TORN_VERTS: u32 = 24u;
override TAIL_VERTS: u32 = 14u;
override FUZZ_VERTS: u32 = 10u;
override FUZZ_PER_YARN: u32 = 10u;
override FUZZ_PER_TAIL: u32 = 12u;

// Node still in the scene (retired debris, x.w = 2, is not drawn).
fn live(n: u32) -> bool { return x[n].w < 1.5; }

fn segOk(owner: u32, f: u32) -> bool {
  let fl = segw[2u * (2u * owner + f) + 1u].y;
  if ((fl & (SEG_EXISTS_R | SEG_BROKEN_R)) != SEG_EXISTS_R) { return false; }
  let L = R.layers[layerOf(owner)].lattice;
  return live(owner) && live(owner + select(L.y, 1u, f == 0u));
}

fn layerOf(n: u32) -> u32 { return nodes[n].ijl >> 24u; }

// Bundled yarns of span (owner, f) that are absent or separated (bit k = yarn k).
fn yarnMask(owner: u32, f: u32) -> u32 { return (segw[2u * (2u * owner + f) + 1u].y >> 8u) & 0xFu; }
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
    tframe[2u * n] = vec4<f32>(normalize(tu), tframe[2u * n].w);
    tframe[2u * n + 1u] = vec4<f32>(normalize(tv), 0.0);
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
    // complete the tangent frame (+u x +v = normal) from the remaining family, else keep the last one
    var eu = tframe[2u * n].xyz;
    var ev = tframe[2u * n + 1u].xyz;
    if (dot(tu, tu) > 1e-20) {
      eu = normalize(tu);
      ev = normalize(cross(nn, eu));
    } else if (dot(tv, tv) > 1e-20) {
      ev = normalize(tv);
      eu = normalize(cross(ev, nn));
    }
    tframe[2u * n] = vec4<f32>(eu, tframe[2u * n].w);
    tframe[2u * n + 1u] = vec4<f32>(ev, 0.0);
  }
  nrm[n] = vec4<f32>(nn, 0.0);
}

fn cellOf(layer: u32, ci: u32, cj: u32) -> u32 {
  let L = R.layers[layer];
  return L.refine.z + cj * (L.lattice.y - 1u) + ci;
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
  // bit 0: the cell is drawn as surface (all four spans carry material); bits 8..23: yarn masks of
  // its left / right warp spans and bottom / top pick spans (yarns gone from the woven surface)
  var flags = 0u;
  let cell = cellOf(l, i, j);
  if (segOk(n, 0u) && segOk(n, 1u) && segOk(n + 1u, 1u) && segOk(n + L.y, 0u)) {
    flags = 1u | (yarnMask(n, 1u) << 8u) | (yarnMask(n + 1u, 1u) << 12u) | (yarnMask(n, 0u) << 16u) | (yarnMask(n + L.y, 0u) << 20u);
  } else if (live(n) || live(n + 1u) || live(n + L.y) || live(n + L.y + 1u)) {
    // torn: each live corner draws its own quarter of the cell (surface.wgsl vsTorn)
    let k = atomicAdd(&lists[2], 1u);
    if (k < MAX_TORN) { atomicStore(&lists[SPAN_BASE + MAX_SPANS + MAX_TAILS + k], cell); }
  }
  cellFlags[cell] = flags;
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
    // the two cells sharing span (n, f): on its +lateral side (+v for u-spans, +u for v-spans) and -side
    var plus = true;
    var minus = true;
    if (f == 0u) {
      plus = cellOk(l, i, j);
      minus = cellOk(l, i, j - 1);
    } else {
      plus = cellOk(l, i, j);
      minus = cellOk(l, i - 1, j);
    }
    if (plus && minus) { continue; }
    let k = atomicAdd(&lists[0], 1u);
    // bits 30 / 31: the +/- side cell is intact surface (its yarns are drawn there, not here)
    if (k < MAX_SPANS) { atomicStore(&lists[SPAN_BASE + k], (2u * n + f) | (select(0u, 1u, plus) << 30u) | (select(0u, 1u, minus) << 31u)); }
  }
  for (var side = 0u; side < 4u; side++) {
    if (live(n) && tails[4u * n + side] > 1e-6) {
      let k = atomicAdd(&lists[1], 1u);
      if (k < MAX_TAILS) { atomicStore(&lists[SPAN_BASE + MAX_SPANS + k], 4u * n + side); }
    }
  }
}

// Indirect draw arguments: [unused, tails, yarn fuzz, tail fuzz, torn corners] x (vertexCount,
// instanceCount, 0, 0). Fuzz on a listed span's yarns is drawn per half yarn.
@compute @workgroup_size(1)
fn args() {
  let spans = min(atomicLoad(&lists[0]), MAX_SPANS);
  let tl = min(atomicLoad(&lists[1]), MAX_TAILS);
  // instances per span / tail: the larger bundle (extra instances of the smaller family are empty)
  let b = bundleMax(R.layers[0]);
  drawArgs[0] = 0u;
  drawArgs[1] = 0u;
  drawArgs[2] = 0u;
  drawArgs[3] = 0u;
  drawArgs[4] = TAIL_VERTS;
  drawArgs[5] = tl * b;
  drawArgs[6] = 0u;
  drawArgs[7] = 0u;
  drawArgs[8] = FUZZ_VERTS;
  drawArgs[9] = spans * b * 2u * FUZZ_PER_YARN;
  drawArgs[10] = 0u;
  drawArgs[11] = 0u;
  drawArgs[12] = FUZZ_VERTS;
  drawArgs[13] = tl * b * FUZZ_PER_TAIL;
  drawArgs[14] = 0u;
  drawArgs[15] = 0u;
  drawArgs[16] = TORN_VERTS;
  drawArgs[17] = min(atomicLoad(&lists[2]), MAX_TORN) * 4u;
  drawArgs[18] = 0u;
  drawArgs[19] = 0u;
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

fn nodeAt(L: vec4<u32>, i: u32, j: u32) -> u32 { return L.x + j * L.y + i; }

// The 4 x 4 Catmull-Rom control net of an intact cell (ci, cj): its own 2 x 2 corners plus the ring
// around them. A ring point is used only when a surviving span connects it to the cell's corner;
// otherwise (lattice border, torn span, debris) it is extrapolated from the cell, so a hole or a
// falling piece never pulls the intact surface toward it.
fn controlNet(L: vec4<u32>, ci: u32, cj: u32) -> array<vec3<f32>, 16> {
  var P: array<vec3<f32>, 16>;
  let nx = L.y;
  let ny = L.z;
  // inner rows q = 1, 2 (j = cj, cj + 1)
  for (var q = 1u; q < 3u; q++) {
    let j = cj + q - 1u;
    let a = nodeAt(L, ci, j);
    let b = a + 1u;
    let pa = x[a].xyz;
    let pb = x[b].xyz;
    P[4u * q + 1u] = pa;
    P[4u * q + 2u] = pb;
    P[4u * q] = select(2.0 * pa - pb, x[a - 1u].xyz, ci > 0u && segOk(a - 1u, 0u));
    P[4u * q + 3u] = select(2.0 * pb - pa, x[b + 1u].xyz, ci + 2u < nx && segOk(b, 0u));
  }
  // outer rows q = 0 (j = cj - 1) and q = 3 (j = cj + 2)
  for (var c = 1u; c < 3u; c++) {
    let i = ci + c - 1u;
    let a = nodeAt(L, i, cj);
    let b = nodeAt(L, i, cj + 1u);
    P[c] = select(2.0 * P[4u + c] - P[8u + c], x[a - nx].xyz, cj > 0u && segOk(a - nx, 1u));
    P[12u + c] = select(2.0 * P[8u + c] - P[4u + c], x[b + nx].xyz, cj + 2u < ny && segOk(b, 1u));
  }
  // corners of the ring: parallelogram completion
  P[0] = P[1] + P[4] - P[5];
  P[3] = P[2] + P[7] - P[6];
  P[12] = P[13] + P[8] - P[9];
  P[15] = P[14] + P[11] - P[10];
  return P;
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
  let net = controlNet(L, ci, cj);
  var rows: array<vec3<f32>, 4>;
  var drows: array<vec3<f32>, 4>;
  for (var q = 0u; q < 4u; q++) {
    rows[q] = cr(net[4u * q], net[4u * q + 1u], net[4u * q + 2u], net[4u * q + 3u], s);
    drows[q] = crd(net[4u * q], net[4u * q + 1u], net[4u * q + 2u], net[4u * q + 3u], s);
  }
  let P = cr(rows[0], rows[1], rows[2], rows[3], t);
  let dPs = cr(drows[0], drows[1], drows[2], drows[3], t);
  let dPt = crd(rows[0], rows[1], rows[2], rows[3], t);
  let n00 = L.x + cj * L.y + ci;
  let w = mix(mix(tframe[2u * n00].w, tframe[2u * (n00 + 1u)].w, s), mix(tframe[2u * (n00 + L.y)].w, tframe[2u * (n00 + L.y + 1u)].w, s), t);
  rsurf[3u * r] = vec4<f32>(P, w);
  rsurf[3u * r + 1u] = vec4<f32>(dPs, 0.0);
  rsurf[3u * r + 2u] = vec4<f32>(dPt, 0.0);
}
