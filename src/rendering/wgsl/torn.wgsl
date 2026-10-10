// Per-frame geometry of torn cells (cells with a separated span), from the compact torn-cell list:
//  - tornGeo: per cell its broken edges, crack crossings and crack curves; per corner group
//    (render_common.wgsl) its free-edge curves and the 4 x 4 Catmull-Rom control net its leader
//    draws: the group's own corners are real, the others
//    rebuilt from the group's tangent frames at rest length, so a separating group carries its side of
//    the crack with it and is never stretched toward material it is no longer joined to;
//  - edgeArgs: indirect draw arguments that redraw the edge field only when the set of cracks changed
//    (an order-independent signature of the torn cells).

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> segw: array<vec4<u32>>; // Seg as 2 x vec4<u32>
@group(0) @binding(3) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> lists: array<u32>;
@group(0) @binding(5) var<storage, read_write> tornGeo: array<vec4<f32>>;
// [0] signature of this frame's torn cells, [1] signature the edge field was drawn for, [2] drawn once
@group(0) @binding(6) var<storage, read_write> edgeSig: array<atomic<u32>>;
// clear draw (4 x u32), crack draw (4 x u32)
@group(0) @binding(7) var<storage, read_write> edgeArgs: array<u32>;

override TORN_BASE: u32 = 131076u;
override MAX_TORN: u32 = 16384u;

fn tornCount() -> u32 { return min(lists[2], MAX_TORN); }

fn liveT(n: u32) -> bool { return x[n].w < 1.5; }

fn segOkT(owner: u32, f: u32, L: vec4<u32>) -> bool {
  let fl = segw[2u * (2u * owner + f) + 1u].y;
  if ((fl & (SEG_EXISTS_R | SEG_BROKEN_R)) != SEG_EXISTS_R) { return false; }
  return liveT(owner) && liveT(owner + select(L.y, 1u, f == 0u));
}

// Ring point of the 4 x 4 net beyond inner corner node n in direction (di, dj): the real neighbour when
// n is real and the span toward it holds, otherwise the extrapolation from the inner pair.
fn ringT(L: vec4<u32>, n: u32, real: bool, di: i32, dj: i32, P: vec3<f32>, Pin: vec3<f32>) -> vec3<f32> {
  let i = i32((n - L.x) % L.y);
  let j = i32((n - L.x) / L.y);
  let mi = i + di;
  let mj = j + dj;
  if (real && mi >= 0 && mj >= 0 && mi < i32(L.y) && mj < i32(L.z)) {
    let m = L.x + u32(mj) * L.y + u32(mi);
    let owner = select(n, m, di + dj < 0);
    if (segOkT(owner, select(1u, 0u, di != 0), L)) { return x[m].xyz; }
  }
  return 2.0 * P - Pin;
}

@compute @workgroup_size(128)
fn tornGeoMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x >> 2u;
  let c = gid.x & 3u;
  if (k >= tornCount()) { return; }
  let cell = lists[TORN_BASE + k];
  var l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (cell >= R.layers[q].refine.z) { l = q; }
  }
  let LR = R.layers[l];
  let L = LR.lattice;
  let local = cell - LR.refine.z;
  let ci = local % (L.y - 1u);
  let cj = local / (L.y - 1u);
  let n00 = L.x + cj * L.y + ci;
  var nodeC = array<u32, 4>(n00, n00 + 1u, n00 + L.y, n00 + L.y + 1u);
  // broken edges, crossings and rest lengths
  var mask = 0u;
  var crossings = 0u;
  var rest: array<f32, 4>;
  for (var e = 0u; e < 4u; e++) {
    let s = edgeSpan(n00, L.y, e);
    let owner = s >> 1u;
    let f = s & 1u;
    rest[e] = bitcast<f32>(segw[2u * s].x);
    if (!segOkT(owner, f, L)) {
      mask |= 1u << e;
      crossings |= u32(round(255.0 * crossingOf(bitcast<f32>(segw[2u * s + 1u].z)))) << (8u * e);
    }
  }
  let base = k * TORN_STRIDE;
  let size = vec2<f32>(f32(bundleOf(LR, 1u)) * LR.pitch.x, f32(bundleOf(LR, 0u)) * LR.pitch.y);
  let torn = makeTorn(mask, crossings, size);
  // each corner thread writes one of the (up to four) cracks
  if (c == 0u) {
    tornGeo[base] = vec4<f32>(bitcast<f32>(cell), bitcast<f32>(mask), bitcast<f32>(crossings), bitcast<f32>(l));
    atomicAdd(&edgeSig[0], hash11((cell * 0x9e3779b1u) ^ (mask << 28u) ^ hash11(crossings + 17u)));
  }
  if (c < crackCount(torn)) {
    let cr = packBezier(crackBezier(torn, c), false);
    tornGeo[base + TORN_CRACKS + 2u * c] = cr[0];
    tornGeo[base + TORN_CRACKS + 2u * c + 1u] = cr[1];
  }
  let slot = tornSlot(k, c);
  // corner c's group; only its leader draws it
  let g = groupOf(mask, ccwCorner(c));
  if (groupLeader(mask, g) != c || !liveT(nodeC[c])) {
    tornGeo[slot] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    return;
  }
  let gc = groupCurves(torn, g);
  for (var q = 0u; q < 4u; q++) { tornGeo[slot + 16u + q] = gc[q]; }
  var real = array<bool, 4>(false, false, false, false);
  var P: array<vec3<f32>, 4>;
  for (var q = 0u; q < 4u; q++) {
    real[q] = liveT(nodeC[q]) && (countOneBits(mask) < 2u || groupOf(mask, ccwCorner(q)) == g);
    P[q] = x[nodeC[q]].xyz;
  }
  // rebuild corners outside the group from adjacent group corners at rest length, then any corner
  // diagonal to a lone corner by parallelogram completion
  var have = real;
  for (var q = 0u; q < 4u; q++) {
    if (real[q]) { continue; }
    var est = vec3<f32>(0.0);
    var cnt = 0.0;
    // across the row edge (bottom e0 / top e2: differs in bit 0) and the column edge (left e3 / right e1)
    let rs = q ^ 1u;
    if (real[rs]) {
      let len = rest[select(0u, 2u, (q >> 1u) == 1u)];
      est += P[rs] + select(-1.0, 1.0, (q & 1u) == 1u) * len * tframe[2u * nodeC[rs]].xyz;
      cnt += 1.0;
    }
    let cs = q ^ 2u;
    if (real[cs]) {
      let len = rest[select(3u, 1u, (q & 1u) == 1u)];
      est += P[cs] + select(-1.0, 1.0, (q >> 1u) == 1u) * len * tframe[2u * nodeC[cs] + 1u].xyz;
      cnt += 1.0;
    }
    if (cnt > 0.0) {
      P[q] = est / cnt;
      have[q] = true;
    }
  }
  for (var q = 0u; q < 4u; q++) {
    if (!have[q]) { P[q] = P[q ^ 1u] + P[q ^ 2u] - P[q ^ 3u]; }
  }
  // 4 x 4 net, rows j - cj + 1, columns i - ci + 1
  var net: array<vec3<f32>, 16>;
  net[5] = P[0];
  net[6] = P[1];
  net[9] = P[2];
  net[10] = P[3];
  net[4] = ringT(L, nodeC[0], real[0], -1, 0, P[0], P[1]);
  net[7] = ringT(L, nodeC[1], real[1], 1, 0, P[1], P[0]);
  net[8] = ringT(L, nodeC[2], real[2], -1, 0, P[2], P[3]);
  net[11] = ringT(L, nodeC[3], real[3], 1, 0, P[3], P[2]);
  net[1] = ringT(L, nodeC[0], real[0], 0, -1, P[0], P[2]);
  net[2] = ringT(L, nodeC[1], real[1], 0, -1, P[1], P[3]);
  net[13] = ringT(L, nodeC[2], real[2], 0, 1, P[2], P[0]);
  net[14] = ringT(L, nodeC[3], real[3], 0, 1, P[3], P[1]);
  net[0] = net[1] + net[4] - net[5];
  net[3] = net[2] + net[7] - net[6];
  net[12] = net[13] + net[8] - net[9];
  net[15] = net[14] + net[11] - net[10];
  for (var q = 0u; q < 16u; q++) { tornGeo[slot + q] = vec4<f32>(net[q], 1.0); }
}

@compute @workgroup_size(1)
fn edgeArgsMain() {
  let count = tornCount();
  let sig = atomicLoad(&edgeSig[0]) ^ hash11(count + 0x51ed27u);
  let changed = sig != atomicLoad(&edgeSig[1]) || atomicLoad(&edgeSig[2]) == 0u;
  atomicStore(&edgeSig[1], sig);
  atomicStore(&edgeSig[2], 1u);
  edgeArgs[0] = 3u;
  edgeArgs[1] = select(0u, 1u, changed);
  edgeArgs[2] = 0u;
  edgeArgs[3] = 0u;
  edgeArgs[4] = 6u;
  edgeArgs[5] = select(0u, 4u * count, changed);
  edgeArgs[6] = 0u;
  edgeArgs[7] = 0u;
}
