// Per-frame geometry of torn cells (cells with a crossing edge), from the compact torn-cell list:
//  - tornInfoMain: per cell its crossing edges, the corners that own no cloth, and which group owns
//    each of its crossings: flooded from each group's corners through uncut yarn segments of the yarn
//    damage field (render_common.wgsl, bit-parallel over the rows of the cell's crossing grid);
//  - tornGeoMain: per corner group the 4 x 4 Catmull-Rom control net its leader draws: the group's
//    own corners are real, the others rebuilt from the group's tangent frames at rest length, so a
//    separating group carries its yarns with it and is never stretched toward material it is no longer
//    joined to.

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> segw: array<vec4<u32>>; // Seg as 2 x vec4<u32>
@group(0) @binding(3) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> lists: array<u32>;
@group(0) @binding(5) var<storage, read_write> tornGeo: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> field: array<u32>;

override TORN_BASE: u32 = 131076u;
override MAX_TORN: u32 = 16384u;

fn tornCount() -> u32 { return min(lists[2], MAX_TORN); }

fn liveT(n: u32) -> bool { return x[n].w < 1.5; }

fn segOkT(owner: u32, f: u32, L: vec4<u32>) -> bool {
  let fl = segw[2u * (2u * owner + f) + 1u].y;
  if ((fl & (SEG_EXISTS_R | SEG_BROKEN_R)) != SEG_EXISTS_R) { return false; }
  return liveT(owner) && liveT(owner + select(L.y, 1u, f == 0u));
}

struct CellRef {
  l: u32,
  ci: u32,
  cj: u32,
  n00: u32,
};

fn cellRef(cell: u32) -> CellRef {
  var o: CellRef;
  o.l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (cell >= R.layers[q].refine.z) { o.l = q; }
  }
  let LR = R.layers[o.l];
  let local = cell - LR.refine.z;
  o.ci = local % (LR.lattice.y - 1u);
  o.cj = local / (LR.lattice.y - 1u);
  o.n00 = LR.lattice.x + o.cj * LR.lattice.y + o.ci;
  return o;
}

fn cornerNode(c: CellRef, q: u32) -> u32 { return c.n00 + (q & 1u) + (q >> 1u) * R.layers[c.l].lattice.y; }

@compute @workgroup_size(128)
fn tornInfoMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x >> 2u;
  if (k >= tornCount() || (gid.x & 3u) != 0u) { return; }
  let cell = lists[TORN_BASE + k];
  let c = cellRef(cell);
  let LR = R.layers[c.l];
  let L = LR.lattice;
  var dead = 0u;
  for (var q = 0u; q < 4u; q++) {
    if (!liveT(cornerNode(c, q))) { dead |= 1u << q; }
  }
  // crossing edges: no longer joining two live corners, with a live end
  var mask = 0u;
  for (var e = 0u; e < 4u; e++) {
    let s = edgeSpan(c.n00, L.y, e);
    let da = ((dead >> ccwCorner(e)) & 1u) == 1u;
    let db = ((dead >> ccwCorner(e + 1u)) & 1u) == 1u;
    if ((!da || !db) && (da || db || !segOkT(s >> 1u, s & 1u, L))) { mask |= 1u << e; }
  }
  let base = k * TORN_STRIDE;
  tornGeo[base] = vec4<f32>(bitcast<f32>(cell), bitcast<f32>(makeMeta(mask, dead)), 0.0, bitcast<f32>(c.l));

  // the cell's crossings: where the weave is intact (both yarns), where a lone yarn remains, and the
  // links their intact segments make. Groups own the woven crossings they reach and, up to two
  // crossings beyond them, the yarn ends sticking out of their edge; yarns with no weave around them belong to
  // loose bundles (threads) or have fallen away.
  let g = cellGrid(LR, c.ci, c.cj);
  var woven: array<u32, 8>;
  var mat: array<u32, 8>;
  var linkH: array<u32, 8>;
  var linkV: array<u32, 8>;
  for (var r = 0u; r < CELL_MAX_M; r++) {
    woven[r] = 0u;
    mat[r] = 0u;
    linkH[r] = 0u;
    linkV[r] = 0u;
  }
  var warpPrev = 0u;
  for (var r = 0u; r < g.w; r++) {
    var pickPrev = false;
    var warpRow = 0u;
    for (var q = 0u; q < g.z; q++) {
      let w = field[fieldIndexR(LR, g.x + q, g.y + r)];
      let pick = yarnCode(w, 0u) != CUT_R;
      let warp = yarnCode(w, 1u) != CUT_R;
      if (pick || warp) { mat[r] |= 1u << q; }
      if (pick && warp) { woven[r] |= 1u << q; }
      // a pick links neighbouring crossings of its row where both of its segments are present
      if (pick && pickPrev) { linkH[r] |= 1u << (q - 1u); }
      pickPrev = pick;
      if (warp) { warpRow |= 1u << q; }
    }
    // a warp links rows r - 1 and r where it is present at both
    if (r > 0u) { linkV[r - 1u] = warpPrev & warpRow; }
    warpPrev = warpRow;
  }
  // flood each group from its corners (the 2 x 2 crossings nearest each of them); a crossing belongs
  // to the first group that reaches it
  var labels: array<u32, 16>;
  for (var q = 0u; q < 16u; q++) { labels[q] = 0xFFFFFFFFu; }
  var taken: array<u32, 8>;
  for (var r = 0u; r < CELL_MAX_M; r++) { taken[r] = 0u; }
  let lastK = g.z - 1u;
  let lastM = g.w - 1u;
  for (var cq = 0u; cq < 4u; cq++) {
    if (((dead >> cq) & 1u) == 1u) { continue; }
    let grp = groupOf(mask, ccwCorner(cq));
    if (groupLeader(mask, grp) != cq) { continue; }
    var reg: array<u32, 8>;
    for (var r = 0u; r < CELL_MAX_M; r++) { reg[r] = 0u; }
    for (var q = 0u; q < 4u; q++) {
      if (((dead >> q) & 1u) == 1u || (countOneBits(mask) >= 2u && groupOf(mask, ccwCorner(q)) != grp)) { continue; }
      let row = select(0u, lastM - 1u, (q >> 1u) == 1u);
      let colBits = select(3u, 3u << (lastK - 1u), (q & 1u) == 1u);
      reg[row] |= colBits & woven[row] & ~taken[row];
      reg[row + 1u] |= colBits & woven[row + 1u] & ~taken[row + 1u];
    }
    for (var it = 0u; it < 32u; it++) {
      var changed = false;
      for (var r = 0u; r < g.w; r++) {
        let h = linkH[r];
        var v = reg[r];
        // along the row through linked neighbours, a few columns per pass
        for (var s = 0u; s < 4u; s++) { v |= ((v & h) << 1u) | ((v >> 1u) & h); }
        if (r > 0u) { v |= reg[r - 1u] & linkV[r - 1u]; }
        if (r + 1u < g.w) { v |= reg[r + 1u] & linkV[r]; }
        v &= woven[r] & ~taken[r];
        if (v != reg[r]) {
          reg[r] = v;
          changed = true;
        }
      }
      if (!changed) { break; }
    }
    // yarn ends up to two crossings beyond the woven region: a worn-through float (at most three
    // crossings of lone pick) stays part of the cloth, a longer run of lone yarn is a thread
    var fringe: array<u32, 8>;
    for (var step = 0u; step < 2u; step++) {
      for (var r = 0u; r < g.w; r++) {
        let h = linkH[r];
        var v = ((reg[r] & h) << 1u) | ((reg[r] >> 1u) & h);
        if (r > 0u) { v |= reg[r - 1u] & linkV[r - 1u]; }
        if (r + 1u < g.w) { v |= reg[r + 1u] & linkV[r]; }
        fringe[r] = v & mat[r] & ~woven[r] & ~taken[r];
      }
      for (var r = 0u; r < g.w; r++) { reg[r] |= fringe[r]; }
    }
    for (var r = 0u; r < g.w; r++) {
      taken[r] |= reg[r];
      for (var q = 0u; q < g.z; q++) {
        if (((reg[r] >> q) & 1u) == 0u) { continue; }
        let idx = r * CELL_MAX_K + q;
        let word = idx >> 3u;
        let sh = 4u * (idx & 7u);
        labels[word] = (labels[word] & ~(0xFu << sh)) | (cq << sh);
      }
    }
  }
  for (var q = 0u; q < 4u; q++) {
    tornGeo[base + TORN_LABELS + q] = bitcast<vec4<f32>>(vec4<u32>(labels[4u * q], labels[4u * q + 1u], labels[4u * q + 2u], labels[4u * q + 3u]));
  }
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
  let base = k * TORN_STRIDE;
  let info = tornInfo(tornGeo[base]);
  let cm = info.y;
  let mask = metaMask(cm);
  let dead = metaDead(cm);
  let cr = cellRef(info.x);
  let L = R.layers[cr.l].lattice;
  var nodeC = array<u32, 4>(cr.n00, cr.n00 + 1u, cr.n00 + L.y, cr.n00 + L.y + 1u);
  var rest: array<f32, 4>;
  for (var e = 0u; e < 4u; e++) { rest[e] = bitcast<f32>(segw[2u * edgeSpan(cr.n00, L.y, e)].x); }
  let slot = tornSlot(k, c);
  // corner c's group; only its leader draws it
  let g = groupOf(mask, ccwCorner(c));
  if (((dead >> c) & 1u) == 1u || groupLeader(mask, g) != c) {
    tornGeo[slot] = vec4<f32>(0.0, 0.0, 0.0, -1.0);
    return;
  }
  var real = array<bool, 4>(false, false, false, false);
  var P: array<vec3<f32>, 4>;
  for (var q = 0u; q < 4u; q++) {
    real[q] = ((dead >> q) & 1u) == 0u && (countOneBits(mask) < 2u || groupOf(mask, ccwCorner(q)) == g);
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
