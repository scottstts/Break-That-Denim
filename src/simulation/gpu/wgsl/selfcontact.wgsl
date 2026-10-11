// Self-contact of a cloth layer: a node against a face (an intact lattice cell, as the two triangles
// c00 c10 c11 and c00 c11 c01). Penalty E = k (delta - d)^3 / (3 delta) on the distance d from the node
// to the face: force k s^2 / delta and stiffness 2 k s / delta at depth s = delta - d, both zero where
// contact begins (the surface fibres compress progressively; the Newton step is limited to where that
// linearization holds, terms). Lagged-normal regularized Coulomb friction acts on their relative step.
// The node takes the force, the face's corners the reaction by the closest point's barycentric weights
// (momentum is conserved). Contributions are scattered with fixed-point integer atomics (WGSL has no float
// atomics) and gathered into ext; the Newton operator sees them as diagonal blocks.
//
// Faces are hashed by centroid once per substep at the predicted iterate (and at the start of the
// substep when that is another cell). Detection keeps each node's K nearest faces within reach and an
// OGC-style directional bound, relative between node and face: against the first face the node's
// predicted path crosses, else the nearest face at the start of the substep, the node's step may close
// only a share of the clearance down to a hard core, and the face's corners may close no more than the
// same share toward it (resolveBounds). Together they never close the gap, so no force (a hand pulling
// cloth up through a flap, a flap swinging onto a strand) carries one through the other, while sliding
// stays free. Faces within two lattice cells of a node, and still linked to it through surviving
// spans, are ignored: on a 3.4 mm lattice they cannot meet it short of an impossibly tight fold, and
// bounding against them would throttle every fast pull of the sheet. Across a tear they are another
// lip or flap and are tested like any other face.

const K: u32 = 4u;
const NONE: u32 = 0xFFFFFFFFu;
// fixed-point scales of the accumulated gradient (N) and Hessian (N/m)
const GS: f32 = 1e8;
const HS: f32 = 1e3;
// accumulator words per node: gradient xyz, Hessian xx xy xz yy yz zz
const ACC: u32 = 9u;
// separation a node never closes toward a face within one substep, m
const HARD_CORE: f32 = 0.3e-3;

struct SelfU {
  // hash cell size (m), table mask (u32 bits), detection reach (m), contact distance delta (m)
  grid: vec4<f32>,
  // penalty stiffness (N/m), friction coefficient, friction regularization speed (m/s), bound fraction
  law: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<uniform> SU: SelfU;
@group(0) @binding(2) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> dx: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> segs: array<Seg>;
// hash table: [0, T) cell counts, scanned into start offsets; [T, 2T) fill cursors
@group(0) @binding(6) var<storage, read_write> cells: array<atomic<u32>>;
// per hashed face, 2 vec4: its centroid now and a radius bounding it now and at the start of the
// substep (w), its centroid at the start of the substep and the face index (w, exact below 2^24)
@group(0) @binding(7) var<storage, read_write> items: array<vec4<f32>>;
// K candidate faces per node, then per node: the reciprocal-bound key (detect, resolveBounds) and the
// damage mark of its lattice row (damageRow)
@group(0) @binding(8) var<storage, read_write> pairs: array<atomic<u32>>;
// 2 per node: the node's own bound (unit normal away from the face, largest approach along it) and the
// reciprocal bound of a face corner (direction toward the bounded node, largest motion along it);
// zero normal: unbounded
@group(0) @binding(9) var<storage, read_write> cmeta: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> ext: array<vec4<f32>>; // 3 per node

fn tableMask() -> u32 { return bitcast<u32>(SU.grid.y); }
fn cur(n: u32) -> vec3<f32> { return x[n].xyz + dx[n].xyz; }
// a node of the solve: cloth or a detached piece (not retired)
fn cloth(n: u32) -> bool { return x[n].w < 1.5; }

fn hashCell(c: vec3<i32>) -> u32 {
  let h = (u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u);
  return h & tableMask();
}
fn cellOf(p: vec3<f32>) -> vec3<i32> { return vec3<i32>(floor(p / SU.grid.x)); }

// Corners (c00, c10, c01, c11) of the face with lower-left node f.
fn corners(f: u32) -> vec4<u32> {
  let L = P.layers[nodes[f].ijl >> 24u].lattice;
  return vec4<u32>(f, f + 1u, f + L.y, f + L.y + 1u);
}

// An intact cell of the cloth: all four spans hold and all corners are in the solve.
fn faceValid(f: u32) -> bool {
  let ijl = nodes[f].ijl;
  let L = P.layers[ijl >> 24u].lattice;
  let i = ijl & 0xFFFu;
  let j = (ijl >> 12u) & 0xFFFu;
  if (i + 1u >= L.y || j + 1u >= L.z) { return false; }
  let c = corners(f);
  if (!cloth(c.x) || !cloth(c.y) || !cloth(c.z) || !cloth(c.w)) { return false; }
  return segActive(segs[2u * c.x]) && segActive(segs[2u * c.x + 1u]) && segActive(segs[2u * c.y + 1u]) && segActive(segs[2u * c.z]);
}

fn centroid(c: vec4<u32>) -> vec3<f32> { return 0.25 * (cur(c.x) + cur(c.y) + cur(c.z) + cur(c.w)); }

// Closest point of triangle (a, b, c) to p: barycentric weights (wa, wb, wc) and squared distance.
fn closestTri(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec4<f32> {
  let ab = b - a;
  let ac = c - a;
  let ap = p - a;
  let d1 = dot(ab, ap);
  let d2 = dot(ac, ap);
  var w = vec3<f32>(1.0, 0.0, 0.0);
  if (d1 <= 0.0 && d2 <= 0.0) {
    w = vec3<f32>(1.0, 0.0, 0.0);
  } else {
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    let vc = d1 * d4 - d3 * d2;
    let vb = d5 * d2 - d1 * d6;
    let va = d3 * d6 - d5 * d4;
    if (d3 >= 0.0 && d4 <= d3) {
      w = vec3<f32>(0.0, 1.0, 0.0);
    } else if (d6 >= 0.0 && d5 <= d6) {
      w = vec3<f32>(0.0, 0.0, 1.0);
    } else if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
      let v = d1 / (d1 - d3);
      w = vec3<f32>(1.0 - v, v, 0.0);
    } else if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
      let v = d2 / (d2 - d6);
      w = vec3<f32>(1.0 - v, 0.0, v);
    } else if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
      let v = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      w = vec3<f32>(0.0, 1.0 - v, v);
    } else {
      let den = 1.0 / (va + vb + vc);
      let wb = vb * den;
      let wc = vc * den;
      w = vec3<f32>(1.0 - wb - wc, wb, wc);
    }
  }
  let q = w.x * a + w.y * b + w.z * c;
  let e = p - q;
  return vec4<f32>(w, dot(e, e));
}

struct FaceHit {
  // corner weights (c00, c10, c01, c11)
  w: vec4<f32>,
  q: vec3<f32>,
  d: f32,
};

fn faceHit(p: vec3<f32>, c: vec4<u32>) -> FaceHit {
  return faceHitP(p, cur(c.x), cur(c.y), cur(c.z), cur(c.w));
}

// The same at the start of the substep (accepted positions).
fn faceHitAt(p: vec3<f32>, c: vec4<u32>) -> FaceHit {
  return faceHitP(p, x[c.x].xyz, x[c.y].xyz, x[c.z].xyz, x[c.w].xyz);
}

fn faceHitP(p: vec3<f32>, p00: vec3<f32>, p10: vec3<f32>, p01: vec3<f32>, p11: vec3<f32>) -> FaceHit {
  let t1 = closestTri(p, p00, p10, p11);
  let t2 = closestTri(p, p00, p11, p01);
  var o: FaceHit;
  if (t1.w <= t2.w) {
    o.w = vec4<f32>(t1.x, t1.y, 0.0, t1.z);
    o.d = sqrt(t1.w);
  } else {
    o.w = vec4<f32>(t2.x, 0.0, t2.z, t2.y);
    o.d = sqrt(t2.w);
  }
  o.q = o.w.x * p00 + o.w.y * p10 + o.w.z * p01 + o.w.w * p11;
  return o;
}

// ---------------------------------------------------------------- detection (once per substep)

// A face is hashed by its centroid at the predicted iterate and, when that lies in another cell, also
// by its centroid at the start of the substep: a node's search near its start position finds the faces
// that were near it then, however fast they move.
fn hashFace(f: u32) -> vec2<u32> {
  let c = corners(f);
  let h1 = hashCell(cellOf(centroid(c)));
  let c0 = cellOf(0.25 * (x[c.x].xyz + x[c.y].xyz + x[c.z].xyz + x[c.w].xyz));
  let h0 = select(hashCell(c0), NONE, all(c0 == cellOf(centroid(c))));
  return vec2<u32>(h1, h0);
}

@compute @workgroup_size(128)
fn faceCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= P.counts.x || !faceValid(f)) { return; }
  let h = hashFace(f);
  atomicAdd(&cells[h.x], 1u);
  if (h.y != NONE) { atomicAdd(&cells[h.y], 1u); }
}

// After the exclusive scan, cells[0, T) holds start offsets.
@compute @workgroup_size(128)
fn faceFill(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= P.counts.x || !faceValid(f)) { return; }
  let c = corners(f);
  // bounding spheres for the query's rejection test (one radius bounds both)
  let a0 = x[c.x].xyz;
  let b0 = x[c.y].xyz;
  let e0 = x[c.z].xyz;
  let g0 = x[c.w].xyz;
  let a1 = a0 + dx[c.x].xyz;
  let b1 = b0 + dx[c.y].xyz;
  let e1 = e0 + dx[c.z].xyz;
  let g1 = g0 + dx[c.w].xyz;
  let m0 = 0.25 * (a0 + b0 + e0 + g0);
  let m1 = 0.25 * (a1 + b1 + e1 + g1);
  let r0 = max(max(dot(a0 - m0, a0 - m0), dot(b0 - m0, b0 - m0)), max(dot(e0 - m0, e0 - m0), dot(g0 - m0, g0 - m0)));
  let r1 = max(max(dot(a1 - m1, a1 - m1), dot(b1 - m1, b1 - m1)), max(dot(e1 - m1, e1 - m1), dot(g1 - m1, g1 - m1)));
  let i1 = vec4<f32>(m1, sqrt(max(r0, r1)));
  let i0 = vec4<f32>(m0, f32(f));
  let h = hashFace(f);
  let slot = atomicLoad(&cells[h.x]) + atomicAdd(&cells[tableMask() + 1u + h.x], 1u);
  items[2u * slot] = i1;
  items[2u * slot + 1u] = i0;
  if (h.y != NONE) {
    let s0 = atomicLoad(&cells[h.y]) + atomicAdd(&cells[tableMask() + 1u + h.y], 1u);
    items[2u * s0] = i1;
    items[2u * s0 + 1u] = i0;
  }
}

// Distance from point q to segment (a, b).
fn segDist(q: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> f32 {
  let e = b - a;
  let s = clamp(dot(q - a, e) / max(dot(e, e), 1e-20), 0.0, 1.0);
  return length(q - a - s * e);
}

// Segment (o, o + d) against triangle (a, b, c): the segment parameter of the crossing, or 2 if none.
fn segTri(o: vec3<f32>, d: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
  let e1 = b - a;
  let e2 = c - a;
  let p = cross(d, e2);
  let det = dot(e1, p);
  if (abs(det) < 1e-18) { return 2.0; }
  let inv = 1.0 / det;
  let s = o - a;
  let u = dot(s, p) * inv;
  if (u < 0.0 || u > 1.0) { return 2.0; }
  let q = cross(s, e1);
  let v = dot(d, q) * inv;
  if (v < 0.0 || u + v > 1.0) { return 2.0; }
  let t = dot(e2, q) * inv;
  return select(2.0, t, t >= 0.0 && t <= 1.0);
}

// Where the node's predicted path (o, o + d) first crosses the face with corners c, at its start
// positions or at its predicted ones (2 if it does not).
fn pathCrossing(o: vec3<f32>, d: vec3<f32>, c: vec4<u32>) -> f32 {
  let a0 = x[c.x].xyz;
  let b0 = x[c.y].xyz;
  let e0 = x[c.z].xyz;
  let g0 = x[c.w].xyz;
  var t = min(segTri(o, d, a0, b0, g0), segTri(o, d, a0, g0, e0));
  t = min(t, min(segTri(o, d, cur(c.x), cur(c.y), cur(c.w)), segTri(o, d, cur(c.x), cur(c.w), cur(c.z))));
  return t;
}

// Lattice node (i, j) of layer offset/stride L.
fn latticeNode(L: vec4<u32>, i: i32, j: i32) -> u32 { return L.x + u32(j) * L.y + u32(i); }

// Surviving material along a lattice row (fixed j) from i0 to i1, or a column (fixed i) from j0 to j1.
fn rowHolds(L: vec4<u32>, j: i32, i0: i32, i1: i32) -> bool {
  for (var i = min(i0, i1); i < max(i0, i1); i++) {
    let a = latticeNode(L, i, j);
    if (!segActive(segs[2u * a]) || !cloth(a) || !cloth(a + 1u)) { return false; }
  }
  return true;
}
fn columnHolds(L: vec4<u32>, i: i32, j0: i32, j1: i32) -> bool {
  for (var j = min(j0, j1); j < max(j0, j1); j++) {
    let a = latticeNode(L, i, j);
    if (!segActive(segs[2u * a + 1u]) || !cloth(a) || !cloth(a + L.y)) { return false; }
  }
  return true;
}

// Whether node (ni, nj) still reaches face (fi, fj) through surviving spans along either L-shaped
// lattice path to the face's nearest corner. Such a face (within the exclusion window) is the node's own
// cloth and cannot meet it short of an impossibly tight fold; across a tear it is another lip or flap,
// and a contact face like any other.
fn latticeLinked(L: vec4<u32>, ni: i32, nj: i32, fi: i32, fj: i32) -> bool {
  let ci = clamp(ni, fi, fi + 1);
  let cj = clamp(nj, fj, fj + 1);
  if (rowHolds(L, nj, ni, ci) && columnHolds(L, ci, nj, cj)) { return true; }
  return columnHolds(L, ni, nj, cj) && rowHolds(L, cj, ni, ci);
}

// Exclusion window reach in lattice cells (detect skips linked faces with di, dj in [-3, 2]; their
// linking paths run through nodes within 3 cells).
const WINDOW: i32 = 3;

// Per node: whether any crossing within WINDOW along its lattice row has a broken span or left the
// cloth. detect reads the marks of the 2 WINDOW + 1 rows around a node: without damage there, every
// face in its exclusion window is its own intact cloth and needs no linking test.
@compute @workgroup_size(128)
fn damageRow(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let ijl = nodes[n].ijl;
  let L = P.layers[ijl >> 24u].lattice;
  let i = i32(ijl & 0xFFFu);
  let j = i32((ijl >> 12u) & 0xFFFu);
  var mark = 0u;
  for (var a = max(i - WINDOW, 0); a <= min(i + WINDOW, i32(L.y) - 1); a++) {
    let m = latticeNode(L, a, j);
    let fu = segs[2u * m].flags;
    let fv = segs[2u * m + 1u].flags;
    if (x[m].w > 0.5 || (fu & SEG_BROKEN) != 0u || (fv & SEG_BROKEN) != 0u) { mark = 1u; }
  }
  atomicStore(&pairs[(K + 1u) * P.counts.x + n], mark);
}

// Bound of the reciprocal pass: (4095 - share in 10 um) << 20 | (node + 1); the largest key is the
// tightest share (ties: any node), zero means none.
const RECIP_UNIT: f32 = 1e-5;

@compute @workgroup_size(128)
fn detect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  for (var k = 0u; k < K; k++) { atomicStore(&pairs[K * n + k], NONE); }
  // retired nodes are out of the scene
  if ((nodes[n].flags & NODE_PINNED) != 0u || x[n].w > 1.5) {
    cmeta[2u * n] = vec4<f32>(0.0, 0.0, 0.0, 1e9);
    return;
  }
  let p = cur(n);
  let p0 = x[n].xyz;
  let path = p - p0;
  let moving = dot(path, path) > HARD_CORE * HARD_CORE;
  let reach = SU.grid.z;
  let cell = SU.grid.x;
  let T = tableMask() + 1u;
  var best: array<u32, 4>;
  var bestD: array<f32, 4>;
  for (var k = 0u; k < K; k++) {
    best[k] = NONE;
    bestD[k] = 1e9;
  }
  // nearest face at the start of the substep (positions x): the bound's normal and clearance
  var d0min = 1e9;
  var n0 = vec3<f32>(0.0);
  var f0 = NONE;
  // the first face the predicted path crosses: its plane bounds the step instead
  var tc = 2.0;
  var fc = NONE;
  // every hash cell within half a cell of the predicted path: each face centroid within half a cell of
  // its start or end (a slow node scans the 2 x 2 x 2 cells nearest to it; a path over 2.5 cells is
  // covered from its start)
  let lo = vec3<i32>(floor(min(p, p0) / cell - 0.5));
  let hi = min(vec3<i32>(floor(max(p, p0) / cell + 0.5)), lo + vec3<i32>(3));
  // faces near the node in the lattice (their nearest corner within two cells) and still linked to it
  // through surviving spans: skipped
  let ijl = nodes[n].ijl;
  let ni = i32(ijl & 0xFFFu);
  let nj = i32((ijl >> 12u) & 0xFFFu);
  let nl = ijl >> 24u;
  let NL = P.layers[nl].lattice;
  var damaged = false;
  for (var b = max(nj - WINDOW, 0); b <= min(nj + WINDOW, i32(NL.z) - 1); b++) {
    if (atomicLoad(&pairs[(K + 1u) * P.counts.x + latticeNode(NL, ni, b)]) != 0u) { damaged = true; }
  }
  for (var cz = lo.z; cz <= hi.z; cz++) {
    for (var cy = lo.y; cy <= hi.y; cy++) {
      for (var cx = lo.x; cx <= hi.x; cx++) {
        let h = hashCell(vec3<i32>(cx, cy, cz));
        let start = atomicLoad(&cells[h]);
        let count = atomicLoad(&cells[T + h]);
        for (var q = 0u; q < count; q++) {
          // the closest-point distance is never below |p - centroid| - radius: faces that cannot
          // matter (beyond the reach now, no nearer than the bound's best at the start, off the path)
          // are rejected from their compact record without touching their corners
          let i1 = items[2u * (start + q)];
          let i0 = items[2u * (start + q) + 1u];
          let nearNow = length(p - i1.xyz) - i1.w < reach;
          let nearStart = length(p0 - i0.xyz) - i1.w < min(d0min, 0.5 * cell);
          let onPath = moving && min(segDist(i0.xyz, p0, p), segDist(i1.xyz, p0, p)) < i1.w;
          if (!nearNow && !nearStart && !onPath) { continue; }
          let f = u32(i0.w);
          let fijl = nodes[f].ijl;
          let di = i32(fijl & 0xFFFu) - ni;
          let dj = i32((fijl >> 12u) & 0xFFFu) - nj;
          if ((fijl >> 24u) == nl && di >= -3 && di <= 2 && dj >= -3 && dj <= 2 && (!damaged || latticeLinked(NL, ni, nj, ni + di, nj + dj))) { continue; }
          let cf = corners(f);
          // ties (a closest point on an edge two faces share) go to the lower face index: the result
          // must not depend on the order the hash was filled in
          if (nearStart) {
            let s0 = faceHitAt(p0, cf);
            if ((s0.d < d0min || (s0.d == d0min && f < f0)) && s0.d > 1e-9) {
              d0min = s0.d;
              n0 = (p0 - s0.q) / s0.d;
              f0 = f;
            }
          }
          if (onPath) {
            let t = pathCrossing(p0, path, cf);
            if (t < tc || (t == tc && t <= 1.0 && f < fc)) {
              tc = t;
              fc = f;
            }
          }
          if (!nearNow) { continue; }
          let hit = faceHit(p, cf);
          if (hit.d >= reach) { continue; }
          var cand = f;
          var cd = hit.d;
          for (var k = 0u; k < K; k++) {
            if (cand == best[k]) { break; }
            if (cd < bestD[k] || (cd == bestD[k] && cand < best[k])) {
              let ts = best[k];
              let td = bestD[k];
              best[k] = cand;
              bestD[k] = cd;
              cand = ts;
              cd = td;
            }
          }
        }
      }
    }
  }
  for (var k = 0u; k < K; k++) { atomicStore(&pairs[K * n + k], best[k]); }
  // the bound: against the first face the predicted path crosses (its plane, on the side the node
  // starts on), else against the nearest face at the start
  var nb = n0;
  var db = d0min;
  var fb = f0;
  if (fc != NONE && fc != f0) {
    let c = corners(fc);
    let a = x[c.x].xyz;
    var nrm = cross(x[c.y].xyz - a, x[c.z].xyz - a);
    let ln = length(nrm);
    if (ln > 1e-12) {
      nrm /= ln;
      let side = dot(p0 - a, nrm);
      nb = select(-nrm, nrm, side >= 0.0);
      db = abs(side);
      fb = fc;
    }
  }
  if (fb == NONE || (fb == f0 && db >= 0.5 * cell)) {
    cmeta[2u * n] = vec4<f32>(0.0, 0.0, 0.0, 1e9);
    return;
  }
  // approach toward the face limited to a share of the clearance above the hard core; the face's
  // corners may close no more than the same share toward the node (resolveBounds), so together they
  // never close the clearance
  let share = SU.law.w * max(db - HARD_CORE, 0.0);
  cmeta[2u * n] = vec4<f32>(nb, share);
  let key = ((4095u - min(4095u, u32(share / RECIP_UNIT))) << 20u) | (n + 1u);
  let c = corners(fb);
  let base = K * P.counts.x;
  atomicMax(&pairs[base + c.x], key);
  atomicMax(&pairs[base + c.y], key);
  atomicMax(&pairs[base + c.z], key);
  atomicMax(&pairs[base + c.w], key);
}

// Reciprocal bounds: a node that is the corner of a face some node bounds itself against may not move
// toward that node by more than the same share (the tightest such node; its normal from its own bound).
// Clears the reciprocal keys for the next substep.
@compute @workgroup_size(128)
fn resolveBounds(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let key = atomicExchange(&pairs[K * P.counts.x + n], 0u);
  if (key == 0u) {
    cmeta[2u * n + 1u] = vec4<f32>(0.0, 0.0, 0.0, 1e9);
    return;
  }
  let j = (key & 0xFFFFFu) - 1u;
  let share = f32(4095u - (key >> 20u)) * RECIP_UNIT;
  cmeta[2u * n + 1u] = vec4<f32>(-cmeta[2u * j].xyz, share);
}

// ---------------------------------------------------------------- terms (every Newton iteration)

fn addTerm(n: u32, g: vec3<f32>, H: mat3x3<f32>) {
  let b = ACC * n;
  let gq = clamp(g * GS, vec3<f32>(-2e9), vec3<f32>(2e9));
  atomicAdd(&acc[b], i32(gq.x));
  atomicAdd(&acc[b + 1u], i32(gq.y));
  atomicAdd(&acc[b + 2u], i32(gq.z));
  let hq = array<f32, 6>(H[0].x, H[0].y, H[0].z, H[1].y, H[1].z, H[2].z);
  for (var k = 0u; k < 6u; k++) { atomicAdd(&acc[b + 3u + k], i32(clamp(hq[k] * HS, -2e9, 2e9))); }
}

@compute @workgroup_size(128)
fn terms(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let p = cur(n);
  let delta = SU.grid.w;
  let kc = SU.law.x;
  let mu = SU.law.y;
  let epsU = SU.law.z * P.hg.x;
  let I3 = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
  let own = cmeta[2u * n];
  // how far this iteration's step may still approach along the node's bound normal (trust region below)
  var reach = 1e9;
  // depth at which the penalty carries the node's weight
  let sWeight = sqrt(nodes[n].mass * length(P.hg.yzw) * delta / kc);
  for (var k = 0u; k < K; k++) {
    let f = atomicLoad(&pairs[K * n + k]);
    if (f == NONE) { continue; }
    let c = corners(f);
    let hit = faceHit(p, c);
    if (hit.d <= 1e-9) { continue; }
    let nn = (p - hit.q) / hit.d;
    let a = dot(nn, own.xyz);
    if (a > 0.1) { reach = min(reach, (hit.d - delta + max(sWeight, 2.0 * max(delta - hit.d, 0.0))) / a); }
    if (hit.d >= delta) { continue; }
    let s = delta - hit.d;
    let lam = kc * s * s / delta;
    let nnT = mat3x3<f32>(nn * nn.x, nn * nn.y, nn * nn.z);
    // friction on the step relative to the face's material point
    let rel = dx[n].xyz - (hit.w.x * dx[c.x].xyz + hit.w.y * dx[c.y].xyz + hit.w.z * dx[c.z].xyz + hit.w.w * dx[c.w].xyz);
    let ut = rel - dot(rel, nn) * nn;
    let kf = mu * lam / max(length(ut), epsU);
    let g = -lam * nn + kf * ut;
    let H = (2.0 * kc * s / delta) * nnT + kf * (I3 - nnT);
    // The operator takes contact as diagonal blocks only (the node-face coupling is non-local), so
    // node and face each answer the contact force on their own. With the exact blocks (H on the node,
    // w^2 H on a corner of weight w) a corner would correct by gap / w and the pair would overshoot
    // and chatter. The stiffness is split instead so that the independent corrections add up to the
    // closure the exact pair would make: 2 H on the node (half the gap) and 2 w H on each corner (the
    // other half, by the barycentric weights).
    addTerm(n, g, 2.0 * H);
    var cs = array<u32, 4>(c.x, c.y, c.z, c.w);
    var ws = array<f32, 4>(hit.w.x, hit.w.y, hit.w.z, hit.w.w);
    for (var q = 0u; q < 4u; q++) {
      if (ws[q] <= 0.0) { continue; }
      addTerm(cs[q], -ws[q] * g, (2.0 * ws[q]) * H);
    }
  }
  // Trust region of the linearized penalty: a step may take the node no deeper into a contact distance
  // than twice its depth now, or the depth that carries its weight (its bound tightened to there for the
  // rest of the substep; step.wgsl newtonApply). Its curvature grows with depth (zero at the onset), so
  // beyond that the Newton model understates it without bound: one iteration took a free node outside
  // contact as deep as gravity carried it (h^2 g = 0.68 mm per substep), and a light strand lying on
  // cloth sank 0.4 mm, was kicked back by 0.25 N and bounced in a cycle two iterations never converged
  // (8 were needed). Within the region the model's curvature is at least half the true one; a node the
  // forces press deeper gets there over successive iterations and substeps.
  if (reach < 1e9 && any(own.xyz != vec3<f32>(0.0))) {
    let approach = -dot(dx[n].xyz, own.xyz);
    cmeta[2u * n] = vec4<f32>(own.xyz, min(own.w, approach + reach));
  }
}

// Accumulated terms into ext (the grip adds its own afterwards); the accumulator is cleared.
@compute @workgroup_size(128)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let b = ACC * n;
  var v: array<f32, 9>;
  for (var k = 0u; k < ACC; k++) {
    let a = atomicExchange(&acc[b + k], 0);
    v[k] = f32(a) / select(HS, GS, k < 3u);
  }
  ext[3u * n] = vec4<f32>(v[0], v[1], v[2], 0.0);
  ext[3u * n + 1u] = vec4<f32>(v[3], v[4], v[5], v[6]);
  ext[3u * n + 2u] = vec4<f32>(v[7], v[8], 0.0, 0.0);
}
