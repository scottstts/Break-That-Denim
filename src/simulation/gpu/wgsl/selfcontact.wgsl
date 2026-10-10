// Self-contact of a cloth layer: a node against a face (an intact lattice cell, as the two triangles
// c00 c10 c11 and c00 c11 c01). Penalty E = 1/2 k (delta - d)^2 on the distance d from the node to the
// face, with lagged-normal regularized Coulomb friction on their relative step. The node takes the
// force, the face's corners the reaction by the closest point's barycentric weights (momentum is
// conserved). Contributions are scattered with fixed-point integer atomics (WGSL has no float
// atomics) and gathered into ext; the Newton operator sees them as diagonal blocks.
//
// Faces are hashed by centroid once per substep at the predicted iterate. Detection keeps each node's
// K nearest faces within reach and an OGC-style directional bound from the nearest face at the start
// of the substep: the step may close only a share of the clearance down to a hard core, so no force
// (a hand pressing a flap down) carries a node through it, while sliding stays free. Faces within two
// lattice cells of a node are ignored: on a 3.4 mm lattice they cannot meet it short of an impossibly
// tight fold, and bounding against them would throttle every fast pull of the sheet.

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
@group(0) @binding(7) var<storage, read_write> items: array<u32>;
@group(0) @binding(8) var<storage, read_write> pairs: array<u32>;
@group(0) @binding(9) var<storage, read_write> cmeta: array<vec4<f32>>; // dx at detection, bound
@group(0) @binding(10) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(11) var<storage, read_write> ext: array<vec4<f32>>; // 3 per node

fn tableMask() -> u32 { return bitcast<u32>(SU.grid.y); }
fn cur(n: u32) -> vec3<f32> { return x[n].xyz + dx[n].xyz; }
// a node of the cloth solve (not debris or retired)
fn cloth(n: u32) -> bool { return x[n].w < 0.5; }

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

@compute @workgroup_size(128)
fn faceCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= P.counts.x || !faceValid(f)) { return; }
  atomicAdd(&cells[hashCell(cellOf(centroid(corners(f))))], 1u);
}

// After the exclusive scan, cells[0, T) holds start offsets.
@compute @workgroup_size(128)
fn faceFill(@builtin(global_invocation_id) gid: vec3<u32>) {
  let f = gid.x;
  if (f >= P.counts.x || !faceValid(f)) { return; }
  let h = hashCell(cellOf(centroid(corners(f))));
  let slot = atomicLoad(&cells[h]) + atomicAdd(&cells[tableMask() + 1u + h], 1u);
  items[slot] = f;
}

@compute @workgroup_size(128)
fn detect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  for (var k = 0u; k < K; k++) { pairs[K * n + k] = NONE; }
  if ((nodes[n].flags & NODE_PINNED) != 0u || !cloth(n)) {
    cmeta[n] = vec4<f32>(0.0, 0.0, 0.0, 1e9);
    return;
  }
  let p = cur(n);
  let reach = SU.grid.z;
  let T = tableMask() + 1u;
  var best: array<u32, 4>;
  var bestD: array<f32, 4>;
  for (var k = 0u; k < K; k++) {
    best[k] = NONE;
    bestD[k] = 1e9;
  }
  // nearest face at the start of the substep (positions x): the bound's normal and clearance
  let p0 = x[n].xyz;
  var d0min = 1e9;
  var n0 = vec3<f32>(0.0);
  // the 2 x 2 x 2 hash cells nearest to the node: every face centroid within half a cell of it
  let g = p / SU.grid.x;
  let c0 = vec3<i32>(floor(g));
  let side = select(vec3<i32>(-1), vec3<i32>(1), fract(g) >= vec3<f32>(0.5));
  // faces near the node in the lattice (their nearest corner within two cells): skipped
  let ijl = nodes[n].ijl;
  let ni = i32(ijl & 0xFFFu);
  let nj = i32((ijl >> 12u) & 0xFFFu);
  let nl = ijl >> 24u;
  for (var oz = 0; oz <= 1; oz++) {
    for (var oy = 0; oy <= 1; oy++) {
      for (var ox = 0; ox <= 1; ox++) {
        let h = hashCell(c0 + vec3<i32>(ox, oy, oz) * side);
        let start = atomicLoad(&cells[h]);
        let count = atomicLoad(&cells[T + h]);
        for (var q = 0u; q < count; q++) {
          let f = items[start + q];
          let fijl = nodes[f].ijl;
          let di = i32(fijl & 0xFFFu) - ni;
          let dj = i32((fijl >> 12u) & 0xFFFu) - nj;
          if ((fijl >> 24u) == nl && di >= -3 && di <= 2 && dj >= -3 && dj <= 2) { continue; }
          let cf = corners(f);
          let s0 = faceHitAt(p0, cf);
          if (s0.d < d0min && s0.d > 1e-9) {
            d0min = s0.d;
            n0 = (p0 - s0.q) / s0.d;
          }
          let hit = faceHit(p, cf);
          if (hit.d >= reach) { continue; }
          var cand = f;
          var cd = hit.d;
          for (var k = 0u; k < K; k++) {
            if (cand == best[k]) { break; }
            if (cd < bestD[k]) {
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
  for (var k = 0u; k < K; k++) { pairs[K * n + k] = best[k]; }
  // approach toward the nearest face limited to a share of the clearance above the hard core
  if (d0min < 0.5 * SU.grid.x) {
    cmeta[n] = vec4<f32>(n0, SU.law.w * max(d0min - HARD_CORE, 0.0));
  } else {
    cmeta[n] = vec4<f32>(0.0, 0.0, 0.0, 1e9);
  }
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
  for (var k = 0u; k < K; k++) {
    let f = pairs[K * n + k];
    if (f == NONE) { continue; }
    let c = corners(f);
    let hit = faceHit(p, c);
    if (hit.d >= delta || hit.d <= 1e-9) { continue; }
    let nn = (p - hit.q) / hit.d;
    let lam = kc * (delta - hit.d);
    let nnT = mat3x3<f32>(nn * nn.x, nn * nn.y, nn * nn.z);
    // friction on the step relative to the face's material point
    let rel = dx[n].xyz - (hit.w.x * dx[c.x].xyz + hit.w.y * dx[c.y].xyz + hit.w.z * dx[c.z].xyz + hit.w.w * dx[c.w].xyz);
    let ut = rel - dot(rel, nn) * nn;
    let kf = mu * lam / max(length(ut), epsU);
    let g = -lam * nn + kf * ut;
    let H = kc * nnT + kf * (I3 - nnT);
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
