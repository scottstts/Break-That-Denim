// Per-substep kernels outside the linear solve: predictor, Newton update, commit, cohesive damage
// and rupture, friction-limited crossing slip. Mirrors ReferenceSolver (reference/solver.ts).

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<uniform> Mode: vec4<u32>; // slip: x = family, y = parity; newtonApply/gate: x = iteration
@group(0) @binding(2) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(3) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> v: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> dx: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> pred: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> segs: array<Seg>;
@group(0) @binding(8) var<storage, read_write> tails: array<f32>;
@group(0) @binding(9) var<storage, read> delta: array<vec4<f32>>;
// [0] ruptures, [1] pulled out, [2] max speed (f32 bits), [3] line-search step length (f32 bits),
// [4 + k] largest update of Newton iteration k
@group(0) @binding(10) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> cmeta: array<vec4<f32>>; // dx at contact detection, displacement bound

fn historyTolerance() -> f32 { return bitcast<f32>(P.counts.w); }

fn pinnedN(n: u32) -> bool { return (nodes[n].flags & NODE_PINNED) != 0u; }

fn lattice(l: u32) -> vec4<u32> { return P.layers[l].lattice; }

// Neighbor of n along family f in direction dir (-1/+1), or -1.
fn nbr(n: u32, f: u32, dir: i32) -> i32 {
  let ijl = nodes[n].ijl;
  let i = i32(ijl & 0xFFFu);
  let j = i32((ijl >> 12u) & 0xFFFu);
  let L = lattice(ijl >> 24u);
  let ni = i + select(0, dir, f == 0u);
  let nj = j + select(0, dir, f == 1u);
  if (ni < 0 || nj < 0 || ni >= i32(L.y) || nj >= i32(L.z)) { return -1; }
  return i32(L.x) + nj * i32(L.y) + ni;
}

@compute @workgroup_size(128)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  if (pinnedN(n)) {
    pred[n] = vec4<f32>(0.0);
    dx[n] = vec4<f32>(0.0);
    return;
  }
  let h = P.hg.x;
  let hv = h * v[n].xyz;
  pred[n] = vec4<f32>(hv + h * h * P.hg.yzw, 0.0);
  dx[n] = vec4<f32>(hv, 0.0);
}

@compute @workgroup_size(128)
fn newtonApply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  if (pinnedN(n)) { return; }
  // OGC-style feasibility: the iterate stays within the bound around the detection-time displacement
  let m = cmeta[n];
  let old = dx[n].xyz;
  // step length chosen by the line search (energy.wgsl)
  let alpha = bitcast<f32>(atomicLoad(&counters[3]));
  var d = old + alpha * delta[n].xyz - m.xyz;
  let l = length(d);
  if (l > m.w) { d = d * (m.w / l); }
  let nd = m.xyz + d;
  // w keeps the size of this node's latest Newton update (local convergence for history updates)
  dx[n] = vec4<f32>(nd, length(nd - old));
  // non-negative f32 bit patterns order like their values
  atomicMax(&counters[4u + Mode.x], bitcast<u32>(length(nd - old)));
}

@compute @workgroup_size(128)
fn commit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let d = dx[n].xyz;
  // v.w carries the node's last Newton update into the history kernels
  v[n] = vec4<f32>(d / P.hg.x, dx[n].w);
  // largest speed since the host last cleared it (rest detection)
  atomicMax(&counters[2], bitcast<u32>(length(d) / P.hg.x));
  x[n] = vec4<f32>(x[n].xyz + d, 0.0);
  dx[n] = vec4<f32>(0.0);
}

@compute @workgroup_size(128)
fn damage(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= 2u * P.counts.x || P.counts.z == 0u) { return; }
  var sg = segs[s];
  if (!segActive(sg)) { return; }
  let a = s >> 1u;
  let f = s & 1u;
  let b = u32(nbr(a, f, 1));
  // an unconverged iterate must not commit damage: defer until both ends have converged locally
  if (max(v[a].w, v[b].w) > historyTolerance()) { return; }
  let ext = length(x[b].xyz - x[a].xyz) - sg.rest;
  if (ext > sg.maxExt) { sg.maxExt = ext; }
  if (isSeparated(sg.rest, sg)) {
    sg.flags = sg.flags | SEG_BROKEN;
    tails[4u * a + 2u * f + 1u] += sg.rest * sg.failSite;
    tails[4u * b + 2u * f] += sg.rest * (1.0 - sg.failSite);
    atomicAdd(&counters[0], 1u);
  }
  segs[s] = sg;
}

// Tension of an active span at committed positions and dT/dL at fixed length.
fn spanTension(owner: u32, f: u32) -> vec2<f32> {
  let sg = segs[2u * owner + f];
  let b = u32(nbr(owner, f, 1));
  let l = length(x[b].xyz - x[owner].xyz);
  let L = sg.rest;
  let y = P.layers[nodes[owner].ijl >> 24u].yarn[f];
  let r = damagedResponse(y, L, sg, l / L - 1.0);
  return vec2<f32>(r.x, max(r.y, 0.0) * l / (L * L));
}

fn spanActive(owner: i32, f: u32) -> bool {
  if (owner < 0) { return false; }
  return segActive(segs[2u * u32(owner) + f]);
}

@compute @workgroup_size(128)
fn slip(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x || P.counts.z == 0u) { return; }
  if (v[n].w > historyTolerance()) { return; }
  let f = Mode.x;
  let ijl = nodes[n].ijl;
  let along = select((ijl >> 12u) & 0xFFFu, ijl & 0xFFFu, f == 0u);
  if ((along & 1u) != Mode.y) { return; }
  if (pinnedN(n)) { return; }
  let prev = nbr(n, f, -1);
  let next = nbr(n, f, 1);
  if (prev < 0 || next < 0) { return; }
  let actL = spanActive(prev, f);
  let actR = spanActive(i32(n), f);
  if (!actL && !actR) { return; }
  var tl = vec2<f32>(0.0);
  var tr = vec2<f32>(0.0);
  if (actL) { tl = spanTension(u32(prev), f); }
  if (actR) { tr = spanTension(n, f); }
  // crossing yarn tension
  let g = 1u - f;
  var tc = 0.0;
  var cnt = 0.0;
  let cp = nbr(n, g, -1);
  if (spanActive(cp, g)) {
    tc += max(0.0, spanTension(u32(cp), g).x);
    cnt += 1.0;
  }
  if (spanActive(i32(n), g)) {
    tc += max(0.0, spanTension(n, g).x);
    cnt += 1.0;
  }
  if (cnt == 0.0) { return; }
  tc /= cnt;
  let mat = P.layers[ijl >> 24u];
  let Fc = mat.a.y * (nodes[n].n0 + mat.a.z * mat.yarn[f].yarnsPerSpan * tc);
  let dT = max(tr.x, 0.0) - max(tl.x, 0.0);
  if (abs(dT) <= Fc) { return; }
  let highR = dT > 0.0;
  let kHigh = select(tl.y, tr.y, highR);
  let lowActive = select(actR, actL, highR);
  let kLow = select(0.0, select(tr.y, tl.y, highR), lowActive);
  var dl = (abs(dT) - Fc) / max(kHigh + kLow, 1e-9);
  let pitch = select(mat.b.x, mat.a.w, f == 0u);
  let sHigh = select(2u * u32(prev) + f, 2u * n + f, highR);
  let sLow = select(2u * n + f, 2u * u32(prev) + f, highR);
  if (lowActive) {
    dl = min(dl, max(0.0, segs[sLow].rest - 0.55 * pitch));
    segs[sLow].rest -= dl;
    segs[sHigh].rest += dl;
  } else {
    let tIdx = select(4u * n + 2u * f + 1u, 4u * n + 2u * f, highR);
    dl = min(dl, tails[tIdx]);
    tails[tIdx] -= dl;
    segs[sHigh].rest += dl;
    if (tails[tIdx] <= 1e-9) {
      tails[tIdx] = 0.0;
      // The yarn leaves this crossing: the high-side span becomes a free tail of its far node.
      let farTail = select(4u * u32(prev) + 2u * f + 1u, 4u * u32(next) + 2u * f, highR);
      tails[farTail] += segs[sHigh].rest;
      segs[sHigh].flags = segs[sHigh].flags | SEG_BROKEN;
      atomicAdd(&counters[1], 1u);
    }
  }
}
