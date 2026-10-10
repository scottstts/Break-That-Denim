// Per frame, per thread run (threads.wgsl): where its held ends are (the cloth they come out of,
// material_map.wgsl), whether the lattice chain carries it (thread_geom.wgsl RunGeo), and where a free
// tail beyond the chain's reach leaves it (solved by thread_tail.wgsl).

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> segw: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> nrm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> runs: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> threadCount: array<u32>;
// per run, RUN_GEO vec4 (thread.wgsl runGeoAt)
@group(0) @binding(7) var<storage, read_write> runGeo: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> rsurf: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> kappaM: array<u32>;
// (runCenter reads the solved tails; threadPrep does not)
var<private> tails: array<vec4<f32>, 1>;

override MAX_RUNS: u32 = 65536u;

// The chain spans over the run's drawn range hold, between nodes of one piece.
fn chainCarries(ri: RunInfo) -> bool {
  let LR = R.layers[ri.layer];
  let L = LR.lattice;
  let f = ri.fam;
  let Bal = f32(select(bundleOf(LR, 0u), bundleOf(LR, 1u), f == 0u));
  let Bcr = select(bundleOf(LR, 1u), bundleOf(LR, 0u), f == 0u);
  let len = select(L.z, L.y, f == 0u);
  let b = ri.yarn / Bcr;
  let a0 = u32(clamp(floor((ri.drawA - 0.5 * (Bal - 1.0)) / Bal), 0.0, f32(len) - 2.0));
  let a1 = u32(clamp(floor((ri.drawB - 0.5 * (Bal - 1.0)) / Bal), 0.0, f32(len) - 2.0));
  let w0 = x[chainNode(L, f, a0, b)].w;
  for (var a = a0; a <= a1; a++) {
    let n = chainNode(L, f, a, b);
    if (!segOkTh(n, f, L) || abs(x[n].w - w0) > 0.5 || abs(x[chainNode(L, f, a + 1u, b)].w - w0) > 0.5) { return false; }
  }
  return true;
}

// A run held at one end only: the drawn parameter up to which (from that end) its chain still holds,
// in one piece. Beyond it the yarn hangs from the chain's last point (thread_geom.wgsl runCenter).
fn chainReach(ri: RunInfo) -> f32 {
  let LR = R.layers[ri.layer];
  let L = LR.lattice;
  let f = ri.fam;
  let Bal = f32(select(bundleOf(LR, 0u), bundleOf(LR, 1u), f == 0u));
  let Bcr = select(bundleOf(LR, 1u), bundleOf(LR, 0u), f == 0u);
  let len = select(L.z, L.y, f == 0u);
  let b = ri.yarn / Bcr;
  let off = 0.5 * (Bal - 1.0);
  let a0 = u32(clamp(floor((ri.drawA - off) / Bal), 0.0, f32(len) - 2.0));
  let a1 = u32(clamp(floor((ri.drawB - off) / Bal), 0.0, f32(len) - 2.0));
  if (ri.heldA) {
    let w0 = x[chainNode(L, f, a0, b)].w;
    for (var a = a0; a <= a1; a++) {
      let n = chainNode(L, f, a, b);
      if (!segOkTh(n, f, L) || abs(x[chainNode(L, f, a + 1u, b)].w - w0) > 0.5) { return clamp(f32(a) * Bal + off, ri.drawA, ri.drawB); }
    }
    return ri.drawB;
  }
  let w1 = x[chainNode(L, f, a1 + 1u, b)].w;
  for (var a = i32(a1); a >= i32(a0); a--) {
    let n = chainNode(L, f, u32(a), b);
    if (!segOkTh(n, f, L) || abs(x[n].w - w1) > 0.5) { return clamp(f32(a + 1) * Bal + off, ri.drawA, ri.drawB); }
  }
  return ri.drawA;
}

// The held end at drawn parameter p (a crossing where the yarn is in the cloth): where that cloth is,
// and the yarn's unit direction there (toward increasing p). Falls back to the chain when the cloth
// is not held there.
fn heldEnd(ri: RunInfo, p: f32) -> array<vec3<f32>, 2> {
  let chain = centerAt(ri, p);
  let a = i32(round(p));
  let k = select(i32(ri.yarn), a, ri.fam == 0u);
  let m = select(a, i32(ri.yarn), ri.fam == 0u);
  let F = R.layers[ri.layer].field;
  var o = array<vec3<f32>, 2>(chain.p, normalize(chain.d + vec3<f32>(1e-9, 0.0, 0.0)));
  if (k < 0 || m < 0 || k >= i32(F.y) || m >= i32(F.z)) { return o; }
  let mp = crossingPoint(ri.layer, k, m);
  if (!mp.valid) { return o; }
  o[0] = mp.p;
  o[1] = normalize(select(mp.dt, mp.ds, ri.fam == 0u) + vec3<f32>(1e-9, 0.0, 0.0));
  return o;
}

@compute @workgroup_size(64)
fn threadPrep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= min(threadCount[0], MAX_RUNS)) { return; }
  let ri = runInfo(runs[r]);
  var g = runGeoZero();
  g.chain = chainCarries(ri);
  let cA = centerAt(ri, ri.drawA).p;
  let cB = centerAt(ri, ri.drawB).p;
  g.pA = cA;
  g.pB = cB;
  // ends held in the cloth (not at the field's border, where the chain is the cloth)
  if (((runs[r].x >> 5u) & 3u) == 0u) {
    let e = heldEnd(ri, ri.drawA);
    g.pA = e[0];
    g.tA = e[1];
  }
  if (((runs[r].x >> 7u) & 3u) == 0u) {
    let e = heldEnd(ri, ri.drawB);
    g.pB = e[0];
    g.tB = e[1];
  }
  // a run held at one end and not carried whole: how far its chain carries it from that end, and the
  // free tail beyond
  var reach = select(ri.drawA, ri.drawB, ri.heldA);
  var tailLen = 0.0;
  if (!g.chain && ri.heldA != ri.heldB) {
    reach = chainReach(ri);
    let LR = R.layers[ri.layer];
    let pitchAl = select(LR.pitch.y, LR.pitch.x, ri.fam == 0u);
    tailLen = abs(select(ri.drawA, ri.drawB, ri.heldA) - reach) * pitchAl;
    g.reach = reach;
    let root = tailRoot(ri, g, cA, cB);
    runGeo[RUN_GEO * r + 7u] = root[0];
    runGeo[RUN_GEO * r + 8u] = root[1];
  }
  // the tail's slot is set by threadTail
  runGeo[RUN_GEO * r] = vec4<f32>(select(0.0, 1.0, g.chain), reach, -1.0, tailLen);
  runGeo[RUN_GEO * r + 1u] = vec4<f32>(g.pA, 0.0);
  runGeo[RUN_GEO * r + 2u] = vec4<f32>(g.pB, 0.0);
  runGeo[RUN_GEO * r + 3u] = vec4<f32>(g.tA, 0.0);
  runGeo[RUN_GEO * r + 4u] = vec4<f32>(g.tB, 0.0);
  runGeo[RUN_GEO * r + 5u] = vec4<f32>(cA, 0.0);
  runGeo[RUN_GEO * r + 6u] = vec4<f32>(cB, 0.0);
}

// Where the free tail of a run held at one end leaves its chain (as runCenter leaves it): its root
// point and the chain node there (w), and the yarn's direction.
fn tailRoot(ri: RunInfo, g: RunGeo, cA: vec3<f32>, cB: vec3<f32>) -> array<vec4<f32>, 2> {
  let LR = R.layers[ri.layer];
  let fromA = ri.heldA;
  let held = select(ri.drawB, ri.drawA, fromA);
  let shift = endOffset(ri, g, cA, cB, g.reach);
  var q = select(g.pB, g.pA, fromA);
  var dir0 = select(-g.tB, g.tA, fromA);
  let L = LR.lattice;
  let f = ri.fam;
  let Bal = f32(select(bundleOf(LR, 0u), bundleOf(LR, 1u), f == 0u));
  let Bcr = select(bundleOf(LR, 1u), bundleOf(LR, 0u), f == 0u);
  let len = select(L.z, L.y, f == 0u);
  let a = u32(clamp(round((g.reach - 0.5 * (Bal - 1.0)) / Bal), 0.0, f32(len) - 1.0));
  let root = chainNode(L, f, a, ri.yarn / Bcr);
  if (g.reach != held) {
    let e = centerAt(ri, g.reach);
    q = e.p + shift;
    dir0 = normalize(e.d * select(-1.0, 1.0, fromA) + vec3<f32>(1e-9, 0.0, 0.0));
  }
  return array<vec4<f32>, 2>(vec4<f32>(q, f32(root)), vec4<f32>(dir0, 0.0));
}
