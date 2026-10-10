// Loose thread geometry (threads.wgsl runs), shared by the per-run preparation and the thread and
// fibre passes. Requires module-scope R (RenderU), nodes, x, segw, tframe and nrm.
//
// A thread follows the mechanical lattice: its bundle (the lattice row of a pick, the column of a
// warp) is a chain of nodes that carries the yarns' load and moves with it, and the thread is that
// chain's Catmull-Rom curve through the spans that still hold. The chain is a one-dimensional strand:
// mechanics resolves the bundle's motion, not where its yarns lie across it. Where a yarn is held in
// the weave it keeps its woven place; away from it, freed of the crossing yarns, the bundle's yarns
// cling together side by side in one to three flat clumps around the chain. No shape is added that the
// chain does not have: a thread swings, sags and settles as its strand does. Beyond the last span its
// chain holds, a yarn held at one end takes its free tail's solved shape (thread_prep.wgsl), and
// near a free end it leaves its clump.

fn liveTh(n: u32) -> bool { return x[n].w < 1.5; }

fn segOkTh(owner: u32, f: u32, L: vec4<u32>) -> bool {
  let fl = segw[2u * (2u * owner + f) + 1u].y;
  if ((fl & (SEG_EXISTS_R | SEG_BROKEN_R)) != SEG_EXISTS_R) { return false; }
  return liveTh(owner) && liveTh(owner + select(L.y, 1u, f == 0u));
}

// Bundle chain of family f through cross index b: node at along index a.
fn chainNode(L: vec4<u32>, f: u32, a: u32, b: u32) -> u32 { return L.x + select(a * L.y + b, b * L.y + a, f == 0u); }

struct Center {
  p: vec3<f32>,
  // d p / d (crossing along the yarn)
  d: vec3<f32>,
  // unit direction across the bundle (the crossing family's tangent) and cloth normal
  side: vec3<f32>,
  nrm: vec3<f32>,
  valid: bool,
};

// Centre line of the yarn's bundle at fine coordinate p (crossings along the yarn).
fn centerAt(ri: RunInfo, p: f32) -> Center {
  var o: Center;
  o.valid = false;
  let LR = R.layers[ri.layer];
  let L = LR.lattice;
  let f = ri.fam;
  let Bal = f32(select(bundleOf(LR, 0u), bundleOf(LR, 1u), f == 0u));
  let Bcr = select(bundleOf(LR, 1u), bundleOf(LR, 0u), f == 0u);
  let len = select(L.z, L.y, f == 0u);
  let b = ri.yarn / Bcr;
  let s = (p - 0.5 * (Bal - 1.0)) / Bal;
  let a0 = u32(clamp(floor(s), 0.0, f32(len) - 2.0));
  let t = s - f32(a0);
  let n0 = chainNode(L, f, a0, b);
  let n1 = chainNode(L, f, a0 + 1u, b);
  let pitchAl = select(LR.pitch.y, LR.pitch.x, f == 0u);
  if (segOkTh(n0, f, L)) {
    let p0 = x[n0].xyz;
    let p1 = x[n1].xyz;
    var pm = 2.0 * p0 - p1;
    if (a0 > 0u) {
      let nm = chainNode(L, f, a0 - 1u, b);
      if (segOkTh(nm, f, L)) { pm = x[nm].xyz; }
    }
    var pp = 2.0 * p1 - p0;
    if (a0 + 2u < len && segOkTh(n1, f, L)) { pp = x[chainNode(L, f, a0 + 2u, b)].xyz; }
    o.p = cmrT(pm, p0, p1, pp, t);
    o.d = cmrdT(pm, p0, p1, pp, t) / Bal;
    o.side = mix(tframe[2u * n0 + 1u - f].xyz, tframe[2u * n1 + 1u - f].xyz, clamp(t, 0.0, 1.0));
    o.nrm = mix(nrm[n0].xyz, nrm[n1].xyz, clamp(t, 0.0, 1.0));
    o.valid = liveTh(n0) && liveTh(n1);
  } else {
    // the bundle has separated in this span: the piece hangs from the node on the yarn's held side,
    // along that node's frame, drooping more the further it reaches
    let fromA = ri.heldA;
    let nh = select(n1, n0, fromA);
    if (!liveTh(nh)) { return o; }
    let dirN = tframe[2u * nh + f].xyz;
    let reach = (p - (f32(select(a0 + 1u, a0, fromA)) * Bal + 0.5 * (Bal - 1.0))) * pitchAl;
    let down = vec3<f32>(0.0, 0.0, -1.0);
    let dr = abs(reach);
    o.p = x[nh].xyz + dirN * reach + down * (0.35 * dr * dr / 1.5e-3);
    o.d = (dirN + down * (0.7 * reach / 1.5e-3)) * pitchAl;
    o.side = tframe[2u * nh + 1u - f].xyz;
    o.nrm = nrm[nh].xyz;
    o.valid = true;
  }
  return o;
}

fn cmrT(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
}
fn cmrdT(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  return 0.5 * ((p2 - p0) + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t + 3.0 * (3.0 * p1 - p0 - 3.0 * p2 + p3) * t * t);
}

// Per run and frame (thread_prep.wgsl). The lattice chain is the bundle's; the cloth a thread comes
// out of is placed per crossing (material_map.wgsl), so its held ends are attached where that cloth
// is: pA / pB (where held). Where the chain carries the thread over its whole drawn range (its spans
// hold, its nodes are of one piece) the thread follows the chain, shifted to meet the ends; otherwise
// (a falling piece, a separated bundle) it runs straight between its held ends, or, held at one end,
// follows its chain as far as that holds and beyond it takes its free tail's shape (thread_prep.wgsl
// tailSolve: tails, TAIL_SEGS + 1 points per tail).
struct RunGeo {
  chain: bool,
  // held at one end only, not carried whole: the drawn parameter up to which the chain carries it from
  // its held end (thread_prep.wgsl chainReach)
  reach: f32,
  // the free tail beyond reach: its slot in tails (-1: none) and its length (m)
  tail: i32,
  tailLen: f32,
  pA: vec3<f32>,
  pB: vec3<f32>,
  tA: vec3<f32>,
  tB: vec3<f32>,
};

// Chain point at drawn parameter p shifted so the thread meets its held ends.
fn endOffset(ri: RunInfo, g: RunGeo, cA: vec3<f32>, cB: vec3<f32>, p: f32) -> vec3<f32> {
  let dA = g.pA - cA;
  let dB = g.pB - cB;
  if (!ri.heldA) { return dB; }
  if (!ri.heldB) { return dA; }
  return mix(dA, dB, clamp((p - ri.drawA) / (ri.drawB - ri.drawA), 0.0, 1.0));
}

// Centre line of run ri at drawn parameter p (see RunGeo); chain points at the drawn ends cA, cB.
fn runCenter(ri: RunInfo, g: RunGeo, cA: vec3<f32>, cB: vec3<f32>, p: f32) -> Center {
  var c = centerAt(ri, p);
  if (g.chain) {
    c.p += endOffset(ri, g, cA, cB, p);
    return c;
  }
  let LR = R.layers[ri.layer];
  let pitchAl = select(LR.pitch.y, LR.pitch.x, ri.fam == 0u);
  let span = ri.drawB - ri.drawA;
  var o: Center;
  o.valid = true;
  o.side = c.side;
  o.nrm = c.nrm;
  if (!c.valid) {
    o.side = cross(vec3<f32>(0.0, 0.0, 1.0), select(g.tA, g.tB, !ri.heldA));
    o.nrm = vec3<f32>(0.0, 0.0, 1.0);
  }
  if (ri.heldA && ri.heldB) {
    let tr = clamp((p - ri.drawA) / span, 0.0, 1.0);
    o.p = mix(g.pA, g.pB, tr);
    o.d = (g.pB - g.pA) / span;
    return o;
  }
  // held at one end: along its chain as far as the chain holds (shifted to meet the held end), then
  // hanging from the chain's last point: it leaves along the yarn's direction there and bends over
  // toward the vertical within a bending length, at its own length (integrated along the arc)
  let fromA = ri.heldA;
  let held = select(ri.drawB, ri.drawA, fromA);
  let shift = endOffset(ri, g, cA, cB, p);
  if (select(p >= g.reach, p <= g.reach, fromA) && c.valid) {
    o.p = c.p + shift;
    o.d = c.d;
    return o;
  }
  if (g.tail >= 0) {
    // the solved tail, by arc length from its root (Catmull-Rom through its points)
    let h = g.tailLen / f32(TAIL_SEGS);
    let u = clamp(abs(p - g.reach) * pitchAl / max(h, 1e-9), 0.0, f32(TAIL_SEGS));
    let i = min(u32(floor(u)), TAIL_SEGS - 1u);
    let t = u - f32(i);
    let b = u32(g.tail) * (TAIL_SEGS + 1u);
    let p0 = tails[b + i].xyz;
    let p1 = tails[b + i + 1u].xyz;
    let pm = select(2.0 * p0 - p1, tails[b + i - 1u].xyz, i > 0u);
    let pp = select(2.0 * p1 - p0, tails[b + i + 2u].xyz, i + 1u < TAIL_SEGS);
    o.p = cmrT(pm, p0, p1, pp, t);
    o.d = cmrdT(pm, p0, p1, pp, t) / max(h, 1e-9) * pitchAl * select(-1.0, 1.0, fromA);
    return o;
  }
  var q = select(g.pB, g.pA, fromA);
  var dir0 = select(-g.tB, g.tA, fromA);
  if (g.reach != held) {
    let e = centerAt(ri, g.reach);
    q = e.p + shift;
    dir0 = normalize(e.d * select(-1.0, 1.0, fromA) + vec3<f32>(1e-9, 0.0, 0.0));
  }
  let reach = abs(p - g.reach) * pitchAl;
  let down = vec3<f32>(0.0, 0.0, -1.0);
  let lb = 1.5e-3;
  var dq = dir0;
  let steps = 6u;
  for (var k = 0u; k < steps; k++) {
    let sm = (f32(k) + 0.5) / f32(steps) * reach;
    dq = normalize(mix(dir0, down, sm / (sm + lb)) + vec3<f32>(1e-9, 0.0, 0.0));
    q += dq * (reach / f32(steps));
  }
  o.p = q;
  o.d = dq * pitchAl * select(-1.0, 1.0, fromA);
  return o;
}

fn runGeoZero() -> RunGeo {
  var g: RunGeo;
  g.chain = true;
  g.reach = 0.0;
  g.tail = -1;
  g.tailLen = 0.0;
  g.pA = vec3<f32>(0.0);
  g.pB = vec3<f32>(0.0);
  g.tA = vec3<f32>(1.0, 0.0, 0.0);
  g.tB = vec3<f32>(1.0, 0.0, 0.0);
  return g;
}

struct ThreadPoint {
  p: vec3<f32>,
  t: vec3<f32>,
  n: vec3<f32>,
  radius: f32,
  // 0 .. 1 toward a free tip (untwisting)
  tip: f32,
  valid: bool,
};

// Point of the thread of run ri at parameter p; cA, cB are the chain points at the drawn ends
// (centerAt), when g.chain.
fn threadAt(ri: RunInfo, g: RunGeo, cA: vec3<f32>, cB: vec3<f32>, p: f32) -> ThreadPoint {
  var o: ThreadPoint;
  let c = runCenter(ri, g, cA, cB, p);
  o.valid = c.valid;
  if (!c.valid) { return o; }
  let LR = R.layers[ri.layer];
  let f = ri.fam;
  let Bcr = select(bundleOf(LR, 1u), bundleOf(LR, 0u), f == 0u);
  let pitchAl = select(LR.pitch.y, LR.pitch.x, f == 0u);
  let pitchCr = select(LR.pitch.x, LR.pitch.y, f == 0u);
  let T = normalize(c.d + vec3<f32>(1e-9, 0.0, 0.0));
  var S = c.side - dot(c.side, T) * T;
  S = normalize(S + vec3<f32>(0.0, 0.0, 1e-7));
  let N = normalize(cross(T, S) * sign(dot(cross(T, S), c.nrm) + 1e-9));
  let b = ri.yarn / Bcr;
  let q = ri.yarn % Bcr;
  let seedY = hash11((ri.layer * 7919u + ri.yarn) * 2u + f + 0x51ed27u);
  let seedB = hash11((ri.layer * 7919u + b) * 2u + f + 0x2c1b3cu);
  let rY = select(LR.loose.y, LR.loose.x, f == 0u) * (0.88 + 0.24 * hashf(seedY + 5u));
  o.radius = rY;
  let along = p * pitchAl;
  // distance (crossings) to where the yarn is held in the weave
  let eA = select(1e6, p - ri.pos + 0.5, ri.heldA);
  let eB = select(1e6, ri.pos + ri.n - 0.5 - p, ri.heldB);
  let e = max(min(eA, eB), 0.0);
  // Freed of the crossing yarns, neighbouring yarns cling together side by side (fibre cohesion and
  // entanglement) within a couple of millimetres of where they leave the weave: a bundle's yarns form
  // one to three flat clumps, each near where its yarns were woven, so a ladder shows groups of pale
  // strands with dark gaps between them rather than evenly spaced lines.
  let eMm = e * pitchAl * 1e3;
  let gather = smoothstep(0.3, 2.5, eMm);
  let gs = clamp((eMm - 0.3) / 2.2, 0.0, 1.0);
  let dGather = select(0.0, 6.0 * gs * (1.0 - gs) / 2.2 * pitchAl * 1e3, e > 0.0 && e < 1e5) * select(1.0, -1.0, eA > eB);
  let woven = (f32(q) - 0.5 * f32(Bcr - 1u)) * pitchCr;
  let hc = hashf(seedB);
  let nClumps = select(select(3u, 2u, hc < 0.75), 1u, hc < 0.3 || Bcr < 3u);
  // contiguous ranges of yarns, split at hashed points
  let cut1 = 1u + u32(hashf(seedB + 21u) * f32(max(Bcr - 1u, 1u)));
  let cut2 = 1u + u32(hashf(seedB + 22u) * f32(max(Bcr - 1u, 1u)));
  let lo = min(cut1, cut2);
  let hi = max(cut1, cut2);
  var c0 = 0u;
  var c1 = Bcr;
  if (nClumps == 2u) {
    if (q < cut1) { c1 = cut1; } else { c0 = cut1; }
  } else if (nClumps == 3u) {
    if (q < lo) { c1 = lo; } else if (q < hi || hi == lo) { c0 = lo; c1 = max(hi, lo + 1u); } else { c0 = hi; }
  }
  let cn = c1 - c0;
  let slot = f32(q - c0) - 0.5 * f32(cn - 1u);
  let clumpLat = (0.5 * f32(c0 + c1 - 1u) - 0.5 * f32(Bcr - 1u)) * pitchCr + (hashf(seedB + 23u + c0) - 0.5) * 0.4 * pitchCr;
  // side by side, touching (loose yarns flatten against each other), the flat clump turning slowly
  // about its axis (pitch 15-30 mm, either hand)
  let spacing = 1.6 * rY;
  let twist = (15e-3 + 15e-3 * hashf(seedB + 3u + c0)) * select(-1.0, 1.0, hashf(seedB + 4u + c0) < 0.5);
  let phi = 6.2831853 * (along / twist + hashf(seedB + 5u + c0));
  let dPhi = 6.2831853 * pitchAl / twist;
  let inCord = vec2<f32>(cos(phi), sin(phi)) * (slot * spacing);
  let dInCord = vec2<f32>(-sin(phi), cos(phi)) * (slot * spacing * dPhi);
  let cordLat = clumpLat;
  let lat = mix(woven, cordLat + inCord.x, gather);
  let up = gather * inCord.y;
  var P = c.p + S * lat + N * up;
  var dP = c.d + S * (gather * dInCord.x + dGather * (cordLat + inCord.x - woven)) + N * (gather * dInCord.y + dGather * inCord.y);
  // a free end leaves the cord over its last millimetre or two: it splays a little away from the
  // other yarns (its weight bends a freed yarn over ~1.4 cm, not within its tip: tailSolve)
  o.tip = 0.0;
  if (!ri.heldA || !ri.heldB) {
    let lt = (0.6e-3 + 1.8e-3 * hashf(seedY + 6u)) / pitchAl;
    let u = clamp(select(p - (ri.drawB - lt), (ri.drawA + lt) - p, !ri.heldA) / lt, 0.0, 1.0);
    let splay = S * (0.35 * (hashf(seedY + 10u) - 0.5)) + N * (0.2 * (hashf(seedY + 11u) - 0.5));
    let off = splay * (lt * pitchAl);
    P += off * (u * u);
    dP += off * (2.0 * u / lt) * select(1.0, -1.0, !ri.heldA);
    o.tip = u;
    o.radius *= 1.0 - 0.45 * u;
  }
  o.p = P;
  o.t = normalize(dP + vec3<f32>(1e-9, 0.0, 0.0));
  o.n = N;
  return o;
}

// The per-frame geometry of run r written by threadPrep (RUN_GEO vec4 per run). Requires module-scope
// runGeo.
fn runGeoAt(r: u32) -> RunGeo {
  var g: RunGeo;
  let h = runGeo[RUN_GEO * r];
  g.chain = h.x > 0.5;
  g.reach = h.y;
  g.tail = i32(h.z);
  g.tailLen = h.w;
  g.pA = runGeo[RUN_GEO * r + 1u].xyz;
  g.pB = runGeo[RUN_GEO * r + 2u].xyz;
  g.tA = runGeo[RUN_GEO * r + 3u].xyz;
  g.tB = runGeo[RUN_GEO * r + 4u].xyz;
  return g;
}
