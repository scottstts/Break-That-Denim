// Loose thread geometry (threads.wgsl runs), shared by the per-run preparation and the thread and
// fibre passes. Requires module-scope R (RenderU), nodes, x, segw, tframe and nrm.
//
// A thread follows the mechanical lattice: its bundle (the lattice row of a pick, the column of a
// warp) is a chain of nodes that carries the yarns' load, and the thread is that chain's Catmull-Rom
// curve through the spans that still hold, offset across the bundle to the yarn's own place in it.
// Where it is held in the weave a yarn keeps its woven place; away from it, freed of the crossing
// yarns, it wanders across by about a pitch (some neighbours pair up, a few stray) and hangs by its
// own slack. Near a free end the yarn droops and curls and thins as it untwists.

struct RunInfo {
  layer: u32,
  fam: u32,
  yarn: u32,
  // first lone crossing and count
  pos: f32,
  n: f32,
  // the ends are held in the weave (or the field's border), not free
  heldA: bool,
  heldB: bool,
  // drawn parameter range (crossings along the yarn)
  drawA: f32,
  drawB: f32,
  // first of its n + 2 simulated points (thread_sim.wgsl): the end A anchor, one per lone crossing,
  // the end B anchor
  base: u32,
};

fn runInfo(rec: vec4<u32>) -> RunInfo {
  var o: RunInfo;
  o.layer = rec.x & 0xFu;
  o.fam = (rec.x >> 4u) & 1u;
  o.heldA = ((rec.x >> 5u) & 3u) != 1u;
  o.heldB = ((rec.x >> 7u) & 3u) != 1u;
  o.yarn = rec.y & 0xFFFFu;
  o.pos = f32(rec.y >> 16u);
  o.n = f32(rec.z);
  o.base = rec.w;
  o.drawA = o.pos - select(0.5, 1.0, o.heldA);
  o.drawB = o.pos + o.n - 1.0 + select(0.5, 1.0, o.heldB);
  return o;
}

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
// (a falling piece, a separated bundle) it runs straight between its held ends, or hangs from its one
// held end along tA, the yarn's direction there.
struct RunGeo {
  arc: f32,
  chain: bool,
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
  // hanging from its held end: it leaves along the yarn's direction there and bends over toward the
  // vertical within a bending length, at its own length (integrated along the arc)
  let fromA = ri.heldA;
  let reach = select(ri.drawB - p, p - ri.drawA, fromA) * pitchAl;
  let dir0 = select(-g.tB, g.tA, fromA);
  let down = vec3<f32>(0.0, 0.0, -1.0);
  let lb = 1.5e-3;
  var q = select(g.pB, g.pA, fromA);
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
  g.arc = 0.0;
  g.chain = true;
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

// Point of the thread of run ri at parameter p; g.arc is the current length of its centre line over
// the drawn range (threadPrep), which the yarn's own slack is measured against; cA, cB are the chain
// points at the drawn ends (centerAt), when g.chain.
fn threadAt(ri: RunInfo, g: RunGeo, cA: vec3<f32>, cB: vec3<f32>, p: f32) -> ThreadPoint {
  var o: ThreadPoint;
  let arc = g.arc;
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
  o.radius = select(LR.loose.y, LR.loose.x, f == 0u) * (0.88 + 0.24 * hashf(seedY + 5u));
  let along = p * pitchAl;
  // distance (crossings) to where the yarn is held in the weave
  let eA = select(1e6, p - ri.pos + 0.5, ri.heldA);
  let eB = select(1e6, ri.pos + ri.n - 0.5 - p, ri.heldB);
  let e = max(min(eA, eB), 0.0);
  let stray = hashf(seedY + 9u) < 0.08;
  // A freed yarn stays near its own place in the weave: nothing gathers a bundle (the lattice's unit
  // of mechanics, not of the cloth), so threads never line up on the lattice. Away from where it is
  // held it wanders across by up to about a pitch, and now and then two neighbours lie together
  // (pairs per yarn pair, not per bundle).
  let woven = (f32(q) - 0.5 * f32(Bcr - 1u)) * pitchCr;
  let loose = 1.0 - exp(-e / 2.5);
  let pairSeed = hash11((ri.layer * 7919u + (ri.yarn >> 1u)) * 2u + f + 0x7a3c11u);
  let paired = hashf(pairSeed) < 0.35;
  let pairC = (f32(i32(q & ~1u)) + 0.5 - 0.5 * f32(Bcr - 1u)) * pitchCr;
  let wander = (0.06e-3 + select(0.45, 1.2, stray) * pitchCr * loose) * (0.6 * noise1(seedY + 3u, along / 3.5e-3) + 0.4 * noise1(seedY + 13u, along / 1.2e-3));
  let lat = mix(woven, pairC + select(-0.4, 0.4, (q & 1u) == 1u) * o.radius, select(0.0, loose, paired)) + wander;
  let up = 0.6 * o.radius * loose * noise1(seedY + 17u, along / 2.0e-3);
  var P = c.p + S * lat + N * up;
  var dP = c.d;
  // the yarn's own slack between its held ends: a parabolic hang below the bundle
  let down = vec3<f32>(0.0, 0.0, -1.0);
  var sagDir = down - dot(down, T) * T;
  sagDir = normalize(sagDir + S * (0.5 * (hashf(seedY + 2u) - 0.5)) + vec3<f32>(0.0, 0.0, -1e-6));
  let span = ri.drawB - ri.drawA;
  let tr = clamp((p - ri.drawA) / span, 0.0, 1.0);
  let sl = hashf(seedY + 1u);
  let slack = (0.002 + 0.03 * sl * sl * sl) * select(1.0, 2.5, stray);
  let rest = span * pitchAl;
  if (ri.heldA && ri.heldB) {
    let hang = sqrt(max(0.375 * arc * (rest * (1.0 + slack) - arc), 0.0));
    P += sagDir * (4.0 * hang * tr * (1.0 - tr));
    dP += sagDir * (4.0 * hang * (1.0 - 2.0 * tr) / span);
  }
  // a free end droops and curls over its last millimetre or two
  o.tip = 0.0;
  if (!ri.heldA || !ri.heldB) {
    let lt = (0.6e-3 + 1.8e-3 * hashf(seedY + 6u)) / pitchAl;
    let u = clamp(select(p - (ri.drawB - lt), (ri.drawA + lt) - p, !ri.heldA) / lt, 0.0, 1.0);
    let curl = vec3<f32>(hashf(seedY + 7u) - 0.5, hashf(seedY + 8u) - 0.5, -0.8) * (0.9 * lt * pitchAl);
    P += curl * (u * u) + S * (0.3 * lt * pitchAl * (hashf(seedY + 10u) - 0.5) * u * u);
    dP += curl * (2.0 * u / lt) * select(1.0, -1.0, !ri.heldA);
    o.tip = u;
    o.radius *= 1.0 - 0.45 * u;
  }
  o.p = P;
  o.t = normalize(dP + vec3<f32>(1e-9, 0.0, 0.0));
  o.n = N;
  return o;
}
