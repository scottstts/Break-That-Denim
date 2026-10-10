// Loose yarn geometry shared by the tail and fuzz passes. Requires module-scope bindings
// R (RenderU), nodes, x, nrm, tframe, segw, tails, lists (u32 view) and the list layout constants.

const SPAN_BASE: u32 = 4u;
override MAX_SPANS: u32 = 65536u;

fn segOkR(owner: u32, f: u32) -> bool {
  let fl = segw[2u * (2u * owner + f) + 1u].y;
  if ((fl & (SEG_EXISTS_R | SEG_BROKEN_R)) != SEG_EXISTS_R) { return false; }
  let L = R.layers[nodes[owner].ijl >> 24u].lattice;
  return x[owner].w < 1.5 && x[owner + select(L.y, 1u, f == 0u)].w < 1.5;
}

fn segRest(owner: u32, f: u32) -> f32 { return bitcast<f32>(segw[2u * (2u * owner + f)].x); }

struct YarnPoint {
  p: vec3<f32>,
  // unit tangent along the yarn
  t: vec3<f32>,
  // cloth normal at this point
  n: vec3<f32>,
  // yarn coordinate along this yarn (pick index for warp, warp index for weft)
  c: f32,
  // fine yarn index across (warp index for warp yarns, pick index for weft yarns)
  yarn: i32,
  fam: u32,
  layer: u32,
  wear: f32,
  radius: f32,
};

fn cmr(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
}
fn cmrd(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  return 0.5 * ((p2 - p0) + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t + 3.0 * (3.0 * p1 - p0 - 3.0 * p2 + p3) * t * t);
}

// Strand of bundled yarn q in span s.
fn strandOf(s: u32, q: u32) -> u32 { return q / yarnsPerStrand(R.layers[nodes[s >> 1u].ijl >> 24u], s & 1u); }

// Failure site of bundled yarn q of span s (fraction from the owning node).
fn yarnSite(s: u32, q: u32) -> f32 { return yarnSiteOf(segw[2u * s + 1u].w, s, strandOf(s, q), q); }

// Span separated on tail side e = 4 n + side (side: 0 u-, 1 u+, 2 v-, 3 v+) of node n.
fn tailSpan(e: u32) -> u32 {
  let n = e >> 2u;
  let side = e & 3u;
  let f = side >> 1u;
  let L = R.layers[nodes[n].ijl >> 24u].lattice;
  let stride = select(L.y, 1u, f == 0u);
  return 2u * select(n - stride, n, (side & 1u) == 1u) + f;
}

// Distance from node e >> 2 along its tail's span to the crack crossing, where the node's torn
// group ends (render_common.wgsl crossingOf).
fn tailRoot(e: u32) -> f32 {
  let s = tailSpan(e);
  let c = crossingOf(bitcast<f32>(segw[2u * s + 1u].z));
  return select(1.0 - c, c, (e & 1u) == 1u) * segRest(s >> 1u, s & 1u);
}

// Shortest loose end drawn as a yarn (m); shorter ends are the rim's fringe (surface and fibres).
const TAIL_MIN: f32 = 0.6e-3;

// Length of loose yarn q of tail e beyond the crack: the yarn's remnant to its own failure site plus
// any length pulled out through the crossing, less the distance to the crack. <= 0: the yarn ends
// inside the torn group's material.
fn tailLength(e: u32, q: u32) -> f32 {
  let s = tailSpan(e);
  let plus = (e & 1u) == 1u;
  let Lr = segRest(s >> 1u, s & 1u);
  let failSite = bitcast<f32>(segw[2u * s + 1u].z);
  let own = select(1.0 - yarnSite(s, q), yarnSite(s, q), plus);
  let spanOwn = select(1.0 - failSite, failSite, plus);
  let extra = max(tails[e] - spanOwn * Lr, 0.0);
  return min(own * Lr + extra - tailRoot(e), 0.008);
}

// Loose yarn end q of tail e, t in [0, 1] from the crack to the free end.
// Render-only shape: it leaves along the node's frame (as the torn group it continues), each yarn of
// the bundle at its own angle, sags under gravity, curls, and fans out toward its untwisted end.
fn tailYarn(e: u32, q: u32, t: f32) -> YarnPoint {
  let n = e >> 2u;
  let sideIdx = e & 3u;
  let f = sideIdx >> 1u;
  let dirSign = select(-1.0, 1.0, (sideIdx & 1u) == 1u);
  let ijl = nodes[n].ijl;
  let layer = ijl >> 24u;
  let LR = R.layers[layer];
  let i = ijl & 0xFFFu;
  let j = (ijl >> 12u) & 0xFFFu;
  let a = select(j, i, f == 0u);
  let N = nrm[n].xyz;
  let tu = tframe[2u * n].xyz;
  let tv = tframe[2u * n + 1u].xyz;
  let d0 = dirSign * select(tv, tu, f == 0u);
  // bundle yarns lie across the yarn direction: picks along +v, warps along +u
  let sideV = select(tu, tv, f == 0u);
  let r0 = tailRoot(e);
  let b = bundleOf(LR, f);
  let lat = f32(q) - 0.5 * (f32(b) - 1.0);
  let seed = hash11(e * 7u + q * 131u + 17u);
  let len = max(tailLength(e, q), 0.0);
  let pitchLat = select(LR.pitch.y, LR.pitch.x, f == 1u);
  let curl = vec3<f32>(hashf(seed + 1u) - 0.5, hashf(seed + 2u) - 0.5, hashf(seed + 3u) - 0.5);
  var o: YarnPoint;
  let jit = hashf(seed + 4u) - 0.5;
  let p0 = x[n].xyz + d0 * r0 + sideV * ((lat + 0.8 * jit) * pitchLat);
  // the freed yarns of a bundle splay apart: each leaves at its own angle
  let ang = (hashf(seed + 5u) - 0.5) * 0.9 + 0.08 * lat;
  let fan = sideV * (ang * len * t) + curl * (0.9 * len * t * t);
  let sag = vec3<f32>(0.0, 0.0, -0.35 * len * t * t) + N * ((0.15 + 0.3 * hashf(seed + 6u)) * len * t * (1.0 - t));
  o.p = p0 + d0 * (len * t) + fan + sag;
  let dp = d0 * len + sideV * (ang * len) + curl * (1.8 * len * t) + vec3<f32>(0.0, 0.0, -0.7 * len * t) + N * ((0.15 + 0.3 * hashf(seed + 6u)) * len * (1.0 - 2.0 * t));
  o.t = normalize(dp + d0 * 1e-6);
  o.n = N;
  o.fam = select(0u, 1u, f == 1u);
  o.yarn = i32(select(j, i, f == 1u) * b + q);
  let ba = f32(bundleOf(LR, 1u - f));
  o.c = f32(a) * ba + 0.5 * (ba - 1.0) + dirSign * (r0 + t * len) / select(LR.pitch.x, LR.pitch.y, f == 1u);
  o.layer = layer;
  // broken ends are abraded and untwisting: their pale cores show more toward the end
  o.wear = min(1.0, wearR[n] + 0.4 + 0.5 * t);
  o.radius = 0.32 * select(LR.weft.x, LR.warp.x, f == 1u) * (1.0 - 0.6 * t);
  return o;
}

fn tailEntry(k: u32) -> u32 { return lists[SPAN_BASE + MAX_SPANS + k]; }
