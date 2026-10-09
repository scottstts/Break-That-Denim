// Explicit yarn geometry shared by the yarn, tail and fuzz passes. Requires module-scope bindings
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

// Yarn q of the bundle carried by span s (owner node s >> 1, family s & 1), at parameter t in [0, 1].
fn spanYarn(s: u32, q: u32, t: f32) -> YarnPoint {
  let n = s >> 1u;
  let f = s & 1u;
  let ijl = nodes[n].ijl;
  let layer = ijl >> 24u;
  let LR = R.layers[layer];
  let L = LR.lattice;
  let i = ijl & 0xFFFu;
  let j = (ijl >> 12u) & 0xFFFu;
  let stride = select(L.y, 1u, f == 0u);
  let a = select(j, i, f == 0u);
  let amax = select(L.z, L.y, f == 0u);
  let nb = n + stride;
  let pa = x[n].xyz;
  let pb = x[nb].xyz;
  var p0 = 2.0 * pa - pb;
  var p3 = 2.0 * pb - pa;
  if (a > 0u && segOkR(n - stride, f)) { p0 = x[n - stride].xyz; }
  if (a + 2u < amax && segOkR(nb, f)) { p3 = x[nb + stride].xyz; }
  var o: YarnPoint;
  let P = cmr(p0, pa, pb, p3, t);
  o.t = normalize(cmrd(p0, pa, pb, p3, t));
  o.n = normalize(mix(nrm[n].xyz, nrm[nb].xyz, t));
  let b = bundleOf(LR, f);
  let bf = f32(b);
  let lat = f32(q) - 0.5 * (bf - 1.0);
  // lateral direction: +U for warp yarns, +V for weft yarns (see cross(N, T) identities)
  let side = normalize(cross(o.n, o.t));
  let pitchLat = select(LR.pitch.y, LR.pitch.x, f == 1u);
  let latSign = select(1.0, -1.0, f == 1u);
  o.fam = select(0u, 1u, f == 1u);
  o.yarn = i32(select(j, i, f == 1u) * b + q);
  // coordinate along the yarn in crossing pitches of the other family
  let ba = f32(bundleOf(LR, 1u - f));
  o.c = (f32(a) + t) * ba + 0.5 * (ba - 1.0);
  o.layer = layer;
  // crimp follows the same over/under order as the intact surface and straightens under strain
  let strain = max(length(pb - pa) / segRest(n, f) - 1.0, 0.0);
  let decrimp = clamp(1.0 - strain / 0.03, 0.0, 1.0);
  var crimp = 0.0;
  if (f == 1u) {
    crimp = LR.warp.z * crimpProfile(warpUnderDist(LR, o.yarn, o.c)).x;
  } else {
    crimp = -LR.weft.z * crimpProfile(weftOverDist(LR, o.yarn, o.c)).x;
  }
  o.p = P + side * (latSign * lat * pitchLat) + o.n * (crimp * decrimp);
  o.wear = mix(wearR[n], wearR[nb], t);
  o.radius = 0.4 * select(LR.weft.x, LR.warp.x, f == 1u);
  return o;
}

// Strand of bundled yarn q in span s.
fn strandOf(s: u32, q: u32) -> u32 { return q / yarnsPerStrand(R.layers[nodes[s >> 1u].ijl >> 24u], s & 1u); }

// Bundled yarn q of span s belongs to an absent or separated strand.
fn yarnBroken(s: u32, q: u32) -> bool { return ((segw[2u * s + 1u].y >> (8u + strandOf(s, q))) & 1u) == 1u; }

// Failure site of bundled yarn q of span s (fraction from the owning node), as the torn corner
// patches clip it (yarnSiteOf in render_common.wgsl).
fn yarnSite(s: u32, q: u32) -> f32 { return yarnSiteOf(segw[2u * s + 1u].w, s, strandOf(s, q), q); }

// Point u in [0, 1] along half h (0: from the owning node, 1: from the far node) of bundled yarn q
// of span s. An intact yarn's halves meet in the middle. A broken yarn is two frayed stubs ending
// short of its failure site (the ends recoil and untwist): their free ends droop, curl, fan apart
// and thin out. Render-only shape on the mechanical span.
fn halfYarn(s: u32, q: u32, h: u32, u: f32) -> YarnPoint {
  if (!yarnBroken(s, q)) { return spanYarn(s, q, select(0.5 * u, 1.0 - 0.5 * u, h == 1u)); }
  let site = yarnSite(s, q);
  let seed = hash11(s * 31u + q * 7u + h * 3u + 5u);
  let recoil = 0.05 + 0.1 * hashf(seed + 9u);
  let reach = max(select(site, 1.0 - site, h == 1u) - recoil, 0.06);
  var o = spanYarn(s, q, select(reach * u, 1.0 - reach * u, h == 1u));
  let n = s >> 1u;
  let L = R.layers[nodes[n].ijl >> 24u].lattice;
  let nb = n + select(L.y, 1u, (s & 1u) == 0u);
  let len = reach * distance(x[n].xyz, x[nb].xyz);
  let curl = vec3<f32>(hashf(seed + 1u) - 0.5, hashf(seed + 2u) - 0.5, hashf(seed + 3u) - 0.5);
  let side = normalize(cross(o.n, o.t));
  let lat = f32(q) - 0.5 * (f32(bundleOf(R.layers[o.layer], s & 1u)) - 1.0);
  let bend = curl * (0.5 * len) + vec3<f32>(0.0, 0.0, -0.25 * len) + side * (lat * 0.35 * len);
  o.p += bend * (u * u);
  o.t = normalize(o.t * select(len, -len, h == 1u) + bend * (2.0 * u));
  // the end untwists into loose fibres (fuzz pass): the core thins toward it, and a stub shorter
  // than the yarn is wide is mostly fibre
  o.radius *= (1.0 - 0.6 * u) * clamp(len / (3.0 * o.radius), 0.25, 1.0);
  return o;
}

// Span separated on tail side e = 4 n + side (side: 0 u-, 1 u+, 2 v-, 3 v+) of node n.
fn tailSpan(e: u32) -> u32 {
  let n = e >> 2u;
  let side = e & 3u;
  let f = side >> 1u;
  let L = R.layers[nodes[n].ijl >> 24u].lattice;
  let stride = select(L.y, 1u, f == 0u);
  return 2u * select(n - stride, n, (side & 1u) == 1u) + f;
}

// Length of loose yarn q of tail e beyond the node's own quarter of the torn cell (the torn corner
// patch draws the woven part up to there): the yarn's remnant to its own failure site plus any length
// pulled out through the crossing, less half the span. <= 0: the yarn ends inside the quarter.
fn tailLength(e: u32, q: u32) -> f32 {
  let s = tailSpan(e);
  let plus = (e & 1u) == 1u;
  let Lr = segRest(s >> 1u, s & 1u);
  let failSite = bitcast<f32>(segw[2u * s + 1u].z);
  let own = select(1.0 - yarnSite(s, q), yarnSite(s, q), plus);
  let spanOwn = select(1.0 - failSite, failSite, plus);
  let extra = max(tails[e] - spanOwn * Lr, 0.0);
  return min(own * Lr + extra - 0.5 * Lr, 0.008);
}

// Loose yarn end q of tail e, t in [0, 1] from the edge of the node's quarter to the free end.
// Render-only shape: it leaves along the node's frame (as the torn corner patch it continues), sags
// under gravity, curls and fans out toward the end.
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
  let s = tailSpan(e);
  let Lr = segRest(s >> 1u, f);
  let b = bundleOf(LR, f);
  let lat = f32(q) - 0.5 * (f32(b) - 1.0);
  let seed = hash11(e * 7u + q * 131u + 17u);
  let len = max(tailLength(e, q), 0.0);
  let pitchLat = select(LR.pitch.y, LR.pitch.x, f == 1u);
  let curl = vec3<f32>(hashf(seed + 1u) - 0.5, hashf(seed + 2u) - 0.5, hashf(seed + 3u) - 0.5);
  var o: YarnPoint;
  let p0 = x[n].xyz + d0 * (0.5 * Lr) + sideV * (lat * pitchLat);
  let fan = sideV * (lat * pitchLat * 1.2 * t) + curl * (0.6 * len * t * t);
  let sag = vec3<f32>(0.0, 0.0, -0.35 * len * t * t) + N * (0.15 * len * t * (1.0 - t));
  o.p = p0 + d0 * (len * t) + fan + sag;
  let dp = d0 * len + sideV * (lat * pitchLat * 1.2) + curl * (1.2 * len * t) + vec3<f32>(0.0, 0.0, -0.7 * len * t) + N * (0.15 * len * (1.0 - 2.0 * t));
  o.t = normalize(dp + d0 * 1e-6);
  o.n = N;
  o.fam = select(0u, 1u, f == 1u);
  o.yarn = i32(select(j, i, f == 1u) * b + q);
  let ba = f32(bundleOf(LR, 1u - f));
  o.c = f32(a) * ba + 0.5 * (ba - 1.0) + dirSign * (0.5 * Lr + t * len) / select(LR.pitch.x, LR.pitch.y, f == 1u);
  o.layer = layer;
  o.wear = wearR[n];
  o.radius = 0.4 * select(LR.weft.x, LR.warp.x, f == 1u) * (1.0 - 0.55 * t);
  return o;
}

fn spanEntry(k: u32) -> u32 { return lists[SPAN_BASE + k] & 0x3FFFFFFFu; }

// Bundled yarn q of list entry k lies inside an intact neighbour cell, whose surface already shows it.
fn yarnInSurface(k: u32, q: u32, b: u32) -> bool {
  let e = lists[SPAN_BASE + k];
  let lat = 2.0 * f32(q) - f32(b - 1u);
  return (lat > 0.0 && (e & 0x40000000u) != 0u) || (lat < 0.0 && (e & 0x80000000u) != 0u);
}
fn tailEntry(k: u32) -> u32 { return lists[SPAN_BASE + MAX_SPANS + k]; }
