// Explicit yarn geometry shared by the yarn, tail and fuzz passes. Requires module-scope bindings
// R (RenderU), nodes, x, nrm, segw, tails, lists (u32 view) and the list layout constants.

const SPAN_BASE: u32 = 4u;
override MAX_SPANS: u32 = 65536u;

fn segOkR(owner: u32, f: u32) -> bool {
  let fl = segw[2u * (2u * owner + f) + 1u].y;
  return (fl & (SEG_EXISTS_R | SEG_BROKEN_R)) == SEG_EXISTS_R;
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

// Loose yarn end q of the tail at node side e = 4 n + side (side: 0 u-, 1 u+, 2 v-, 3 v+), t in [0, 1]
// from the node to the free end. Render-only shape: it continues the yarn direction, sags under
// gravity, curls and fans out toward the end; its length is the tail's material length.
fn tailYarn(e: u32, q: u32, t: f32) -> YarnPoint {
  let n = e >> 2u;
  let sideIdx = e & 3u;
  let f = sideIdx >> 1u;
  let dirSign = select(-1.0, 1.0, (sideIdx & 1u) == 1u);
  let ijl = nodes[n].ijl;
  let layer = ijl >> 24u;
  let LR = R.layers[layer];
  let L = LR.lattice;
  let i = ijl & 0xFFFu;
  let j = (ijl >> 12u) & 0xFFFu;
  let stride = select(L.y, 1u, f == 0u);
  let a = select(j, i, f == 0u);
  let amax = select(L.z, L.y, f == 0u);
  let N = nrm[n].xyz;
  // continuation of the yarn through the node
  var d0 = vec3<f32>(0.0);
  if (dirSign > 0.0 && a > 0u && segOkR(n - stride, f)) { d0 = x[n].xyz - x[n - stride].xyz; }
  if (dirSign < 0.0 && a + 1u < amax && segOkR(n, f)) { d0 = x[n].xyz - x[n + stride].xyz; }
  if (dot(d0, d0) < 1e-14) {
    let ref_ = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), f == 0u);
    d0 = dirSign * (ref_ - dot(ref_, N) * N);
  }
  d0 = normalize(d0);
  let b = bundleOf(LR, f);
  let bf = f32(b);
  let lat = f32(q) - 0.5 * (bf - 1.0);
  let seed = hash11(e * 7u + q * 131u + 17u);
  let len = min(tails[e], 0.006) * (0.75 + 0.4 * hashf(seed));
  let side = normalize(cross(N, d0));
  let pitchLat = select(LR.pitch.y, LR.pitch.x, f == 1u);
  let curl = vec3<f32>(hashf(seed + 1u) - 0.5, hashf(seed + 2u) - 0.5, hashf(seed + 3u) - 0.5);
  var o: YarnPoint;
  let fan = side * (lat * pitchLat * (1.0 + 1.8 * t)) + curl * (0.6 * len * t * t);
  let sag = vec3<f32>(0.0, 0.0, -0.35 * len * t * t) + N * (0.15 * len * t * (1.0 - t));
  o.p = x[n].xyz + d0 * (len * t) + fan + sag;
  let dp = d0 * len + side * (lat * pitchLat * 1.8) + curl * (1.2 * len * t) + vec3<f32>(0.0, 0.0, -0.7 * len * t) + N * (0.15 * len * (1.0 - 2.0 * t));
  o.t = normalize(dp);
  o.n = N;
  o.fam = select(0u, 1u, f == 1u);
  o.yarn = i32(select(j, i, f == 1u) * b + q);
  let ba = f32(bundleOf(LR, 1u - f));
  o.c = f32(a) * ba + 0.5 * (ba - 1.0) + dirSign * t * len / select(LR.pitch.x, LR.pitch.y, f == 1u);
  o.layer = layer;
  o.wear = wearR[n];
  o.radius = 0.4 * select(LR.weft.x, LR.warp.x, f == 1u) * (1.0 - 0.55 * t);
  return o;
}

fn spanEntry(k: u32) -> u32 { return lists[SPAN_BASE + k]; }
fn tailEntry(k: u32) -> u32 { return lists[SPAN_BASE + MAX_SPANS + k]; }
