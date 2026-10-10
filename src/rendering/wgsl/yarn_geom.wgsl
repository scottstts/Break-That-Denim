// Loose thread geometry shared by the thread and fibre passes. Requires module-scope bindings
// R (RenderU), nodes, x, nrm, tframe, segw, cellFlagsY, lists (u32 view), fieldY (yarn damage field)
// and the list layout constants.
//
// Cloth is drawn where the weave is intact (torn.wgsl); a yarn whose crossing yarns are gone runs on
// alone as a thread: a pick across a ladder of worn-through warps, a bridge over an opening, an end
// hanging from its edge. Per lattice span next to a torn cell (refine.wgsl lists them) each of its
// yarns is drawn over its runs of lone crossings (yarn present, crossing yarn separated): the first
// run and the last, as two halves. A run that ends at a separated segment of the yarn itself has a
// free end that droops and curls; one ending in woven cloth or at the span's end continues there.
// Threads follow the mechanical nodes (they carry the bundle's load and move with it), sit side by side
// across the bundle (drifting into small clusters, as freed picks do) and each hangs by its own slack:
// a yarn that slid a little further through the weave is a few percent longer, so as a span's ends
// approach each other its threads sag one after another instead of moving as a ribbon.

const SPAN_BASE: u32 = 4u;
// Points per drawn half thread.
const STRAND_PTS: u32 = 9u;
// Instances per listed span: up to 9 yarns, each drawn in two halves.
const BUNDLE_HALVES: u32 = 18u;

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

fn bundleEntry(k: u32) -> u32 { return lists[SPAN_BASE + k]; }

// Chain neighbour of node n along family f in direction dir through an intact span, else the
// extrapolation from `other` (the node on the opposite side).
fn chainPoint(n: u32, f: u32, dir: i32, other: vec3<f32>) -> vec3<f32> {
  let ijl = nodes[n].ijl;
  let L = R.layers[ijl >> 24u].lattice;
  let a = i32(select((ijl >> 12u) & 0xFFFu, ijl & 0xFFFu, f == 0u));
  let len = i32(select(L.z, L.y, f == 0u));
  let stride = select(L.y, 1u, f == 0u);
  if (dir > 0 && a + 1 < len && segOkR(n, f)) { return x[n + stride].xyz; }
  if (dir < 0 && a > 0 && segOkR(n - stride, f)) { return x[n - stride].xyz; }
  return 2.0 * x[n].xyz - other;
}

// Point at fraction u of drawn half `half` of yarn q of listed span s; valid = false when that half
// draws nothing.
struct StrandPoint {
  y: YarnPoint,
  valid: bool,
};

fn strandPoint(s: u32, q: u32, half: u32, u: f32) -> StrandPoint {
  var o: StrandPoint;
  o.valid = false;
  let a = s >> 1u;
  let f = s & 1u;
  let ijl = nodes[a].ijl;
  let layer = ijl >> 24u;
  let LR = R.layers[layer];
  let L = LR.lattice;
  let B = bundleOf(LR, f);
  if (q >= B) { return o; }
  let ci = ijl & 0xFFFu;
  let cj = (ijl >> 12u) & 0xFFFu;
  let b = a + select(L.y, 1u, f == 0u);
  // the cell this yarn lies in (beside the span); an intact cell draws its own yarns
  var cell: i32;
  if (f == 0u) {
    let below = q < B / 2u;
    if (below && cj == 0u) { return o; }
    cell = i32(select(cj, cj - 1u, below) * (L.y - 1u) + ci);
  } else {
    let left = q < B / 2u;
    if (left && ci == 0u) { return o; }
    if (!left && ci + 1u >= L.y) { return o; }
    cell = i32(cj * (L.y - 1u) + select(ci, ci - 1u, left));
  }
  if (cellFlagsY[u32(cell) + LR.refine.z] != 0u) { return o; }
  // the yarn along the span: lone crossings (thread), woven ones (cloth), separated ones (gap)
  let bu = bundleOf(LR, 1u);
  let bv = bundleOf(LR, 0u);
  let Pn = select(bu, bv, f == 1u);
  let FL = LR.field;
  var state: array<u32, 16>;
  for (var p = 0u; p < Pn; p++) {
    var c: u32;
    if (f == 1u) {
      c = FL.x + (cj * bv + bv / 2u + p) * FL.y + ci * bu + q;
    } else {
      c = FL.x + (cj * bv + q) * FL.y + ci * bu + bu / 2u + p;
    }
    let w = fieldY[c];
    let own = yarnCode(w, f) != CUT_R;
    let other = yarnCode(w, 1u - f) != CUT_R;
    state[p] = select(0u, select(1u, 2u, other), own);
  }
  // first and last run of lone crossings
  var r0s = -1;
  var r0e = -1;
  var r1s = -1;
  var r1e = -1;
  var p = 0;
  loop {
    if (p >= i32(Pn)) { break; }
    if (state[p] != 1u) {
      p++;
      continue;
    }
    let st = p;
    loop {
      if (p >= i32(Pn) || state[p] != 1u) { break; }
      p++;
    }
    if (r0s < 0) {
      r0s = st;
      r0e = p;
    }
    r1s = st;
    r1e = p;
  }
  if (r0s < 0) { return o; }
  // one run: draw it in two halves; two or more: the first and the last
  var rs: i32;
  var re: i32;
  var ta: f32;
  var tb: f32;
  if (r0s == r1s) {
    let mid = 0.5 * f32(r0s + r0e);
    ta = select(mid, f32(r0s), half == 0u);
    tb = select(f32(r0e), mid, half == 0u);
    rs = r0s;
    re = r0e;
  } else {
    rs = select(r1s, r0s, half == 0u);
    re = select(r1e, r0e, half == 0u);
    ta = f32(rs);
    tb = f32(re);
  }
  // ends: free where the yarn itself has separated next to the run, held elsewhere
  let freeA = rs > 0 && state[rs - 1] == 0u;
  let freeB = re < i32(Pn) && state[re] == 0u;
  if (freeA && freeB) { return o; }
  o.valid = true;
  let t = mix(ta, tb, u) / f32(Pn);
  let pa = x[a].xyz;
  let pb = x[b].xyz;
  let rest = segRest(a, f);
  let seedY = hash11((u32(layer) * 7919u + u32(select(ci, cj, f == 0u)) * 131u + q) * 2654435761u + f);
  let seed = hash11(s * 977u + q * 131u + 7u);
  // across the bundle: yarns side by side over its width along the other family's frame, drifting
  // into clusters of two or three (per chain of spans, so a thread stays continuous)
  let side = normalize(mix(tframe[2u * a + f].xyz, tframe[2u * b + f].xyz, t) + vec3<f32>(1e-7, 0.0, 0.0));
  let across = select(tframe[2u * a].xyz, tframe[2u * a + 1u].xyz, f == 0u);
  let lateral = normalize(across - dot(across, side) * side + vec3<f32>(0.0, 0.0, 1e-7));
  let width = f32(B) * select(LR.pitch.y, LR.pitch.x, f == 1u);
  let cl = hash11(u32(layer) * 31u + u32(select(ci, cj, f == 0u)) * 7u + f);
  let size = 2u + ((cl >> 1u) & 1u);
  let cluster = (q + (cl & 1u)) / size;
  let c0 = f32(cluster * size) - f32(cl & 1u);
  let cmid = (clamp(c0 + 0.5 * f32(size - 1u), 0.0, f32(B) - 1.0) + 0.5) / f32(B) - 0.5;
  let ownLat = (f32(q) + 0.5) / f32(B) - 0.5 + 0.3 * (hashf(seedY) - 0.5) / f32(B);
  let lat = mix(ownLat, cmid, 0.3);
  var pos: vec3<f32>;
  var dp: vec3<f32>;
  let broken = (segw[2u * s + 1u].y & SEG_BROKEN_R) != 0u;
  if (broken) {
    // the span has separated: each piece hangs from its own node along that node's frame
    let fromA = !freeA || rs == 0;
    let n = select(b, a, fromA);
    let dirN = select(-1.0, 1.0, fromA) * tframe[2u * n + f].xyz;
    let along = select(1.0 - t, t, fromA) * rest;
    pos = x[n].xyz + dirN * along;
    dp = dirN * select(-1.0, 1.0, fromA);
  } else {
    let p0 = chainPoint(a, f, -1, pb);
    let p3 = chainPoint(b, f, 1, pa);
    pos = cmr(p0, pa, pb, p3, t);
    dp = cmrd(p0, pa, pb, p3, t);
  }
  pos += lateral * (lat * width);
  // gravity sag of this yarn's own slack over its run (parabolic hang: arc = chord + 8 h^2 / (3 chord))
  let chord = length(pb - pa);
  let slack = hashf(seedY + 1u);
  let restK = rest * (1.0 + 0.05 * slack * slack);
  let down = vec3<f32>(0.0, 0.0, -1.0);
  let dch = normalize(dp + vec3<f32>(1e-9, 0.0, 0.0));
  var sagDir = down - dot(down, dch) * dch;
  sagDir = normalize(sagDir + lateral * (0.6 * (hashf(seedY + 2u) - 0.5)) + vec3<f32>(0.0, 0.0, -1e-6));
  let ra = f32(rs) / f32(Pn);
  let rb = f32(re) / f32(Pn);
  let tr = clamp((t - ra) / max(rb - ra, 1e-3), 0.0, 1.0);
  let runLen = (rb - ra) * rest;
  if (freeA || freeB) {
    // a loose end: droops and curls away from where it is held, more toward its tip
    let fu = select(tr, 1.0 - tr, freeA);
    let curl = vec3<f32>(hashf(seed + 3u) - 0.5, hashf(seed + 4u) - 0.5, -0.7) * (0.9 * runLen);
    pos += curl * fu * fu + lateral * (0.25 * runLen * fu);
    dp += curl * (2.0 * fu) * select(1.0, -1.0, freeA) / max(runLen, 1e-6) * rest;
  } else if (!broken) {
    let hang = sqrt(max(0.375 * chord * (restK - chord), 0.0)) * (rb - ra);
    pos += sagDir * (4.0 * hang * tr * (1.0 - tr));
    dp += sagDir * (4.0 * hang * (1.0 - 2.0 * tr)) / max(rb - ra, 1e-3);
  }
  o.y.p = pos;
  o.y.t = normalize(dp + side * 1e-7);
  o.y.n = normalize(nrm[a].xyz + nrm[b].xyz + vec3<f32>(0.0, 0.0, 1e-6));
  o.y.fam = f;
  let crossI = select(i32(ci), i32(cj), f == 0u);
  let alongI = select(i32(cj), i32(ci), f == 0u);
  o.y.yarn = crossI * i32(B) + i32(q);
  let ba = f32(bundleOf(LR, 1u - f));
  o.y.c = (f32(alongI) + t) * ba + 0.5 * (ba - 1.0);
  o.y.layer = layer;
  // freed from the weave the yarn is abraded and loosened: its pale core shows
  o.y.wear = min(1.0, 0.5 * (wearR[a] + wearR[b]) + 0.35);
  // a loosened, untwisting round yarn: fuller than its flattened width in the weave, thinning toward
  // a free end
  let yw = select(LR.weft.x, LR.warp.x, f == 1u);
  let tip = select(0.0, select(tr, 1.0 - tr, freeA), freeA || freeB);
  o.y.radius = 0.5 * yw * (0.85 + 0.3 * hashf(seedY + 5u)) * (1.0 - 0.5 * tip);
  return o;
}
