// Shared render definitions: frame/layer uniforms, hashing and noise, the woven yarn height model,
// yarn appearance and the fiber BRDF. Used by the surface, explicit-yarn, fuzz and shadow passes so
// intact cloth and freed yarns are the same material.

struct FrameU {
  viewProj: mat4x4<f32>,
  lightViewProj: mat4x4<f32>,
  // xyz camera position, w = angular size of one drawing-buffer pixel (rad)
  camPos: vec4<f32>,
  // xyz key light position, w = key intensity
  keyPos: vec4<f32>,
  // xyz key axis (from the light toward the scene), w = cos of the cone's soft edge start
  keyAxis: vec4<f32>,
  // rgb ambient (hemisphere top), w = shadow-map texel size (m)
  ambient: vec4<f32>,
  // width, height, 1/width, 1/height (drawing buffer)
  viewport: vec4<f32>,
  // exposure, time (s), key distance to the view target (irradiance normalisation, m), frame index
  params: vec4<f32>,
};

struct LayerR {
  // node offset, nx, ny, bundle bytes (warps per column, picks per row, warp strands, pick strands)
  lattice: vec4<u32>,
  // fine yarn pitch u (between warps), pitch v (between picks), twill direction (+-1), unused
  pitch: vec4<f32>,
  // warp (v-family): width, height, crimp amplitude, surface fiber twist angle (rad)
  warp: vec4<f32>,
  // weft (u-family): width, height, crimp amplitude, surface fiber twist angle (rad)
  weft: vec4<f32>,
  // twill repeat, warp-over count, refinement S (sub-quads per cell edge), unused
  twill: vec4<u32>,
  // refined-vertex offset, refined row length, cell offset, material id
  refine: vec4<u32>,
  // edge field: first texel row, texel size (m), texture width, texture height (texels)
  edge: vec4<f32>,
};

struct RenderU {
  // node count, layer count, cell count, refined vertex count
  counts: vec4<u32>,
  layers: array<LayerR, 3>,
};

struct NodeStaticR {
  ijl: u32,
  flags: u32,
  mass: f32,
  n0: f32,
};

const SEG_EXISTS_R: u32 = 1u;
const SEG_BROKEN_R: u32 = 2u;
const PI: f32 = 3.14159265;

// Yarns carried by one lattice span of family f (1 = warp spans: warps per column; 0 = picks per row).
fn bundleOf(L: LayerR, f: u32) -> u32 { return select((L.lattice.w >> 8u) & 0xFFu, L.lattice.w & 0xFFu, f == 1u); }
fn bundleMax(L: LayerR) -> u32 { return max((L.lattice.w >> 8u) & 0xFFu, L.lattice.w & 0xFFu); }
// Yarns per failure strand of family f's spans.
fn yarnsPerStrand(L: LayerR, f: u32) -> u32 {
  return bundleOf(L, f) / select(L.lattice.w >> 24u, (L.lattice.w >> 16u) & 0xFFu, f == 1u);
}

// ---------------------------------------------------------------- hashing and noise

fn hash11(x: u32) -> u32 {
  var v = x;
  v ^= v >> 16u;
  v *= 0x7feb352du;
  v ^= v >> 15u;
  v *= 0x846ca68bu;
  v ^= v >> 16u;
  return v;
}
fn hashf(x: u32) -> f32 { return f32(hash11(x)) * (1.0 / 4294967296.0); }
fn hash2f(a: u32, b: u32) -> f32 { return hashf(a ^ hash11(b + 0x9e3779b9u)); }
fn hashi(a: i32, b: u32) -> f32 { return hash2f(bitcast<u32>(a), b); }

// Smooth 1D value noise in [-1, 1] along a coordinate (seeded).
fn noise1(seed: u32, x: f32) -> f32 {
  let i = floor(x);
  let f = x - i;
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash2f(seed, u32(i32(i) + 65536));
  let b = hash2f(seed, u32(i32(i) + 65537));
  return mix(a, b, u) * 2.0 - 1.0;
}

// Smooth 2D value noise in [-1, 1].
fn noise2(seed: u32, p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = p - i;
  let u = f * f * (3.0 - 2.0 * f);
  let ix = u32(i32(i.x) + 32768);
  let iy = u32(i32(i.y) + 32768);
  let a = hash2f(seed ^ hash11(iy), ix);
  let b = hash2f(seed ^ hash11(iy), ix + 1u);
  let c = hash2f(seed ^ hash11(iy + 1u), ix);
  let d = hash2f(seed ^ hash11(iy + 1u), ix + 1u);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 2.0 - 1.0;
}

// ---------------------------------------------------------------- woven yarn geometry

// Longitudinal crimp profile: -1 at the centre of an under-run, +1 one pitch away and beyond.
// Returns (profile, d profile / d coordinate) for a coordinate measured in crossing pitches.
fn crimpProfile(d: f32) -> vec2<f32> {
  let a = min(abs(d), 1.0);
  // smoothstep-shaped transition (C1, cheap): -1 at the run centre, +1 from one pitch on
  return vec2<f32>(-1.0 + 2.0 * a * a * (3.0 - 2.0 * a), 6.0 * a * (1.0 - a) * sign(d));
}

// Signed distance (in pitches) from coordinate c to the nearest centre c0 + R n.
fn periodic(c: f32, c0: f32, R: f32) -> f32 {
  return c - c0 - R * round((c - c0) / R);
}

// Warp yarn k: signed pick distance from V to the centre of its nearest under-run.
fn warpUnderDist(L: LayerR, k: i32, V: f32) -> f32 {
  let R = f32(L.twill.x);
  let over = f32(L.twill.y);
  let c0 = L.pitch.z * f32(k) + over + 0.5 * (R - over - 1.0);
  return periodic(V, c0, R);
}

// Weft yarn m: signed warp distance from U to the centre of its nearest over-run (where it floats).
fn weftOverDist(L: LayerR, m: i32, U: f32) -> f32 {
  let R = f32(L.twill.x);
  let over = f32(L.twill.y);
  // warp k is under pick m when (m - s k) mod R >= over  =>  s k = m - over - (R - over - 1) / 2 (centre)
  let c0 = L.pitch.z * (f32(m) - over - 0.5 * (R - over - 1.0));
  return periodic(U, c0, R);
}

struct Hit {
  covered: bool,
  // 1 = warp (runs along v), 0 = weft (runs along u)
  fam: u32,
  yarn: i32,
  // which float of this yarn (for per-float variation)
  float_: i32,
  // physical position along the yarn (m)
  along: f32,
  // lateral position across the yarn, -1..1
  x: f32,
  // height above the midsurface (m)
  h: f32,
  // dh/du_phys, dh/dv_phys (surface slope in physical tangent coordinates)
  slope: vec2<f32>,
};

// Height of warp k at (U, V). Physical scales: su, sv are the local stretch ratios of the lattice.
fn warpHit(L: LayerR, k: i32, U: f32, V: f32, su: f32, sv: f32) -> Hit {
  var o: Hit;
  o.fam = 1u;
  o.yarn = k;
  let a = 0.5 * L.warp.x;
  let ac = (U - f32(k)) * L.pitch.x * su;
  o.x = ac / a;
  o.covered = abs(o.x) < 1.0;
  let d = warpUnderDist(L, k, V);
  let R = f32(L.twill.x);
  o.float_ = i32(round((V - d) / R));
  let cp = crimpProfile(d);
  let e = sqrt(max(1.0 - o.x * o.x, 0.0));
  o.h = L.warp.z * cp.x + 0.5 * L.warp.y * e;
  let de = -o.x / max(e, 0.12) / a;
  o.slope = vec2<f32>(0.5 * L.warp.y * de, L.warp.z * cp.y / (L.pitch.y * sv));
  o.along = V * L.pitch.y;
  return o;
}

fn weftHit(L: LayerR, m: i32, U: f32, V: f32, su: f32, sv: f32) -> Hit {
  var o: Hit;
  o.fam = 0u;
  o.yarn = m;
  let a = 0.5 * L.weft.x;
  let ac = (V - f32(m)) * L.pitch.y * sv;
  o.x = ac / a;
  o.covered = abs(o.x) < 1.0;
  let d = weftOverDist(L, m, U);
  let R = f32(L.twill.x);
  o.float_ = i32(round((U - d) / R));
  let cp = crimpProfile(d);
  let e = sqrt(max(1.0 - o.x * o.x, 0.0));
  // floats (profile -1 at the over-run centre) rise; elsewhere the weft lies below the warp
  o.h = -L.weft.z * cp.x + 0.5 * L.weft.y * e;
  let de = -o.x / max(e, 0.12) / a;
  o.slope = vec2<f32>(-L.weft.z * cp.y / (L.pitch.x * su), 0.5 * L.weft.y * de);
  o.along = U * L.pitch.x;
  return o;
}

// ---------------------------------------------------------------- torn cells: groups and cracks
//
// A torn cell (one with a separated span) splits into groups of corners still joined by its intact
// spans. Edges in counter-clockwise order: 0 bottom (c0-c1), 1 right (c1-c3), 2 top (c3-c2), 3 left
// (c2-c0); the corner before edge k in that order is ccwCorner(k). Each broken edge is crossed by the
// crack where its yarns failed. A crack leaves every crossing at right angles to the yarns it cut, so
// it continues smoothly into the neighbouring cell instead of following the lattice; a two-way split
// joins its two crossings directly, three or more cracks meet at a junction, and a lone broken edge is
// a crack tip ending inside the cell. Each group owns the part of the cell on its side of the cracks
// and carries it away when it separates: the crack is the free edge, not a lattice line.
// Geometry is in metres in the cell's rest frame (origin at c0, +u along s, +v along t).

// tornGeo record per torn cell (torn.wgsl): info (cell, broken mask, packed crossings, layer as bits);
// the cell's cracks (TORN_CRACKS + 2 i: Bezier lo, hi in metres from c0, up to four); then per corner a
// 16-point control net (w < 0: the corner does not lead a group) followed by its group's free-edge
// curves (groupCurves: four vec4).
const TORN_STRIDE: u32 = 89u;
const TORN_CRACKS: u32 = 1u;
fn tornSlot(k: u32, c: u32) -> u32 { return k * TORN_STRIDE + 9u + 20u * c; }
// Edge fibres rooted along the cracks of one torn cell (fuzz.wgsl).
const FIBERS_PER_CELL: u32 = 96u;

fn tornInfo(geo: vec4<f32>) -> vec4<u32> { return bitcast<vec4<u32>>(geo); }

// Corner index (bit 0: +i, bit 1: +j) at counter-clockwise position k.
fn ccwCorner(k: u32) -> u32 { return (k & 3u) ^ ((k & 3u) >> 1u); }
fn cornerPos(c: u32, size: vec2<f32>) -> vec2<f32> { return vec2<f32>(f32(c & 1u), f32(c >> 1u)) * size; }

// Span (2 * owner + family) of edge e of the cell whose lower-left node is n00, in a lattice of row
// length nx.
fn edgeSpan(n00: u32, nx: u32, e: u32) -> u32 {
  switch e {
    case 0u: { return 2u * n00; }
    case 1u: { return 2u * (n00 + 1u) + 1u; }
    case 2u: { return 2u * (n00 + nx); }
    default: { return 2u * n00 + 1u; }
  }
}

// Where the crack crosses a broken span (fraction from its owning node): the yarns' failure site,
// drawn a little toward the middle so neighbouring crossings do not zigzag.
fn crossingOf(failSite: f32) -> f32 { return 0.5 + 0.7 * (clamp(failSite, 0.0, 1.0) - 0.5); }

fn edgeCross(e: u32, c: f32, size: vec2<f32>) -> vec2<f32> {
  switch e {
    case 0u: { return vec2<f32>(c, 0.0) * size; }
    case 1u: { return vec2<f32>(1.0, c) * size; }
    case 2u: { return vec2<f32>(c, 1.0) * size; }
    default: { return vec2<f32>(0.0, c) * size; }
  }
}

fn edgeInward(e: u32) -> vec2<f32> {
  switch e {
    case 0u: { return vec2<f32>(0.0, 1.0); }
    case 1u: { return vec2<f32>(-1.0, 0.0); }
    case 2u: { return vec2<f32>(0.0, -1.0); }
    default: { return vec2<f32>(1.0, 0.0); }
  }
}

struct Torn {
  // broken edges (bit e)
  mask: u32,
  // rest cell size (m)
  size: vec2<f32>,
  // crack crossings of the edges (m from c0; meaningful for broken edges)
  P: array<vec2<f32>, 4>,
  // junction of three or more cracks, or the end of a lone crack (tip)
  J: vec2<f32>,
};

// Crossing fractions are packed as u8 per edge (bits 8e).
fn makeTorn(mask: u32, crossings: u32, size: vec2<f32>) -> Torn {
  var o: Torn;
  o.mask = mask;
  o.size = size;
  var sum = vec2<f32>(0.0);
  for (var e = 0u; e < 4u; e++) {
    o.P[e] = edgeCross(e, f32((crossings >> (8u * e)) & 0xFFu) / 255.0, size);
    if (((mask >> e) & 1u) == 1u) { sum += o.P[e]; }
  }
  let k = countOneBits(mask);
  if (k == 1u) {
    let e = firstTrailingBit(mask);
    o.J = mix(o.P[e], 0.5 * size, 0.8);
  } else {
    o.J = sum / f32(max(k, 1u));
  }
  return o;
}

// Edge of the i-th set bit of mask.
fn nthEdge(mask: u32, i: u32) -> u32 {
  var seen = 0u;
  for (var e = 0u; e < 4u; e++) {
    if (((mask >> e) & 1u) == 1u) {
      if (seen == i) { return e; }
      seen++;
    }
  }
  return 0u;
}

// Number of cracks: one for a two-way split, else one per broken edge (to the junction or tip).
fn crackCount(t: Torn) -> u32 {
  let k = countOneBits(t.mask);
  return select(k, 1u, k == 2u);
}

// Cubic Bezier control points of crack i. A two-way split runs from its lower to its higher edge;
// otherwise crack i runs from the i-th broken edge to the junction.
fn crackBezier(t: Torn, i: u32) -> array<vec2<f32>, 4> {
  if (countOneBits(t.mask) == 2u) {
    let a = firstTrailingBit(t.mask);
    let b = firstLeadingBit(t.mask);
    let al = 0.39 * distance(t.P[a], t.P[b]);
    return array<vec2<f32>, 4>(t.P[a], t.P[a] + al * edgeInward(a), t.P[b] + al * edgeInward(b), t.P[b]);
  }
  let e = nthEdge(t.mask, i);
  let b1 = t.P[e] + 0.45 * distance(t.P[e], t.J) * edgeInward(e);
  return array<vec2<f32>, 4>(t.P[e], b1, mix(t.J, b1, 0.35), t.J);
}


// Segments per crack when its curve is evaluated as a polyline.
const CRACK_SEGS: u32 = 6u;

fn segmentDistance(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
  let ab = b - a;
  let h = clamp(dot(p - a, ab) / max(dot(ab, ab), 1e-20), 0.0, 1.0);
  return length(p - a - h * ab);
}

// Cubic Bezier packed as (b0, b1) and (b2, b3), evaluated without private arrays (per-fragment code).
fn bezierP(lo: vec4<f32>, hi: vec4<f32>, s: f32) -> vec2<f32> {
  let r = 1.0 - s;
  return r * r * r * lo.xy + 3.0 * r * r * s * lo.zw + 3.0 * r * s * s * hi.xy + s * s * s * hi.zw;
}
fn bezierTangentP(lo: vec4<f32>, hi: vec4<f32>, s: f32) -> vec2<f32> {
  let r = 1.0 - s;
  return 3.0 * r * r * (lo.zw - lo.xy) + 6.0 * r * s * (hi.xy - lo.zw) + 3.0 * s * s * (hi.zw - hi.xy);
}

fn bezierDistance(lo: vec4<f32>, hi: vec4<f32>, p: vec2<f32>) -> f32 {
  var d = 1e9;
  var a = lo.xy;
  for (var k = 1u; k <= CRACK_SEGS; k++) {
    let b = bezierP(lo, hi, f32(k) / f32(CRACK_SEGS));
    d = min(d, segmentDistance(p, a, b));
    a = b;
  }
  return d;
}

fn packBezier(B: array<vec2<f32>, 4>, reverse: bool) -> array<vec4<f32>, 2> {
  if (reverse) { return array<vec4<f32>, 2>(vec4<f32>(B[3], B[2]), vec4<f32>(B[1], B[0])); }
  return array<vec4<f32>, 2>(vec4<f32>(B[0], B[1]), vec4<f32>(B[2], B[3]));
}

// Group of the corner at counter-clockwise position k: the broken edges entering (a) and leaving (b)
// its run of corners, packed a | b << 2. With fewer than two broken edges the cell is one group.
fn groupOf(mask: u32, k: u32) -> u32 {
  var a = (k + 3u) & 3u;
  for (var q = 0u; q < 4u; q++) {
    if (((mask >> a) & 1u) == 1u) { break; }
    a = (a + 3u) & 3u;
  }
  var b = k & 3u;
  for (var q = 0u; q < 4u; q++) {
    if (((mask >> b) & 1u) == 1u) { break; }
    b = (b + 1u) & 3u;
  }
  return a | (b << 2u);
}

// The group's leader: its first corner counter-clockwise, which draws the group.
fn groupLeader(mask: u32, g: u32) -> u32 {
  if (countOneBits(mask) < 2u) { return 0u; }
  return ccwCorner((g & 3u) + 1u);
}

// Free edge of group g (a | b << 2) as two Beziers (lo, hi, lo, hi): from crossing b back to
// crossing a. A two-way split has one curve (the second unused); three or more cracks give b's crack
// out to the junction and a's crack back from it. A crack tip is the first curve alone.
fn groupCurves(t: Torn, g: u32) -> array<vec4<f32>, 4> {
  let k = countOneBits(t.mask);
  if (k < 2u) {
    let c = packBezier(crackBezier(t, 0u), false);
    return array<vec4<f32>, 4>(c[0], c[1], c[0], c[1]);
  }
  let a = g & 3u;
  let b = g >> 2u;
  if (k == 2u) {
    // the crack runs from the lower edge: forward when that is b
    let c = packBezier(crackBezier(t, 0u), firstTrailingBit(t.mask) != b);
    return array<vec4<f32>, 4>(c[0], c[1], c[0], c[1]);
  }
  var ib = 0u;
  var ia = 0u;
  for (var e = 0u; e < 4u; e++) {
    if (((t.mask >> e) & 1u) == 0u) { continue; }
    if (e < b) { ib++; }
    if (e < a) { ia++; }
  }
  let cb = packBezier(crackBezier(t, ib), false);
  let ca = packBezier(crackBezier(t, ia), true);
  return array<vec4<f32>, 4>(cb[0], cb[1], ca[0], ca[1]);
}

// Edge a -> b crosses the +x ray from p.
fn rayCross(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> u32 {
  if ((a.y > p.y) != (b.y > p.y)) {
    if (p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)) { return 1u; }
  }
  return 0u;
}

// Signed distance (m) from cell point p to the free edge of group g, positive on the group's side,
// from its boundary curves (groupCurves). The region (crossing a, its corners, crossing b, back along
// the curves) is streamed edge by edge for the inside test: no private arrays in per-fragment code.
fn groupSd(mask: u32, g: u32, size: vec2<f32>, c1lo: vec4<f32>, c1hi: vec4<f32>, c2lo: vec4<f32>, c2hi: vec4<f32>, p: vec2<f32>) -> f32 {
  let k = countOneBits(mask);
  if (k < 2u) { return bezierDistance(c1lo, c1hi, p); }
  let a = g & 3u;
  let b = g >> 2u;
  let pb = c1lo.xy;
  let pa = select(c2hi.zw, c1hi.zw, k == 2u);
  var n = 0u;
  var prev = pa;
  var idx = (a + 1u) & 3u;
  for (var q = 0u; q < 4u; q++) {
    let cur = cornerPos(ccwCorner(idx), size);
    n += rayCross(p, prev, cur);
    prev = cur;
    if (idx == b) { break; }
    idx = (idx + 1u) & 3u;
  }
  n += rayCross(p, prev, pb);
  var d = 1e9;
  prev = pb;
  for (var s = 1u; s <= CRACK_SEGS; s++) {
    let cur = bezierP(c1lo, c1hi, f32(s) / f32(CRACK_SEGS));
    n += rayCross(p, prev, cur);
    d = min(d, segmentDistance(p, prev, cur));
    prev = cur;
  }
  if (k >= 3u) {
    for (var s = 1u; s <= CRACK_SEGS; s++) {
      let cur = bezierP(c2lo, c2hi, f32(s) / f32(CRACK_SEGS));
      n += rayCross(p, prev, cur);
      d = min(d, segmentDistance(p, prev, cur));
      prev = cur;
    }
  }
  return select(-d, d, (n & 1u) == 1u);
}

// ---------------------------------------------------------------- edge field
//
// Distance (mm) from every material point to the nearest crack, in a texture over the rest chart
// (edgefield.wgsl draws it). The torn rim, its fringe and the abraded halo around holes are material
// features at a fixed distance from the free edge, so they follow it smoothly across cells.

// Largest stored distance (mm); everything beyond is intact cloth.
const EDGE_RMAX: f32 = 7.0;

// How far a crack has developed, 0..1: a lone crack tip (one broken edge in its cell) is a fresh cut
// through a few yarns whose sides still hold each other, so its rim is narrow and barely frayed. The
// edge field stores distances divided by it, which narrows every rim feature around a tip at once.
fn crackMaturity(mask: u32) -> f32 { return select(1.0, 0.3, countOneBits(mask) < 2u); }

// Cracks of a torn cell from its broken edges alone (crackCount without the geometry).
fn crackCountOf(mask: u32) -> u32 {
  let k = countOneBits(mask);
  return select(k, 1u, k == 2u);
}

// Edge-field texel of rest point (m) of layer L.
fn edgeTexel(L: LayerR, rest: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(rest.x / L.edge.y, rest.y / L.edge.y + L.edge.x);
}

// Rest-chart point (m) of fine yarn coordinates (U, V): node (i, j) sits at U = i bu + (bu - 1) / 2.
fn restOfUV(L: LayerR, uv: vec2<f32>) -> vec2<f32> {
  let bu = f32(bundleOf(L, 1u));
  let bv = f32(bundleOf(L, 0u));
  return vec2<f32>((uv.x - 0.5 * (bu - 1.0)) * L.pitch.x, (uv.y - 0.5 * (bv - 1.0)) * L.pitch.y);
}

// ---------------------------------------------------------------- woven surface lookup

// The two highest yarns at a point and the soft visibility weight of the second: real yarn surfaces
// are hairy, so the boundary between overlapping yarns is a soft transition over ~tau of height.
struct Hit2 {
  a: Hit,
  b: Hit,
  wb: f32,
};

fn weave2(L: LayerR, U: f32, V: f32, su: f32, sv: f32, front: bool, tau: f32) -> Hit2 {
  // heights only for every candidate; full hits for the two highest
  let sgn = select(-1.0, 1.0, front);
  let k0 = i32(round(U));
  let m0 = i32(round(V));
  let aw = 0.5 * L.warp.x;
  let af = 0.5 * L.weft.x;
  // warps overlap their neighbours (width > pitch): three candidates; picks only when stretched apart
  let wefts = select(1, 3, af > 0.5 * L.pitch.y * sv);
  var ha = -1.0;
  var hb = -1.0;
  var ia = -1;
  var ib = -1;
  for (var q = 0; q < 3 + wefts; q++) {
    var x: f32;
    var hh: f32;
    if (q < 3) {
      x = (U - f32(k0 + q - 1)) * L.pitch.x * su / aw;
      if (abs(x) >= 1.0) { continue; }
      hh = L.warp.z * crimpProfile(warpUnderDist(L, k0 + q - 1, V)).x + 0.5 * L.warp.y * sqrt(1.0 - x * x);
    } else {
      let m = m0 + select(0, q - 4, wefts == 3);
      x = (V - f32(m)) * L.pitch.y * sv / af;
      if (abs(x) >= 1.0) { continue; }
      hh = -L.weft.z * crimpProfile(weftOverDist(L, m, U)).x + 0.5 * L.weft.y * sqrt(1.0 - x * x);
    }
    hh *= sgn;
    if (hh > ha) {
      hb = ha;
      ib = ia;
      ha = hh;
      ia = q;
    } else if (hh > hb) {
      hb = hh;
      ib = q;
    }
  }
  var o: Hit2;
  o.a.covered = false;
  o.b.covered = false;
  o.wb = 0.0;
  if (ia < 0) { return o; }
  if (ia < 3) { o.a = warpHit(L, k0 + ia - 1, U, V, su, sv); } else { o.a = weftHit(L, m0 + select(0, ia - 4, wefts == 3), U, V, su, sv); }
  if (ib >= 0) {
    if (ib < 3) { o.b = warpHit(L, k0 + ib - 1, U, V, su, sv); } else { o.b = weftHit(L, m0 + select(0, ib - 4, wefts == 3), U, V, su, sv); }
    o.wb = 1.0 / (1.0 + exp((ha - hb) / tau));
  }
  return o;
}

// Failure site of bundled yarn q (strand st) of span s from its packed strand sites: the strand's
// site, staggered per yarn (the yarns of a strand part a little apart). Used by the tail pass.
fn yarnSiteOf(sites: u32, s: u32, st: u32, q: u32) -> f32 {
  let site = f32((sites >> (8u * st)) & 0xFFu) / 255.0;
  return clamp(site + 0.18 * (hashf(s * 7u + q * 131u + 3u) - 0.5), 0.1, 0.9);
}

// ---------------------------------------------------------------- appearance

// Occlusion of a point at normalized height hn (0 = bottom of the weave, 1 = top of the floats):
// the twill grooves and interlacings are deep and shaded by the floats around them.
fn weaveOcclusion(hn: f32) -> f32 {
  let t = hn * hn * (3.0 - 2.0 * hn);
  return mix(0.08, 1.0, t * t);
}

struct Look {
  albedo: vec3<f32>,
  // fraction of loose surface fibre (sheen strength)
  fuzz: f32,
  // cavity occlusion of the local weave position
  ao: f32,
};

// Ring-dyed indigo warp and undyed weft. `pix` is the pixel footprint in metres (filters fibre-scale
// detail to its mean), `wear` the abrasion field, `crown` the exposure of this point on its yarn.
fn yarnLook(L: LayerR, layerIdx: u32, fam: u32, yarn: i32, float_: i32, along: f32, x: f32, wear: f32, pix: f32) -> Look {
  var o: Look;
  let seedY = bitcast<u32>(yarn) * 2u + fam + 977u * layerIdx;
  let crown = sqrt(max(0.0, 1.0 - x * x));
  // fibre-scale streaks along the twisted fibres (twist angle from the yarn axis)
  let tw = select(L.weft.w, L.warp.w, fam == 1u);
  let width = select(L.weft.x, L.warp.x, fam == 1u);
  let across = 0.5 * width * x;
  let fiberCoord = (across * cos(tw) - along * sin(tw)) / 22e-6;
  let fiberVis = clamp(1.6 - pix / 40e-6, 0.0, 1.0);
  let fiberIdx = u32(i32(floor(fiberCoord)) + 1048576);
  let fiberRnd = hash2f(seedY * 31u + 7u, fiberIdx);
  if (fam == 1u) {
    // per-yarn dye depth, slubs along the yarn (mm to cm scale) and per-float variation
    let yarnDye = 0.92 + 0.16 * hash2f(seedY, 3u);
    let slub = 0.6 * noise1(seedY + 11u, along / 0.0035) + 0.4 * noise1(seedY + 23u, along / 0.014);
    let floatV = hash2f(seedY ^ 0x5bd1e995u, bitcast<u32>(float_));
    // sub-millimetre streaks along the yarn: twist irregularity exposes more or less undyed fibre
    let streak = 0.6 * noise1(seedY + 41u, along / 0.3e-3) + 0.4 * noise1(seedY + 53u, along / 0.85e-3);
    // exposure of the pale core: abrasion acts on the crowns of the floats
    let abr = clamp(wear * (0.35 + 0.65 * crown) * (0.6 + 0.8 * floatV) + 0.1 * floatV * crown, 0.0, 1.0);
    let dye = clamp(yarnDye * (1.0 + 0.12 * slub) * (1.0 - 0.75 * abr), 0.0, 1.5);
    // washed indigo: the dyed sheath is a greyish blue, the exposed core a cool off-white
    let deep = vec3<f32>(0.03, 0.05, 0.13);
    let pale = vec3<f32>(0.42, 0.45, 0.55);
    // fibre clumps and uneven ring dyeing mottle the surface: elongated along the warp (slub-like,
    // ~3 mm long, ~0.5 mm wide) plus fine isotropic clumps; faded below the pixel footprint
    let pc = vec2<f32>(f32(yarn) * L.pitch.x + across, along);
    let m1 = noise2(977u + layerIdx, pc / 0.4e-3) * clamp(1.5 - pix / 0.4e-3, 0.0, 1.0);
    let m2 = noise2(1913u + layerIdx, pc / vec2<f32>(0.55e-3, 3.0e-3)) * clamp(1.5 - pix / 0.55e-3, 0.0, 1.0);
    let mottle = 0.3 * m1 + 0.7 * m2;
    // mean fraction of undyed fibre showing on the surface; individual fibres alternate around it
    let paleMean = clamp(0.12 + 0.45 * (1.0 - min(dye, 1.0)) + 0.06 * streak + 0.1 * mottle, 0.0, 1.0);
    let paleFiber = select(0.0, 1.0, fiberRnd < paleMean);
    let paleFrac = mix(paleMean, paleFiber, fiberVis * 0.2);
    o.albedo = mix(deep * (0.75 + 0.5 * min(dye, 1.2)), pale, paleFrac);
    o.fuzz = 0.25 + 0.6 * wear;
  } else {
    let v = 0.85 + 0.2 * hash2f(seedY, 5u) + 0.08 * noise1(seedY + 3u, along / 0.004);
    let fib = mix(1.0, 0.85 + 0.3 * fiberRnd, fiberVis);
    // undyed weft picks up indigo from neighbouring warps in washing; it is greyish, not white
    o.albedo = vec3<f32>(0.26, 0.27, 0.28) * v * fib;
    o.fuzz = 0.35;
  }
  o.ao = 1.0;
  return o;
}

