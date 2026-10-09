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

// One corner's share of a torn cell (a cell with a separated span): the corner node owns the quarter
// of the cell its own warp and pick bundles occupy. Its yarns end at their failure sites when their
// span (or strand) has separated, and authored worn-through strands are absent.
struct TornInfo {
  // 0 for surface cells; else 1 + corner (a | b << 1: a = 1 for the right column, b = 1 for the top row)
  mode: u32,
  // bits 0-3: strands of the corner's warp span separated, 4-7: absent; 8-11 / 12-15: same for its pick
  // span; bit 16: the warp span separated entirely, bit 17: the pick span
  flags: u32,
  // packed per-strand failure sites (u8, fraction from the span's owning node) of the warp / pick span
  sitesV: u32,
  sitesU: u32,
  // span indices (2 * owner + family) of the corner's warp and pick spans
  spanV: u32,
  spanU: u32,
};

fn noTorn() -> TornInfo { return TornInfo(0u, 0u, 0u, 0u, 0u, 0u); }

// Failure site of bundled yarn q (strand st) of span s from its packed strand sites: the strand's
// site, staggered per yarn (the yarns of a strand part a little apart). Shared with the tail pass.
fn yarnSiteOf(sites: u32, s: u32, st: u32, q: u32) -> f32 {
  let site = f32((sites >> (8u * st)) & 0xFFu) / 255.0;
  return clamp(site + 0.18 * (hashf(s * 7u + q * 131u + 3u) - 0.5), 0.1, 0.9);
}

// Yarns missing from one lattice cell (strands absent or separated), from the strand masks of its
// four spans. Fine warp k sits at U = k; the cell's left warp bundle holds warps warpBase ..
// warpBase + bu - 1, its right bundle the next bu; picks likewise from pickBase (bottom, then top).
struct CellYarns {
  warpBase: i32,
  pickBase: i32,
  bu: i32,
  bv: i32,
  // yarns per strand: warp, pick
  su: i32,
  sv: i32,
  // strand bits 0-3 left warp bundle, 4-7 right, 8-11 bottom pick bundle, 12-15 top
  masks: u32,
  tr: TornInfo,
};

fn bundleYarnGone(base: i32, b: i32, ys: i32, k: i32, masks: u32, shift: u32) -> bool {
  let d = k - base;
  if (d < 0 || d >= 2 * b) { return false; }
  let side = select(0, 1, d >= b);
  let bit = u32((d - side * b) / ys + 4 * side);
  return ((masks >> (shift + bit)) & 1u) == 1u;
}
fn warpGone(c: CellYarns, k: i32) -> bool { return bundleYarnGone(c.warpBase, c.bu, c.su, k, c.masks, 0u); }
fn pickGone(c: CellYarns, m: i32) -> bool { return bundleYarnGone(c.pickBase, c.bv, c.sv, m, c.masks, 8u); }

// A torn corner's yarn of one family at bundle offset d from the cell's first bundle, at parameter p
// along the yarn's span (0 at the span's owning node): present only if it belongs to the corner's own
// bundle (side) and, when its strand or span has separated, only up to its failure site less a short
// recoil (the broken ends spring back and untwist).
fn tornGone(tr: TornInfo, fam: u32, d: i32, b: i32, ys: i32, side: u32, p: f32) -> bool {
  if (d < 0 || d >= 2 * b) { return true; }
  let own = select(0u, 1u, d >= b);
  if (own != side) { return true; }
  let q = d - i32(own) * b;
  let st = u32(q / ys);
  let sh = select(8u, 0u, fam == 1u);
  if (((tr.flags >> (sh + 4u + st)) & 1u) == 1u) { return true; }
  let spanBroken = ((tr.flags >> select(17u, 16u, fam == 1u)) & 1u) == 1u;
  if (!spanBroken && ((tr.flags >> (sh + st)) & 1u) == 0u) { return false; }
  let span = select(tr.spanU, tr.spanV, fam == 1u);
  let site = yarnSiteOf(select(tr.sitesU, tr.sitesV, fam == 1u), span, st, u32(q));
  let recoil = 0.02 + 0.05 * hashf(span * 13u + u32(q) * 29u + 1u);
  // the corner sits at the span's owning end (p = 0) or at its far end (p = 1)
  let atOwner = select(((tr.mode - 1u) >> 1u) == 0u, ((tr.mode - 1u) & 1u) == 0u, fam == 0u);
  return select((p < site + recoil), (p > site - recoil), atOwner);
}

// Warp k at pick coordinate V (and pick m at warp coordinate U) is missing from this cell or corner.
// At a torn edge a yarn is also lost where none of the nearest crossing yarns still reaches it: with
// its crossings gone it slips out, so the edge frays back irregularly pick by pick (warp by warp)
// instead of ending on the lattice line.
fn warpGoneAt(c: CellYarns, k: i32, V: f32) -> bool {
  if (c.tr.mode == 0u) { return warpGone(c, k); }
  let a = (c.tr.mode - 1u) & 1u;
  let b = (c.tr.mode - 1u) >> 1u;
  let t = (V - (f32(c.pickBase) + 0.5 * f32(c.bv - 1))) / f32(c.bv);
  if (tornGone(c.tr, 1u, k - c.warpBase, c.bu, c.su, a, t)) { return true; }
  if (((c.tr.flags >> 17u) & 1u) == 0u) { return false; }
  let s = (f32(k) - (f32(c.warpBase) + 0.5 * f32(c.bu - 1))) / f32(c.bu);
  let m0 = i32(round(V));
  for (var dm = -1; dm <= 1; dm++) {
    if (!tornGone(c.tr, 0u, m0 + dm - c.pickBase, c.bv, c.sv, b, s)) { return false; }
  }
  return true;
}
fn pickGoneAt(c: CellYarns, m: i32, U: f32) -> bool {
  if (c.tr.mode == 0u) { return pickGone(c, m); }
  let a = (c.tr.mode - 1u) & 1u;
  let b = (c.tr.mode - 1u) >> 1u;
  let s = (U - (f32(c.warpBase) + 0.5 * f32(c.bu - 1))) / f32(c.bu);
  if (tornGone(c.tr, 0u, m - c.pickBase, c.bv, c.sv, b, s)) { return true; }
  if (((c.tr.flags >> 16u) & 1u) == 0u) { return false; }
  let t = (f32(m) - (f32(c.pickBase) + 0.5 * f32(c.bv - 1))) / f32(c.bv);
  let k0 = i32(round(U));
  for (var dk = -1; dk <= 1; dk++) {
    if (!tornGone(c.tr, 1u, k0 + dk - c.warpBase, c.bu, c.su, a, t)) { return false; }
  }
  return true;
}

// The two highest yarns at a point and the soft visibility weight of the second: real yarn surfaces
// are hairy, so the boundary between overlapping yarns is a soft transition over ~tau of height.
struct Hit2 {
  a: Hit,
  b: Hit,
  wb: f32,
};

fn weave2(L: LayerR, U: f32, V: f32, su: f32, sv: f32, front: bool, tau: f32, cy: CellYarns) -> Hit2 {
  // heights only for every candidate; full hits for the two highest
  let sgn = select(-1.0, 1.0, front);
  let k0 = i32(round(U));
  let m0 = i32(round(V));
  let aw = 0.5 * L.warp.x;
  let af = 0.5 * L.weft.x;
  // warps overlap their neighbours (width > pitch): three candidates; picks only when stretched apart
  let wefts = select(1, 3, af > 0.5 * L.pitch.y * sv || (cy.masks & 0xFF00u) != 0u || cy.tr.mode != 0u);
  var ha = -1.0;
  var hb = -1.0;
  var ia = -1;
  var ib = -1;
  for (var q = 0; q < 3 + wefts; q++) {
    var x: f32;
    var hh: f32;
    if (q < 3) {
      x = (U - f32(k0 + q - 1)) * L.pitch.x * su / aw;
      if (abs(x) >= 1.0 || warpGoneAt(cy, k0 + q - 1, V)) { continue; }
      hh = L.warp.z * crimpProfile(warpUnderDist(L, k0 + q - 1, V)).x + 0.5 * L.warp.y * sqrt(1.0 - x * x);
    } else {
      let m = m0 + select(0, q - 4, wefts == 3);
      x = (V - f32(m)) * L.pitch.y * sv / af;
      if (abs(x) >= 1.0 || pickGoneAt(cy, m, U)) { continue; }
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

