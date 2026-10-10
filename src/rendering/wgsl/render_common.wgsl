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
  // edge field: first texel row, unused, texture width, texture height (texels)
  edge: vec4<f32>,
  // yarn damage field: first crossing, warps K, picks M, unused
  field: vec4<u32>,
  // loose (freed, round) yarn radius of a pick and of a warp (m), unused, unused
  loose: vec4<f32>,
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

// ---------------------------------------------------------------- lattice cells at yarn resolution

// Crossings of a cell's grid at most (columns x rows); cells near damage are drawn per crossing.
const CELL_MAX_K: u32 = 16u;
const CELL_MAX_M: u32 = 8u;

// Crossing grid of cell (ci, cj): first warp and pick (fine indices) and the counts inside the cell
// (crossings on its edge lines included).
fn cellGrid(L: LayerR, ci: u32, cj: u32) -> vec4<u32> {
  let bu = f32(bundleOf(L, 1u));
  let bv = f32(bundleOf(L, 0u));
  let ua = f32(ci) * bu + 0.5 * (bu - 1.0);
  let va = f32(cj) * bv + 0.5 * (bv - 1.0);
  let k0 = u32(ceil(ua - 1e-3));
  let m0 = u32(ceil(va - 1e-3));
  let nk = min(u32(floor(ua + bu + 1e-3)) - k0 + 1u, CELL_MAX_K);
  let nm = min(u32(floor(va + bv + 1e-3)) - m0 + 1u, CELL_MAX_M);
  return vec4<u32>(k0, m0, nk, nm);
}

// ---------------------------------------------------------------- yarn damage field and edge field
//
// The yarn damage field (simulation/yarnField.ts) holds per crossing the damage code of its pick (bits
// 0-7) and warp (bits 8-15) segment, 255 = cut, 254 = pulled out of the weave (yarnField.ts). The edge field is a texture at crossing
// resolution over each layer's field (layers stacked vertically, LayerR.edge): r = distance (mm) to the
// nearest opening, a crossing where neither yarn is part of the cloth (EDGE_RMAX beyond), g / b =
// intact fraction of the warp / pick there (0 for a yarn drawn as a loose thread), a = density of separated yarn segments around it (how much of an edge it is). edgefield.wgsl rebuilds it whenever new yarns separated. The torn rim, its fringe and the
// abraded halo are material features at a distance from the free edge, so they follow it smoothly;
// g and b let a weave below the pixel footprint still lose its worn-through yarns.

// Loose threads (threads.wgsl): runs of at least MIN_RUN lone crossings, drawn in chunks of CHUNK
// crossings (THREAD_VERTS: two per centre-line sample, two samples per crossing) with
// FIBERS_PER_CHUNK fibres each (FIBER_VERTS vertices).
const MIN_RUN: u32 = 4u;
const CHUNK: u32 = 8u;
const THREAD_VERTS: u32 = 34u;
const FIBERS_PER_CHUNK: u32 = 12u;
const FIBER_VERTS: u32 = 14u;
// vec4 per run of the per-frame thread geometry (thread_prep.wgsl)
const RUN_GEO: u32 = 7u;

// The material at crossing c belongs to a falling piece: bit c of the debris region of the thread bits
// (threads.wgsl), which starts after the 2-bit-per-crossing region of fieldTotal crossings.
fn debrisWord(c: u32, fieldTotal: u32) -> u32 { return (fieldTotal + 15u) / 16u + (c >> 5u); }

// Crossings of all layers' yarn fields, from the last layer.
fn fieldTotalOf(last: LayerR) -> u32 { return last.field.x + last.field.y * last.field.z; }

// Largest stored distance (mm); everything beyond is intact cloth.
const EDGE_RMAX: f32 = 10.0;
const CUT_R: u32 = 255u;
// pulled out of the weave at the crossing: the end has slid back toward where the yarn is held, so
// the crossing no longer has this yarn (drawn like a cut, yarnField.ts DAMAGE_PULLED)
const PULLED_R: u32 = 254u;

fn fieldIndexR(L: LayerR, k: u32, m: u32) -> u32 { return L.field.x + m * L.field.y + k; }
fn yarnCode(word: u32, fam: u32) -> u32 { return (word >> (8u * fam)) & 0xFFu; }

// Normalised edge-field texture coordinates of fine yarn coordinates (U, V) of layer L.
fn edgeUV(L: LayerR, uv: vec2<f32>) -> vec2<f32> {
  return vec2<f32>((uv.x + 0.5) / L.edge.z, (uv.y + 0.5 + L.edge.x) / L.edge.w);
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
// detail to its mean), `wear` the abrasion field, `crown` the exposure of this point on its yarn,
// `loose` 1 for a yarn freed of the crossing yarns (a thread, a pick across a ladder): no longer
// compacted under the warp floats, it shows its own fibre all round, untwisted and fluffed.
fn yarnLook(L: LayerR, layerIdx: u32, fam: u32, yarn: i32, float_: i32, along: f32, x: f32, wear: f32, pix: f32, loose: f32) -> Look {
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
    let yarnDye = 0.95 + 0.1 * hash2f(seedY, 3u);
    let slub = 0.6 * noise1(seedY + 11u, along / 0.0035) + 0.4 * noise1(seedY + 23u, along / 0.014);
    let floatV = hash2f(seedY ^ 0x5bd1e995u, bitcast<u32>(float_));
    // sub-millimetre streaks along the yarn: twist irregularity exposes more or less undyed fibre
    let streak = 0.6 * noise1(seedY + 41u, along / 0.3e-3) + 0.4 * noise1(seedY + 53u, along / 0.85e-3);
    // exposure of the pale core: abrasion acts on the crowns of the floats
    let abr = clamp(wear * (0.35 + 0.65 * crown) * (0.6 + 0.8 * floatV) + 0.1 * floatV * crown, 0.0, 1.0);
    let dye = clamp(yarnDye * (1.0 + 0.12 * slub) * (1.0 - 0.75 * abr), 0.0, 1.5);
    // washed indigo: the dyed sheath is a deep blue, abraded fibre a light blue with the off-white
    // core showing through
    let deep = vec3<f32>(0.03, 0.05, 0.13);
    let pale = vec3<f32>(0.36, 0.44, 0.6);
    // fibre clumps and uneven ring dyeing mottle the surface: elongated along the warp (slub-like,
    // ~3 mm long, ~0.5 mm wide) plus fine isotropic clumps; faded below the pixel footprint
    let pc = vec2<f32>(f32(yarn) * L.pitch.x + across, along);
    let m1 = noise2(977u + layerIdx, pc / 0.4e-3) * clamp(1.5 - pix / 0.4e-3, 0.0, 1.0);
    let m2 = noise2(1913u + layerIdx, pc / vec2<f32>(0.55e-3, 3.0e-3)) * clamp(1.5 - pix / 0.55e-3, 0.0, 1.0);
    let mottle = 0.55 * m1 + 0.3 * m2;
    // mean fraction of undyed fibre showing on the surface; individual fibres alternate around it
    let paleMean = clamp(0.12 + 0.45 * (1.0 - min(dye, 1.0)) + 0.06 * streak + 0.1 * mottle, 0.0, 1.0);
    let paleFiber = select(0.0, 1.0, fiberRnd < paleMean);
    let paleFrac = mix(paleMean, paleFiber, fiberVis * 0.2);
    // floats differ from one another (twist, hairiness, how far they stand out): the face of the
    // twill reads as a pebbly run of separate floats along each wale, not a printed stripe
    let floatB = 0.78 + 0.44 * floatV;
    o.albedo = mix(deep * (0.75 + 0.5 * min(dye, 1.2)), pale, paleFrac) * floatB;
    // a freed warp untwists: its pale core shows between the dyed surface fibres
    o.albedo = mix(o.albedo, mix(pale, vec3<f32>(0.6, 0.62, 0.66), 0.5), 0.45 * loose);
    o.fuzz = 0.25 + 0.6 * wear + 0.3 * loose;
  } else {
    let v = 0.85 + 0.2 * hash2f(seedY, 5u) + 0.08 * noise1(seedY + 3u, along / 0.004);
    let fib = mix(1.0, 0.85 + 0.3 * fiberRnd, fiberVis);
    // undyed weft picks up indigo from neighbouring warps in washing; it is greyish while compacted
    // under the floats, until abrasion and loosening expose its raw, fluffy ecru cotton
    let woven = mix(vec3<f32>(0.26, 0.27, 0.28), vec3<f32>(0.5, 0.5, 0.49), 0.8 * wear * wear);
    o.albedo = mix(woven, vec3<f32>(0.64, 0.61, 0.55), loose) * v * fib;
    o.fuzz = 0.35 + 0.35 * loose;
  }
  o.ao = 1.0;
  return o;
}

