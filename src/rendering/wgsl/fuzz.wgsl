// The free edges' fibre (render-only, attached to its material). Each open side of an edge crossing
// (pieces.wgsl edgeList, roots placed by edge_roots.wgsl) is where the yarn running across it ended.
// Of its fibres (its linear density over a fibre's) those that slipped out of the break instead of
// breaking at it protrude, FRAY_LEN on average; the yarns along the edge, which lost their crossings,
// shed hair over it. Per side the strands are:
//  - sheets of the rim yarns' hair, rooted behind the edge and lying out over it;
//  - the compact yarn end, up to where its twist opens;
//  - fans of the end's protruding fibre, splaying from there over the opening.
// A sheet's fibres are evaluated in the fragment shader (sheetCover): each leaves the root at its
// own place, diverges across the fan, crimps and curls, and ends at its own point, so the fringe has
// the coverage of the real fibre count, and in close views shows its separate fibres. Strands rise
// toward the camera in slot order and are drawn in that order (back to front within a crossing). They
// are lit per vertex: through the fringe's fibre between them and the key (single scattering) and by
// light diffusing through it (multiple scattering, damped by the fibre's absorption, so the depth of
// an indigo fringe darkens toward blue).

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> edgeList: array<u32>;
@group(0) @binding(3) var<storage, read> edgeRoots: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> fieldY: array<u32>;
@group(0) @binding(9) var shadowMap: texture_depth_2d;
@group(0) @binding(10) var shadowSampler: sampler_comparison;
@group(0) @binding(11) var overMap: texture_depth_2d;

const FIBER_PTS: u32 = FIBER_VERTS / 2u;
// strands per open side (FIBERS_PER_EDGE are dealt round a crossing's open sides)
const SIDE_STRANDS: u32 = 6u;
// mean projected width of a cotton fibre (a twisted ribbon, 12-20 um across)
const FIBRE_W: f32 = 16e-6;
// fraction of a broken staple yarn's fibres that slip out of the break rather than break across it:
// the twist presses the core fibres hardest, so the outer (sheath) fibres slip far more often
const SLIP_SHEATH: f32 = 0.8;
const SLIP_CORE: f32 = 0.3;
// fraction of a ring-dyed warp's fibre carrying indigo: its sheath (dye penetrates ~30% of the radius)
const RING_DYED: f32 = 0.5;
// the protruding fraction of a broken yarn's fibre, and the dyed share of a warp's protruding fibre
const FRAY_PROTRUDE: f32 = RING_DYED * SLIP_SHEATH + (1.0 - RING_DYED) * SLIP_CORE;
const FRAY_DYED: f32 = RING_DYED * SLIP_SHEATH / FRAY_PROTRUDE;
// mean length by which they protrude (cotton staple is ~25 mm, but abraded fibre is broken short and
// a loosened end curls back on itself)
const FRAY_LEN: f32 = 1.5e-3;
// hair shed by a loosened yarn along the edge, as fibre length per length of yarn (Uster H of an
// abraded, untwisted coarse yarn), the band of such yarns behind the edge, the hairs' mean length,
// and their dyed share on a warp (hair is its surface fibre: the dyed sheath)
const RIM_HAIR: f32 = 10.0;
const RIM_BAND: f32 = 1.5e-3;
const RIM_HAIR_LEN: f32 = 0.9e-3;
const RIM_DYED: f32 = 0.85;
// half-angle tangent of a fan: the protruding fibres diverge by ~20 degrees from the end's axis
const FAN_CONE: f32 = 0.35;
// fibres evaluated per sheet (a sheet of more gives each the coverage of several)
const SHEET_FIBRES: u32 = 48u;
// a loose fibre's lateral wander: cotton's convolutions (amplitude, range of wavelengths), and the
// gentler curl of its free length (amplitude per length free, range of wavelengths: loose cotton
// bends with radii of a millimetre or two)
const CRIMP_AMP: f32 = 0.04e-3;
const CRIMP_MIN: f32 = 0.3e-3;
const CRIMP_MAX: f32 = 1.2e-3;
const CURL: f32 = 0.15;
const CURL_MIN: f32 = 2.0e-3;
const CURL_MAX: f32 = 6.0e-3;
// single-scattering albedo of a cotton fibre, from the woven material's reflectances (yarnLook): a
// deep mass of fibre of albedo w reflects R = (1 - s) / (1 + s), s = sqrt(1 - w). Indigo-dyed (the
// dyed yarn surface, ~(0.054, 0.078, 0.16)), undyed warp core (0.6, 0.62, 0.66), loose weft
// (0.64, 0.61, 0.55).
const DYED_FIBRE: vec3<f32> = vec3<f32>(0.195, 0.27, 0.476);
const CORE_FIBRE: vec3<f32> = vec3<f32>(0.938, 0.945, 0.958);
const WEFT_FIBRE: vec3<f32> = vec3<f32>(0.952, 0.941, 0.916);

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  // coverage of a compact yarn end (its width over the drawn width)
  @location(2) alpha: f32,
  // kind (0 fan of an end's fibre, 1 yarn end, 2 rim hair), its undyed fibre (0 warp core, 2 weft),
  // layer, seed
  @location(3) @interpolate(flat) id: vec4<u32>,
  // 0 at the root .. 1 at the tip
  @location(4) along: f32,
  // light leaving the strand's undyed fibre (or the yarn end) and its dyed fibre, single scattered
  // (litU, litD) and multiply scattered with the fill (msU, msD), before the strand's own shadow
  // across its width (per vertex: light varies slowly along a strand; the fragment adds the shadow
  // and mixes them by what covers it)
  @location(5) litU: vec3<f32>,
  @location(8) litD: vec3<f32>,
  @location(9) msU: vec3<f32>,
  @location(10) msD: vec3<f32>,
  // the key's direction across the ribbon (its component along the ribbon's width)
  @location(11) keyAcross: f32,
  // across the drawn ribbon (m, signed), the sheet's half width here (m)
  @location(6) lat: vec2<f32>,
  // fibres in the sheet, root half width (m), length (m), fraction dyed
  @location(7) @interpolate(flat) sheet: vec4<f32>,
};

fn hidden() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

// Fibres of one family per yarn (LayerR.loose: z picks, w warps).
fn fibresPerYarn(L: LayerR, fam: u32) -> f32 { return select(L.loose.z, L.loose.w, fam == 1u); }

// Fraction of the end's protruding fibre that still reaches distance x (m) beyond the edge: their
// protrusion is exponential with mean FRAY_LEN. Behind the edge, little of it folds back.
fn endReach(x: f32) -> f32 {
  return select(exp(x / 0.3e-3), exp(-x / FRAY_LEN), x >= 0.0);
}

// Light leaving fibre of single-scattering albedo a at P (tangent T, toward the camera V), lit by the
// key through the fringe's optical depth tau toward it (the fringe is about as thick as it reaches,
// its strands rising at up to ~40 degrees, so the path to the light is its depth whatever the light's
// angle) and by the fill. Nc: the cloth normal. Multiple scattering: light diffusing through fibre of
// albedo a decays by sqrt(3 (1 - a)) per unit optical depth.
// Returns the single-scattered part and the multiply scattered part with the fill, apart, since the
// strand's own fibre attenuates them differently (fsFuzz).
struct FibreLit {
  single: vec3<f32>,
  multi: vec3<f32>,
};

fn diffusion(a: vec3<f32>) -> vec3<f32> { return sqrt(3.0 * (vec3<f32>(1.0) - a)); }

// The key at P (direction, irradiance with its shadow) for a fibre of tangent T seen from V.
fn fibreKey(P: vec3<f32>, T: vec3<f32>, V: vec3<f32>) -> vec4<f32> {
  let key = keyLight(P);
  return vec4<f32>(key.xyz, key.w * shadowFactor(P, normalize(V - dot(V, T) * T)));
}

// The overhead light at P (direction, its shadow) for a fibre of tangent T seen from V.
fn fibreOver(P: vec3<f32>, T: vec3<f32>, V: vec3<f32>) -> vec4<f32> {
  return vec4<f32>(F.overDir.xyz, shadowOver(P, normalize(V - dot(V, T) * T)));
}

// One light of direction Ld and irradiance E through optical depth tau.
fn fibreDirect(Ld: vec3<f32>, E: vec3<f32>, T: vec3<f32>, V: vec3<f32>, Nc: vec3<f32>, a: vec3<f32>, tau: f32) -> FibreLit {
  let N = normalize(V - dot(V, T) * T);
  let direct = exp(-tau);
  var o: FibreLit;
  o.single = fiberShade(N, T, V, Ld, a, 0.3) * direct * E;
  o.multi = a * (1.0 / PI) * abs(dot(Nc, Ld)) * max(exp(-diffusion(a) * tau) - vec3<f32>(direct), vec3<f32>(0.0)) * E;
  return o;
}

// The key (through tau), the overhead light (through tauO), the camera's light (through tauC, from the
// camera's side) and the fill.
fn fibreLight(key: vec4<f32>, over: vec4<f32>, cam: vec4<f32>, T: vec3<f32>, V: vec3<f32>, Nc: vec3<f32>, a: vec3<f32>, tau: f32, tauO: f32, tauC: f32) -> FibreLit {
  let k = fibreDirect(key.xyz, key.w * F.keyColor.rgb, T, V, Nc, a, tau);
  let o2 = fibreDirect(over.xyz, F.overColor.rgb * over.w, T, V, Nc, a, tauO);
  var o: FibreLit;
  o.single = k.single + o2.single;
  o.multi = k.multi + o2.multi + F.ambient.rgb * a * exp(-diffusion(a) * tau);
  if (cam.w > 0.0) {
    let c = fibreDirect(cam.xyz, cam.w * F.torchColor.rgb, T, V, Nc, a, tauC);
    o.single += c.single;
    o.multi += c.multi;
  }
  return o;
}

// What a strand is made of: fibres in it (0: a compact yarn end), its root width (m), how far its
// fibres diverge (half-angle tangent), its dyed fraction and its undyed fibre (0 warp core, 2 weft).
struct Make {
  n: f32,
  w0: f32,
  fan: f32,
  dyed: f32,
  undyed: u32,
};

// Vertex vi of a strand from `root` along unit `dir` over `len`, bent by `bend` (tip offset growing
// quadratically) and crimped by `wave` (amplitude of `waves` half periods). `x0` is the root's distance
// beyond the edge along `outW`, `tauEnd` and `tauRim` the optical depths (along the cloth normal N) of
// the end's fibre at the edge and of the rim's hair, and `h` the strand's height in the fringe (0
// lowest .. 1 toward the camera).
fn strand(root: vec3<f32>, dir: vec3<f32>, len: f32, bend: vec3<f32>, wave: vec3<f32>, waves: f32, m: Make, vi: u32, side: vec3<f32>, N: vec3<f32>, outW: vec3<f32>, x0: f32, tauEnd: f32, tauRim: f32, h: f32, id: vec4<u32>) -> VOut {
  let t = f32(vi >> 1u) / f32(FIBER_PTS - 1u);
  let ph = PI * waves * t;
  let P = root + dir * (len * t) + bend * (t * t) + wave * (sin(ph) * t);
  let dP = normalize(dir * len + bend * (2.0 * t) + wave * (sin(ph) + PI * waves * t * cos(ph)));
  let sideSign = select(-1.0, 1.0, (vi & 1u) == 1u);
  let V = normalize(F.camPos.xyz - P);
  var B = cross(dP, V);
  let bl = length(B);
  if (bl < 1e-6) { B = side; } else { B = B / bl; }
  let pixW = F.camPos.w * length(F.camPos.xyz - P);
  var o: VOut;
  var halfW: f32;
  let H = 0.5 * m.w0 + m.fan * len * t;
  if (m.n > 0.0) {
    // the fan, its fibres' wander and width, and a pixel's margin for the coverage filter
    halfW = H + 1.6 * (CRIMP_AMP + CURL * len * t) + FIBRE_W * max(1.0, m.n / f32(SHEET_FIBRES)) + 0.5 * pixW;
    o.alpha = 1.0;
  } else {
    halfW = max(0.5 * m.w0, 0.5 * pixW);
    o.alpha = clamp(m.w0 / max(pixW, 1e-9), 0.0, 1.0);
  }
  let W = P + sideSign * halfW * B;
  o.pos = F.viewProj * vec4<f32>(W, 1.0);
  o.world = W;
  o.id = id;
  o.along = t;
  o.lat = vec2<f32>(sideSign * halfW, H);
  o.sheet = vec4<f32>(m.n, 0.5 * m.w0, len, m.dyed);
  // the fringe toward the light: the strands above this one (h is its height toward the camera; the
  // key may light the fringe from the other side)
  let x = x0 + dot(P - root, outW);
  let up = select(h, 1.0 - h, dot(N, F.keyPos.xyz - P) > 0.0);
  let rim = select(1.0, exp(-x / RIM_HAIR_LEN), x > 0.0);
  let tau0 = tauEnd * endReach(x) + tauRim * rim;
  let tau = up * tau0;
  let tauO = select(h, 1.0 - h, dot(N, F.overDir.xyz) > 0.0) * tau0;
  let undyed = select(WEFT_FIBRE, CORE_FIBRE, m.undyed == 0u);
  o.keyAcross = dot(normalize(F.keyPos.xyz - P), B) * sideSign;
  let key = fibreKey(P, dP, V);
  let over = fibreOver(P, dP, V);
  // the camera's light reaches the fringe from the camera's side (h: height toward the camera)
  let cam = torchLight(P);
  let tauC = h * tau0;
  if (m.n > 0.0) {
    let u = fibreLight(key, over, cam, dP, V, N, undyed, tau, tauO, tauC);
    o.litU = u.single;
    o.msU = u.multi;
    if (m.dyed > 0.0) {
      let d = fibreLight(key, over, cam, dP, V, N, DYED_FIBRE, tau, tauO, tauC);
      o.litD = d.single;
      o.msD = d.multi;
    } else {
      o.litD = u.single;
      o.msD = u.multi;
    }
  } else {
    // the compact yarn end: a warp shows its dyed surface, paling as it untwists toward the fans
    let a = select(WEFT_FIBRE, mix(vec3<f32>(0.054, 0.078, 0.16), mix(DYED_FIBRE, CORE_FIBRE, 0.5), t * t), m.undyed == 0u);
    let e = fibreLight(key, over, cam, dP, V, N, a, tau, tauO, tauC);
    o.litU = e.single + e.multi;
  }
  return o;
}

@vertex
fn vsEdgeFuzz(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let ei = ii / FIBERS_PER_EDGE;
  let fi = ii % FIBERS_PER_EDGE;
  let root0 = edgeRoots[3u * ei];
  if (root0.w < 0.5) { return hidden(); }
  let entry = edgeList[ei];
  let c = entry & 0x0FFFFFFFu;
  let sides = entry >> 28u;
  // the slot's open side (slots are dealt round them in turn) and its place among the side's strands
  let nOpen = countOneBits(sides);
  let slots = min(SIDE_STRANDS, FIBERS_PER_EDGE / nOpen);
  let j = fi / nOpen;
  if (j >= slots) { return hidden(); }
  var pick = fi % nOpen;
  var side = 0u;
  for (var b = 0u; b < 4u; b++) {
    if (((sides >> b) & 1u) == 0u) { continue; }
    if (pick == 0u) {
      side = b;
      break;
    }
    pick--;
  }
  var l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (c >= R.layers[q].field.x) { l = q; }
  }
  let LR = R.layers[l];
  // how much of the yarns loosened and slipped here: yarns break at scattered places along their
  // length and loosen unevenly, so tufted stretches of edge lie next to nearly bare ones (noise over
  // the rest chart, a few crossings to several millimetres; mean 1)
  let FL = LR.field;
  let restC = vec2<f32>(f32((c - FL.x) % FL.y) * LR.pitch.x, f32((c - FL.x) / FL.y) * LR.pitch.y);
  let tuftN = 0.6 * noise2(6007u + l, restC / 5.0e-3) + 0.4 * noise2(6011u + l, restC / 1.6e-3);
  let density = 0.3 + 1.4 * smoothstep(-0.45, 0.5, tuftN);
  let dU = edgeRoots[3u * ei + 1u].xyz;
  let dV = edgeRoots[3u * ei + 2u].xyz;
  var N = normalize(cross(dU, dV));
  if (dot(N, F.camPos.xyz - root0.xyz) < 0.0) { N = -N; }
  let sg = select(-1.0, 1.0, (side & 1u) == 1u);
  // in the rest chart: out of the side, and along it
  let outR = select(vec2<f32>(0.0, sg), vec2<f32>(sg, 0.0), side < 2u);
  let alongR = vec2<f32>(-outR.y, outR.x);
  let outW = normalize(dU * outR.x + dV * outR.y);
  let acrossW = normalize(dU * alongR.x + dV * alongR.y);
  let pitchOut = select(LR.pitch.y, LR.pitch.x, side < 2u);
  // the yarn running across the side ends here (picks across the u sides, warps across the v sides);
  // the yarns along it are the loosened rim
  let fam = select(1u, 0u, side < 2u);
  let famAlong = 1u - fam;
  // side length along the edge: the spacing of the yarns that end across it
  let sideLen = select(LR.pitch.y, LR.pitch.x, fam == 1u);
  let endHere = yarnCode(fieldY[c], fam) < PULLED_R;
  // fibre of the side: the protruding end (absent when the yarn was pulled out of this crossing) and
  // the hair of the band of rim yarns along the edge (RIM_BAND / pitchOut of them, each shedding
  // RIM_HAIR of fibre length per length in hairs RIM_HAIR_LEN long)
  let endFibres = select(0.0, FRAY_PROTRUDE * fibresPerYarn(LR, fam) * density, endHere);
  let rimHair = RIM_HAIR * sideLen * (RIM_BAND / pitchOut) / RIM_HAIR_LEN * density;
  // optical depths of the end's fibre at the edge and of the rim's hair, seen along the cloth normal
  let tauEnd = endFibres * FIBRE_W / sideLen;
  let tauRim = rimHair * FIBRE_W / sideLen;
  // the side's strands, lowest first: rim hair, the yarn end, fans
  let nRim = select(1u, 2u, slots >= 5u);
  let nFan = slots - 1u - nRim;
  let h = (f32(j) + 0.5) / f32(slots);
  let seed = hash11(c * 2654435761u + side * 97u + j * 7919u + 11u);
  let cv = vec3<f32>(hashf(seed + 8u) - 0.5, hashf(seed + 9u) - 0.5, hashf(seed + 10u) - 0.5);
  let wv = vec3<f32>(hashf(seed + 13u) - 0.5, hashf(seed + 14u) - 0.5, hashf(seed + 15u) - 0.5);
  let atSide = root0.xyz + (dU * outR.x + dV * outR.y) * (0.5 - 0.15 * hashf(seed + 2u)) + (dU * alongR.x + dV * alongR.y) * (hashf(seed + 1u) - 0.5);

  if (j < nRim) {
    // hair of the loosened rim yarns: rooted up to the band's depth behind the edge, drawn out over it
    var m: Make;
    m.n = rimHair / f32(nRim);
    if (m.n < 0.05) { return hidden(); }
    m.w0 = 0.2e-3;
    m.fan = FAN_CONE;
    m.dyed = select(0.0, RIM_DYED, famAlong == 1u);
    m.undyed = select(2u, 0u, famAlong == 1u);
    let back = (0.2e-3 + RIM_BAND * pow(hashf(seed + 17u), 1.5)) / pitchOut;
    let root = atSide - (dU * outR.x + dV * outR.y) * back;
    let spread = (hashf(seed + 3u) - 0.5) * 1.2;
    let dir = normalize(outW + acrossW * spread + N * (0.03 + 0.12 * h));
    let len = back * pitchOut + clamp(-2.0 * RIM_HAIR_LEN * log(1.0 - 0.95 * hashf(seed + 5u)), 0.2e-3, 4.0e-3);
    let bend = cv * (0.7 * len) - N * (0.1 * len);
    return strand(root + N * 0.06e-3, dir, len, bend, wv * (0.4 * len), 2.0 + 4.0 * hashf(seed + 16u), m, vi, acrossW, N, outW, -back * pitchOut, tauEnd, tauRim, h, vec4<u32>(2u, m.undyed, l, seed));
  }
  if (!endHere) { return hidden(); }
  // the yarn end standing out of the edge, untwisting from part-way along it; its fans splay from
  // there. All derive from one seed per side.
  let sb = hash11(c * 2654435761u + side * 13u + 7u);
  let atEnd = root0.xyz + (dU * outR.x + dV * outR.y) * (0.5 - 0.15 * hashf(sb + 2u)) + (dU * alongR.x + dV * alongR.y) * (hashf(sb + 1u) - 0.5);
  let liftB = 0.1 + 0.45 * hashf(sb + 4u);
  let dirB = normalize(outW + acrossW * ((hashf(sb + 3u) - 0.5) * 0.8) + N * liftB);
  let lenB = clamp(-2.0 * FRAY_LEN * log(1.0 - 0.98 * hashf(sb + 5u)), 0.6e-3, 8.0 * FRAY_LEN);
  let droopB = vec3<f32>(0.0, 0.0, -0.3 * lenB) - N * (0.25 * lenB * liftB);
  let curlB = (acrossW * (hashf(sb + 6u) - 0.5) + N * (hashf(sb + 7u) - 0.5)) * (0.45 * lenB);
  // where the twist opens up: the core runs as a yarn up to t0 of lenB
  let t0 = 0.15 + 0.3 * hashf(sb + 9u);
  let yarnD = 2.0 * select(LR.loose.y, LR.loose.x, fam == 0u);
  var m: Make;
  m.undyed = select(2u, 0u, fam == 1u);
  if (j == nRim) {
    m.n = 0.0;
    m.w0 = yarnD;
    m.fan = 0.0;
    m.dyed = 0.0;
    return strand(atEnd, dirB, lenB * t0, (droopB + curlB) * (t0 * t0), curlB * 0.2, 1.0, m, vi, acrossW, N, outW, 0.0, tauEnd, tauRim, h, vec4<u32>(1u, m.undyed, l, sb));
  }
  // a fan of the end's protruding fibre (its share of them; a warp's mostly from the dyed sheath, a
  // weft's undyed throughout), from where the twist opened, splaying and drooping over the opening;
  // fans rise from the end's own direction toward the camera
  m.n = endFibres / f32(nFan);
  m.w0 = yarnD;
  m.fan = FAN_CONE;
  m.dyed = select(0.0, FRAY_DYED, fam == 1u);
  let tip = atEnd + dirB * (lenB * t0) + (droopB + curlB) * (t0 * t0);
  let tangent = normalize(dirB * lenB + (droopB + curlB) * (2.0 * t0));
  let k = f32(j - nRim - 1u) / max(f32(nFan) - 1.0, 1.0);
  let fs = vec3<f32>(hashf(seed + 3u), hashf(seed + 4u), hashf(seed + 5u)) - 0.5;
  let dir = normalize(tangent + acrossW * (0.9 * fs.x) + N * (0.1 + 0.6 * k + 0.3 * fs.y));
  // fan lengths: exponential with mean 2 FRAY_LEN, so that its fibres, ending evenly along it,
  // protrude FRAY_LEN on average beyond the end (cut off where a fibre's abraded length runs out)
  let len = clamp(-2.0 * FRAY_LEN * log(1.0 - 0.98 * hashf(seed + 6u)), 0.4e-3, 4.0 * FRAY_LEN);
  let droop = clamp(len / 1.5e-3, 0.0, 1.0);
  let bend = cv * (0.8 * len) - N * (0.4 * len * droop) + vec3<f32>(0.0, 0.0, -0.3 * len * droop);
  return strand(tip, dir, len, bend, wv * (0.3 * len), 1.5 + 2.5 * hashf(seed + 16u), m, vi, acrossW, N, outW, lenB * t0 * dot(dirB, outW), tauEnd, tauRim, h, vec4<u32>(0u, m.undyed, l, seed));
}

// A smooth wave of period 1 between -1 and 1 (a parabola per half period: within 0.06 of a sine).
fn wave(x: f32) -> f32 {
  let f = fract(x) * 2.0 - 1.0;
  return -4.0 * f * (1.0 - abs(f));
}

// Coverage of a fibre sheet at lateral position y (m) across it and fraction t along it, over a pixel
// pix (m) wide, with the dyed share of what covers. Each fibre leaves the root at its own place (within
// r0 of the centre), diverges across the fan (to within H), crimps and curls, and ends at its own t,
// so the sheet thins evenly toward its tip. Where neighbouring fibres lie further apart than a pixel
// they are evaluated one by one (however faint, a few separate fibres over the dark opening are
// streaks, not a veil);
// where many share a pixel only their mean coverage is visible, evaluated in closed form (their
// distribution across the sheet, Poisson overlap), blended across the range where they begin to
// resolve.
fn sheetCover(seed: u32, n: f32, r0: f32, H: f32, len: f32, dyedFrac: f32, y: f32, t: f32, pix: f32, overCloth: bool) -> vec2<f32> {
  // past SHEET_FIBRES, each evaluated fibre stands for several side by side
  let d = FIBRE_W * max(1.0, n / f32(SHEET_FIBRES));
  let wander = CRIMP_AMP + CURL * len * t;
  let half = mix(r0, H, t) + wander;
  let alive = n * (1.0 - t);
  let spacing = 2.0 * half / max(alive, 1.0);
  // (hair lying over the pale cloth has little contrast there until it covers a good share of a pixel)
  let resolve = smoothstep(0.5, 1.0, spacing / pix) * select(1.0, smoothstep(0.06, 0.12, d / pix), overCloth);
  var mean = 0.0;
  if (resolve < 1.0) {
    // a fibre lies at (1 - t) a r0 + t b H, a and b uniform over -1 .. 1: the sum of two uniform
    // spreads, flat out to |A - B| and falling linearly to zero at A + B (crimp widens both)
    let A = (1.0 - t) * r0 + 0.5 * wander;
    let Bs = t * H + 0.5 * wander;
    let lo = abs(A - Bs);
    let hi = A + Bs;
    let density = clamp((hi - abs(y)) / max(hi - lo, 1e-9), 0.0, 1.0) / (2.0 * max(A, Bs));
    mean = 1.0 - exp(-alive * FIBRE_W * density);
  }
  if (resolve <= 0.0 || abs(y) > half + 0.6 * wander + 0.5 * pix + d) { return vec2<f32>(mean, dyedFrac); }
  // ends stratified along the sheet (fibre k ends in the k-th of count equal steps from the tip), so
  // only the fibres still present at t are visited
  let count = min(u32(ceil(n)), SHEET_FIBRES);
  let present = min(count, u32(f32(count) * (1.0 - t)) + 1u);
  var clear = 1.0;
  var all = 0.0;
  var dyed = 0.0;
  for (var k = 0u; k < present; k++) {
    // one hash per fibre for its end, root, direction and phase, remixed for the rest
    let h0 = hash11(seed * 7919u + k * 104729u + 1u);
    let h1 = (h0 ^ (h0 >> 15u)) * 0x2c1b3c6du;
    let h2 = (h1 ^ (h1 >> 13u)) * 0x297a2d39u;
    let r = vec4<f32>(vec4<u32>(h0 & 255u, (h0 >> 8u) & 255u, (h0 >> 16u) & 255u, h0 >> 24u)) * (1.0 / 255.0);
    if (t > 1.0 - (f32(k) + r.x) / f32(count)) { continue; }
    let q = vec4<f32>(vec4<u32>(h1 & 255u, (h1 >> 8u) & 255u, (h1 >> 16u) & 255u, h1 >> 24u)) * (1.0 / 255.0);
    // the convolutions and the free length's curl
    let along = t * len;
    let ph = along / mix(CRIMP_MIN, CRIMP_MAX, q.x) + r.w;
    let p = vec4<f32>(vec4<u32>(h2 & 255u, (h2 >> 8u) & 255u, (h2 >> 16u) & 255u, h2 >> 24u)) * (1.0 / 255.0);
    let crimp = CRIMP_AMP * (0.4 + 1.2 * q.y) * smoothstep(0.0, 0.2, t) * wave(ph);
    let curl = CURL * along * (0.3 + 1.4 * p.x) * wave(along / mix(CURL_MIN, CURL_MAX, q.w) + p.y);
    let yk = mix((2.0 * r.y - 1.0) * r0, (2.0 * r.z - 1.0) * H, t) + crimp + curl;
    let dk = d * (0.75 + 0.5 * q.z);
    let ov = clamp((min(y + 0.5 * pix, yk + 0.5 * dk) - max(y - 0.5 * pix, yk - 0.5 * dk)) / pix, 0.0, 1.0);
    let w = ov * clear;
    all += w;
    dyed += select(0.0, w, p.z < dyedFrac);
    clear *= 1.0 - ov;
  }
  let each = vec2<f32>(1.0 - clear, select(dyedFrac, dyed / all, all > 1e-6));
  return mix(vec2<f32>(mean, dyedFrac), each, resolve);
}

@fragment
fn fsFuzz(in: VOut) -> @location(0) vec4<f32> {
  if (in.id.x == 1u) { return vec4<f32>(in.litU, in.alpha); }
  let pix = F.camPos.w * length(F.camPos.xyz - in.world);
  let cov = sheetCover(in.id.w, in.sheet.x, in.sheet.y, in.lat.y, in.sheet.z, in.sheet.w, in.lat.x, in.along, pix, in.id.x == 2u);
  if (cov.x < 0.002) { discard; }
  // the strand shadows itself across its width: a tuft is lit on the side toward the key and dark on
  // the far side, through the optical depth its own fibre presents across it (as for its coverage)
  let half = mix(in.sheet.y, in.lat.y, in.along) + CRIMP_AMP + CURL * in.sheet.z * in.along;
  let across = in.sheet.x * (1.0 - in.along) * FIBRE_W / (2.0 * half);
  let s = clamp(0.5 - 0.5 * sign(in.keyAcross) * in.lat.x / half, 0.0, 1.0);
  let tauX = across * mix(0.5, s, abs(in.keyAcross));
  let undyed = select(WEFT_FIBRE, CORE_FIBRE, in.id.y == 0u);
  let cu = in.litU * exp(-tauX) + in.msU * exp(-diffusion(undyed) * tauX);
  let cd = in.litD * exp(-tauX) + in.msD * exp(-diffusion(DYED_FIBRE) * tauX);
  return vec4<f32>(mix(cu, cd, cov.y), cov.x);
}
