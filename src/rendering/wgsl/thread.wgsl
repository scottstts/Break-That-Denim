// Loose threads (threads.wgsl runs, thread_geom.wgsl geometry): each chunk of a run is a ribbon
// facing the camera around the yarn's centre line, shaded as a round fibrous cylinder with the same
// material identity as the woven surface; fibres stand out of it and splay from untwisting free ends.
// Threads far below a pixel are drawn at their true coverage (alpha to coverage on 4x MSAA), never
// as solid one-pixel lines.

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> segw: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> nrm: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> runs: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> chunks: array<u32>;
// per run, RUN_GEO vec4 (thread_prep.wgsl)
@group(0) @binding(8) var<storage, read> runGeo: array<vec4<f32>>;
@group(0) @binding(9) var shadowMap: texture_depth_2d;
@group(0) @binding(10) var shadowSampler: sampler_comparison;
// free tails of threads held at one end (thread_prep.wgsl tailSolve)
@group(0) @binding(11) var<storage, read> tails: array<vec4<f32>>;
@group(0) @binding(12) var overMap: texture_depth_2d;

const CHUNK_PTS: u32 = THREAD_VERTS / 2u;
const FIBER_PTS: u32 = FIBER_VERTS / 2u;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) side: f32,
  @location(2) tangent: vec3<f32>,
  @location(3) binormal: vec3<f32>,
  // parameter along the yarn (crossings)
  @location(4) c: f32,
  // yarn (fine index), family, layer, unused
  @location(5) @interpolate(flat) id: vec4<u32>,
  @location(6) tip: f32,
  // true width over drawn width (pixel coverage of a thread thinner than its ribbon)
  @location(7) alpha: f32,
};

fn hiddenT() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

struct ChunkRef {
  ri: RunInfo,
  r: u32,
  a: f32,
  b: f32,
};

fn chunkRef(ii: u32) -> ChunkRef {
  var o: ChunkRef;
  let rec = chunks[ii];
  o.r = rec >> 10u;
  o.ri = runInfo(runs[o.r]);
  o.a = o.ri.drawA + f32((rec & 1023u) * CHUNK);
  o.b = min(o.a + f32(CHUNK), o.ri.drawB);
  return o;
}

// Point of the thread of chunk ch at drawn parameter p, from its strand's chain (thread_geom.wgsl).
fn threadOf(ch: ChunkRef, p: f32) -> ThreadPoint {
  let g = runGeoAt(ch.r);
  let cA = runGeo[RUN_GEO * ch.r + 5u].xyz;
  let cB = runGeo[RUN_GEO * ch.r + 6u].xyz;
  return threadAt(ch.ri, g, cA, cB, p);
}

fn threadVertex(vi: u32, ii: u32, fromLight: bool) -> VOut {
  let ch = chunkRef(ii);
  if (ch.a >= ch.b) { return hiddenT(); }
  let p = mix(ch.a, ch.b, f32(vi >> 1u) / f32(CHUNK_PTS - 1u));
  let tp = threadOf(ch, p);
  if (!tp.valid) { return hiddenT(); }
  let sideSign = select(-1.0, 1.0, (vi & 1u) == 1u);
  var V: vec3<f32>;
  if (fromLight) { V = normalize(F.keyPos.xyz - tp.p); } else { V = normalize(F.camPos.xyz - tp.p); }
  var B = cross(tp.t, V);
  let bl = length(B);
  if (bl < 1e-6) { B = normalize(cross(tp.t, tp.n)); } else { B = B / bl; }
  var half = tp.radius;
  var alpha = 1.0;
  if (!fromLight) {
    // at least ~0.7 px wide; a thinner thread covers its true fraction of the ribbon
    let minHalf = 0.35 * F.camPos.w * length(F.camPos.xyz - tp.p);
    alpha = clamp(half / minHalf, 0.0, 1.0);
    half = max(half, minHalf);
  }
  let W = tp.p + sideSign * half * B;
  var o: VOut;
  if (fromLight) { o.pos = F.lightViewProj * vec4<f32>(W, 1.0); } else { o.pos = F.viewProj * vec4<f32>(W, 1.0); }
  o.world = W;
  o.side = sideSign;
  o.tangent = tp.t;
  o.binormal = B;
  o.c = p;
  o.id = vec4<u32>(ch.ri.yarn, ch.ri.fam, ch.ri.layer, 0u);
  o.tip = tp.tip;
  o.alpha = alpha;
  return o;
}

@vertex fn vsThread(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut { return threadVertex(vi, ii, false); }
@vertex fn vsThreadShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> { return threadVertex(vi, ii, true).pos; }

@fragment
fn fsThread(in: VOut) -> @location(0) vec4<f32> {
  let LR = R.layers[in.id.z];
  let s = clamp(in.side, -1.0, 1.0);
  let V = normalize(F.camPos.xyz - in.world);
  let T = normalize(in.tangent);
  let B = normalize(in.binormal);
  let Vp = normalize(V - dot(V, T) * T);
  let crown = sqrt(max(0.0, 1.0 - s * s));
  let N = normalize(B * s + Vp * crown);
  let fam = in.id.y;
  let yarn = i32(in.id.x);
  let pitchAl = select(LR.pitch.y, LR.pitch.x, fam == 0u);
  let along = in.c * pitchAl;
  let pix = F.camPos.w * length(F.camPos.xyz - in.world);
  // a yarn freed from the weave is abraded and loosened: its pale core shows, more toward an
  // untwisting end
  let look = yarnLook(LR, in.id.z, fam, yarn, i32(floor(in.c / f32(LR.twill.x))), along, s, min(1.0, 0.8 + 0.2 * in.tip), pix, 1.0);
  let tw = select(LR.weft.w, LR.warp.w, fam == 1u) * (1.0 - in.tip);
  let Tf = normalize(cos(tw) * T + sin(tw) * cross(N, T));
  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, N);
  let amb = F.ambient.rgb * (0.55 + 0.45 * N.z);
  let ao = mix(0.6, 1.0, crown);
  var c = fiberShade(N, Tf, V, key.xyz, look.albedo, look.fuzz + 0.25) * (key.w * sh) * F.keyColor.rgb + overShade(N, Tf, V, look.albedo, look.fuzz + 0.25, in.world, N) + amb * look.albedo * ao;
  let cam = torchLight(in.world);
  if (cam.w > 0.0) { c += fiberShade(N, Tf, V, cam.xyz, look.albedo, look.fuzz + 0.25) * cam.w * F.torchColor.rgb; }
  return vec4<f32>(c, in.alpha);
}

// ---------------------------------------------------------------- fibres

struct FOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) tangent: vec3<f32>,
  @location(2) alpha: f32,
  // unused, family, layer, seed
  @location(3) @interpolate(flat) id: vec4<u32>,
};

fn hiddenF() -> FOut {
  var o: FOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

// Vertex vi of a fibre from `root` along unit `dir` over `len`, bent by `bend` (tip offset growing
// quadratically) and crimped by `wave`, `width` wide; drawn >= 1 px with alpha = true coverage.
fn fibre(root: vec3<f32>, dir: vec3<f32>, len: f32, bend: vec3<f32>, wave: vec3<f32>, waves: f32, width: f32, vi: u32, side: vec3<f32>, id: vec4<u32>) -> FOut {
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
  let halfW = max(0.5 * width, 0.5 * pixW);
  var o: FOut;
  let W = P + sideSign * halfW * B;
  o.pos = F.viewProj * vec4<f32>(W, 1.0);
  o.world = W;
  o.tangent = dP;
  o.alpha = clamp(width / max(pixW, 1e-9), 0.0, 1.0) * (1.0 - 0.6 * t);
  o.id = id;
  return o;
}

// Fibres of a loose thread: instance = chunk x fibre. Freed yarns untwist, so their fibres lie
// mostly along the thread; at an untwisting free end they splay out of it in a brush.
@vertex
fn vsThreadFuzz(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> FOut {
  let ch = chunkRef(ii / FIBERS_PER_CHUNK);
  if (ch.a >= ch.b) { return hiddenF(); }
  let seed = hash11((ch.ri.layer * 7919u + ch.ri.yarn) * 613u + u32(ch.a) * 131u + (ii % FIBERS_PER_CHUNK) * 6007u + ch.ri.fam);
  let p = mix(ch.a, ch.b, hashf(seed));
  let tp = threadOf(ch, p);
  if (!tp.valid) { return hiddenF(); }
  let az = 6.2831853 * hashf(seed + 1u);
  let u1 = normalize(cross(tp.t, tp.n));
  let u2 = cross(u1, tp.t);
  let radial = cos(az) * u1 + sin(az) * u2;
  let toward = select(-1.0, 1.0, hashf(seed + 11u) < 0.5);
  let splay = tp.tip * tp.tip;
  let dir = normalize(radial * mix(0.45 + 0.6 * hashf(seed + 2u), 0.8, splay) + tp.t * toward * mix(1.6, 1.0, splay));
  let rl = hashf(seed + 3u);
  let len = (0.15e-3 + 0.8e-3 * rl * rl) * (1.0 + 2.0 * splay);
  let bend = vec3<f32>(hashf(seed + 4u) - 0.5, hashf(seed + 5u) - 0.5, hashf(seed + 6u) - 0.8) * (0.9 * len);
  let wave = vec3<f32>(hashf(seed + 7u) - 0.5, hashf(seed + 8u) - 0.5, hashf(seed + 9u) - 0.5) * (0.3 * len);
  return fibre(tp.p + radial * tp.radius, dir, len, bend, wave, 1.0 + 2.0 * hashf(seed + 10u), 25e-6, vi, u1, vec4<u32>(0u, ch.ri.fam, ch.ri.layer, seed));
}

@fragment
fn fsThreadFuzz(in: FOut) -> @location(0) vec4<f32> {
  let V = normalize(F.camPos.xyz - in.world);
  let T = normalize(in.tangent);
  let N = normalize(V - dot(V, T) * T);
  // loose fibres are mostly the pale core and abraded surface fibres
  let pale = hashf(in.id.w + 9u) < 0.75;
  var albedo = vec3<f32>(0.40, 0.42, 0.47);
  if (in.id.y == 0u) { albedo = vec3<f32>(0.44, 0.44, 0.43); }
  if (!pale && in.id.y == 1u) { albedo = vec3<f32>(0.08, 0.12, 0.27); }
  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, N);
  var c = fiberShade(N, T, V, key.xyz, albedo, 0.2) * (key.w * sh) * F.keyColor.rgb + overShade(N, T, V, albedo, 0.2, in.world, N) + F.ambient.rgb * albedo;
  let cam = torchLight(in.world);
  if (cam.w > 0.0) { c += fiberShade(N, T, V, cam.xyz, albedo, 0.2) * cam.w * F.torchColor.rgb; }
  return vec4<f32>(c, in.alpha);
}
