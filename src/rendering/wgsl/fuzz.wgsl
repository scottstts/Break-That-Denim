// The free edges' fibre (render-only, attached to their material): per edge crossing (pieces.wgsl
// edgeList, roots placed by edge_roots.wgsl) FIBERS_PER_EDGE strands along its open sides:
//  - the ends of the yarns that ran across the side, untwisted into short splayed brushes;
//  - loose fibre standing out of the edge and fanning over the opening;
//  - fibre lying across the loosened rim behind the edge, drawn out over it.
// Together they make the soft pale band a torn denim edge reads as. Fibres are tens of microns wide:
// drawn as >= 1 px ribbons whose alpha is their true pixel coverage (alpha to coverage on 4x MSAA).

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> edgeList: array<u32>;
@group(0) @binding(3) var<storage, read> edgeRoots: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> fieldY: array<u32>;
@group(0) @binding(9) var shadowMap: texture_depth_2d;
@group(0) @binding(10) var shadowSampler: sampler_comparison;

const FIBER_PTS: u32 = FIBER_VERTS / 2u;
// strands of one yarn-end brush: its core and the fibres it splays into
const YARN_END_FIBRES: u32 = 6u;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) tangent: vec3<f32>,
  @location(2) alpha: f32,
  // kind (0 fibre, 1 yarn end, 2 fibre of a yarn end), family, layer, seed
  @location(3) @interpolate(flat) id: vec4<u32>,
  // 0 at the root .. 1 at the tip
  @location(4) along: f32,
};

fn hidden() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

// Vertex vi of a strand from `root` along unit `dir` over `len`, bent by `bend` (tip offset growing
// quadratically) and crimped by `wave` (amplitude of `waves` half periods), `width` wide at the root
// and `tipW` at the tip. Cotton fibres are convoluted ribbons and loose ones keep their yarn's twist
// crimp, so they never stand out as straight needles.
fn strand(root: vec3<f32>, dir: vec3<f32>, len: f32, bend: vec3<f32>, wave: vec3<f32>, waves: f32, width: f32, tipW: f32, vi: u32, side: vec3<f32>, id: vec4<u32>) -> VOut {
  let t = f32(vi >> 1u) / f32(FIBER_PTS - 1u);
  let ph = PI * waves * t;
  let P = root + dir * (len * t) + bend * (t * t) + wave * (sin(ph) * t);
  let dP = normalize(dir * len + bend * (2.0 * t) + wave * (sin(ph) + PI * waves * t * cos(ph)));
  let sideSign = select(-1.0, 1.0, (vi & 1u) == 1u);
  let V = normalize(F.camPos.xyz - P);
  var B = cross(dP, V);
  let bl = length(B);
  if (bl < 1e-6) { B = side; } else { B = B / bl; }
  let w = mix(width, tipW, t);
  let pixW = F.camPos.w * length(F.camPos.xyz - P);
  let halfW = max(0.5 * w, 0.5 * pixW);
  var o: VOut;
  let W = P + sideSign * halfW * B;
  o.pos = F.viewProj * vec4<f32>(W, 1.0);
  o.world = W;
  o.tangent = dP;
  o.alpha = clamp(w / max(pixW, 1e-9), 0.0, 1.0) * (1.0 - 0.5 * t * t);
  o.id = id;
  o.along = t;
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
  var l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (c >= R.layers[q].field.x) { l = q; }
  }
  let LR = R.layers[l];
  // how much loose material this stretch of edge carries: tufted stretches next to nearly bare ones
  // (yarns break at scattered places along their length and loosen unevenly), from noise over the
  // rest chart at the scale of a few crossings to several millimetres
  let FL = LR.field;
  let restC = vec2<f32>(f32((c - FL.x) % FL.y) * LR.pitch.x, f32((c - FL.x) / FL.y) * LR.pitch.y);
  let tuftN = 0.6 * noise2(6007u + l, restC / 5.0e-3) + 0.4 * noise2(6011u + l, restC / 1.6e-3);
  let density = smoothstep(-0.45, 0.5, tuftN);
  let dU = edgeRoots[3u * ei + 1u].xyz;
  let dV = edgeRoots[3u * ei + 2u].xyz;
  var N = normalize(cross(dU, dV));
  if (dot(N, F.camPos.xyz - root0.xyz) < 0.0) { N = -N; }
  let nOpen = countOneBits(sides);
  // the first YARN_END_FIBRES strands share one side: one yarn end and the fibres it splays into
  let brush = fi < YARN_END_FIBRES;
  let seedSide = select(hash11(c * 2654435761u + fi * 97u + 11u), hash11(c * 2654435761u + 7u), brush);
  var pick = u32(hashf(seedSide) * f32(nOpen)) % nOpen;
  var side = 0u;
  for (var b = 0u; b < 4u; b++) {
    if (((sides >> b) & 1u) == 0u) { continue; }
    if (pick == 0u) {
      side = b;
      break;
    }
    pick--;
  }
  let sg = select(-1.0, 1.0, (side & 1u) == 1u);
  // in the rest chart: out of the side, and along it
  let outR = select(vec2<f32>(0.0, sg), vec2<f32>(sg, 0.0), side < 2u);
  let alongR = vec2<f32>(-outR.y, outR.x);
  let outW = normalize(dU * outR.x + dV * outR.y);
  let acrossW = normalize(dU * alongR.x + dV * alongR.y);
  let pitchOut = select(LR.pitch.y, LR.pitch.x, side < 2u);
  // the yarn running across the side ends here: picks across the u sides, warps across the v sides
  let fam = select(1u, 0u, side < 2u);
  let seed = hash11(c * 2654435761u + fi * 97u + 11u);
  if (brush) {
    // a cut yarn end standing out of the edge (it was cut or pulled back here); its fibres untwist
    // from part-way along it into a splayed brush. All strands of the brush derive from one seed.
    if (yarnCode(fieldY[c], fam) >= PULLED_R) { return hidden(); }
    let sb = hash11(c * 2654435761u + 13u);
    let along = hashf(sb + 1u) - 0.5;
    let atSide = root0.xyz + (dU * outR.x + dV * outR.y) * (0.5 - 0.15 * hashf(sb + 2u)) + (dU * alongR.x + dV * alongR.y) * along;
    let spread = (hashf(sb + 3u) - 0.5) * 0.8;
    let lift = 0.1 + 0.45 * hashf(sb + 4u);
    let dir = normalize(outW + acrossW * spread + N * lift);
    let r = hashf(sb + 5u);
    let len = (0.6e-3 + 2.4e-3 * r * r) * (0.55 + 0.9 * density);
    let droop = vec3<f32>(0.0, 0.0, -0.3 * len) - N * (0.25 * len * lift);
    let curl = (acrossW * (hashf(sb + 6u) - 0.5) + N * (hashf(sb + 7u) - 0.5)) * (0.45 * len);
    let yw = select(LR.loose.y, LR.loose.x, fam == 0u) * (1.1 + 0.5 * hashf(sb + 8u));
    // where the twist opens up: the core runs as a yarn up to t0, its fibres splay beyond
    let t0 = 0.3 + 0.4 * hashf(sb + 9u);
    if (fi == 0u) {
      return strand(atSide, dir, len * t0, (droop + curl) * (t0 * t0), curl * 0.2, 1.0, yw, 0.7 * yw, vi, acrossW, vec4<u32>(1u, fam, l, sb));
    }
    let tip = atSide + dir * (len * t0) + (droop + curl) * (t0 * t0);
    let tangent = normalize(dir * len + (droop + curl) * (2.0 * t0));
    let fs = (vec3<f32>(hashf(seed + 3u), hashf(seed + 4u), hashf(seed + 5u)) - 0.5);
    let fdir = normalize(tangent + 0.55 * fs + N * 0.1 * hashf(seed + 6u));
    let flen = len * (1.0 - t0) * (0.6 + 0.9 * hashf(seed + 7u));
    let wv = (vec3<f32>(hashf(seed + 13u), hashf(seed + 14u), hashf(seed + 15u)) - 0.5) * (0.35 * flen);
    let bend = vec3<f32>(0.0, 0.0, -0.4 * flen) + fs * (0.5 * flen);
    let width = 18e-6 + 22e-6 * hashf(seed + 11u);
    return strand(tip, fdir, flen, bend, wv, 1.5 + 2.5 * hashf(seed + 16u), width, width, vi, acrossW, vec4<u32>(2u, fam, l, seed));
  }
  let famId = select(fam, 1u - fam, hashf(seed + 12u) < 0.3);
  let width = 18e-6 + 24e-6 * hashf(seed + 11u);
  let cv = vec3<f32>(hashf(seed + 8u) - 0.5, hashf(seed + 9u) - 0.5, hashf(seed + 10u) - 0.5);
  let wv = vec3<f32>(hashf(seed + 13u) - 0.5, hashf(seed + 14u) - 0.5, hashf(seed + 15u) - 0.5);
  let along = hashf(seed + 1u) - 0.5;
  let atSide = root0.xyz + (dU * outR.x + dV * outR.y) * (0.5 - 0.15 * hashf(seed + 2u)) + (dU * alongR.x + dV * alongR.y) * along;
  if (fi >= FIBERS_PER_EDGE - FIBERS_PER_EDGE / 3u) {
    // fibre of the loosened rim: rooted up to a few millimetres behind the edge, lying out over it;
    // thinner where the edge is bare
    if (hashf(seed + 19u) > 0.3 + 0.6 * density) { return hidden(); }
    let back = (0.4e-3 + 3.0e-3 * pow(hashf(seed + 17u), 1.5)) / pitchOut;
    let root = atSide - (dU * outR.x + dV * outR.y) * back;
    let spread = (hashf(seed + 3u) - 0.5) * 1.2;
    let lift = 0.03 + 0.15 * hashf(seed + 4u);
    let dir = normalize(outW + acrossW * spread + N * lift);
    let len = back * pitchOut + 0.2e-3 + 1.2e-3 * hashf(seed + 5u);
    let bend = cv * (0.7 * len) - N * (0.1 * len);
    return strand(root + N * 0.06e-3, dir, len, bend, wv * (0.4 * len), 2.0 + 4.0 * hashf(seed + 16u), width, width, vi, acrossW, vec4<u32>(0u, famId, l, seed));
  }
  // loose fibre standing out of the edge and curling over the opening: cotton fibres are short and
  // convoluted, so mostly short crimped wisps, a few longer ones drooping; only where the edge is tufted
  if (hashf(seed + 19u) > density) { return hidden(); }
  let spread = (hashf(seed + 3u) - 0.5) * 2.0;
  let lift = 0.1 + 0.7 * hashf(seed + 4u);
  let dir = normalize(outW + acrossW * spread + N * lift);
  let r = hashf(seed + 5u);
  var len = 0.5e-3 + 2.0e-3 * r * r;
  if (hashf(seed + 6u) < 0.12) { len += 1.0e-3 + 2.5e-3 * hashf(seed + 7u); }
  let droop = clamp(len / 1.5e-3, 0.0, 1.0);
  let bend = cv * (1.1 * len) - N * (0.6 * len * lift * droop) + vec3<f32>(0.0, 0.0, -0.3 * len * droop);
  return strand(atSide, dir, len, bend, wv * (0.45 * len), 2.0 + 4.0 * hashf(seed + 16u), width, width, vi, acrossW, vec4<u32>(0u, famId, l, seed));
}

@fragment
fn fsFuzz(in: VOut) -> @location(0) vec4<f32> {
  let V = normalize(F.camPos.xyz - in.world);
  let T = normalize(in.tangent);
  let N = normalize(V - dot(V, T) * T);
  // loose fibres and opened yarn ends are mostly the pale core and abraded surface fibre; a warp end
  // keeps some of its indigo near the root
  let pale = hashf(in.id.w + 9u) < 0.7;
  var albedo = vec3<f32>(0.42, 0.44, 0.49);
  if (in.id.y == 0u) { albedo = vec3<f32>(0.46, 0.46, 0.45); }
  if (in.id.y == 1u && (!pale || (in.id.x == 1u && in.along < 0.4))) { albedo = mix(vec3<f32>(0.1, 0.14, 0.3), albedo, select(0.0, in.along / 0.4, in.id.x == 1u)); }
  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, N);
  let c = fiberShade(N, T, V, key.xyz, albedo, 0.3) * (key.w * sh) + F.ambient.rgb * albedo;
  return vec4<f32>(c, in.alpha);
}

// ---------------------------------------------------------------- aggregate fringe

// The fringe of a free edge as the eye sees it beyond a few fibres per pixel: one translucent ribbon
// per open side of an edge crossing, from the side out over the opening, whose opacity is the
// coverage of the fibres and yarn ends standing out there (falling off toward their tips, tufted
// along the edge) and whose texture streaks along them. Where single fibres resolve (close views) it
// fades out and the individual strands above carry the fringe alone.

struct TOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  // along the side (m), out of it (0 root .. 1 tip)
  @location(1) st: vec2<f32>,
  @location(2) out: vec3<f32>,
  // fibre length here (m), aggregate weight, layer, unused
  @location(3) @interpolate(flat) info: vec4<f32>,
};

fn hiddenT() -> TOut {
  var o: TOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

@vertex
fn vsEdgeTuft(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> TOut {
  let ei = ii / 4u;
  let side = ii % 4u;
  let root0 = edgeRoots[3u * ei];
  if (root0.w < 0.5) { return hiddenT(); }
  let entry = edgeList[ei];
  if (((entry >> (28u + side)) & 1u) == 0u) { return hiddenT(); }
  let c = entry & 0x0FFFFFFFu;
  var l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (c >= R.layers[q].field.x) { l = q; }
  }
  let LR = R.layers[l];
  let FL = LR.field;
  let dU = edgeRoots[3u * ei + 1u].xyz;
  let dV = edgeRoots[3u * ei + 2u].xyz;
  var N = normalize(cross(dU, dV));
  if (dot(N, F.camPos.xyz - root0.xyz) < 0.0) { N = -N; }
  let sg = select(-1.0, 1.0, (side & 1u) == 1u);
  let outR = select(vec2<f32>(0.0, sg), vec2<f32>(sg, 0.0), side < 2u);
  let alongR = vec2<f32>(-outR.y, outR.x);
  let outW = dU * outR.x + dV * outR.y;
  let alongW = dU * alongR.x + dV * alongR.y;
  // the same tuft density as the strands (tufted stretches next to nearly bare ones)
  let restC = vec2<f32>(f32((c - FL.x) % FL.y) * LR.pitch.x, f32((c - FL.x) / FL.y) * LR.pitch.y);
  let tuftN = 0.6 * noise2(6007u + l, restC / 5.0e-3) + 0.4 * noise2(6011u + l, restC / 1.6e-3);
  let density = smoothstep(-0.45, 0.5, tuftN);
  let len = (2.0e-3 + 3.0e-3 * density) * (0.8 + 0.4 * hashf(c * 2654435761u + side));
  let pix = F.camPos.w * length(F.camPos.xyz - root0.xyz);
  // individual fibres (~30 um) resolve below ~40 um per pixel
  let agg = clamp((pix - 40e-6) / 120e-6, 0.0, 1.0);
  if (agg <= 0.0) { return hiddenT(); }
  let su = select(-0.5, 0.5, (vi & 1u) == 1u);
  let tv = f32(vi >> 1u);
  let olen = length(outW);
  let dir = normalize(outW / max(olen, 1e-12) + N * 0.25) ;
  // from the side's midpoint (half a crossing out), fanned slightly wider at the tip, drooping a little
  let base = root0.xyz + outW * 0.5;
  let P = base + alongW * (su * (1.0 + 0.4 * tv)) + dir * (len * tv) + vec3<f32>(0.0, 0.0, -0.2 * len * tv * tv);
  var o: TOut;
  o.pos = F.viewProj * vec4<f32>(P, 1.0);
  o.world = P;
  o.st = vec2<f32>(su * length(alongW), tv);
  o.out = dir;
  o.info = vec4<f32>(len, agg * (0.6 + 0.35 * density), f32(l), 0.0);
  return o;
}

@fragment
fn fsEdgeTuft(in: TOut) -> @location(0) vec4<f32> {
  let l = u32(in.info.z);
  let t = clamp(in.st.y, 0.0, 1.0);
  // fibres end at scattered lengths: coverage falls toward the tips; streaked along the fibres
  let streak = 0.5 + 0.5 * noise2(7001u + l, vec2<f32>(in.st.x / 0.12e-3, t * in.info.x / 1.2e-3));
  let clump = 0.5 + 0.5 * noise2(7013u + l, vec2<f32>(in.st.x / 0.5e-3, t * in.info.x / 2.0e-3));
  let cover = in.info.y * pow(1.0 - t, 0.9) * mix(0.4, 1.0, streak) * mix(0.65, 1.0, clump);
  if (cover <= 0.003) { discard; }
  let V = normalize(F.camPos.xyz - in.world);
  let T = normalize(in.out);
  let Nf = normalize(V - dot(V, T) * T);
  // loose fibre: pale cores and weft, some keeping their indigo near the root
  let albedo = mix(vec3<f32>(0.3, 0.36, 0.52), vec3<f32>(0.62, 0.64, 0.68), smoothstep(0.0, 0.4, t) * 0.75 + 0.25 * clump);
  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, Nf);
  let c = fiberShade(Nf, T, V, key.xyz, albedo, 0.4) * (key.w * sh) + F.ambient.rgb * albedo;
  return vec4<f32>(c, cover);
}
