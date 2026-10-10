// Explicit yarns: loose threads beside torn cells (yarn_geom.wgsl).
// Each yarn is a camera-facing ribbon around its centreline shaded as a round fibrous cylinder with
// the same material identity as the intact surface.

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> nodes: array<NodeStaticR>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> nrm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> segw: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> cellFlagsY: array<u32>;
@group(0) @binding(7) var<storage, read> lists: array<u32>;
@group(0) @binding(8) var<storage, read> wearR: array<f32>;
@group(0) @binding(9) var shadowMap: texture_depth_2d;
@group(0) @binding(10) var shadowSampler: sampler_comparison;
@group(0) @binding(11) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read> fieldY: array<u32>;

override SEGS: u32 = 8u;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) side: f32,
  @location(2) tangent: vec3<f32>,
  @location(3) binormal: vec3<f32>,
  @location(4) c: f32,
  @location(5) @interpolate(flat) id: vec4<u32>, // yarn (bitcast), family, layer, unused
  @location(6) wearV: f32,
  @location(7) radius: f32,
};

fn ribbon(y: YarnPoint, vi: u32, fromLight: bool) -> VOut {
  let sideSign = select(-1.0, 1.0, (vi & 1u) == 1u);
  var V: vec3<f32>;
  if (fromLight) { V = normalize(F.keyPos.xyz - y.p); } else { V = normalize(F.camPos.xyz - y.p); }
  var B = cross(y.t, V);
  let bl = length(B);
  if (bl < 1e-6) { B = normalize(cross(y.t, y.n)); } else { B = B / bl; }
  var half = y.radius;
  if (!fromLight) {
    // never thinner than ~0.7 px on screen; thinner yarns are drawn at coverage via MSAA
    half = max(half, 0.35 * F.camPos.w * length(F.camPos.xyz - y.p));
  }
  let W = y.p + sideSign * half * B;
  var o: VOut;
  if (fromLight) { o.pos = F.lightViewProj * vec4<f32>(W, 1.0); } else { o.pos = F.viewProj * vec4<f32>(W, 1.0); }
  o.world = W;
  o.side = sideSign;
  o.tangent = y.t;
  o.binormal = B;
  o.c = y.c;
  o.id = vec4<u32>(bitcast<u32>(y.yarn), y.fam, y.layer, 0u);
  o.wearV = y.wear;
  o.radius = y.radius;
  return o;
}

fn degenerate() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

@fragment
fn fsYarn(in: VOut) -> @location(0) vec4<f32> {
  let LR = R.layers[in.id.z];
  let s = clamp(in.side, -1.0, 1.0);
  let V = normalize(F.camPos.xyz - in.world);
  let T = normalize(in.tangent);
  let B = normalize(in.binormal);
  let Vp = normalize(V - dot(V, T) * T);
  let crown = sqrt(max(0.0, 1.0 - s * s));
  let N = normalize(B * s + Vp * crown);
  let fam = in.id.y;
  let yarn = bitcast<i32>(in.id.x);
  let pitchAlong = select(LR.pitch.x, LR.pitch.y, fam == 1u);
  let along = in.c * pitchAlong;
  let pix = F.camPos.w * length(F.camPos.xyz - in.world);
  // freed yarns are worn: frayed, pale cores show more than on the intact face
  let look = yarnLook(LR, in.id.z, fam, yarn, i32(floor(in.c / f32(LR.twill.x))), along, s, min(1.0, in.wearV + 0.1), pix);
  let tw = select(LR.weft.w, LR.warp.w, fam == 1u);
  let Tf = normalize(cos(tw) * T + sin(tw) * cross(N, T));
  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, N);
  let amb = F.ambient.rgb * (0.55 + 0.45 * N.z);
  let ao = mix(0.6, 1.0, crown);
  let c = fiberShade(N, Tf, V, key.xyz, look.albedo, look.fuzz + 0.15) * (key.w * sh) + amb * look.albedo * ao;
  return vec4<f32>(c, 1.0);
}

// Loose bundle threads (yarn_geom.wgsl strandPoint): instance = bundle span x (strand, half).
fn strandVertex(vi: u32, ii: u32, fromLight: bool) -> VOut {
  let s = bundleEntry(ii / BUNDLE_HALVES);
  let k = (ii % BUNDLE_HALVES) >> 1u;
  let half = ii & 1u;
  let u = f32(vi >> 1u) / f32(STRAND_PTS - 1u);
  let sp = strandPoint(s, k, half, u);
  if (!sp.valid) { return degenerate(); }
  return ribbon(sp.y, vi, fromLight);
}

@vertex fn vsStrand(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut { return strandVertex(vi, ii, false); }
@vertex fn vsStrandShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> { return strandVertex(vi, ii, true).pos; }
