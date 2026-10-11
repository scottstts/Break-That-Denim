// The floor beneath the cloth: a textured dielectric (base colour, normal, roughness, AO from a
// scanned set; base colour and normal at full resolution cropped to the floor ever seen, the slowly
// varying roughness and AO as the whole tile) lit by the key, the overhead lamp and the fill, each seen through the cloth above it.
// The cloth's occlusion of a light comes from its shadow map prefiltered into coverage (coverage.wgsl):
// the fraction of the source a floor point sees is one minus the cloth's coverage over the penumbra
// footprint, whose radius is the cloth's height above the point (along the light) times the source's
// angular radius, so contact with nothing is sharp and a hole's light falls soft-edged. The fill is
// the hemisphere above, seen through the openings: the cloth's coverage weighted by the cosine kernel
// of a plane at the cloth's height (rings of equal energy).

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var baseTex: texture_2d<f32>;
@group(0) @binding(2) var normalTex: texture_2d<f32>;
// roughness, ambient occlusion
@group(0) @binding(3) var raTex: texture_2d<f32>;
@group(0) @binding(4) var matSampler: sampler;
// the crops: clamped (they do not tile)
@group(0) @binding(11) var detailSampler: sampler;
// per light: coverage of its shadow projection and coverage-weighted cloth depth, mipmapped
@group(0) @binding(5) var keyCovTex: texture_2d<f32>;
@group(0) @binding(6) var overCovTex: texture_2d<f32>;
@group(0) @binding(7) var covSampler: sampler;
// declared for the shared lighting (render_light.wgsl); the floor sees the cloth's shadows through the
// coverage instead, so these are never bound
@group(0) @binding(8) var shadowMap: texture_depth_2d;
@group(0) @binding(9) var shadowSampler: sampler_comparison;
@group(0) @binding(10) var overMap: texture_depth_2d;

// polished stone / resin: the dielectric's reflectance at normal incidence
const F0: f32 = 0.04;
// half extent of the floor quad around the view target (m)
const FLOOR_HALF: f32 = 1.5;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
};

@vertex
fn vsFloor(@builtin(vertex_index) vi: u32) -> VOut {
  var c = array<vec2<f32>, 6>(vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
  let W = vec3<f32>(c[vi] * FLOOR_HALF, F.floor.x);
  var o: VOut;
  o.world = W;
  o.pos = F.viewProj * vec4<f32>(W, 1.0);
  return o;
}

fn lightUV(M: mat4x4<f32>, P: vec3<f32>) -> vec3<f32> {
  let p = M * vec4<f32>(P, 1.0);
  return vec3<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5, p.z);
}

// Mip level whose texels span a footprint of the given diameter (m).
fn covLevel(L: vec4<f32>, diameter: f32) -> f32 {
  return log2(max(diameter / (L.z / L.w), 1.0));
}

// Fraction of a light seen from the floor point P through the cloth. L: the light's coverage
// parameters (FrameU keyCov / overCov); nominalGap: the cloth plane's distance from P along the light.
fn lightVis(tex: texture_2d<f32>, M: mat4x4<f32>, L: vec4<f32>, P: vec3<f32>, nominalGap: f32) -> f32 {
  let q = lightUV(M, P);
  // beyond the projection the light's path crosses the cloth's plane outside the coupon: the mount
  if (any(q.xy < vec2<f32>(0.0)) || any(q.xy > vec2<f32>(1.0))) { return 0.0; }
  // the cloth's mean depth over the nominal penumbra (the blocker search)
  let s = textureSampleLevel(tex, covSampler, q.xy, covLevel(L, 2.0 * nominalGap * L.x));
  if (s.x < 1e-4) { return 1.0; }
  let gap = max((q.z - s.y / s.x) * L.y, 0.0);
  let R = gap * L.x;
  // coverage over the penumbra disc: four taps of the footprint half its size, tent-like
  let lv = covLevel(L, R);
  let o = 0.5 * R / L.z;
  var c = 0.0;
  c += textureSampleLevel(tex, covSampler, q.xy + vec2<f32>(o, o), lv).x;
  c += textureSampleLevel(tex, covSampler, q.xy + vec2<f32>(-o, o), lv).x;
  c += textureSampleLevel(tex, covSampler, q.xy + vec2<f32>(o, -o), lv).x;
  c += textureSampleLevel(tex, covSampler, q.xy + vec2<f32>(-o, -o), lv).x;
  return clamp(1.0 - 0.25 * c, 0.0, 1.0);
}

struct Sky {
  // cosine-weighted fraction of the sky seen, and the cloth's height above the point (m)
  vis: f32,
  gap: f32,
};

// The fill seen from P through the openings of the cloth above it (the overhead projection is close
// to vertical, so its texels are the floor's xy).
fn skyVis(P: vec3<f32>) -> Sky {
  let L = F.overCov;
  let q = lightUV(F.overViewProj, P);
  var sk: Sky;
  sk.vis = 1.0;
  sk.gap = -F.floor.x;
  if (any(q.xy < vec2<f32>(0.0)) || any(q.xy > vec2<f32>(1.0))) { return sk; }
  let s = textureSampleLevel(overCovTex, covSampler, q.xy, covLevel(L, 2.0 * sk.gap));
  if (s.x > 1e-4) { sk.gap = max((q.z - s.y / s.x) * L.y, 1e-3); }
  // a parallel plane at height g: the cosine-weighted share within radius r is r^2 / (r^2 + g^2);
  // rings of a quarter of it each at their median radius, four taps around each
  var radii = array<f32, 4>(0.378, 0.775, 1.29, 2.65);
  var cov = 0.0;
  for (var k = 0u; k < 4u; k++) {
    let r = radii[k] * sk.gap;
    let lv = covLevel(L, 1.4 * r);
    let a = 0.785398 * f32(k & 1u);
    for (var j = 0u; j < 4u; j++) {
      let t = a + 1.570796 * f32(j);
      cov += textureSampleLevel(overCovTex, covSampler, q.xy + vec2<f32>(cos(t), -sin(t)) * (r / L.z), lv).x;
    }
  }
  sk.vis = clamp(1.0 - cov / 16.0, 0.0, 1.0);
  return sk;
}

// GGX specular (Smith height-correlated visibility, Schlick Fresnel) times cos for unit irradiance.
fn ggx(N: vec3<f32>, V: vec3<f32>, Ld: vec3<f32>, alpha: f32) -> f32 {
  let NL = dot(N, Ld);
  if (NL <= 0.0) { return 0.0; }
  let NV = max(dot(N, V), 1e-3);
  let H = normalize(Ld + V);
  let NH = max(dot(N, H), 0.0);
  let a2 = alpha * alpha;
  let dd = NH * NH * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * dd * dd);
  let vis = 0.5 / (NL * sqrt(NV * NV * (1.0 - a2) + a2) + NV * sqrt(NL * NL * (1.0 - a2) + a2));
  let Fr = F0 + (1.0 - F0) * pow(1.0 - max(dot(V, H), 0.0), 5.0);
  return D * vis * Fr * NL;
}

@fragment
fn fsFloor(in: VOut) -> @location(0) vec4<f32> {
  let P = in.world;
  let V = normalize(F.camPos.xyz - P);
  // texture space: u along +x, v down the image = -y (a +Y-up normal map then points along +y); the
  // tile is centred on the view (the visible floor is far smaller than one), so its edges never show
  let uv = vec2<f32>(P.x, -P.y) * F.floor.y + 0.5;
  let duv = (uv - 0.5) * F.floor.w / vec2<f32>(textureDimensions(baseTex)) + 0.5;
  let base = textureSample(baseTex, detailSampler, duv).rgb;
  let ra = textureSample(raTex, matSampler, uv).rg;
  let nt = textureSample(normalTex, detailSampler, duv).xyz * 2.0 - 1.0;
  let nl = max(length(nt), 1e-3);
  let N = normalize(nt);
  // Toksvig: the normals a texel averages widen the lobe by the length they lose
  let alpha = sqrt(min(ra.x * ra.x * ra.x * ra.x + (1.0 - min(nl, 1.0)) / nl, 1.0));
  let ao = ra.y;
  let diff = base * (1.0 / PI);

  // key: a broad source, seen through the cloth
  let key = keyLight(P);
  let visK = lightVis(keyCovTex, F.lightViewProj, F.keyCov, P, -F.floor.x / max(key.z, 0.05));
  let aK = sqrt(alpha * alpha + 0.25 * F.keyCov.x * F.keyCov.x);
  var c = (diff * max(dot(N, key.xyz), 0.0) + vec3<f32>(ggx(N, V, key.xyz, aK))) * (key.w * visK) * F.keyColor.rgb;

  // overhead lamp: a distant disc; its size widens the highlight (half-vector spread of half its radius)
  let Lo = F.overDir.xyz;
  let visO = lightVis(overCovTex, F.overViewProj, F.overCov, P, -F.floor.x / max(Lo.z, 0.05));
  let aO = sqrt(alpha * alpha + 0.25 * F.overColor.w * F.overColor.w);
  c += (diff * max(dot(N, Lo), 0.0) + vec3<f32>(ggx(N, V, Lo, aO))) * F.overColor.rgb * visO;

  // fill through the openings, and its mirror image where the reflected view leaves through one
  let sky = skyVis(P);
  c += F.ambient.rgb * base * ao * sky.vis;
  let Rv = reflect(-V, N);
  if (Rv.z > 0.05) {
    let q = lightUV(F.overViewProj, P);
    let off = Rv.xy * (sky.gap / Rv.z);
    let lv = covLevel(F.overCov, 2.0 * sky.gap * alpha);
    let open = 1.0 - textureSampleLevel(overCovTex, covSampler, q.xy + vec2<f32>(off.x, -off.y) / F.overCov.z, lv).x;
    let Fr = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    c += F.ambient.rgb * (Fr * ao * open);
  }
  return vec4<f32>(c, 1.0);
}
