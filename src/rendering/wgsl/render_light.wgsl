// Lighting shared by the shaded passes (requires a module-scope `F: FrameU` uniform and the shadow
// maps shadowMap (key), overMap (overhead light) with shadowSampler).

// ---------------------------------------------------------------- lighting

// Key light: a broad soft source above the top-left of the frame; returns (direction to light,
// irradiance scale). Its colour (F.keyColor) multiplies the irradiance where it is used.
fn keyLight(P: vec3<f32>) -> vec4<f32> {
  let d = F.keyPos.xyz - P;
  let dist = length(d);
  let Ld = d / dist;
  let c = dot(-Ld, F.keyAxis.xyz);
  let cone = smoothstep(F.keyAxis.w, F.keyShape.x, c);
  let r0 = F.params.z;
  let fall = (r0 * r0) / (dist * dist);
  return vec4<f32>(Ld, F.keyPos.w * fall * mix(F.keyShape.y, 1.0, cone));
}

// The camera's light (direction to it, irradiance; times F.torchColor where used): a small source at the
// lens, so nothing the camera sees is in its shadow (no shadow map); inverse square, its cone.
fn torchLight(P: vec3<f32>) -> vec4<f32> {
  if (F.torchPos.w <= 0.0) { return vec4<f32>(0.0, 0.0, 1.0, 0.0); }
  let d = F.torchPos.xyz - P;
  let r2 = max(dot(d, d), 1e-4);
  let Ld = d * inverseSqrt(r2);
  let cone = smoothstep(F.torchAxis.w, F.torchColor.w, dot(-Ld, F.torchAxis.xyz));
  return vec4<f32>(Ld, F.torchPos.w * cone / r2);
}

// Fibre BRDF times cos for one light: diffuse from a fibrous yarn surface, a longitudinal specular
// lobe on the fibre direction T, and grazing sheen from loose surface fibres.
fn fiberShade(N: vec3<f32>, T: vec3<f32>, V: vec3<f32>, Ldir: vec3<f32>, albedo: vec3<f32>, fuzz: f32) -> vec3<f32> {
  let NL = dot(N, Ldir);
  let NV = max(dot(N, V), 1e-3);
  // fibrous diffuse: Lambert softened toward the fibre cylinder response
  let TL = dot(T, Ldir);
  let sinTL = sqrt(max(0.0, 1.0 - TL * TL));
  let lam = max(NL, 0.0);
  let wrapd = clamp((NL + 0.35) / 1.35, 0.0, 1.0) * sinTL;
  let diff = albedo * (0.72 * lam + 0.28 * wrapd) * (1.0 / PI);
  // longitudinal fibre specular (Marschner R lobe approximation): sin(thetaI) + sin(thetaR) = 0
  let TV = dot(T, V);
  // rough cotton fibres: weak, broad longitudinal lobe
  let beta = 0.35;
  let sh = TL + TV;
  let lobe = exp(-(sh * sh) / (2.0 * beta * beta));
  let H = normalize(Ldir + V);
  let cosPhi = clamp(dot(normalize(Ldir - TL * T), normalize(V - TV * T)), -1.0, 1.0);
  let spec = vec3<f32>(0.018) * lobe * sqrt(0.5 + 0.5 * cosPhi) * max(NL + 0.1, 0.0) / (2.0 * PI * beta);
  // asperity (fuzz) sheen: Charlie distribution with Neubelt visibility
  let NH = max(dot(N, H), 0.0);
  let r = 0.65;
  let inv = 1.0 / r;
  let sin2 = max(1.0 - NH * NH, 0.0078125);
  let D = (2.0 + inv) * pow(sin2, inv * 0.5) / (2.0 * PI);
  let vis = 1.0 / (4.0 * (lam + NV - lam * NV) + 1e-4);
  let sheenCol = mix(albedo * 2.5, vec3<f32>(0.30, 0.36, 0.50), 0.5);
  let sheen = sheenCol * fuzz * 0.22 * D * vis * lam;
  return diff + spec + sheen;
}

// Soft shadow from a broad light: PCF over a rotated Poisson disk of radiusTexels shadow-map texels
// (the penumbra the light's size gives a nominal 2 mm occluder gap).
// bias: the depth offset of the comparison (m) over the projection's depth range (m).
fn shadowPCF(map: texture_depth_2d, smp: sampler_comparison, M: mat4x4<f32>, texel: f32, radiusTexels: f32, bias: f32, world: vec3<f32>, nrmW: vec3<f32>) -> f32 {
  let p = M * vec4<f32>(world + nrmW * (1.5 * texel), 1.0);
  let uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 1.0; }
  let depth = p.z - bias;
  var taps = array<vec2<f32>, 8>(
    vec2<f32>(-0.326, -0.406), vec2<f32>(-0.840, -0.074), vec2<f32>(-0.696, 0.457), vec2<f32>(-0.203, 0.621),
    vec2<f32>(0.962, -0.195), vec2<f32>(0.473, -0.480), vec2<f32>(0.519, 0.767), vec2<f32>(0.185, -0.893));
  let dims = vec2<f32>(textureDimensions(map));
  let ang = hash2f(u32(world.x * 4e5 + 1e5), u32(world.y * 4e5 + 1e5)) * 6.2831853;
  let cs = vec2<f32>(cos(ang), sin(ang));
  let radius = clamp(radiusTexels, 1.5, 6.0) / dims.x;
  var acc = 0.0;
  for (var k = 0u; k < 8u; k++) {
    let d = taps[k];
    let r = vec2<f32>(d.x * cs.x - d.y * cs.y, d.x * cs.y + d.y * cs.x);
    acc += textureSampleCompareLevel(map, smp, uv + r * radius, depth);
  }
  return acc / 8.0;
}

// Depth offset of the shadow comparison (m).
const SHADOW_BIAS: f32 = 0.24e-3;

// The key's shadow (module-scope shadowMap / shadowSampler).
fn shadowFactor(world: vec3<f32>, nrmW: vec3<f32>) -> f32 {
  return shadowPCF(shadowMap, shadowSampler, F.lightViewProj, F.ambient.w, 3.0, SHADOW_BIAS / F.keyCov.y, world, nrmW);
}

// The overhead light's shadow (module-scope overMap). The small lamp's penumbra over a nominal 2 mm
// gap is a fraction of a texel, so four bilinear comparisons a texel apart (a fixed 4 x 4 texel
// footprint, no rotated disk) resolve it.
fn shadowOver(world: vec3<f32>, nrmW: vec3<f32>) -> f32 {
  let texel = F.overDir.w;
  let p = F.overViewProj * vec4<f32>(world + nrmW * (1.5 * texel), 1.0);
  let uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) { return 1.0; }
  let depth = p.z - SHADOW_BIAS / F.overCov.y;
  let o = 0.5 / f32(textureDimensions(overMap).x);
  var acc = textureSampleCompareLevel(overMap, shadowSampler, uv + vec2<f32>(-o, -o), depth);
  acc += textureSampleCompareLevel(overMap, shadowSampler, uv + vec2<f32>(o, -o), depth);
  acc += textureSampleCompareLevel(overMap, shadowSampler, uv + vec2<f32>(-o, o), depth);
  acc += textureSampleCompareLevel(overMap, shadowSampler, uv + vec2<f32>(o, o), depth);
  return 0.25 * acc;
}

// Fibre BRDF times cos for the overhead light, with its shadow and colour.
fn overShade(N: vec3<f32>, T: vec3<f32>, V: vec3<f32>, albedo: vec3<f32>, fuzz: f32, world: vec3<f32>, nrmW: vec3<f32>) -> vec3<f32> {
  return fiberShade(N, T, V, F.overDir.xyz, albedo, fuzz) * F.overColor.rgb * shadowOver(world, nrmW);
}
