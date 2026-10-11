// A staged creature: a skinned mesh (linear blend skinning on the GPU) lit by the scene's lights with
// their shadow maps (the cloth shades it through its openings, and it shades the cloth), plus the
// stage's own pit light below the cloth. Skin: GGX specular on a wrapped diffuse (light scattered
// under the skin reaches past the terminator, reddened).

struct CreatureU {
  // pit light: position (xyz), unused
  pit: vec4<f32>,
  // pit light: radiant intensity (rgb, W/sr-equivalent in the scene's units), unused
  pitColor: vec4<f32>,
};

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> CU: CreatureU;
@group(0) @binding(2) var<storage, read> joints: array<mat4x4<f32>>;
@group(0) @binding(3) var shadowMap: texture_depth_2d;
@group(0) @binding(4) var shadowSampler: sampler_comparison;
@group(0) @binding(5) var overMap: texture_depth_2d;
@group(0) @binding(6) var baseTex: texture_2d<f32>;
@group(0) @binding(7) var normalTex: texture_2d<f32>;
@group(0) @binding(8) var ormTex: texture_2d<f32>;
@group(0) @binding(9) var texSampler: sampler;

struct VIn {
  @location(0) pos: vec3<f32>,
  @location(1) nrm: vec3<f32>,
  @location(2) tan: vec4<f32>,
  @location(3) uv: vec2<f32>,
  @location(4) j: vec4<u32>,
  @location(5) w: vec4<f32>,
};

struct VOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) nrm: vec3<f32>,
  @location(2) tan: vec4<f32>,
  @location(3) uv: vec2<f32>,
};

fn skinMatrix(j: vec4<u32>, w: vec4<f32>) -> mat4x4<f32> {
  return joints[j.x] * w.x + joints[j.y] * w.y + joints[j.z] * w.z + joints[j.w] * w.w;
}

@vertex
fn vsMain(v: VIn) -> VOut {
  let M = skinMatrix(v.j, v.w);
  let P = (M * vec4<f32>(v.pos, 1.0)).xyz;
  let M3 = mat3x3<f32>(M[0].xyz, M[1].xyz, M[2].xyz);
  var o: VOut;
  o.clip = F.viewProj * vec4<f32>(P, 1.0);
  o.world = P;
  o.nrm = M3 * v.nrm;
  o.tan = vec4<f32>(M3 * v.tan.xyz, v.tan.w);
  o.uv = v.uv;
  return o;
}

// depth from the light whose projection the bound frame uniform carries (key or overhead)
@vertex
fn vsShadow(v: VIn) -> @builtin(position) vec4<f32> {
  let M = skinMatrix(v.j, v.w);
  return F.lightViewProj * (M * vec4<f32>(v.pos, 1.0));
}

fn ggxSpec(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, a: f32, f0: f32) -> f32 {
  let H = normalize(L + V);
  let NL = max(dot(N, L), 0.0);
  let NV = max(dot(N, V), 1e-3);
  let NH = max(dot(N, H), 0.0);
  let a2 = a * a;
  let d = NH * NH * (a2 - 1.0) + 1.0;
  let D = a2 / (PI * d * d);
  let k = a * 0.5;
  let G = (NL / (NL * (1.0 - k) + k)) * (NV / (NV * (1.0 - k) + k));
  let Fr = f0 + (1.0 - f0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
  return D * G * Fr / max(4.0 * NL * NV, 1e-4) * NL;
}

// skin diffuse times cos: wrapped (light scattered under the skin), the wrapped part reddened
fn skinDiffuse(N: vec3<f32>, L: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
  let NL = dot(N, L);
  let lam = max(NL, 0.0);
  let wrap = clamp((NL + 0.3) / 1.3, 0.0, 1.0);
  let scatter = (wrap - lam) * vec3<f32>(1.0, 0.45, 0.35);
  return albedo * (lam + scatter) * (1.0 / PI);
}

@fragment
fn fsMain(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
  let base = textureSample(baseTex, texSampler, in.uv);
  let orm = textureSample(ormTex, texSampler, in.uv);
  let tn = textureSample(normalTex, texSampler, in.uv).xyz * 2.0 - 1.0;
  var Ng = normalize(in.nrm);
  if (!front) { Ng = -Ng; }
  let T = normalize(in.tan.xyz - Ng * dot(in.tan.xyz, Ng));
  let B = cross(Ng, T) * in.tan.w;
  let N = normalize(T * tn.x + B * tn.y + Ng * tn.z);
  let V = normalize(F.camPos.xyz - in.world);
  let albedo = base.rgb;
  let ao = orm.r;
  // skin and drying blood: never mirror-like
  let rough = clamp(orm.g, 0.38, 1.0);
  let a = rough * rough;
  let f0 = 0.028;

  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, Ng);
  var c = (skinDiffuse(N, key.xyz, albedo) + vec3<f32>(ggxSpec(N, V, key.xyz, a, f0))) * (key.w * sh) * F.keyColor.rgb;

  let Lo = F.overDir.xyz;
  c += (skinDiffuse(N, Lo, albedo) + vec3<f32>(ggxSpec(N, V, Lo, a, f0))) * F.overColor.rgb * shadowOver(in.world, Ng);

  // the pit light below the cloth (a small bulb: inverse square, unshadowed)
  let dp = CU.pit.xyz - in.world;
  let r2 = max(dot(dp, dp), 1e-4);
  let Lp = dp * inverseSqrt(r2);
  c += (skinDiffuse(N, Lp, albedo) + vec3<f32>(ggxSpec(N, V, Lp, a, f0))) * CU.pitColor.rgb / r2;

  // the camera's light
  if (F.torchPos.w > 0.0) {
    let tl = torchLight(in.world);
    c += (skinDiffuse(N, tl.xyz, albedo) + vec3<f32>(ggxSpec(N, V, tl.xyz, a, f0))) * tl.w * F.torchColor.rgb;
  }

  c += F.ambient.rgb * albedo * ao * (0.55 + 0.45 * N.z);
  return vec4<f32>(c, 1.0);
}
