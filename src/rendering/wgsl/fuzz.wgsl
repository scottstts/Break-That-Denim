// Loose surface fibres on freed yarns and frayed tail ends (render-only, attached to their yarn).
// Each drawn fibre is a short curved tuft rooted on its parent yarn's surface, tens of microns wide:
// far below a pixel, so it is drawn as a >= 1 px ribbon whose alpha is the true pixel coverage,
// resolved by alpha-to-coverage on the 4x MSAA target.

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> nodes: array<NodeStaticR>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> nrm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> segw: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> tails: array<f32>;
@group(0) @binding(7) var<storage, read> lists: array<u32>;
@group(0) @binding(8) var<storage, read> wearR: array<f32>;
@group(0) @binding(9) var shadowMap: texture_depth_2d;
@group(0) @binding(10) var shadowSampler: sampler_comparison;

override FUZZ_PER_YARN: u32 = 10u;
override FUZZ_PER_TAIL: u32 = 24u;
const FIBER_PTS: u32 = 5u;
// a drawn fibre is a small tuft of a few fibres (single cotton fibres are ~15 um)
const FIBER_WIDTH: f32 = 30e-6;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) tangent: vec3<f32>,
  @location(2) alpha: f32,
  @location(3) @interpolate(flat) id: vec4<u32>, // yarn (bitcast), family, layer, seed
  @location(4) wearV: f32,
};

fn fiber(y: YarnPoint, seed: u32, vi: u32, tailEnd: bool) -> VOut {
  // root on the yarn surface at a random azimuth; the fibre leaves mostly outward, partly along the yarn
  let az = 6.2831853 * hashf(seed + 1u);
  let u1 = normalize(cross(y.t, y.n));
  let u2 = cross(u1, y.t);
  let radial = cos(az) * u1 + sin(az) * u2;
  let root = y.p + radial * y.radius;
  let tilt = (hashf(seed + 2u) - 0.5) * 2.2;
  var dir = normalize(radial + y.t * tilt);
  // length: log-normal-ish, longer and more aligned at frayed ends
  let r = hashf(seed + 3u);
  var len = 0.12e-3 + 0.75e-3 * r * r;
  if (tailEnd) {
    len *= 3.2;
    dir = normalize(dir + y.t * 1.2);
  }
  let bend = vec3<f32>(hashf(seed + 4u) - 0.5, hashf(seed + 5u) - 0.5, hashf(seed + 6u) - 0.8) * (0.9 * len);
  let k = vi >> 1u;
  let t = f32(k) / f32(FIBER_PTS - 1u);
  let P = root + dir * (len * t) + bend * (t * t);
  let dP = normalize(dir * len + bend * (2.0 * t));
  let sideSign = select(-1.0, 1.0, (vi & 1u) == 1u);
  let V = normalize(F.camPos.xyz - P);
  var B = cross(dP, V);
  let bl = length(B);
  if (bl < 1e-6) { B = u1; } else { B = B / bl; }
  let pixW = F.camPos.w * length(F.camPos.xyz - P);
  let halfW = max(0.5 * FIBER_WIDTH, 0.5 * pixW);
  var o: VOut;
  let W = P + sideSign * halfW * B;
  o.pos = F.viewProj * vec4<f32>(W, 1.0);
  o.world = W;
  o.tangent = dP;
  // coverage of a sub-pixel fibre, thinning toward its free tip
  o.alpha = clamp(FIBER_WIDTH / max(pixW, 1e-9), 0.0, 1.0) * (1.0 - 0.6 * t);
  o.id = vec4<u32>(bitcast<u32>(y.yarn), y.fam, y.layer, seed);
  o.wearV = y.wear;
  return o;
}

fn hidden() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

@vertex
fn vsYarnFuzz(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let b = bundleMax(R.layers[0]);
  let yi = ii / FUZZ_PER_YARN;
  let k = yi / (2u * b);
  let s = spanEntry(k);
  let q = (yi % (2u * b)) >> 1u;
  let h = yi & 1u;
  let bs = bundleOf(R.layers[nodes[s >> 1u].ijl >> 24u], s & 1u);
  if (q >= bs || yarnInSurface(k, q, bs)) { return hidden(); }
  let seed = hash11(s * 977u + q * 131u + h * 53u + (ii % FUZZ_PER_YARN) * 7919u);
  // on a broken yarn the fibres concentrate toward its frayed end
  let broken = yarnBroken(s, q);
  let r = hashf(seed);
  let u = select(r, 1.0 - r * r * 0.85, broken);
  return fiber(halfYarn(s, q, h, u), seed, vi, broken && u > 0.6);
}

@vertex
fn vsTailFuzz(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let b = bundleMax(R.layers[0]);
  let yi = ii / FUZZ_PER_TAIL;
  let e = tailEntry(yi / b);
  let q = yi % b;
  if (q >= bundleOf(R.layers[nodes[e >> 2u].ijl >> 24u], (e & 3u) >> 1u)) { return hidden(); }
  let seed = hash11(e * 613u + q * 137u + (ii % FUZZ_PER_TAIL) * 6007u + 3u);
  // fibres concentrate toward the untwisted free end
  let r = hashf(seed);
  let t = 1.0 - r * r * 0.8;
  return fiber(tailYarn(e, q, t), seed, vi, true);
}

@fragment
fn fsFuzz(in: VOut) -> @location(0) vec4<f32> {
  let LR = R.layers[in.id.z];
  let V = normalize(F.camPos.xyz - in.world);
  let T = normalize(in.tangent);
  let N = normalize(V - dot(V, T) * T);
  // loose fibres are mostly the pale core and abraded surface fibres
  let pale = hashf(in.id.w + 9u) < 0.7;
  var albedo = vec3<f32>(0.46, 0.5, 0.58);
  if (in.id.y == 0u) { albedo = vec3<f32>(0.55, 0.52, 0.46); }
  if (!pale && in.id.y == 1u) { albedo = vec3<f32>(0.06, 0.1, 0.24); }
  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, N);
  let amb = F.ambient.rgb;
  let c = fiberShade(N, T, V, key.xyz, albedo, 0.2) * (key.w * sh) + amb * albedo;
  return vec4<f32>(c, in.alpha);
}
