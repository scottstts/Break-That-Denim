// The mount the coupon is stretched in: an opaque board in the cloth's plane, just beneath it, from
// under the pinned collar outward. It shades the floor beneath the cloth from the lights (they would
// otherwise reach under the coupon's edge) and is seen only past the coupon's sides; a dark matte
// surface lit like the floor's lights, with the cloth's shadows.

@group(0) @binding(0) var<uniform> F: FrameU;
// inner (under the collar) corners 0..3, outer corners 4..7, in order around the coupon
@group(0) @binding(1) var<storage, read> corners: array<vec4<f32>>;
@group(0) @binding(2) var shadowMap: texture_depth_2d;
@group(0) @binding(3) var shadowSampler: sampler_comparison;
@group(0) @binding(4) var overMap: texture_depth_2d;

const MOUNT_ALBEDO: vec3<f32> = vec3<f32>(0.045, 0.045, 0.047);

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
};

// One side of the ring per pair of triangles: inner k, inner k+1, outer k+1, outer k.
fn mountCorner(vi: u32) -> vec3<f32> {
  let side = vi / 6u;
  var q = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
  let c = q[vi % 6u];
  let k = (side + select(0u, 1u, c == 1u || c == 2u)) % 4u;
  return corners[select(k + 4u, k, c < 2u)].xyz;
}

@vertex
fn vsMount(@builtin(vertex_index) vi: u32) -> VOut {
  let W = mountCorner(vi);
  var o: VOut;
  o.world = W;
  o.pos = F.viewProj * vec4<f32>(W, 1.0);
  return o;
}

@vertex
fn vsMountShadow(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  return F.lightViewProj * vec4<f32>(mountCorner(vi), 1.0);
}

@fragment
fn fsMount(in: VOut) -> @location(0) vec4<f32> {
  let N = vec3<f32>(0.0, 0.0, 1.0);
  let key = keyLight(in.world);
  let a = MOUNT_ALBEDO * (1.0 / PI);
  var c = a * max(key.z, 0.0) * key.w * F.keyColor.rgb * shadowFactor(in.world, N);
  c += a * max(F.overDir.z, 0.0) * F.overColor.rgb * shadowOver(in.world, N);
  let cam = torchLight(in.world);
  c += a * max(cam.z, 0.0) * cam.w * F.torchColor.rgb;
  c += F.ambient.rgb * MOUNT_ALBEDO;
  return vec4<f32>(c, 1.0);
}
