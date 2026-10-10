// A shadow map prefiltered for the floor (floor.wgsl): per texel, the fraction of the shadow map's
// texels it spans that hold cloth (everything drawn there is cloth above the floor), and the mean
// cloth depth weighted the same way, then a box-filtered mip chain of both. Both are linear in the
// covered area, so any level is the exact average over its texels.

struct CovU {
  // shadow-map texels per coverage texel along each side
  block: vec4<u32>,
};

@group(0) @binding(0) var<uniform> U: CovU;
@group(0) @binding(1) var depthIn: texture_depth_2d;
@group(0) @binding(2) var covIn: texture_2d<f32>;
@group(0) @binding(3) var covOut: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var gatherSampler: sampler;

@compute @workgroup_size(8, 8)
fn fromDepth(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(covOut);
  if (any(gid.xy >= dims)) { return; }
  let B = U.block.x;
  let texel = 1.0 / vec2<f32>(textureDimensions(depthIn));
  var c = 0.0;
  var cd = 0.0;
  // 2 x 2 texels per gather (sampled at their shared corner)
  for (var j = 0u; j < B; j += 2u) {
    for (var i = 0u; i < B; i += 2u) {
      let d = textureGather(depthIn, gatherSampler, vec2<f32>(f32(gid.x * B + i + 1u), f32(gid.y * B + j + 1u)) * texel);
      let m = select(vec4<f32>(0.0), vec4<f32>(1.0), d < vec4<f32>(1.0));
      c += dot(m, vec4<f32>(1.0));
      cd += dot(m, d);
    }
  }
  let inv = 1.0 / f32(B * B);
  textureStore(covOut, gid.xy, vec4<f32>(c * inv, cd * inv, 0.0, 0.0));
}

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(covOut);
  if (any(gid.xy >= dims)) { return; }
  let p = gid.xy * 2u;
  let s = textureLoad(covIn, p, 0) + textureLoad(covIn, p + vec2<u32>(1u, 0u), 0) + textureLoad(covIn, p + vec2<u32>(0u, 1u), 0) + textureLoad(covIn, p + vec2<u32>(1u, 1u), 0);
  textureStore(covOut, gid.xy, 0.25 * s);
}
