// Floor material textures at load: mip levels by a bilinear tap at the centre of each 2 x 2 block of
// the level above (the box average, filtered in linear space for sRGB views), and the roughness and
// ambient occlusion maps packed into one texture.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src2: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;

struct FsIn {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> FsIn {
  let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: FsIn;
  o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
  return o;
}

@fragment
fn fsDown(in: FsIn) -> @location(0) vec4<f32> {
  // the level below is half the size, rounded down (odd sizes: the tap stays at the texel's place)
  let uv = in.pos.xy / vec2<f32>(max(textureDimensions(src) / 2u, vec2<u32>(1u)));
  return textureSampleLevel(src, smp, uv, 0.0);
}

@fragment
fn fsPack(in: FsIn) -> @location(0) vec4<f32> {
  let p = vec2<i32>(in.pos.xy);
  return vec4<f32>(textureLoad(src, p, 0).r, textureLoad(src2, p, 0).r, 0.0, 1.0);
}
