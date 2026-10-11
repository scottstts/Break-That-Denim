// Resolve HDR to the display: one tone map and one output encoding. A staged experience adds the
// camera's film: a grade, grain and vignette, and digital signal corruption (glitch).

struct PostU {
  // staged grade amount, film grain, vignette, time (s)
  grade: vec4<f32>,
  // glitch amount, its seed (held over a few frames), unused
  glitch: vec4<f32>,
};

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var hdr: texture_2d<f32>;
@group(0) @binding(2) var<uniform> PU: PostU;

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

// ACES fitted curve (Narkowicz)
fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linearToSrgb(c: vec3<f32>) -> vec3<f32> {
  let lo = c * 12.92;
  let hi = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn hdrAt(p: vec2<f32>) -> vec3<f32> {
  let dims = vec2<i32>(textureDimensions(hdr));
  return textureLoad(hdr, clamp(vec2<i32>(floor(p)), vec2<i32>(0), dims - 1), 0).rgb;
}

fn h2(a: f32, b: f32) -> f32 { return hash2f(u32(i32(a) + 65536), u32(i32(b) + 65536)); }

// Signal corruption of a digital camera: rows torn sideways in bands, colour planes out of register,
// macroblocks copied from elsewhere and posterised, interlace lines and noise. Returns linear HDR.
fn corrupted(p: vec2<f32>, g: f32, seed: f32) -> vec3<f32> {
  let W = F.viewport.x;
  let H = F.viewport.y;
  let s = seed * 97.0;
  let bandH = mix(4.0, 0.06 * H, h2(s, 1.0));
  let band = floor((p.y + h2(s, 2.0) * bandH) / bandH);
  var q = p;
  let rb = h2(band, s);
  if (rb < 0.55 * g) { q.x += (h2(band, s + 3.0) - 0.5) * W * 0.3 * g; }
  // larger slabs drift as a whole
  let slab = floor(p.y / (0.18 * H));
  if (h2(slab, s + 5.0) < 0.35 * g) { q.x += (h2(slab, s + 6.0) - 0.5) * W * 0.08; q.y += (h2(slab, s + 7.0) - 0.5) * 0.05 * H; }
  // macroblocks copied from another place
  let B = 16.0 * (1.0 + floor(h2(s, 8.0) * 3.0));
  let blk = floor(q / B);
  let mb = h2(blk.x + 311.0 * blk.y, s + 9.0) < 0.12 * g;
  if (mb) { q += (vec2<f32>(h2(blk.x, s + 10.0), h2(blk.y, s + 11.0)) - 0.5) * B * 6.0; }
  let split = (h2(band, s + 12.0) - 0.3) * 14.0 * g * (W / 1600.0);
  var c = vec3<f32>(hdrAt(q + vec2<f32>(split, 0.0)).r, hdrAt(q).g, hdrAt(q - vec2<f32>(split, 0.0)).b);
  if (mb) { c = floor(c * 6.0 / (c + 1.0)) / 4.0; }
  let line = select(1.0, 1.0 - 0.35 * g, (u32(p.y) & 1u) == 1u);
  let noise = hash2f(u32(p.x) + 7919u * u32(seed), u32(p.y) + 104729u * u32(F.params.w));
  return c * line + vec3<f32>(noise * noise * 0.12 * g);
}

@fragment
fn fsTonemap(in: FsIn) -> @location(0) vec4<f32> {
  let staged = PU.grade.x + PU.grade.y + PU.grade.z + PU.glitch.x;
  if (staged == 0.0) {
    let c = textureLoad(hdr, vec2<i32>(in.pos.xy), 0).rgb * F.params.x;
    return vec4<f32>(linearToSrgb(aces(c)), 1.0);
  }
  var lin = hdrAt(in.pos.xy);
  if (PU.glitch.x > 0.0) { lin = corrupted(in.pos.xy, PU.glitch.x, PU.glitch.y); }
  var c = aces(lin * F.params.x);
  // grade: colour drained toward a cold luminance, shadows crushed
  let a = PU.grade.x;
  let y = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  c = mix(c, y * vec3<f32>(0.93, 0.98, 1.04), 0.35 * a);
  c = mix(c, c * c * (3.0 - 2.0 * c), 0.25 * a);
  // vignette of the lens
  let uv = in.pos.xy * F.viewport.zw - 0.5;
  let r = length(uv * vec2<f32>(F.viewport.x * F.viewport.w, 1.0));
  c *= 1.0 - PU.grade.z * smoothstep(0.3, 0.95, r);
  // sensor grain, strongest in the shadows, new every frame
  let n = hash2f(u32(in.pos.x) + 1301u * u32(F.params.w), u32(in.pos.y) * 7u + 31u) - 0.5;
  c += PU.grade.y * n * 0.09 * (1.0 - 0.6 * sqrt(y));
  return vec4<f32>(linearToSrgb(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
