// Edge field: distance (mm, up to EDGE_RMAX) from each rest-chart point to the nearest crack, drawn
// into an r16float texture by min-blending one quad per crack (its bounding box grown by EDGE_RMAX).
// Redrawn only when the cracks change (edgeArgs in torn.wgsl); a full-target clear precedes a redraw.

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> tornGeo: array<vec4<f32>>;

@vertex
fn vsClear(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fsClear() -> @location(0) vec4<f32> { return vec4<f32>(EDGE_RMAX, 0.0, 0.0, 1.0); }

struct EOut {
  @builtin(position) pos: vec4<f32>,
  // rest-chart point (m)
  @location(0) rest: vec2<f32>,
  // crack control points (rest chart, m)
  @location(1) @interpolate(flat) b01: vec4<f32>,
  @location(2) @interpolate(flat) b23: vec4<f32>,
  @location(3) @interpolate(flat) maturity: f32,
};

@vertex
fn vsCrack(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> EOut {
  var o: EOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  let k = ii >> 2u;
  let i = ii & 3u;
  let base = k * TORN_STRIDE;
  let info = tornInfo(tornGeo[base]);
  if (i >= crackCountOf(info.y)) { return o; }
  let LR = R.layers[info.w];
  let size = vec2<f32>(f32(bundleOf(LR, 1u)) * LR.pitch.x, f32(bundleOf(LR, 0u)) * LR.pitch.y);
  let local = info.x - LR.refine.z;
  let cw = LR.lattice.y - 1u;
  let origin = vec2<f32>(f32(local % cw), f32(local / cw)) * size;
  let c01 = tornGeo[base + TORN_CRACKS + 2u * i] + vec4<f32>(origin, origin);
  let c23 = tornGeo[base + TORN_CRACKS + 2u * i + 1u] + vec4<f32>(origin, origin);
  let r = EDGE_RMAX * 1e-3;
  let lo = min(min(c01.xy, c01.zw), min(c23.xy, c23.zw)) - vec2<f32>(r);
  let hi = max(max(c01.xy, c01.zw), max(c23.xy, c23.zw)) + vec2<f32>(r);
  var cx = array<f32, 6>(0.0, 1.0, 1.0, 0.0, 1.0, 0.0);
  var cy = array<f32, 6>(0.0, 0.0, 1.0, 0.0, 1.0, 1.0);
  let p = mix(lo, hi, vec2<f32>(cx[vi], cy[vi]));
  let tx = edgeTexel(LR, p) / LR.edge.zw;
  o.pos = vec4<f32>(2.0 * tx.x - 1.0, 1.0 - 2.0 * tx.y, 0.0, 1.0);
  o.rest = p;
  o.b01 = c01;
  o.b23 = c23;
  o.maturity = crackMaturity(info.y);
  return o;
}

@fragment
fn fsCrack(in: EOut) -> @location(0) vec4<f32> {
  let d = bezierDistance(in.b01, in.b23, in.rest);
  return vec4<f32>(min(d * 1e3 / in.maturity, EDGE_RMAX), 0.0, 0.0, 1.0);
}
