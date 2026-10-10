// Prefiltered weave tile: one twill repeat (R warps x R picks, unstretched, face side) per layer,
// with every mip level computed directly as the average of the analytic weave over the texel's
// footprint. The surface shader samples it with hardware anisotropic filtering wherever the yarns
// approach the pixel size, so the twill keeps its true filtered structure instead of aliasing.
// Texel: (warp-on-top fraction among covered points, occlusion x coverage, mean dh/du, mean dh/dv).

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<uniform> Lvl: vec4<u32>; // x: layer, y: texel count per side at this level
@group(0) @binding(2) var tileOut: texture_storage_2d<rgba16float, write>;

const SUB: u32 = 8u;

@compute @workgroup_size(8, 8)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let size = Lvl.y;
  if (gid.x >= size || gid.y >= size) { return; }
  let L = R.layers[Lvl.x];
  let rep = f32(L.twill.x);
  let texel = rep / f32(size);
  let hTop = L.warp.z + 0.5 * L.warp.y;
  var wf = 0.0;
  var aoc = 0.0;
  var slope = vec2<f32>(0.0);
  var covered = 0.0;
  for (var a = 0u; a < SUB; a++) {
    for (var b = 0u; b < SUB; b++) {
      // stratified samples over the texel (at coarse levels the texel spans several yarns)
      let U = (f32(gid.x) + (f32(a) + 0.5) / f32(SUB)) * texel;
      let V = (f32(gid.y) + (f32(b) + 0.5) / f32(SUB)) * texel;
      let w2 = weave2(L, U, V, 1.0, 1.0, true, 0.018e-3);
      if (!w2.a.covered) { continue; }
      var top = select(0.0, 1.0, w2.a.fam == 1u);
      var s = w2.a.slope;
      var h = w2.a.h;
      if (w2.b.covered) {
        top = mix(top, select(0.0, 1.0, w2.b.fam == 1u), w2.wb);
        s = mix(s, w2.b.slope, w2.wb);
        h = mix(h, w2.b.h, w2.wb);
      }
      let hn = clamp((h + hTop) / (2.0 * hTop), 0.0, 1.0);
      wf += top;
      aoc += weaveOcclusion(hn);
      slope += s;
      covered += 1.0;
    }
  }
  let n = f32(SUB * SUB);
  textureStore(tileOut, vec2<i32>(gid.xy), vec4<f32>(wf / max(covered, 1.0), aoc / n, slope / n));
}
