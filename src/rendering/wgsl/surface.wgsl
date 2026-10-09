// Intact cloth: one instance per lattice cell, S x S sub-quads of the refined bicubic surface.
// Cells missing any span are not drawn here (their surviving yarns are drawn explicitly).
// The fragment shader evaluates the woven yarn surface at the four MSAA sample positions: each
// sample is either a yarn (shaded) or a gap in the weave (uncovered), so stretched cloth opens up.

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> rsurf: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cellFlags: array<u32>;
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;
@group(0) @binding(6) var weaveTile: texture_2d_array<f32>;
@group(0) @binding(7) var tileSampler: sampler;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) dPs: vec3<f32>,
  @location(3) dPt: vec3<f32>,
  @location(4) wearV: f32,
  @location(5) @interpolate(flat) layer: u32,
  // cell (i, j) and its missing-yarn masks (see CellYarns)
  @location(6) @interpolate(flat) cell: vec4<u32>,
};

fn degenerate() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

fn surfaceVertex(vi: u32, cell: u32) -> VOut {
  var l = 0u;
  for (var k = 1u; k < R.counts.y; k++) {
    if (cell >= R.layers[k].refine.z) { l = k; }
  }
  let LR = R.layers[l];
  if (cell >= R.counts.z) { return degenerate(); }
  let flags = cellFlags[cell];
  if (flags == 0u) { return degenerate(); }
  let L = LR.lattice;
  let S = LR.twill.z;
  let local = cell - LR.refine.z;
  let ci = local % (L.y - 1u);
  let cj = local / (L.y - 1u);
  let q = vi / 6u;
  let corner = vi % 6u;
  var cu = array<u32, 6>(0u, 1u, 1u, 0u, 1u, 0u);
  var cv = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
  let I = ci * S + (q % S) + cu[corner];
  let J = cj * S + (q / S) + cv[corner];
  let r = LR.refine.x + J * LR.refine.y + I;
  let a = rsurf[3u * r];
  var o: VOut;
  o.world = a.xyz;
  o.wearV = a.w;
  o.dPs = rsurf[3u * r + 1u].xyz;
  o.dPt = rsurf[3u * r + 2u].xyz;
  let bu = f32(bundleOf(LR, 1u));
  let bv = f32(bundleOf(LR, 0u));
  o.uv = vec2<f32>((f32(I) / f32(S)) * bu + 0.5 * (bu - 1.0), (f32(J) / f32(S)) * bv + 0.5 * (bv - 1.0));
  o.layer = l;
  o.cell = vec4<u32>(ci, cj, flags >> 8u, 0u);
  o.pos = F.viewProj * vec4<f32>(a.xyz, 1.0);
  return o;
}

@vertex
fn vsMain(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  return surfaceVertex(vi, ii);
}

@vertex
fn vsShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
  let o = surfaceVertex(vi, ii);
  if (o.pos.z < -1.0) { return o.pos; }
  return F.lightViewProj * vec4<f32>(o.world, 1.0);
}

struct FsOut {
  @location(0) color: vec4<f32>,
  @builtin(sample_mask) mask: u32,
};

@fragment
fn fsMain(in: VOut, @builtin(front_facing) ff: bool) -> FsOut {
  let LR = R.layers[in.layer];
  let bu = i32(bundleOf(LR, 1u));
  let bv = i32(bundleOf(LR, 0u));
  let cy = CellYarns(i32(in.cell.x) * bu, i32(in.cell.y) * bv, bu, bv, i32(yarnsPerStrand(LR, 1u)), i32(yarnsPerStrand(LR, 0u)), in.cell.z);
  let ls = length(in.dPs);
  let lt = length(in.dPt);
  let Tu = in.dPs / ls;
  let Tv0 = in.dPt / lt;
  var Ng = normalize(cross(in.dPs, in.dPt));
  let Tv = normalize(Tv0 - dot(Tv0, Tu) * Tu);
  // local stretch of the yarn spacing relative to rest
  let su = ls / (f32(bundleOf(LR, 1u)) * LR.pitch.x);
  let sv = lt / (f32(bundleOf(LR, 0u)) * LR.pitch.y);
  let Vv = normalize(F.camPos.xyz - in.world);
  let front = dot(Ng, Vv) > 0.0;
  let Nf = select(-Ng, Ng, front);

  let dist = length(F.camPos.xyz - in.world);
  let pix = F.camPos.w * dist;
  let dx = dpdx(in.uv);
  let dy = dpdy(in.uv);
  let fy = max(length(dx), length(dy));
  // yarns near the pixel size: the prefiltered tile shades the weave. Cells with missing yarns still
  // evaluate the exact weave per sample for their coverage (the gaps are the damage) and for which
  // family shows, so both kinds of cell share one appearance
  let intactCell = in.cell.z == 0u;
  let lodTile = smoothstep(0.28, 0.55, fy);
  let lod = select(smoothstep(0.45, 1.1, fy), 0.0, intactCell);
  var warpTop = 0.0;

  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, Nf);
  let amb = F.ambient.rgb * (0.55 + 0.45 * Nf.z);

  // 4x MSAA standard sample offsets (pixel units, y down); each also jittered over the hairiness
  // radius (~45 um), which blurs the weave the way loose fibres do (and fills sub-fuzz gaps)
  var offs = array<vec2<f32>, 4>(vec2<f32>(-0.125, -0.375), vec2<f32>(0.375, -0.125), vec2<f32>(-0.375, 0.125), vec2<f32>(0.125, 0.375));
  let fuzzR = vec2<f32>(45e-6 / (LR.pitch.x * su), 45e-6 / (LR.pitch.y * sv));
  let rotSeed = hash11(u32(in.pos.x) * 7919u + u32(in.pos.y));
  let rot = hashf(rotSeed) * 6.2831853;
  // weave below the pixel footprint: coverage left by the cell's missing yarns (warp-faced: a missing
  // warp exposes the picks beneath, which still cover most of their area)
  let missW = f32(countOneBits(in.cell.z & 0xFFu) * yarnsPerStrand(LR, 1u)) / f32(2 * bu);
  let missP = f32(countOneBits(in.cell.z & 0xFF00u) * yarnsPerStrand(LR, 0u)) / f32(2 * bv);
  let lodCover = 1.0 - missW * (0.3 + 0.7 * missP);
  let hTop = LR.warp.z + 0.5 * LR.warp.y;
  let sgn = select(-1.0, 1.0, front);
  // family-mean albedos for the softly blended second yarn
  let warpMean = vec3<f32>(0.045, 0.065, 0.135);
  let weftMean = vec3<f32>(0.26, 0.27, 0.28);
  // per sample: woven geometry and appearance; lighting once per pixel on the averaged inputs
  // (Lambert is linear in the normal; the weak specular lobe is the only approximation)
  var albDirect = vec3<f32>(0.0);
  var albAmb = vec3<f32>(0.0);
  var nSum = vec3<f32>(0.0);
  var tSum = vec3<f32>(0.0);
  var fuzz = 0.0;
  var groove = 0.0;
  var cnt = 0.0;
  var mask = 0u;
  for (var k = 0u; k < 4u; k++) {
    if (lodTile >= 1.0 && intactCell) { break; }
    let ang = rot + f32(k) * 2.39996;
    let jr = sqrt((f32(k) + 0.5) / 4.0);
    let uvk = in.uv + dx * offs[k].x + dy * offs[k].y + vec2<f32>(cos(ang), sin(ang)) * jr * fuzzR;
    let w2 = weave2(LR, uvk.x, uvk.y, su, sv, front, 0.018e-3, cy);
    if (w2.a.covered || (lod >= 0.99 && hashf(rotSeed + k) < lodCover)) { mask |= (1u << k); }
    if (!w2.a.covered) { continue; }
    let hit = w2.a;
    let wb = w2.wb;
    warpTop += select(0.0, 1.0, hit.fam == 1u) * (1.0 - wb) + select(0.0, 1.0, w2.b.covered && w2.b.fam == 1u) * wb;
    let look = yarnLook(LR, in.layer, hit.fam, hit.yarn, hit.float_, hit.along, hit.x, in.wearV, pix);
    var albedo = look.albedo;
    var slope = hit.slope;
    var h = hit.h;
    if (wb > 0.02) {
      albedo = mix(albedo, select(weftMean, warpMean, w2.b.fam == 1u) * (1.0 - 0.4 * in.wearV) + vec3<f32>(0.12) * in.wearV, wb);
      slope = mix(slope, w2.b.slope, wb);
      h = mix(h, w2.b.h, wb);
    }
    let Ny = normalize(Nf - sgn * (slope.x * Tu + slope.y * Tv));
    var axis = select(Tu, Tv, hit.fam == 1u);
    let slopeAlong = select(slope.x, slope.y, hit.fam == 1u);
    axis = normalize(axis + sgn * slopeAlong * Nf);
    let tw = select(LR.weft.w, LR.warp.w, hit.fam == 1u);
    let hn = clamp((sgn * h + hTop) / (2.0 * hTop), 0.0, 1.0);
    // interstices between floats (where the weft shows) are deep and shadowed by the neighbours
    let ao = weaveOcclusion(hn);
    albDirect += albedo * mix(0.55, 1.0, ao);
    albAmb += albedo * ao;
    nSum += Ny;
    tSum += cos(tw) * axis + sin(tw) * cross(Ny, axis);
    fuzz += look.fuzz;
    groove += 1.0 - hn;
    cnt += 1.0;
  }
  var acc = vec3<f32>(0.0);
  if (cnt > 0.0) {
    let inv = 1.0 / cnt;
    let Nl = normalize(nSum);
    let Tl = normalize(tSum - dot(tSum, Nl) * Nl + 1e-6 * Tu);
    acc = fiberShade(Nl, Tl, Vv, key.xyz, albDirect * inv, fuzz * inv) * (key.w * sh) + amb * albAmb * inv;
    // loose fibres bridging the grooves: a faint haze lifts the deepest points
    acc = mix(acc, (key.w * sh * max(dot(Nf, key.xyz), 0.0) * 0.06 + amb * 0.6) * vec3<f32>(0.32, 0.36, 0.48), 0.12 * groove * inv * fuzz * inv);
    acc *= cnt;
  }
  var color = acc / max(cnt, 1.0);
  if (lodTile > 0.0) {
    let rep = f32(LR.twill.x);
    let tex = textureSampleGrad(weaveTile, tileSampler, in.uv / rep, in.layer, dx / rep, dy / rep);
    // the back of a warp-faced twill shows mostly weft; a damaged cell shows what its samples hit
    let wfTile = select(1.0 - tex.x, tex.x, front);
    let wf = select(warpTop / max(cnt, 1.0), wfTile, intactCell || cnt == 0.0);
    let Ny = normalize(Nf - sgn * (tex.z * Tu + tex.w * Tv));
    let k = i32(round(in.uv.x));
    let m = i32(round(in.uv.y));
    let fl = i32(round((in.uv.y - warpUnderDist(LR, k, in.uv.y)) / rep));
    let lw = yarnLook(LR, in.layer, 1u, k, fl, in.uv.y * LR.pitch.y, 0.3, in.wearV, pix);
    let lf = yarnLook(LR, in.layer, 0u, m, 0, in.uv.x * LR.pitch.x, 0.3, in.wearV, pix);
    let albedo = mix(lf.albedo, lw.albedo, wf);
    let ao = tex.y;
    let tw = mix(LR.weft.w, LR.warp.w, wf);
    let Ta = select(Tu, Tv, wf > 0.5);
    let Tf = normalize(cos(tw) * Ta + sin(tw) * cross(Ny, Ta));
    let fz = mix(lf.fuzz, lw.fuzz, wf);
    var ct = fiberShade(Ny, Tf, Vv, key.xyz, albedo * mix(0.55, 1.0, ao), fz) * (key.w * sh) + amb * albedo * ao;
    ct = mix(ct, (key.w * sh * max(dot(Nf, key.xyz), 0.0) * 0.06 + amb * 0.6) * vec3<f32>(0.32, 0.36, 0.48), 0.12 * (1.0 - ao) * fz);
    color = select(mix(color, ct, lodTile), ct, cnt == 0.0 || (!intactCell && lodTile >= 1.0));
    if (intactCell) { mask = 0xFu; }
  }
  if (lod > 0.0) {
    // filtered appearance once the weave is below the pixel footprint
    let mw = yarnLook(LR, in.layer, 1u, i32(round(in.uv.x)), 0, in.uv.y * LR.pitch.y, 0.4, in.wearV, 1.0);
    let mf = yarnLook(LR, in.layer, 0u, i32(round(in.uv.y)), 0, in.uv.x * LR.pitch.x, 0.4, in.wearV, 1.0);
    let albedo = select(mix(mw.albedo, mf.albedo, 0.14), mix(mf.albedo, mw.albedo, 0.25), !front);
    let Tf = select(Tu, Tv, front);
    let cm = fiberShade(Nf, Tf, Vv, key.xyz, albedo, mw.fuzz) * (key.w * sh) * 0.85 + amb * albedo * 0.8;
    color = mix(color, cm, lod);
  }
  if (mask == 0u) { discard; }
  var o: FsOut;
  o.color = vec4<f32>(color, 1.0);
  o.mask = mask;
  return o;
}
