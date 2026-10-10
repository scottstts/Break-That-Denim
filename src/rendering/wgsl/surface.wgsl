// Woven cloth surface. Intact cells: one instance per lattice cell, S x S sub-quads of the refined
// bicubic surface. Torn cells (a separated span) are drawn per corner group (vsTorn): the group's
// leader draws the whole cell on the group's own control net (torn.wgsl) and keeps the part on its
// side of the cracks, so the free edge is a smooth crack curve, not a lattice line.
// The fragment shader evaluates the woven yarn surface at the four MSAA sample positions: each sample
// is either a yarn (shaded) or a gap in the weave (uncovered), so stretched cloth opens up. Near a free
// edge (edge field) the cloth turns into its torn rim: an abraded halo, then a pale, fibrous, slightly
// rolled band whose yarn ends fringe out across the crack.

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> rsurf: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cellFlags: array<u32>;
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;
@group(0) @binding(6) var weaveTile: texture_2d_array<f32>;
@group(0) @binding(7) var tileSampler: sampler;
@group(0) @binding(8) var<storage, read> tornGeo: array<vec4<f32>>;
@group(0) @binding(9) var edgeTex: texture_2d<f32>;
@group(0) @binding(10) var edgeSampler: sampler;

// sub-quads per edge of a torn group's patch (over the whole cell)
const TQ: u32 = 4u;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) dPs: vec3<f32>,
  @location(3) dPt: vec3<f32>,
  @location(4) wearV: f32,
  @location(5) @interpolate(flat) layer: u32,
  // cell (i, j), strand masks of its spans (bits 0-15, surface cells), unused
  @location(6) @interpolate(flat) cell: vec4<u32>,
  // torn groups: 1, broken edges, group (a | b << 2), packed crack crossings; x = 0 for surface cells
  @location(7) @interpolate(flat) torn: vec4<u32>,
  // torn groups: the group's free-edge curves (groupCurves, metres from the cell's c0)
  @location(8) @interpolate(flat) c1lo: vec4<f32>,
  @location(9) @interpolate(flat) c1hi: vec4<f32>,
  @location(10) @interpolate(flat) c2lo: vec4<f32>,
  @location(11) @interpolate(flat) c2hi: vec4<f32>,
};

fn degenerate() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

fn layerOfCell(cell: u32) -> u32 {
  var l = 0u;
  for (var k = 1u; k < R.counts.y; k++) {
    if (cell >= R.layers[k].refine.z) { l = k; }
  }
  return l;
}

fn surfaceVertex(vi: u32, cell: u32) -> VOut {
  let l = layerOfCell(cell);
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
  o.cell = vec4<u32>(ci, cj, (flags >> 8u) & 0xFFFFu, 0u);
  o.torn = vec4<u32>(0u);
  o.pos = F.viewProj * vec4<f32>(a.xyz, 1.0);
  return o;
}

// ---------------------------------------------------------------- torn groups

fn crS(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * (2.0 * p1 + (p2 - p0) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
}
fn crdS(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  return 0.5 * ((p2 - p0) + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t + 3.0 * (3.0 * p1 - p0 - 3.0 * p2 + p3) * t * t);
}

fn tornVertex(vi: u32, ii: u32) -> VOut {
  let k = ii >> 2u;
  let c = ii & 3u;
  let base = k * TORN_STRIDE;
  let slot = tornSlot(k, c);
  if (tornGeo[slot].w < 0.0) { return degenerate(); }
  let info = tornInfo(tornGeo[base]);
  let l = info.w;
  let LR = R.layers[l];
  let L = LR.lattice;
  let local = info.x - LR.refine.z;
  let ci = local % (L.y - 1u);
  let cj = local / (L.y - 1u);
  let q = vi / 6u;
  let k6 = vi % 6u;
  var cuo = array<u32, 6>(0u, 1u, 1u, 0u, 1u, 0u);
  var cvo = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
  let s = f32(q % TQ + cuo[k6]) / f32(TQ);
  let t = f32(q / TQ + cvo[k6]) / f32(TQ);
  var rows: array<vec3<f32>, 4>;
  var drows: array<vec3<f32>, 4>;
  for (var r = 0u; r < 4u; r++) {
    let p0 = tornGeo[slot + 4u * r].xyz;
    let p1 = tornGeo[slot + 4u * r + 1u].xyz;
    let p2 = tornGeo[slot + 4u * r + 2u].xyz;
    let p3 = tornGeo[slot + 4u * r + 3u].xyz;
    rows[r] = crS(p0, p1, p2, p3, s);
    drows[r] = crdS(p0, p1, p2, p3, s);
  }
  let W = crS(rows[0], rows[1], rows[2], rows[3], t);
  var o: VOut;
  o.world = W;
  o.dPs = crS(drows[0], drows[1], drows[2], drows[3], t);
  o.dPt = crdS(rows[0], rows[1], rows[2], rows[3], t);
  // wear from the cell's refined corner samples
  let S = LR.twill.z;
  let r00 = LR.refine.x + (cj * S) * LR.refine.y + ci * S;
  let w00 = rsurf[3u * r00].w;
  let w10 = rsurf[3u * (r00 + S)].w;
  let w01 = rsurf[3u * (r00 + S * LR.refine.y)].w;
  let w11 = rsurf[3u * (r00 + S * LR.refine.y + S)].w;
  o.wearV = mix(mix(w00, w10, s), mix(w01, w11, s), t);
  let bu = f32(bundleOf(LR, 1u));
  let bv = f32(bundleOf(LR, 0u));
  o.uv = vec2<f32>((f32(ci) + s) * bu + 0.5 * (bu - 1.0), (f32(cj) + t) * bv + 0.5 * (bv - 1.0));
  o.layer = l;
  o.cell = vec4<u32>(ci, cj, 0u, 0u);
  o.torn = vec4<u32>(1u, info.y, groupOf(info.y, ccwCorner(c)), info.z);
  o.c1lo = tornGeo[slot + 16u];
  o.c1hi = tornGeo[slot + 17u];
  o.c2lo = tornGeo[slot + 18u];
  o.c2hi = tornGeo[slot + 19u];
  o.pos = F.viewProj * vec4<f32>(W, 1.0);
  return o;
}

fn cellSize(LR: LayerR) -> vec2<f32> {
  return vec2<f32>(f32(bundleOf(LR, 1u)) * LR.pitch.x, f32(bundleOf(LR, 0u)) * LR.pitch.y);
}

// Signed distance (m) from a fragment of a torn group to the group's free edge.
fn tornDistance(in: VOut, LR: LayerR) -> f32 {
  let size = cellSize(LR);
  let p = restOfUV(LR, in.uv) - vec2<f32>(f32(in.cell.x), f32(in.cell.y)) * size;
  return groupSd(in.torn.y, in.torn.z, size, in.c1lo, in.c1hi, in.c2lo, in.c2hi, p);
}

@vertex
fn vsTorn(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  return tornVertex(vi, ii);
}

@vertex
fn vsTornShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o = tornVertex(vi, ii);
  if (o.pos.z < -1.0) { return o; }
  o.pos = F.lightViewProj * vec4<f32>(o.world, 1.0);
  return o;
}

// Torn groups cast shadows only where they keep material.
@fragment
fn fsTornShadow(in: VOut) {
  if (tornDistance(in, R.layers[in.layer]) < -0.2e-3) { discard; }
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

// ---------------------------------------------------------------- torn rim

// Shape of the rim at one point: how far it reaches (mm), and the local response at distance d.
struct Rim {
  // 0 intact .. 1 at the free edge
  rim: f32,
  // abraded halo around the damage, 0 .. 1
  halo: f32,
  // lip slope (height change per unit distance from the edge, dimensionless) and its direction in the
  // rest chart (unit, pointing away from the edge)
  slope: f32,
  dir: vec2<f32>,
  // occlusion of the rolled edge
  ao: f32,
};

// d: distance to the free edge (mm, divided by the crack's maturity), dir: unit rest-chart gradient
// of d, rest: rest point (m).
fn rimAt(d: f32, dir: vec2<f32>, rest: vec2<f32>, layer: u32) -> Rim {
  var o: Rim;
  if (d >= EDGE_RMAX) {
    o.rim = 0.0;
    o.halo = 0.0;
    o.slope = 0.0;
    o.dir = dir;
    o.ao = 1.0;
    return o;
  }
  // the rim's width wanders along the edge: tufted stretches next to nearly bare ones
  let n = 0.65 * noise2(4441u + layer, rest / 0.0065) + 0.35 * noise2(4447u + layer, rest / 0.0022);
  let w = clamp(3.0 + 2.2 * n, 1.0, 5.5);
  let x = d / w;
  o.rim = 1.0 - smoothstep(0.0, 1.0, x);
  o.halo = 1.0 - smoothstep(0.0, EDGE_RMAX, d);
  // the loosened edge rolls up a little: height H x/x0 exp(1 - x/x0), peaking a third into the rim
  let x0 = 0.32;
  let H = min(0.28 * w, 1.0);
  let ex = exp(1.0 - x / x0);
  o.slope = select(0.0, (H / (w * x0)) * ex * (1.0 - x / x0), x < 4.0);
  o.dir = dir;
  // the fold where the roll meets the flat cloth is partly shaded; its outer wall is loose, lit fibre
  o.ao = mix(0.88, 1.0, smoothstep(0.0, 0.22, x)) * mix(1.0, 0.86, exp(-pow((x - 0.75) / 0.25, 2.0)));
  return o;
}

// The rim's loosened, untwisted fibres at a rest point: their albedo (undyed weft and pale warp cores
// with some dyed fibre) in rgb and how densely they cover the weave in w. Fibre tufts are drawn out
// across the edge (dir: away from it), so the band is streaked like a fringe, with gaps where the
// dyed weave still shows.
fn rimFibres(rest: vec2<f32>, dir: vec2<f32>, layer: u32, pix: f32) -> vec4<f32> {
  let fine = clamp(1.6 - pix / 0.2e-3, 0.0, 1.0);
  let q = vec2<f32>(dot(rest, dir), dot(rest, vec2<f32>(-dir.y, dir.x)));
  let streak = noise2(5003u + layer, q / vec2<f32>(1.4e-3, 0.22e-3));
  let tuft = noise2(5009u + layer, rest / 0.6e-3);
  let clump = noise2(5021u + layer, rest / 2.2e-3);
  let blue = 0.5 + 0.5 * noise2(5011u + layer, rest / 1.6e-3);
  let pale = vec3<f32>(0.52, 0.53, 0.56);
  let dyed = vec3<f32>(0.16, 0.21, 0.36);
  let alb = mix(pale, dyed, 0.45 * blue * blue) * (0.88 + 0.22 * (0.6 * streak + 0.4 * tuft) * fine);
  let cover = clamp(0.72 + 0.25 * clump + 0.35 * (0.65 * streak + 0.35 * tuft) * fine, 0.0, 1.0);
  return vec4<f32>(alb, cover);
}

@fragment
fn fsMain(in: VOut, @builtin(front_facing) ff: bool) -> FsOut {
  let LR = R.layers[in.layer];
  let torn = in.torn.x != 0u;
  let rest = restOfUV(LR, in.uv);
  // signed distance to this group's free edge (m); intact cells are far inside
  var sd = 1.0;
  var maturity = 1.0;
  if (torn) {
    sd = tornDistance(in, LR);
    maturity = crackMaturity(in.torn.y);
  }
  let sdx = dpdx(sd);
  let sdy = dpdy(sd);
  let dx = dpdx(in.uv);
  let dy = dpdy(in.uv);
  // beyond the farthest yarn end of this group's free edge
  if (sd < -1.3e-3) { discard; }

  // distance to the nearest free edge (mm) from the edge field and its rest-chart gradient (central
  // differences): one continuous field shades the rim across torn and intact cells alike, while the
  // exact crack distance above only decides coverage
  let et = edgeTexel(LR, rest);
  let tsz = LR.edge.zw;
  let dEdge = textureSampleLevel(edgeTex, edgeSampler, et / tsz, 0.0).x;
  var gEdge = vec2<f32>(0.0);
  if (dEdge < EDGE_RMAX) {
    let h = 1.5;
    let gx = textureSampleLevel(edgeTex, edgeSampler, (et + vec2<f32>(h, 0.0)) / tsz, 0.0).x - textureSampleLevel(edgeTex, edgeSampler, (et - vec2<f32>(h, 0.0)) / tsz, 0.0).x;
    let gy = textureSampleLevel(edgeTex, edgeSampler, (et + vec2<f32>(0.0, h)) / tsz, 0.0).x - textureSampleLevel(edgeTex, edgeSampler, (et - vec2<f32>(0.0, h)) / tsz, 0.0).x;
    gEdge = vec2<f32>(gx, gy);
  }
  let gl = length(gEdge);
  let dirEdge = select(vec2<f32>(0.0, 1.0), gEdge / gl, gl > 1e-12);
  let rim = rimAt(dEdge, dirEdge, rest, in.layer);
  // broken strands inside an intact cell: a damaged, fuzzier spot, never a see-through gap
  let damage = f32(countOneBits(in.cell.z)) / 16.0;

  var wearF = min(1.0, in.wearV + 0.4 * rim.halo * rim.halo + 0.3 * damage);
  let ls = length(in.dPs);
  let lt = length(in.dPt);
  let Tu = in.dPs / ls;
  let Tv0 = in.dPt / lt;
  let Ng = normalize(cross(in.dPs, in.dPt));
  let Tv = normalize(Tv0 - dot(Tv0, Tu) * Tu);
  // local stretch of the yarn spacing relative to rest
  let su = ls / (f32(bundleOf(LR, 1u)) * LR.pitch.x);
  let sv = lt / (f32(bundleOf(LR, 0u)) * LR.pitch.y);
  let Vv = normalize(F.camPos.xyz - in.world);
  let front = dot(Ng, Vv) > 0.0;
  let sgn = select(-1.0, 1.0, front);
  // the rolled rim tilts the cloth normal toward the edge
  let lipDir = normalize(Tu * rim.dir.x + Tv * rim.dir.y);
  let Nf = normalize(select(-Ng, Ng, front) - sgn * rim.slope * lipDir);

  let dist = length(F.camPos.xyz - in.world);
  let pix = F.camPos.w * dist;
  let fy = max(length(dx), length(dy));
  // yarns near the pixel size: the prefiltered tile shades the weave. Torn groups still evaluate the
  // exact weave per sample for their coverage, so both kinds of cell share one appearance
  let intactCell = !torn;
  let lodTile = smoothstep(0.28, 0.55, fy);
  let lod = select(smoothstep(0.45, 1.1, fy), 0.0, intactCell);
  var warpTop = 0.0;

  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, Nf);
  let amb = F.ambient.rgb * (0.55 + 0.45 * Nf.z);
  // the rim's loose fibres: pale albedo, more sheen, weave washed out toward the edge
  var fib = vec4<f32>(0.0);
  if (rim.rim > 0.0) { fib = rimFibres(rest, rim.dir, in.layer, pix); }
  let rimPale = pow(rim.rim, 0.7) * fib.w;
  let rimAlb = fib.rgb;
  let rimFuzz = 0.9 * rim.rim;

  // 4x MSAA standard sample offsets (pixel units, y down); each also jittered over the hairiness
  // radius (~45 um, wider in the loosened rim), which blurs the weave the way loose fibres do
  var offs = array<vec2<f32>, 4>(vec2<f32>(-0.125, -0.375), vec2<f32>(0.375, -0.125), vec2<f32>(-0.375, 0.125), vec2<f32>(0.125, 0.375));
  let hairy = 45e-6 * (1.0 + 5.0 * rim.rim);
  let fuzzR = vec2<f32>(hairy / (LR.pitch.x * su), hairy / (LR.pitch.y * sv));
  let rotSeed = hash11(u32(in.pos.x) * 7919u + u32(in.pos.y));
  let rot = hashf(rotSeed) * 6.2831853;
  let cellSeed = hash11((in.cell.x * 73856093u) ^ (in.cell.y * 19349663u));
  let hTop = LR.warp.z + 0.5 * LR.warp.y;
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
    let sdk = sd + sdx * offs[k].x + sdy * offs[k].y;
    let w2 = weave2(LR, uvk.x, uvk.y, su, sv, front, 0.018e-3);
    // at the free edge every yarn ends on its own: yarns crossing the crack fringe out past it,
    // yarns running along it fall away from the outermost rows
    var reach = 0.0;
    if (torn) {
      if (w2.a.covered) {
        let axis = select(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0), w2.a.fam == 1u);
        let across = abs(dot(axis, rim.dir));
        let hy = hash2f(cellSeed ^ (w2.a.fam * 0x68e31da4u), bitcast<u32>(w2.a.yarn));
        let hz = hash2f(cellSeed + 0x1b873593u, bitcast<u32>(w2.a.yarn) * 2u + w2.a.fam);
        reach = maturity * (across * (1.2 * hy * hy - 0.2) * 1e-3 - (1.0 - across) * 0.75e-3 * hz);
      } else {
        reach = maturity * (hashf(rotSeed + k * 31u) - 0.6) * 0.8e-3;
      }
    }
    let inEdge = sdk + reach > 0.0;
    if (inEdge && (w2.a.covered || lod >= 0.99)) { mask |= (1u << k); }
    if (!w2.a.covered || !inEdge) { continue; }
    let hit = w2.a;
    let wb = w2.wb;
    warpTop += select(0.0, 1.0, hit.fam == 1u) * (1.0 - wb) + select(0.0, 1.0, w2.b.covered && w2.b.fam == 1u) * wb;
    // yarns below the pixel: the prefiltered tile shades a torn group, its samples only decide coverage
    if (lodTile >= 1.0) {
      cnt += 1.0;
      continue;
    }
    let look = yarnLook(LR, in.layer, hit.fam, hit.yarn, hit.float_, hit.along, hit.x, wearF, pix);
    var albedo = look.albedo;
    var slope = hit.slope;
    var h = hit.h;
    if (wb > 0.02) {
      albedo = mix(albedo, select(weftMean, warpMean, w2.b.fam == 1u) * (1.0 - 0.4 * wearF) + vec3<f32>(0.12) * wearF, wb);
      slope = mix(slope, w2.b.slope, wb);
      h = mix(h, w2.b.h, wb);
    }
    // the rim's loosened fibres wash the weave out
    albedo = mix(albedo, rimAlb, rimPale);
    slope *= 1.0 - 0.6 * rim.rim;
    let Ny = normalize(Nf - sgn * (slope.x * Tu + slope.y * Tv));
    var axis = select(Tu, Tv, hit.fam == 1u);
    let slopeAlong = select(slope.x, slope.y, hit.fam == 1u);
    axis = normalize(axis + sgn * slopeAlong * Nf);
    let tw = select(LR.weft.w, LR.warp.w, hit.fam == 1u);
    let hn = mix(clamp((sgn * h + hTop) / (2.0 * hTop), 0.0, 1.0), 0.7, rim.rim);
    // interstices between floats (where the weft shows) are deep and shadowed by the neighbours
    let ao = weaveOcclusion(hn) * rim.ao;
    albDirect += albedo * mix(0.55, 1.0, ao);
    albAmb += albedo * ao;
    nSum += Ny;
    tSum += cos(tw) * axis + sin(tw) * cross(Ny, axis);
    fuzz += look.fuzz + rimFuzz;
    groove += 1.0 - hn;
    cnt += 1.0;
  }
  var acc = vec3<f32>(0.0);
  if (cnt > 0.0 && lodTile < 1.0) {
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
    let ts = 1.0 - 0.6 * rim.rim;
    let Ny = normalize(Nf - sgn * ts * (tex.z * Tu + tex.w * Tv));
    let k = i32(round(in.uv.x));
    let m = i32(round(in.uv.y));
    let fl = i32(round((in.uv.y - warpUnderDist(LR, k, in.uv.y)) / rep));
    let lw = yarnLook(LR, in.layer, 1u, k, fl, in.uv.y * LR.pitch.y, 0.3, wearF, pix);
    let lf = yarnLook(LR, in.layer, 0u, m, 0, in.uv.x * LR.pitch.x, 0.3, wearF, pix);
    let albedo = mix(mix(lf.albedo, lw.albedo, wf), rimAlb, rimPale);
    let ao = mix(tex.y, 0.8, rim.rim) * rim.ao;
    let tw = mix(LR.weft.w, LR.warp.w, wf);
    let Ta = select(Tu, Tv, wf > 0.5);
    let Tf = normalize(cos(tw) * Ta + sin(tw) * cross(Ny, Ta));
    let fz = mix(lf.fuzz, lw.fuzz, wf) + rimFuzz;
    var ct = fiberShade(Ny, Tf, Vv, key.xyz, albedo * mix(0.55, 1.0, ao), fz) * (key.w * sh) + amb * albedo * ao;
    ct = mix(ct, (key.w * sh * max(dot(Nf, key.xyz), 0.0) * 0.06 + amb * 0.6) * vec3<f32>(0.32, 0.36, 0.48), 0.12 * (1.0 - ao) * fz);
    color = select(mix(color, ct, lodTile), ct, cnt == 0.0 || (!intactCell && lodTile >= 1.0));
    if (intactCell) { mask = 0xFu; }
  }
  if (lod > 0.0) {
    // filtered appearance once the weave is below the pixel footprint
    let mw = yarnLook(LR, in.layer, 1u, i32(round(in.uv.x)), 0, in.uv.y * LR.pitch.y, 0.4, wearF, 1.0);
    let mf = yarnLook(LR, in.layer, 0u, i32(round(in.uv.y)), 0, in.uv.x * LR.pitch.x, 0.4, wearF, 1.0);
    let albedo = mix(select(mix(mw.albedo, mf.albedo, 0.14), mix(mf.albedo, mw.albedo, 0.25), !front), rimAlb, rimPale);
    let Tf = select(Tu, Tv, front);
    let cm = fiberShade(Nf, Tf, Vv, key.xyz, albedo, mw.fuzz + rimFuzz) * (key.w * sh) * 0.85 + amb * albedo * 0.8 * rim.ao;
    color = mix(color, cm, lod);
  }
  if (mask == 0u) { discard; }
  var o: FsOut;
  o.color = vec4<f32>(color, 1.0);
  o.mask = mask;
  return o;
}
