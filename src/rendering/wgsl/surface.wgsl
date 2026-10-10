// Woven cloth surface. Intact cells: one instance per lattice cell, S x S sub-quads of the refined
// bicubic surface. Cells near damage (pieces.wgsl gives a node of theirs a connection field, or a span
// of theirs has separated) are drawn per crossing instead (vsFine): one quad per crossing that has
// surface material, each placed from the cell's corners it is connected to (material_map.wgsl), so
// the cloth opens exactly where its yarns separated and is drawn only where material is.
// The fragment shader evaluates the woven yarn surface at the four MSAA sample positions: each sample
// is either a yarn (shaded) or a gap (uncovered): a gap in stretched weave, or a yarn that is not part
// of the cloth surface there (separated, or drawn as a loose thread), where the yarn below shows
// through, or nothing. Near a free edge (edge field) the cloth turns into its torn rim: an abraded
// halo, then a pale fibrous band.

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> rsurf: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cellFlags: array<u32>;
@group(0) @binding(4) var shadowMap: texture_depth_2d;
@group(0) @binding(5) var shadowSampler: sampler_comparison;
@group(0) @binding(6) var weaveTile: texture_2d_array<f32>;
@group(0) @binding(7) var tileSampler: sampler;
// render lists: fine cells from FINE_BASE (refine.wgsl)
@group(0) @binding(8) var<storage, read> lists: array<u32>;
@group(0) @binding(9) var edgeTex: texture_2d<f32>;
@group(0) @binding(10) var edgeSampler: sampler;
@group(0) @binding(11) var<storage, read> fieldR: array<u32>;
// per 16 crossings, 2 bits each: the pick / warp there belongs to a loose thread (threads.wgsl)
@group(0) @binding(12) var<storage, read> threadBitsR: array<u32>;
@group(0) @binding(13) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read> kappaM: array<u32>;


struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) uv: vec2<f32>,
  @location(2) dPs: vec3<f32>,
  @location(3) dPt: vec3<f32>,
  @location(4) wearV: f32,
  @location(5) @interpolate(flat) layer: u32,
  // cell (i, j), unused, unused
  @location(6) @interpolate(flat) cell: vec4<u32>,
  // x: 1 for a fine (per-crossing) quad, 0 for an intact cell
  @location(7) @interpolate(flat) torn: vec4<u32>,
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
  if ((cellFlags[cell] & 1u) == 0u) { return degenerate(); }
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
  o.cell = vec4<u32>(ci, cj, 0u, 0u);
  o.torn = vec4<u32>(0u);
  o.pos = F.viewProj * vec4<f32>(a.xyz, 1.0);
  return o;
}

// ---------------------------------------------------------------- cells near damage, per crossing

fn debrisS(c: u32, total: u32) -> bool { return ((threadBitsR[debrisWord(c, total)] >> (c & 31u)) & 1u) == 1u; }

// The yarn of family f at crossing c is part of the cloth surface (not separated, not a thread).
fn surfaceYarnS(c: u32, f: u32) -> bool {
  return yarnCode(fieldR[c], f) < PULLED_R && ((threadBitsR[c >> 4u] >> (2u * (c & 15u) + f)) & 1u) == 0u;
}

// Vertex vi of instance ii (a fine cell): six per crossing quad of the cell's crossing grid. A quad
// covers its crossing's area clipped to the cell, so the cells tile; each vertex is placed with the
// connections of its own crossing to the corners, averaged with the neighbouring crossings around
// the vertex that its material reaches directly (along a pick or a warp both are part of): within a
// piece of cloth neighbouring quads meet exactly, across a crack they part.
fn fineVertex(vi: u32, ii: u32) -> VOut {
  let cell = lists[FINE_BASE + ii];
  let l = layerOfCell(cell);
  let LR = R.layers[l];
  let L = LR.lattice;
  let local = cell - LR.refine.z;
  let ci = local % (L.y - 1u);
  let cj = local / (L.y - 1u);
  let g = cellGrid(LR, ci, cj);
  let q = vi / 6u;
  let qk = q % FINE_QK;
  let qm = q / FINE_QK;
  if (qk >= g.z || qm >= g.w) { return degenerate(); }
  let k = i32(g.x + qk);
  let m = i32(g.y + qm);
  let c = fieldIndexR(LR, u32(k), u32(m));
  if (!surfaceYarnS(c, 0u) && !surfaceYarnS(c, 1u)) { return degenerate(); }
  let bu = f32(bundleOf(LR, 1u));
  let bv = f32(bundleOf(LR, 0u));
  let ua = f32(ci) * bu + 0.5 * (bu - 1.0);
  let va = f32(cj) * bv + 0.5 * (bv - 1.0);
  let u0 = max(ua, f32(k) - 0.5);
  let u1 = min(ua + bu, f32(k) + 0.5);
  let v0 = max(va, f32(m) - 0.5);
  let v1 = min(va + bv, f32(m) + 0.5);
  if (u1 - u0 < 1e-4 || v1 - v0 < 1e-4) { return degenerate(); }
  var cuo = array<u32, 6>(0u, 1u, 1u, 0u, 1u, 0u);
  var cvo = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
  let cu = cuo[vi % 6u];
  let cv = cvo[vi % 6u];
  let U = select(u0, u1, cu == 1u);
  let V = select(v0, v1, cv == 1u);
  // neighbouring crossings around the vertex (where it lies on a crossing corner, not clipped)
  let sx = select(-1, 1, cu == 1u);
  let sy = select(-1, 1, cv == 1u);
  let onX = abs(U - (f32(k) + 0.5 * f32(sx))) < 1e-4;
  let onY = abs(V - (f32(m) + 0.5 * f32(sy))) < 1e-4;
  let K = i32(LR.field.y);
  let M = i32(LR.field.z);
  let kx = k + sx;
  let my = m + sy;
  let inX = onX && kx >= 0 && kx < K;
  let inY = onY && my >= 0 && my < M;
  let cx = fieldIndexR(LR, u32(clamp(kx, 0, K - 1)), u32(m));
  let cy = fieldIndexR(LR, u32(k), u32(clamp(my, 0, M - 1)));
  let cxy = fieldIndexR(LR, u32(clamp(kx, 0, K - 1)), u32(clamp(my, 0, M - 1)));
  // joined along a yarn, within one piece (cloth or a falling piece)
  let total = fieldTotalOf(R.layers[R.counts.y - 1u]);
  let deb = debrisS(c, total);
  let linkX = inX && surfaceYarnS(c, 0u) && surfaceYarnS(cx, 0u) && debrisS(cx, total) == deb;
  let linkY = inY && surfaceYarnS(c, 1u) && surfaceYarnS(cy, 1u) && debrisS(cy, total) == deb;
  let linkXY = inX && inY && debrisS(cxy, total) == deb && ((linkX && surfaceYarnS(cx, 1u) && surfaceYarnS(cxy, 1u)) || (linkY && surfaceYarnS(cy, 0u) && surfaceYarnS(cxy, 0u)));
  let n00 = L.x + cj * L.y + ci;
  var nc = array<u32, 4>(n00, n00 + 1u, n00 + L.y, n00 + L.y + 1u);
  var kap = vec4<f32>(0.0);
  var cnt = 0.0;
  for (var e = 0u; e < 4u; e++) {
    var kk = k;
    var mm = m;
    var take = e == 0u;
    if (e == 1u) {
      kk = kx;
      take = linkX;
    } else if (e == 2u) {
      mm = my;
      take = linkY;
    } else if (e == 3u) {
      kk = kx;
      mm = my;
      take = linkXY;
    }
    if (!take) { continue; }
    kap += vec4<f32>(kappaAt(l, nc[0], kk, mm), kappaAt(l, nc[1], kk, mm), kappaAt(l, nc[2], kk, mm), kappaAt(l, nc[3], kk, mm));
    cnt += 1.0;
  }
  kap /= cnt;
  let mp = matPoint(l, ci, cj, (U - ua) / bu, (V - va) / bv, kap);
  if (!mp.valid) { return degenerate(); }
  var o: VOut;
  o.world = mp.p;
  o.wearV = mp.wear;
  o.dPs = mp.ds;
  o.dPt = mp.dt;
  o.uv = vec2<f32>(U, V);
  o.layer = l;
  o.cell = vec4<u32>(ci, cj, 0u, 0u);
  o.torn = vec4<u32>(1u, 0u, 0u, 0u);
  o.pos = F.viewProj * vec4<f32>(mp.p, 1.0);
  return o;
}

// Crossing of the yarn segment hit at fine point uv: a warp at its nearest pick, a pick at its
// nearest warp.
fn hitCrossing(L: LayerR, h: Hit, uv: vec2<f32>) -> u32 {
  var k: i32;
  var m: i32;
  if (h.fam == 1u) {
    k = h.yarn;
    m = i32(round(uv.y));
  } else {
    k = i32(round(uv.x));
    m = h.yarn;
  }
  k = clamp(k, 0, i32(L.field.y) - 1);
  m = clamp(m, 0, i32(L.field.z) - 1);
  return fieldIndexR(L, u32(k), u32(m));
}

// Damage code of the yarn segment hit at fine point uv.
fn hitDamage(L: LayerR, h: Hit, uv: vec2<f32>) -> u32 {
  return yarnCode(fieldR[hitCrossing(L, h, uv)], h.fam);
}

// The yarn segment hit at uv is gone from the cloth: separated, or a loose thread drawn on its own.
fn hitAbsent(L: LayerR, h: Hit, uv: vec2<f32>) -> bool {
  let c = hitCrossing(L, h, uv);
  if (yarnCode(fieldR[c], h.fam) >= PULLED_R) { return true; }
  return ((threadBitsR[c >> 4u] >> (2u * (c & 15u) + h.fam)) & 1u) == 1u;
}

// A yarn with damage code d is missing at a sample: separated, or (weakened, loosened) frayed through
// at random fibre positions fs, more often the more it is damaged.
fn yarnGone(d: u32, fs: u32) -> bool {
  if (d >= PULLED_R) { return true; }
  let t = f32(d) / 255.0;
  return hashf(fs) < 0.55 * t * t;
}

@vertex
fn vsFine(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  return fineVertex(vi, ii);
}

@vertex
fn vsFineShadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4<f32> {
  let o = fineVertex(vi, ii);
  if (o.pos.z < -1.0) { return o.pos; }
  return F.lightViewProj * vec4<f32>(o.world, 1.0);
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

// d: distance to the nearest opening (mm), dir: unit rest-chart gradient of d, rest: rest point (m),
// density: separated yarn ends around it (edge field). A torn edge is abraded and loosened: its
// yarns untwist into a dense band of pale fibre (the rim) inside a wider abraded halo, so it reads as
// a fuzzy pale border, not a cut. A few cut yarns fray a little; an opening whose edge is all yarn
// ends has the full rim.
fn rimAt(d: f32, dir: vec2<f32>, rest: vec2<f32>, layer: u32, density: f32) -> Rim {
  var o: Rim;
  if (d >= EDGE_RMAX) {
    o.rim = 0.0;
    o.halo = 0.0;
    o.slope = 0.0;
    o.dir = dir;
    o.ao = 1.0;
    return o;
  }
  // the loosened band is the yarns beside the cut that lost the crossings holding them: a crossing or
  // a few wide, wandering along the edge from nearly bare stretches to tufted ones
  let n = 0.65 * noise2(4441u + layer, rest / 0.0065) + 0.35 * noise2(4447u + layer, rest / 0.0022);
  let edge = smoothstep(0.03, 0.22, density);
  let w = clamp(1.3 + 1.2 * n, 0.35, 2.6) * mix(0.5, 1.0, edge);
  let x = d / w;
  o.rim = (1.0 - smoothstep(0.0, 1.0, x)) * mix(0.3, 1.0, edge);
  o.halo = (1.0 - smoothstep(0.0, 6.5, d)) * edge;
  // a loosened edge does not roll into a lip: its relief is the fibre itself (no normal tilt)
  o.slope = 0.0;
  o.dir = dir;
  o.ao = 1.0;
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
  // undyed weft and the pale cores of untwisted warp, with some fibre that keeps its indigo
  let pale = vec3<f32>(0.64, 0.67, 0.73);
  let dyed = vec3<f32>(0.2, 0.28, 0.5);
  let alb = mix(pale, dyed, 0.3 * blue) * (0.86 + 0.26 * (0.6 * streak + 0.4 * tuft) * fine);
  let cover = clamp(0.8 + 0.2 * clump + 0.3 * (0.65 * streak + 0.35 * tuft) * fine, 0.0, 1.0);
  return vec4<f32>(alb, cover);
}

@fragment
fn fsMain(in: VOut, @builtin(front_facing) ff: bool) -> FsOut {
  let LR = R.layers[in.layer];
  let rest = restOfUV(LR, in.uv);
  let dx = dpdx(in.uv);
  let dy = dpdy(in.uv);

  // edge field: distance to the nearest separated yarn (mm), its rest-chart gradient (central
  // differences over one crossing), and the filtered intact fraction of warp and pick
  let euv = edgeUV(LR, in.uv);
  let ef = textureSampleLevel(edgeTex, edgeSampler, euv, 0.0);
  let dEdge = ef.x;
  var gEdge = vec2<f32>(0.0);
  if (dEdge < EDGE_RMAX) {
    let tu = vec2<f32>(1.0 / LR.edge.z, 0.0);
    let tv = vec2<f32>(0.0, 1.0 / LR.edge.w);
    let gx = textureSampleLevel(edgeTex, edgeSampler, euv + tu, 0.0).x - textureSampleLevel(edgeTex, edgeSampler, euv - tu, 0.0).x;
    let gy = textureSampleLevel(edgeTex, edgeSampler, euv + tv, 0.0).x - textureSampleLevel(edgeTex, edgeSampler, euv - tv, 0.0).x;
    gEdge = vec2<f32>(gx / LR.pitch.x, gy / LR.pitch.y);
  }
  let gl = length(gEdge);
  let dirEdge = select(vec2<f32>(0.0, 1.0), gEdge / gl, gl > 1e-12);
  let rim = rimAt(dEdge, dirEdge, rest, in.layer, ef.w);
  // weakened (partly separated) yarns around the damage are abraded and pale
  let weak = 1.0 - min(ef.y, 1.0);

  var wearF = min(1.0, in.wearV + 0.4 * rim.halo * rim.halo + 0.35 * weak);
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
  // yarns near the pixel size: the prefiltered tile shades the weave
  let lodTile = smoothstep(0.28, 0.55, fy);

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
  let hairy = 45e-6 * (1.0 + 2.5 * rim.rim);
  let fuzzR = vec2<f32>(hairy / (LR.pitch.x * su), hairy / (LR.pitch.y * sv));
  let rotSeed = hash11(u32(in.pos.x) * 7919u + u32(in.pos.y));
  let rot = hashf(rotSeed) * 6.2831853;
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
    if (lodTile >= 1.0) { break; }
    let ang = rot + f32(k) * 2.39996;
    let jr = sqrt((f32(k) + 0.5) / 4.0);
    let uvk = in.uv + dx * offs[k].x + dy * offs[k].y + vec2<f32>(cos(ang), sin(ang)) * jr * fuzzR;
    let w2 = weave2(LR, uvk.x, uvk.y, su, sv, front, 0.018e-3);
    // the yarn damage field: a separated yarn is gone here (the one below shows, if any), a weakened
    // one is loosened: frayed through in places, pale and fuzzy
    var hit = w2.a;
    var wb = w2.wb;
    var dmg = 0.0;
    var present = w2.a.covered;
    if (present) {
      let fs = hash11(rotSeed + 977u * k + 13u);
      let ca = hitDamage(LR, w2.a, uvk);
      if (yarnGone(ca, fs) || hitAbsent(LR, w2.a, uvk)) {
        present = w2.b.covered && !hitAbsent(LR, w2.b, uvk);
        hit = w2.b;
        if (!present && w2.a.fam == 1u) {
          // a warp worn through over its pick: the loosened pick spreads into the gap, over its full
          // pitch at rest (stretched cloth still opens between picks)
          let m = round(uvk.y);
          let wh = weftHit(LR, i32(m), uvk.x, uvk.y, su, sv);
          if (abs(uvk.y - m) * sv < 0.5 && !hitAbsent(LR, wh, uvk)) {
            hit = wh;
            hit.x = clamp(hit.x * LR.weft.x / LR.pitch.y, -0.95, 0.95);
            // no longer in the groove under a float: exposed to the light, near the face of the weave
            hit.h = sgn * 0.45 * (LR.warp.z + 0.5 * LR.warp.y);
            hit.slope *= 0.5;
            present = true;
          }
        }
        wb = 0.0;
        dmg = select(0.0, f32(hitDamage(LR, hit, uvk)) / 255.0, present);
      } else {
        dmg = f32(ca) / 255.0;
        if (wb > 0.02 && w2.b.covered && hitAbsent(LR, w2.b, uvk)) { wb = 0.0; }
      }
    }
    if (present) { mask |= (1u << k); }
    if (!present) { continue; }
    let look = yarnLook(LR, in.layer, hit.fam, hit.yarn, hit.float_, hit.along, hit.x, min(1.0, wearF + 0.7 * dmg), pix);
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
    fuzz += look.fuzz + rimFuzz + 0.5 * dmg;
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
  // filtered presence of warp and pick (edge field) for weave below the pixel
  let warpP = clamp(ef.y, 0.0, 1.0);
  let pickP = clamp(ef.z, 0.0, 1.0);
  if (lodTile > 0.0) {
    let rep = f32(LR.twill.x);
    let tex = textureSampleGrad(weaveTile, tileSampler, in.uv / rep, in.layer, dx / rep, dy / rep);
    // the back of a warp-faced twill shows mostly weft; a damaged cell shows what its samples hit;
    // where warps are worn through, the picks beneath them show
    let wfTile0 = select(1.0 - tex.x, tex.x, front);
    let wfTile = wfTile0 * warpP / max(wfTile0 * warpP + (1.0 - wfTile0) * pickP, 1e-3);
    let wf = wfTile;
    let ts = 1.0 - 0.6 * rim.rim;
    let Ny = normalize(Nf - sgn * ts * (tex.z * Tu + tex.w * Tv));
    let kk = i32(round(in.uv.x));
    let m = i32(round(in.uv.y));
    let fl = i32(round((in.uv.y - warpUnderDist(LR, kk, in.uv.y)) / rep));
    let lw = yarnLook(LR, in.layer, 1u, kk, fl, in.uv.y * LR.pitch.y, 0.3, wearF, pix);
    let lf = yarnLook(LR, in.layer, 0u, m, 0, in.uv.x * LR.pitch.x, 0.3, wearF, pix);
    let albedo = mix(mix(lf.albedo, lw.albedo, wf), rimAlb, rimPale);
    let ao = mix(tex.y, 0.8, rim.rim) * rim.ao;
    let tw = mix(LR.weft.w, LR.warp.w, wf);
    let Ta = select(Tu, Tv, wf > 0.5);
    let Tf = normalize(cos(tw) * Ta + sin(tw) * cross(Ny, Ta));
    let fz = mix(lf.fuzz, lw.fuzz, wf) + rimFuzz + 0.5 * weak;
    var ct = fiberShade(Ny, Tf, Vv, key.xyz, albedo * mix(0.55, 1.0, ao), fz) * (key.w * sh) + amb * albedo * ao;
    ct = mix(ct, (key.w * sh * max(dot(Nf, key.xyz), 0.0) * 0.06 + amb * 0.6) * vec3<f32>(0.32, 0.36, 0.48), 0.12 * (1.0 - ao) * fz);
    color = select(mix(color, ct, lodTile), ct, cnt == 0.0);
    if (lodTile >= 1.0) {
      // coverage of the filtered cloth: holes where both families are gone, dithered over the samples
      let cover = 1.0 - (1.0 - warpP) * (1.0 - pickP);
      mask = 0u;
      for (var k = 0u; k < 4u; k++) {
        if (cover > (f32(k) + hashf(rotSeed + 31u * k)) * 0.25) { mask |= 1u << k; }
      }
    }
  }
  if (mask == 0u) { discard; }
  var o: FsOut;
  o.color = vec4<f32>(color, 1.0);
  o.mask = mask;
  return o;
}
