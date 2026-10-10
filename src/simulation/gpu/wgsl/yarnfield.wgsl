// Yarn damage field (src/simulation/yarnField.ts, mirrored exactly): per crossing the damage of its
// warp and pick segments, advanced after each substep from the strain interpolated from the active
// parallel spans around it; the spans then take their intact fraction from it, and a span whose yarns
// have all separated ruptures. Passes, in order: spanStrain (strain snapshot per span), fieldUpdate
// (per crossing), couple (per span).

@group(0) @binding(0) var<uniform> P: SimParams;
// per layer: first crossing, warps K, picks M, unused
@group(0) @binding(1) var<uniform> FU: array<vec4<u32>, 3>;
@group(0) @binding(2) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
// per node: strain of its u and v spans (NO_STRAIN when they carry none); the solver's predictor
// buffer, free between the commit and the next predictor
@group(0) @binding(4) var<storage, read_write> strain: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> segs: array<Seg>;
@group(0) @binding(6) var<storage, read_write> tails: array<f32>;
// per crossing: damage code of its pick segment (bits 0-7) and warp segment (bits 8-15), 255 = cut;
// cross-section code of the abraded pick (bits 16-23) and warp (24-31) segment
@group(0) @binding(7) var<storage, read_write> field: array<u32>;
// per crossing: peak tension of its pick (x) and warp (y) segment, N
@group(0) @binding(8) var<storage, read> strength: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> counters: array<atomic<u32>>;
// [0] set to 1 whenever a damage code changed (the renderer clears it once it has rebuilt its edge field)
@group(0) @binding(10) var<storage, read_write> fieldEvents: array<atomic<u32>>;

const NO_STRAIN: f32 = 1e30;
const CUT: u32 = 255u;

fn latticeOf(l: u32) -> vec4<u32> { return P.layers[l].lattice; }
fn cloth(n: u32) -> bool { return x[n].w < 0.5; }
fn bundleF(l: u32, f: u32) -> u32 { return u32(P.layers[l].yarn[f].yarnsPerSpan); }
fn damageAt(c: u32, f: u32) -> u32 { return (field[c] >> (8u * f)) & 0xFFu; }
fn areaOfCode(code: u32) -> f32 { return max(0.02, f32(code) / 255.0); }
fn areaAt(c: u32, f: u32) -> f32 { return areaOfCode((field[c] >> (16u + 8u * f)) & 0xFFu); }

@compute @workgroup_size(128)
fn spanStrain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  var e = vec2<f32>(NO_STRAIN);
  if (cloth(n)) {
    let ijl = nodes[n].ijl;
    let L = latticeOf(ijl >> 24u);
    let i = ijl & 0xFFFu;
    let j = (ijl >> 12u) & 0xFFFu;
    for (var f = 0u; f < 2u; f++) {
      let sg = segs[2u * n + f];
      let inside = select(j + 1u < L.z, i + 1u < L.y, f == 0u);
      if (!inside || !segActive(sg)) { continue; }
      let b = n + select(L.y, 1u, f == 0u);
      if (cloth(b)) { e[f] = (length(x[b].xyz - x[n].xyz) - sg.rest) / sg.rest; }
    }
  }
  strain[n] = vec4<f32>(e, 0.0, 0.0);
}

// Strain of family f at crossing (k, m): bilinear over the active parallel spans at their midpoints.
fn interpolatedStrain(l: u32, f: u32, k: u32, m: u32) -> f32 {
  let L = latticeOf(l);
  let bu = f32(bundleF(l, 1u));
  let bv = f32(bundleF(l, 0u));
  let s = select((f32(k) - (bu - 0.5)) / bu, (f32(k) - 0.5 * (bu - 1.0)) / bu, f == 1u);
  let t = select((f32(m) - 0.5 * (bv - 1.0)) / bv, (f32(m) - (bv - 0.5)) / bv, f == 1u);
  let i0 = i32(floor(s));
  let j0 = i32(floor(t));
  let fu = s - f32(i0);
  let fv = t - f32(j0);
  let imax = i32(L.y) - select(2, 1, f == 1u);
  let jmax = i32(L.z) - select(1, 2, f == 1u);
  var num = 0.0;
  var den = 0.0;
  for (var dj = 0; dj < 2; dj++) {
    for (var di = 0; di < 2; di++) {
      let i = i0 + di;
      let j = j0 + dj;
      if (i < 0 || j < 0 || i > imax || j > jmax) { continue; }
      let e = strain[L.x + u32(j) * L.y + u32(i)][f];
      if (e >= 0.5 * NO_STRAIN) { continue; }
      let w = select(1.0 - fu, fu, di == 1) * select(1.0 - fv, fv, dj == 1);
      num += w * e;
      den += w;
    }
  }
  return select(NO_STRAIN, num / den, den > 1e-6);
}

@compute @workgroup_size(128)
fn fieldUpdate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (P.counts.z == 0u) { return; }
  var l = 0u;
  for (var q = 1u; q < P.counts.y; q++) {
    if (c >= FU[q].x) { l = q; }
  }
  let F = FU[l];
  if (c >= F.x + F.y * F.z) { return; }
  let local = c - F.x;
  let k = local % F.y;
  let m = local / F.y;
  let L = latticeOf(l);
  let bu = bundleF(l, 1u);
  let bv = bundleF(l, 0u);
  var word = field[c];
  let st = strength[c];
  for (var f = 0u; f < 2u; f++) {
    let code = (word >> (8u * f)) & 0xFFu;
    if (code == CUT) { continue; }
    // the yarn's span and its index q in the bundle
    var i: i32;
    var j: i32;
    var q: u32;
    if (f == 1u) {
      i = i32(k / bu);
      j = (i32(m) - i32(bv / 2u)) / i32(bv);
      if (i32(m) < i32(bv / 2u)) { j = -1; }
      q = k - u32(max(i, 0)) * bu;
      if (i >= i32(L.y) || j < 0 || j > i32(L.z) - 2) { continue; }
    } else {
      j = i32(m / bv);
      i = (i32(k) - i32(bu / 2u)) / i32(bu);
      if (i32(k) < i32(bu / 2u)) { i = -1; }
      q = m - u32(max(j, 0)) * bv;
      if (j >= i32(L.z) || i < 0 || i > i32(L.y) - 2) { continue; }
    }
    let a = L.x + u32(j) * L.y + u32(i);
    let s = 2u * a + f;
    let sg = segs[s];
    if (!segActive(sg) || ((sg.cutMask >> q) & 1u) == 1u) { continue; }
    let b = a + select(L.y, 1u, f == 0u);
    if (!cloth(a) || !cloth(b)) { continue; }
    let eps = interpolatedStrain(l, f, k, m);
    if (eps >= 0.5 * NO_STRAIN) { continue; }
    let B = f32(select(bv, bu, f == 1u));
    let d = yarnDamage(P.layers[l].yarn[f], B, areaOfCode((word >> (16u + 8u * f)) & 0xFFu), st[f], sg.soft / sg.rest, eps);
    let nc = select(min(u32(floor(d * 255.0)), 254u), CUT, d >= 1.0);
    if (nc > code) { word = (word & ~(0xFFu << (8u * f))) | (nc << (8u * f)); }
  }
  if (word != field[c]) {
    field[c] = word;
    atomicStore(&fieldEvents[0], 1u);
  }
}

@compute @workgroup_size(128)
fn couple(@builtin(global_invocation_id) gid: vec3<u32>) {
  let s = gid.x;
  if (s >= 2u * P.counts.x || P.counts.z == 0u) { return; }
  var sg = segs[s];
  if (!segActive(sg)) { return; }
  let a = s >> 1u;
  let f = s & 1u;
  let ijl = nodes[a].ijl;
  let l = ijl >> 24u;
  let L = latticeOf(l);
  let i = ijl & 0xFFFu;
  let j = (ijl >> 12u) & 0xFFFu;
  let b = a + select(L.y, 1u, f == 0u);
  if (!cloth(a) || !cloth(b)) { return; }
  let F = FU[l];
  let bu = bundleF(l, 1u);
  let bv = bundleF(l, 0u);
  let B = select(bv, bu, f == 1u);
  let Pn = select(bu, bv, f == 1u);
  var sum = 0.0;
  var mask = 0u;
  var site = 0.0;
  for (var q = 0u; q < B; q++) {
    var dmax = 0u;
    var first = -1;
    var compliance = 0.0;
    for (var p = 0u; p < Pn; p++) {
      var c: u32;
      if (f == 1u) {
        c = F.x + (j * bv + bv / 2u + p) * F.y + i * bu + q;
      } else {
        c = F.x + (j * bv + q) * F.y + i * bu + bu / 2u + p;
      }
      let d = damageAt(c, f);
      compliance += 1.0 / areaAt(c, f);
      dmax = max(dmax, d);
      if (d == CUT && first < 0) { first = i32(p); }
    }
    sum += (1.0 - f32(dmax) / 255.0) * (f32(Pn) / compliance);
    if (dmax == CUT) {
      mask |= 1u << q;
      site += (f32(first) + 0.5) / f32(Pn);
    }
  }
  sg.intact = sum / f32(B);
  sg.cutMask = mask;
  let all = select((1u << B) - 1u, 0xFFFFFFFFu, B >= 32u);
  if (mask == all) {
    let phi = site / f32(B);
    sg.flags = sg.flags | SEG_BROKEN;
    sg.failSite = phi;
    tails[4u * a + 2u * f + 1u] += sg.rest * phi;
    tails[4u * b + 2u * f] += sg.rest * (1.0 - phi);
    atomicAdd(&counters[0], 1u);
  }
  segs[s] = sg;
}
