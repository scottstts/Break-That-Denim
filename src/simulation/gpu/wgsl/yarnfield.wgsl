// Yarn damage field (src/simulation/yarnField.ts, mirrored exactly): per crossing the damage of its
// warp and pick segments, advanced after each substep from the strain interpolated from the active
// parallel spans around it; the spans then take their intact fraction from it, and a span whose yarns
// have all separated ruptures. Passes, in order: spanStrain (strain snapshot per span), fieldUpdatePick
// and fieldUpdateWarp (per span and yarn, after sepSnapshot), pullMark and pullApply (per 16 crossings),
// coupleU and coupleV (per span, one family each).

@group(0) @binding(0) var<uniform> P: SimParams;
// per layer: first crossing, warps K, picks M, load-sharing reach along picks | along warps << 16
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
// per 16 crossings, bit 2 e + f: the separated state before the step (sepSnapshot -> fieldUpdate*),
// then the pull-out marks (pullMark -> pullApply)
@group(0) @binding(11) var<storage, read_write> pullBits: array<u32>;
// sound source anchors (common.wgsl nearestSource): positions of the held grips, w > 0 in use
@group(0) @binding(12) var<uniform> SRC: array<vec4<f32>, 4>;

const NO_STRAIN: f32 = 1e30;
const CUT: u32 = 255u;
// pulled out of the weave at the crossing (yarnField.ts DAMAGE_PULLED); codes >= PULLED separate it
const PULLED: u32 = 254u;

fn latticeOf(l: u32) -> vec4<u32> { return P.layers[l].lattice; }
// in the solve: cloth or a detached piece (not retired)
fn cloth(n: u32) -> bool { return x[n].w < 1.5; }
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

// ---------------------------------------------------------------- local load sharing (yarnField.ts)

override SHARE_FRACTION: f32 = 0.6;
override SHARE_RUN: u32 = 6u;

@compute @workgroup_size(128)
fn sepSnapshot(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  let total = fieldTotalY();
  if (w >= (total + 15u) / 16u || P.counts.z == 0u) { return; }
  var bits = 0u;
  for (var e = 0u; e < 16u; e++) {
    let c = 16u * w + e;
    if (c >= total) { break; }
    for (var f = 0u; f < 2u; f++) {
      if (damageAt(c, f) >= PULLED) { bits |= 1u << (2u * e + f); }
    }
  }
  pullBits[w] = bits;
}

fn sepAt(c: u32, f: u32) -> bool { return ((pullBits[c >> 4u] >> (2u * (c & 15u) + f)) & 1u) == 1u; }

// Overload of the yarn of family f at crossing (k, m) of layer field F, at most 1 + cap (shareFactor).
fn shareFactor(F: vec4<u32>, f: u32, k: u32, m: u32, cap: f32) -> f32 {
  let reach = i32(select(F.w & 0xFFFFu, F.w >> 16u, f == 1u));
  let along = i32(select(k, m, f == 1u));
  let lenAlong = i32(select(F.y, F.z, f == 1u));
  let across = i32(select(m, k, f == 1u));
  let lenAcross = i32(select(F.z, F.y, f == 1u));
  var runs = 0u;
  for (var side = 0; side < 2; side++) {
    let dir = select(-1, 1, side == 1);
    for (var d = 1; d <= i32(SHARE_RUN); d++) {
      let y = across + dir * d;
      if (y < 0 || y >= lenAcross) { break; }
      var broken = false;
      for (var a = max(0, along - reach); a <= min(lenAlong - 1, along + reach); a++) {
        let c = F.x + u32(select(y * i32(F.y) + a, a * i32(F.y) + y, f == 1u));
        if (sepAt(c, f)) {
          broken = true;
          break;
        }
      }
      if (!broken) { break; }
      runs++;
    }
  }
  return 1.0 + min(SHARE_FRACTION * f32(runs) * 0.5, cap);
}

// Yarn q of the family-f span owned by node n (invocation n * MAX_BUNDLE_F + q): its crossings within
// the span are in series and carry the same tension, so it softens at its weakest crossing only (the
// lowest peak strain), at the yarn's strain interpolated across the bundle at the yarn and at the
// middle of the span; mirrored exactly by updateField (yarnField.ts). One pass per family: within a
// pass every crossing's byte of the field word has a single writer.
const MAX_BUNDLE_F: u32 = 16u;

fn fieldUpdateYarn(gid: u32, f: u32) {
  let n = gid / MAX_BUNDLE_F;
  let q = gid % MAX_BUNDLE_F;
  if (n >= P.counts.x || P.counts.z == 0u) { return; }
  let ijl = nodes[n].ijl;
  let l = ijl >> 24u;
  let L = latticeOf(l);
  let i = ijl & 0xFFFu;
  let j = (ijl >> 12u) & 0xFFFu;
  let bu = bundleF(l, 1u);
  let bv = bundleF(l, 0u);
  let B = select(bv, bu, f == 1u);
  if (q >= B) { return; }
  let inside = select(j + 1u < L.z, i + 1u < L.y, f == 0u);
  if (!inside) { return; }
  let s = 2u * n + f;
  let sg = segs[s];
  if (!segActive(sg) || ((sg.cutMask >> q) & 1u) == 1u) { return; }
  let b = n + select(L.y, 1u, f == 0u);
  if (!cloth(n) || !cloth(b)) { return; }
  let F = FU[l];
  let Pn = select(bu, bv, f == 1u);
  let soft = sg.soft / sg.rest;
  // first crossing of the yarn in the span and the step along it
  var k0: u32;
  var m0: u32;
  if (f == 1u) {
    k0 = i * bu + q;
    m0 = j * bv + bv / 2u;
  } else {
    k0 = i * bu + bu / 2u;
    m0 = j * bv + q;
  }
  let dk = select(1u, 0u, f == 1u);
  let dm = select(0u, 1u, f == 1u);
  let eps = interpolatedStrain(l, f, k0 + (Pn >> 1u) * dk, m0 + (Pn >> 1u) * dm);
  if (eps >= 0.5 * NO_STRAIN) { return; }
  let y = P.layers[l].yarn[f];
  // no crossing can soften even at the largest overload: nothing to do (exact early out; most yarns)
  var minPeak = 1e30;
  for (var p = 0u; p < Pn; p++) {
    let c = F.x + (m0 + p * dm) * F.y + k0 + p * dk;
    minPeak = min(minPeak, strainAtTension(y, strength[c][f] * f32(B) / areaAt(c, f)));
  }
  if (eps * (1.0 + SHARE_FRACTION * f32(SHARE_RUN)) <= minPeak) { return; }
  // friction the crossings can pass to this yarn (shareCap): compaction of one crossing plus the wrap
  // force of the crossing yarn's tension at the middle of the span
  let g = 1u - f;
  let kM = k0 + (Pn >> 1u) * dk;
  let mM = m0 + (Pn >> 1u) * dm;
  let ec = interpolatedStrain(l, g, kM, mM);
  let Bg = select(bv, bu, g == 1u);
  let tCross = select(max(0.0, axialTension(P.layers[l].yarn[g], ec)) * areaAt(F.x + mM * F.y + kM, g) / f32(Bg), 0.0, ec >= 0.5 * NO_STRAIN);
  let reach = select(F.w & 0xFFFFu, F.w >> 16u, f == 1u);
  let transfer = f32(2u * reach + 1u) * P.layers[l].a.y * (nodes[n].n0 / f32(bu * bv) + P.layers[l].a.z * tCross);
  // where the yarn softens: where it already has, else where overload / peak strain is largest
  var best = F.x + m0 * F.y + k0;
  var bestScore = -1.0;
  var bestOmega = 1.0;
  var soft0 = 0xFFFFFFFFu;
  var softCode = 0u;
  for (var p = 0u; p < Pn; p++) {
    let kk = k0 + p * dk;
    let mm = m0 + p * dm;
    let c = F.x + mm * F.y + kk;
    let omega = shareFactor(F, f, kk, mm, transfer / strength[c][f]);
    let peak = strainAtTension(y, strength[c][f] * f32(B) / areaAt(c, f));
    let score = omega / peak;
    if (score > bestScore) {
      best = c;
      bestScore = score;
      bestOmega = omega;
    }
    let code = damageAt(c, f);
    if (code > softCode) {
      soft0 = c;
      softCode = code;
    }
  }
  if (soft0 != 0xFFFFFFFFu) {
    best = soft0;
    bestOmega = shareFactor(F, f, (best - F.x) % F.y, (best - F.x) / F.y, transfer / strength[best][f]);
  }
  let d = yarnDamage(y, f32(B), areaAt(best, f), strength[best][f], soft, eps * bestOmega);
  let bestCode = select(min(u32(floor(d * 255.0)), PULLED - 1u), CUT, d >= 1.0);
  let word = field[best];
  let oldCode = (word >> (8u * f)) & 0xFFu;
  if (bestCode > oldCode) {
    field[best] = (word & ~(0xFFu << (8u * f))) | (bestCode << (8u * f));
    atomicStore(&fieldEvents[0], 1u);
    emitFracture(best, f, B, y, oldCode, bestCode, x[n].xyz);
  }
}

// Acoustic emission of fibre fracture (common.wgsl ACOUSTIC): the strain energy per unit length the
// yarn held at its peak, in the share of its fibres that broke between two damage codes.
fn emitFracture(c: u32, f: u32, B: u32, y: Yarn, c0: u32, c1: u32, at: vec3<f32>) {
  if (c0 >= PULLED) { return; }
  let area = areaAt(c, f);
  let u = axialEnergyDensity(y, strainAtTension(y, strength[c][f] * f32(B) / area)) * area / f32(B);
  let d1 = select(f32(c1) / 255.0, 1.0, c1 == CUT);
  let e = u32(round((d1 - f32(c0) / 255.0) * u * 1e5));
  let s = nearestSource(at, SRC);
  atomicAdd(&counters[acousticSlot(s, 0u)], e);
  let w = i32(c1 - c0);
  let mm = vec3<i32>(round(at * 1000.0)) * w;
  atomicAdd(&counters[acousticSlot(s, 1u)], bitcast<u32>(mm.x));
  atomicAdd(&counters[acousticSlot(s, 2u)], bitcast<u32>(mm.y));
  atomicAdd(&counters[acousticSlot(s, 3u)], bitcast<u32>(mm.z));
  atomicAdd(&counters[acousticSlot(s, 4u)], u32(w));
  if (c1 == CUT) { atomicAdd(&counters[acousticSlot(s, 5u)], 1u); }
}

@compute @workgroup_size(128)
fn fieldUpdatePick(@builtin(global_invocation_id) gid: vec3<u32>) { fieldUpdateYarn(gid.x, 0u); }

@compute @workgroup_size(128)
fn fieldUpdateWarp(@builtin(global_invocation_id) gid: vec3<u32>) { fieldUpdateYarn(gid.x, 1u); }

// ---------------------------------------------------------------- pull-out (yarnField.ts pullOut)

fn fieldTotalY() -> u32 {
  let F = FU[P.counts.y - 1u];
  return F.x + F.y * F.z;
}

// Pull-out of free yarn ends (yarnField.ts pullOut, recoilLength): a crossing still interlaced leaves
// the weave once the nearest cut along its yarn (through crossings already pulled out) lies within the
// end's shear-lag retraction (T eps - W) / f at the local strain, its tension capped at the yarn's peak.
const RECOIL_MAX: i32 = 12;

fn breakDistance(F: vec4<u32>, f: u32, k: u32, m: u32) -> i32 {
  let pos = i32(select(m, k, f == 0u));
  let len = i32(select(F.z, F.y, f == 0u));
  let stride = i32(select(F.y, 1u, f == 0u));
  let c = i32(F.x + m * F.y + k);
  var best = -1;
  for (var side = 0; side < 2; side++) {
    let dir = select(-1, 1, side == 1);
    for (var d = 1; d <= RECOIL_MAX; d++) {
      let p = pos + dir * d;
      if (p < 0 || p >= len) { break; }
      let code = damageAt(u32(c + dir * d * stride), f);
      if (code == CUT) {
        if (best < 0 || d < best) { best = d; }
        break;
      }
      if (code != PULLED) { break; }
    }
  }
  return best;
}

fn pullsOut(l: u32, F: vec4<u32>, c: u32, f: u32, k: u32, m: u32) -> bool {
  let d = breakDistance(F, f, k, m);
  if (d < 0) { return false; }
  let eps = interpolatedStrain(l, f, k, m);
  if (eps >= 0.5 * NO_STRAIN || eps <= 0.0) { return false; }
  let mat = P.layers[l];
  let g = 1u - f;
  let ec = interpolatedStrain(l, g, k, m);
  let bu = bundleF(l, 1u);
  let bv = bundleF(l, 0u);
  let Bg = select(bv, bu, g == 1u);
  let tCross = select(max(0.0, axialTension(mat.yarn[g], ec)) * areaAt(c, g) / f32(Bg), 0.0, ec >= 0.5 * NO_STRAIN);
  let L = latticeOf(l);
  let n = L.x + min(L.z - 1u, m / bv) * L.y + min(L.y - 1u, k / bu);
  // pitch of the individual weave along the yarn: the bundled pitch over the crossing family's count
  let pitch = select(mat.a.w / f32(bu), mat.b.x / f32(bv), f == 1u);
  let B = select(bv, bu, f == 1u);
  let y = mat.yarn[f];
  // no yarn carries more than its own peak tension (the bundle strain near a crack includes its opening)
  let share = areaAt(c, f) / f32(B);
  let e = min(eps, strainAtTension(y, strength[c][f] / share));
  let comp = share * (axialTension(y, e) * e - axialEnergyDensity(y, e));
  let fric = mat.a.y * (nodes[n].n0 / f32(bu * bv) + mat.a.z * tCross);
  return comp * pitch >= (f32(d) - 0.5) * pitch * fric;
}

// One invocation per 16 crossings: which ends pull out of their next crossing this substep.
@compute @workgroup_size(128)
fn pullMark(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  let total = fieldTotalY();
  if (w >= (total + 15u) / 16u || P.counts.z == 0u) { return; }
  var bits = 0u;
  for (var e = 0u; e < 16u; e++) {
    let c = 16u * w + e;
    if (c >= total) { break; }
    var l = 0u;
    for (var q = 1u; q < P.counts.y; q++) {
      if (c >= FU[q].x) { l = q; }
    }
    let F = FU[l];
    let k = (c - F.x) % F.y;
    let m = (c - F.x) / F.y;
    for (var f = 0u; f < 2u; f++) {
      if (damageAt(c, f) >= PULLED) { continue; }
      if (pullsOut(l, F, c, f, k, m)) { bits |= 1u << (2u * e + f); }
    }
  }
  pullBits[w] = bits;
}

@compute @workgroup_size(128)
fn pullApply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  let total = fieldTotalY();
  if (w >= (total + 15u) / 16u || P.counts.z == 0u) { return; }
  let bits = pullBits[w];
  if (bits == 0u) { return; }
  for (var e = 0u; e < 16u; e++) {
    let c = 16u * w + e;
    if (c >= total) { break; }
    var word = field[c];
    for (var f = 0u; f < 2u; f++) {
      if (((bits >> (2u * e + f)) & 1u) == 1u) { word = (word & ~(0xFFu << (8u * f))) | (PULLED << (8u * f)); }
    }
    field[c] = word;
  }
  atomicStore(&fieldEvents[0], 1u);
}

fn coupleSpan(a: u32, f: u32) {
  if (a >= P.counts.x || P.counts.z == 0u) { return; }
  let s = 2u * a + f;
  var sg = segs[s];
  let slipped = (sg.flags & SEG_SLIPPED) != 0u;
  if (!slipped && !segActive(sg)) { return; }
  let ijl = nodes[a].ijl;
  let l = ijl >> 24u;
  let L = latticeOf(l);
  let i = ijl & 0xFFFu;
  let j = (ijl >> 12u) & 0xFFFu;
  let F = FU[l];
  let bu = bundleF(l, 1u);
  let bv = bundleF(l, 0u);
  if (slipped) {
    partYarns(F, i, j, f, bu, bv, select(1, 0, sg.failSite < 0.5));
    segs[s].flags = sg.flags & ~SEG_SLIPPED;
    return;
  }
  let b = a + select(L.y, 1u, f == 0u);
  if (!cloth(a) || !cloth(b)) { return; }
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
      let c = spanCrossingF(F, i, j, f, bu, bv, q, p);
      let d = damageAt(c, f);
      compliance += 1.0 / areaAt(c, f);
      dmax = max(dmax, d);
      if (d >= PULLED && first < 0) { first = i32(p); }
    }
    sum += select(1.0 - f32(dmax) / 255.0, 0.0, dmax >= PULLED) * (f32(Pn) / compliance);
    if (dmax >= PULLED) {
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
    partYarns(F, i, j, f, bu, bv, -1);
  }
  segs[s] = sg;
}

// A separated span's yarns part from the other side's weave (yarnField.ts partSpanYarns). A ruptured
// span's yarn belongs, up to its first separation, to the span's owner a and beyond it to b (their
// tails took its rest length; the renderer draws it so); a span whose yarns slid out of the crossings
// at one end (slipEnd 0 at a, 1 at b) belongs wholly to the other. The span's first half of crossings
// is interlaced with yarns of a's column (row, for a v-span), its second half with b's: where the
// yarn's side and the crossing's disagree, the crossing would still tie a to b after the lattice let
// them part. The yarn's end slides out there (the crossing yarn has no free end near): pulled out.
fn partYarns(F: vec4<u32>, i: u32, j: u32, f: u32, bu: u32, bv: u32, slipEnd: i32) {
  let B = select(bv, bu, f == 1u);
  let Pn = select(bu, bv, f == 1u);
  // crossings p < half are interlaced with a's yarns
  let half = Pn - Pn / 2u;
  for (var q = 0u; q < B; q++) {
    var ofA = slipEnd != 0;
    for (var p = 0u; p < Pn; p++) {
      let c = spanCrossingF(F, i, j, f, bu, bv, q, p);
      let word = field[c];
      if (((word >> (8u * f)) & 0xFFu) >= PULLED) {
        if (slipEnd < 0) { ofA = false; }
        continue;
      }
      if (ofA != (p < half)) {
        field[c] = (word & ~(0xFFu << (8u * f))) | (PULLED << (8u * f));
        atomicStore(&fieldEvents[0], 1u);
      }
    }
  }
}

// Crossing of yarn q of span (i, j, f) at its p-th crossing from the owner (yarnField.ts spanCrossing).
fn spanCrossingF(F: vec4<u32>, i: u32, j: u32, f: u32, bu: u32, bv: u32, q: u32, p: u32) -> u32 {
  if (f == 1u) { return F.x + (j * bv + bv / 2u + p) * F.y + i * bu + q; }
  return F.x + (j * bv + q) * F.y + i * bu + bu / 2u + p;
}

// One pass per family: a crossing's word is then written by the one span of that family holding it.
@compute @workgroup_size(128)
fn coupleU(@builtin(global_invocation_id) gid: vec3<u32>) { coupleSpan(gid.x, 0u); }

@compute @workgroup_size(128)
fn coupleV(@builtin(global_invocation_id) gid: vec3<u32>) { coupleSpan(gid.x, 1u); }
