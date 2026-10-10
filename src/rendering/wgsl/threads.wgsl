// Loose threads from the yarn damage field. Where a yarn is no longer interlaced with the yarns
// crossing it, it is a thread: picks across a ladder of worn-through warps, a bridge over an opening,
// a lone warp across such a ladder, an end standing out of an edge. Every maximal run of such lone
// crossings is one thread, drawn continuously from where it leaves the cloth to where it re-enters it
// or ends at its own cut (thread.wgsl): every yarn end standing out of an edge, and every longer run
// (at least MIN_RUN crossings) held at both ends. A shorter run held at both ends (a worn-through
// float of the twill) stays part of the cloth surface. A run ends where its bundle no longer holds in
// the mechanics (emitSplit).
//
// Per crossing and family, in order:
//  - present: the field has not separated the yarn segment (cut, or pulled out: the end slid away from
//    the crossing) and its material is still in the scene. A
//    separated span's yarns belong, up to their cut, to the node on each side, and a piece that fell
//    out of the scene (retired) takes its part with it, except where its yarn is still interlaced
//    with a retained yarn of the other family: the lattice moves a bundle of yarns as one, so it
//    would otherwise carry off whole bundles along straight lattice lines; a yarn held in the
//    retained weave tears at the weave's edge instead;
//  - anchored: present and part of a continuous piece of at least three crossings (a shorter fragment,
//    cut on both sides, is held by nothing and holds nothing);
//  - cloth: both yarns anchored at all four crossings of some 2 x 2 block around it. Interlacing is a
//    two-dimensional property: a single warp surviving across a ladder of picks is a thread crossing
//    threads, not a strip of cloth;
//  - lone: anchored where the crossing is not cloth.
//
// Rebuilt only when the field changed or a piece was retired, with the edge field (its indirect
// dispatch): threadClear, threadPresence, threadAnchor, threadCloth, then threadScan finds the runs
// (one invocation per crossing, a run is walked by the invocation at its start) and appends them with
// their draw chunks; threadArgs writes the indirect draws every frame; retireCount counts retired
// nodes every frame (the rebuild test). The thread bits tell the surface which yarns it must not draw
// (threads, fragments, departed material).

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> field: array<u32>;
// per 16 crossings: bit 2 (c & 15) + f = the yarn of family f at crossing c is not part of the cloth
// surface (drawn as a thread, a fragment, or departed); then, from word words16(), 1 bit per crossing:
// its material belongs to a falling piece (debrisBit, render_common.wgsl)
@group(0) @binding(2) var<storage, read_write> threadBits: array<atomic<u32>>;
// [0] runs, [1] chunks, [2] retired nodes (this frame), [3] unused
@group(0) @binding(3) var<storage, read_write> threadCount: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> runs: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> chunks: array<u32>;
// indirect draws: threads (vertexCount, instanceCount, 0, 0), thread fibres (same); then the per-run
// preparation dispatch (x, y, z)
@group(0) @binding(6) var<storage, read_write> threadDraw: array<u32>;
@group(0) @binding(7) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> segw: array<vec4<u32>>;
// rebuild scratch: present bits (2 per crossing, 16 per word), then anchored bits (same layout), then
// cloth bits (1 per crossing, 32 per word)
@group(0) @binding(9) var<storage, read_write> scratch: array<u32>;

override MAX_RUNS: u32 = 65536u;
override MAX_CHUNKS: u32 = 262144u;

fn fieldTotalT() -> u32 {
  let F = R.layers[R.counts.y - 1u].field;
  return F.x + F.y * F.z;
}
fn words16() -> u32 { return (fieldTotalT() + 15u) / 16u; }

@compute @workgroup_size(128)
fn threadClear(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  if (w == 0u) {
    atomicStore(&threadCount[0], 0u);
    atomicStore(&threadCount[1], 0u);
  }
  if (w < words16() + (words16() + 1u) / 2u) { atomicStore(&threadBits[w], 0u); }
}

@compute @workgroup_size(128)
fn retireCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < R.counts.x && x[gid.x].w > 1.5) { atomicAdd(&threadCount[2], 1u); }
}

struct Cross {
  l: u32,
  k: u32,
  m: u32,
  valid: bool,
};

fn crossOf(c: u32) -> Cross {
  var o: Cross;
  o.l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (c >= R.layers[q].field.x) { o.l = q; }
  }
  let F = R.layers[o.l].field;
  o.valid = c < F.x + F.y * F.z;
  let local = c - F.x;
  o.k = local % F.y;
  o.m = local / F.y;
  return o;
}

// State of the node owning the yarn of family f at crossing (k, m) of layer l (the field says it has
// not separated there): 0 cloth, 1 a falling piece (debris), 2 retired. A separated span's yarn
// belongs, up to its first separation, to the span's owner node, beyond it to the other node.
fn nodeState(n: u32) -> u32 { return select(select(0u, 1u, x[n].w > 0.5), 2u, x[n].w > 1.5); }

fn ownerState(l: u32, k: u32, m: u32, f: u32) -> u32 {
  let LR = R.layers[l];
  let L = LR.lattice;
  let FL = LR.field;
  let bu = bundleOf(LR, 1u);
  let bv = bundleOf(LR, 0u);
  // the yarn's span (yarnField.ts ownSpan), its index q in the bundle and crossing p along it
  var i: u32;
  var j: u32;
  var q: u32;
  var p: u32;
  if (f == 1u) {
    if (m < bv / 2u) { return 0u; }
    i = k / bu;
    j = (m - bv / 2u) / bv;
    if (i >= L.y || j + 2u > L.z) { return 0u; }
    q = k - i * bu;
    p = m - (j * bv + bv / 2u);
  } else {
    if (k < bu / 2u) { return 0u; }
    j = m / bv;
    i = (k - bu / 2u) / bu;
    if (j >= L.z || i + 2u > L.y) { return 0u; }
    q = m - j * bv;
    p = k - (i * bu + bu / 2u);
  }
  let a = L.x + j * L.y + i;
  let b = a + select(L.y, 1u, f == 0u);
  let s = 2u * a + f;
  if ((segw[2u * s + 1u].y & SEG_BROKEN_R) == 0u) { return max(nodeState(a), nodeState(b)); }
  // separated span: the part before the yarn's first separation (cut or pulled out) is a's, after it b's
  let P = select(bu, bv, f == 1u);
  var first = P;
  for (var pp = 0u; pp < P; pp++) {
    var c: u32;
    if (f == 1u) { c = FL.x + (j * bv + bv / 2u + pp) * FL.y + i * bu + q; } else { c = FL.x + (j * bv + q) * FL.y + i * bu + bu / 2u + pp; }
    if (yarnCode(field[c], f) >= PULLED_R) {
      first = pp;
      break;
    }
  }
  return nodeState(select(b, a, p < first));
}

// One invocation per 16 crossings: the present bits of both families, and which crossings belong to
// a falling piece (threadBits debris region, debrisBit). A yarn owned by the cloth stays; a yarn of a
// falling piece goes with it, unless it is interlaced there with a yarn the cloth keeps, which tears
// it out of the piece (the lattice moves whole bundles: without this a piece would take the cloth's
// yarns along straight bundle lines, or the cloth would stay joined to a piece that has left); a
// retired piece's yarn stays only where it is interlaced with a cloth yarn.
@compute @workgroup_size(128)
fn threadPresence(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  let total = fieldTotalT();
  if (w >= words16()) { return; }
  var bits = 0u;
  var debris = 0u;
  for (var e = 0u; e < 16u; e++) {
    let c = 16u * w + e;
    if (c >= total) { break; }
    let t = crossOf(c);
    let word = field[c];
    let uncut = vec2<bool>(yarnCode(word, 0u) < PULLED_R, yarnCode(word, 1u) < PULLED_R);
    let st = vec2<u32>(select(3u, ownerState(t.l, t.k, t.m, 0u), uncut.x), select(3u, ownerState(t.l, t.k, t.m, 1u), uncut.y));
    let clothHere = st.x == 0u || st.y == 0u;
    var any = false;
    var deb = false;
    for (var f = 0u; f < 2u; f++) {
      var keep = false;
      if (st[f] == 0u) { keep = true; }
      // a falling piece's yarn: with the piece, unless the cloth keeps the yarn crossing it here
      if (st[f] == 1u && !clothHere) {
        keep = true;
        deb = true;
      }
      // a retired piece's yarn interlaced with a cloth yarn stays with the cloth
      if (st[f] == 2u && st[1u - f] == 0u) { keep = true; }
      if (keep) {
        bits |= 1u << (2u * e + f);
        any = true;
      }
    }
    if (any && deb) { debris |= 1u << e; }
  }
  scratch[w] = bits;
  if (debris != 0u) { atomicOr(&threadBits[words16() + (w >> 1u)], debris << (16u * (w & 1u))); }
}

fn presentAt(c: u32, f: u32) -> bool { return ((scratch[c >> 4u] >> (2u * (c & 15u) + f)) & 1u) == 1u; }
fn anchoredAt(c: u32, f: u32) -> bool { return ((scratch[words16() + (c >> 4u)] >> (2u * (c & 15u) + f)) & 1u) == 1u; }
fn clothAt(c: u32) -> bool { return ((scratch[2u * words16() + (c >> 5u)] >> (c & 31u)) & 1u) == 1u; }

// One invocation per 16 crossings: the anchored bits (the field's border counts as present).
@compute @workgroup_size(128)
fn threadAnchor(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  let total = fieldTotalT();
  if (w >= words16()) { return; }
  var bits = 0u;
  for (var e = 0u; e < 16u; e++) {
    let c = 16u * w + e;
    if (c >= total) { break; }
    let t = crossOf(c);
    let F = R.layers[t.l].field;
    for (var f = 0u; f < 2u; f++) {
      if (!presentAt(c, f)) { continue; }
      let pos = i32(select(t.m, t.k, f == 0u));
      let len = i32(select(F.z, F.y, f == 0u));
      let stride = i32(select(F.y, 1u, f == 0u));
      var pr: array<bool, 5>;
      for (var d = -2; d <= 2; d++) {
        let q = pos + d;
        pr[d + 2] = q < 0 || q >= len || presentAt(u32(i32(c) + d * stride), f);
      }
      if ((pr[1] && pr[3]) || (pr[3] && pr[4]) || (pr[1] && pr[0])) { bits |= 1u << (2u * e + f); }
    }
  }
  scratch[words16() + w] = bits;
}

// Both yarns anchored at crossing (k, m) of field F (outside the field: cloth).
fn interlaced(F: vec4<u32>, k: i32, m: i32) -> bool {
  if (k < 0 || m < 0 || k >= i32(F.y) || m >= i32(F.z)) { return true; }
  let c = F.x + u32(m) * F.y + u32(k);
  return anchoredAt(c, 0u) && anchoredAt(c, 1u);
}

// One invocation per 32 crossings: the cloth bits.
@compute @workgroup_size(128)
fn threadCloth(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w = gid.x;
  let total = fieldTotalT();
  if (w >= (total + 31u) / 32u) { return; }
  var bits = 0u;
  for (var e = 0u; e < 32u; e++) {
    let c = 32u * w + e;
    if (c >= total) { break; }
    let t = crossOf(c);
    let F = R.layers[t.l].field;
    let k = i32(t.k);
    let m = i32(t.m);
    if (!interlaced(F, k, m)) { continue; }
    let l = interlaced(F, k - 1, m);
    let r = interlaced(F, k + 1, m);
    let d = interlaced(F, k, m - 1);
    let u = interlaced(F, k, m + 1);
    let cloth = (l && d && interlaced(F, k - 1, m - 1)) || (r && d && interlaced(F, k + 1, m - 1)) || (l && u && interlaced(F, k - 1, m + 1)) || (r && u && interlaced(F, k + 1, m + 1));
    if (cloth) { bits |= 1u << e; }
  }
  scratch[2u * words16() + w] = bits;
}

// 0: the yarn of family f is not part of the cloth here (separated, a loose fragment, departed), 1:
// lone, 2: woven.
fn yarnState(c: u32, f: u32) -> u32 {
  if (!anchoredAt(c, f)) { return 0u; }
  return select(1u, 2u, clothAt(c));
}

// Append run [pos, pos + n) of a yarn (family f, cross index yarn) with its ends (0 held in the cloth,
// 1 free, 2 the field's border). A piece cut free at both ends has nothing holding it: it has fallen.
fn emitRun(l: u32, f: u32, yarn: u32, pos: u32, n: u32, endA: u32, endB: u32) {
  if (n == 0u || (endA == 1u && endB == 1u)) { return; }
  let r = atomicAdd(&threadCount[0], 1u);
  if (r >= MAX_RUNS) { return; }
  runs[r] = vec4<u32>(l | (f << 4u) | (endA << 5u) | (endB << 7u), yarn | (pos << 16u), n, 0u);
  // drawn over n + 1 crossings: half a crossing beyond each end (into the weave, or to the cut)
  let nc = n / CHUNK + 1u;
  let c0 = atomicAdd(&threadCount[1], nc);
  for (var q = 0u; q < nc; q++) {
    if (c0 + q < MAX_CHUNKS) { chunks[c0 + q] = (r << 10u) | q; }
  }
}

// The run as the mechanics holds it. A bundle span that no longer holds (every yarn in it separated,
// or the bundle slipped out of a crossing) carries no yarn across it: its material went to the two
// nodes as tails, split at the span's failure site (yarnfield.wgsl couple). A yarn of that bundle may
// still be uncut along its own lone run (its cut lies just past where it is held), but nothing joins
// its two sides any more: the run ends there, each part hanging from its own side. Drawing it held at
// both ends would bridge an opening the mechanics opened, straight through whatever lies between.
fn emitSplit(l: u32, f: u32, yarn: u32, pos: u32, n: u32, endA: u32, endB: u32) {
  // a yarn with a free end already hangs from its held side (thread_geom.wgsl runCenter)
  if (endA == 1u || endB == 1u) {
    emitRun(l, f, yarn, pos, n, endA, endB);
    return;
  }
  let LR = R.layers[l];
  let L = LR.lattice;
  let bu = bundleOf(LR, 1u);
  let bv = bundleOf(LR, 0u);
  let Bal = select(bv, bu, f == 0u);
  let Bcr = select(bu, bv, f == 0u);
  let len = select(L.z, L.y, f == 0u);
  let b = yarn / Bcr;
  // crossings the drawn run covers, held ends included
  let lo = select(pos, pos - 1u, endA == 0u && pos > 0u);
  let hi = select(pos + n - 1u, pos + n, endB == 0u);
  let half = Bal / 2u;
  let a0 = select(0u, (lo - half) / Bal, lo >= half);
  let a1 = min(select(0u, (hi - half) / Bal, hi >= half), len - 2u);
  var cur = pos;
  var curA = endA;
  for (var a = a0; a <= a1; a++) {
    let owner = L.x + select(a * L.y + b, b * L.y + a, f == 0u);
    let w = segw[2u * (2u * owner + f) + 1u];
    if ((w.y & (SEG_EXISTS_R | SEG_BROKEN_R)) == SEG_EXISTS_R) { continue; }
    let site = a * Bal + half + u32(round(clamp(bitcast<f32>(w.z), 0.0, 1.0) * f32(Bal)));
    let sp = clamp(site, cur, pos + n);
    if (sp > cur) { emitRun(l, f, yarn, cur, sp - cur, curA, 1u); }
    cur = sp;
    curA = 1u;
  }
  emitRun(l, f, yarn, cur, pos + n - cur, curA, endB);
}

@compute @workgroup_size(128)
fn threadScan(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  let t = crossOf(c);
  if (!t.valid) { return; }
  let F = R.layers[t.l].field;
  for (var f = 0u; f < 2u; f++) {
    let st = yarnState(c, f);
    // a fragment that fell out of the weave, or material that left with a retired piece: not drawn,
    // and no longer cloth
    if (st == 0u && yarnCode(field[c], f) < PULLED_R) { atomicOr(&threadBits[c >> 4u], 1u << (2u * (c & 15u) + f)); }
    if (st != 1u) { continue; }
    // along the yarn: a pick (f = 0) runs over the warps k, a warp over the picks m
    let pos = select(t.m, t.k, f == 0u);
    let len = select(F.z, F.y, f == 0u);
    let stride = select(F.y, 1u, f == 0u);
    if (pos > 0u && yarnState(c - stride, f) == 1u) { continue; }
    var n = 1u;
    loop {
      if (pos + n >= len || n >= 4096u || yarnState(c + n * stride, f) != 1u) { break; }
      n++;
    }
    // ends: 0 held in the cloth, 1 free (the yarn itself ends there), 2 the field's border
    var endA = 2u;
    if (pos > 0u) { endA = select(0u, 1u, yarnState(c - stride, f) == 0u); }
    var endB = 2u;
    if (pos + n < len) { endB = select(0u, 1u, yarnState(c + n * stride, f) == 0u); }
    // a short run held at both ends is a worn-through float of the cloth; a yarn end standing out of
    // an edge is a thread however short
    if (n < MIN_RUN && endA != 1u && endB != 1u) { continue; }
    emitSplit(t.l, f, select(t.k, t.m, f == 0u), pos, n, endA, endB);
    for (var q = 0u; q < n; q++) {
      let cc = c + q * stride;
      atomicOr(&threadBits[cc >> 4u], 1u << (2u * (cc & 15u) + f));
    }
  }
}

@compute @workgroup_size(1)
fn threadArgs() {
  let nc = min(atomicLoad(&threadCount[1]), MAX_CHUNKS);
  threadDraw[0] = THREAD_VERTS;
  threadDraw[1] = nc;
  threadDraw[2] = 0u;
  threadDraw[3] = 0u;
  threadDraw[4] = FIBER_VERTS;
  threadDraw[5] = nc * FIBERS_PER_CHUNK;
  threadDraw[6] = 0u;
  threadDraw[7] = 0u;
  // per-run preparation (thread_prep.wgsl), 64 runs per workgroup
  threadDraw[8] = (min(atomicLoad(&threadCount[0]), MAX_RUNS) + 63u) / 64u;
  threadDraw[9] = 1u;
  threadDraw[10] = 1u;
}
