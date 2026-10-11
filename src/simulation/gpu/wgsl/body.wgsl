// Contact of the cloth with a kinematic body made of capsules (a staged creature's limbs, head and
// torso). Per Newton iteration, each cloth node inside the contact distance of a capsule gets a penalty
// term along the capsule's outward normal and Coulomb friction (regularised) on its step relative to
// the body's surface; gradient and Hessian are added to ext after the self-contact terms (which
// overwrite it) and before the grips (which add theirs). The body is prescribed: it takes no reaction.
// Contact is swept: a node's step and a capsule's motion over the substep are both linear, and where
// the node's path relative to the capsule enters it, the node meets the surface where it first touched
// (the tangent plane there, carried with the capsule to its end pose) instead of the closest point of
// its end position, which lies on the far side once a fast limb or a large step has crossed it.
// Claws pierce: once per substep (after commit), a node under a sharp capsule's point (beyond the end of
// its segment, within the point's half-width of its axis) that the capsule loads past the yarns'
// piercing force has every crossing within the point's half-width cut (the field update and span
// coupling part them). Nodes a claw holds (exempt 1) are on the claw: they neither collide with the
// body nor are pierced. Nodes it let go (exempt 2) slide off it: they meet the body again once clear.

const BODY_MAX: u32 = 48u;

struct Capsule {
  // segment start (xyz), radius (w), at the end of the substep
  a: vec4<f32>,
  // segment end (xyz), 1 if the capsule is in contact this substep (w)
  b: vec4<f32>,
  // displacement of the start and end over the substep (xyz); w = 1 if its end is a claw's point
  da: vec4<f32>,
  db: vec4<f32>,
};

struct BodyU {
  // capsule count, unused...
  count: vec4<u32>,
  // penalty stiffness (N/m), friction coefficient, regularisation speed x h (m), contact distance (m)
  law: vec4<f32>,
  // bounds of every capsule over the substep, start and end poses (expanded by its radius and the
  // contact distance): lo xyz, hi xyz
  lo: vec4<f32>,
  hi: vec4<f32>,
  // piercing force of the yarns under a claw's point (N), the point's half-width (m), unused...
  pierce: vec4<f32>,
  caps: array<Capsule, BODY_MAX>,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<uniform> B: BodyU;
@group(0) @binding(2) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> dx: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> ext: array<vec4<f32>>; // 3 per node (see assemble.wgsl)
// 1 where a claw holds the node, 2 where it let go of it and the node has not yet come clear of the body
@group(0) @binding(6) var<storage, read_write> exempt: array<u32>;
// yarn damage field (yarnfield.wgsl) and its layout per layer: first crossing, crossings per row, rows
@group(0) @binding(7) var<storage, read_write> field: array<u32>;
@group(0) @binding(8) var<uniform> FU: array<vec4<u32>, 3>;
@group(0) @binding(9) var<storage, read_write> fieldEvents: array<u32>;

// Distance of p from the surface of capsule (a, b, radius r) (x) and the closest point's parameter (y).
fn capsuleGap(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r: f32) -> vec2<f32> {
  let e = b - a;
  let ee = dot(e, e);
  let s = select(0.0, clamp(dot(p - a, e) / ee, 0.0, 1.0), ee > 1e-12);
  return vec2<f32>(length(p - a - s * e) - r, s);
}

// First time in (0, 1] at which the point x0 + t dp enters capsule C (each end moving linearly over the
// substep to its pose at t = 1), or -1 where it does not (or starts inside: the closest point then
// decides). Conservative advancement: the gap shrinks no faster than the largest speed of the point
// relative to any point of the segment, which is at one of its ends.
const ENTRY_STEPS: i32 = 16;

fn entryTime(x0: vec3<f32>, dp: vec3<f32>, C: Capsule) -> f32 {
  let a0 = C.a.xyz - C.da.xyz;
  let b0 = C.b.xyz - C.db.xyz;
  let vmax = max(length(dp - C.da.xyz), length(dp - C.db.xyz));
  var g = capsuleGap(x0, a0, b0, C.a.w).x;
  if (g <= 0.0 || g >= vmax) { return -1.0; }
  let tol = 1e-3 * C.a.w;
  var t = 0.0;
  for (var k = 0; k < ENTRY_STEPS; k++) {
    t += g / vmax;
    if (t >= 1.0) { return -1.0; }
    g = capsuleGap(x0 + t * dp, a0 + t * C.da.xyz, b0 + t * C.db.xyz, C.a.w).x;
    if (g <= tol) { return t; }
  }
  return -1.0;
}

// w rotated by the rotation taking unit u to unit v (the shortest arc).
fn rotateArc(w: vec3<f32>, u: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  let c = dot(u, v);
  if (c < -0.99) { return w; }
  let k = cross(u, v);
  return c * w + cross(k, w) + k * (dot(k, w) / (1.0 + c));
}

@compute @workgroup_size(128)
fn bodyTerms(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x || B.count.x == 0u) { return; }
  let ex = exempt[n];
  if ((nodes[n].flags & 1u) != 0u || x[n].w > 1.5 || ex == 1u) { return; }
  let x0 = x[n].xyz;
  let p = x0 + dx[n].xyz;
  if (any(min(x0, p) > B.hi.xyz) || any(max(x0, p) < B.lo.xyz)) {
    if (ex == 2u) { exempt[n] = 0u; }
    return;
  }
  let kc = B.law.x;
  let mu = B.law.y;
  let epsU = B.law.z;
  let delta = B.law.w;
  var g = vec3<f32>(0.0);
  var H0 = vec3<f32>(0.0); // xx yy zz
  var H1 = vec3<f32>(0.0); // xy xz yz
  var touched = false;
  for (var c = 0u; c < min(B.count.x, BODY_MAX); c++) {
    let C = B.caps[c];
    if (C.b.w < 0.5) { continue; }
    let e = C.b.xyz - C.a.xyz;
    var s: f32;
    var d: f32;
    var nn: vec3<f32>;
    let te = entryTime(x0, dx[n].xyz, C);
    if (te >= 0.0) {
      // the path entered the capsule: the tangent plane where it first touched, moved with the capsule
      let at = C.a.xyz - (1.0 - te) * C.da.xyz;
      let bt = C.b.xyz - (1.0 - te) * C.db.xyz;
      let xt = x0 + te * dx[n].xyz;
      s = capsuleGap(xt, at, bt, C.a.w).y;
      let rt = xt - at - s * (bt - at);
      let lt = length(rt);
      if (lt <= 1e-7) { continue; }
      let et = bt - at;
      nn = rt / lt;
      if (dot(et, et) > 1e-12 && dot(e, e) > 1e-12) { nn = normalize(rotateArc(nn, normalize(et), normalize(e))); }
      d = dot(p - C.a.xyz - s * e, nn) - C.a.w;
      if (d >= delta) { continue; }
    } else {
      let ee = dot(e, e);
      s = select(0.0, clamp(dot(p - C.a.xyz, e) / ee, 0.0, 1.0), ee > 1e-12);
      let r = p - C.a.xyz - s * e;
      let dist = length(r);
      d = dist - C.a.w;
      if (d >= delta || dist <= 1e-7) { continue; }
      nn = r / dist;
    }
    let lam = kc * (delta - d);
    // friction on the step relative to the body's surface there
    let rel = dx[n].xyz - mix(C.da.xyz, C.db.xyz, s);
    let ut = rel - dot(rel, nn) * nn;
    let kf = mu * lam / max(length(ut), epsU);
    g += -lam * nn + kf * ut;
    let a = kc - kf;
    H0 += a * nn * nn + vec3<f32>(kf);
    H1 += a * vec3<f32>(nn.x * nn.y, nn.x * nn.z, nn.y * nn.z);
    touched = true;
  }
  if (ex == 2u) {
    if (!touched) { exempt[n] = 0u; }
    return;
  }
  if (!touched) { return; }
  let b = 3u * n;
  ext[b] = ext[b] + vec4<f32>(g, 0.0);
  ext[b + 1u] = ext[b + 1u] + vec4<f32>(H0.x, H1.x, H1.y, H0.y);
  ext[b + 2u] = ext[b + 2u] + vec4<f32>(H1.z, H0.z, 0.0, 0.0);
}

// Cut both yarns of every crossing within `radius` of node n's place in the weave (a node sits at
// the centre of its span's bundle: crossing (i bu + (bu - 1) / 2, j bv + (bv - 1) / 2)).
fn cutAround(n: u32, radius: f32) {
  let ijl = nodes[n].ijl;
  let l = ijl >> 24u;
  let L = P.layers[l];
  let bu = u32(L.yarn[1].yarnsPerSpan);
  let bv = u32(L.yarn[0].yarnsPerSpan);
  let pu = L.a.w / f32(bu);
  let pv = L.b.x / f32(bv);
  let k0 = f32((ijl & 0xFFFu) * bu) + 0.5 * f32(bu - 1u);
  let m0 = f32(((ijl >> 12u) & 0xFFFu) * bv) + 0.5 * f32(bv - 1u);
  let F = FU[l];
  let mLo = u32(max(0.0, ceil(m0 - radius / pv)));
  let mHi = u32(min(f32(F.z) - 1.0, floor(m0 + radius / pv)));
  for (var m = mLo; m <= mHi; m++) {
    let dv = (f32(m) - m0) * pv;
    let half = sqrt(max(0.0, radius * radius - dv * dv)) / pu;
    let kLo = u32(max(0.0, ceil(k0 - half)));
    let kHi = min(F.y - 1u, u32(max(0.0, floor(k0 + half))));
    for (var k = kLo; k <= kHi; k++) {
      let c = F.x + m * F.y + k;
      field[c] = field[c] | 0xFFFFu;
    }
  }
}

@compute @workgroup_size(128)
fn pierce(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x || B.count.x == 0u) { return; }
  if ((nodes[n].flags & 1u) != 0u || x[n].w > 1.5 || exempt[n] != 0u) { return; }
  let p = x[n].xyz;
  if (any(p < B.lo.xyz) || any(p > B.hi.xyz)) { return; }
  let kc = B.law.x;
  let delta = B.law.w;
  for (var c = 0u; c < min(B.count.x, BODY_MAX); c++) {
    let C = B.caps[c];
    if (C.b.w < 0.5 || C.da.w < 0.5) { continue; }
    // only under the point: past the end of the segment and within the point's half-width of its axis
    // (the finger's rounded tip around the nail is blunt)
    let ax = normalize(C.b.xyz - C.a.xyz);
    let v = p - C.b.xyz;
    let along = dot(v, ax);
    if (along <= 0.0 || length(v - along * ax) > B.pierce.y) { continue; }
    let d = length(v) - C.a.w;
    if (kc * (delta - d) < B.pierce.x) { continue; }
    cutAround(n, B.pierce.y);
    fieldEvents[0] = 1u;
    return;
  }
}
