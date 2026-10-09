// Newton line search: change of the incremental potential along the Newton direction (delta) for
// step lengths alpha in {1, 1/2, 1/4}, as sums of per-term differences (stable in f32: no
// subtraction of large totals). Each node owns its inertia/external/drag terms, its "+" spans, the
// bending triples centred on it and the shear diagonals of the cell it is the lower-left corner of.
// Mirrors the reference line search (reference/newtonMultigrid.ts) up to the axial quadrature.

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> dx: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> pred: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> segs: array<Seg>;
@group(0) @binding(6) var<storage, read> delta: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> ext: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> partial4: array<vec4<f32>>;
// [3] receives the chosen step length (f32 bits)
@group(0) @binding(9) var<storage, read_write> counters: array<u32>;

const GROUPS: u32 = 256u;
var<workgroup> red: array<vec4<f32>, 256>;

fn lat(l: u32) -> vec4<u32> { return P.layers[l].lattice; }

fn nodeAt(l: u32, i: i32, j: i32) -> i32 {
  let L = lat(l);
  if (i < 0 || j < 0 || i >= i32(L.y) || j >= i32(L.z)) { return -1; }
  return i32(L.x) + j * i32(L.y) + i;
}

fn pinned(n: u32) -> bool { return (nodes[n].flags & NODE_PINNED) != 0u; }

fn spanOn(owner: i32, f: u32) -> bool {
  if (owner < 0) { return false; }
  return segActive(segs[2u * u32(owner) + f]);
}

// alpha-scaled direction (pinned nodes never move)
fn dir(n: u32) -> vec3<f32> { return select(delta[n].xyz, vec3<f32>(0.0), pinned(n)); }

const ALPHA: vec3<f32> = vec3<f32>(1.0, 0.5, 0.25);

// Quadratic term change: alpha g.d + 1/2 alpha^2 q for the three step lengths.
fn quad(gd: f32, q: f32) -> vec3<f32> { return ALPHA * gd + 0.5 * ALPHA * ALPHA * q; }

// |e0 + a de| - |e0| computed without cancellation.
fn lenChange(e0: vec3<f32>, de: vec3<f32>, a: f32, l0: f32) -> vec2<f32> {
  let la = length(e0 + a * de);
  return vec2<f32>((2.0 * a * dot(e0, de) + a * a * dot(de, de)) / max(la + l0, 1e-12), la);
}

fn nodeDelta(n: u32) -> vec3<f32> {
  let st = nodes[n];
  if ((st.flags & NODE_PINNED) != 0u) { return vec3<f32>(0.0); }
  let i = i32(st.ijl & 0xFFFu);
  let j = i32((st.ijl >> 12u) & 0xFFFu);
  let l = st.ijl >> 24u;
  let mat = P.layers[l];
  let h = P.hg.x;
  let d0 = dx[n].xyz;
  let dn = delta[n].xyz;
  // inertia
  let w = st.mass / (h * h);
  var acc = quad(w * dot(dn, d0 - pred[n].xyz), w * dot(dn, dn));
  // external terms (grip, contact): quadratic model from their gradient and Hessian at this iterate
  let e0 = ext[3u * n];
  let e1 = ext[3u * n + 1u];
  let e2 = ext[3u * n + 2u];
  let He = Sym(vec3<f32>(e1.x, e1.w, e2.y), vec3<f32>(e1.y, e1.z, e2.x));
  acc += quad(dot(e0.xyz, dn), dot(dn, symMul(He, dn)));
  // normal air drag (normal of the accepted state, see assemble.wgsl)
  if (mat.b.y > 0.0) {
    let a = nodeAt(l, i - 1, j);
    let b = nodeAt(l, i + 1, j);
    let c = nodeAt(l, i, j - 1);
    let e = nodeAt(l, i, j + 1);
    let pa = x[u32(select(a, i32(n), a < 0))].xyz;
    let pb = x[u32(select(b, i32(n), b < 0))].xyz;
    let pc = x[u32(select(c, i32(n), c < 0))].xyz;
    let pe = x[u32(select(e, i32(n), e < 0))].xyz;
    let nn = cross(pb - pa, pe - pc);
    let len = length(nn);
    if (len > 1e-12) {
      let nu = nn / len;
      let cd = mat.b.y * mat.a.w * mat.b.x / h;
      acc += quad(cd * dot(nu, d0) * dot(nu, dn), cd * dot(nu, dn) * dot(nu, dn));
    }
  }
  for (var f = 0u; f < 2u; f++) {
    let du = select(0, 1, f == 0u);
    let dv = select(0, 1, f == 1u);
    let y = mat.yarn[f];
    // the span n -> next (axial: trapezoid of the damaged force law; viscosity exact)
    let next = nodeAt(l, i + du, j + dv);
    if (next >= 0 && spanOn(i32(n), f)) {
      let b = u32(next);
      let s = segs[2u * n + f];
      let L = s.rest;
      let en = x[b].xyz - x[n].xyz;
      let ec = en + dx[b].xyz - d0;
      let de = dir(b) - dn;
      let l0 = length(ec);
      let ln = length(en);
      let T0 = damagedResponse(y, L, s, l0 / L - 1.0).x;
      let c = y.axialVisc / (L * h);
      for (var k = 0u; k < 3u; k++) {
        let al = ALPHA;
        let r = lenChange(ec, de, al[k], l0);
        let Ta = damagedResponse(y, L, s, r.y / L - 1.0).x;
        acc[k] += 0.5 * (T0 + Ta) * r.x + 0.5 * c * r.x * (r.y + l0 - 2.0 * ln);
      }
    }
    // bending triple centred at n: (prev, n, next)
    let prev = nodeAt(l, i - du, j - dv);
    if (prev >= 0 && next >= 0 && spanOn(prev, f) && spanOn(i32(n), f)) {
      let a = u32(prev);
      let c = u32(next);
      let Lm = 0.5 * (segs[2u * a + f].rest + segs[2u * n + f].rest);
      let ke = y.bendRigidity / (Lm * Lm * Lm);
      let kd = ke * y.bendVisc / h;
      let rn = x[a].xyz - 2.0 * x[n].xyz + x[c].xyz;
      let r0 = rn + dx[a].xyz - 2.0 * d0 + dx[c].xyz;
      let dr = dir(a) - 2.0 * dn + dir(c);
      acc += quad(ke * dot(r0, dr) + kd * dot(r0 - rn, dr), (ke + kd) * dot(dr, dr));
    }
  }
  // shear diagonals of the cell with lower-left corner n
  let r = nodeAt(l, i + 1, j);
  let t = nodeAt(l, i, j + 1);
  let rt = nodeAt(l, i + 1, j + 1);
  if (r >= 0 && t >= 0 && spanOn(i32(n), 0u) && spanOn(i32(n), 1u) && spanOn(r, 1u) && spanOn(t, 0u)) {
    let s0 = segs[2u * n].rest;
    let s1 = segs[2u * n + 1u].rest;
    let Ld = sqrt(s0 * s0 + s1 * s1);
    let ks = mat.a.x;
    var pa = array<u32, 2>(n, u32(r));
    var pb = array<u32, 2>(u32(rt), u32(t));
    for (var q = 0; q < 2; q++) {
      let p = pa[q];
      let o = pb[q];
      let ec = (x[o].xyz + dx[o].xyz) - (x[p].xyz + dx[p].xyz);
      let de = dir(o) - dir(p);
      let l0 = length(ec);
      for (var k = 0u; k < 3u; k++) {
        let al = ALPHA;
        let rr = lenChange(ec, de, al[k], l0);
        acc[k] += 0.5 * ks * rr.x * (rr.y + l0 - 2.0 * Ld);
      }
    }
  }
  return acc;
}

@compute @workgroup_size(256)
fn energyDelta(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  var acc = vec3<f32>(0.0);
  for (var n = gid.x; n < P.counts.x; n += 256u * GROUPS) { acc += nodeDelta(n); }
  red[lid] = vec4<f32>(acc, 0.0);
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid == 0u) { partial4[wg.x] = red[0]; }
}

// Step length with the lowest potential; if none of them decreases it (a non-descent or converged
// direction at f32 resolution), a short safe step.
@compute @workgroup_size(256)
fn lineChoose(@builtin(local_invocation_index) lid: u32) {
  red[lid] = partial4[lid];
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid == 0u) {
    let e = red[0].xyz;
    var a = 1.0;
    var best = e.x;
    if (e.y < best) {
      a = 0.5;
      best = e.y;
    }
    if (e.z < best) {
      a = 0.25;
      best = e.z;
    }
    if (best > 0.0) { a = 0.125; }
    counters[3] = bitcast<u32>(a);
  }
}
