// Newton line search: change of the incremental potential along the Newton direction (delta) for
// step lengths alpha in {1, 1/2, 1/4, 1/8}, as sums of per-term differences (stable in f32: no
// subtraction of large totals), each with a rounding scale (the sum of the magnitudes of its terms)
// that decides whether a difference is conclusive (lineChoose). Each node owns its inertia/external/drag terms, its "+" spans, the
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
// [3] receives the chosen step length (f32 bits); [12] counts directions along which no evaluated
// step decreased the potential
@group(0) @binding(9) var<storage, read_write> counters: array<u32>;

const GROUPS: u32 = 256u;
var<workgroup> red: array<vec4<f32>, 256>;
var<workgroup> redM: array<vec4<f32>, 256>;
// per-node energy changes and their magnitude sums (rounding scale) for the four step lengths
var<private> acc: vec4<f32>;
var<private> mag: vec4<f32>;

fn add4(v: vec4<f32>) {
  acc += v;
  mag += abs(v);
}
fn add1(k: u32, v: f32) {
  acc[k] += v;
  mag[k] += abs(v);
}

fn lat(l: u32) -> vec4<u32> { return P.layers[l].lattice; }

fn nodeAt(l: u32, i: i32, j: i32) -> i32 {
  let L = lat(l);
  if (i < 0 || j < 0 || i >= i32(L.y) || j >= i32(L.z)) { return -1; }
  return i32(L.x) + j * i32(L.y) + i;
}

fn pinned(n: u32) -> bool { return (nodes[n].flags & NODE_PINNED) != 0u || x[n].w > 0.5; }

fn spanOn(owner: i32, f: u32) -> bool {
  if (owner < 0) { return false; }
  return segActive(segs[2u * u32(owner) + f]);
}

// alpha-scaled direction (pinned nodes never move)
fn dir(n: u32) -> vec3<f32> { return select(delta[n].xyz, vec3<f32>(0.0), pinned(n)); }

const ALPHA: vec4<f32> = vec4<f32>(1.0, 0.5, 0.25, 0.125);

// Quadratic term change: alpha g.d + 1/2 alpha^2 q for the four step lengths.
fn quad(gd: f32, q: f32) {
  acc += ALPHA * gd + 0.5 * ALPHA * ALPHA * q;
  mag += abs(ALPHA * gd) + abs(0.5 * ALPHA * ALPHA * q);
}

// |e0 + a de| - |e0| computed without cancellation.
fn lenChange(e0: vec3<f32>, de: vec3<f32>, a: f32, l0: f32) -> vec2<f32> {
  let la = length(e0 + a * de);
  return vec2<f32>((2.0 * a * dot(e0, de) + a * a * dot(de, de)) / max(la + l0, 1e-12), la);
}

fn nodeDelta(n: u32) {
  acc = vec4<f32>(0.0);
  mag = vec4<f32>(0.0);
  let st = nodes[n];
  let i = i32(st.ijl & 0xFFFu);
  let j = i32((st.ijl >> 12u) & 0xFFFu);
  let l = st.ijl >> 24u;
  let mat = P.layers[l];
  let h = P.hg.x;
  let d0 = dx[n].xyz;
  let dn = dir(n);
  // a mounted node has no inertial, external or drag terms, but the spans, bending triples and shear
  // cells it owns still act on their free nodes (the gradient includes them)
  if (!pinned(n)) {
    // inertia
    let w = st.mass / (h * h);
    quad(w * dot(dn, d0 - pred[n].xyz), w * dot(dn, dn));
    // external terms (grip, contact): quadratic model from their gradient and Hessian at this iterate
    let e0 = ext[3u * n];
    let e1 = ext[3u * n + 1u];
    let e2 = ext[3u * n + 2u];
    let He = Sym(vec3<f32>(e1.x, e1.w, e2.y), vec3<f32>(e1.y, e1.z, e2.x));
    quad(dot(e0.xyz, dn), dot(dn, symMul(He, dn)));
    // air drag (drag.wgsl): quadratic with a fixed block
    let Dd = airDragBlock(n);
    quad(dot(symMul(Dd, d0), dn), dot(dn, symMul(Dd, dn)));
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
      let T0 = spanResponse(y, s, l0 / L - 1.0).x;
      let c = y.axialVisc / (L * h);
      for (var k = 0u; k < 4u; k++) {
        let al = ALPHA;
        let r = lenChange(ec, de, al[k], l0);
        let Ta = spanResponse(y, s, r.y / L - 1.0).x;
        add1(k, 0.5 * (T0 + Ta) * r.x);
        add1(k, 0.5 * c * r.x * (r.y + l0 - 2.0 * ln));
      }
    }
    // bending triple centred at n: (prev, n, next)
    let prev = nodeAt(l, i - du, j - dv);
    if (prev >= 0 && next >= 0 && spanOn(prev, f) && spanOn(i32(n), f)) {
      let a = u32(prev);
      let c = u32(next);
      let sa = segs[2u * a + f];
      let sn = segs[2u * n + f];
      let geo = bendGeometry(sa.rest, sn.rest);
      let ke = y.bendRigidity * bendScale(sa.intact, sn.intact, sa.cutMask, sn.cutMask, u32(y.yarnsPerSpan)) * geo.z;
      let kd = ke * y.bendVisc / h;
      let rn = geo.x * x[a].xyz - 2.0 * x[n].xyz + geo.y * x[c].xyz;
      let r0 = rn + geo.x * dx[a].xyz - 2.0 * d0 + geo.y * dx[c].xyz;
      let dr = geo.x * dir(a) - 2.0 * dn + geo.y * dir(c);
      // elastic: exact change of the fold-stiffening energy; viscous: quadratic
      quad(kd * dot(r0 - rn, dr), kd * dot(dr, dr));
      let q0 = dot(r0, r0);
      let ab = geo.w;
      for (var k = 0u; k < 4u; k++) {
        let alv = ALPHA;
        let al = alv[k];
        let dq = 2.0 * al * dot(r0, dr) + al * al * dot(dr, dr);
        add1(k, 0.5 * ke * bendPsiDelta(q0, q0 + dq, dq, ab));
      }
    }
  }
  // shear diagonals of the cell with lower-left corner n
  let r = nodeAt(l, i + 1, j);
  let t = nodeAt(l, i, j + 1);
  let rt = nodeAt(l, i + 1, j + 1);
  if (r >= 0 && t >= 0) {
    // a diagonal acts while one of its triangles is intact (diagActive in assemble.wgsl)
    let cu = spanOn(i32(n), 0u);
    let cv = spanOn(i32(n), 1u);
    let rv = spanOn(r, 1u);
    let tu = spanOn(t, 0u);
    let on = array<bool, 2>((cu && rv) || (cv && tu), (cu && cv) || (rv && tu));
    let s0 = segs[2u * n].rest;
    let s1 = segs[2u * n + 1u].rest;
    let Ld = sqrt(s0 * s0 + s1 * s1);
    let a = cellArea(select(0.0, segs[2u * n].intact, cu), select(0.0, segs[2u * u32(t)].intact, tu), select(0.0, segs[2u * n + 1u].intact, cv), select(0.0, segs[2u * u32(r) + 1u].intact, rv));
    let ks = mat.a.x * st.n0 / mat.b.z;
    let jam = shearJam(a);
    var pa = array<u32, 2>(n, u32(r));
    var pb = array<u32, 2>(u32(rt), u32(t));
    for (var q = 0; q < 2; q++) {
      if (!on[q]) { continue; }
      let p = pa[q];
      let o = pb[q];
      let ec = (x[o].xyz + dx[o].xyz) - (x[p].xyz + dx[p].xyz);
      let de = dir(o) - dir(p);
      let l0 = length(ec);
      let f0 = shearLaw(ks, jam, Ld, l0).x;
      for (var k = 0u; k < 4u; k++) {
        let al = ALPHA;
        let rr = lenChange(ec, de, al[k], l0);
        // trapezoid of the shear force law (exact on its linear branch)
        add1(k, 0.5 * (f0 + shearLaw(ks, jam, Ld, rr.y).x) * rr.x);
      }
    }
  }
}

@compute @workgroup_size(256)
fn energyDelta(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  var sum = vec4<f32>(0.0);
  var sumM = vec4<f32>(0.0);
  for (var n = gid.x; n < P.counts.x; n += 256u * GROUPS) {
    nodeDelta(n);
    // firewall: one non-finite node must not decide the global step length
    if (finite3(acc.xyz) && finite1(acc.w) && finite3(mag.xyz) && finite1(mag.w)) {
      sum += acc;
      sumM += mag;
    }
  }
  red[lid] = sum;
  redM[lid] = sumM;
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) {
      red[lid] += red[lid + s];
      redM[lid] += redM[lid + s];
    }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid == 0u) {
    partial4[2u * wg.x] = red[0];
    partial4[2u * wg.x + 1u] = redM[0];
  }
}

// Relative rounding bound of an energy change: f32 per-term rounding accumulated over the
// reduction tree, as a multiple of the sum of the terms' magnitudes (newtonMultigrid.ts LINE_NOISE).
const LINE_NOISE: f32 = 4e-6;

// Step length from the evaluated changes (newtonMultigrid.ts chooseStep): the lowest potential if it is
// a conclusive decrease; no step if every evaluated step conclusively increases the potential (a
// non-descent direction, counted); otherwise the changes are within rounding (a converged iterate) and
// the full Newton step is taken. No unevaluated step is ever taken on a conclusive increase.
@compute @workgroup_size(256)
fn lineChoose(@builtin(local_invocation_index) lid: u32) {
  red[lid] = partial4[2u * lid];
  redM[lid] = partial4[2u * lid + 1u];
  workgroupBarrier();
  var s = 128u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) {
      red[lid] += red[lid + s];
      redM[lid] += redM[lid + s];
    }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid == 0u) {
    let e = red[0];
    let tol = LINE_NOISE * redM[0];
    var best = 0u;
    for (var k = 1u; k < 4u; k++) {
      if (e[k] < e[best]) { best = k; }
    }
    var a = 1.0;
    if (e[best] < -tol[best]) {
      a = ALPHA[best];
    } else if (all(e > tol)) {
      a = 0.0;
      counters[12] += 1u;
    }
    counters[3] = bitcast<u32>(a);
  }
}
