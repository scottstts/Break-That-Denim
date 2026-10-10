// Fine-level Newton system assembly: projected Hessian row (upper-half stencil) and rhs = -grad.
// Mirrors ReferenceSolver.nodeGradient and assembleFineOperator (reference/newtonMultigrid.ts).

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> dx: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> pred: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> segs: array<Seg>;
@group(0) @binding(6) var<storage, read_write> op: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> rhs: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> ext: array<vec4<f32>>; // 3 per node: grad, H(xx xy xz yy), H(yz zz)

fn fineLevel() -> LevelInfo {
  var lv: LevelInfo;
  lv.counts = P.counts;
  for (var l = 0u; l < 3u; l++) { lv.layers[l] = P.layers[l].lattice; }
  return lv;
}

fn isPinned(m: i32) -> bool { return (nodes[u32(m)].flags & NODE_PINNED) != 0u || x[u32(m)].w > 0.5; }

fn segAt(owner: i32, f: u32) -> Seg { return segs[2u * u32(owner) + f]; }

fn activeSeg(owner: i32, f: u32) -> bool {
  if (owner < 0) { return false; }
  return segActive(segAt(owner, f));
}

struct SegForce {
  fs: f32,
  kPar: f32,
  kPerp: f32,
  e: vec3<f32>,
};

// Axial response of the span owned by `a` toward `b` (family f) at the current iterate.
fn segForce(a: u32, b: u32, f: u32, mat: LayerMat) -> SegForce {
  let en = x[b].xyz - x[a].xyz;
  let e = en + (dx[b].xyz - dx[a].xyz);
  let l = length(e);
  let ln = length(en);
  let s = segs[2u * a + f];
  let y = mat.yarn[f];
  let L = s.rest;
  let r = spanResponse(y, s, l / L - 1.0);
  let c = y.axialVisc / (L * P.hg.x);
  let fs = r.x + c * (l - ln);
  return SegForce(fs, max(r.y, 0.0) / L + c, max(fs, 0.0) / l, e / l);
}

// Bending of the triple (a, b, c) centered at b (fold-stiffening energy, see bendPsi): the gradient
// coefficient ke Psi' and a scalar stiffness k = ke (Psi' + 2 Psi'' q) + kd that majorizes the
// exact Hessian ke (Psi' I + 2 Psi'' r r^T) + kd I (the operator stores this stencil as scalars).
struct Triple {
  ke: f32,
  kd: f32,
  k: f32,
  r: vec3<f32>,
  rn: vec3<f32>,
};

fn triple(a: u32, b: u32, c: u32, f: u32, mat: LayerMat) -> Triple {
  let L = 0.5 * (segs[2u * a + f].rest + segs[2u * b + f].rest);
  let y = mat.yarn[f];
  let kb = y.bendRigidity / (L * L * L);
  let kd = kb * y.bendVisc / P.hg.x;
  let rn = x[a].xyz - 2.0 * x[b].xyz + x[c].xyz;
  let r = rn + dx[a].xyz - 2.0 * dx[b].xyz + dx[c].xyz;
  let q = dot(r, r);
  let ps = bendPsi(q, 4.0 * L * L);
  return Triple(kb * ps.y, kd, kb * (ps.y + 2.0 * ps.z * q) + kd, r, rn);
}

fn tripleGrad(t: Triple) -> vec3<f32> { return t.ke * t.r + t.kd * (t.r - t.rn); }

// Shear diagonal d of the cell with lower-left corner (ci, cj): d = 0 joins c and rt, d = 1 joins r
// and t. A diagonal acts while one of its two triangles is intact (the crossings there still lock
// the yarn angle), so a cell keeps its shear stiffness when a single span breaks and hole edges do
// not turn into free hinges.
fn diagActive(lv: LevelInfo, l: u32, ci: i32, cj: i32, d: u32) -> bool {
  let c = levelNode(lv, l, ci, cj);
  let r = levelNode(lv, l, ci + 1, cj);
  let t = levelNode(lv, l, ci, cj + 1);
  if (c < 0 || r < 0 || t < 0) { return false; }
  let cu = activeSeg(c, 0u);
  let cv = activeSeg(c, 1u);
  let rv = activeSeg(r, 1u);
  let tu = activeSeg(t, 0u);
  if (d == 0u) { return (cu && rv) || (cv && tu); }
  return (cu && cv) || (rv && tu);
}

fn shearRest(lv: LevelInfo, l: u32, ci: i32, cj: i32) -> f32 {
  let c = levelNode(lv, l, ci, cj);
  let s0 = segAt(c, 0u).rest;
  let s1 = segAt(c, 1u).rest;
  return sqrt(s0 * s0 + s1 * s1);
}

// Fine row layout (see rowfine.wgsl): 18 vec4, block b at 2b / 2b+1, scalars in spare lanes.
const ROW_VEC4: u32 = 18u;

fn storeSym(v: u32, b: Sym, spare: vec2<f32>) {
  op[v] = vec4<f32>(b.d.x, b.o.x, b.o.y, b.d.y);
  op[v + 1u] = vec4<f32>(b.o.z, b.d.z, spare.x, spare.y);
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let st = nodes[n];
  let lv = fineLevel();
  let i = i32(st.ijl & 0xFFFu);
  let j = i32((st.ijl >> 12u) & 0xFFFu);
  let l = st.ijl >> 24u;
  let mat = P.layers[l];
  let row = n * ROW_VEC4;

  if (isPinned(i32(n))) {
    storeSym(row, symId(1.0), vec2<f32>(0.0));
    for (var k = 1u; k < 9u; k++) { storeSym(row + 2u * k, symZero(), vec2<f32>(0.0)); }
    rhs[n] = vec4<f32>(0.0);
    return;
  }

  let h = P.hg.x;
  let w = st.mass / (h * h);
  var D = symId(w);
  var g = w * (dx[n].xyz - pred[n].xyz);
  // implicit air drag (drag.wgsl): 1/2 dx^T Dd dx from the accepted state
  let Dd = airDragBlock(n);
  g += symMul(Dd, dx[n].xyz);
  D = symAdd(D, Dd);
  // External node terms (contact, friction, grip): gradient and symmetric Hessian
  let e0 = ext[3u * n];
  let e1 = ext[3u * n + 1u];
  let e2 = ext[3u * n + 2u];
  g += e0.xyz;
  D = symAdd(D, Sym(vec3<f32>(e1.x, e1.w, e2.y), vec3<f32>(e1.y, e1.z, e2.x)));

  // couplings: [+u, -u, +v, -v] axial/bending, scalars [+2u, -2u, +2v, -2v]
  var C1: array<Sym, 4>;
  var C2: array<f32, 4>;
  for (var k = 0u; k < 4u; k++) {
    C1[k] = symZero();
    C2[k] = 0.0;
  }

  for (var f = 0u; f < 2u; f++) {
    let du = select(0, 1, f == 0u);
    let dv = select(0, 1, f == 1u);
    let prev = levelNode(lv, l, i - du, j - dv);
    let next = levelNode(lv, l, i + du, j + dv);
    let prev2 = levelNode(lv, l, i - 2 * du, j - 2 * dv);
    let next2 = levelNode(lv, l, i + 2 * du, j + 2 * dv);
    let actPrev = activeSeg(prev, f);
    let actNext = activeSeg(i32(n), f) && next >= 0;
    let freeNext = next >= 0 && !isPinned(next);
    let freePrev = prev >= 0 && !isPinned(prev);
    let kp = 2u * f;
    let km = 2u * f + 1u;
    if (actPrev) {
      let s = segForce(u32(prev), n, f, mat);
      g += s.fs * s.e;
      let H = axialBlock(s.kPar, s.kPerp, s.e);
      D = symAdd(D, H);
      if (freePrev) { C1[km] = symAdd(C1[km], symScale(H, -1.0)); }
    }
    if (actNext) {
      let s = segForce(n, u32(next), f, mat);
      g -= s.fs * s.e;
      let H = axialBlock(s.kPar, s.kPerp, s.e);
      D = symAdd(D, H);
      if (freeNext) { C1[kp] = symAdd(C1[kp], symScale(H, -1.0)); }
    }
    // triple centered at n: (prev, n, next)
    if (actPrev && actNext) {
      let t = triple(u32(prev), n, u32(next), f, mat);
      g += -2.0 * tripleGrad(t);
      let k = t.k;
      D = symAdd(D, symId(4.0 * k));
      if (freeNext) { C1[kp] = symAdd(C1[kp], symId(-2.0 * k)); }
      if (freePrev) { C1[km] = symAdd(C1[km], symId(-2.0 * k)); }
    }
    // triple centered at next: (n, next, next2)
    if (actNext && activeSeg(next, f) && next2 >= 0) {
      let t = triple(n, u32(next), u32(next2), f, mat);
      g += tripleGrad(t);
      let k = t.k;
      D = symAdd(D, symId(k));
      if (freeNext) { C1[kp] = symAdd(C1[kp], symId(-2.0 * k)); }
      if (!isPinned(next2)) { C2[kp] += k; }
    }
    // triple centered at prev: (prev2, prev, n)
    if (actPrev && activeSeg(prev2, f)) {
      let t = triple(u32(prev2), u32(prev), n, f, mat);
      g += tripleGrad(t);
      let k = t.k;
      D = symAdd(D, symId(k));
      if (freePrev) { C1[km] = symAdd(C1[km], symId(-2.0 * k)); }
      if (!isPinned(prev2)) { C2[km] += k; }
    }
  }

  // shear diagonals around n; partner order matches couplings 4..7: +u+v, -u-v, -u+v, +u-v
  let ks = mat.a.x;
  var CD: array<Sym, 4>;
  for (var c = 0; c < 4; c++) {
    CD[c] = symZero();
    var ci = i;
    var cj = j;
    var pi = i + 1;
    var pj = j + 1;
    if (c == 1) { ci = i - 1; cj = j - 1; pi = i - 1; pj = j - 1; }
    if (c == 2) { ci = i - 1; cj = j; pi = i - 1; pj = j + 1; }
    if (c == 3) { ci = i; cj = j - 1; pi = i + 1; pj = j - 1; }
    let p = levelNode(lv, l, pi, pj);
    if (p < 0 || !diagActive(lv, l, ci, cj, select(1u, 0u, c < 2))) { continue; }
    let d = (x[p].xyz - x[n].xyz) + (dx[p].xyz - dx[n].xyz);
    let len = length(d);
    let e = d / len;
    let sl = shearLaw(ks, shearRest(lv, l, ci, cj), len);
    let fs = sl.x;
    g -= fs * e;
    let H = axialBlock(sl.y, max(fs, 0.0) / len, e);
    D = symAdd(D, H);
    if (!isPinned(p)) { CD[c] = symScale(H, -1.0); }
  }

  storeSym(row, D, vec2<f32>(C2[0], C2[1]));
  storeSym(row + 2u, C1[0], vec2<f32>(C2[2], C2[3]));
  for (var k = 1u; k < 4u; k++) { storeSym(row + 2u + 2u * k, C1[k], vec2<f32>(0.0)); }
  for (var k = 0u; k < 4u; k++) { storeSym(row + 10u + 2u * k, CD[k], vec2<f32>(0.0)); }
  rhs[n] = vec4<f32>(-g, 0.0);
}
