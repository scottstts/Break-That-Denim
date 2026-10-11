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
// contact bounds per node (selfcontact.wgsl / contact.wgsl; step.wgsl feasible): own and reciprocal
@group(0) @binding(9) var<storage, read> cmeta: array<vec4<f32>>;

fn fineLevel() -> LevelInfo {
  var lv: LevelInfo;
  lv.counts = P.counts;
  for (var l = 0u; l < 3u; l++) { lv.layers[l] = P.layers[l].lattice; }
  return lv;
}

// mounted or retired (detached pieces stay in the solve, state.wgsl)
fn isPinned(m: i32) -> bool { return (nodes[u32(m)].flags & NODE_PINNED) != 0u || x[u32(m)].w > 1.5; }

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

// Bending of the triple (a, b, c) centered at b (fold-stiffening energy, see bendPsi, with the
// rest-length weights r = wa x_a - 2 x_b + wc x_c of bendGeometry): the gradient coefficient ke Psi'
// and a scalar stiffness k = ke (Psi' + 2 Psi'' q) + kd that majorizes the exact Hessian
// ke (Psi' I + 2 Psi'' r r^T) + kd H_v (the operator stores this stencil as scalars, times w w^T).
// Viscosity acts on the rate of the bending measure |r| (viscous potential 1/2 kd (|r| - |rn|)^2 over
// the substep): a rigid rotation of the triple changes r but not |r|, and is not resisted. Its Hessian
// H_v = r^ r^T + (1 - |rn| / |r|) (I - r^ r^T) is at most I.
struct Triple {
  ke: f32,
  kd: f32,
  k: f32,
  wa: f32,
  wc: f32,
  r: vec3<f32>,
  rn: vec3<f32>,
};

fn triple(a: u32, b: u32, c: u32, f: u32, mat: LayerMat) -> Triple {
  let sa = segs[2u * a + f];
  let sb = segs[2u * b + f];
  let geo = bendGeometry(sa.rest, sb.rest);
  let y = mat.yarn[f];
  // only yarns continuous through b bend (laws.ts bendScale)
  let kb = y.bendRigidity * bendScale(sa.intact, sb.intact, sa.cutMask, sb.cutMask, u32(y.yarnsPerSpan)) * geo.z;
  let kd = kb * y.bendVisc / P.hg.x;
  let rn = geo.x * x[a].xyz - 2.0 * x[b].xyz + geo.y * x[c].xyz;
  let r = rn + geo.x * dx[a].xyz - 2.0 * dx[b].xyz + geo.y * dx[c].xyz;
  let q = dot(r, r);
  let ps = bendPsi(q, geo.w);
  return Triple(kb * ps.y, kd, kb * (ps.y + 2.0 * ps.z * q) + kd, geo.x, geo.y, r, rn);
}

fn tripleGrad(t: Triple) -> vec3<f32> {
  let lr = length(t.r);
  return (t.ke + t.kd * (lr - length(t.rn)) / max(lr, 1e-12)) * t.r;
}

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

fn intactOf(owner: i32, f: u32) -> f32 { return select(0.0, segAt(owner, f).intact, activeSeg(owner, f)); }

// Shear stiffness and jam strain of the cell with lower-left corner (ci, cj): a worn, thinned weave
// turns further before its yarns jam and resists turning less (laws.ts cellShear).
fn cellShear(lv: LevelInfo, l: u32, ci: i32, cj: i32, mat: LayerMat) -> vec2<f32> {
  let c = levelNode(lv, l, ci, cj);
  let r = levelNode(lv, l, ci + 1, cj);
  let t = levelNode(lv, l, ci, cj + 1);
  let a = cellArea(intactOf(c, 0u), intactOf(t, 0u), intactOf(c, 1u), intactOf(r, 1u));
  return vec2<f32>(mat.a.x * nodes[u32(c)].n0 / mat.b.z, shearJam(a));
}

fn shearRest(lv: LevelInfo, l: u32, ci: i32, cj: i32) -> f32 {
  let c = levelNode(lv, l, ci, cj);
  let s0 = segAt(c, 0u).rest;
  let s1 = segAt(c, 1u).rest;
  return sqrt(s0 * s0 + s1 * s1);
}

// Normal stiffness of an active contact bound in the Newton operator, relative to the node's own
// diagonal: the held node keeps ~1 % of the normal correction it would otherwise take.
const ACTIVE_BOUND: f32 = 100.0;

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
    // triple centered at n: (prev, n, next), weight of n -2
    if (actPrev && actNext) {
      let t = triple(u32(prev), n, u32(next), f, mat);
      g += -2.0 * tripleGrad(t);
      let k = t.k;
      D = symAdd(D, symId(4.0 * k));
      if (freeNext) { C1[kp] = symAdd(C1[kp], symId(-2.0 * t.wc * k)); }
      if (freePrev) { C1[km] = symAdd(C1[km], symId(-2.0 * t.wa * k)); }
    }
    // triple centered at next: (n, next, next2), weight of n wa
    if (actNext && activeSeg(next, f) && next2 >= 0) {
      let t = triple(n, u32(next), u32(next2), f, mat);
      g += t.wa * tripleGrad(t);
      let k = t.k;
      D = symAdd(D, symId(t.wa * t.wa * k));
      if (freeNext) { C1[kp] = symAdd(C1[kp], symId(-2.0 * t.wa * k)); }
      if (!isPinned(next2)) { C2[kp] += t.wa * t.wc * k; }
    }
    // triple centered at prev: (prev2, prev, n), weight of n wc
    if (actPrev && activeSeg(prev2, f)) {
      let t = triple(u32(prev2), u32(prev), n, f, mat);
      g += t.wc * tripleGrad(t);
      let k = t.k;
      D = symAdd(D, symId(t.wc * t.wc * k));
      if (freePrev) { C1[km] = symAdd(C1[km], symId(-2.0 * t.wc * k)); }
      if (!isPinned(prev2)) { C2[km] += t.wa * t.wc * k; }
    }
  }

  // shear diagonals around n; partner order matches couplings 4..7: +u+v, -u-v, -u+v, +u-v
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
    let cs = cellShear(lv, l, ci, cj, mat);
    let sl = shearLaw(cs.x, cs.y, shearRest(lv, l, ci, cj), len);
    let fs = sl.x;
    g -= fs * e;
    let H = axialBlock(sl.y, max(fs, 0.0) / len, e);
    D = symAdd(D, H);
    if (!isPinned(p)) { CD[c] = symScale(H, -1.0); }
  }

  // Active contact bounds (step.wgsl feasible): a node whose iterate already approaches a face by all
  // its bound allows, and whose net force still drives it on, is held along the bound's normal in the
  // Newton direction (stiffness ACTIVE_BOUND times its own), so the rest of the system takes up what the
  // clip would otherwise leave as a stretched span. Only the direction changes (the line search uses
  // the true potential); a node the forces pull away from the face is free.
  let kAct = ACTIVE_BOUND * (D.d.x + D.d.y + D.d.z);
  for (var b = 0u; b < 2u; b++) {
    let m = cmeta[2u * n + b];
    if (m.w >= 1e8 || all(m.xyz == vec3<f32>(0.0))) { continue; }
    if (-dot(dx[n].xyz, m.xyz) >= m.w - 1e-3 * abs(m.w) && dot(g, m.xyz) > 0.0) { D = symAdd(D, axialBlock(kAct, 0.0, m.xyz)); }
  }

  storeSym(row, D, vec2<f32>(C2[0], C2[1]));
  storeSym(row + 2u, C1[0], vec2<f32>(C2[2], C2[3]));
  for (var k = 1u; k < 4u; k++) { storeSym(row + 2u + 2u * k, C1[k], vec2<f32>(0.0)); }
  for (var k = 0u; k < 4u; k++) { storeSym(row + 10u + 2u * k, CD[k], vec2<f32>(0.0)); }
  rhs[n] = vec4<f32>(-g, 0.0);
}
