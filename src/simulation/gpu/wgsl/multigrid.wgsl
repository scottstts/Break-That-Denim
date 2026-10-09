// Galerkin multigrid kernels on lattice stencils, compiled with the row reader of the finer level
// (rowfine.wgsl or rowcoarse.wgsl). Coarse rows are always written in the coarse layout.
// Mirrors galerkin / vcycle in reference/newtonMultigrid.ts.

@group(0) @binding(0) var<uniform> F: LevelInfo;
@group(0) @binding(1) var<uniform> C: LevelInfo;
@group(0) @binding(2) var<storage, read> op: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> fixedF: array<u32>;
@group(0) @binding(4) var<storage, read_write> opC: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> fixedC: array<u32>;
@group(0) @binding(6) var<storage, read> vecIn: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> vecIn2: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> vecOut: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> vecOut2: array<vec4<f32>>;
// Prolongation weights of each level-F node to its bracketing coarse nodes, slot ki + 2 kj (lo = 0).
@group(0) @binding(10) var<storage, read_write> pw: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> segsF: array<Seg>;

// Coarse row writer (layout of rowcoarse.wgsl): offset index o = (dj + 2) * 5 + (di + 2).
fn storeC(n: u32, o: u32, b: Sym) {
  var blk = 0u;
  if (o != 12u) { blk = 1u + select(o, o - 1u, o > 12u); }
  let v = n * 50u + 2u * blk;
  opC[v] = vec4<f32>(b.d.x, b.o.x, b.o.y, b.d.y);
  opC[v + 1u] = vec4<f32>(b.o.z, b.d.z, 0.0, 0.0);
}

// Weight of fine index i in coarse index I (extent n), 0 if outside its bracket.
fn weightOf(i: u32, I: u32, n: u32) -> f32 {
  let b = bracketOf(i, n);
  if (b.lo == I) { return b.wlo; }
  if (b.hi == I && b.whi > 0.0) { return b.whi; }
  return 0.0;
}

// ---------------------------------------------------------------- connectivity-aware prolongation

// Whether the level-F node at (i, j) and its unit-offset neighbour are connected: an intact span on
// the fine lattice; on coarse levels a nonzero Galerkin coupling (pieces separated by a tear have
// none) or a fixed neighbour (mount boundaries keep their bilinear weights).
fn linked(l: u32, i: i32, j: i32, di: i32, dj: i32) -> bool {
  let a = levelNode(F, l, i, j);
  let b = levelNode(F, l, i + di, j + dj);
  if (a < 0 || b < 0) { return false; }
  if (FINE_ROWS) {
    let f = select(1u, 0u, di != 0);
    let owner = select(b, a, di + dj > 0);
    return segActive(segsF[2u * u32(owner) + f]);
  }
  if (fixedF[u32(b)] != 0u) { return true; }
  return !symIsZero(rowCoupling(u32(a), couplingIndex(vec2<i32>(di, dj))));
}

// Bilinear weights masked to the coarse nodes a node reaches through connected edges inside their
// shared cell, renormalized. Across a tear, plain bilinear interpolation ties both sides to the same
// coarse unknowns, so a torn flap's own rigid motion has no coarse representation and its swing
// converges poorly. Mirrors buildProlongation (reference/newtonMultigrid.ts).
@compute @workgroup_size(128)
fn prolongWeights(@builtin(global_invocation_id) gid: vec3<u32>) {
  let a = gid.x;
  if (a >= F.counts.x) { return; }
  if (fixedF[a] != 0u) {
    pw[a] = vec4<f32>(0.0);
    return;
  }
  let fc = levelCoords(F, a);
  let l = fc.z;
  let Lf = F.layers[l];
  let i = i32(fc.x);
  let j = i32(fc.y);
  let bi = bracketOf(fc.x, Lf.y);
  let bj = bracketOf(fc.y, Lf.z);
  var w = vec4<f32>(0.0);
  var sum = 0.0;
  for (var kj = 0u; kj < 2u; kj++) {
    let J = select(bj.lo, bj.hi, kj == 1u);
    let wj = select(bj.wlo, bj.whi, kj == 1u);
    if (wj == 0.0) { continue; }
    let dj = i32(coarseToFine(J, Lf.z)) - j;
    for (var ki = 0u; ki < 2u; ki++) {
      let I = select(bi.lo, bi.hi, ki == 1u);
      let wi = select(bi.wlo, bi.whi, ki == 1u);
      if (wi == 0.0) { continue; }
      let di = i32(coarseToFine(I, Lf.y)) - i;
      var ok = true;
      if (di != 0 && dj != 0) {
        ok = (linked(l, i, j, di, 0) && linked(l, i + di, j, 0, dj)) || (linked(l, i, j, 0, dj) && linked(l, i, j + dj, di, 0));
      } else if (di != 0 || dj != 0) {
        ok = linked(l, i, j, di, dj);
      }
      if (ok) {
        w[ki + 2u * kj] = wi * wj;
        sum += wi * wj;
      }
    }
  }
  pw[a] = select(vec4<f32>(0.0), w / sum, sum > 0.0);
}

// Prolongation weight of level-F node a (at fine coordinates fi, fj) for coarse node (I, J).
fn weightAt(a: u32, fi: u32, fj: u32, I: u32, J: u32, nxf: u32, nyf: u32) -> f32 {
  let bi = bracketOf(fi, nxf);
  let bj = bracketOf(fj, nyf);
  let ki = select(1u, 0u, bi.lo == I);
  let kj = select(1u, 0u, bj.lo == J);
  return pw[a][ki + 2u * kj];
}

// ---------------------------------------------------------------- Galerkin coarse operator P^T A P

@compute @workgroup_size(64)
fn rap(@builtin(global_invocation_id) gid: vec3<u32>) {
  let I = gid.x;
  if (I >= C.counts.x) { return; }
  let cc = levelCoords(C, I);
  let l = cc.z;
  let Lf = F.layers[l];
  let nxf = Lf.y;
  let nyf = Lf.z;
  var acc: array<Sym, 25>;
  for (var o = 0u; o < 25u; o++) { acc[o] = symZero(); }
  let fi0 = i32(coarseToFine(cc.x, nxf));
  let fj0 = i32(coarseToFine(cc.y, nyf));
  for (var aj = fj0 - 1; aj <= fj0 + 1; aj++) {
    if (aj < 0 || aj >= i32(nyf)) { continue; }
    let wj = weightOf(u32(aj), cc.y, nyf);
    if (wj == 0.0) { continue; }
    for (var ai = fi0 - 1; ai <= fi0 + 1; ai++) {
      if (ai < 0 || ai >= i32(nxf)) { continue; }
      let wi = weightOf(u32(ai), cc.x, nxf);
      if (wi == 0.0) { continue; }
      let a = u32(levelNode(F, l, ai, aj));
      if (fixedF[a] != 0u) { continue; }
      let wa = weightAt(a, u32(ai), u32(aj), cc.x, cc.y, nxf, nyf);
      if (wa == 0.0) { continue; }
      for (var k = 0u; k <= ROW_COUPLINGS; k++) {
        var off = vec2<i32>(0, 0);
        var A: Sym;
        if (k == ROW_COUPLINGS) {
          A = rowDiag(a);
        } else {
          off = rowOffset(k);
          A = rowCoupling(a, k);
        }
        if (symIsZero(A)) { continue; }
        let bi = ai + off.x;
        let bj = aj + off.y;
        if (bi < 0 || bj < 0 || bi >= i32(nxf) || bj >= i32(nyf)) { continue; }
        let bri = bracketOf(u32(bi), nxf);
        let brj = bracketOf(u32(bj), nyf);
        let wb = pw[u32(levelNode(F, l, bi, bj))];
        for (var kj = 0u; kj < 2u; kj++) {
          let J = select(brj.lo, brj.hi, kj == 1u);
          for (var ki = 0u; ki < 2u; ki++) {
            let Ii = select(bri.lo, bri.hi, ki == 1u);
            let wJ = wb[ki + 2u * kj];
            if (wJ == 0.0) { continue; }
            let o = u32((i32(J) - i32(cc.y) + 2) * 5 + (i32(Ii) - i32(cc.x) + 2));
            acc[o] = symAdd(acc[o], symScale(A, wa * wJ));
          }
        }
      }
    }
  }
  let center = acc[12];
  if (center.d.x + center.d.y + center.d.z <= 0.0) {
    for (var o = 0u; o < 25u; o++) { storeC(I, o, symZero()); }
    storeC(I, 12u, symId(1.0));
    fixedC[I] = 1u;
    return;
  }
  for (var o = 0u; o < 25u; o++) { storeC(I, o, acc[o]); }
  fixedC[I] = 0u;
}

// ---------------------------------------------------------------- residual r = b - A x  (level F)

@compute @workgroup_size(128)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= F.counts.x) { return; }
  if (fixedF[n] != 0u) {
    vecOut[n] = vec4<f32>(0.0);
    return;
  }
  let cc = levelCoords(F, n);
  var r = vecIn2[n].xyz - symMul(rowDiag(n), vecIn[n].xyz);
  for (var k = 0u; k < ROW_COUPLINGS; k++) {
    let off = rowOffset(k);
    let m = levelNode(F, cc.z, i32(cc.x) + off.x, i32(cc.y) + off.y);
    if (m < 0) { continue; }
    r -= symMul(rowCoupling(n, k), vecIn[u32(m)].xyz);
  }
  vecOut[n] = vec4<f32>(r, 0.0);
}

// ---------------------------------------------------------------- restriction b_c = P^T r_f, x_c = 0

@compute @workgroup_size(128)
fn restrictResidual(@builtin(global_invocation_id) gid: vec3<u32>) {
  let I = gid.x;
  if (I >= C.counts.x) { return; }
  vecOut2[I] = vec4<f32>(0.0);
  if (fixedC[I] != 0u) {
    vecOut[I] = vec4<f32>(0.0);
    return;
  }
  let cc = levelCoords(C, I);
  let l = cc.z;
  let Lf = F.layers[l];
  let fi0 = i32(coarseToFine(cc.x, Lf.y));
  let fj0 = i32(coarseToFine(cc.y, Lf.z));
  var s = vec3<f32>(0.0);
  for (var aj = fj0 - 1; aj <= fj0 + 1; aj++) {
    if (aj < 0 || aj >= i32(Lf.z)) { continue; }
    let wj = weightOf(u32(aj), cc.y, Lf.z);
    if (wj == 0.0) { continue; }
    for (var ai = fi0 - 1; ai <= fi0 + 1; ai++) {
      if (ai < 0 || ai >= i32(Lf.y)) { continue; }
      let wi = weightOf(u32(ai), cc.x, Lf.y);
      if (wi == 0.0) { continue; }
      let a = u32(levelNode(F, l, ai, aj));
      if (fixedF[a] != 0u) { continue; }
      s += weightAt(a, u32(ai), u32(aj), cc.x, cc.y, Lf.y, Lf.z) * vecIn[a].xyz;
    }
  }
  vecOut[I] = vec4<f32>(s, 0.0);
}

// ---------------------------------------------------------------- prolongation x_f += P x_c

@compute @workgroup_size(128)
fn prolongAdd(@builtin(global_invocation_id) gid: vec3<u32>) {
  let a = gid.x;
  if (a >= F.counts.x) { return; }
  if (fixedF[a] != 0u) { return; }
  let fc = levelCoords(F, a);
  let Lf = F.layers[fc.z];
  let bi = bracketOf(fc.x, Lf.y);
  let bj = bracketOf(fc.y, Lf.z);
  let w = pw[a];
  var s = vec3<f32>(0.0);
  for (var kj = 0u; kj < 2u; kj++) {
    let J = select(bj.lo, bj.hi, kj == 1u);
    for (var ki = 0u; ki < 2u; ki++) {
      let Ii = select(bi.lo, bi.hi, ki == 1u);
      let wk = w[ki + 2u * kj];
      if (wk == 0.0) { continue; }
      let c = u32(levelNode(C, fc.z, i32(Ii), i32(J)));
      s += wk * vecIn[c].xyz;
    }
  }
  vecOut[a] = vec4<f32>(vecOut[a].xyz + s, 0.0);
}
