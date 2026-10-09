// Yarn-line smoother. One 32-thread workgroup solves one line exactly as a block-tridiagonal system
// with the partition method: each thread eliminates a contiguous chunk in thread-private scratch,
// expressing interior unknowns as x_i = v_i + V_i x_s + W_i x_e in terms of its chunk ends; the
// 2P chunk-end equations form a small block-tridiagonal system solved in workgroup memory; interior
// unknowns are then recovered in parallel. Compiled with rowfine.wgsl or rowcoarse.wgsl.
// Couplings to other lines (other colors) are lagged; the +-2 coupling along the line is majorized
// onto the diagonal. Mirrors smoothLines (CPU).

const WG: u32 = 32u;
// Per-node scratch (vec4): G 0..2, h/v 3, Hs/V 4..6, He/W 7..9.
const SCR: u32 = 10u;

@group(0) @binding(0) var<uniform> L: LevelInfo;
@group(0) @binding(1) var<uniform> DP: vec4<u32>; // x: first line index of this color
@group(0) @binding(2) var<storage, read> lines: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> op: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> fixedN: array<u32>;
@group(0) @binding(5) var<storage, read_write> xv: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> bv: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> scr: array<vec4<f32>>;

var<workgroup> qL: array<mat3x3<f32>, 64>;
var<workgroup> qD: array<mat3x3<f32>, 64>;
var<workgroup> qU: array<mat3x3<f32>, 64>;
var<workgroup> qr: array<vec3<f32>, 64>;
var<workgroup> sLen: u32;

fn ldM(b: u32) -> mat3x3<f32> { return mat3x3<f32>(scr[b].xyz, scr[b + 1u].xyz, scr[b + 2u].xyz); }
fn stM(b: u32, m: mat3x3<f32>) {
  scr[b] = vec4<f32>(m[0], 0.0);
  scr[b + 1u] = vec4<f32>(m[1], 0.0);
  scr[b + 2u] = vec4<f32>(m[2], 0.0);
}

struct LineRow {
  A: mat3x3<f32>, // coupling to the previous node on the line
  B: mat3x3<f32>, // diagonal incl. majorization
  C: mat3x3<f32>, // coupling to the next node on the line
  r: vec3<f32>,
};

fn lineRow(n: u32, k: u32, len: u32, fam: u32) -> LineRow {
  let zero3 = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  if (fixedN[n] != 0u) { return LineRow(zero3, symMat(symId(1.0)), zero3, vec3<f32>(0.0)); }
  let cc = levelCoords(L, n);
  let ci = i32(cc.x);
  let cj = i32(cc.y);
  var row: LineRow;
  row.A = zero3;
  row.C = zero3;
  var D = rowDiag(n);
  var r = bv[n].xyz;
  var maj = 0.0;
  // Couplings to fixed nodes are zero by construction, so no neighbor flag checks are needed.
  for (var c = 0u; c < ROW_COUPLINGS; c++) {
    let off = rowOffset(c);
    let along = select(off.y, off.x, fam == 0u);
    let across = select(off.x, off.y, fam == 0u);
    if (across == 0 && along == -1) {
      if (k > 0u) { row.A = symMat(rowCoupling(n, c)); }
      continue;
    }
    if (across == 0 && along == 1) {
      if (k + 1u < len) { row.C = symMat(rowCoupling(n, c)); }
      continue;
    }
    let m = levelNode(L, cc.z, ci + off.x, cj + off.y);
    if (m < 0) { continue; }
    let Bc = rowCoupling(n, c);
    if (across == 0) { maj += symFrob(Bc); }
    r -= symMul(Bc, xv[u32(m)].xyz);
  }
  D = symAdd(D, symId(maj));
  row.B = symMat(D);
  row.r = r + maj * xv[n].xyz;
  return row;
}

@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let line = lines[DP.x + wg.x];
  if (t == 0u) { sLen = line.z; }
  let len = workgroupUniformLoad(&sLen);
  let fam = line.w;
  let n0 = line.x;
  let st = line.y;
  let zero3 = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  let I3 = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
  let P = max(1u, min(WG, len / 2u));
  let base = len / P;
  let extra = len % P;
  let ks = t * base + min(t, extra);
  let m = base + select(0u, 1u, t < extra);
  let ke = ks + m - 1u;

  if (t < P) {
    let rs = lineRow(n0 + ks * st, ks, len, fam);
    let re = lineRow(n0 + ke * st, ke, len, fam);
    // chunk-interior elimination with right-hand sides [r | x_s | x_e]
    var Gp = zero3;
    var hp = vec3<f32>(0.0);
    var Hsp = I3;
    var Hep = zero3;
    for (var k = ks + 1u; k < ke; k++) {
      let n = n0 + k * st;
      let rk = lineRow(n, k, len, fam);
      let Minv = inverse3(rk.B - rk.A * Gp);
      var G = zero3;
      var Cext = zero3;
      if (k + 1u < ke) { G = Minv * rk.C; } else { Cext = rk.C; }
      let h = Minv * (rk.r - rk.A * hp);
      let Hs = Minv * (zero3 - rk.A * Hsp);
      let He = Minv * (zero3 - rk.A * Hep - Cext);
      let b = SCR * n;
      stM(b, G);
      scr[b + 3u] = vec4<f32>(h, 0.0);
      stM(b + 4u, Hs);
      stM(b + 7u, He);
      Gp = G;
      hp = h;
      Hsp = Hs;
      Hep = He;
    }
    // back substitution inside the chunk: (v, V, W) of each interior node
    var vS = vec3<f32>(0.0); // at ks + 1 (defaults for m == 2: x_{s+1} = x_e)
    var VS = zero3;
    var WS = I3;
    var vE = vec3<f32>(0.0); // at ke - 1 (defaults for m == 2: x_{e-1} = x_s)
    var VE = I3;
    var WE = zero3;
    if (m >= 3u) {
      var vn = vec3<f32>(0.0);
      var Vn = zero3;
      var Wn = zero3;
      for (var k = ke - 1u; k > ks; k--) {
        let b = SCR * (n0 + k * st);
        let G = ldM(b);
        let v = scr[b + 3u].xyz - G * vn;
        let V = ldM(b + 4u) - G * Vn;
        let W = ldM(b + 7u) - G * Wn;
        scr[b + 3u] = vec4<f32>(v, 0.0);
        stM(b + 4u, V);
        stM(b + 7u, W);
        if (k == ke - 1u) {
          vE = v;
          VE = V;
          WE = W;
        }
        vn = v;
        Vn = V;
        Wn = W;
      }
      vS = vn;
      VS = Vn;
      WS = Wn;
    }
    // reduced equations for x_s (index 2t) and x_e (index 2t + 1)
    qL[2u * t] = rs.A;
    qD[2u * t] = rs.B + rs.C * VS;
    qU[2u * t] = rs.C * WS;
    qr[2u * t] = rs.r - rs.C * vS;
    qL[2u * t + 1u] = re.A * VE;
    qD[2u * t + 1u] = re.B + re.A * WE;
    qU[2u * t + 1u] = re.C;
    qr[2u * t + 1u] = re.r - re.A * vE;
  }
  workgroupBarrier();

  if (t == 0u) {
    let Q = 2u * P;
    var Gp = zero3;
    var yp = vec3<f32>(0.0);
    for (var q = 0u; q < Q; q++) {
      let Minv = inverse3(qD[q] - qL[q] * Gp);
      let G = Minv * qU[q];
      let y = Minv * (qr[q] - qL[q] * yp);
      qU[q] = G;
      qr[q] = y;
      Gp = G;
      yp = y;
    }
    var zn = vec3<f32>(0.0);
    for (var q = Q; q > 0u; q--) {
      let z = qr[q - 1u] - qU[q - 1u] * zn;
      qr[q - 1u] = z;
      zn = z;
    }
  }
  workgroupBarrier();

  if (t < P) {
    let xs = qr[2u * t];
    let xe = qr[2u * t + 1u];
    let ns = n0 + ks * st;
    let ne = n0 + ke * st;
    if (fixedN[ns] == 0u) { xv[ns] = vec4<f32>(xs, 0.0); }
    if (fixedN[ne] == 0u) { xv[ne] = vec4<f32>(xe, 0.0); }
    for (var k = ks + 1u; k < ke; k++) {
      let n = n0 + k * st;
      if (fixedN[n] != 0u) { continue; }
      let b = SCR * n;
      xv[n] = vec4<f32>(scr[b + 3u].xyz + ldM(b + 4u) * xs + ldM(b + 7u) * xe, 0.0);
    }
  }
}
