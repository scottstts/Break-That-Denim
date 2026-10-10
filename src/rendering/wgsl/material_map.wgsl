// Where a material point of the cloth is (pieces.wgsl): the refined lattice surface of its cell,
// corrected toward the nodes it is still connected to. Requires module-scope R (RenderU), rsurf
// (refine.wgsl samples), x (positions), tframe (node tangent frames) and kappaM (pieces.wgsl).
//
// Each corner c of the cell enters with the connection kap.c of the point to it. Where that is 1 the
// corner is used as it is; where it is 0 the corner is replaced by a ghost: where the point's own
// material would put it, from the corners it is connected to, each carrying its rest offset to c in its
// own tangent frame. The correction blends with the bilinear weight of the corner, so it vanishes on
// a cell edge for the corners off that edge: a point on an edge is placed the same way from either
// cell. A piece cut off from some corners therefore keeps its rest shape there, and the gap a crack
// opens is what the lattice has stretched across it.

// Connection (0..1) of crossing (k, m) of layer l to node n (pieces.wgsl).
fn kappaAt(l: u32, n: u32, k: i32, m: i32) -> f32 {
  let s = kappaM[KAPPA_TABLE + n];
  if (s == KAPPA_NONE) { return 1.0; }
  if (s == KAPPA_ZERO) { return 0.0; }
  let LR = R.layers[l];
  let L = LR.lattice;
  let local = n - L.x;
  let bu = f32(bundleOf(LR, 1u));
  let bv = f32(bundleOf(LR, 0u));
  let Un = f32(local % L.y) * bu + 0.5 * (bu - 1.0);
  let Vn = f32(local / L.y) * bv + 0.5 * (bv - 1.0);
  let dk = k - i32(ceil(Un - bu - 1.0));
  let dm = m - i32(ceil(Vn - bv - 1.0));
  if (dk < 0 || dm < 0 || dk >= i32(KAPPA_RW) || dm >= i32(KAPPA_RH)) { return 1.0; }
  let e = u32(dm) * KAPPA_RW + u32(dk);
  let w = kappaM[KAPPA_TABLE + R.counts.x + KAPPA_SLOTS + s * KAPPA_WORDS + (e >> 2u)];
  return f32((w >> (8u * (e & 3u))) & 0xFFu) / 255.0;
}

struct MatPoint {
  // false: the material is connected to none of the cell's corners (it is not held by this cell)
  valid: bool,
  p: vec3<f32>,
  // derivatives over the cell (s along u, t along v)
  ds: vec3<f32>,
  dt: vec3<f32>,
  wear: f32,
};

// Point (s, t) of cell (ci, cj) of layer l whose material has connections kap to the cell's corners
// (corner index: bit 0 = +i, bit 1 = +j).
fn matPoint(l: u32, ci: u32, cj: u32, s: f32, t: f32, kap: vec4<f32>) -> MatPoint {
  let LR = R.layers[l];
  let L = LR.lattice;
  let S = LR.twill.z;
  // refined surface: bilinear within the refined sub-quad, as the intact cells' triangles draw it
  let sI = clamp(s, 0.0, 1.0) * f32(S);
  let tI = clamp(t, 0.0, 1.0) * f32(S);
  let I0 = min(u32(floor(sI)), S - 1u);
  let J0 = min(u32(floor(tI)), S - 1u);
  let fs = sI - f32(I0);
  let ft = tI - f32(J0);
  let r00 = LR.refine.x + (cj * S + J0) * LR.refine.y + ci * S + I0;
  let r10 = r00 + 1u;
  let r01 = r00 + LR.refine.y;
  let r11 = r01 + 1u;
  let w00 = (1.0 - fs) * (1.0 - ft);
  let w10 = fs * (1.0 - ft);
  let w01 = (1.0 - fs) * ft;
  let w11 = fs * ft;
  var o: MatPoint;
  o.valid = any(kap > vec4<f32>(0.02));
  let a = rsurf[3u * r00] * w00 + rsurf[3u * r10] * w10 + rsurf[3u * r01] * w01 + rsurf[3u * r11] * w11;
  o.p = a.xyz;
  o.wear = a.w;
  o.ds = (rsurf[3u * r00 + 1u] * w00 + rsurf[3u * r10 + 1u] * w10 + rsurf[3u * r01 + 1u] * w01 + rsurf[3u * r11 + 1u] * w11).xyz;
  o.dt = (rsurf[3u * r00 + 2u] * w00 + rsurf[3u * r10 + 2u] * w10 + rsurf[3u * r01 + 2u] * w01 + rsurf[3u * r11 + 2u] * w11).xyz;
  if (all(kap >= vec4<f32>(0.999))) { return o; }
  // Partly connected: the corners are corrected (a corner the point is cut off from is replaced by
  // where the point's own material would put it), and the point placed bilinearly between the
  // corrected corners; the bicubic surface's curvature detail over that is kept only as far as the
  // point is connected to all four (a far corner, e.g. of a piece that fell away, never bends it).
  let n00 = L.x + cj * L.y + ci;
  var nc = array<u32, 4>(n00, n00 + 1u, n00 + L.y, n00 + L.y + 1u);
  var h = array<f32, 4>((1.0 - s) * (1.0 - t), s * (1.0 - t), (1.0 - s) * t, s * t);
  var hs = array<f32, 4>(-(1.0 - t), 1.0 - t, -t, t);
  var ht = array<f32, 4>(-(1.0 - s), -s, 1.0 - s, s);
  let restU = f32(bundleOf(LR, 1u)) * LR.pitch.x;
  let restV = f32(bundleOf(LR, 0u)) * LR.pitch.y;
  var Pb = vec3<f32>(0.0);
  var Pbs = vec3<f32>(0.0);
  var Pbt = vec3<f32>(0.0);
  var Qb = vec3<f32>(0.0);
  var Qbs = vec3<f32>(0.0);
  var Qbt = vec3<f32>(0.0);
  for (var c = 0u; c < 4u; c++) {
    let Pc = x[nc[c]].xyz;
    var Qc = Pc;
    if (kap[c] < 0.999) {
      var G = vec3<f32>(0.0);
      var W = 0.0;
      for (var q = 0u; q < 4u; q++) {
        if (q == c) { continue; }
        let wq = (h[q] + 1e-3) * kap[q];
        if (wq <= 0.0) { continue; }
        let di = f32(i32(c & 1u) - i32(q & 1u));
        let dj = f32(i32(c >> 1u) - i32(q >> 1u));
        G += wq * (x[nc[q]].xyz + di * restU * tframe[2u * nc[q]].xyz + dj * restV * tframe[2u * nc[q] + 1u].xyz);
        W += wq;
      }
      if (W > 1e-9) { Qc = mix(G / W, Pc, kap[c]); }
    }
    Pb += h[c] * Pc;
    Pbs += hs[c] * Pc;
    Pbt += ht[c] * Pc;
    Qb += h[c] * Qc;
    Qbs += hs[c] * Qc;
    Qbt += ht[c] * Qc;
  }
  let detail = min(min(kap.x, kap.y), min(kap.z, kap.w));
  o.p = Qb + detail * (o.p - Pb);
  o.ds = Qbs + detail * (o.ds - Pbs);
  o.dt = Qbt + detail * (o.dt - Pbt);
  return o;
}

// Cell (ci, cj) of the fine point (U, V) of layer l and its coordinates (s, t) in it.
fn cellOfUV(l: u32, U: f32, V: f32) -> vec4<f32> {
  let LR = R.layers[l];
  let L = LR.lattice;
  let bu = f32(bundleOf(LR, 1u));
  let bv = f32(bundleOf(LR, 0u));
  let su = (U - 0.5 * (bu - 1.0)) / bu;
  let sv = (V - 0.5 * (bv - 1.0)) / bv;
  let ci = clamp(floor(su), 0.0, f32(L.y) - 2.0);
  let cj = clamp(floor(sv), 0.0, f32(L.z) - 2.0);
  return vec4<f32>(ci, cj, su - ci, sv - cj);
}

// Material point of crossing (k, m) of layer l, from its own connections to its cell's corners.
fn crossingPoint(l: u32, k: i32, m: i32) -> MatPoint {
  let cc = cellOfUV(l, f32(k), f32(m));
  let ci = u32(cc.x);
  let cj = u32(cc.y);
  let L = R.layers[l].lattice;
  let n00 = L.x + cj * L.y + ci;
  let kap = vec4<f32>(kappaAt(l, n00, k, m), kappaAt(l, n00 + 1u, k, m), kappaAt(l, n00 + L.y, k, m), kappaAt(l, n00 + L.y + 1u, k, m));
  return matPoint(l, ci, cj, cc.z, cc.w, kap);
}
