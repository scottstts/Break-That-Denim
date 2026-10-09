// Shared definitions for the mechanics kernels. Pure functions only (no bindings).

struct Yarn {
  eaLow: f32,
  eaHigh: f32,
  crimpStrain: f32,
  eaCompress: f32,
  axialVisc: f32,
  bendRigidity: f32,
  bendVisc: f32,
  // parallel yarns per lattice span (scales the crossing wrap force of a sliding bundle)
  yarnsPerSpan: f32,
};

struct LayerMat {
  // offset, nx, ny, unused
  lattice: vec4<u32>,
  yarn: array<Yarn, 2>,
  // shearStiffness, crossingFriction, crossingWrap, pitchU
  a: vec4<f32>,
  // pitchV, normal air drag (N s / m^3), failure strands per span of family u, of family v
  b: vec4<f32>,
};

struct SimParams {
  // h, gx, gy, gz
  hg: vec4<f32>,
  // node count, layer count, history updates enabled (damage, slip; 0 while settling a drape),
  // history tolerance (f32 bits): largest last Newton update at which a node may advance history
  counts: vec4<u32>,
  layers: array<LayerMat, 3>,
};

struct NodeStatic {
  // i | j << 12 | layer << 24
  ijl: u32,
  // bit0 pinned
  flags: u32,
  mass: f32,
  // crossing normal force at zero tension (wear-reduced), N
  n0: f32,
};

struct Seg {
  rest: f32,
  // bundle peak tension (yarn k peaks at peak * multiplier k)
  peak: f32,
  soft: f32,
  // packed u8 strength multiplier per bundled yarn (0 = absent), see axial.ts unpackYarnPeak
  yarnPeaks: u32,
  maxExt: f32,
  // bits 0..2 SEG_*, bits 8..11 yarn k absent or separated
  flags: u32,
  failSite: f32,
  // packed u8 failure site per bundled yarn (fraction from the owning node, /255; render only)
  yarnSites: u32,
};

const SEG_EXISTS: u32 = 1u;
const SEG_BROKEN: u32 = 2u;
const SEG_YARN_SHIFT: u32 = 8u;
const NODE_PINNED: u32 = 1u;

struct LevelInfo {
  // node count, layer count, unused, unused
  counts: vec4<u32>,
  // (offset, nx, ny, 0) per layer
  layers: array<vec4<u32>, 3>,
};

// ---------------------------------------------------------------- axial law (mirrors axial.ts)

fn axialTension(y: Yarn, eps: f32) -> f32 {
  if (eps <= 0.0) { return y.eaCompress * eps; }
  let ec = y.crimpStrain;
  let dk = y.eaHigh - y.eaLow;
  if (eps <= ec) { return y.eaLow * eps + dk * eps * eps / (2.0 * ec); }
  return y.eaLow * ec + dk * ec * 0.5 + y.eaHigh * (eps - ec);
}

fn axialStiffness(y: Yarn, eps: f32) -> f32 {
  if (eps <= 0.0) { return y.eaCompress; }
  let ec = y.crimpStrain;
  if (eps <= ec) { return y.eaLow + (y.eaHigh - y.eaLow) * eps / ec; }
  return y.eaHigh;
}

// Strain at which the undamaged curve reaches tension t > 0 (cancellation-free quadratic root).
fn strainAtTension(y: Yarn, t: f32) -> f32 {
  let ec = y.crimpStrain;
  let dk = y.eaHigh - y.eaLow;
  let tc = y.eaLow * ec + dk * ec * 0.5;
  if (t >= tc) { return ec + (t - tc) / y.eaHigh; }
  let a = dk / (2.0 * ec);
  return 2.0 * t / (y.eaLow + sqrt(y.eaLow * y.eaLow + 4.0 * a * t));
}

fn envelopeTension(peak: f32, peakExt: f32, soft: f32, ext: f32) -> f32 {
  if (ext <= peakExt) { return peak; }
  return max(0.0, peak * (1.0 - (ext - peakExt) / soft));
}

// Damaged tension and PSD solver tangent of one yarn law (peak tension `peak`) for committed history
// maxExt. Returns (T, K).
fn damagedResponse(y: Yarn, L: f32, peak: f32, soft: f32, maxExt: f32, eps: f32) -> vec2<f32> {
  let peakExt = strainAtTension(y, peak) * L;
  let ext = eps * L;
  let tEl = axialTension(y, eps);
  let kEl = axialStiffness(y, eps);
  var f = 1.0;
  if (maxExt > peakExt) {
    let tAtMax = axialTension(y, maxExt / L);
    f = select(0.0, envelopeTension(peak, peakExt, soft, maxExt) / tAtMax, tAtMax > 0.0);
  }
  let tSec = f * tEl;
  if (ext > peakExt) {
    let tEnv = envelopeTension(peak, peakExt, soft, ext);
    if (tEnv < tSec) {
      return vec2<f32>(tEnv, select(0.0, tEnv / tEl * kEl, tEl > 0.0));
    }
  }
  return vec2<f32>(tSec, f * kEl);
}

fn yarnPeakMultiplier(packed: u32, k: u32) -> f32 {
  let u = (packed >> (8u * k)) & 0xFFu;
  return select(0.4 + 1.2 * f32(u - 1u) / 254.0, 0.0, u == 0u);
}

fn strandsOf(mat: LayerMat, f: u32) -> u32 { return u32(select(mat.b.z, mat.b.w, f == 1u)); }

// Strand-level damage in a bundled span of b strands (axial.ts bundleResponse): mean of the
// per-strand responses.
fn bundleResponse(y: Yarn, b: u32, L: f32, s: Seg, eps: f32) -> vec2<f32> {
  var r = vec2<f32>(0.0);
  for (var k = 0u; k < b; k++) {
    let m = yarnPeakMultiplier(s.yarnPeaks, k);
    if (m == 0.0) { continue; }
    r += damagedResponse(y, L, s.peak * m, s.soft, s.maxExt, eps);
  }
  return r / f32(b);
}

// Bit k: strand k of the bundle (of b strands) is absent or fully separated.
fn bundleBrokenMask(y: Yarn, b: u32, L: f32, s: Seg) -> u32 {
  var mask = 0u;
  for (var k = 0u; k < b; k++) {
    let m = yarnPeakMultiplier(s.yarnPeaks, k);
    if (m == 0.0 || s.maxExt >= strainAtTension(y, s.peak * m) * L + s.soft) { mask |= 1u << k; }
  }
  return mask;
}

// Finiteness by bit pattern (immune to fast-math assumptions about NaN).
fn finite1(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn finite3(v: vec3<f32>) -> bool { return finite1(v.x) && finite1(v.y) && finite1(v.z); }

// Inverse of an SPD pivot block, regularized relative to its scale: in f32 a Schur complement can
// lose all of its inertia digits next to stiff yarn terms (exact cancellation); the shift keeps the
// preconditioner finite and positive definite.
fn pivotInverse(m: mat3x3<f32>) -> mat3x3<f32> {
  let s = max(max(abs(m[0][0]), abs(m[1][1])), abs(m[2][2]));
  let e = 2e-6 * s + 1e-30;
  let r = m + mat3x3<f32>(vec3<f32>(e, 0.0, 0.0), vec3<f32>(0.0, e, 0.0), vec3<f32>(0.0, 0.0, e));
  let inv = inverse3(r);
  if (!finite3(inv[0]) || !finite3(inv[1]) || !finite3(inv[2])) {
    let d = 1.0 / max(s, 1e-30);
    return mat3x3<f32>(vec3<f32>(d, 0.0, 0.0), vec3<f32>(0.0, d, 0.0), vec3<f32>(0.0, 0.0, d));
  }
  return inv;
}

// ---------------------------------------------------------------- fold-stiffening bending

// Bending energy of a yarn triple: 0.5 kb Psi(q), q = |x_a - 2 x_b + x_c|^2, a = 4 Lm^2 (fold
// stiffening, see src/simulation/laws.ts, mirrored exactly here).
const BEND_S0: f32 = 0.85;

// (Psi, Psi', Psi'') at q.
fn bendPsi(q: f32, a: f32) -> vec3<f32> {
  let s = q / a;
  if (s < BEND_S0) {
    let d = 1.0 - s;
    return vec3<f32>(q / d, 1.0 / (d * d), 2.0 / (a * d * d * d));
  }
  let d0 = 1.0 - BEND_S0;
  let q0 = BEND_S0 * a;
  let p1 = 1.0 / (d0 * d0);
  let p2 = 2.0 / (a * d0 * d0 * d0);
  let t = q - q0;
  return vec3<f32>(q0 / d0 + p1 * t + 0.5 * p2 * t * t, p1 + p2 * t, p2);
}

// Psi(q1) - Psi(q0) without cancellation for small changes (dq = q1 - q0 computed stably).
fn bendPsiDelta(q0: f32, q1: f32, dq: f32, a: f32) -> f32 {
  let s0 = q0 / a;
  let s1 = q1 / a;
  if (s0 < BEND_S0 && s1 < BEND_S0) { return dq / ((1.0 - s0) * (1.0 - s1)); }
  return bendPsi(q1, a).x - bendPsi(q0, a).x;
}

// Shear-locking diagonal (see shearLaw in src/simulation/laws.ts): (force, tangent stiffness).
const SHEAR_JAM: f32 = 0.16;
const SHEAR_RAMP: f32 = 0.08;
const SHEAR_LOCK: f32 = 40.0;

fn shearLaw(ks: f32, Ld: f32, l: f32) -> vec2<f32> {
  var f = ks * (l - Ld);
  var k = ks;
  let c = (Ld - l) / Ld - SHEAR_JAM;
  if (c > 0.0) {
    let extra = ks * (SHEAR_LOCK - 1.0);
    if (c < SHEAR_RAMP) {
      f -= extra * Ld * c * c / (2.0 * SHEAR_RAMP);
      k += extra * c / SHEAR_RAMP;
    } else {
      f -= extra * Ld * (c - 0.5 * SHEAR_RAMP);
      k += extra;
    }
  }
  return vec2<f32>(f, k);
}


fn segActive(s: Seg) -> bool {
  return (s.flags & (SEG_EXISTS | SEG_BROKEN)) == SEG_EXISTS;
}

// ---------------------------------------------------------------- symmetric 3x3 blocks

// d = (xx, yy, zz), o = (xy, xz, yz)
struct Sym {
  d: vec3<f32>,
  o: vec3<f32>,
};

fn symZero() -> Sym { return Sym(vec3<f32>(0.0), vec3<f32>(0.0)); }
fn symId(s: f32) -> Sym { return Sym(vec3<f32>(s), vec3<f32>(0.0)); }
fn symAdd(a: Sym, b: Sym) -> Sym { return Sym(a.d + b.d, a.o + b.o); }
fn symScale(a: Sym, s: f32) -> Sym { return Sym(a.d * s, a.o * s); }
fn symMul(a: Sym, v: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    a.d.x * v.x + a.o.x * v.y + a.o.y * v.z,
    a.o.x * v.x + a.d.y * v.y + a.o.z * v.z,
    a.o.y * v.x + a.o.z * v.y + a.d.z * v.z);
}
fn symMat(a: Sym) -> mat3x3<f32> {
  return mat3x3<f32>(vec3<f32>(a.d.x, a.o.x, a.o.y), vec3<f32>(a.o.x, a.d.y, a.o.z), vec3<f32>(a.o.y, a.o.z, a.d.z));
}
fn symFrob(a: Sym) -> f32 { return sqrt(dot(a.d, a.d) + 2.0 * dot(a.o, a.o)); }
fn symIsZero(a: Sym) -> bool { return all(a.d == vec3<f32>(0.0)) && all(a.o == vec3<f32>(0.0)); }

// kPar e e^T + kPerp (I - e e^T)
fn axialBlock(kPar: f32, kPerp: f32, e: vec3<f32>) -> Sym {
  let dk = kPar - kPerp;
  return Sym(dk * e * e + vec3<f32>(kPerp), dk * vec3<f32>(e.x * e.y, e.x * e.z, e.y * e.z));
}

fn inverse3(m: mat3x3<f32>) -> mat3x3<f32> {
  let a = m[0];
  let b = m[1];
  let c = m[2];
  let r0 = cross(b, c);
  let r1 = cross(c, a);
  let r2 = cross(a, b);
  let invDet = 1.0 / dot(a, r0);
  return transpose(mat3x3<f32>(r0 * invDet, r1 * invDet, r2 * invDet));
}

// ---------------------------------------------------------------- lattice helpers

// Returns (i, j, layer) of node n in a level.
fn levelCoords(lv: LevelInfo, n: u32) -> vec3<u32> {
  var l = 0u;
  for (var k = 1u; k < lv.counts.y; k++) {
    if (n >= lv.layers[k].x) { l = k; }
  }
  let L = lv.layers[l];
  let local = n - L.x;
  return vec3<u32>(local % L.y, local / L.y, l);
}

// Node at (i, j) in layer l, or -1 outside the lattice.
fn levelNode(lv: LevelInfo, l: u32, i: i32, j: i32) -> i32 {
  let L = lv.layers[l];
  if (i < 0 || j < 0 || i >= i32(L.y) || j >= i32(L.z)) { return -1; }
  return i32(L.x) + j * i32(L.y) + i;
}

// Coarse index set for one coarsening of extent n: 2I, except the last which is n - 1.
fn coarseCount(n: u32) -> u32 { return n / 2u + 1u; }
fn coarseToFine(I: u32, n: u32) -> u32 { return min(2u * I, n - 1u); }

// Bracketing coarse indices of fine index i: (lo, hi) and weights (wlo, whi). whi = 0 when exact.
struct Bracket {
  lo: u32,
  hi: u32,
  wlo: f32,
  whi: f32,
};
fn bracketOf(i: u32, n: u32) -> Bracket {
  let nc = coarseCount(n);
  var lo = min(nc - 1u, i >> 1u);
  if (coarseToFine(lo, n) > i) { lo = lo - 1u; }
  let a = coarseToFine(lo, n);
  if (a == i) { return Bracket(lo, lo, 1.0, 0.0); }
  let b = coarseToFine(lo + 1u, n);
  let inv = 1.0 / f32(b - a);
  return Bracket(lo, lo + 1u, f32(b - i) * inv, f32(i - a) * inv);
}
