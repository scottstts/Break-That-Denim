// Free tails of loose threads held at one end (threads.wgsl runs; thread_prep.wgsl places their roots),
// per frame: each solved into TAIL_SEGS + 1 points that runCenter (thread_geom.wgsl) follows.

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> runs: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> threadCount: array<u32>;
@group(0) @binding(4) var<storage, read_write> runGeo: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> tails: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> tailCount: array<atomic<u32>>;
// the cloth's faces: the solver's self-contact face hash (selfcontact.wgsl), rebuilt every substep
@group(0) @binding(7) var<storage, read> nodes: array<NodeStaticR>;
@group(0) @binding(8) var<storage, read> faceCells: array<u32>;
@group(0) @binding(9) var<storage, read> faceItems: array<vec4<f32>>;

override MAX_RUNS: u32 = 65536u;

// A yarn held at one end hangs beyond the last span its chain holds as a heavy elastic rod clamped along
// the yarn's direction there (as runCenter left it). Freed and untwisted, its fibres slide on one
// another, so it bends as its fibres do side by side (count x a fibre's stiffness), under its own
// weight, with the curl its spinning set in it. Where its shape would pass through the cloth it rests
// on it instead, on the side it came from. Solved afresh each frame (equilibrium: a yarn this light and
// stiff settles faster than a frame), so it carries no state across the runs' rebuilds.

// bending stiffness of a cotton fibre: E (8 GPa) x I of its collapsed ribbon (20 x 7 um), N m^2
const FIBRE_EI: f32 = 4.6e-12;
const GRAVITY: f32 = 9.81;
// relaxed fixed-point iterations of the shape at most; the load is ramped in over more of them the
// longer the tail is against its bending length (a short tail barely bends), and the iteration stops
// once no segment turns by more than TAIL_TOL (rad) between iterations
const TAIL_ITERS: u32 = 16u;
const TAIL_TOL: f32 = 2e-4;
// the most a segment turns per iteration (rad): bounds the fixed point's steps
const TAIL_MAX_TURN: f32 = 0.6;
// the clearance a tail keeps from the cloth's collision faces (m) beyond its own radius: the cloth's
// half thickness and the rendered surface's bulge over the faces (selfcontact.ts distance)
const TAIL_CLEAR: f32 = 0.5e-3;
// the tail ignores the faces around its root (the cloth it comes out of) for this arc length (m)
const TAIL_ROOT_FREE: f32 = 3.0e-3;

// v turned about unit axis a by angle phi (Rodrigues).
fn turn(v: vec3<f32>, a: vec3<f32>, phi: f32) -> vec3<f32> {
  let c = cos(phi);
  let s = sin(phi);
  return v * c + cross(a, v) * s + a * (dot(a, v) * (1.0 - c));
}

// v turned by rotation vector w (axis w / |w|, angle |w|), the angle bounded.
fn turnBy(v: vec3<f32>, w: vec3<f32>) -> vec3<f32> {
  let phi = length(w);
  if (phi < 1e-9) { return v; }
  return turn(v, w / phi, min(phi, TAIL_MAX_TURN));
}

struct Contact {
  hit: bool,
  // where the tail rests on the face, and the face's normal toward the tail
  p: vec3<f32>,
  n: vec3<f32>,
};

// The cloth faces a tail can meet: those of the solver's face hash whose bounds come within its reach
// of the sphere it can occupy (its length about its root), gathered once per tail.
const MAX_CAND: u32 = 64u;
var<private> cand: array<u32, MAX_CAND>;
var<private> candN: u32;

fn gatherFaces(q: vec3<f32>, radius: f32) {
  candN = 0u;
  let T = FACE_MASK + 1u;
  // a face is hashed by its centroid; a cell's faces reach at most a cell's fraction beyond it
  let lo = vec3<i32>(floor((q - vec3<f32>(radius)) / FACE_CELL));
  let hi = vec3<i32>(floor((q + vec3<f32>(radius)) / FACE_CELL));
  for (var cz = lo.z; cz <= hi.z; cz++) {
    for (var cy = lo.y; cy <= hi.y; cy++) {
      for (var cx = lo.x; cx <= hi.x; cx++) {
        let h = ((u32(cx) * 73856093u) ^ (u32(cy) * 19349663u) ^ (u32(cz) * 83492791u)) & FACE_MASK;
        let start = faceCells[h];
        let count = faceCells[T + h];
        for (var k = 0u; k < count; k++) {
          let i1 = faceItems[2u * (start + k)];
          if (length(q - i1.xyz) - i1.w > radius || candN >= MAX_CAND) { continue; }
          cand[candN] = u32(faceItems[2u * (start + k) + 1u].w);
          candN++;
        }
      }
    }
  }
}

// The gathered face that segment a -> b of a tail meets (crosses, or comes within reach of at b);
// the faces around the root node are skipped while `near` is set.
fn tailContact(a: vec3<f32>, b: vec3<f32>, reach: f32, root: u32, near: bool) -> Contact {
  var o: Contact;
  o.hit = false;
  let rijl = nodes[root].ijl;
  let ri = i32(rijl & 0xFFFu);
  let rj = i32((rijl >> 12u) & 0xFFFu);
  var best = 1e9;
  for (var k = 0u; k < candN; k++) {
    let f = cand[k];
    let fijl = nodes[f].ijl;
    if (near && (fijl >> 24u) == (rijl >> 24u) && abs(i32(fijl & 0xFFFu) - ri) <= 2 && abs(i32((fijl >> 12u) & 0xFFFu) - rj) <= 2) { continue; }
    let L = R.layers[fijl >> 24u].lattice;
    let p00 = x[f].xyz;
    let p10 = x[f + 1u].xyz;
    let p01 = x[f + L.y].xyz;
    let p11 = x[f + L.y + 1u].xyz;
    let cen = 0.25 * (p00 + p10 + p01 + p11);
    // the face's corners lie within its half diagonal of the centroid
    let fr = 0.5 * max(length(p11 - p00), length(p10 - p01));
    if (length(b - cen) > fr + reach + length(b - a)) { continue; }
    let nf = cross(p11 - p00, p01 - p10);
    let nl = length(nf);
    if (nl < 1e-12) { continue; }
    let n = nf / nl;
    let sa = dot(a - cen, n);
    let sb = dot(b - cen, n);
    // where the segment meets the face's plane (or b, if it only comes near): within the face?
    let crosses = sa * sb < 0.0;
    let t = select(1.0, sa / (sa - sb), crosses);
    let m = mix(a, b, t);
    let u = p10 - p00;
    let v = p01 - p00;
    let uu = dot(u, u);
    let uv = dot(u, v);
    let vv = dot(v, v);
    let den = uu * vv - uv * uv;
    if (abs(den) < 1e-18) { continue; }
    let em = m - p00;
    let s1 = (vv * dot(em, u) - uv * dot(em, v)) / den;
    let s2 = (uu * dot(em, v) - uv * dot(em, u)) / den;
    if (s1 < -0.05 || s1 > 1.05 || s2 < -0.05 || s2 > 1.05) { continue; }
    if (!crosses && abs(sb) >= reach) { continue; }
    let d = select(abs(sb), 0.0, crosses);
    if (d >= best) { continue; }
    best = d;
    // rest on the side the tail came from
    let up = select(-n, n, sa >= 0.0);
    o.hit = true;
    o.n = up;
    o.p = m + up * reach - n * dot(m - cen, n);
  }
  return o;
}

@compute @workgroup_size(64)
fn threadTail(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= min(threadCount[0], MAX_RUNS)) { return; }
  let h = runGeo[RUN_GEO * r];
  let ri = runInfo(runs[r]);
  if (h.x > 0.5 || ri.heldA == ri.heldB || h.w <= 0.1e-3) { return; }
  let k = atomicAdd(&tailCount[0], 1u);
  if (k >= MAX_TAILS) { return; }
  let root = runGeo[RUN_GEO * r + 7u];
  tailSolve(ri, root.xyz, runGeo[RUN_GEO * r + 8u].xyz, u32(root.w), k, h.w);
  runGeo[RUN_GEO * r] = vec4<f32>(h.xy, f32(k), h.w);
}

fn tailSolve(ri: RunInfo, q: vec3<f32>, dir0: vec3<f32>, root: u32, k: u32, len: f32) {
  let LR = R.layers[ri.layer];
  let fibres = select(LR.loose.z, LR.loose.w, ri.fam == 1u);
  let EI = fibres * FIBRE_EI;
  let w = fibres * FIBRE_LIN * GRAVITY;
  let h = len / f32(TAIL_SEGS);
  let rY = select(LR.loose.x, LR.loose.y, ri.fam == 1u);
  // the curl spinning set in the yarn: radius 5 to 20 mm about an axis across it
  let seed = hash11((ri.layer * 7919u + ri.yarn) * 2u + ri.fam + 0x7a11u);
  let across = normalize(cross(dir0, vec3<f32>(hashf(seed) - 0.5, hashf(seed + 1u) - 0.5, hashf(seed + 2u) - 0.5)) + vec3<f32>(1e-9, 0.0, 0.0));
  let curl = across / (5e-3 + 15e-3 * hashf(seed + 3u));
  // rotation vectors per segment (its turn from the one before, per unit length): the weight beyond
  // each node bends it by its moment over EI, plus the curl
  var kap: array<vec3<f32>, TAIL_SEGS>;
  var t: array<vec3<f32>, TAIL_SEGS>;
  var pts: array<vec3<f32>, TAIL_SEGS_1>;
  for (var i = 0u; i < TAIL_SEGS; i++) { kap[i] = curl; }
  // bending length (EI / w)^(1/3): the scale over which the weight bends it
  let ramp = clamp(ceil(4.0 * len / pow(EI / w, 1.0 / 3.0)), 1.0, f32(TAIL_ITERS / 2u));
  for (var it = 0u; it < TAIL_ITERS; it++) {
    var ti = dir0;
    pts[0] = q;
    for (var i = 0u; i < TAIL_SEGS; i++) {
      ti = turnBy(ti, kap[i] * select(h, 0.5 * h, i == 0u));
      t[i] = ti;
      pts[i + 1u] = pts[i] + ti * h;
    }
    let load = w * h * min(1.0, f32(it + 1u) / ramp);
    var S = vec3<f32>(0.0);
    var m = 0.0;
    var change = 0.0;
    for (var j = 0; j < i32(TAIL_SEGS); j++) {
      let i = TAIL_SEGS - 1u - u32(j);
      S += pts[i] + 0.5 * h * t[i];
      m += 1.0;
      // moment about node i of the weight of the segments beyond it
      let M = cross(S - m * pts[i], vec3<f32>(0.0, 0.0, -load));
      let kn = mix(kap[i], M / EI + curl, 0.5);
      change = max(change, length(kn - kap[i]) * h);
      kap[i] = kn;
    }
    if (f32(it + 1u) >= ramp && change < TAIL_TOL) { break; }
  }
  // the shape of the converged turns
  var tf = dir0;
  for (var i = 0u; i < TAIL_SEGS; i++) {
    tf = turnBy(tf, kap[i] * select(h, 0.5 * h, i == 0u));
    t[i] = tf;
  }
  // rest on the cloth: march the shape; where a segment would pass through a face (or come within the
  // clearance of one), it ends on the face instead and the rest of the shape turns with it
  let reach = rY + TAIL_CLEAR;
  // faces within the tail's length of its root (plus its reach and a face's half diagonal, ~3 mm)
  if (FACE_HASH) { gatherFaces(q, len + reach + 3e-3); }
  var p = q;
  var arc = 0.0;
  let b = k * (TAIL_SEGS + 1u);
  tails[b] = vec4<f32>(q, 0.0);
  for (var i = 0u; i < TAIL_SEGS; i++) {
    var nxt = p + t[i] * h;
    arc += h;
    if (FACE_HASH) {
      let c = tailContact(p, nxt, reach, root, arc < TAIL_ROOT_FREE);
      if (c.hit) {
        // keep the segment's length along the face from where it touched
        let dir = normalize(c.p - p + vec3<f32>(1e-12, 0.0, 0.0));
        let ax = cross(t[i], dir);
        let al = length(ax);
        if (al > 1e-6) {
          let phi = atan2(al, dot(t[i], dir));
          for (var j = i + 1u; j < TAIL_SEGS; j++) { t[j] = turn(t[j], ax / al, phi); }
        }
        nxt = p + dir * h;
        // a segment pressed into the face slides along it
        let into = dot(nxt - c.p, c.n);
        if (into < 0.0) { nxt -= c.n * into; }
      }
    }
    // the floor, likewise: a segment that would pass below it (its axis a radius above the plane) ends
    // where it touches and the rest lies along the floor
    let zf = R.floor.x + rY;
    if (R.floor.y > 0.5 && nxt.z < zf) {
      let dz = p.z - nxt.z;
      let touch = mix(p, nxt, select(0.0, clamp((p.z - zf) / dz, 0.0, 1.0), dz > 1e-12));
      var dir = touch - p;
      if (dot(dir, dir) < 1e-14) { dir = vec3<f32>(t[i].xy, 0.0); }
      let dl = length(dir);
      if (dl > 1e-9) {
        dir /= dl;
        let ax = cross(t[i], dir);
        let al = length(ax);
        if (al > 1e-6) {
          let phi = atan2(al, dot(t[i], dir));
          for (var j = i + 1u; j < TAIL_SEGS; j++) { t[j] = turn(t[j], ax / al, phi); }
        }
        nxt = p + dir * h;
      }
      nxt.z = max(nxt.z, min(zf, p.z));
    }
    tails[b + i + 1u] = vec4<f32>(nxt, 0.0);
    p = nxt;
  }
}
