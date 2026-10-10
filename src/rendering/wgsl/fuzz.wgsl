// Loose fibres (render-only, attached to their material): the fringe of untwisted fibre standing out
// of every free edge (rooted along the cracks of torn cells, on each side's own surface), and fibres
// splaying from frayed tail ends. Each drawn fibre is a short curved tuft tens of microns wide: far
// below a pixel, so it is drawn as a >= 1 px ribbon whose alpha is the true pixel coverage, resolved
// by alpha-to-coverage on the 4x MSAA target.

@group(0) @binding(0) var<uniform> F: FrameU;
@group(0) @binding(1) var<uniform> R: RenderU;
@group(0) @binding(2) var<storage, read> nodes: array<NodeStaticR>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> nrm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> segw: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> tails: array<f32>;
@group(0) @binding(7) var<storage, read> lists: array<u32>;
@group(0) @binding(8) var<storage, read> wearR: array<f32>;
@group(0) @binding(9) var shadowMap: texture_depth_2d;
@group(0) @binding(10) var shadowSampler: sampler_comparison;
@group(0) @binding(11) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> tornGeo: array<vec4<f32>>;

override FUZZ_PER_TAIL: u32 = 12u;
const FIBER_PTS: u32 = 7u;

struct VOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) tangent: vec3<f32>,
  @location(2) alpha: f32,
  @location(3) @interpolate(flat) id: vec4<u32>, // unused, family, layer, seed
};

fn hidden() -> VOut {
  var o: VOut;
  o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
  return o;
}

// Vertex vi of a fibre from `root` along unit `dir` over `len`, bent by `bend` (offset at the tip,
// growing quadratically) and crimped by `wave` (offset amplitude of `waves` half-periods along it),
// `width` wide (m), thinning toward its free tip. Cotton fibres are convoluted ribbons and loose ones
// keep the twist crimp of their yarn, so they never stand out as straight needles.
fn strand(root: vec3<f32>, dir: vec3<f32>, len: f32, bend: vec3<f32>, wave: vec3<f32>, waves: f32, width: f32, vi: u32, side: vec3<f32>, id: vec4<u32>) -> VOut {
  let t = f32(vi >> 1u) / f32(FIBER_PTS - 1u);
  let ph = PI * waves * t;
  let P = root + dir * (len * t) + bend * (t * t) + wave * (sin(ph) * t);
  let dP = normalize(dir * len + bend * (2.0 * t) + wave * (sin(ph) + PI * waves * t * cos(ph)));
  let sideSign = select(-1.0, 1.0, (vi & 1u) == 1u);
  let V = normalize(F.camPos.xyz - P);
  var B = cross(dP, V);
  let bl = length(B);
  if (bl < 1e-6) { B = side; } else { B = B / bl; }
  let pixW = F.camPos.w * length(F.camPos.xyz - P);
  let halfW = max(0.5 * width, 0.5 * pixW);
  var o: VOut;
  let W = P + sideSign * halfW * B;
  o.pos = F.viewProj * vec4<f32>(W, 1.0);
  o.world = W;
  o.tangent = dP;
  o.alpha = clamp(width / max(pixW, 1e-9), 0.0, 1.0) * (1.0 - 0.65 * t);
  o.id = id;
  return o;
}

// ---------------------------------------------------------------- edge fringe

// Point (s, t) of a torn group's surface (its 16-point net in tornGeo at `slot`): position, d/ds, d/dt.
struct NetPoint {
  p: vec3<f32>,
  ds: vec3<f32>,
  dt: vec3<f32>,
};

fn netAt(slot: u32, s: f32, t: f32) -> NetPoint {
  var rows: array<vec3<f32>, 4>;
  var drows: array<vec3<f32>, 4>;
  for (var r = 0u; r < 4u; r++) {
    let p0 = tornGeo[slot + 4u * r].xyz;
    let p1 = tornGeo[slot + 4u * r + 1u].xyz;
    let p2 = tornGeo[slot + 4u * r + 2u].xyz;
    let p3 = tornGeo[slot + 4u * r + 3u].xyz;
    rows[r] = cmr(p0, p1, p2, p3, s);
    drows[r] = cmrd(p0, p1, p2, p3, s);
  }
  var o: NetPoint;
  o.p = cmr(rows[0], rows[1], rows[2], rows[3], t);
  o.ds = cmr(drows[0], drows[1], drows[2], drows[3], t);
  o.dt = cmrd(rows[0], rows[1], rows[2], rows[3], t);
  return o;
}

@vertex
fn vsEdgeFuzz(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let k = ii / FIBERS_PER_CELL;
  let fi = ii % FIBERS_PER_CELL;
  let base = k * TORN_STRIDE;
  let info = tornInfo(tornGeo[base]);
  let LR = R.layers[info.w];
  let size = vec2<f32>(f32(bundleOf(LR, 1u)) * LR.pitch.x, f32(bundleOf(LR, 0u)) * LR.pitch.y);
  let mask = info.y;
  let nc = crackCountOf(mask);
  let seed = hash11(info.x * 2654435761u + fi * 97u + 11u);
  let ci = fi % nc;
  let blo = tornGeo[base + TORN_CRACKS + 2u * ci];
  let bhi = tornGeo[base + TORN_CRACKS + 2u * ci + 1u];
  let tau = hashf(seed + 1u);
  let pc = bezierP(blo, bhi, tau);
  let tg = normalize(bezierTangentP(blo, bhi, tau) + vec2<f32>(1e-9, 0.0));
  var nrm2 = vec2<f32>(-tg.y, tg.x);
  if (((seed >> 9u) & 1u) == 1u) { nrm2 = -nrm2; }
  // the corner on this side of the crack picks the group the fibre grows from
  let kb = countOneBits(mask);
  var corner = 0u;
  if (kb >= 2u) {
    var best = -1e9;
    var lo = 0u;
    var hi = 4u;
    if (kb >= 3u) {
      // a crack to the junction separates the corners on either side of its own edge
      let e = nthEdge(mask, ci);
      lo = e;
      hi = e + 2u;
    }
    for (var q = lo; q < hi; q++) {
      let c = ccwCorner(q);
      let d = dot(cornerPos(c, size) - pc, nrm2);
      if (d > best) {
        best = d;
        corner = c;
      }
    }
    if (dot(cornerPos(corner, size) - pc, nrm2) < 0.0) { nrm2 = -nrm2; }
    corner = groupLeader(mask, groupOf(mask, ccwCorner(corner)));
  }
  let slot = tornSlot(k, corner);
  if (tornGeo[slot].w < 0.0) { return hidden(); }
  // rooted a little inside the edge, densest at the very edge
  let h2 = hashf(seed + 2u);
  let rootRest = clamp(pc + nrm2 * (h2 * h2 * 0.9e-3), vec2<f32>(0.0), size);
  let np = netAt(slot, rootRest.x / size.x, rootRest.y / size.y);
  let Tu = normalize(np.ds);
  let Tv = normalize(np.dt);
  let N = normalize(cross(np.ds, np.dt));
  // outward across the crack, drawn toward the cut yarns' direction (the fringe is their ends),
  // standing up out of the rolled rim
  let out = -nrm2;
  let fam = select(vec2<f32>(0.0, sign(out.y + 1e-9)), vec2<f32>(sign(out.x + 1e-9), 0.0), abs(out.x) > abs(out.y));
  let spread = (hashf(seed + 3u) - 0.5) * 2.2;
  let d2 = normalize(mix(out, fam, 0.4) + vec2<f32>(-out.y, out.x) * spread);
  let lift = 0.1 + 0.9 * hashf(seed + 4u);
  let dir = normalize(Tu * d2.x + Tv * d2.y + N * lift);
  // lengths: mostly short fibre, a few long untwisted ends that droop
  let r = hashf(seed + 5u);
  var len = 0.25e-3 + 1.5e-3 * r * r * r;
  if (hashf(seed + 6u) < 0.05) { len += 1.0e-3 + 2.5e-3 * hashf(seed + 7u); }
  let curl = vec3<f32>(hashf(seed + 8u) - 0.5, hashf(seed + 9u) - 0.5, hashf(seed + 10u) - 0.5);
  let droop = clamp(len / 1.5e-3, 0.0, 1.0);
  let bend = curl * (0.9 * len) - N * (0.6 * len * lift * droop) + vec3<f32>(0.0, 0.0, -0.3 * len * droop);
  let wv = vec3<f32>(hashf(seed + 13u) - 0.5, hashf(seed + 14u) - 0.5, hashf(seed + 15u) - 0.5);
  let wave = wv * (0.35 * len);
  let waves = 1.0 + 3.0 * hashf(seed + 16u);
  // tufts of a few fibres
  let width = 25e-6 + 30e-6 * hashf(seed + 11u);
  let famId = select(0u, 1u, hashf(seed + 12u) < 0.35);
  return strand(np.p, dir, len, bend, wave, waves, width, vi, Tu, vec4<u32>(0u, famId, info.w, seed));
}

// ---------------------------------------------------------------- frayed tail ends

@vertex
fn vsTailFuzz(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  let b = bundleMax(R.layers[0]);
  let yi = ii / FUZZ_PER_TAIL;
  let e = tailEntry(yi / b);
  let q = yi % b;
  if (q >= bundleOf(R.layers[nodes[e >> 2u].ijl >> 24u], (e & 3u) >> 1u) || tailLength(e, q) < TAIL_MIN) { return hidden(); }
  let seed = hash11(e * 613u + q * 137u + (ii % FUZZ_PER_TAIL) * 6007u + 3u);
  // fibres concentrate toward the untwisted free end
  let r = hashf(seed);
  let y = tailYarn(e, q, 1.0 - r * r * 0.8);
  // root on the yarn surface at a random azimuth; untwisted end fibres splay mostly along the yarn
  let az = 6.2831853 * hashf(seed + 1u);
  let u1 = normalize(cross(y.t, y.n));
  let u2 = cross(u1, y.t);
  let radial = cos(az) * u1 + sin(az) * u2;
  let tilt = (hashf(seed + 2u) - 0.5) * 2.2;
  let dir = normalize(normalize(radial + y.t * tilt) + y.t * 2.2);
  let rl = hashf(seed + 3u);
  let len = 1.8 * (0.12e-3 + 0.75e-3 * rl * rl);
  let bend = vec3<f32>(hashf(seed + 4u) - 0.5, hashf(seed + 5u) - 0.5, hashf(seed + 6u) - 0.8) * (0.9 * len);
  let wave = vec3<f32>(hashf(seed + 7u) - 0.5, hashf(seed + 8u) - 0.5, hashf(seed + 9u) - 0.5) * (0.3 * len);
  return strand(y.p + radial * y.radius, dir, len, bend, wave, 1.0 + 2.0 * hashf(seed + 10u), 30e-6, vi, u1, vec4<u32>(0u, y.fam, y.layer, seed));
}

@fragment
fn fsFuzz(in: VOut) -> @location(0) vec4<f32> {
  let V = normalize(F.camPos.xyz - in.world);
  let T = normalize(in.tangent);
  let N = normalize(V - dot(V, T) * T);
  // loose fibres are mostly the pale core and abraded surface fibres
  let pale = hashf(in.id.w + 9u) < 0.7;
  var albedo = vec3<f32>(0.34, 0.37, 0.44);
  if (in.id.y == 0u) { albedo = vec3<f32>(0.38, 0.38, 0.38); }
  if (!pale && in.id.y == 1u) { albedo = vec3<f32>(0.08, 0.12, 0.27); }
  let key = keyLight(in.world);
  let sh = shadowFactor(in.world, N);
  let amb = F.ambient.rgb;
  let c = fiberShade(N, T, V, key.xyz, albedo, 0.2) * (key.w * sh) + amb * albedo;
  return vec4<f32>(c, in.alpha);
}
