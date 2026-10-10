// Edge field (render_common.wgsl): per crossing the distance to the nearest crossing with a separated
// yarn, the intact fraction of its warp and pick, and the density of separated yarn ends around it,
// written to an rgba16float texture at crossing resolution. Distances come from jump flooding over the crossing grid with physical (anisotropic)
// pitches: nearest seeds are propagated over steps 16, 8, 4, 2, 1, 1 crossings, enough for EDGE_RMAX.
// Rebuilt only when the yarn field changed (edgeArgsMain consumes the solver's field event flag).

@group(0) @binding(0) var<uniform> R: RenderU;
// x: jump step (crossings)
@group(0) @binding(1) var<uniform> Step: vec4<u32>;
@group(0) @binding(2) var<storage, read> field: array<u32>;
@group(0) @binding(3) var<storage, read> seedIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> seedOut: array<u32>;
@group(0) @binding(5) var edgeOut: texture_storage_2d<rgba16float, write>;
// [0] built at least once since the renderer was created or reset
@group(0) @binding(6) var<storage, read_write> edgeSig: array<u32>;
// solver yarn field: [0] = 1 when a damage code changed since the last rebuild
@group(0) @binding(7) var<storage, read_write> fieldEvents: array<u32>;
// indirect dispatch of the rebuild passes (x, y, z)
@group(0) @binding(8) var<storage, read_write> edgeArgs: array<u32>;

const NO_SEED: u32 = 0xFFFFFFFFu;

struct Texel {
  l: u32,
  k: u32,
  m: u32,
  valid: bool,
};

fn texelOf(c: u32) -> Texel {
  var o: Texel;
  o.l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (c >= R.layers[q].field.x) { o.l = q; }
  }
  let F = R.layers[o.l].field;
  o.valid = c < F.x + F.y * F.z;
  let local = c - F.x;
  o.k = local % F.y;
  o.m = local / F.y;
  return o;
}

fn fieldTotal() -> u32 {
  let F = R.layers[R.counts.y - 1u].field;
  return F.x + F.y * F.z;
}

@compute @workgroup_size(1)
fn edgeArgsMain() {
  let changed = fieldEvents[0] != 0u || edgeSig[0] == 0u;
  fieldEvents[0] = 0u;
  edgeSig[0] = 1u;
  edgeArgs[0] = select(0u, (fieldTotal() + 127u) / 128u, changed);
  edgeArgs[1] = 1u;
  edgeArgs[2] = 1u;
}

@compute @workgroup_size(128)
fn seedInit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  let t = texelOf(c);
  if (!t.valid) { return; }
  let w = field[c];
  let cut = yarnCode(w, 0u) == CUT_R || yarnCode(w, 1u) == CUT_R;
  seedOut[c] = select(NO_SEED, t.k | (t.m << 16u), cut);
}

fn seedDist(L: LayerR, k: u32, m: u32, s: u32) -> f32 {
  if (s == NO_SEED) { return 1e9; }
  let d = vec2<f32>((f32(s & 0xFFFFu) - f32(k)) * L.pitch.x, (f32(s >> 16u) - f32(m)) * L.pitch.y);
  return dot(d, d);
}

@compute @workgroup_size(128)
fn seedStep(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  let t = texelOf(c);
  if (!t.valid) { return; }
  let LR = R.layers[t.l];
  let F = LR.field;
  let st = i32(Step.x);
  var best = seedIn[c];
  var bd = seedDist(LR, t.k, t.m, best);
  for (var dj = -1; dj <= 1; dj++) {
    for (var di = -1; di <= 1; di++) {
      if (di == 0 && dj == 0) { continue; }
      let k = i32(t.k) + di * st;
      let m = i32(t.m) + dj * st;
      if (k < 0 || m < 0 || k >= i32(F.y) || m >= i32(F.z)) { continue; }
      let s = seedIn[F.x + u32(m) * F.y + u32(k)];
      let d = seedDist(LR, t.k, t.m, s);
      if (d < bd) {
        bd = d;
        best = s;
      }
    }
  }
  seedOut[c] = best;
}

@compute @workgroup_size(128)
fn edgeWrite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  let t = texelOf(c);
  if (!t.valid) { return; }
  let LR = R.layers[t.l];
  let w = field[c];
  let d = min(sqrt(seedDist(LR, t.k, t.m, seedIn[c])) * 1e3, EDGE_RMAX);
  let warp = 1.0 - f32(yarnCode(w, 1u)) / 255.0;
  let pick = 1.0 - f32(yarnCode(w, 0u)) / 255.0;
  // separated yarn segments within about 1.2 mm (7 warps x 5 picks), where the rim needs it
  var density = 0.0;
  if (d < EDGE_RMAX) {
    let F = LR.field;
    var cuts = 0u;
    var n = 0u;
    for (var dm = -2; dm <= 2; dm++) {
      for (var dk = -3; dk <= 3; dk++) {
        let k = i32(t.k) + dk;
        let m = i32(t.m) + dm;
        if (k < 0 || m < 0 || k >= i32(F.y) || m >= i32(F.z)) { continue; }
        let ww = field[F.x + u32(m) * F.y + u32(k)];
        cuts += select(0u, 1u, yarnCode(ww, 0u) == CUT_R) + select(0u, 1u, yarnCode(ww, 1u) == CUT_R);
        n += 2u;
      }
    }
    density = f32(cuts) / f32(max(n, 1u));
  }
  textureStore(edgeOut, vec2<u32>(t.k, u32(LR.edge.x) + t.m), vec4<f32>(d, warp, pick, density));
}
