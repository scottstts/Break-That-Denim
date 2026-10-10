// Cross-layer contact between crossing nodes. The contact distance (midsurface separation of two plies,
// ~0.85 mm) is about twice the crossing spacing, so the summed node-node penalty field is a smooth
// sheet-like barrier that still opens where material is missing. Detection runs once per substep on a
// spatial hash keyed by (cell, layer) and also produces an OGC-style displacement bound per node;
// contact and friction terms are re-evaluated at every Newton iterate from the stored pairs.

const K: u32 = 4u;
const NONE: u32 = 0xFFFFFFFFu;

struct ContactU {
  // cell size (m), table mask, unused, unused
  grid: vec4<f32>,
  // penalty stiffness (N/m), friction coefficient, friction regularization speed (m/s), bound fraction
  law: vec4<f32>,
  // per layer pair contact distance: [0-1, 0-2, 1-2, same layer], m
  dist: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<uniform> CU: ContactU;
@group(0) @binding(2) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> dx: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> segs: array<Seg>;
@group(0) @binding(6) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> cellCursor: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> cellItems: array<u32>;
@group(0) @binding(9) var<storage, read_write> pairs: array<u32>;
// 2 per node (selfcontact.wgsl): own bound (normal, largest approach), reciprocal bound (none here)
@group(0) @binding(10) var<storage, read_write> cmeta: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> ext: array<vec4<f32>>;  // 3 per node

fn layerOf(n: u32) -> u32 { return nodes[n].ijl >> 24u; }

fn cellOf(p: vec3<f32>) -> vec3<i32> { return vec3<i32>(floor(p / CU.grid.x)); }

fn hashCell(c: vec3<i32>, layer: u32) -> u32 {
  let h = (u32(c.x) * 73856093u) ^ (u32(c.y) * 19349663u) ^ (u32(c.z) * 83492791u) ^ (layer * 2654435761u);
  return h & bitcast<u32>(CU.grid.y);
}

fn pairDistance(la: u32, lb: u32) -> f32 {
  if (la == lb) { return CU.dist.w; }
  let s = la + lb;
  if (s == 1u) { return CU.dist.x; }
  if (s == 2u) { return CU.dist.y; }
  return CU.dist.z;
}

fn cur(n: u32) -> vec3<f32> { return x[n].xyz + dx[n].xyz; }

@compute @workgroup_size(128)
fn hashCount(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  atomicAdd(&cellCount[hashCell(cellOf(cur(n)), layerOf(n))], 1u);
}

// After the exclusive scan, cellCount holds cell start offsets.
@compute @workgroup_size(128)
fn hashFill(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let h = hashCell(cellOf(cur(n)), layerOf(n));
  let slot = atomicLoad(&cellCount[h]) + atomicAdd(&cellCursor[h], 1u);
  cellItems[slot] = n;
}

@compute @workgroup_size(128)
fn detect(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let p = cur(n);
  let ln = layerOf(n);
  var best: array<u32, 4>;
  var bestD: array<f32, 4>;
  for (var k = 0u; k < K; k++) {
    best[k] = NONE;
    bestD[k] = 1e9;
  }
  var dmin = 1e9;
  var nmin = vec3<f32>(0.0);
  let c = cellOf(p);
  for (var lo = 0u; lo < P.counts.y; lo++) {
    if (lo == ln) { continue; }
    let r = pairDistance(ln, lo);
    for (var dz = -1; dz <= 1; dz++) {
      for (var dy = -1; dy <= 1; dy++) {
        for (var dxc = -1; dxc <= 1; dxc++) {
          let h = hashCell(c + vec3<i32>(dxc, dy, dz), lo);
          let start = atomicLoad(&cellCount[h]);
          let count = atomicLoad(&cellCursor[h]);
          for (var q = 0u; q < count; q++) {
            let m = cellItems[start + q];
            if (layerOf(m) != lo) { continue; }
            let d = distance(p, cur(m));
            if (d - r < dmin && d > 1e-9) {
              dmin = d - r;
              nmin = (p - cur(m)) / d;
            }
            if (d >= r + 0.25 * CU.grid.x) { continue; }
            var cand = m;
            var cd = d - r;
            for (var k = 0u; k < K; k++) {
              if (cd < bestD[k]) {
                let ts = best[k];
                let td = bestD[k];
                best[k] = cand;
                bestD[k] = cd;
                cand = ts;
                cd = td;
              }
            }
          }
        }
      }
    }
  }
  for (var k = 0u; k < K; k++) { pairs[K * n + k] = best[k]; }
  // Directional bound (step.wgsl newtonApply): the step may approach the nearest foreign node only by
  // a share of its clearance plus half the contact distance, so it cannot tunnel through a layer.
  if (dmin < 1e8) {
    cmeta[2u * n] = vec4<f32>(nmin, CU.law.w * (max(0.0, dmin) + 0.5 * CU.dist.x));
  } else {
    cmeta[2u * n] = vec4<f32>(0.0, 0.0, 0.0, 1e9);
  }
  cmeta[2u * n + 1u] = vec4<f32>(0.0, 0.0, 0.0, 1e9);
}

@compute @workgroup_size(128)
fn terms(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  var g = vec3<f32>(0.0);
  var H = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
  let p = cur(n);
  let ln = layerOf(n);
  let kc = CU.law.x;
  let mu = CU.law.y;
  let epsU = CU.law.z * P.hg.x;
  let I3 = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
  for (var k = 0u; k < K; k++) {
    let m = pairs[K * n + k];
    if (m == NONE) { continue; }
    let dv = p - cur(m);
    let d = length(dv);
    let r = pairDistance(ln, layerOf(m));
    if (d >= r || d <= 1e-9) { continue; }
    let nn = dv / d;
    let lam = kc * (r - d);
    g -= lam * nn;
    let nnT = mat3x3<f32>(nn * nn.x, nn * nn.y, nn * nn.z);
    H += kc * nnT;
    // lagged-normal regularized Coulomb friction on the relative tangential step
    let rel = dx[n].xyz - dx[m].xyz;
    let ut = rel - dot(rel, nn) * nn;
    let kf = mu * lam / max(length(ut), epsU);
    g += kf * ut;
    H += kf * (I3 - nnT);
  }
  ext[3u * n] = vec4<f32>(g, 0.0);
  ext[3u * n + 1u] = vec4<f32>(H[0].x, H[0].y, H[0].z, H[1].y);
  ext[3u * n + 2u] = vec4<f32>(H[1].z, H[2].z, 0.0, 0.0);
}
