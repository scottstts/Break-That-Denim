// Exclusive prefix sum over u32 counts in blocks of 512 (two elements per thread), with a block-sum
// array for the next level. Levels are chained by the host until one block remains.

const BLOCK: u32 = 512u;

@group(0) @binding(0) var<uniform> S: vec4<u32>; // x: element count
@group(0) @binding(1) var<storage, read_write> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> sums: array<u32>;

var<workgroup> tmp: array<u32, 512>;

@compute @workgroup_size(256)
fn scanBlocks(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * BLOCK;
  let n = S.x;
  let a = base + 2u * t;
  tmp[2u * t] = select(0u, data[a], a < n);
  tmp[2u * t + 1u] = select(0u, data[a + 1u], a + 1u < n);
  // up-sweep
  var offset = 1u;
  for (var d = BLOCK >> 1u; d > 0u; d = d >> 1u) {
    workgroupBarrier();
    if (t < d) {
      let ai = offset * (2u * t + 1u) - 1u;
      let bi = offset * (2u * t + 2u) - 1u;
      tmp[bi] += tmp[ai];
    }
    offset = offset << 1u;
  }
  workgroupBarrier();
  if (t == 0u) {
    sums[wg.x] = tmp[BLOCK - 1u];
    tmp[BLOCK - 1u] = 0u;
  }
  // down-sweep
  for (var d = 1u; d < BLOCK; d = d << 1u) {
    offset = offset >> 1u;
    workgroupBarrier();
    if (t < d) {
      let ai = offset * (2u * t + 1u) - 1u;
      let bi = offset * (2u * t + 2u) - 1u;
      let v = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += v;
    }
  }
  workgroupBarrier();
  if (a < n) { data[a] = tmp[2u * t]; }
  if (a + 1u < n) { data[a + 1u] = tmp[2u * t + 1u]; }
}

// Add the scanned block sums back to every element of each block.
@compute @workgroup_size(256)
fn addBlockSums(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let base = wg.x * BLOCK;
  let n = S.x;
  let add = sums[wg.x];
  let a = base + 2u * t;
  if (a < n) { data[a] += add; }
  if (a + 1u < n) { data[a + 1u] += add; }
}
