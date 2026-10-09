// Adaptive Newton gate (see GpuSolver.encodeSubstep).

@group(0) @binding(0) var<uniform> Mode: vec4<u32>; // x = Newton iteration just completed
@group(0) @binding(1) var<storage, read> counters: array<u32>; // [4 + k] largest update of iteration k (f32 bits)
@group(0) @binding(2) var<storage, read_write> gateArgs: array<u32>;
@group(0) @binding(3) var<storage, read> gateSizes: array<u32>; // [count, tolerance bits, sizes...]

// After Newton iteration Mode.x: enable the next (optional) iteration only while this one still moved
// some node by more than the tolerance. Writes indirect dispatch arguments for every dispatch size of
// a Newton iteration (zero workgroups when converged); a skipped iteration leaves its slot at 0, so
// every later optional iteration stays off.
@compute @workgroup_size(64)
fn gate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let count = gateSizes[0];
  let k = gid.x;
  if (k >= count) { return; }
  let moved = bitcast<f32>(counters[4u + Mode.x]);
  let on = moved > bitcast<f32>(gateSizes[1]);
  let base = 3u * count * (Mode.x + 1u);
  gateArgs[base + 3u * k] = select(0u, gateSizes[2u + k], on);
  gateArgs[base + 3u * k + 1u] = 1u;
  gateArgs[base + 3u * k + 2u] = 1u;
}

