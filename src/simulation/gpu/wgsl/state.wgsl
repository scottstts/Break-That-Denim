// Dynamic node state (x.w): 0 = cloth, 1 = ballistic debris (part of a piece no longer connected to
// a mount or the grip: it leaves the linear solve and falls freely), 2 = retired (fell out of the
// scene: frozen and not drawn). Detached pieces are found on the CPU from the packed connectivity;
// their rigid modes are resolved only by inertia, which f32 cannot represent next to yarn stiffness.

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> segs: array<Seg>;
@group(0) @binding(2) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> fixedN: array<u32>;
@group(0) @binding(4) var<storage, read_write> packed: array<u32>;
@group(0) @binding(5) var<storage, read> freed: array<u32>;

// Per node: bit0 / bit1 = active span of family u / v owned by the node, bit2 = not cloth (x.w > 0).
@compute @workgroup_size(128)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  var b = 0u;
  if (segActive(segs[2u * n])) { b |= 1u; }
  if (segActive(segs[2u * n + 1u])) { b |= 2u; }
  if (x[n].w > 0.5) { b |= 4u; }
  packed[n] = b;
}

// Nodes flagged by the CPU become ballistic (irreversible) and leave the linear solve.
@compute @workgroup_size(128)
fn applyFreed(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  if (freed[n] == 0u || x[n].w > 0.5) { return; }
  x[n].w = 1.0;
  fixedN[n] = 1u;
}
