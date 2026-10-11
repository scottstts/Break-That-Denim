// Dynamic node state (x.w): 0 = cloth, 1 = detached (part of a piece no longer connected to a mount or
// a grip), 2 = retired (left the scene: frozen, disconnected and not drawn). A detached piece stays in
// the cloth solve with the same mechanics: its spans carry load, damage and break, and it collides as
// cloth. The state only decides what may hold it (no grip takes it), how it is drawn (a surface of its
// own) and that it is retired once it leaves the scene. Detached pieces are found on the CPU from the
// packed connectivity.

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> segs: array<Seg>;
@group(0) @binding(2) var<storage, read_write> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> packed: array<u32>;
@group(0) @binding(5) var<storage, read> freed: array<u32>;

// Per node: bit0 / bit1 = active span of family u / v owned by the node, bit2 = not cloth (x.w > 0),
// bit3 = retired (x.w > 1.5).
@compute @workgroup_size(128)
fn pack(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  var b = 0u;
  if (segActive(segs[2u * n])) { b |= 1u; }
  if (segActive(segs[2u * n + 1u])) { b |= 2u; }
  if (x[n].w > 0.5) { b |= 4u; }
  if (x[n].w > 1.5) { b |= 8u; }
  packed[n] = b;
}

// Nodes flagged by the CPU become detached (irreversible).
@compute @workgroup_size(128)
fn applyFreed(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  if (freed[n] == 0u || x[n].w > 0.5) { return; }
  x[n].w = 1.0;
}
