// Per frame, per free-edge crossing (pieces.wgsl edgeList): where its material is (material_map.wgsl),
// shared by all the fibres rooted on it (fuzz.wgsl).

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rsurf: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> kappaM: array<u32>;
@group(0) @binding(5) var<storage, read> edgeList: array<u32>;
// per edge crossing: position (w = 1, or 0 when it is not held), d/du and d/dv per crossing pitch
@group(0) @binding(6) var<storage, read_write> edgeRoots: array<vec4<f32>>;

@compute @workgroup_size(64)
fn edgeRootsMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = gid.x;
  if (e >= edgeList[MAX_EDGES]) { return; }
  let c = edgeList[e] & 0x0FFFFFFFu;
  var l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (c >= R.layers[q].field.x) { l = q; }
  }
  let LR = R.layers[l];
  let FL = LR.field;
  let mp = crossingPoint(l, i32((c - FL.x) % FL.y), i32((c - FL.x) / FL.y));
  let bu = f32(bundleOf(LR, 1u));
  let bv = f32(bundleOf(LR, 0u));
  edgeRoots[3u * e] = vec4<f32>(mp.p, select(0.0, 1.0, mp.valid));
  edgeRoots[3u * e + 1u] = vec4<f32>(mp.ds / bu, 0.0);
  edgeRoots[3u * e + 2u] = vec4<f32>(mp.dt / bv, 0.0);
}
