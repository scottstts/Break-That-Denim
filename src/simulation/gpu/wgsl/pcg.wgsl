// PCG vector kernels on the fine level. Scalars stay on the GPU: slot 2k = r.z of iteration k,
// slot 2k+1 = p.Ap of iteration k. Mirrors pcg() in reference/newtonMultigrid.ts.

const RED: u32 = 256u;
const GROUPS: u32 = 256u;

@group(0) @binding(0) var<uniform> F: LevelInfo;
@group(0) @binding(1) var<uniform> It: vec4<u32>; // x: iteration k, y: scalar slot, z: partial count
@group(0) @binding(2) var<storage, read> op: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> fixedN: array<u32>;
@group(0) @binding(4) var<storage, read_write> va: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> vb: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> vc: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> vd: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> ve: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> partial: array<f32>;
@group(0) @binding(10) var<storage, read_write> scal: array<f32>;

var<workgroup> red: array<f32, 256>;

fn reduceStore(lid: u32, wgid: u32, v: f32) {
  red[lid] = v;
  workgroupBarrier();
  var s = RED / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid == 0u) { partial[wgid] = red[0]; }
}

// vb = A va ; partial sums of va . vb
@compute @workgroup_size(256)
fn matvecDot(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  var acc = 0.0;
  for (var n = gid.x; n < F.counts.x; n += RED * GROUPS) {
    var y = vec3<f32>(0.0);
    if (fixedN[n] == 0u) {
      let cc = levelCoords(F, n);
      y = symMul(rowDiag(n), va[n].xyz);
      for (var c = 0u; c < ROW_COUPLINGS; c++) {
        let off = rowOffset(c);
        let m = levelNode(F, cc.z, i32(cc.x) + off.x, i32(cc.y) + off.y);
        if (m < 0) { continue; }
        y += symMul(rowCoupling(n, c), va[u32(m)].xyz);
      }
    }
    vb[n] = vec4<f32>(y, 0.0);
    acc += dot(y, va[n].xyz);
  }
  reduceStore(lid, wg.x, acc);
}

// partial sums of va . vb
@compute @workgroup_size(256)
fn dotPartial(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  var acc = 0.0;
  for (var n = gid.x; n < F.counts.x; n += RED * GROUPS) {
    acc += dot(va[n].xyz, vb[n].xyz);
  }
  reduceStore(lid, wg.x, acc);
}

@compute @workgroup_size(256)
fn reduceFinal(@builtin(local_invocation_index) lid: u32) {
  var acc = 0.0;
  for (var k = lid; k < It.z; k += RED) { acc += partial[k]; }
  red[lid] = acc;
  workgroupBarrier();
  var s = RED / 2u;
  loop {
    if (s == 0u) { break; }
    if (lid < s) { red[lid] += red[lid + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  if (lid == 0u) { scal[It.y] = red[0]; }
}

fn safeDiv(a: f32, b: f32) -> f32 { return select(0.0, a / b, b > 0.0); }

// va = delta, vb = p, vc = r, vd = Ap, ve = z:  delta += a p ; r -= a Ap ; z = 0
@compute @workgroup_size(256)
fn updateXR(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= F.counts.x) { return; }
  let k = It.x;
  let alpha = safeDiv(scal[2u * k], scal[2u * k + 1u]);
  va[n] = vec4<f32>(va[n].xyz + alpha * vb[n].xyz, 0.0);
  vc[n] = vec4<f32>(vc[n].xyz - alpha * vd[n].xyz, 0.0);
  ve[n] = vec4<f32>(0.0);
}

// va = p, vb = z: p = z + beta p (beta = rz_{k+1} / rz_k); k = 0xFFFFFFFF means p = z
@compute @workgroup_size(256)
fn updateP(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= F.counts.x) { return; }
  let k = It.x;
  var beta = 0.0;
  if (k != 0xFFFFFFFFu) { beta = safeDiv(scal[2u * k + 2u], scal[2u * k]); }
  va[n] = vec4<f32>(vb[n].xyz + beta * va[n].xyz, 0.0);
}

// va = 0 (and vb = 0)
@compute @workgroup_size(256)
fn clear2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= F.counts.x) { return; }
  va[n] = vec4<f32>(0.0);
  vb[n] = vec4<f32>(0.0);
}
