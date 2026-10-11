// Rigid-mode start of the Newton direction for detached pieces (rigid_frame.wgsl): PCG starts from
// x0 = Z E^-1 Z^T b, each piece's exact rigid response, with the residual b - A x0, instead of from
// zero. PCG minimises the error in the operator's norm over x0 plus its Krylov space, so the direction
// is never worse than that start. (Adding Z E^-1 Z^T r to every preconditioned residual instead
// counted a single node twice, its V-cycle having resolved it already: the direction turned against
// the contact holding it, and nodes resting on cloth were pumped off it.)
// rigidReduce: per piece, the rhs force and torque about the centroid and the rigid step
// (t, w) = E^-1 (force, torque); rigidApply: per node of a piece, x0 = t + w x r; rigidResidual:
// b -= A x0 (A x0 from the PCG matvec).

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> dx: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> pieceStart: array<u32>;
@group(0) @binding(4) var<storage, read> pieceNodes: array<u32>;
// per node: its piece, or NO_PIECE
@group(0) @binding(5) var<storage, read> pieceOf: array<u32>;
@group(0) @binding(6) var<storage, read> pieceFrame: array<vec4<f32>>;
// per piece: t (xyz), w (xyz) as two vec4
@group(0) @binding(7) var<storage, read_write> pieceStep: array<vec4<f32>>;
// the fine level's rhs b (B, made the residual of x0), the Newton direction (delta) and A x0 (PCG Ap)
@group(0) @binding(8) var<storage, read_write> r: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> z: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> Az: array<vec4<f32>>;

const RIGID_FRAME: u32 = 10u;
const NO_PIECE: u32 = 0xFFFFFFFFu;
const WG: u32 = 64u;
var<workgroup> red: array<array<vec3<f32>, 2>, 64>;
var<workgroup> count: u32;

fn pos(n: u32) -> vec3<f32> { return x[n].xyz + dx[n].xyz; }

fn inverseAt(o: u32, q: u32) -> f32 { return pieceFrame[o + 1u + q / 4u][q % 4u]; }

@compute @workgroup_size(64)
fn rigidReduce(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32, @builtin(num_workgroups) ng: vec3<u32>) {
  if (t == 0u) { count = pieceStart[0]; }
  let pieces = workgroupUniformLoad(&count);
  for (var p = wg.x; p < pieces; p += ng.x) {
    let o = RIGID_FRAME * p;
    let c = pieceFrame[o].xyz;
    let a = pieceStart[1u + p];
    let b = pieceStart[2u + p];
    var f = vec3<f32>(0.0);
    var m = vec3<f32>(0.0);
    for (var k = a + t; k < b; k += WG) {
      let n = pieceNodes[k];
      if (x[n].w > 1.5) { continue; }
      let rn = r[n].xyz;
      f += rn;
      m += cross(pos(n) - c, rn);
    }
    red[t] = array<vec3<f32>, 2>(f, m);
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
      if (t < s) {
        red[t][0] += red[t + s][0];
        red[t][1] += red[t + s][1];
      }
      workgroupBarrier();
    }
    if (t == 0u) {
      let g = array<f32, 6>(red[0][0].x, red[0][0].y, red[0][0].z, red[0][1].x, red[0][1].y, red[0][1].z);
      var q: array<f32, 6>;
      for (var i = 0u; i < 6u; i++) {
        var s = 0.0;
        for (var j = 0u; j < 6u; j++) { s += inverseAt(o, 6u * i + j) * g[j]; }
        q[i] = select(0.0, s, pieceFrame[o].w > 0.5);
      }
      pieceStep[2u * p] = vec4<f32>(q[0], q[1], q[2], 0.0);
      pieceStep[2u * p + 1u] = vec4<f32>(q[3], q[4], q[5], 0.0);
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(128)
fn rigidApply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let p = pieceOf[n];
  if (p == NO_PIECE || x[n].w > 1.5) { return; }
  let c = pieceFrame[RIGID_FRAME * p].xyz;
  let d = pieceStep[2u * p].xyz + cross(pieceStep[2u * p + 1u].xyz, pos(n) - c);
  z[n] = vec4<f32>(d, 0.0);
}

@compute @workgroup_size(128)
fn rigidResidual(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  r[n] = vec4<f32>(r[n].xyz - Az[n].xyz, 0.0);
}
