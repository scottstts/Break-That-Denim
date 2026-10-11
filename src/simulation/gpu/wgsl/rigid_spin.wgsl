// Spin start of a detached piece's substep (solver.ts encodePredict), after the predictor.
//
// The Newton iterate starts from h v. A substep's velocity is its chord (x1 - x0) / h, so for a
// spinning piece the rotation it made over the last substep (angle th = |w| h): repeated from the
// rotated position, the chord points outside the circle, and the start comes out stretched by th^2
// (0.3 % at 10 rad/s). That tension's geometric stiffness, ~1500 times a node's inertia, resists any
// change of the spin in the linearized system: two Newton iterations moved a piece spinning on the
// floor by a few percent of the friction torque's deceleration (it spun on for a second, as on ice;
// 8 iterations stop it in three substeps, as Coulomb friction does). The start is moved onto the
// rotation continued over the substep: repeating chord (I - R^T) r instead of (R - I) r is
// dx = h v + (R + R^T - 2 I) r = h v - 2 (1 - cos th) r_perp about the piece's mass centroid. Only where
// the iteration starts changes; the inertial target (pred) is untouched.
//
// spinReduce: per piece (one workgroup at a time) its centroid, momentum and inertia tensor, and
// w = I^-1 L; spinApply: per node of a piece, the correction.
//
// curveSet, each Newton iteration after the rigid-mode start (rigid_apply.wgsl rigidReduce): per node,
// its piece's rigid rotation in the direction and its arm from the piece's centroid (zero off pieces),
// along which the line search and the update move it (energy.wgsl steps, curveApply).

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> dx: array<vec4<f32>>;
// [0] piece count, [1 + p] first entry of piece p in pieceNodes, [1 + count] their total
@group(0) @binding(5) var<storage, read> pieceStart: array<u32>;
@group(0) @binding(6) var<storage, read> pieceNodes: array<u32>;
@group(0) @binding(7) var<storage, read> pieceOf: array<u32>;
// per piece: centroid (xyz), then the rotation vector w h (xyz); free until the Newton iterations
// (rigid_apply.wgsl pieceStep)
@group(0) @binding(8) var<storage, read_write> pieceStep: array<vec4<f32>>;
// per piece: centroid at the iterate (xyz, w = 1 if valid), then E^-1 (rigid_frame.wgsl)
@group(0) @binding(9) var<storage, read> pieceFrame: array<vec4<f32>>;
// 2 per node (energy.wgsl curveRef)
@group(0) @binding(10) var<storage, read_write> curveRef: array<vec4<f32>>;

const RIGID_FRAME: u32 = 10u;

const NO_PIECE: u32 = 0xFFFFFFFFu;
const WG: u32 = 64u;
var<workgroup> redA: array<vec4<f32>, 64>;
var<workgroup> redB: array<vec4<f32>, 64>;
var<workgroup> redI: array<array<f32, 6>, 64>;
var<workgroup> count: u32;

fn inScene(n: u32) -> bool { return x[n].w < 1.5; }

@compute @workgroup_size(64)
fn spinReduce(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32, @builtin(num_workgroups) ng: vec3<u32>) {
  if (t == 0u) { count = pieceStart[0]; }
  let pieces = workgroupUniformLoad(&count);
  for (var p = wg.x; p < pieces; p += ng.x) {
    let a = pieceStart[1u + p];
    let b = pieceStart[2u + p];
    // mass, centroid and mean velocity
    var mx = vec4<f32>(0.0);
    var mv = vec4<f32>(0.0);
    for (var k = a + t; k < b; k += WG) {
      let n = pieceNodes[k];
      if (!inScene(n)) { continue; }
      let m = nodes[n].mass;
      mx += vec4<f32>(m * x[n].xyz, m);
      mv += vec4<f32>(m * v[n].xyz, 0.0);
    }
    redA[t] = mx;
    redB[t] = mv;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
      if (t < s) {
        redA[t] += redA[t + s];
        redB[t] += redB[t + s];
      }
      workgroupBarrier();
    }
    let M = redA[0].w;
    let c = select(vec3<f32>(0.0), redA[0].xyz / M, M > 0.0);
    let vc = select(vec3<f32>(0.0), redB[0].xyz / M, M > 0.0);
    workgroupBarrier();
    // angular momentum and inertia tensor (xx yy zz xy xz yz) about the centroid
    var L = vec3<f32>(0.0);
    var I: array<f32, 6>;
    for (var q = 0u; q < 6u; q++) { I[q] = 0.0; }
    for (var k = a + t; k < b; k += WG) {
      let n = pieceNodes[k];
      if (!inScene(n)) { continue; }
      let m = nodes[n].mass;
      let r = x[n].xyz - c;
      L += m * cross(r, v[n].xyz - vc);
      let rr = dot(r, r);
      I[0] += m * (rr - r.x * r.x);
      I[1] += m * (rr - r.y * r.y);
      I[2] += m * (rr - r.z * r.z);
      I[3] -= m * r.x * r.y;
      I[4] -= m * r.x * r.z;
      I[5] -= m * r.y * r.z;
    }
    redA[t] = vec4<f32>(L, 0.0);
    redI[t] = I;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
      if (t < s) {
        redA[t] += redA[t + s];
        for (var q = 0u; q < 6u; q++) { redI[t][q] += redI[t + s][q]; }
      }
      workgroupBarrier();
    }
    if (t == 0u) {
      let J = redI[0];
      // regularized as rigid_frame.wgsl: a node, or a straight strand about its own axis, has no
      // inertia there (and no momentum about it)
      let eps = 1e-6 * (J[0] + J[1] + J[2]) / 3.0 + 1e-30;
      let T = mat3x3<f32>(vec3<f32>(J[0] + eps, J[3], J[4]), vec3<f32>(J[3], J[1] + eps, J[5]), vec3<f32>(J[4], J[5], J[2] + eps));
      var th = inverse3(T) * redA[0].xyz * P.hg.x;
      if (!(M > 0.0) || !finite3(th)) { th = vec3<f32>(0.0); }
      pieceStep[2u * p] = vec4<f32>(c, 0.0);
      pieceStep[2u * p + 1u] = vec4<f32>(th, 0.0);
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(128)
fn spinApply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x || !inScene(n)) { return; }
  let p = pieceOf[n];
  if (p == NO_PIECE || p >= pieceStart[0]) { return; }
  let th = pieceStep[2u * p + 1u].xyz;
  let phi = length(th);
  if (phi < 1e-9) { return; }
  let k = th / phi;
  let r = x[n].xyz - pieceStep[2u * p].xyz;
  // k x (k x r) = -r_perp
  dx[n] = vec4<f32>(dx[n].xyz + 2.0 * (1.0 - cos(phi)) * cross(k, cross(k, r)), dx[n].w);
}

@compute @workgroup_size(128)
fn curveSet(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= P.counts.x) { return; }
  let p = pieceOf[n];
  var w = vec4<f32>(0.0);
  var r = vec4<f32>(0.0);
  if (p != NO_PIECE && p < pieceStart[0] && inScene(n)) {
    let fr = pieceFrame[RIGID_FRAME * p];
    if (fr.w > 0.5) {
      w = vec4<f32>(pieceStep[2u * p + 1u].xyz, 1.0);
      r = vec4<f32>(x[n].xyz + dx[n].xyz - fr.xyz, 0.0);
    }
  }
  curveRef[2u * n] = w;
  curveRef[2u * n + 1u] = r;
}
