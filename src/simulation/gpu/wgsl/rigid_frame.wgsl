// Rigid modes of detached pieces (solver.ts encodeRigid), once per Newton iteration after assembly.
//
// A detached piece's rigid motion is resisted by its inertia, drag and contact alone, against yarn
// couplings ~1e5 times stiffer: the near-null modes the multigrid hierarchy does not carry (a piece
// keeps few coarse unknowns and the coarsest level is only smoothed). Two PCG iterations resolved 4 %
// of a falling piece's gravity step, and pieces hovered. Each piece's six rigid modes Z (translation,
// rotation about its centroid) are added to the preconditioner as an exact coarse space: z = M r +
// Z E^-1 Z^T r (rigid_apply.wgsl), with E = Z^T A Z, the Newton operator on the rigid modes. Summed as
// is, A's yarn couplings would cancel in f32 to the inertia 1e5 times smaller; instead each term enters
// with what it puts on a rigid motion (assemble.wgsl rigidK): the external blocks (inertia, drag,
// contact, active bounds) fully, the internal terms only on rotation, as their tension's geometric
// stiffness (a spinning piece's spans, stretched by the predictor, resist rotation 1e4 times more
// than its inertia: without them the rotation correction swamped the step).
//
// Per piece p (one workgroup at a time): the mass centroid at the iterate, then E and its inverse.

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> nodes: array<NodeStatic>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> dx: array<vec4<f32>>;
// per detached node: external block and internal rotation stiffness (assemble.wgsl)
@group(0) @binding(4) var<storage, read> rigidK: array<vec4<f32>>;
// [0] piece count, [1 + p] first entry of piece p in pieceNodes, [1 + count] their total
@group(0) @binding(5) var<storage, read> pieceStart: array<u32>;
@group(0) @binding(6) var<storage, read> pieceNodes: array<u32>;
// per piece RIGID_FRAME vec4: centroid (xyz, w = 1 if valid), then E^-1 row-major (36 floats)
@group(0) @binding(7) var<storage, read_write> pieceFrame: array<vec4<f32>>;

const RIGID_FRAME: u32 = 10u;
const WG: u32 = 64u;
var<workgroup> redM: array<vec4<f32>, 64>;
var<workgroup> redE: array<array<f32, 21>, 64>;
var<workgroup> count: u32;

fn pos(n: u32) -> vec3<f32> { return x[n].xyz + dx[n].xyz; }
fn inScene(n: u32) -> bool { return x[n].w < 1.5; }

// Node n's external block and its internal terms' rotation stiffness (assemble.wgsl rigidK).
fn externalBlock(n: u32) -> mat3x3<f32> {
  let a = rigidK[3u * n];
  let b = rigidK[3u * n + 1u];
  return symMat(Sym(vec3<f32>(a.x, a.w, b.y), vec3<f32>(a.y, a.z, b.x)));
}
fn rotationBlock(n: u32) -> mat3x3<f32> {
  let b = rigidK[3u * n + 1u];
  let c = rigidK[3u * n + 2u];
  return symMat(Sym(vec3<f32>(b.z, c.y, c.w), vec3<f32>(b.w, c.x, c.z)));
}

// Upper triangle index of (i, j), i <= j, in a symmetric 6 x 6.
fn tri(i: u32, j: u32) -> u32 { return i * 6u - (i * (i + 1u)) / 2u + j; }

@compute @workgroup_size(64)
fn rigidFrame(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32, @builtin(num_workgroups) ng: vec3<u32>) {
  if (t == 0u) { count = pieceStart[0]; }
  let pieces = workgroupUniformLoad(&count);
  for (var p = wg.x; p < pieces; p += ng.x) {
    let a = pieceStart[1u + p];
    let b = pieceStart[2u + p];
    // mass centroid
    var m = vec4<f32>(0.0);
    for (var k = a + t; k < b; k += WG) {
      let n = pieceNodes[k];
      if (inScene(n)) { m += vec4<f32>(nodes[n].mass * pos(n), nodes[n].mass); }
    }
    redM[t] = m;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
      if (t < s) { redM[t] += redM[t + s]; }
      workgroupBarrier();
    }
    let tot = redM[0];
    workgroupBarrier();
    let c = select(vec3<f32>(0.0), tot.xyz / tot.w, tot.w > 0.0);
    // E = sum Z^T K Z, Z = [I, -[r]x] (a rigid displacement t + w x r)
    var e: array<f32, 21>;
    for (var q = 0u; q < 21u; q++) { e[q] = 0.0; }
    for (var k = a + t; k < b; k += WG) {
      let n = pieceNodes[k];
      if (!inScene(n)) { continue; }
      let K = externalBlock(n);
      let r = pos(n) - c;
      // columns of Z: e_x, e_y, e_z, then -[r]x e_i = e_i x r
      var Z: array<vec3<f32>, 6>;
      Z[0] = vec3<f32>(1.0, 0.0, 0.0);
      Z[1] = vec3<f32>(0.0, 1.0, 0.0);
      Z[2] = vec3<f32>(0.0, 0.0, 1.0);
      Z[3] = cross(Z[0], r);
      Z[4] = cross(Z[1], r);
      Z[5] = cross(Z[2], r);
      for (var i = 0u; i < 6u; i++) {
        let KZi = K * Z[i];
        for (var j = i; j < 6u; j++) { e[tri(i, j)] += dot(Z[j], KZi); }
      }
      let Kr = rotationBlock(n);
      for (var i = 0u; i < 3u; i++) {
        for (var j = i; j < 3u; j++) { e[tri(3u + i, 3u + j)] += Kr[i][j]; }
      }
    }
    redE[t] = e;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s /= 2u) {
      if (t < s) {
        for (var q = 0u; q < 21u; q++) { redE[t][q] += redE[t + s][q]; }
      }
      workgroupBarrier();
    }
    if (t == 0u) {
      let o = RIGID_FRAME * p;
      pieceFrame[o] = vec4<f32>(c, select(0.0, 1.0, tot.w > 0.0));
      // Cholesky of E; the rotation block regularized relative to its scale (a node, or a straight
      // strand, has no stiffness about its own axis: no torque reaches that mode either)
      var A: array<array<f32, 6>, 6>;
      let E = redE[0];
      let st = (E[tri(0u, 0u)] + E[tri(1u, 1u)] + E[tri(2u, 2u)]) / 3.0;
      let sr = (E[tri(3u, 3u)] + E[tri(4u, 4u)] + E[tri(5u, 5u)]) / 3.0;
      let eps = 1e-6 * max(sr, 1e-8 * st) + 1e-30;
      for (var i = 0u; i < 6u; i++) {
        for (var j = 0u; j < 6u; j++) { A[i][j] = E[tri(min(i, j), max(i, j))]; }
        if (i >= 3u) { A[i][i] += eps; }
      }
      var ok = st > 0.0;
      var Lc: array<array<f32, 6>, 6>;
      for (var i = 0u; i < 6u; i++) {
        for (var j = 0u; j <= i; j++) {
          var s = A[i][j];
          for (var k = 0u; k < j; k++) { s -= Lc[i][k] * Lc[j][k]; }
          if (i == j) {
            if (s <= 0.0) { ok = false; s = 1.0; }
            Lc[i][i] = sqrt(s);
          } else {
            Lc[i][j] = s / Lc[j][j];
          }
        }
      }
      // E^-1 column by column (forward and back substitution)
      for (var col = 0u; col < 6u; col++) {
        var y: array<f32, 6>;
        for (var i = 0u; i < 6u; i++) {
          var s = select(0.0, 1.0, i == col);
          for (var k = 0u; k < i; k++) { s -= Lc[i][k] * y[k]; }
          y[i] = s / Lc[i][i];
        }
        var z: array<f32, 6>;
        for (var ii = 0u; ii < 6u; ii++) {
          let i = 5u - ii;
          var s = y[i];
          for (var k = i + 1u; k < 6u; k++) { s -= Lc[k][i] * z[k]; }
          z[i] = s / Lc[i][i];
        }
        for (var i = 0u; i < 6u; i++) {
          let q = 6u * i + col;
          let v = select(0.0, z[i], ok);
          var cell = pieceFrame[o + 1u + q / 4u];
          cell[q % 4u] = v;
          pieceFrame[o + 1u + q / 4u] = cell;
        }
      }
    }
    workgroupBarrier();
  }
}
