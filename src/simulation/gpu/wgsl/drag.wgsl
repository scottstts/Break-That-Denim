// Implicit air drag of a node: energy 1/2 dx^T D dx per substep with D built from the accepted state
// (a fixed quadratic within the substep, exact for Newton). Requires module-scope `P: SimParams`,
// `nodes`, `x` and `segs`. Mirrors ReferenceSolver.airDrag.
//
// Geometry comes from the node's connected spans only: a sheet node (spans in both families) drags
// along its surface normal; a strand node (one family left: a thread or a one-cell strip) drags on
// all motion across the strand; an isolated node drags isotropically. Using lattice neighbours that
// a tear has disconnected gave hanging strips a meaningless normal, and they swung undamped.

fn dragSpanOn(owner: i32, f: u32) -> bool {
  if (owner < 0) { return false; }
  return segActive(segs[2u * u32(owner) + f]);
}

fn dragNodeAt(l: u32, i: i32, j: i32) -> i32 {
  let L = P.layers[l].lattice;
  if (i < 0 || j < 0 || i >= i32(L.y) || j >= i32(L.z)) { return -1; }
  return i32(L.x) + j * i32(L.y) + i;
}

// Chord through the node's connected neighbours along family f (zero without any).
fn dragTangent(n: u32, l: u32, i: i32, j: i32, f: u32) -> vec3<f32> {
  let du = select(0, 1, f == 0u);
  let dv = select(0, 1, f == 1u);
  let prev = dragNodeAt(l, i - du, j - dv);
  let next = dragNodeAt(l, i + du, j + dv);
  var a = x[n].xyz;
  var b = x[n].xyz;
  if (dragSpanOn(prev, f)) { a = x[u32(prev)].xyz; }
  if (next >= 0 && dragSpanOn(i32(n), f)) { b = x[u32(next)].xyz; }
  return b - a;
}

fn airDragBlock(n: u32) -> Sym {
  let st = nodes[n];
  let l = st.ijl >> 24u;
  let mat = P.layers[l];
  if (mat.b.y <= 0.0) { return symZero(); }
  let i = i32(st.ijl & 0xFFFu);
  let j = i32((st.ijl >> 12u) & 0xFFFu);
  // c A / h with A the node's share of cloth area
  let k = mat.b.y * mat.a.w * mat.b.x / P.hg.x;
  let tu = dragTangent(n, l, i, j, 0u);
  let tv = dragTangent(n, l, i, j, 1u);
  let lu = length(tu);
  let lv = length(tv);
  if (lu > 0.0 && lv > 0.0) {
    let c = cross(tu, tv);
    let lc = length(c);
    if (lc > 1e-3 * lu * lv) { return axialBlock(k, 0.0, c / lc); }
  }
  if (lu > 0.0 || lv > 0.0) {
    let t = select(tv / lv, tu / lu, lu >= lv);
    return axialBlock(0.0, k, t);
  }
  return symId(k);
}
