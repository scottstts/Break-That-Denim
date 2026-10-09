// Fine-level operator rows: 18 vec4 per node. Block b (0 = diagonal, 1..8 = couplings
// +u, -u, +v, -v, +u+v, -u-v, -u+v, +u-v) occupies vec4 2b = (xx, xy, xz, yy) and 2b+1 = (yz, zz, *, *).
// The four scalar bending 2-hop couplings live in spare lanes: vec4 1.zw = (+2u, -2u),
// vec4 3.zw = (+2v, -2v). Requires a module-scope `op: array<vec4<f32>>`.

const ROW_VEC4: u32 = 18u;
const ROW_COUPLINGS: u32 = 12u;
// Fine lattice rows: material connectivity comes from the spans (prolongation masking).
const FINE_ROWS: bool = true;

// Coupling index of a unit offset (+u, -u, +v, -v).
fn couplingIndex(off: vec2<i32>) -> u32 {
  if (off.x == 1) { return 0u; }
  if (off.x == -1) { return 1u; }
  if (off.y == 1) { return 2u; }
  return 3u;
}

fn rowOffset(k: u32) -> vec2<i32> {
  var offs = array<vec2<i32>, 12>(
    vec2<i32>(1, 0), vec2<i32>(-1, 0), vec2<i32>(0, 1), vec2<i32>(0, -1),
    vec2<i32>(1, 1), vec2<i32>(-1, -1), vec2<i32>(-1, 1), vec2<i32>(1, -1),
    vec2<i32>(2, 0), vec2<i32>(-2, 0), vec2<i32>(0, 2), vec2<i32>(0, -2));
  return offs[k];
}

fn symFromVec4(a: vec4<f32>, b: vec4<f32>) -> Sym {
  return Sym(vec3<f32>(a.x, a.w, b.y), vec3<f32>(a.y, a.z, b.x));
}

fn rowDiag(n: u32) -> Sym {
  let r = n * ROW_VEC4;
  return symFromVec4(op[r], op[r + 1u]);
}

fn rowCoupling(n: u32, k: u32) -> Sym {
  let r = n * ROW_VEC4;
  if (k < 8u) { return symFromVec4(op[r + 2u * k + 2u], op[r + 2u * k + 3u]); }
  let s = op[r + select(1u, 3u, k >= 10u)];
  return symId(select(s.z, s.w, (k & 1u) == 1u));
}
