// Coarse-level operator rows: 50 vec4 per node. Block 0 is the diagonal, blocks 1..24 the
// radius-2 couplings in offset order (dj, di) = (-2..2, -2..2) skipping the center; block b uses
// vec4 2b = (xx, xy, xz, yy) and 2b+1 = (yz, zz, 0, 0). Requires `op: array<vec4<f32>>`.

const ROW_VEC4: u32 = 50u;
const ROW_COUPLINGS: u32 = 24u;
const FINE_ROWS: bool = false;

// Coupling index of an offset within the radius-2 stencil.
fn couplingIndex(off: vec2<i32>) -> u32 {
  let o = u32((off.y + 2) * 5 + (off.x + 2));
  return select(o, o - 1u, o > 12u);
}

fn rowOffset(k: u32) -> vec2<i32> {
  let o = select(k, k + 1u, k >= 12u);
  return vec2<i32>(i32(o % 5u) - 2, i32(o / 5u) - 2);
}

fn symFromVec4(a: vec4<f32>, b: vec4<f32>) -> Sym {
  return Sym(vec3<f32>(a.x, a.w, b.y), vec3<f32>(a.y, a.z, b.x));
}

fn rowDiag(n: u32) -> Sym {
  let r = n * ROW_VEC4;
  return symFromVec4(op[r], op[r + 1u]);
}

fn rowCoupling(n: u32, k: u32) -> Sym {
  let r = n * ROW_VEC4 + 2u * k + 2u;
  return symFromVec4(op[r], op[r + 1u]);
}
