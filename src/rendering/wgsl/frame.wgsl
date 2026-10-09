// Per-node cloth frame: normal from connected yarn neighbors only (torn spans never bridge holes).

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> nodes: array<NodeStaticR>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> segFlags: array<vec4<u32>>; // Seg struct viewed as 2 x vec4<u32>; flags at [1].y
@group(0) @binding(4) var<storage, read_write> nrm: array<vec4<f32>>;

fn segOk(owner: u32, f: u32) -> bool {
  let fl = segFlags[2u * (2u * owner + f) + 1u].y;
  return (fl & (SEG_EXISTS_R | SEG_BROKEN_R)) == SEG_EXISTS_R;
}

// Tangent of family f at node n from connected neighbors; zero if isolated in that family.
fn tangent(n: u32, i: i32, j: i32, L: vec4<u32>, f: u32) -> vec3<f32> {
  let du = select(0, 1, f == 0u);
  let dv = select(0, 1, f == 1u);
  let ip = i + du;
  let jp = j + dv;
  let im = i - du;
  let jm = j - dv;
  let stride = select(L.y, 1u, f == 0u);
  var t = vec3<f32>(0.0);
  if (ip < i32(L.y) && jp < i32(L.z) && segOk(n, f)) { t += x[n + stride].xyz - x[n].xyz; }
  if (im >= 0 && jm >= 0 && segOk(n - stride, f)) { t += x[n].xyz - x[n - stride].xyz; }
  return t;
}

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= R.counts.x) { return; }
  let ijl = nodes[n].ijl;
  let i = i32(ijl & 0xFFFu);
  let j = i32((ijl >> 12u) & 0xFFFu);
  let L = R.layers[ijl >> 24u].lattice;
  let tu = tangent(n, i, j, L, 0u);
  let tv = tangent(n, i, j, L, 1u);
  let prev = nrm[n].xyz;
  var nn = cross(tu, tv);
  let l = length(nn);
  if (l > 1e-12) {
    nn = nn / l;
  } else {
    // one family missing: keep the normal perpendicular to the remaining tangent, closest to the previous one
    let t = select(tv, tu, dot(tu, tu) > dot(tv, tv));
    let tl = length(t);
    var p = select(vec3<f32>(0.0, 0.0, 1.0), prev, dot(prev, prev) > 0.5);
    if (tl > 1e-12) {
      let td = t / tl;
      p = p - dot(p, td) * td;
    }
    nn = normalize(p + vec3<f32>(0.0, 0.0, 1e-6));
  }
  nrm[n] = vec4<f32>(nn, 0.0);
}
