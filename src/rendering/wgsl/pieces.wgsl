// Material connectivity of the cloth around each lattice node (rebuilt with the loose threads, when
// the yarn field changed).
//
// The mechanical lattice carries forces at bundle resolution; where the cloth is torn, which material
// moves with which node is decided at yarn resolution. Surface material (a crossing with a yarn that
// is neither separated nor drawn as a loose thread, threads.wgsl) forms a graph: neighbouring
// crossings are joined along a pick where the pick is surface material at both, along a warp
// likewise. For a node near damage, the connection of a crossing to it is the ratio of their rest
// (Manhattan) distance to the shortest path between them through that graph: 1 where the material
// between them is whole, falling toward 0 the further a crack forces the path around, 0 where none
// exists. A crossing across a long crack is therefore cut off from the node (it moves with the
// material on its own side), one near the crack's tip stays partly attached, so an opening tapers to
// nothing at the tip, as an elastic crack does (the visibility / diffraction criterion of meshfree
// crack methods). The surface (surface.wgsl vsFine) places every crossing from the nodes it is
// connected to.
//
// Passes: piecesClear, cellDirty (per cell: does it lack any surface material), nodeAlloc (per node: a
// slot when its domain has a dirty cell), piecesArgs, kappaSolve (one workgroup per slot: shortest
// paths over the domain, then the connection of the crossings of the node's cells; a node whose
// crossings all stay connected gives its slot up as clean), edgeListMain (crossings of surface
// material with an open side: the free edges, at yarn resolution) and edgeDrawArgs.

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> field: array<u32>;
@group(0) @binding(2) var<storage, read> threadBits: array<u32>;
@group(0) @binding(3) var<storage, read> nodes: array<NodeStaticR>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
// slot count, slot table per node, node of each slot, connection fields (render_common.wgsl KAPPA_*)
@group(0) @binding(5) var<storage, read_write> kappa: array<atomic<u32>>;
// per cell: lacks surface material somewhere
@group(0) @binding(6) var<storage, read_write> cellDirty: array<u32>;
// [0..2] indirect dispatch of kappaSolve, [3] edge crossings, [4..7] indirect draw of the edge fibres,
// [8..10] indirect dispatch of their roots (edge_roots.wgsl, 64 per group), [12..15] indirect draw of
// the aggregate fringe
@group(0) @binding(7) var<storage, read_write> piecesArgs: array<atomic<u32>>;
// edge crossings (crossing | open sides << 28: bit 0 -u, 1 +u, 2 -v, 3 +v), for the edge fibres; their
// count at [MAX_EDGES]
@group(0) @binding(8) var<storage, read_write> edgeList: array<u32>;

fn retiredP(n: u32) -> bool { return x[n].w > 1.5; }

fn nodeIJ(n: u32) -> vec2<u32> { return vec2<u32>(nodes[n].ijl & 0xFFFu, (nodes[n].ijl >> 12u) & 0xFFFu); }

// Slots persist across rebuilds (cleared only by a reset): a node keeps its slot, and a slot whose
// node's domain graph is unchanged keeps its connection field (kappaSolve compares a hash). During a
// tear only the domains the new separations touch are solved again.
fn hashBase() -> u32 { return KAPPA_TABLE + R.counts.x + KAPPA_SLOTS + KAPPA_SLOTS * KAPPA_WORDS; }
// per slot: hash of the domain graph of its last solve (0: never), bit 31 set when it came out clean
fn slotHash(s: u32) -> u32 { return hashBase() + s; }
// per node: its slot + 1 (0: none)
fn nodeSlot(n: u32) -> u32 { return hashBase() + KAPPA_SLOTS + n; }

@compute @workgroup_size(128)
fn piecesClear(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x == 0u) { atomicStore(&piecesArgs[3], 0u); }
}

// One invocation per cell (global cell index over all layers).
@compute @workgroup_size(128)
fn cellDirtyMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell = gid.x;
  if (cell >= R.counts.z) { return; }
  var l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (cell >= R.layers[q].refine.z) { l = q; }
  }
  let LR = R.layers[l];
  let local = cell - LR.refine.z;
  let ci = local % (LR.lattice.y - 1u);
  let cj = local / (LR.lattice.y - 1u);
  let g = cellGrid(LR, ci, cj);
  var dirty = 0u;
  for (var r = 0u; r < g.w && dirty == 0u; r++) {
    for (var q = 0u; q < g.z; q++) {
      let c = fieldIndexR(LR, g.x + q, g.y + r);
      if (!surfaceYarn(c, 0u) || !surfaceYarn(c, 1u)) {
        dirty = 1u;
        break;
      }
    }
  }
  cellDirty[cell] = dirty;
}

fn debrisP(c: u32) -> bool { return ((threadBits[debrisWord(c, fieldTotalOf(R.layers[R.counts.y - 1u]))] >> (c & 31u)) & 1u) == 1u; }

fn surfaceYarn(c: u32, f: u32) -> bool {
  return yarnCode(field[c], f) < PULLED_R && ((threadBits[c >> 4u] >> (2u * (c & 15u) + f)) & 1u) == 0u;
}

@compute @workgroup_size(128)
fn nodeAlloc(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= R.counts.x) { return; }
  if (retiredP(n)) {
    atomicStore(&kappa[KAPPA_TABLE + n], KAPPA_ZERO);
    return;
  }
  let l = nodes[n].ijl >> 24u;
  let LR = R.layers[l];
  let L = LR.lattice;
  let ij = nodeIJ(n);
  // domain: the 2 x 2 cells around the node and one cell beyond
  var dirty = false;
  for (var dj = -2; dj <= 1 && !dirty; dj++) {
    for (var di = -2; di <= 1; di++) {
      let ci = i32(ij.x) + di;
      let cj = i32(ij.y) + dj;
      if (ci < 0 || cj < 0 || ci >= i32(L.y) - 1 || cj >= i32(L.z) - 1) { continue; }
      if (cellDirty[LR.refine.z + u32(cj) * (L.y - 1u) + u32(ci)] != 0u) {
        dirty = true;
        break;
      }
    }
  }
  if (!dirty) {
    atomicStore(&kappa[KAPPA_TABLE + n], KAPPA_NONE);
    return;
  }
  var s = atomicLoad(&kappa[nodeSlot(n)]);
  if (s == 0u) {
    let a = atomicAdd(&kappa[0], 1u);
    if (a >= KAPPA_SLOTS) {
      atomicStore(&kappa[KAPPA_TABLE + n], KAPPA_NONE);
      return;
    }
    s = a + 1u;
    atomicStore(&kappa[nodeSlot(n)], s);
    atomicStore(&kappa[KAPPA_TABLE + R.counts.x + a], n);
  }
  // kappaSolve keeps this, or gives the slot's table entry up when the node's cells are all connected
  atomicStore(&kappa[KAPPA_TABLE + n], s - 1u);
}

@compute @workgroup_size(1)
fn piecesArgsMain() {
  atomicStore(&piecesArgs[0], min(atomicLoad(&kappa[0]), KAPPA_SLOTS));
  atomicStore(&piecesArgs[1], 1u);
  atomicStore(&piecesArgs[2], 1u);
}

// After kappaSolve: no further solve until the next rebuild sets the dispatch again.
@compute @workgroup_size(1)
fn piecesArgsReset() { atomicStore(&piecesArgs[0], 0u); }

// One invocation per crossing (global over the layers): a crossing of surface material is on a free
// edge on each side where the material does not continue (no neighbour joined along that yarn).
@compute @workgroup_size(128)
fn edgeListMain(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  var l = 0u;
  for (var q = 1u; q < R.counts.y; q++) {
    if (c >= R.layers[q].field.x) { l = q; }
  }
  let F = R.layers[l].field;
  if (c >= F.x + F.y * F.z) { return; }
  let pk = surfaceYarn(c, 0u);
  let wp = surfaceYarn(c, 1u);
  if (!pk && !wp) { return; }
  let k = (c - F.x) % F.y;
  let m = (c - F.x) / F.y;
  // a side is open where the yarn running across it ends here (or is absent) and the material there
  // is missing: picks run along u, warps along v
  var sides = 0u;
  let deb = debrisP(c);
  let hasL = k > 0u && pk && surfaceYarn(c - 1u, 0u);
  let hasR = k + 1u < F.y && pk && surfaceYarn(c + 1u, 0u);
  let hasD = m > 0u && wp && surfaceYarn(c - F.y, 1u);
  let hasU = m + 1u < F.z && wp && surfaceYarn(c + F.y, 1u);
  let matL = k > 0u && (surfaceYarn(c - 1u, 0u) || surfaceYarn(c - 1u, 1u)) && debrisP(c - 1u) == deb;
  let matR = k + 1u < F.y && (surfaceYarn(c + 1u, 0u) || surfaceYarn(c + 1u, 1u)) && debrisP(c + 1u) == deb;
  let matD = m > 0u && (surfaceYarn(c - F.y, 0u) || surfaceYarn(c - F.y, 1u)) && debrisP(c - F.y) == deb;
  let matU = m + 1u < F.z && (surfaceYarn(c + F.y, 0u) || surfaceYarn(c + F.y, 1u)) && debrisP(c + F.y) == deb;
  if (!hasL && !matL && k > 0u) { sides |= 1u; }
  if (!hasR && !matR && k + 1u < F.y) { sides |= 2u; }
  if (!hasD && !matD && m > 0u) { sides |= 4u; }
  if (!hasU && !matU && m + 1u < F.z) { sides |= 8u; }
  if (sides == 0u) { return; }
  let e = atomicAdd(&piecesArgs[3], 1u);
  if (e < MAX_EDGES) { edgeList[e] = c | (sides << 28u); }
}

@compute @workgroup_size(1)
fn edgeDrawArgs() {
  let n = min(atomicLoad(&piecesArgs[3]), MAX_EDGES);
  atomicStore(&piecesArgs[4], FIBER_VERTS);
  atomicStore(&piecesArgs[5], n * FIBERS_PER_EDGE);
  atomicStore(&piecesArgs[6], 0u);
  atomicStore(&piecesArgs[7], 0u);
  atomicStore(&piecesArgs[8], (n + 63u) / 64u);
  atomicStore(&piecesArgs[9], 1u);
  atomicStore(&piecesArgs[10], 1u);
  edgeList[MAX_EDGES] = n;
}

// Shortest material paths over the node's domain (DW x DH crossings), then its connection field.
var<workgroup> dist: array<f32, KAPPA_DOMAIN>;
// bit 0: surface material, bit 1: joined to the next crossing along u (pick), bit 2: along v (warp)
var<workgroup> graph: array<u32, KAPPA_DOMAIN>;
var<workgroup> changed: u32;
var<workgroup> unclean: u32;
var<workgroup> graphHash: atomic<u32>;
var<workgroup> decide: u32;
// nearest material of the node's own piece within its half cell (rest distance, um)
var<workgroup> nearest: atomic<u32>;

@compute @workgroup_size(256)
fn kappaSolve(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let slot = wg.x;
  let n = atomicLoad(&kappa[KAPPA_TABLE + R.counts.x + slot]);
  // only nodes whose domain has a cell without surface material this rebuild (nodeAlloc)
  if (t == 0u) { decide = select(0u, 1u, atomicLoad(&kappa[KAPPA_TABLE + n]) != slot); }
  if (workgroupUniformLoad(&decide) != 0u) { return; }
  let l = nodes[n].ijl >> 24u;
  let LR = R.layers[l];
  let ij = nodeIJ(n);
  let bu = f32(bundleOf(LR, 1u));
  let bv = f32(bundleOf(LR, 0u));
  let Un = f32(ij.x) * bu + 0.5 * (bu - 1.0);
  let Vn = f32(ij.y) * bv + 0.5 * (bv - 1.0);
  let dk0 = i32(ceil(Un - 2.0 * bu - 1.0));
  let dm0 = i32(ceil(Vn - 2.0 * bv - 1.0));
  let K = i32(LR.field.y);
  let M = i32(LR.field.z);
  let pu = LR.pitch.x;
  let pv = LR.pitch.y;
  for (var e = t; e < KAPPA_DOMAIN; e += 256u) {
    let k = dk0 + i32(e % KAPPA_DW);
    let m = dm0 + i32(e / KAPPA_DW);
    var gb = 0u;
    var d = 1e30;
    if (k >= 0 && m >= 0 && k < K && m < M) {
      let c = fieldIndexR(LR, u32(k), u32(m));
      let pk = surfaceYarn(c, 0u);
      let wp = surfaceYarn(c, 1u);
      // material of a falling piece and of the cloth are never joined
      let deb = debrisP(c);
      if (pk || wp) { gb |= 1u; }
      if (pk && k + 1 < K && surfaceYarn(c + 1u, 0u) && debrisP(c + 1u) == deb) { gb |= 2u; }
      if (wp && m + 1 < M && surfaceYarn(c + LR.field.y, 1u) && debrisP(c + LR.field.y) == deb) { gb |= 4u; }
      // candidate seeds: material of the node's own piece within its half cell (bit 3)
      if ((gb & 1u) != 0u && deb == (x[n].w > 0.5) && abs(f32(k) - Un) <= 0.5 * bu && abs(f32(m) - Vn) <= 0.5 * bv) {
        gb |= 8u;
        d = abs(f32(k) - Un) * pu + abs(f32(m) - Vn) * pv;
      }
    }
    graph[e] = gb;
    dist[e] = d;
    if (gb != 0u) { atomicXor(&graphHash, hash11(e * 16u + gb + 0x9e3779b9u)); }
  }
  if (t == 0u) { atomicStore(&nearest, 0xFFFFFFFFu); }
  workgroupBarrier();
  // the same graph as at this slot's last solve: its connection field still holds
  let h = ((atomicLoad(&graphHash) ^ select(0u, 0x55555555u, x[n].w > 0.5)) & 0x7FFFFFFFu) | 1u;
  if (t == 0u) {
    let prev = atomicLoad(&kappa[slotHash(slot)]);
    decide = select(0u, 1u, (prev & 0x7FFFFFFFu) == h);
    if (decide != 0u && (prev & 0x80000000u) != 0u) { atomicStore(&kappa[KAPPA_TABLE + n], KAPPA_NONE); }
  }
  if (workgroupUniformLoad(&decide) != 0u) { return; }
  for (var e = t; e < KAPPA_DOMAIN; e += 256u) {
    if ((graph[e] & 8u) != 0u) { atomicMin(&nearest, u32(dist[e] * 1e6)); }
  }
  workgroupBarrier();
  // seeds: the nearest material cluster only (within a pitch of the nearest), so that a node standing
  // in a hole holds the cloth beside it, and a crack passing near a node still cuts off its far side
  let d0 = f32(atomicLoad(&nearest)) * 1e-6 + max(pu, pv);
  for (var e = t; e < KAPPA_DOMAIN; e += 256u) {
    if ((graph[e] & 8u) == 0u || dist[e] > d0) { dist[e] = 1e30; }
  }
  workgroupBarrier();
  // Bellman-Ford relaxation in place: distances only decrease toward the unique shortest-path fixed
  // point, so the order of the racing updates does not change the result
  for (var it = 0u; it < 4u * (KAPPA_DW + KAPPA_DH); it++) {
    if (t == 0u) { changed = 0u; }
    workgroupBarrier();
    var any = false;
    for (var e = t; e < KAPPA_DOMAIN; e += 256u) {
      let gb = graph[e];
      if ((gb & 1u) == 0u) { continue; }
      let ek = e % KAPPA_DW;
      var d = dist[e];
      if (ek > 0u && (graph[e - 1u] & 2u) != 0u) { d = min(d, dist[e - 1u] + pu); }
      if (ek + 1u < KAPPA_DW && (gb & 2u) != 0u) { d = min(d, dist[e + 1u] + pu); }
      if (e >= KAPPA_DW && (graph[e - KAPPA_DW] & 4u) != 0u) { d = min(d, dist[e - KAPPA_DW] + pv); }
      if (e + KAPPA_DW < KAPPA_DOMAIN && (gb & 4u) != 0u) { d = min(d, dist[e + KAPPA_DW] + pv); }
      if (d < dist[e]) {
        dist[e] = d;
        any = true;
      }
    }
    // every writer stores the same value
    if (any) { changed = 1u; }
    workgroupBarrier();
    if (workgroupUniformLoad(&changed) == 0u) { break; }
  }
  // connection field over the node's own cells and one crossing beyond (RW x RH from rk0, rm0)
  if (t == 0u) { unclean = 0u; }
  workgroupBarrier();
  let rk0 = i32(ceil(Un - bu - 1.0));
  let rm0 = i32(ceil(Vn - bv - 1.0));
  let base = KAPPA_TABLE + R.counts.x + KAPPA_SLOTS + slot * KAPPA_WORDS;
  for (var w = t; w < KAPPA_WORDS; w += 256u) {
    var word = 0u;
    for (var b = 0u; b < 4u; b++) {
      let e = 4u * w + b;
      if (e >= KAPPA_RW * KAPPA_RH) { break; }
      let k = rk0 + i32(e % KAPPA_RW);
      let m = rm0 + i32(e / KAPPA_RW);
      let de = (k - dk0) + (m - dm0) * i32(KAPPA_DW);
      var kap = 1.0;
      if (de >= 0 && de < i32(KAPPA_DOMAIN) && (graph[de] & 1u) != 0u) {
        let dm = abs(f32(k) - Un) * pu + abs(f32(m) - Vn) * pv;
        let ratio = (dm + pu) / (dist[de] + pu);
        kap = smoothstep(KAPPA_LO, KAPPA_HI, ratio);
      }
      if (kap < 0.995) { unclean = 1u; }
      word |= u32(round(kap * 255.0)) << (8u * b);
    }
    atomicStore(&kappa[base + w], word);
  }
  workgroupBarrier();
  if (t == 0u) {
    if (unclean == 0u) { atomicStore(&kappa[KAPPA_TABLE + n], KAPPA_NONE); }
    atomicStore(&kappa[slotHash(slot)], h | select(0u, 0x80000000u, unclean == 0u));
  }
}
