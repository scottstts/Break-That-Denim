// Loose thread dynamics (render side). A loose thread carries no load the lattice does not already
// carry, but it moves: each run (threads.wgsl) is a chain of points, one per lone crossing, between
// anchors at its held ends (on the cloth they come out of, thread_prep.wgsl RunGeo). Every frame the
// points are advanced in steps of SIM_H: Verlet under gravity with air drag, then the yarn's length
// (each segment no longer than its rest length: a yarn cannot be stretched beyond its slack, but it
// can bend and fold) and a bending stiffness over a few millimetres are enforced by a few Gauss-Seidel
// sweeps along the chain. A thread slackened by a pull therefore swings, droops and settles; a taut
// bridge stays taut between its anchors.
//
// State persists across rebuilds of the run list in a hash table keyed by yarn and crossing
// (2 c + f + 1; 0 empty): position and stamp (the frame it was last advanced in), previous position.
// A point not advanced in the previous frame starts at the thread's kinematic shape (threadAt) at
// rest. The table is cleared when more than half full (rare: one pop of the threads in a long
// session). threadSimBegin advances the frame counter and the step count from the frame's time;
// threadSim runs one invocation per run and writes the points of the frame to runPts.

@group(0) @binding(0) var<uniform> R: RenderU;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> segw: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> tframe: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> nrm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> runs: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> threadCount: array<u32>;
@group(0) @binding(7) var<storage, read> runGeo: array<vec4<f32>>;
// header [0] last time (f32 bits), [1] frame, [2] steps this frame, [3] entries, [4] clear the table
// this frame; keys from
// SIM_KEYS, values (2 vec4 per entry) from SIM_VALUES
@group(0) @binding(8) var<storage, read_write> simState: array<atomic<u32>>;
// per run point: position (w = 1 valid)
@group(0) @binding(9) var<storage, read_write> runPts: array<vec4<f32>>;
@group(0) @binding(10) var<uniform> F: FrameU;

override MAX_RUNS: u32 = 65536u;
const SIM_H: f32 = 1.0 / 240.0;
const SIM_MAX_STEPS: u32 = 10u;
const SIM_KEYS: u32 = 8u;
const SIM_VALUES: u32 = SIM_KEYS + SIM_CAP;
// air drag time constant of a loose yarn (s): a few-millimetre thread settles within about a second
const SIM_DRAG: f32 = 0.12;
const SIM_SWEEPS: u32 = 6u;

@compute @workgroup_size(1)
fn threadSimBegin() {
  let t = F.params.y;
  let last = bitcast<f32>(atomicLoad(&simState[0]));
  let dt = clamp(t - last, 0.0, f32(SIM_MAX_STEPS) * SIM_H);
  atomicStore(&simState[0], bitcast<u32>(t));
  atomicStore(&simState[2], u32(round(dt / SIM_H)));
  let full = atomicLoad(&simState[3]) > SIM_CAP / 2u;
  // a rare full reset: the threads restart from their kinematic shape (the frame skips ahead so no
  // stamp of an old entry is taken for a recent one)
  atomicStore(&simState[1], atomicLoad(&simState[1]) + select(1u, 3u, full));
  atomicStore(&simState[4], select(0u, 1u, full));
  if (full) { atomicStore(&simState[3], 0u); }
}

@compute @workgroup_size(256)
fn threadSimClear(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (atomicLoad(&simState[4]) == 0u) { return; }
  if (gid.x < SIM_CAP) { atomicStore(&simState[SIM_KEYS + gid.x], 0u); }
}

// Slot of key, inserting it if absent (linear probing).
fn slotOf(key: u32) -> u32 {
  var h = hash11(key) & (SIM_CAP - 1u);
  for (var probe = 0u; probe < 64u; probe++) {
    let k = atomicLoad(&simState[SIM_KEYS + h]);
    if (k == key) { return h; }
    if (k == 0u) {
      let r = atomicCompareExchangeWeak(&simState[SIM_KEYS + h], 0u, key);
      if (r.exchanged) {
        atomicAdd(&simState[3], 1u);
        return h;
      }
      if (r.old_value == key) { return h; }
    }
    h = (h + 1u) & (SIM_CAP - 1u);
  }
  return 0xFFFFFFFFu;
}

fn loadV(slot: u32, q: u32) -> vec4<f32> {
  let b = SIM_VALUES + 8u * slot + 4u * q;
  return vec4<f32>(bitcast<f32>(atomicLoad(&simState[b])), bitcast<f32>(atomicLoad(&simState[b + 1u])), bitcast<f32>(atomicLoad(&simState[b + 2u])), bitcast<f32>(atomicLoad(&simState[b + 3u])));
}
fn storeV(slot: u32, q: u32, v: vec4<f32>) {
  let b = SIM_VALUES + 8u * slot + 4u * q;
  atomicStore(&simState[b], bitcast<u32>(v.x));
  atomicStore(&simState[b + 1u], bitcast<u32>(v.y));
  atomicStore(&simState[b + 2u], bitcast<u32>(v.z));
  atomicStore(&simState[b + 3u], bitcast<u32>(v.w));
}

const MAX_CHAIN: u32 = 72u;

@compute @workgroup_size(64)
fn threadSim(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= min(threadCount[0], MAX_RUNS)) { return; }
  let ri = runInfo(runs[r]);
  let h = runGeo[RUN_GEO * r];
  var g: RunGeo;
  g.arc = h.x;
  g.chain = h.y > 0.5;
  g.pA = runGeo[RUN_GEO * r + 1u].xyz;
  g.pB = runGeo[RUN_GEO * r + 2u].xyz;
  g.tA = runGeo[RUN_GEO * r + 3u].xyz;
  g.tB = runGeo[RUN_GEO * r + 4u].xyz;
  let cA = runGeo[RUN_GEO * r + 5u].xyz;
  let cB = runGeo[RUN_GEO * r + 6u].xyz;
  let LR = R.layers[ri.layer];
  let F0 = LR.field;
  let n = min(u32(ri.n), MAX_CHAIN - 2u);
  let pitchAl = select(LR.pitch.y, LR.pitch.x, ri.fam == 0u);
  let seedY = hash11((ri.layer * 7919u + ri.yarn) * 2u + ri.fam + 0x51ed27u);
  let sl = hashf(seedY + 1u);
  let restSeg = pitchAl * (1.0 + 0.004 + 0.04 * sl * sl * sl);
  let frame = atomicLoad(&simState[1]);
  let steps = atomicLoad(&simState[2]);
  // the chain: [0] anchor A, [1..n] the lone crossings, [n + 1] anchor B
  var p: array<vec3<f32>, MAX_CHAIN>;
  var q: array<vec3<f32>, MAX_CHAIN>;
  var slots: array<u32, MAX_CHAIN>;
  p[0] = g.pA;
  p[n + 1u] = g.pB;
  for (var i = 1u; i <= n; i++) {
    let a = u32(ri.pos) + i - 1u;
    let k = select(ri.yarn, a, ri.fam == 0u);
    let m = select(a, ri.yarn, ri.fam == 0u);
    let c = F0.x + m * F0.y + k;
    let s = slotOf(2u * c + ri.fam + 1u);
    slots[i] = s;
    var cur = vec4<f32>(0.0);
    var prev = vec4<f32>(0.0);
    if (s != 0xFFFFFFFFu) {
      cur = loadV(s, 0u);
      prev = loadV(s, 1u);
    }
    // the stamp is kept as an integer (a small integer bitcast to f32 is a denormal, which the GPU
    // may flush to zero)
    let stamp = select(0u, atomicLoad(&simState[SIM_VALUES + 8u * s + 3u]), s != 0xFFFFFFFFu);
    if (s == 0xFFFFFFFFu || (stamp != frame && stamp + 1u != frame)) {
      // new: the kinematic shape, at rest
      let tp = threadAt(ri, g, cA, cB, f32(a));
      cur = vec4<f32>(tp.p, 0.0);
      prev = cur;
    }
    p[i] = cur.xyz;
    q[i] = prev.xyz;
  }
  let freeA = !ri.heldA;
  let freeB = !ri.heldB;
  let gvec = vec3<f32>(0.0, 0.0, -9.81);
  let damp = exp(-SIM_H / SIM_DRAG);
  // bending: neighbours pull a point toward their midpoint, with this weight per sweep
  let kb = 0.35;
  for (var st = 0u; st < steps; st++) {
    for (var i = 1u; i <= n; i++) {
      let v = (p[i] - q[i]) * damp;
      q[i] = p[i];
      p[i] = p[i] + v + gvec * (SIM_H * SIM_H);
    }
    for (var sw = 0u; sw < SIM_SWEEPS; sw++) {
      // bending
      for (var i = 1u; i <= n; i++) {
        let hasL = i > 1u || !freeA;
        let hasR = i < n || !freeB;
        if (hasL && hasR) { p[i] += kb * (0.5 * (p[i - 1u] + p[i + 1u]) - p[i]); }
      }
      // length, both ways along the chain (unilateral: segments only shortened)
      let first = select(0u, 1u, freeA);
      let last = select(n + 1u, n, freeB);
      for (var i = first; i < last; i++) {
        let d = p[i + 1u] - p[i];
        let l = length(d);
        let rl = select(restSeg, 0.5 * restSeg, i == 0u || i == n);
        if (l <= rl || l < 1e-12) { continue; }
        let corr = d * ((l - rl) / l);
        let mA = select(1.0, 0.0, i == 0u && !freeA);
        let mB = select(1.0, 0.0, i + 1u == n + 1u && !freeB);
        let wsum = mA + mB;
        if (wsum <= 0.0) { continue; }
        p[i] += corr * (mA / wsum);
        p[i + 1u] -= corr * (mB / wsum);
      }
      for (var j = 0u; j < last - first; j++) {
        let i = last - 1u - j;
        let d = p[i + 1u] - p[i];
        let l = length(d);
        let rl = select(restSeg, 0.5 * restSeg, i == 0u || i == n);
        if (l <= rl || l < 1e-12) { continue; }
        let corr = d * ((l - rl) / l);
        let mA = select(1.0, 0.0, i == 0u && !freeA);
        let mB = select(1.0, 0.0, i + 1u == n + 1u && !freeB);
        let wsum = mA + mB;
        if (wsum <= 0.0) { continue; }
        p[i] += corr * (mA / wsum);
        p[i + 1u] -= corr * (mB / wsum);
      }
    }
  }
  // a free end: its anchor slot extrapolates the chain by half a crossing
  if (freeA) { p[0] = p[1] + 0.5 * (p[1] - p[2]); }
  if (freeB) { p[n + 1u] = p[n] + 0.5 * (p[n] - p[n - 1u]); }
  for (var i = 1u; i <= n; i++) {
    if (slots[i] != 0xFFFFFFFFu) {
      storeV(slots[i], 0u, vec4<f32>(p[i], 0.0));
      storeV(slots[i], 1u, vec4<f32>(q[i], 0.0));
      atomicStore(&simState[SIM_VALUES + 8u * slots[i] + 3u], frame);
    }
  }
  for (var i = 0u; i <= n + 1u; i++) { runPts[ri.base + i] = vec4<f32>(p[i], 1.0); }
}
