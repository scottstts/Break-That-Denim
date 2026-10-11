/**
 * Interaction and contact contracts on the production kernels (headless: Node + Dawn), on the settled
 * drape. Picking sees only visible material, a grab imposes no displacement, held material survives a
 * pending connectivity analysis, material torn out from under a pinch's pads slips out of it (a claw's
 * hold does not slip), contact bounds are relative and swept, a body's contact is swept over the
 * substep, and a claw's point parts the yarns under it.
 *   npm run test:gpu
 * Exits non-zero on failure.
 */
import { ComputeModule, createBufferWith, readBuffer, UNIFORM } from '../../src/gpu/util';
import { cameraMatrices, DEFAULT_CAMERA } from '../../src/rendering/renderer';
import { normalize, scale, sub, transformPoint, type Vec3 } from '../../src/rendering/math';
import { BodyContact, BODY_CONTACT, CLAW_PIERCE } from '../../src/simulation/gpu/body';
import { Interaction } from '../../src/simulation/gpu/interaction';
import { Topology } from '../../src/simulation/gpu/topology';
import { Hand } from '../../src/simulation/hand';
import { GRIP, hookPatch } from '../../src/simulation/grip';
import { CLAW_HOOK } from '../../src/stage/claws';
import type { GpuSolver } from '../../src/simulation/gpu/solver';
import type { CompiledScene } from '../../src/simulation/scene';
import common from '../../src/simulation/gpu/wgsl/common.wgsl?raw';
import step from '../../src/simulation/gpu/wgsl/step.wgsl?raw';
import { drain, gpuDevice } from '../../tools/gpu/env';
import { settledSolver } from '../../tools/gpu/scripts/render';

const results: Array<{ name: string; ok: boolean }> = [];
function check(name: string, ok: boolean, detail: string): void {
  results.push({ name, ok });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
}

const W = 1600;
const H = 1000;
const vec = (a: Float32Array, n: number): Vec3 => [a[4 * n], a[4 * n + 1], a[4 * n + 2]];
const dist = (a: Vec3, b: Vec3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** A state with every node retired (out of the scene) except those given positions. */
function sparse(initial: Float32Array, placed: Map<number, Vec3>): Float32Array {
  const xs = initial.slice();
  for (let n = 0; n < xs.length / 4; n++) xs[4 * n + 3] = 2;
  for (const [n, p] of placed) xs.set([...p, 0], 4 * n);
  return xs;
}

async function picking(device: GPUDevice, solver: GpuSolver, interaction: Interaction, initial: Float32Array, nx: number): Promise<void> {
  const m = cameraMatrices(DEFAULT_CAMERA, W / H);
  const forward = normalize(sub(DEFAULT_CAMERA.target, m.eye));
  const pixel = (p: Vec3) => {
    const c = transformPoint(m.viewProj, p);
    return [(W / 2) * (1 + c[0] / c[3]), (H / 2) * (1 - c[1] / c[3])];
  };
  // an intact face, and a crossing 4 mm behind its interior along the line of sight (a fold's far layer)
  const f = 20 * nx + 20;
  const corners = [f, f + 1, f + nx, f + nx + 1];
  const back = 50 * nx + 50;
  const placed = new Map<number, Vec3>(corners.map((c, q) => [c, [((q % 2) - 0.5) * 0.00342, (Math.floor(q / 2) - 0.5) * 0.00348, 0]]));
  const behind = scale(forward, 0.004);
  placed.set(back, behind);
  solver.setState(sparse(initial, placed));
  const [bx, by] = pixel(behind);
  const hit = await interaction.pickAt(m.viewProj, bx, by, W, H, 16);
  check('pick: a crossing behind a face is not picked through it', !!hit && corners.includes(hit.node) && hit.kind === 'surface', `picked ${hit?.node} (${hit?.kind}), front corners ${corners.join(',')}`);
  // the same crossing beside the face (nothing in front of it) is picked by proximity
  placed.set(back, [0.006, 0, -0.004]);
  solver.setState(sparse(initial, placed));
  const [sx, sy] = pixel([0.006, 0, -0.004]);
  const side = await interaction.pickAt(m.viewProj, sx, sy, W, H, 16);
  check('pick: an unoccluded crossing over an opening is picked by proximity', side?.node === back && side.kind === 'proximity', `picked ${side?.node} (${side?.kind})`);

  // an end of a strand: its incoming span survives, both of its own are broken
  const n = Math.floor(solver.N / 2);
  solver.setState(sparse(initial, new Map<number, Vec3>([[n, [0, 0, 0]], [n - 1, [-0.00342, 0, 0]]])));
  const segs = new Uint32Array(await readBuffer(device, solver.segs));
  const flags = segs.slice();
  flags[16 * n + 5] |= 2;
  flags[16 * n + 13] |= 2;
  device.queue.writeBuffer(solver.segs, 0, flags);
  const [ex, ey] = pixel([0, 0, 0]);
  const end = await interaction.pickAt(m.viewProj, ex, ey, W, H, 2);
  check('pick: a strand end held only by its incoming span is grabbable', end?.node === n, `picked ${end?.node}, end ${n}`);
  device.queue.writeBuffer(solver.segs, 0, segs);
  solver.setState(initial);
}

async function acquisition(device: GPUDevice, solver: GpuSolver, interaction: Interaction, initial: Float32Array): Promise<void> {
  const m = cameraMatrices(DEFAULT_CAMERA, W / H);
  solver.setState(initial);
  const hit = await interaction.pickAt(m.viewProj, W / 2, H / 2, W, H, 16);
  if (!hit) {
    check('grab: a stationary grab moves nothing', false, 'no pick at the centre');
    return;
  }
  // the host's view of the picked material is 5 mm stale; the target asks for no motion from it
  const stale: Vec3 = [hit.position[0] + 0.005, hit.position[1], hit.position[2]];
  interaction.attach(hit.node, stale, 0, hit.window);
  interaction.setTarget(stale);
  interaction.setPinch(1);
  const p0 = vec(initial, hit.node);
  for (let k = 0; k < 24; k++) {
    const enc = device.createCommandEncoder();
    solver.encodeSubstep(enc);
    device.queue.submit([enc.finish()]);
  }
  await drain(device);
  const x = new Float32Array(await readBuffer(device, solver.x));
  interaction.release(0);
  const moved = dist(vec(x, hit.node), p0);
  check('grab: a stationary grab moves nothing, however stale the host anchor', moved < 5e-5, `picked node moved ${(moved * 1e3).toFixed(4)} mm in 0.2 s (anchor 5 mm off)`);
  solver.setState(initial);
}

async function topologyRace(device: GPUDevice, solver: GpuSolver, interaction: Interaction, initial: Float32Array, nx: number): Promise<void> {
  const scene = solver.scene;
  const topology = new Topology(device, solver, scene, () => interaction.coreNodes);
  const n = Math.floor(solver.N / 2);
  const segs0 = await readBuffer(device, solver.segs);
  // a 3 x 3 piece around the pinch cut free on all four sides (all of it inside the pinch)
  const flags = scene.segFlags.slice();
  const ci = n % nx;
  const cj = Math.floor(n / nx);
  const r = 1;
  for (let j = cj - r; j <= cj + r; j++) {
    flags[2 * (j * nx + ci - r - 1)] |= 2;
    flags[2 * (j * nx + ci + r)] |= 2;
  }
  for (let i = ci - r; i <= ci + r; i++) {
    flags[2 * ((cj - r - 1) * nx + i) + 1] |= 2;
    flags[2 * ((cj + r) * nx + i) + 1] |= 2;
  }
  solver.setSegments({ ...scene, segFlags: flags }, scene.tails);
  // an analysis starts before the finger lands on the piece, and applies after
  const pending = topology.request();
  interaction.attach(n, vec(initial, n));
  await pending;
  const x = new Float32Array(await readBuffer(device, solver.x));
  const lost = interaction.coreNodes.filter((k) => x[4 * k + 3] > 0.5).length;
  check('topology: a piece grabbed while an analysis is pending stays held cloth', interaction.active && lost === 0, `${lost} of ${interaction.coreNodes.length} held core nodes retired`);
  // pinched and held: the piece is crushed toward the pinch point (nothing around it to buckle into)
  const hand = new Hand(vec(initial, n), [0, 0, 1]);
  const step = (k: number, iterations = 0) => {
    for (let q = 0; q < k; q++) {
      const enc = device.createCommandEncoder();
      solver.encodeSubstep(enc, iterations);
      device.queue.submit([enc.finish()]);
    }
  };
  for (let k = 0; k < 36; k++) {
    hand.step(1 / 120, [0, 0, 0]);
    interaction.setTarget(hand.target);
    interaction.setPinch(hand.pinch);
    step(1);
  }
  await drain(device);
  // let go as the worker does: the analysis runs alongside the substeps, the fingers open once it applied
  interaction.letGo(0);
  const opened = topology.request().then(() => interaction.release(0));
  step(4);
  await opened;
  step(12, 4);
  await drain(device);
  const x2 = new Float32Array(await readBuffer(device, solver.x));
  const v2 = new Float32Array(await readBuffer(device, solver.v));
  // the piece (the square inside the cuts) stays in the solve once detached: its pinch compression
  // relaxes and it falls as one body, spans at their rest length; it may tumble (rigid rotation from
  // the uneven release). Flying apart (the failure: metres per second, spread over centimetres) shows
  // as stretched spans and fast relative motion.
  const piece: number[] = [];
  const spans: Array<[number, number]> = [];
  for (let j = cj - r; j <= cj + r; j++) {
    for (let i = ci - r; i <= ci + r; i++) {
      piece.push(j * nx + i);
      if (i < ci + r) spans.push([j * nx + i, 0]);
      if (j < cj + r) spans.push([j * nx + i, 1]);
    }
  }
  const strain = Math.max(...spans.map(([a, f]) => Math.abs(dist(vec(x2, a), vec(x2, a + (f === 0 ? 1 : nx))) / scene.restLength[2 * a + f] - 1)));
  const mean = [0, 1, 2].map((q) => piece.reduce((a, k) => a + v2[4 * k + q], 0) / piece.length);
  const apart = Math.max(...piece.map((k) => Math.hypot(v2[4 * k] - mean[0], v2[4 * k + 1] - mean[1], v2[4 * k + 2] - mean[2])));
  check('release: a crushed piece only the fingers held is freed, and falls as one body', x2[4 * n + 3] > 0.5 && strain < 0.02 && apart < 0.2, `state ${x2[4 * n + 3]}, fall ${Math.hypot(...mean).toFixed(3)} m/s, largest span strain ${(100 * strain).toFixed(2)} %, largest relative speed ${apart.toFixed(3)} m/s after 0.13 s`);
  device.queue.writeBuffer(solver.segs, 0, segs0);
  topology.reset();
  solver.setState(initial);
}

async function contact(device: GPUDevice, solver: GpuSolver, initial: Float32Array, nx: number): Promise<void> {
  const mod = new ComputeModule(device, 'test.apply', common + step);
  const pipe = mod.pipeline('newtonApply');
  const modeU = createBufferWith(device, 'test.mode', new Uint32Array([0, 0, 0, 0]), UNIFORM);
  const group = mod.bindGroup('test.apply', { 0: solver.simParams, 1: modeU, 2: solver.nodes, 3: solver.x, 5: solver.dx, 9: solver.delta, 10: solver.counters, 11: solver.contact.meta });
  const f = 20 * nx + 20;
  const cs = [f, f + 1, f + nx, f + nx + 1];
  const node = 50 * nx + 50;
  const face = new Map<number, Vec3>(cs.map((c, q) => [c, [((q % 2) - 0.5) * 0.00342, (Math.floor(q / 2) - 0.5) * 0.00348, 0]]));
  /** One Newton update (full step) of the given candidate displacements, or a predicted velocity. */
  const apply = async (placed: Map<number, Vec3>, delta: Map<number, Vec3>, velocity: Map<number, Vec3>) => {
    const v = new Float32Array(4 * solver.N);
    const d = new Float32Array(4 * solver.N);
    for (const [n, q] of velocity) v.set(q, 4 * n);
    for (const [n, q] of delta) d.set(q, 4 * n);
    solver.setState(sparse(initial, placed), v);
    device.queue.writeBuffer(solver.dx, 0, new Float32Array(4 * solver.N));
    device.queue.writeBuffer(solver.delta, 0, d);
    device.queue.writeBuffer(solver.counters, 12, new Float32Array([1]));
    const enc = device.createCommandEncoder();
    solver.contact.encodeClear(enc);
    const p = enc.beginComputePass();
    if (velocity.size > 0) solver.encodePredict(p);
    solver.contact.encodeDetect(p);
    p.setPipeline(pipe);
    p.setBindGroup(0, group);
    p.dispatchWorkgroups(Math.ceil(solver.N / 128));
    p.end();
    device.queue.submit([enc.finish()]);
    return new Float32Array(await readBuffer(device, solver.dx));
  };
  const above = new Map(face).set(node, [0, 0, 0.0015]);
  const gap = (out: Float32Array) => 0.0015 + out[4 * node + 2] - cs.reduce((a, c) => a + out[4 * c + 2], 0) / 4;
  const cases: Array<[string, Map<number, Vec3>, Map<number, Vec3>]> = [
    ['a node driven through a still face', new Map([[node, [0, 0, -0.002]]]), new Map()],
    ['a face driven through a still node', new Map(cs.map((c) => [c, [0, 0, 0.003] as Vec3])), new Map()],
    ['node and face driven through each other', new Map<number, Vec3>([[node, [0, 0, -0.002]], ...cs.map((c) => [c, [0, 0, 0.002]] as [number, Vec3])]), new Map()],
  ];
  for (const [name, delta] of cases) {
    const g = gap(await apply(above, delta, new Map()));
    check(`contact: ${name} stays apart`, g > 0.3e-3, `gap ${(g * 1e3).toFixed(3)} mm`);
  }
  // 2.4 m/s through the face in one substep: the end of its path is far beyond any face near it
  const fast = await apply(above, new Map(), new Map([[node, [0, 0, -2.4]]]));
  // the predictor itself is the iterate here (no Newton direction): the bound applies to it
  const g = gap(fast);
  check('contact: a fast node is stopped by the face its path crosses', g > 0.3e-3, `gap ${(g * 1e3).toFixed(3)} mm`);

  // a lip across a tear within the lattice exclusion window: the node three rows above the face is
  // linked to it through its column while it holds, and is another lip once that column is cut
  const lip = f + 3 * nx;
  const link = f + 2 * nx;
  const torn = new Map(face).set(lip, [0, 0, 0.0015]).set(link, [0.01, 0.0, 0.0]);
  const segs = new Uint32Array(await readBuffer(device, solver.segs));
  const lipDelta = new Map<number, Vec3>([[lip, [0, 0, -0.002]]]);
  const linked = await apply(torn, lipDelta, new Map());
  const flags = segs.slice();
  flags[16 * (f + nx) + 13] |= 2;
  device.queue.writeBuffer(solver.segs, 0, flags);
  const cut = await apply(torn, lipDelta, new Map());
  device.queue.writeBuffer(solver.segs, 0, segs);
  const lipGap = (out: Float32Array) => 0.0015 + out[4 * lip + 2];
  check('contact: its own linked cloth in the exclusion window is not a contact', lipGap(linked) < 0, `gap ${(lipGap(linked) * 1e3).toFixed(3)} mm`);
  check('contact: a lip across a tear in the exclusion window is', lipGap(cut) > 0.3e-3, `gap ${(lipGap(cut) * 1e3).toFixed(3)} mm`);
  solver.setState(initial);
}

/**
 * A sharp capsule's point loading a node past the piercing force cuts both yarns of exactly the crossings
 * within its half-width of the node's place in the weave; a blunt capsule, or a node a claw holds, is not cut.
 */
async function piercing(device: GPUDevice, solver: GpuSolver, scene: CompiledScene, initial: Float32Array, nx: number): Promise<void> {
  const body = new BodyContact(device, solver, scene);
  const before = new Uint32Array(await readBuffer(device, solver.field));
  const L = scene.layers[0];
  const F = scene.field[0];
  const w = L.weave;
  const n = L.nodeOffset + 60 * nx + 90;
  const held = n + 12;
  const p = vec(initial, n);
  const r = 0.004;
  // the point presses 0.5 mm into the node from below: a penalty above the piercing force
  const press = (k: number, sharp: boolean) => {
    const q = vec(initial, k);
    const b = [q[0], q[1], q[2] - r + 0.0005];
    return { a: [b[0], b[1], b[2] - 0.02], b, radius: r, active: true, sharp };
  };
  if (BODY_CONTACT.stiffness * (BODY_CONTACT.distance + 0.0005) <= CLAW_PIERCE.force) throw new Error('test press below the piercing force');
  const run = async (caps: ReturnType<typeof press>[]) => {
    body.setPose(caps, 1 / 120);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    body.encodePierce(pass);
    pass.end();
    device.queue.submit([enc.finish()]);
    return new Uint32Array(await readBuffer(device, solver.field));
  };
  // which crossings lie within the point's half-width of a node's place in the weave
  const within = (k: number, c: number) => {
    const [i, j] = [(k - L.nodeOffset) % nx, Math.floor((k - L.nodeOffset) / nx)];
    const kk = (c - F.offset) % F.K;
    const m = Math.floor((c - F.offset) / F.K);
    return Math.hypot((kk - (i * F.bu + (F.bu - 1) / 2)) * w.pitchU, (m - (j * F.bv + (F.bv - 1) / 2)) * w.pitchV) <= CLAW_PIERCE.radius + 1e-9;
  };
  body.setExempt([held], true);
  const blunt = await run([press(n, false)]);
  const after = await run([press(n, true), press(held, true)]);
  let cut = 0;
  let wrong = 0;
  for (let c = 0; c < before.length; c++) {
    const expect = c >= F.offset && c < F.offset + F.K * F.M && within(n, c);
    if (expect) cut++;
    const want = expect ? (before[c] | 0xffff) >>> 0 : before[c];
    if (after[c] !== want || blunt[c] !== before[c]) wrong++;
  }
  check('pierce: a claw point cuts the crossings within its half-width, a blunt one and a held node none', cut > 0 && wrong === 0, `${cut} crossings to cut, ${wrong} differ (node at ${p.map((x) => (x * 1e3).toFixed(1)).join(', ')} mm)`);
  body.clear();
  device.queue.writeBuffer(solver.field, 0, before);
}

/**
 * A pinch entry displaced farther than a pad's radius from where the rest of the patch holds it (cloth
 * torn away from under the pad) slips out; one displaced within the pad, and the rest, stay held. A
 * claw's hold keeps the same entry.
 */
async function pinchSlip(device: GPUDevice, solver: GpuSolver, scene: CompiledScene, interaction: Interaction, initial: Float32Array, nx: number): Promise<void> {
  const grip = (interaction as unknown as { grip: GPUBuffer }).grip;
  const n0 = 60 * nx + 60;
  const p0 = vec(initial, n0);
  // entries (node, weight) of slot 0 after one substep with node `far` moved 20 mm and `near` 3 mm
  const run = async (attach: () => void): Promise<{ before: Map<number, number>; after: Map<number, number>; far: number; near: number }> => {
    solver.setState(initial);
    attach();
    interaction.setTarget(p0);
    // held entries (weight > 0) among the patch's `count` entries
    const read = async (count: number) => {
      const g = new Float32Array(await readBuffer(device, grip));
      const u = new Uint32Array(g.buffer);
      const m = new Map<number, number>();
      for (let q = 0; q < count; q++) if (g[8 * q + 1] > 0) m.set(u[8 * q], g[8 * q + 1]);
      return m;
    };
    const count = (interaction as unknown as { held: Array<{ count: number } | null> }).held[0]?.count ?? 0;
    const before = await read(count);
    const rim = [...before].filter(([, w]) => w < 0.5).map(([n]) => n);
    const far = rim[0];
    const near = rim[rim.length - 1];
    const xs = initial.slice();
    xs[4 * far + 2] += 0.02;
    xs[4 * near + 2] += 0.003;
    solver.setState(xs);
    const enc = device.createCommandEncoder();
    solver.encodeSubstep(enc);
    device.queue.submit([enc.finish()]);
    await drain(device);
    const after = await read(count);
    interaction.release(0);
    return { before, after, far, near };
  };
  const pinch = await run(() => interaction.attach(n0, p0, 0));
  const kept = [...pinch.before].filter(([n, w]) => n !== pinch.far && pinch.after.get(n) === w).length;
  check('grip: cloth torn from under a pad slips out of the pinch, the rest stays held', !pinch.after.has(pinch.far) && kept === pinch.before.size - 1, `entry 20 mm off ${pinch.after.has(pinch.far) ? 'still held' : 'slipped'}, ${kept} of ${pinch.before.size - 1} others unchanged (pad radius ${GRIP.radius * 1e3} mm)`);
  const hook = hookPatch(scene, [n0], CLAW_HOOK, interaction.isFree);
  const claw = hook ? await run(() => interaction.attachPatch(0, n0, hook, p0, CLAW_HOOK.maxLead)) : null;
  check('grip: a claw keeps what it hooks however it is displaced', !!claw && claw.after.size === claw.before.size, `${claw?.after.size} of ${claw?.before.size} entries held`);
  solver.setState(initial);
}

/**
 * Body contact over the substep: a step through a capsule, or a capsule sweeping through a node, meets
 * the surface on the side the node came from; a node carried along on a capsule, or one it passes by,
 * sees only its end-pose contact.
 */
async function sweptBody(device: GPUDevice, solver: GpuSolver, scene: CompiledScene, initial: Float32Array, nx: number): Promise<void> {
  const body = new BodyContact(device, solver, scene);
  const n = 40 * nx + 40;
  const r = 0.004;
  const k = BODY_CONTACT.stiffness;
  const delta = BODY_CONTACT.distance;
  // a limb along x at height z, y off the node's line
  const along = (z: number, y = 0) => ({ a: [-0.01, y, z], b: [0.01, y, z], radius: r, active: true });
  // gradient of the body terms at node n placed at p0 with step dp, the capsules moving from c0 to c1
  const run = async (p0: Vec3, dp: Vec3, c0: ReturnType<typeof along>[], c1: ReturnType<typeof along>[]): Promise<Vec3> => {
    solver.setState(sparse(initial, new Map([[n, p0]])));
    const dx = new Float32Array(4 * solver.N);
    dx.set(dp, 4 * n);
    device.queue.writeBuffer(solver.dx, 0, dx);
    body.clear();
    body.setPose(c0, solver.params.h);
    body.setPose(c1, solver.params.h);
    const enc = device.createCommandEncoder();
    enc.clearBuffer(solver.ext);
    const pass = enc.beginComputePass();
    body.encode((pipe, group, groups) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(groups);
    });
    pass.end();
    device.queue.submit([enc.finish()]);
    const ext = new Float32Array(await readBuffer(device, solver.ext));
    return [ext[12 * n], ext[12 * n + 1], ext[12 * n + 2]];
  };
  const sphere = [{ a: [0, 0, 0], b: [0, 0, 0], radius: r, active: true }];
  const fmt = (g: Vec3) => g.map((v) => v.toFixed(2)).join(', ');
  // a 20 mm step from 10 mm above a 4 mm sphere to 10 mm below it: pushed back up, 14 mm deep
  const through = await run([0, 0, 0.01], [0, 0, -0.02], sphere, sphere);
  const want = -k * (delta + 0.014);
  check('body: a step through a capsule meets it on the side it came from', Math.abs(through[2] - want) < 0.01 * -want && Math.hypot(through[0], through[1]) < 1e-3, `gradient (${fmt(through)}) N, expected z ${want.toFixed(2)}`);
  // a limb sweeping 30 mm up through a resting node: the node is on its upper face, 19 mm deep
  const swept = await run([0, 0, 0], [0, 0, 0], [along(-0.015)], [along(0.015)]);
  const wantS = -k * (delta + 0.019);
  check('body: a capsule sweeping through a node carries it on its leading face', Math.abs(swept[2] - wantS) < 0.01 * -wantS, `gradient (${fmt(swept)}) N, expected z ${wantS.toFixed(2)}`);
  // a node riding 1 mm above a limb that moves 20 mm with it: the end-pose penalty only, no friction
  const riding = await run([0, 0, -0.01 + r + 0.001], [0, 0, 0.02], [along(-0.01)], [along(0.01)]);
  const wantR = -k * (delta - 0.001);
  check('body: a node carried on a moving capsule keeps its end-pose contact', Math.abs(riding[2] - wantR) < 0.02 * -wantR && Math.hypot(riding[0], riding[1]) < 1e-3, `gradient (${fmt(riding)}) N, expected z ${wantR.toFixed(2)}`);
  // a limb passing 10 mm beside a node (6 mm clear of its surface): nothing
  const beside = await run([0, 0, 0], [0, 0, 0], [along(-0.015, 0.01)], [along(0.015, 0.01)]);
  check('body: a capsule passing beside a node does not touch it', Math.hypot(...beside) === 0, `gradient (${fmt(beside)}) N`);
  body.clear();
  solver.setState(initial);
}

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (p) => interaction.encodeSubstep(p);
  solver.preAssemble = (d) => interaction.encode(d);
  const initial = new Float32Array(await readBuffer(device, solver.x));
  const nx = scene.layers[0].nx;
  await picking(device, solver, interaction, initial, nx);
  await acquisition(device, solver, interaction, initial);
  await topologyRace(device, solver, interaction, initial, nx);
  await contact(device, solver, initial, nx);
  await piercing(device, solver, scene, initial, nx);
  await sweptBody(device, solver, scene, initial, nx);
  await pinchSlip(device, solver, scene, interaction, initial, nx);
  device.destroy();
  const failed = results.filter((r) => !r.ok).length;
  console.log(`=== ${results.length - failed}/${results.length} passed`);
  if (failed) throw new Error(`${failed} interaction check(s) failed`);
}
