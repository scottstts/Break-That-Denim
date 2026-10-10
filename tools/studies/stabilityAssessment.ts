/** Assessment-only probes. Production code is not changed. See artifacts/assessment/README.md. */
import { readBuffer, ComputeModule, createBufferWith, UNIFORM } from '../../src/gpu/util';
import { cameraMatrices, DEFAULT_CAMERA } from '../../src/rendering/renderer';
import { add, sub, scale, dot, normalize, transformPoint, type Vec3 } from '../../src/rendering/math';
import { GRIP, gripPatch, pinchScale } from '../../src/simulation/grip';
import { Interaction } from '../../src/simulation/gpu/interaction';
import { Topology } from '../../src/simulation/gpu/topology';
import common from '../../src/simulation/gpu/wgsl/common.wgsl?raw';
import step from '../../src/simulation/gpu/wgsl/step.wgsl?raw';
import { settledSolver } from '../gpu/scripts/render';
import { drain, gpuDevice } from '../gpu/env';

const vec = (a: Float32Array, n: number): Vec3 => [a[4 * n], a[4 * n + 1], a[4 * n + 2]];
const norm = (a: Vec3) => Math.hypot(...a);
const log = (o: unknown) => console.log(JSON.stringify(o));

export default async function main(args: string[]): Promise<void> {
  // GRIP='{"pinch":0.6}' overrides the grip parameters
  Object.assign(GRIP, JSON.parse(process.env.GRIP ?? '{}') as object);
  const device = await gpuDevice();
  const mode = args[0] ?? 'geometry';
  const refined = args.includes('refined');
  const half = args.includes('half');
  const quarter = args.includes('quarter');
  const newton = refined || args.includes('newton6') ? 6 : 2;
  const pcg = refined || args.includes('pcg4') ? 4 : 2;
  const { solver, scene } = await settledSolver(device, { newtonIterations: newton, newtonMax: newton, pcgIterations: pcg, ...(half || quarter ? { h: quarter ? 1 / 480 : 1 / 240 } : {}) });
  const N = solver.N;
  const nx = scene.layers[0].nx;
  const initial = new Float32Array(await readBuffer(device, solver.x));
  const interaction = new Interaction(device, solver, scene);
  solver.preSubstep = (p) => interaction.encodeSubstep(p);
  solver.preAssemble = (d) => interaction.encode(d);
  const m = cameraMatrices(DEFAULT_CAMERA, 1.6);
  const forward = normalize(sub(DEFAULT_CAMERA.target, m.eye));
  const project = (p: Vec3) => {
    const c = transformPoint(m.viewProj, p);
    return [800 * (1 + c[0] / c[3]), 500 * (1 - c[1] / c[3]), c[2] / c[3]];
  };
  const planePoint = (px: number, py: number, p: Vec3): Vec3 => {
    const a = transformPoint(m.invViewProj, [px / 800 - 1, 1 - py / 500, 0]);
    const b = transformPoint(m.invViewProj, [px / 800 - 1, 1 - py / 500, 1]);
    const origin: Vec3 = [a[0] / a[3], a[1] / a[3], a[2] / a[3]];
    const dir = normalize(sub([b[0] / b[3], b[1] / b[3], b[2] / b[3]], origin));
    return add(origin, scale(dir, dot(sub(p, origin), forward) / dot(dir, forward)));
  };
  log({ mode, refined, half, quarter, newton, pcg, nodes: N, nx, h: solver.params.h, camera: DEFAULT_CAMERA });

  if (mode === 'geometry') {
    const picks: Array<{ pixel: number[]; node: number; snapMm: number; wear: number }> = [];
    for (let y = 250; y <= 750; y += 50) for (let x = 400; x <= 1200; x += 50) {
      const hit = await interaction.pickAt(m.viewProj, x, y, 1600, 1000, 16);
      if (hit) picks.push({ pixel: [x, y], node: hit.node, snapMm: 1000 * norm(sub(planePoint(x, y, hit.position), hit.position)), wear: scene.wear[hit.node] });
    }
    picks.sort((a, b) => b.snapMm - a.snapMm);
    log({ kind: 'default-picking', count: picks.length, medianSnapMm: picks[Math.floor(picks.length / 2)].snapMm, worst: picks.slice(0, 8) });

    // Fold-like occlusion fixture: four corners of a front face surround the cursor, and a
    // material-distant crossing is directly behind its interior. Actual GPU picker, unchanged.
    const xs = initial.slice();
    for (let n = 0; n < N; n++) xs[4 * n + 3] = 2;
    const f = 20 * nx + 20;
    const back = 50 * nx + 50;
    const corners = [f, f + 1, f + nx, f + nx + 1];
    for (let q = 0; q < 4; q++) xs.set([(q % 2 - 0.5) * 0.00342, (Math.floor(q / 2) - 0.5) * 0.00348, 0, 0], 4 * corners[q]);
    const backPoint = scale(forward, 0.004);
    xs.set([...backPoint, 0], 4 * back);
    solver.setState(xs);
    const pp = project(backPoint);
    const hit = await interaction.pickAt(m.viewProj, pp[0], pp[1], 1600, 1000, 16);
    log({ kind: 'occluded-pick', back, frontCorners: corners, picked: hit?.node, depthGap: project(backPoint)[2] - project([0, 0, 0])[2], tolerance: 0.0004, backDistanceMm: 4 });

    // A cut through a pinch does not affect gripPatch: it consults rest UV / pins / ownership only.
    const n = Math.floor(N / 2);
    const patch = gripPatch(scene, n)!;
    const W = patch.weights.reduce((a, b) => a + b, 0);
    let contractEnergy = 0;
    let sumForces = 0;
    for (let k = 0; k < patch.nodes.length; k++) {
      const d = norm(sub(vec(initial, patch.nodes[k]), vec(initial, n))) * (1 - GRIP.pinch);
      contractEnergy += 0.5 * patch.stiffness * patch.weights[k] * d * d;
      sumForces += patch.stiffness * patch.weights[k] * d;
    }
    log({ kind: 'grip-scales', node: n, nodes: patch.nodes.length, totalStiffness: W * patch.stiffness, massKg: patch.nodes.reduce((a, b) => a + scene.mass[b], 0), contractionEnergyAtInitialShapeJ: contractEnergy, contractionSumForceN: sumForces, leadEnergyAtCapJ: 0.5 * W * patch.stiffness * GRIP.maxLead ** 2 });
    solver.setState(initial);
    const anchor = vec(initial, n);
    interaction.attach(n, anchor);
    interaction.setTarget(add(anchor, [0, 0, 0.1]));
    const dx = new Float32Array(4 * N);
    for (const node of patch.nodes) dx[4 * node + 2] = -0.005;
    device.queue.writeBuffer(solver.dx, 0, dx);
    const enc = device.createCommandEncoder();
    enc.clearBuffer(solver.ext);
    const p = enc.beginComputePass();
    interaction.encodeSubstep(p);
    interaction.encode((pipe, group, groups) => { p.setPipeline(pipe); p.setBindGroup(0, group); p.dispatchWorkgroups(groups); });
    p.end();
    device.queue.submit([enc.finish()]);
    const ext = new Float32Array(await readBuffer(device, solver.ext));
    log({ kind: 'force-cap-iterate', committedLeadMm: 6, oppositeIterateMm: 5, summedForceN: Math.abs(patch.nodes.reduce((a, node) => a + ext[12 * node + 2], 0)), advertisedCapN: GRIP.maxForce });
    interaction.release();
    // A material endpoint with one incoming span and neither outgoing span is still cloth.
    const endpoints = initial.slice();
    for (let k = 0; k < N; k++) endpoints[4 * k + 3] = 2;
    endpoints.set([0, 0, 0, 0], 4 * n);
    endpoints.set([-0.00342, 0, 0, 0], 4 * (n - 1));
    solver.setState(endpoints);
    const segData = new Uint32Array(await readBuffer(device, solver.segs));
    segData[16 * n + 5] |= 2;
    segData[16 * n + 13] |= 2;
    device.queue.writeBuffer(solver.segs, 0, segData);
    const ep = project([0, 0, 0]);
    const endpointHit = await interaction.pickAt(m.viewProj, ep[0], ep[1], 1600, 1000, 1);
    log({ kind: 'endpoint-pick', incomingFlags: segData[16 * (n - 1) + 5], outgoingFlags: [segData[16 * n + 5], segData[16 * n + 13]], picked: endpointHit?.node ?? null });
  } else if (mode === 'contact') {
    const f = 20 * nx + 20;
    const n = 50 * nx + 50;
    const cs = [f, f + 1, f + nx, f + nx + 1];
    const mod = new ComputeModule(device, 'assessment.apply', common + step);
    const pipe = mod.pipeline('newtonApply');
    const modeU = createBufferWith(device, 'assessment.mode', new Uint32Array([0, 0, 0, 0]), UNIFORM);
    const group = mod.bindGroup('assessment.apply', { 0: solver.simParams, 1: modeU, 2: solver.nodes, 3: solver.x, 5: solver.dx, 9: solver.delta, 10: solver.counters, 11: solver.contact.meta });
    const fixtures = [
      { name: 'stationary-face-approach', nodeDz: -0.002, faceDz: 0, prediction: false },
      { name: 'moving-face-crossing', nodeDz: 0, faceDz: 0.003, prediction: false },
      { name: 'fast-predictor-crossing', nodeDz: -0.020, faceDz: 0, prediction: true },
    ];
    for (const fixture of fixtures) {
      const xs = initial.slice();
      const v = new Float32Array(4 * N);
      const dx = new Float32Array(4 * N);
      const delta = new Float32Array(4 * N);
      for (let k = 0; k < N; k++) xs[4 * k + 3] = 2;
      for (let q = 0; q < 4; q++) xs.set([(q % 2 - 0.5) * 0.00342, (Math.floor(q / 2) - 0.5) * 0.00348, 0, 0], 4 * cs[q]);
      xs.set([0, 0, 0.0015, 0], 4 * n);
      if (fixture.prediction) {
        v[4 * n + 2] = fixture.nodeDz / solver.params.h;
      } else {
        delta[4 * n + 2] = fixture.nodeDz;
        for (const c of cs) delta[4 * c + 2] = fixture.faceDz;
      }
      solver.setState(xs, v);
      device.queue.writeBuffer(solver.dx, 0, dx);
      device.queue.writeBuffer(solver.delta, 0, delta);
      device.queue.writeBuffer(solver.counters, 12, new Float32Array([1]));
      const enc = device.createCommandEncoder();
      solver.contact.encodeClear(enc);
      const p = enc.beginComputePass();
      if (fixture.prediction) solver.encodePredict(p);
      solver.contact.encodeDetect(p);
      p.setPipeline(pipe);
      p.setBindGroup(0, group);
      p.dispatchWorkgroups(Math.ceil(N / 128));
      p.end();
      device.queue.submit([enc.finish()]);
      const out = new Float32Array(await readBuffer(device, solver.dx));
      const meta = new Float32Array(await readBuffer(device, solver.contact.meta));
      const gap = 0.0015 + out[4 * n + 2] - cs.reduce((a, c) => a + out[4 * c + 2], 0) / 4;
      log({ kind: 'contact-fixture', ...fixture, nodeBound: [...meta.slice(4 * n, 4 * n + 4)], appliedNodeDz: out[4 * n + 2], appliedFaceDz: cs.map(c => out[4 * c + 2]), signedGapMm: 1000 * gap });
    }
  } else if (mode === 'topology') {
    const topology = new Topology(device, solver, scene);
    const n = Math.floor(N / 2);
    const patch = gripPatch(scene, n)!;
    const flags = scene.segFlags.slice();
    // Isolate a square enclosing the patch by cutting its boundary in material coordinates.
    const ci = n % nx, cj = Math.floor(n / nx), r = 4;
    for (let j = cj - r; j <= cj + r; j++) {
      flags[2 * (j * nx + ci - r - 1)] |= 2;
      flags[2 * (j * nx + ci + r)] |= 2;
    }
    for (let i = ci - r; i <= ci + r; i++) {
      flags[2 * ((cj - r - 1) * nx + i) + 1] |= 2;
      flags[2 * ((cj + r) * nx + i) + 1] |= 2;
    }
    solver.setSegments({ ...scene, segFlags: flags }, scene.tails);
    const pending = topology.request([]);
    interaction.attach(n, vec(initial, n));
    const follow = topology.request(interaction.coreNodes);
    await Promise.all([pending, follow]);
    const after = new Float32Array(await readBuffer(device, solver.x));
    log({ kind: 'anchor-race', held: interaction.active, heldCore: interaction.coreNodes.length, coreDebris: interaction.coreNodes.filter(k => after[4 * k + 3] > 0.5).length, patchNodes: patch.nodes.length });
    // A live crossing beside debris is a legal picker result, but its patch includes dead neighbours.
    interaction.release();
    after[4 * n + 3] = 0;
    after[4 * (n + 1) + 3] = 0;
    solver.setState(after);
    interaction.attach(n, vec(after, n));
    log({ kind: 'retired-patch-members', attached: interaction.active, patchDebris: interaction.patchNodes.filter(k => after[4 * k + 3] > 0.5).length, patchNodes: interaction.patchNodes.length });
    // Control: if the same anchors are included in the initial analysis, the piece stays cloth.
    solver.setState(initial);
    topology.reset();
    await topology.request(interaction.coreNodes);
    const control = new Float32Array(await readBuffer(device, solver.x));
    log({ kind: 'anchor-control', coreDebris: interaction.coreNodes.filter(k => control[4 * k + 3] > 0.5).length });
  } else if (mode === 'dynamics') {
    const variant = args[1] ?? 'click';
    const noPinch = variant === 'no-pinch' || variant === 'rest';
    const offsetMm = variant === 'offset' ? 4 : variant === 'tiny' ? 1 : 0;
    const topology = new Topology(device, solver, scene);
    await topology.request([]);
    const x0 = new Float32Array(await readBuffer(device, solver.x));
    const px = Number(process.env.PICK_X ?? 800);
    const py = Number(process.env.PICK_Y ?? 500);
    const hit = await interaction.pickAt(m.viewProj, px, py, 1600, 1000, 16);
    if (!hit) throw new Error('No pick');
    const anchor = hit.position;
    const cursorPoint = variant === 'pointer' ? planePoint(px, py, anchor) : variant === 'offset' ? add(anchor, [offsetMm / 1000, 0, 0]) : anchor;
    const patch = gripPatch(scene, hit.node)!;
    if (variant !== 'rest') interaction.attach(hit.node, anchor);
    let handle: Vec3 = [...anchor];
    const h = solver.params.h;
    const perSample = Math.round(1 / (30 * h));
    const samples: unknown[] = [];
    let maxSpeed = 0, maxP99 = 0, maxDisplacement = 0, maxEnergy = 0, maxUpdate = 0;
    let releaseSpeed = 0, releaseEnergy = 0, postSpeed = 0, postEnergy = 0;
    let topologyEvents = 0;
    const t0 = performance.now();
    const totalSteps = Math.round(1.2 / h);
    const releaseStep = Math.round(0.3 / h);
    for (let k = 0; k < totalSteps; k++) {
      if (k === releaseStep) {
        interaction.release();
        await topology.request([]);
      }
      if (k < releaseStep && variant !== 'rest') {
        // Match worker's camera-normal lift. "batched" advances its wall clock in four-step
        // chunks; other variants deliberately use simulation time to isolate constitutive response.
        const since = variant === 'batched' ? (Math.floor(k / 4) + 1) * 4 * h : (k + 1) * h;
        const target = variant === 'tiny' ? add(anchor, [offsetMm / 1000 * Math.min(1, since / 0.15), 0, 0]) : cursorPoint;
        const drag = norm(sub(target, anchor));
        const lift = 0.002 * Math.min(1, since / GRIP.engageTime) + Math.min(0.05, 0.35 * drag);
        const goal = add(target, scale(forward, -lift));
        const d = sub(goal, handle);
        handle = add(handle, scale(d, Math.min(1, 2.5 * h / Math.max(norm(d), 1e-15))));
        interaction.setTarget(handle);
        interaction.setPinch(noPinch ? 1 : pinchScale(since));
      }
      const enc = device.createCommandEncoder();
      if (k % perSample === 0) enc.clearBuffer(solver.counters, 8, 4);
      solver.encodeSubstep(enc);
      device.queue.submit([enc.finish()]);
      if (k % perSample !== perSample - 1) continue;
      await drain(device);
      const x = new Float32Array(await readBuffer(device, solver.x));
      const v = new Float32Array(await readBuffer(device, solver.v));
      const raw = await readBuffer(device, solver.counters, 64);
      const c = new Uint32Array(raw);
      const cf = new Float32Array(raw);
      const speeds: number[] = [];
      let kinetic = 0, disp = 0, bad = 0;
      for (let n = 0; n < N; n++) {
        if (x[4 * n + 3] > 0.5 || scene.pinned[n]) continue;
        const speed = norm(vec(v, n));
        speeds.push(speed);
        kinetic += 0.5 * scene.mass[n] * speed ** 2;
        disp = Math.max(disp, norm(sub(vec(x, n), vec(x0, n))));
        if (!Number.isFinite(speed)) bad++;
      }
      speeds.sort((a, b) => a - b);
      const peak = speeds[speeds.length - 1];
      const p99 = speeds[Math.floor(speeds.length * 0.99)];
      maxSpeed = Math.max(maxSpeed, peak); maxP99 = Math.max(maxP99, p99);
      maxDisplacement = Math.max(maxDisplacement, disp); maxEnergy = Math.max(maxEnergy, kinetic);
      maxUpdate = Math.max(maxUpdate, cf[4 + solver.params.newtonMax - 1]);
      if (k + 1 === releaseStep) { releaseSpeed = peak; releaseEnergy = kinetic; }
      if (k >= releaseStep) { postSpeed = Math.max(postSpeed, peak); postEnergy = Math.max(postEnergy, kinetic); }
      if (c[0] + c[1] !== topologyEvents) {
        topologyEvents = c[0] + c[1];
        await topology.request(interaction.coreNodes);
      }
      const row = { t: (k + 1) * h, maxSpeed: peak, p99Speed: p99, kineticJ: kinetic, maxDisplacementMm: 1000 * disp, ruptures: c[0], pulledOut: c[1], rejected: c[12], lastUpdateMm: 1000 * cf[4 + solver.params.newtonMax - 1], nonfinite: bad };
      samples.push(row);
      if (process.env.FULL_TRACE || (k + 1) % (3 * perSample) === 0) log({ kind: 'sample', variant, refined, half, quarter, newton, pcg, ...row });
    }
    log({ kind: 'summary', variant, refined, half, quarter, newton, pcg, pixel: [px, py], picked: hit.node, wear: scene.wear[hit.node], patchNodes: patch.nodes.length, maxSpeed, maxP99, maxDisplacementMm: 1000 * maxDisplacement, maxKineticJ: maxEnergy, maxLastUpdateMm: 1000 * maxUpdate, releaseSpeed, releaseEnergy, postSpeed, postEnergy, last: samples.at(-1), wallSeconds: (performance.now() - t0) / 1000 });
  } else throw new Error(`Unknown mode ${mode}`);
  device.destroy();
}
