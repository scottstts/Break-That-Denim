/** Two-claw centre-piece classification, including cuts/release while readback is pending.
 * node tools/gpu/run.mjs tests/gpu/twohand-topology.ts
 * No contact/force integration: this isolates classification from subsequent motion.
 */
import { readBuffer } from '../../src/gpu/util';
import { Interaction } from '../../src/simulation/gpu/interaction';
import { Topology } from '../../src/simulation/gpu/topology';
import { NO_PIECE } from '../../src/simulation/gpu/solver';
import { gpuDevice } from '../../tools/gpu/env';
import { settledSolver } from '../../tools/gpu/scripts/render';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const { solver, scene } = await settledSolver(device);
  const interaction = new Interaction(device, solver, scene);
  const topology = new Topology(device, solver, scene, () => interaction.coreNodes);
  const initial = new Float32Array(await readBuffer(device, solver.x));
  const nx = scene.layers[0].nx;
  const node = (i: number, j: number) => (50 + j) * nx + 50 + i;
  const strip = Array.from({ length: 27 }, (_, k) => node(k % 9, Math.floor(k / 9)));
  const centre = strip.filter((n) => n % nx >= 53 && n % nx <= 55);
  let flags = scene.segFlags.slice();
  let checks = 0;
  const check = (name: string, ok: boolean) => {
    console.log(`${ok ? 'PASS' : 'FAIL'} ${name}`);
    if (!ok) throw new Error(name);
    checks++;
  };
  const upload = () => solver.setSegments({ ...scene, segFlags: flags }, scene.tails);
  const cut = (side: number) => {
    for (let j = 0; j < 3; j++) flags[2 * node(side === 0 ? 2 : 5, j)] |= 2;
    upload();
  };
  const state = async () => new Float32Array(await readBuffer(device, solver.x));
  const setup = async () => {
    interaction.releaseAll();
    topology.reset();
    const xs = initial.slice();
    for (let n = 0; n < solver.N; n++) xs[4 * n + 3] = 2;
    for (const n of strip) xs[4 * n + 3] = 0;
    solver.setState(xs);
    flags = scene.segFlags.map((v) => v | 2);
    for (let j = 0; j < 3; j++) for (let i = 0; i < 9; i++) {
      if (i < 8) flags[2 * node(i, j)] = 1;
      if (j < 2) flags[2 * node(i, j) + 1] = 1;
    }
    upload();
    for (let slot = 0; slot < 2; slot++) {
      const n = node(slot === 0 ? 0 : 8, 1);
      // Both claws retain a weak rim entry on the centre. It is not an anchor and must stop
      // contributing any force or stiffness after classification, even before the claws release.
      interaction.attachPatch(slot, n, { nodes: [n, centre[slot]], weights: [1, 0.25], sides: [0, 0], stiffness: 1000, padPitch: 0, between: [], slip: 0 }, [initial[4 * n], initial[4 * n + 1], initial[4 * n + 2]], 0.01);
    }
    await topology.request();
  };
  const verify = async (name: string) => {
    const xs = await state();
    check(`${name}: all 9 centre nodes detached; both 9-node held sides retained`, strip.every((n) => xs[4 * n + 3] === (centre.includes(n) ? 1 : 0)));
    const ids = new Uint32Array(await readBuffer(device, (solver as unknown as { pieceOf: GPUBuffer }).pieceOf));
    check(`${name}: centre is one rigid component; held sides excluded`, ids[centre[0]] !== NO_PIECE && centre.every((n) => ids[n] === ids[centre[0]]) && strip.filter((n) => !centre.includes(n)).every((n) => ids[n] === NO_PIECE));
    const enc = device.createCommandEncoder();
    enc.clearBuffer(solver.ext);
    const pass = enc.beginComputePass();
    interaction.encode((pipe, group, groups) => {
      pass.setPipeline(pipe); pass.setBindGroup(0, group); pass.dispatchWorkgroups(groups);
    });
    pass.end();
    device.queue.submit([enc.finish()]);
    const ext = new Float32Array(await readBuffer(device, solver.ext));
    check(`${name}: lingering claw rim entries exert zero force/stiffness on detached centre`, centre.every((n) => ext.slice(12 * n, 12 * n + 12).every((v) => v === 0)) && ext[12 * node(0, 1) + 4] > 0);
  };
  for (const first of [0, 1]) {
    await setup();
    cut(first);
    await topology.request();
    const xs = await state();
    check(`sequential ${first}: centre stays held by remaining hand after first cut`, centre.every((n) => xs[4 * n + 3] === 0));
    cut(1 - first);
    await topology.request();
    await verify(`sequential ${first}`);
  }
  await setup();
  cut(0); cut(1);
  await topology.request();
  await verify('both cuts before analysis');
  for (const first of [0, 1]) {
    await setup();
    cut(first);
    const pending = topology.request();
    cut(1 - first);
    const rerun = topology.request();
    await Promise.all([pending, rerun]);
    await verify(`second cut during pending analysis ${first}`);
  }
  await setup();
  const pending = topology.request();
  interaction.letGo(0);
  const left = topology.request();
  interaction.letGo(1);
  const right = topology.request();
  await Promise.all([pending, left, right]);
  const released = await state();
  check('both releases during pending analysis: whole strip detaches', strip.every((n) => released[4 * n + 3] === 1));
  console.log(`${checks}/${checks} two-hand topology checks passed`);
  device.destroy();
}
