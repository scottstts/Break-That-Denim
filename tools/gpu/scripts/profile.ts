/** Per-stage GPU timing (timestamp queries; each stage repeated 10x inside one pass). */
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { GpuSolver, GPU_DEFAULTS } from '../../../src/simulation/gpu/solver';
import { drain, gpuDevice } from '../env';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const scene = compileScene(defaultSceneSpecs());
  const s = new GpuSolver(device, scene, { ...GPU_DEFAULTS });
  console.log('nodes', scene.nodeCount, 'levels', s.levels.map((l) => `${l.layers[0].nx}x${l.layers[0].ny}`).join(' '));
  const stages: Array<[string, (p: GPUComputePassEncoder) => void]> = [
    ['assemble', (p) => s.encodeKernel(p, 'assemble')],
    ['assemble+rap', (p) => s.encodeAssemble(p)],
  ];
  s.levels.forEach((_, k) => stages.push([`smooth sweep L${k}`, (p) => s.encodeSmooth(p, k, false)]));
  stages.push(['residual L0', (p) => s.encodeKernel(p, 'residual', 0)]);
  stages.push(['vcycle', (p) => s.encodeVCycle(p, 0)]);
  stages.push(['matvec', (p) => s.encodeKernel(p, 'matvec')]);
  stages.push(['pcg', (p) => s.encodePCG(p)]);
  stages.push(['newton iteration', (p) => s.encodeNewtonIteration(p, 0)]);
  stages.push(['predict+finalize', (p) => { s.encodePredict(p); s.encodeFinalize(p); }]);
  const qs = device.createQuerySet({ type: 'timestamp', count: 2 * stages.length });
  const res = device.createBuffer({ size: 16 * stages.length, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const rd = device.createBuffer({ size: 16 * stages.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  for (let rep = 0; rep < 3; rep++) {
    const lines: string[] = [];
    for (let k = 0; k < stages.length; k++) {
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
      for (let r = 0; r < 10; r++) stages[k][1](p);
      p.end();
      device.queue.submit([enc.finish()]);
      await drain(device);
    }
    const enc = device.createCommandEncoder();
    enc.resolveQuerySet(qs, 0, 2 * stages.length, res, 0);
    enc.copyBufferToBuffer(res, 0, rd, 0, 16 * stages.length);
    device.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(rd.getMappedRange().slice(0));
    rd.unmap();
    stages.forEach(([name], k) => lines.push(`${name.padEnd(18)} ${(Number(t[2 * k + 1] - t[2 * k]) / 1e7).toFixed(3)} ms`));
    if (rep === 2) console.log(lines.join('\n'));
  }
  device.destroy();
}
