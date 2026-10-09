/** Timing of one fine-level smoothing sweep, with and without the serial reduced-system phase. */
import { compileScene, defaultSceneSpecs } from '../../../src/simulation/scene';
import { GpuSolver, GPU_DEFAULTS } from '../../../src/simulation/gpu/solver';
import { drain, gpuDevice } from '../env';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const scene = compileScene(defaultSceneSpecs());
  const variants: Array<[string, ((m: string, c: string) => string) | undefined]> = [
    ['baseline', undefined],
    ['no reduced solve', (m, c) => (m === 'smooth' ? c.replace('if (t == 0u) {\n    let Q = 2u * P;', 'if (t == 999u) {\n    let Q = 2u * P;') : c)],
  ];
  for (const [name, patch] of variants) {
    const s = new GpuSolver(device, scene, { ...GPU_DEFAULTS, patch });
    const qs = device.createQuerySet({ type: 'timestamp', count: 8 });
    const res = device.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const rd = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    let out = '';
    for (let rep = 0; rep < 4; rep++) {
      const enc = device.createCommandEncoder();
      for (const [k, level] of [[0, 0], [1, 1]]) {
        const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
        for (let r = 0; r < 10; r++) s.encodeSmooth(p, level, false);
        p.end();
      }
      enc.resolveQuerySet(qs, 0, 4, res, 0);
      enc.copyBufferToBuffer(res, 0, rd, 0, 32);
      device.queue.submit([enc.finish()]);
      await drain(device);
      await rd.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(rd.getMappedRange().slice(0));
      rd.unmap();
      out = `L0 sweep ${(Number(t[1] - t[0]) / 1e7).toFixed(3)} ms, L1 sweep ${(Number(t[3] - t[2]) / 1e7).toFixed(3)} ms`;
    }
    console.log(name.padEnd(18), out);
  }
  device.destroy();
}
