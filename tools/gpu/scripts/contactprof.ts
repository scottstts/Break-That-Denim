/** GPU time of the self-contact stages on the settled drape (timestamp queries, mean of 10 repeats). */
import type { SelfContact } from '../../../src/simulation/gpu/selfcontact';
import { drain, gpuDevice } from '../env';
import { settledSolver } from './render';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const { solver } = await settledSolver(device);
  const c = solver.contact as SelfContact;
  const stages: Array<[string, (p: GPUComputePassEncoder) => void]> = [
    ['detect (hash + scan + query)', (p) => c.encodeDetect(p)],
    ['  face count', (p) => c.encodeDetect(p, 1)],
    ['  scan', (p) => c.encodeDetect(p, 2)],
    ['  fill', (p) => c.encodeDetect(p, 4)],
    ['  query', (p) => c.encodeDetect(p, 8)],
    ['terms + gather', (p) => c.encodeTerms(p)],
  ];
  const qs = device.createQuerySet({ type: 'timestamp', count: 2 * stages.length });
  const res = device.createBuffer({ size: 256 * stages.length, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const rd = device.createBuffer({ size: 16 * stages.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const total = new Array<number>(stages.length).fill(0);
  for (let r = 0; r < 10; r++) {
    for (let k = 0; k < stages.length; k++) {
      const enc = device.createCommandEncoder();
      // each repeat starts from a cleared face hash
      if (k === 0) c.encodeClear(enc);
      const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
      stages[k][1](p);
      p.end();
      enc.resolveQuerySet(qs, 2 * k, 2, res, 256 * k);
      enc.copyBufferToBuffer(res, 256 * k, rd, 16 * k, 16);
      device.queue.submit([enc.finish()]);
      await drain(device);
      await rd.mapAsync(GPUMapMode.READ, 16 * k, 16);
      const t = new BigUint64Array(rd.getMappedRange(16 * k, 16).slice(0));
      rd.unmap();
      total[k] += Number(t[1] - t[0]) / 1e6;
    }
  }
  stages.forEach(([name], k) => console.log(`${name.padEnd(30)} ${(total[k] / 10).toFixed(3)} ms`));
  device.destroy();
}
