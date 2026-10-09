/** Microbenchmark: cost of N dependent trivial dispatches in one compute pass. */
import { drain, gpuDevice } from '../env';

export default async function main(): Promise<void> {
  const device = await gpuDevice();
  const buf = device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE });
  const mod = device.createShaderModule({ code: `@group(0) @binding(0) var<storage, read_write> b: array<f32>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) { b[g.x % 1024u] = b[g.x % 1024u] + 1.0; }` });
  const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'main' } });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
  for (const [n, groups] of [[10, 1], [500, 1], [500, 64], [2000, 1]]) {
    const ts: number[] = [];
    for (let rep = 0; rep < 5; rep++) {
      const t0 = performance.now();
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipe);
      pass.setBindGroup(0, bg);
      for (let k = 0; k < n; k++) pass.dispatchWorkgroups(groups);
      pass.end();
      device.queue.submit([enc.finish()]);
      ts.push(await drain(device, t0));
    }
    ts.sort((a, b) => a - b);
    console.log(`${n} dispatches x ${groups} groups: ${ts[2].toFixed(2)} ms  (${((ts[2] * 1000) / n).toFixed(1)} us each)`);
  }
  device.destroy();
}
