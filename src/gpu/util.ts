/** Small WebGPU helpers: buffers, shader modules with diagnostics, pipelines. */

export const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
export const UNIFORM = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export function createBuffer(device: GPUDevice, label: string, size: number, usage: number = STORAGE): GPUBuffer {
  return device.createBuffer({ label, size: Math.max(16, align(size, 16)), usage });
}

export function createBufferWith(device: GPUDevice, label: string, data: ArrayBufferView, usage: number = STORAGE): GPUBuffer {
  const buf = createBuffer(device, label, data.byteLength, usage);
  device.queue.writeBuffer(buf, 0, data.buffer, data.byteOffset, data.byteLength);
  return buf;
}

export function align(n: number, a: number): number {
  return Math.ceil(n / a) * a;
}

const moduleCache = new WeakMap<GPUDevice, Map<string, GPUShaderModule>>();

/** Create (or reuse) a shader module; compilation errors are surfaced with source line context. */
export function shaderModule(device: GPUDevice, label: string, code: string): GPUShaderModule {
  let cache = moduleCache.get(device);
  if (!cache) {
    cache = new Map();
    moduleCache.set(device, cache);
  }
  const hit = cache.get(code);
  if (hit) return hit;
  const module = device.createShaderModule({ label, code });
  cache.set(code, module);
  void module.getCompilationInfo().then((info) => {
    const lines = code.split('\n');
    for (const m of info.messages) {
      if (m.type === 'info') continue;
      const ctx = lines.slice(Math.max(0, m.lineNum - 3), m.lineNum + 1).join('\n');
      console[m.type === 'error' ? 'error' : 'warn'](`[${label}] ${m.type} ${m.lineNum}:${m.linePos} ${m.message}\n${ctx}`);
    }
  });
  return module;
}

export function computePipeline(device: GPUDevice, label: string, code: string, entryPoint = 'main', constants?: Record<string, number>): GPUComputePipeline {
  return device.createComputePipeline({
    label,
    layout: 'auto',
    compute: { module: shaderModule(device, label, code), entryPoint, constants },
  });
}

export type BindingResource = GPUBuffer | { buffer: GPUBuffer; offset?: number; size?: number };

/** Bind group for group 0 from an ordered list of buffers (binding i = entries[i]). */
export function bindGroup(device: GPUDevice, label: string, pipeline: GPUComputePipeline | GPURenderPipeline, entries: Array<BindingResource | null>, group = 0): GPUBindGroup {
  const list: GPUBindGroupEntry[] = [];
  entries.forEach((e, i) => {
    if (!e) return;
    list.push({ binding: i, resource: e instanceof GPUBuffer ? { buffer: e } : e });
  });
  return device.createBindGroup({ label, layout: pipeline.getBindGroupLayout(group), entries: list });
}

/** Read a buffer back (debug/test paths only; never on the per-frame path). */
export async function readBuffer(device: GPUDevice, src: GPUBuffer, size = src.size, offset = 0): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, offset, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const out = staging.getMappedRange().slice(0);
  staging.unmap();
  staging.destroy();
  return out;
}

type BindingKind = 'uniform' | 'read-only-storage' | 'storage';

/** Parse `@group(0) @binding(N) var<...>` declarations of a WGSL module. */
export function parseBindings(code: string): Map<number, BindingKind> {
  const out = new Map<number, BindingKind>();
  const re = /@group\(0\)\s*@binding\((\d+)\)\s*var<([^>]+)>/g;
  for (let m = re.exec(code); m; m = re.exec(code)) {
    const decl = m[2].replace(/\s/g, '');
    const kind: BindingKind = decl === 'uniform' ? 'uniform' : decl.includes('read_write') ? 'storage' : 'read-only-storage';
    out.set(Number(m[1]), kind);
  }
  return out;
}

/**
 * One WGSL module with an explicit bind group layout shared by all of its entry points.
 * Bind groups may omit bindings an entry point does not use; omitted slots get dummy buffers.
 */
export class ComputeModule {
  readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly bindings: Map<number, BindingKind>;
  private readonly module: GPUShaderModule;
  private readonly dummies = new Map<number, GPUBuffer>();
  private readonly pipelines = new Map<string, GPUComputePipeline>();
  private readonly device: GPUDevice;
  readonly label: string;

  constructor(device: GPUDevice, label: string, code: string) {
    this.device = device;
    this.label = label;
    this.bindings = parseBindings(code);
    this.layout = device.createBindGroupLayout({
      label,
      entries: [...this.bindings].map(([binding, type]) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } })),
    });
    this.pipelineLayout = device.createPipelineLayout({ label, bindGroupLayouts: [this.layout] });
    this.module = shaderModule(device, label, code);
  }

  pipeline(entryPoint: string): GPUComputePipeline {
    let p = this.pipelines.get(entryPoint);
    if (!p) {
      p = this.device.createComputePipeline({ label: `${this.label}.${entryPoint}`, layout: this.pipelineLayout, compute: { module: this.module, entryPoint } });
      this.pipelines.set(entryPoint, p);
    }
    return p;
  }

  private dummy(binding: number, kind: BindingKind): GPUBuffer {
    let b = this.dummies.get(binding);
    if (!b) {
      b = this.device.createBuffer({
        label: `${this.label}.dummy${binding}`,
        size: kind === 'uniform' ? 4096 : 65536,
        usage: kind === 'uniform' ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE,
      });
      this.dummies.set(binding, b);
    }
    return b;
  }

  bindGroup(label: string, entries: Record<number, BindingResource>): GPUBindGroup {
    const list: GPUBindGroupEntry[] = [];
    for (const [binding, kind] of this.bindings) {
      const e = entries[binding];
      const resource = e ? (e instanceof GPUBuffer ? { buffer: e } : e) : { buffer: this.dummy(binding, kind) };
      list.push({ binding, resource });
    }
    for (const k of Object.keys(entries)) {
      if (!this.bindings.has(Number(k))) throw new Error(`${this.label}: no binding ${k}`);
    }
    return this.device.createBindGroup({ label, layout: this.layout, entries: list });
  }
}
