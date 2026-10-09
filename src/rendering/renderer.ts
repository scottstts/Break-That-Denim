import type { CompiledScene } from '../simulation/scene';
import type { GpuSolver } from '../simulation/gpu/solver';
import { ComputeModule, createBuffer, createBufferWith, shaderModule, STORAGE, UNIFORM } from '../gpu/util';
import { invert, lookAt, multiply, normalize, orthographic, perspective, sub, type Mat4, type Vec3 } from './math';
import commonWgsl from './wgsl/render_common.wgsl?raw';
import lightWgsl from './wgsl/render_light.wgsl?raw';
import refineWgsl from './wgsl/refine.wgsl?raw';
import surfaceWgsl from './wgsl/surface.wgsl?raw';
import yarnGeomWgsl from './wgsl/yarn_geom.wgsl?raw';
import yarnWgsl from './wgsl/yarn.wgsl?raw';
import fuzzWgsl from './wgsl/fuzz.wgsl?raw';
import postWgsl from './wgsl/post.wgsl?raw';
import weaveTileWgsl from './wgsl/weave_tile.wgsl?raw';

export interface CameraState {
  target: Vec3;
  distance: number;
  /** Tilt from the cloth normal toward -y (rad). */
  pitch: number;
  /** Rotation about the cloth normal (rad). */
  yaw: number;
  fovY: number;
}

/** About 35 cm of cloth across the view height: the projected weave density of the reference. */
export const DEFAULT_CAMERA: CameraState = {
  target: [0.0, 0.0, -0.01],
  distance: 0.98,
  pitch: 0.16,
  yaw: 0,
  fovY: (20 * Math.PI) / 180,
};

const SAMPLES = 4;
/** Texels per side of the prefiltered weave tile (one twill repeat) at mip level 0. */
const TILE_SIZE = 128;
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const SHADOW_SIZE = 4096;
const SHADOW_HALF = 0.34;
/** Sub-quads per lattice cell edge of the bicubic surface. */
const REFINE = 2;
const MAX_SPANS = 65536;
const MAX_TAILS = 65536;
/** Vertices per explicit-yarn ribbon (2 per centreline sample), tails, and fibres; see the WGSL. */
const SPAN_VERTS = 18;
const TAIL_VERTS = 14;
const FUZZ_VERTS = 10;

export interface ViewMatrices {
  eye: Vec3;
  view: Mat4;
  proj: Mat4;
  viewProj: Mat4;
  invViewProj: Mat4;
}

export function cameraMatrices(c: CameraState, aspect: number): ViewMatrices {
  const dir: Vec3 = [Math.sin(c.yaw) * Math.sin(c.pitch), -Math.cos(c.yaw) * Math.sin(c.pitch), Math.cos(c.pitch)];
  const eye: Vec3 = [c.target[0] + dir[0] * c.distance, c.target[1] + dir[1] * c.distance, c.target[2] + dir[2] * c.distance];
  const up: Vec3 = [-Math.sin(c.yaw), Math.cos(c.yaw), 0];
  const view = lookAt(eye, c.target, up);
  const proj = perspective(c.fovY, aspect, 0.05, 3.0);
  const viewProj = multiply(proj, view);
  return { eye, view, proj, viewProj, invViewProj: invert(viewProj) };
}

/**
 * Renders the accepted mechanical state: intact cells as a bicubic surface with procedural woven
 * microstructure, damaged neighbourhoods as explicit yarns with frayed tails and loose fibres.
 */
export class Renderer {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  readonly solver: GpuSolver;
  readonly scene: CompiledScene;
  /** Key light position relative to the camera target, its intensity, and exposure. */
  keyOffset: Vec3 = [-0.3, 0.46, 0.56];
  keyIntensity = 4.2;
  exposure = 1.0;
  /** Sky/bounce fill (hemisphere top): shadowed cloth stays a dark blue, as in the reference. */
  ambient: Vec3 = [0.07, 0.08, 0.11];

  private readonly frameU: GPUBuffer;
  private readonly renderU: GPUBuffer;
  private readonly nrm: GPUBuffer;
  private readonly wear: GPUBuffer;
  private readonly cellFlags: GPUBuffer;
  private readonly lists: GPUBuffer;
  private readonly drawArgs: GPUBuffer;
  private readonly rsurf: GPUBuffer;
  private readonly cellCount: number;
  private readonly refinedCount: number;
  private readonly refine: ComputeModule;
  private readonly refineGroup: GPUBindGroup;
  private readonly pipes: Record<string, GPURenderPipeline> = {};
  private readonly cpipes: Record<string, GPUComputePipeline> = {};
  private readonly surfaceLayout: GPUBindGroupLayout;
  private readonly yarnLayout: GPUBindGroupLayout;
  private readonly postLayout: GPUBindGroupLayout;
  private readonly shadowTex: GPUTexture;
  private readonly weaveTile: GPUTexture;
  private readonly tileSampler: GPUSampler;
  private readonly shadowSampler: GPUSampler;
  private readonly dummyDepth: GPUTexture;
  private surfaceGroup: GPUBindGroup;
  private surfaceShadowGroup: GPUBindGroup;
  private yarnGroup: GPUBindGroup;
  private yarnShadowGroup: GPUBindGroup;
  private postGroup: GPUBindGroup | null = null;
  private msaaColor: GPUTexture | null = null;
  private msaaDepth: GPUTexture | null = null;
  private hdr: GPUTexture | null = null;
  width = 1;
  height = 1;
  private frameIndex = 0;

  constructor(device: GPUDevice, format: GPUTextureFormat, solver: GpuSolver, scene: CompiledScene) {
    this.device = device;
    this.format = format;
    this.solver = solver;
    this.scene = scene;
    const N = scene.nodeCount;

    let cells = 0;
    let refined = 0;
    for (const L of scene.layers) {
      cells += (L.nx - 1) * (L.ny - 1);
      refined += ((L.nx - 1) * REFINE + 1) * ((L.ny - 1) * REFINE + 1);
    }
    this.cellCount = cells;
    this.refinedCount = refined;

    this.frameU = createBuffer(device, 'frameU', 256, UNIFORM);
    this.renderU = createBufferWith(device, 'renderU', new Uint8Array(this.renderParams()), UNIFORM);
    const nInit = new Float32Array(4 * N);
    for (let n = 0; n < N; n++) nInit[4 * n + 2] = 1;
    this.nrm = createBufferWith(device, 'nrm', nInit);
    this.wear = createBufferWith(device, 'wear', Float32Array.from(scene.wear));
    this.cellFlags = createBuffer(device, 'cellFlags', 4 * cells);
    this.lists = createBuffer(device, 'lists', 4 * (4 + MAX_SPANS + MAX_TAILS));
    this.drawArgs = createBuffer(device, 'drawArgs', 64, STORAGE | GPUBufferUsage.INDIRECT);
    this.rsurf = createBuffer(device, 'rsurf', 48 * refined);

    this.refine = new ComputeModule(device, 'refine', commonWgsl + refineWgsl);
    for (const e of ['nodeFrame', 'classifyCells', 'classifySpans', 'args', 'refine']) this.cpipes[e] = this.refine.pipeline(e);
    this.refineGroup = this.refine.bindGroup('refine', {
      0: this.renderU,
      1: solver.nodes,
      2: solver.x,
      3: solver.segs,
      4: solver.tails,
      5: this.nrm,
      6: this.cellFlags,
      7: this.lists,
      8: this.rsurf,
      9: this.drawArgs,
      10: this.wear,
    });

    const V = GPUShaderStage.VERTEX;
    const F = GPUShaderStage.FRAGMENT;
    const VF = V | F;
    const ro = (binding: number, visibility = V): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'read-only-storage' } });
    this.surfaceLayout = device.createBindGroupLayout({
      label: 'surface',
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 1, visibility: VF, buffer: { type: 'uniform' } },
        ro(2),
        ro(3),
        { binding: 4, visibility: F, texture: { sampleType: 'depth' } },
        { binding: 5, visibility: F, sampler: { type: 'comparison' } },
        { binding: 6, visibility: F, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 7, visibility: F, sampler: { type: 'filtering' } },
      ],
    });
    this.yarnLayout = device.createBindGroupLayout({
      label: 'yarn',
      entries: [
        { binding: 0, visibility: VF, buffer: { type: 'uniform' } },
        { binding: 1, visibility: VF, buffer: { type: 'uniform' } },
        ro(2),
        ro(3),
        ro(4),
        ro(5),
        ro(6),
        ro(7),
        ro(8),
        { binding: 9, visibility: F, texture: { sampleType: 'depth' } },
        { binding: 10, visibility: F, sampler: { type: 'comparison' } },
      ],
    });

    const depth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' };
    const shadowDepth: GPUDepthStencilState = { ...depth, depthBias: 2, depthBiasSlopeScale: 2.0 };
    const ms: GPUMultisampleState = { count: SAMPLES };
    const strip: GPUPrimitiveState = { topology: 'triangle-strip', cullMode: 'none' };
    const list: GPUPrimitiveState = { topology: 'triangle-list', cullMode: 'none' };

    const surfaceModule = shaderModule(device, 'surface', commonWgsl + lightWgsl + surfaceWgsl);
    const surfacePL = device.createPipelineLayout({ bindGroupLayouts: [this.surfaceLayout] });
    this.pipes.surface = device.createRenderPipeline({
      label: 'surface',
      layout: surfacePL,
      vertex: { module: surfaceModule, entryPoint: 'vsMain' },
      fragment: { module: surfaceModule, entryPoint: 'fsMain', targets: [{ format: HDR_FORMAT }] },
      primitive: list,
      depthStencil: depth,
      multisample: ms,
    });
    this.pipes.surfaceShadow = device.createRenderPipeline({
      label: 'surface.shadow',
      layout: surfacePL,
      vertex: { module: surfaceModule, entryPoint: 'vsShadow' },
      primitive: list,
      depthStencil: shadowDepth,
    });

    const yarnPL = device.createPipelineLayout({ bindGroupLayouts: [this.yarnLayout] });
    const yarnModule = shaderModule(device, 'yarn', commonWgsl + lightWgsl + yarnGeomWgsl + yarnWgsl);
    for (const [name, vs] of [
      ['span', 'vsSpan'],
      ['tail', 'vsTail'],
    ]) {
      this.pipes[name] = device.createRenderPipeline({
        label: name,
        layout: yarnPL,
        vertex: { module: yarnModule, entryPoint: vs },
        fragment: { module: yarnModule, entryPoint: 'fsYarn', targets: [{ format: HDR_FORMAT }] },
        primitive: strip,
        depthStencil: depth,
        multisample: ms,
      });
      this.pipes[`${name}Shadow`] = device.createRenderPipeline({
        label: `${name}.shadow`,
        layout: yarnPL,
        vertex: { module: yarnModule, entryPoint: `${vs}Shadow` },
        primitive: strip,
        depthStencil: shadowDepth,
      });
    }
    const fuzzModule = shaderModule(device, 'fuzz', commonWgsl + lightWgsl + yarnGeomWgsl + fuzzWgsl);
    for (const [name, vs] of [
      ['yarnFuzz', 'vsYarnFuzz'],
      ['tailFuzz', 'vsTailFuzz'],
    ]) {
      this.pipes[name] = device.createRenderPipeline({
        label: name,
        layout: yarnPL,
        vertex: { module: fuzzModule, entryPoint: vs },
        fragment: { module: fuzzModule, entryPoint: 'fsFuzz', targets: [{ format: HDR_FORMAT }] },
        primitive: strip,
        depthStencil: depth,
        multisample: { count: SAMPLES, alphaToCoverageEnabled: true },
      });
    }

    this.shadowTex = device.createTexture({
      label: 'shadowMap',
      size: [SHADOW_SIZE, SHADOW_SIZE],
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.dummyDepth = device.createTexture({ size: [1, 1], format: 'depth32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
    this.shadowSampler = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });

    this.weaveTile = this.buildWeaveTile();
    this.tileSampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 16 });
    const tileView = this.weaveTile.createView({ dimension: '2d-array' });
    const surfaceEntries = (shadow: GPUTextureView): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: this.frameU } },
      { binding: 1, resource: { buffer: this.renderU } },
      { binding: 2, resource: { buffer: this.rsurf } },
      { binding: 3, resource: { buffer: this.cellFlags } },
      { binding: 4, resource: shadow },
      { binding: 5, resource: this.shadowSampler },
      { binding: 6, resource: tileView },
      { binding: 7, resource: this.tileSampler },
    ];
    const yarnEntries = (shadow: GPUTextureView): GPUBindGroupEntry[] => [
      { binding: 0, resource: { buffer: this.frameU } },
      { binding: 1, resource: { buffer: this.renderU } },
      { binding: 2, resource: { buffer: solver.nodes } },
      { binding: 3, resource: { buffer: solver.x } },
      { binding: 4, resource: { buffer: this.nrm } },
      { binding: 5, resource: { buffer: solver.segs } },
      { binding: 6, resource: { buffer: solver.tails } },
      { binding: 7, resource: { buffer: this.lists } },
      { binding: 8, resource: { buffer: this.wear } },
      { binding: 9, resource: shadow },
      { binding: 10, resource: this.shadowSampler },
    ];
    const shadowView = this.shadowTex.createView();
    const dummyView = this.dummyDepth.createView();
    this.surfaceGroup = device.createBindGroup({ layout: this.surfaceLayout, entries: surfaceEntries(shadowView) });
    this.surfaceShadowGroup = device.createBindGroup({ layout: this.surfaceLayout, entries: surfaceEntries(dummyView) });
    this.yarnGroup = device.createBindGroup({ layout: this.yarnLayout, entries: yarnEntries(shadowView) });
    this.yarnShadowGroup = device.createBindGroup({ layout: this.yarnLayout, entries: yarnEntries(dummyView) });

    const postModule = shaderModule(device, 'post', commonWgsl + postWgsl);
    this.postLayout = device.createBindGroupLayout({
      label: 'post',
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, texture: { sampleType: 'unfilterable-float' } },
      ],
    });
    this.pipes.post = device.createRenderPipeline({
      label: 'post',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.postLayout] }),
      vertex: { module: postModule, entryPoint: 'vsFull' },
      fragment: { module: postModule, entryPoint: 'fsTonemap', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  /** Prefiltered weave tile per layer: every mip level averaged analytically (weave_tile.wgsl). */
  private buildWeaveTile(): GPUTexture {
    const layers = this.scene.layers.length;
    const levels = Math.log2(TILE_SIZE) + 1;
    const tex = this.device.createTexture({
      label: 'weaveTile',
      size: [TILE_SIZE, TILE_SIZE, layers],
      format: 'rgba16float',
      mipLevelCount: levels,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const module = shaderModule(this.device, 'weaveTile', commonWgsl + weaveTileWgsl);
    const layout = this.device.createBindGroupLayout({
      label: 'weaveTile',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
      ],
    });
    const pipe = this.device.createComputePipeline({
      label: 'weaveTile.build',
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'build' },
    });
    const enc = this.device.createCommandEncoder({ label: 'weaveTile' });
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    const temps: GPUBuffer[] = [];
    for (let l = 0; l < layers; l++) {
      for (let k = 0; k < levels; k++) {
        const size = TILE_SIZE >> k;
        const u = createBufferWith(this.device, 'weaveTile.level', new Uint32Array([l, size, 0, 0]), UNIFORM);
        temps.push(u);
        const view = tex.createView({ dimension: '2d', baseMipLevel: k, mipLevelCount: 1, baseArrayLayer: l, arrayLayerCount: 1 });
        pass.setBindGroup(0, this.device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: this.renderU } }, { binding: 1, resource: { buffer: u } }, { binding: 2, resource: view }] }));
        pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8));
      }
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
    void this.device.queue.onSubmittedWorkDone().then(() => temps.forEach((b) => b.destroy()));
    return tex;
  }

  private renderParams(): ArrayBuffer {
    const buf = new ArrayBuffer(16 + 3 * 96);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    const s = this.scene;
    u[0] = s.nodeCount;
    u[1] = s.layers.length;
    u[2] = this.cellCount;
    u[3] = this.refinedCount;
    let cellOffset = 0;
    let refinedOffset = 0;
    s.layers.forEach((L, l) => {
      const b = (16 + 96 * l) / 4;
      const m = L.weave;
      u[b] = L.nodeOffset;
      u[b + 1] = L.nx;
      u[b + 2] = L.ny;
      // yarns per span (warp column, pick row) and failure strands per span (warp, pick), one byte each
      u[b + 3] = L.bundle[0] | (L.bundle[1] << 8) | ((L.material.yarns[1].strandsPerSpan ?? 1) << 16) | ((L.material.yarns[0].strandsPerSpan ?? 1) << 24);
      f[b + 4] = m.pitchU;
      f[b + 5] = m.pitchV;
      f[b + 6] = 1; // twill direction: wales rise to the right at ~57 degrees in view, as in the reference
      const warp = m.yarns[1];
      const weft = m.yarns[0];
      f.set([warp.width, warp.height, warp.crimpAmplitude, 0.42], b + 8);
      f.set([weft.width, weft.height, weft.crimpAmplitude, -0.38], b + 12);
      u[b + 16] = m.twillRepeat;
      u[b + 17] = m.twillWarpOver;
      u[b + 18] = REFINE;
      const rnx = (L.nx - 1) * REFINE + 1;
      u[b + 20] = refinedOffset;
      u[b + 21] = rnx;
      u[b + 22] = cellOffset;
      u[b + 23] = m.id;
      cellOffset += (L.nx - 1) * (L.ny - 1);
      refinedOffset += rnx * ((L.ny - 1) * REFINE + 1);
    });
    return buf;
  }

  resize(width: number, height: number): void {
    if (width === this.width && height === this.height && this.hdr) return;
    this.width = width;
    this.height = height;
    this.msaaColor?.destroy();
    this.msaaDepth?.destroy();
    this.hdr?.destroy();
    const size = [width, height];
    this.msaaColor = this.device.createTexture({ label: 'msaaColor', size, format: HDR_FORMAT, sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.msaaDepth = this.device.createTexture({ label: 'msaaDepth', size, format: 'depth32float', sampleCount: SAMPLES, usage: GPUTextureUsage.RENDER_ATTACHMENT });
    this.hdr = this.device.createTexture({ label: 'hdr', size, format: HDR_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.postGroup = this.device.createBindGroup({ layout: this.postLayout, entries: [{ binding: 0, resource: { buffer: this.frameU } }, { binding: 1, resource: this.hdr.createView() }] });
  }

  private lightMatrix(target: Vec3): Mat4 {
    const key: Vec3 = [target[0] + this.keyOffset[0], target[1] + this.keyOffset[1], target[2] + this.keyOffset[2]];
    const dir = normalize(sub(key, target));
    const eye: Vec3 = [target[0] + dir[0] * 0.9, target[1] + dir[1] * 0.9, target[2] + dir[2] * 0.9];
    const view = lookAt(eye, target, Math.abs(dir[2]) > 0.95 ? [0, 1, 0] : [0, 0, 1]);
    const proj = orthographic(-SHADOW_HALF, SHADOW_HALF, -SHADOW_HALF, SHADOW_HALF, 0.6, 1.2);
    return multiply(proj, view);
  }

  /**
   * Encode the full frame for the current accepted state into `target`. `profile` (development)
   * receives begin/end timestamps of the prep, shadow, main and post passes (8 queries).
   */
  encode(encoder: GPUCommandEncoder, camera: CameraState, time: number, target: GPUTextureView, profile?: GPUQuerySet): void {
    const tw = (k: number) => (profile ? { querySet: profile, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } : undefined);
    if (!this.hdr || !this.msaaColor || !this.msaaDepth || !this.postGroup) return;
    const m = cameraMatrices(camera, this.width / this.height);
    const key: Vec3 = [camera.target[0] + this.keyOffset[0], camera.target[1] + this.keyOffset[1], camera.target[2] + this.keyOffset[2]];
    // aim the key slightly above the frame centre so illumination falls off toward the bottom
    const aim: Vec3 = [camera.target[0] + 0.03, camera.target[1] + 0.065, camera.target[2]];
    const axis = normalize(sub(aim, key));
    const fu = new Float32Array(64);
    fu.set(m.viewProj, 0);
    fu.set(this.lightMatrix(camera.target), 16);
    fu.set([...m.eye, camera.fovY / this.height], 32);
    // keyPos relative intensity normalisation uses |keyPos| (see keyLight); positions are in world space
    fu.set([...key, this.keyIntensity], 36);
    fu.set([...axis, 0.93], 40);
    fu.set([...this.ambient, (2 * SHADOW_HALF) / SHADOW_SIZE], 44);
    fu.set([this.width, this.height, 1 / this.width, 1 / this.height], 48);
    const kd = Math.hypot(key[0] - camera.target[0], key[1] - camera.target[1], key[2] - camera.target[2]);
    fu.set([this.exposure, time, kd, this.frameIndex++], 52);
    this.device.queue.writeBuffer(this.frameU, 0, fu);

    const N = this.scene.nodeCount;
    encoder.clearBuffer(this.lists, 0, 16);
    const cp = encoder.beginComputePass({ label: 'renderPrep', timestampWrites: tw(0) });
    cp.setBindGroup(0, this.refineGroup);
    const run = (name: string, groups: number) => {
      cp.setPipeline(this.cpipes[name]);
      cp.dispatchWorkgroups(groups);
    };
    run('nodeFrame', Math.ceil(N / 128));
    run('classifyCells', Math.ceil(N / 128));
    run('classifySpans', Math.ceil(N / 128));
    run('args', 1);
    run('refine', Math.ceil(this.refinedCount / 128));
    cp.end();

    const surfaceVerts = REFINE * REFINE * 6;
    const sp = encoder.beginRenderPass({
      label: 'shadow',
      timestampWrites: tw(1),
      colorAttachments: [],
      depthStencilAttachment: { view: this.shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    sp.setPipeline(this.pipes.surfaceShadow);
    sp.setBindGroup(0, this.surfaceShadowGroup);
    sp.draw(surfaceVerts, this.cellCount);
    sp.setBindGroup(0, this.yarnShadowGroup);
    sp.setPipeline(this.pipes.spanShadow);
    sp.drawIndirect(this.drawArgs, 0);
    sp.setPipeline(this.pipes.tailShadow);
    sp.drawIndirect(this.drawArgs, 16);
    sp.end();

    const mp = encoder.beginRenderPass({
      label: 'main',
      timestampWrites: tw(2),
      colorAttachments: [{ view: this.msaaColor.createView(), resolveTarget: this.hdr.createView(), clearValue: { r: 0.002, g: 0.002, b: 0.003, a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
      depthStencilAttachment: { view: this.msaaDepth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    mp.setPipeline(this.pipes.surface);
    mp.setBindGroup(0, this.surfaceGroup);
    mp.draw(surfaceVerts, this.cellCount);
    mp.setBindGroup(0, this.yarnGroup);
    mp.setPipeline(this.pipes.span);
    mp.drawIndirect(this.drawArgs, 0);
    mp.setPipeline(this.pipes.tail);
    mp.drawIndirect(this.drawArgs, 16);
    mp.setPipeline(this.pipes.yarnFuzz);
    mp.drawIndirect(this.drawArgs, 32);
    mp.setPipeline(this.pipes.tailFuzz);
    mp.drawIndirect(this.drawArgs, 48);
    mp.end();

    const pp = encoder.beginRenderPass({
      label: 'post',
      timestampWrites: tw(3),
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pp.setPipeline(this.pipes.post);
    pp.setBindGroup(0, this.postGroup);
    pp.draw(3);
    pp.end();
  }
}

export { SPAN_VERTS, TAIL_VERTS, FUZZ_VERTS };
