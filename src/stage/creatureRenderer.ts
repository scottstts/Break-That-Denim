import { createBuffer, createBufferWith, shaderModule, UNIFORM } from '../gpu/util';
import type { FloorImage, Renderer, SceneExtra } from '../rendering/renderer';
import mipgenWgsl from '../rendering/wgsl/mipgen.wgsl?raw';
import type { GltfSkinnedMesh } from './gltf';
import creatureWgsl from './wgsl/creature.wgsl?raw';

/** The creature's material images, decoded as stored (base colour, tangent-space normal, occlusion/roughness/metallic). */
export interface CreatureImages {
  base: FloorImage;
  normal: FloorImage;
  orm: FloorImage;
}

/**
 * GPU side of a staged skinned creature: its vertex data, material textures (mipmapped once), the
 * per-frame skinning matrices, and its draws in the renderer's two shadow passes and main pass
 * (Renderer.extras).
 */
export class CreatureRenderer implements SceneExtra {
  private readonly device: GPUDevice;
  private readonly vertex: GPUBuffer[];
  private readonly index: GPUBuffer;
  private readonly indexCount: number;
  private readonly jointBuf: GPUBuffer;
  private readonly uniform: GPUBuffer;
  private readonly textures: GPUTexture[];
  private readonly main: GPURenderPipeline;
  private readonly shadow: GPURenderPipeline;
  private readonly mainGroup: GPUBindGroup;
  private readonly keyGroup: GPUBindGroup;
  private readonly overGroup: GPUBindGroup;
  /** Drawn at all (the stage hides it outside its experience). */
  visible = false;

  constructor(renderer: Renderer, model: GltfSkinnedMesh, images: CreatureImages) {
    const device = renderer.device;
    this.device = device;
    const shared = renderer.shared();
    this.vertex = [
      createBufferWith(device, 'creature.pos', model.position, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST),
      createBufferWith(device, 'creature.nrm', model.normal, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST),
      createBufferWith(device, 'creature.tan', model.tangent, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST),
      createBufferWith(device, 'creature.uv', model.uv, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST),
      createBufferWith(device, 'creature.joints', model.jointIndex, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST),
      createBufferWith(device, 'creature.weights', model.weight, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST),
    ];
    this.index = createBufferWith(device, 'creature.index', model.index, GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST);
    this.indexCount = model.index.length;
    this.jointBuf = createBuffer(device, 'creature.skin', 64 * model.joints.length);
    this.uniform = createBuffer(device, 'creature.u', 32, UNIFORM);
    this.textures = [this.upload(images.base, 'rgba8unorm-srgb'), this.upload(images.normal, 'rgba8unorm'), this.upload(images.orm, 'rgba8unorm')];

    const module = shaderModule(device, 'creature', shared.common + shared.light + creatureWgsl);
    const F = GPUShaderStage.FRAGMENT;
    const V = GPUShaderStage.VERTEX;
    const layout = device.createBindGroupLayout({
      label: 'creature',
      entries: [
        { binding: 0, visibility: V | F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, buffer: { type: 'uniform' } },
        { binding: 2, visibility: V, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: F, texture: { sampleType: 'depth' } },
        { binding: 4, visibility: F, sampler: { type: 'comparison' } },
        { binding: 5, visibility: F, texture: { sampleType: 'depth' } },
        { binding: 6, visibility: F, texture: { sampleType: 'float' } },
        { binding: 7, visibility: F, texture: { sampleType: 'float' } },
        { binding: 8, visibility: F, texture: { sampleType: 'float' } },
        { binding: 9, visibility: F, sampler: { type: 'filtering' } },
      ],
    });
    const buffers: GPUVertexBufferLayout[] = [
      { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
      { arrayStride: 16, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x4' }] },
      { arrayStride: 8, attributes: [{ shaderLocation: 3, offset: 0, format: 'float32x2' }] },
      { arrayStride: 8, attributes: [{ shaderLocation: 4, offset: 0, format: 'uint16x4' }] },
      { arrayStride: 16, attributes: [{ shaderLocation: 5, offset: 0, format: 'float32x4' }] },
    ];
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const prim: GPUPrimitiveState = { topology: 'triangle-list', cullMode: 'none' };
    this.main = device.createRenderPipeline({
      label: 'creature',
      layout: pl,
      vertex: { module, entryPoint: 'vsMain', buffers },
      fragment: { module, entryPoint: 'fsMain', targets: [{ format: shared.hdrFormat }] },
      primitive: prim,
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' },
      multisample: { count: shared.samples },
    });
    this.shadow = device.createRenderPipeline({
      label: 'creature.shadow',
      layout: pl,
      vertex: { module, entryPoint: 'vsShadow', buffers },
      primitive: prim,
      depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less', depthBias: 2, depthBiasSlopeScale: 2.0 },
    });
    const sampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 8 });
    const group = (frame: GPUBuffer, shadow: GPUTextureView, over: GPUTextureView) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: frame } },
          { binding: 1, resource: { buffer: this.uniform } },
          { binding: 2, resource: { buffer: this.jointBuf } },
          { binding: 3, resource: shadow },
          { binding: 4, resource: shared.shadowSampler },
          { binding: 5, resource: over },
          { binding: 6, resource: this.textures[0].createView() },
          { binding: 7, resource: this.textures[1].createView() },
          { binding: 8, resource: this.textures[2].createView() },
          { binding: 9, resource: sampler },
        ],
      });
    this.mainGroup = group(shared.frameU, shared.shadowView, shared.overView);
    this.keyGroup = group(shared.frameU, shared.dummyDepthView, shared.dummyDepthView);
    this.overGroup = group(shared.frameOverU, shared.dummyDepthView, shared.dummyDepthView);
  }

  /** Mipmapped texture from a decoded image (box-filtered levels, linear for sRGB). */
  private upload(img: FloorImage, format: GPUTextureFormat): GPUTexture {
    const device = this.device;
    const levels = Math.floor(Math.log2(Math.max(img.width, img.height))) + 1;
    const t = device.createTexture({ label: 'creature.tex', size: [img.width, img.height], format, mipLevelCount: levels, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    if ('data' in img) device.queue.writeTexture({ texture: t }, img.data, { bytesPerRow: 4 * img.width }, [img.width, img.height]);
    else device.queue.copyExternalImageToTexture({ source: img }, { texture: t }, [img.width, img.height]);
    const module = shaderModule(device, 'mipgen', mipgenWgsl);
    const pipe = device.createRenderPipeline({ label: 'creature.mips', layout: 'auto', vertex: { module, entryPoint: 'vsFull' }, fragment: { module, entryPoint: 'fsDown', targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
    const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const enc = device.createCommandEncoder({ label: 'creature.mips' });
    for (let k = 1; k < levels; k++) {
      const src = t.createView({ baseMipLevel: k - 1, mipLevelCount: 1 });
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: t.createView({ baseMipLevel: k, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      pass.setPipeline(pipe);
      pass.setBindGroup(0, device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: src }, { binding: 2, resource: linear }] }));
      pass.draw(3);
      pass.end();
    }
    device.queue.submit([enc.finish()]);
    return t;
  }

  /** This frame's skinning matrices (Rig.skin) and pit light (position, radiant intensity). */
  setFrame(skin: Float32Array, pit: readonly number[], pitColor: readonly number[]): void {
    this.device.queue.writeBuffer(this.jointBuf, 0, skin);
    this.device.queue.writeBuffer(this.uniform, 0, new Float32Array([pit[0], pit[1], pit[2], 0, pitColor[0], pitColor[1], pitColor[2], 0]));
  }

  private draw(pass: GPURenderPassEncoder, pipe: GPURenderPipeline, group: GPUBindGroup): void {
    if (!this.visible) return;
    pass.setPipeline(pipe);
    pass.setBindGroup(0, group);
    this.vertex.forEach((b, k) => pass.setVertexBuffer(k, b));
    pass.setIndexBuffer(this.index, 'uint32');
    pass.drawIndexed(this.indexCount);
  }

  drawShadow(pass: GPURenderPassEncoder, light: 'key' | 'over'): void {
    this.draw(pass, this.shadow, light === 'key' ? this.keyGroup : this.overGroup);
  }

  drawMain(pass: GPURenderPassEncoder): void {
    this.draw(pass, this.main, this.mainGroup);
  }
}
