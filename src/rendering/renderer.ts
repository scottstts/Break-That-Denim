import type { CompiledScene } from '../simulation/scene';
import { COTTON_FIBRE_DENSITY } from '../simulation/material';
import type { GpuSolver } from '../simulation/gpu/solver';
import { SELF_CONTACT, SelfContact } from '../simulation/gpu/selfcontact';
import { ComputeModule, createBuffer, createBufferWith, shaderModule, STORAGE, UNIFORM } from '../gpu/util';
import { cameraMatrices, type CameraState } from './camera';
import { lookAt, multiply, normalize, orthographic, sub, type Mat4, type Vec3 } from './math';
import commonWgsl from './wgsl/render_common.wgsl?raw';
import lightWgsl from './wgsl/render_light.wgsl?raw';
import refineWgsl from './wgsl/refine.wgsl?raw';
import surfaceWgsl from './wgsl/surface.wgsl?raw';
import materialMapWgsl from './wgsl/material_map.wgsl?raw';
import piecesWgsl from './wgsl/pieces.wgsl?raw';
import threadsWgsl from './wgsl/threads.wgsl?raw';
import threadGeomWgsl from './wgsl/thread_geom.wgsl?raw';
import threadPrepWgsl from './wgsl/thread_prep.wgsl?raw';
import threadTailWgsl from './wgsl/thread_tail.wgsl?raw';
import threadWgsl from './wgsl/thread.wgsl?raw';
import fuzzWgsl from './wgsl/fuzz.wgsl?raw';
import postWgsl from './wgsl/post.wgsl?raw';
import weaveTileWgsl from './wgsl/weave_tile.wgsl?raw';
import edgeFieldWgsl from './wgsl/edgefield.wgsl?raw';
import edgeRootsWgsl from './wgsl/edge_roots.wgsl?raw';
import floorWgsl from './wgsl/floor.wgsl?raw';
import coverageWgsl from './wgsl/coverage.wgsl?raw';
import mipgenWgsl from './wgsl/mipgen.wgsl?raw';
import mountWgsl from './wgsl/mount.wgsl?raw';

export { DEFAULT_CAMERA, cameraMatrices, type CameraState, type ViewMatrices } from './camera';

const SAMPLES = 4;
/** Bytes per layer in the RenderU uniform (render_common.wgsl LayerR). */
const LAYER_BYTES = 144;
/** Texels per side of the prefiltered weave tile (one twill repeat) at mip level 0. */
const TILE_SIZE = 128;
const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
const SHADOW_SIZE = 4096;
const SHADOW_HALF = 0.34;
/**
 * Depth range of the key's and the overhead light's shadow projections (m): from the key's, deep
 * enough to hold the mount wherever the floor's view of the key crosses it.
 */
const SHADOW_NEAR = 0.3;
const SHADOW_FAR = 1.2;
const OVER_NEAR = 0.7;
const OVER_FAR = 1.3;
/** Overhead light's shadow map: texels per side and half its extent (m), covering the visible floor and its fill's reach. */
const OVER_SIZE = 2048;
const OVER_HALF = 0.45;
/** Shadow-map texels per side of a floor coverage texel (coverage.wgsl). */
const COV_BLOCK = 4;
/**
 * Angular radius of the key (rad): the penumbra its cloth shadows use (3 texels over a nominal 2 mm
 * occluder gap, render_light.wgsl), i.e. a source about 0.33 m across at its distance.
 */
const KEY_ANGULAR_RADIUS = (3 * (2 * SHADOW_HALF)) / SHADOW_SIZE / 2e-3;
const FRAME_BYTES = 512;

/**
 * Opaque geometry staged into the scene beside the cloth (Renderer.extras): drawn into both lights'
 * shadow maps (the bound frame uniform carries that light's projection) and into the main pass.
 */
export interface SceneExtra {
  drawShadow(pass: GPURenderPassEncoder, light: 'key' | 'over'): void;
  drawMain(pass: GPURenderPassEncoder): void;
}

/** What staged geometry needs to shade in the scene's lighting (Renderer.shared). */
export interface SharedResources {
  /** Render prelude and common definitions, then the lighting functions (render_light.wgsl). */
  common: string;
  light: string;
  frameU: GPUBuffer;
  frameOverU: GPUBuffer;
  shadowView: GPUTextureView;
  overView: GPUTextureView;
  dummyDepthView: GPUTextureView;
  shadowSampler: GPUSampler;
  hdrFormat: GPUTextureFormat;
  samples: number;
}

/**
 * Camera-film treatment applied in the post pass (all zero: the product's plain tone map).
 * grade: amount of the staged grade, film grain, vignette; glitch: amount of signal corruption and its
 * seed (held for a few frames).
 */
export interface PostEffects {
  grade: number;
  grain: number;
  vignette: number;
  glitch: number;
  glitchSeed: number;
}

export const NO_POST_EFFECTS: PostEffects = { grade: 0, grain: 0, vignette: 0, glitch: 0, glitchSeed: 0 };

/** An image for the floor material: decoded RGBA8 bytes (headless) or a browser image bitmap. */
export type FloorImage = ImageBitmap | { width: number; height: number; data: Uint8Array };

/** The floor's scanned material set. */
export interface FloorImages {
  base: FloorImage;
  normal: FloorImage;
  roughness: FloorImage;
  ao: FloorImage;
}
/** Sub-quads per lattice cell edge of the bicubic surface. */
const REFINE = 2;
/** Vec4s per run of the per-frame thread geometry (render_common.wgsl RUN_GEO). */
const RUN_GEO = 9;
/** Loose-thread runs and their draw chunks (threads.wgsl). */
const MAX_RUNS = 65536;
const MAX_CHUNKS = 262144;
/** Apparent density of a freed cotton yarn: fibre density 1520 kg/m^3 at 0.4 packing (abraded and partly untwisted once out of the weave), kg/m^3. */
const LOOSE_YARN_DENSITY = 1520 * 0.4;
/** Cells drawn per crossing (refine.wgsl list from FINE_BASE). */
const MAX_FINE = 32768;
/** Nodes with a connection field (pieces.wgsl) and the free-edge crossings (edge fibres). */
const KAPPA_SLOTS = 16384;
const MAX_EDGES = 262144;
/** Edge-fibre strands per free-edge crossing, dealt round its open sides (fuzz.wgsl). */
const FIBERS_PER_EDGE = 12;
/** Free tails of threads held at one end (thread_prep.wgsl tailSolve): segments per tail, most per frame. */
const TAIL_SEGS = 16;
const MAX_TAILS = 8192;
/**
 * Connection of a crossing to a node from the ratio of rest to material path distance (pieces.wgsl):
 * a smoothstep from KAPPA_LO (cut off) to KAPPA_HI (joined). With the node about a cell from a crack,
 * the opening tapers to nothing over about a cell from the crack's tip.
 */
const KAPPA_LO = 0.25;
const KAPPA_HI = 0.75;
/** Edge field: one texel per yarn crossing (render_common.wgsl), distance and warp / pick presence. */
const EDGE_FORMAT: GPUTextureFormat = 'rgba16float';
/** Jump-flooding steps (crossings) of the edge field: enough to reach EDGE_RMAX across warps and picks. */
const JFA_STEPS = [16, 8, 4, 2, 1, 1];


/** Layout constants of the yarn-resolution passes, prepended to every render module. */
function prelude(scene: CompiledScene, faceHash: boolean): string {
  const bu = Math.max(...scene.layers.map((L) => L.bundle[0]));
  const bv = Math.max(...scene.layers.map((L) => L.bundle[1]));
  const RW = 2 * bu + 3;
  const RH = 2 * bv + 3;
  const lines = [
    ['KAPPA_TABLE', 'u32', '4u'],
    ['KAPPA_NONE', 'u32', '0xFFFFFFFFu'],
    ['KAPPA_ZERO', 'u32', '0xFFFFFFFEu'],
    ['KAPPA_SLOTS', 'u32', `${KAPPA_SLOTS}u`],
    ['KAPPA_DW', 'u32', `${4 * bu + 3}u`],
    ['KAPPA_DH', 'u32', `${4 * bv + 3}u`],
    ['KAPPA_DOMAIN', 'u32', `${(4 * bu + 3) * (4 * bv + 3)}u`],
    ['KAPPA_RW', 'u32', `${RW}u`],
    ['KAPPA_RH', 'u32', `${RH}u`],
    ['KAPPA_WORDS', 'u32', `${Math.ceil((RW * RH) / 4)}u`],
    ['KAPPA_LO', 'f32', `${KAPPA_LO}`],
    ['KAPPA_HI', 'f32', `${KAPPA_HI}`],
    ['FINE_BASE', 'u32', '4u'],
    ['MAX_FINE', 'u32', `${MAX_FINE}u`],
    ['FINE_QK', 'u32', `${bu + 1}u`],
    ['FINE_QM', 'u32', `${bv + 1}u`],
    ['MAX_EDGES', 'u32', `${MAX_EDGES}u`],
    ['FIBERS_PER_EDGE', 'u32', `${FIBERS_PER_EDGE}u`],
    ['FIBRE_LIN', 'f32', `${COTTON_FIBRE_DENSITY}`],
    ['TAIL_SEGS', 'u32', `${TAIL_SEGS}u`],
    ['TAIL_SEGS_1', 'u32', `${TAIL_SEGS + 1}u`],
    ['MAX_TAILS', 'u32', `${MAX_TAILS}u`],
    // the solver's self-contact face hash, which the tails collide against (absent in a layered scene)
    ['FACE_HASH', 'bool', `${faceHash}`],
    ['FACE_CELL', 'f32', `${SELF_CONTACT.cell}`],
    ['FACE_MASK', 'u32', `${(1 << SELF_CONTACT.tableBits) - 1}u`],
  ];
  return lines.map(([n, t, v]) => `const ${n}: ${t} = ${v};`).join('\n') + '\n';
}

/**
 * Renders the accepted mechanical state: intact cells as a bicubic surface with procedural woven
 * microstructure; near damage, the cloth per crossing, placed from the nodes its material is still
 * connected to, with loose threads and the free edges' fibre.
 */
export class Renderer {
  readonly device: GPUDevice;
  readonly format: GPUTextureFormat;
  readonly solver: GpuSolver;
  readonly scene: CompiledScene;
  /** Key light position relative to the camera target, its intensity, and exposure. */
  /** A broad key grazing in from the top of the frame, as in the reference: folds read as relief and
   * the light falls off toward the bottom. */
  keyOffset: Vec3 = [-0.05, 0.6, 0.3];
  keyIntensity = 6.0;
  exposure = 1.0;
  /** Sky/bounce fill (hemisphere top): shadowed cloth stays a dark blue, as in the reference. */
  ambient: Vec3 = [0.03, 0.035, 0.05];
  /**
   * Where the key's cone is aimed, relative to the camera target; the cosines where its soft edge
   * starts and where it reaches full strength; the share of light outside the cone (the fixture's
   * spill); its colour.
   */
  keyAim: Vec3 = [0.0, 0.12, 0];
  keyCone = 0.9;
  keyInner = 1.0;
  keyLeak = 0.05;
  keyColor: Vec3 = [1, 1, 1];
  /**
   * A light on the camera (a staged experience; off at zero intensity): position, aim, radiant intensity
   * (W/sr), cosines of its cone's soft edge start and full strength, colour. Unshadowed: it sits at the lens.
   */
  torchPos: Vec3 = [0, 0, 1];
  torchAim: Vec3 = [0, 0, -1];
  torchIntensity = 0;
  torchCone: [number, number] = [0.8, 0.95];
  torchColor: Vec3 = [1, 1, 1];
  /** Radiance behind the cloth (the void seen through openings). */
  background: Vec3 = [0.002, 0.002, 0.003];
  /**
   * The room's overhead lamp: direction toward it, irradiance on a surface facing it, colour, and
   * angular radius (rad). A dim fixture high above, tilted toward the top of the frame like the key:
   * light falling through an opening lands on the floor close beneath it.
   */
  overDir: Vec3 = normalize([-0.03, 0.17, 1]);
  overIntensity = 0.1;
  overColor: Vec3 = [1.0, 0.92, 0.82];
  overAngle = 0.065;
  /**
   * Floor beneath the cloth (m): parallel to the mount and below everything the cloth hangs to (pulled
   * flaps reach ~6 cm, a grip pushes 3 cm), so it never meets the cloth; and its material's texture
   * repeat (m, the scanned set's real size).
   */
  floorZ = -0.12;
  floorTile = 2.5;
  /** Texels per tile of the floor's base colour and normal maps (public/texture holds their centre crop). */
  floorDetailTexels = 8192;
  /** Draw the floor (a staged experience plays over the void). */
  showFloor = true;
  /**
   * Where the lights are placed and their shadow maps centred: null follows the camera's target (the
   * product's framing); a staged camera that moves pins them to the room instead.
   */
  lightAnchor: Vec3 | null = null;
  /** Opaque staged geometry (SceneExtra). */
  readonly extras: SceneExtra[] = [];
  /** Post-pass treatment (PostEffects). */
  post: PostEffects = { ...NO_POST_EFFECTS };
  /** Present the last rendered frame again through the post pass only (a frozen picture). */
  freeze = false;

  private readonly frameU: GPUBuffer;
  /** The frame uniform as the overhead light's shadow pass sees it (its projection in place of the key's). */
  private readonly frameOverU: GPUBuffer;
  private readonly overTex: GPUTexture;
  /** Floor coverage of the key's and the overhead light's shadow maps (coverage.wgsl), mipmapped. */
  private readonly keyCov: GPUTexture;
  private readonly overCov: GPUTexture;
  private readonly covGroups: { pipe: GPUComputePipeline; group: GPUBindGroup; size: number }[] = [];
  private floorGroup: GPUBindGroup | null = null;
  private readonly floorLayout: GPUBindGroupLayout;
  private readonly mountCorners: GPUBuffer;
  private readonly renderU: GPUBuffer;
  private readonly nrm: GPUBuffer;
  /** Per node: unit +u tangent with the node's wear in w, unit +v tangent. */
  private readonly tframe: GPUBuffer;
  private readonly cellFlags: GPUBuffer;
  private readonly lists: GPUBuffer;
  private readonly drawArgs: GPUBuffer;
  private readonly rsurf: GPUBuffer;
  private readonly cellCount: number;
  private readonly refinedCount: number;
  private readonly fieldCount: number;
  /** Field-derived state, rebuilt when the yarn field changed: its gate and indirect dispatches. */
  private readonly edgeSig: GPUBuffer;
  private readonly edgeArgs: GPUBuffer;
  /** Loose threads: per-crossing thread bits, run / chunk counts, runs, chunks, per-run geometry, draws. */
  private readonly threadBits: GPUBuffer;
  private readonly threadScratch: GPUBuffer;
  private readonly threadCount: GPUBuffer;
  private readonly runs: GPUBuffer;
  private readonly chunks: GPUBuffer;
  private readonly runGeo: GPUBuffer;
  private readonly threadDraw: GPUBuffer;
  /** Free tails of threads held at one end (thread_prep.wgsl tailSolve) and their count. */
  private readonly tails: GPUBuffer;
  private readonly tailCount: GPUBuffer;
  /** Material connectivity (pieces.wgsl): slots and connection fields, dirty cells, args, free edges. */
  private readonly kappa: GPUBuffer;
  private readonly cellDirty: GPUBuffer;
  private readonly piecesArgs: GPUBuffer;
  private readonly edgeList: GPUBuffer;
  private readonly edgeRoots: GPUBuffer;
  /** Edge field: distance from each rest-chart point to the nearest opening (edgefield.wgsl). */
  private readonly seedA: GPUBuffer;
  private readonly seedB: GPUBuffer;
  private readonly edgeTex: GPUTexture;
  private readonly edgeSampler: GPUSampler;
  private readonly edgeArgsGroup: GPUBindGroup;
  private readonly edgeGroups: GPUBindGroup[] = [];
  private readonly groups: Record<string, GPUBindGroup> = {};
  private readonly pipes: Record<string, GPURenderPipeline> = {};
  private readonly cpipes: Record<string, GPUComputePipeline> = {};
  private readonly postLayout: GPUBindGroupLayout;
  private readonly shadowTex: GPUTexture;
  private readonly weaveTile: GPUTexture;
  private readonly tileSampler: GPUSampler;
  private readonly shadowSampler: GPUSampler;
  private readonly dummyDepth: GPUTexture;
  /** Render prelude and common definitions of every render module. */
  private readonly common: string;
  private readonly postU: GPUBuffer;
  private postGroup: GPUBindGroup | null = null;
  private msaaColor: GPUTexture | null = null;
  private msaaDepth: GPUTexture | null = null;
  private hdr: GPUTexture | null = null;
  width = 1;
  height = 1;
  private frameIndex = 0;
  /** Edge-field texture: first row of each layer's region, and its size (texels). */
  private edgeRows: number[] = [];
  private edgeSize: [number, number] = [1, 1];

  constructor(device: GPUDevice, format: GPUTextureFormat, solver: GpuSolver, scene: CompiledScene) {
    this.device = device;
    this.format = format;
    this.solver = solver;
    this.scene = scene;
    const N = scene.nodeCount;
    // the free tails collide with the cloth through the solver's self-contact face hash
    const faceHash = solver.contactEnabled && solver.contact instanceof SelfContact ? solver.contact : null;
    const common = prelude(scene, faceHash !== null) + commonWgsl;
    this.common = common;

    let cells = 0;
    let refined = 0;
    for (const L of scene.layers) {
      cells += (L.nx - 1) * (L.ny - 1);
      refined += ((L.nx - 1) * REFINE + 1) * ((L.ny - 1) * REFINE + 1);
    }
    this.cellCount = cells;
    this.refinedCount = refined;
    this.fieldCount = scene.fieldDamage.length / 2;
    const bu = Math.max(...scene.layers.map((L) => L.bundle[0]));
    const bv = Math.max(...scene.layers.map((L) => L.bundle[1]));
    const kappaWords = Math.ceil(((2 * bu + 3) * (2 * bv + 3)) / 4);

    // edge field layout (needed by renderParams)
    let edgeW = 1;
    let edgeH = 0;
    this.edgeRows = scene.field.map((F) => {
      const row = edgeH;
      edgeW = Math.max(edgeW, F.K);
      edgeH += F.M;
      return row;
    });
    this.edgeSize = [edgeW, edgeH];

    this.frameU = createBuffer(device, 'frameU', FRAME_BYTES, UNIFORM);
    this.frameOverU = createBuffer(device, 'frameOverU', FRAME_BYTES, UNIFORM);
    this.renderU = createBufferWith(device, 'renderU', new Uint8Array(this.renderParams()), UNIFORM);
    const nInit = new Float32Array(4 * N);
    for (let n = 0; n < N; n++) nInit[4 * n + 2] = 1;
    this.nrm = createBufferWith(device, 'nrm', nInit);
    const fInit = new Float32Array(8 * N);
    for (let n = 0; n < N; n++) {
      fInit[8 * n] = 1;
      fInit[8 * n + 3] = scene.wear[n];
      fInit[8 * n + 5] = 1;
    }
    this.tframe = createBufferWith(device, 'tframe', fInit);
    this.cellFlags = createBuffer(device, 'cellFlags', 4 * cells);
    this.lists = createBuffer(device, 'lists', 4 * (4 + MAX_FINE));
    this.drawArgs = createBuffer(device, 'drawArgs', 96, STORAGE | GPUBufferUsage.INDIRECT);
    this.rsurf = createBuffer(device, 'rsurf', 48 * refined);
    this.edgeSig = createBuffer(device, 'edgeSig', 16);
    this.edgeArgs = createBuffer(device, 'edgeArgs', 64, STORAGE | GPUBufferUsage.INDIRECT);
    this.seedA = createBuffer(device, 'edge.seedA', 4 * this.fieldCount);
    this.seedB = createBuffer(device, 'edge.seedB', 4 * this.fieldCount);
    // 2 bits per crossing (threads, absent material), then 1 bit per crossing (falling pieces)
    this.threadBits = createBuffer(device, 'threadBits', 4 * (Math.ceil(this.fieldCount / 16) + Math.ceil(this.fieldCount / 32) + 1));
    this.threadScratch = createBuffer(device, 'threadScratch', 4 * (2 * Math.ceil(this.fieldCount / 16) + Math.ceil(this.fieldCount / 32)));
    this.threadCount = createBuffer(device, 'threadCount', 16);
    this.tails = createBuffer(device, 'threadTails', 16 * (TAIL_SEGS + 1) * MAX_TAILS);
    this.tailCount = createBuffer(device, 'threadTailCount', 16);
    this.runs = createBuffer(device, 'threadRuns', 16 * MAX_RUNS);
    this.chunks = createBuffer(device, 'threadChunks', 4 * MAX_CHUNKS);
    this.runGeo = createBuffer(device, 'threadRunGeo', RUN_GEO * 16 * MAX_RUNS);
    this.threadDraw = createBuffer(device, 'threadDraw', 48, STORAGE | GPUBufferUsage.INDIRECT);
    // slot table initialised to "no field" so the first frame before a rebuild draws everything intact
    // header, node table, slot nodes, slot fields, slot hashes, persistent node slots (pieces.wgsl)
    const kappaInit = new Uint32Array(4 + N + KAPPA_SLOTS + KAPPA_SLOTS * kappaWords + KAPPA_SLOTS + N);
    kappaInit.fill(0xffffffff, 4, 4 + N);
    this.kappa = createBufferWith(device, 'kappa', kappaInit);
    this.cellDirty = createBuffer(device, 'cellDirty', 4 * cells);
    this.piecesArgs = createBuffer(device, 'piecesArgs', 64, STORAGE | GPUBufferUsage.INDIRECT);
    this.edgeList = createBuffer(device, 'edgeList', 4 * (MAX_EDGES + 1));
    this.edgeRoots = createBuffer(device, 'edgeRoots', 48 * MAX_EDGES);

    // ---------------------------------------------------------------- compute modules
    const refine = new ComputeModule(device, 'refine', common + refineWgsl);
    for (const e of ['nodeFrame', 'classifyCells', 'args', 'refine']) this.cpipes[e] = refine.pipeline(e);
    this.groups.refine = refine.bindGroup('refine', {
      0: this.renderU,
      1: solver.nodes,
      2: solver.x,
      3: solver.segs,
      5: this.nrm,
      6: this.cellFlags,
      7: this.lists,
      8: this.rsurf,
      9: this.drawArgs,
      10: this.tframe,
      11: this.kappa,
    });
    const threads = new ComputeModule(device, 'threads', common + threadsWgsl);
    for (const e of ['retireCount', 'threadClear', 'threadPresence', 'threadAnchor', 'threadCloth', 'threadScan', 'threadArgs']) this.cpipes[e] = threads.pipeline(e);
    this.groups.threads = threads.bindGroup('threads', {
      0: this.renderU,
      1: solver.field,
      2: this.threadBits,
      3: this.threadCount,
      4: this.runs,
      5: this.chunks,
      6: this.threadDraw,
      7: solver.x,
      8: solver.segs,
      9: this.threadScratch,
    });
    const pieces = new ComputeModule(device, 'pieces', common + piecesWgsl);
    for (const e of ['piecesClear', 'cellDirtyMain', 'nodeAlloc', 'piecesArgsMain', 'kappaSolve', 'piecesArgsReset', 'edgeListMain', 'edgeDrawArgs']) this.cpipes[e] = pieces.pipeline(e);
    this.groups.pieces = pieces.bindGroup('pieces', {
      0: this.renderU,
      1: solver.field,
      2: this.threadBits,
      3: solver.nodes,
      4: solver.x,
      5: this.kappa,
      6: this.cellDirty,
      7: this.piecesArgs,
      8: this.edgeList,
    });
    // kappaSolve is dispatched from piecesArgs, which it must not have bound for writing
    this.groups.piecesSolve = pieces.bindGroup('pieces.solve', { 0: this.renderU, 1: solver.field, 2: this.threadBits, 3: solver.nodes, 4: solver.x, 5: this.kappa });
    const edgeRoots = new ComputeModule(device, 'edgeRoots', common + materialMapWgsl + edgeRootsWgsl);
    this.cpipes.edgeRootsMain = edgeRoots.pipeline('edgeRootsMain');
    this.groups.edgeRoots = edgeRoots.bindGroup('edgeRoots', { 0: this.renderU, 1: solver.x, 2: this.tframe, 3: this.rsurf, 4: this.kappa, 5: this.edgeList, 6: this.edgeRoots });
    const threadPrep = new ComputeModule(device, 'threadPrep', common + threadGeomWgsl + materialMapWgsl + threadPrepWgsl);
    this.cpipes.threadPrep = threadPrep.pipeline('threadPrep');
    this.groups.threadPrep = threadPrep.bindGroup('threadPrep', {
      0: this.renderU,
      1: solver.x,
      2: solver.segs,
      3: this.tframe,
      4: this.nrm,
      5: this.runs,
      6: this.threadCount,
      7: this.runGeo,
      8: this.rsurf,
      9: this.kappa,
    });
    const threadTail = new ComputeModule(device, 'threadTail', common + threadTailWgsl);
    this.cpipes.threadTail = threadTail.pipeline('threadTail');
    this.groups.threadTail = threadTail.bindGroup('threadTail', {
      0: this.renderU,
      1: solver.x,
      2: this.runs,
      3: this.threadCount,
      4: this.runGeo,
      5: this.tails,
      6: this.tailCount,
      7: solver.nodes,
      8: faceHash ? faceHash.cells : createBuffer(device, 'noFaceCells', 16),
      9: faceHash ? faceHash.items : createBuffer(device, 'noFaceItems', 32),
    });

    // edge field
    this.edgeTex = device.createTexture({ label: 'edgeField', size: [edgeW, edgeH], format: EDGE_FORMAT, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    this.edgeSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    const C = GPUShaderStage.COMPUTE;
    const edgeLayout = device.createBindGroupLayout({
      label: 'edgeField',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, buffer: { type: 'uniform' } },
        { binding: 2, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: C, buffer: { type: 'storage' } },
        { binding: 5, visibility: C, storageTexture: { access: 'write-only', format: EDGE_FORMAT } },
        { binding: 6, visibility: C, buffer: { type: 'storage' } },
        { binding: 7, visibility: C, buffer: { type: 'storage' } },
        { binding: 8, visibility: C, buffer: { type: 'storage' } },
        { binding: 9, visibility: C, buffer: { type: 'read-only-storage' } },
        { binding: 10, visibility: C, buffer: { type: 'read-only-storage' } },
      ],
    });
    const edgeModule = shaderModule(device, 'edgeField', common + edgeFieldWgsl);
    const edgePL = device.createPipelineLayout({ bindGroupLayouts: [edgeLayout] });
    for (const e of ['edgeArgsMain', 'seedInit', 'seedStep', 'edgeWrite']) {
      this.cpipes[e] = device.createComputePipeline({ label: `edgeField.${e}`, layout: edgePL, compute: { module: edgeModule, entryPoint: e } });
    }
    const texView = this.edgeTex.createView();
    const edgeBind = (label: string, step: number, seedIn: GPUBuffer, seedOut: GPUBuffer, args: GPUBuffer, events: GPUBuffer, sig: GPUBuffer) =>
      device.createBindGroup({
        label,
        layout: edgeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.renderU } },
          { binding: 1, resource: { buffer: createBufferWith(device, `${label}.step`, new Uint32Array([step, 0, 0, 0]), UNIFORM) } },
          { binding: 2, resource: { buffer: solver.field } },
          { binding: 3, resource: { buffer: seedIn } },
          { binding: 4, resource: { buffer: seedOut } },
          { binding: 5, resource: texView },
          { binding: 6, resource: { buffer: sig } },
          { binding: 7, resource: { buffer: events } },
          { binding: 8, resource: { buffer: args } },
          { binding: 9, resource: { buffer: this.threadBits } },
          { binding: 10, resource: { buffer: this.threadCount } },
        ],
      });
    const scratch = (label: string) => createBuffer(device, label, 64);
    this.edgeArgsGroup = edgeBind('edgeField.args', 0, this.seedB, this.seedA, this.edgeArgs, solver.fieldEvents, this.edgeSig);
    // seeds into A; steps alternate A -> B -> A ...; the final write reads the last output
    this.edgeGroups.push(edgeBind('edgeField.init', 0, this.seedB, this.seedA, scratch('edgeField.dA'), scratch('edgeField.dE'), scratch('edgeField.dS')));
    JFA_STEPS.forEach((st, k) => {
      const [a, b] = k % 2 === 0 ? [this.seedA, this.seedB] : [this.seedB, this.seedA];
      this.edgeGroups.push(edgeBind(`edgeField.step${k}`, st, a, b, scratch(`edgeField.dA${k}`), scratch(`edgeField.dE${k}`), scratch(`edgeField.dS${k}`)));
    });
    const last = JFA_STEPS.length % 2 === 0 ? this.seedA : this.seedB;
    this.edgeGroups.push(edgeBind('edgeField.write', 0, last, last === this.seedA ? this.seedB : this.seedA, scratch('edgeField.dAw'), scratch('edgeField.dEw'), scratch('edgeField.dSw')));

    // ---------------------------------------------------------------- render pipelines
    this.shadowTex = device.createTexture({ label: 'shadowMap', size: [SHADOW_SIZE, SHADOW_SIZE], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.dummyDepth = device.createTexture({ size: [1, 1], format: 'depth32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
    this.shadowSampler = device.createSampler({ compare: 'less', magFilter: 'linear', minFilter: 'linear' });
    this.overTex = device.createTexture({ label: 'overShadowMap', size: [OVER_SIZE, OVER_SIZE], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    const covTex = (label: string, size: number) =>
      device.createTexture({ label, size: [size, size], format: 'rgba16float', mipLevelCount: Math.log2(size) + 1, usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    this.keyCov = covTex('keyCoverage', SHADOW_SIZE / COV_BLOCK);
    this.overCov = covTex('overCoverage', OVER_SIZE / COV_BLOCK);
    this.buildCoverage([
      [this.shadowTex, this.keyCov],
      [this.overTex, this.overCov],
    ]);
    this.weaveTile = this.buildWeaveTile(common);
    this.tileSampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 16 });

    const V = GPUShaderStage.VERTEX;
    const F = GPUShaderStage.FRAGMENT;
    const VF = V | F;
    const ro = (binding: number, visibility = V): GPUBindGroupLayoutEntry => ({ binding, visibility, buffer: { type: 'read-only-storage' } });
    const uni = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: VF, buffer: { type: 'uniform' } });
    const shadowEntries = (b: number, visibility = F): GPUBindGroupLayoutEntry[] => [
      { binding: b, visibility, texture: { sampleType: 'depth' } },
      { binding: b + 1, visibility, sampler: { type: 'comparison' } },
    ];
    const surfaceLayout = device.createBindGroupLayout({
      label: 'surface',
      entries: [
        uni(0),
        uni(1),
        ro(2),
        ro(3),
        ...shadowEntries(4),
        { binding: 6, visibility: F, texture: { sampleType: 'float', viewDimension: '2d-array' } },
        { binding: 7, visibility: F, sampler: { type: 'filtering' } },
        ro(8),
        { binding: 9, visibility: F, texture: { sampleType: 'float' } },
        { binding: 10, visibility: F, sampler: { type: 'filtering' } },
        // the yarn damage field and loose-thread bits: per sample, and for the fine quads' links
        ro(11, VF),
        ro(12, VF),
        ro(13),
        ro(14),
        ro(15),
        { binding: 16, visibility: F, texture: { sampleType: 'depth' } },
      ],
    });
    const fuzzLayout = device.createBindGroupLayout({
      label: 'fuzz',
      // fibre strands are lit per vertex
      entries: [uni(0), uni(1), ro(2), ro(3), ro(4), ...shadowEntries(9, VF), { binding: 11, visibility: VF, texture: { sampleType: 'depth' } }],
    });
    const threadLayout = device.createBindGroupLayout({
      label: 'thread',
      entries: [uni(0), uni(1), ro(2), ro(3), ro(4), ro(5), ro(6), ro(7), ro(8), ...shadowEntries(9), ro(11), { binding: 12, visibility: F, texture: { sampleType: 'depth' } }],
    });

    const depth: GPUDepthStencilState = { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less' };
    const shadowDepth: GPUDepthStencilState = { ...depth, depthBias: 2, depthBiasSlopeScale: 2.0 };
    const ms: GPUMultisampleState = { count: SAMPLES };
    const coverage: GPUMultisampleState = { count: SAMPLES, alphaToCoverageEnabled: true };
    const strip: GPUPrimitiveState = { topology: 'triangle-strip', cullMode: 'none' };
    const list: GPUPrimitiveState = { topology: 'triangle-list', cullMode: 'none' };
    // fibres far thinner than a pixel cover a fraction of it: blended over what is behind (depth
    // tested, not written; a crossing's edge-fibre strands are drawn back to front, fuzz.wgsl).
    // Alpha to coverage would quantise their sub-pixel coverage to 0 or 1 of 4 samples (speckle).
    const blended: GPUBlendState = { color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' }, alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha' } };
    const pipe = (label: string, layout: GPUBindGroupLayout, module: GPUShaderModule, vs: string, fs: string | null, prim: GPUPrimitiveState, msaa: GPUMultisampleState | null, blend?: GPUBlendState) =>
      device.createRenderPipeline({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: vs },
        fragment: fs ? { module, entryPoint: fs, targets: [{ format: HDR_FORMAT, blend }] } : undefined,
        primitive: prim,
        depthStencil: msaa ? (blend ? { ...depth, depthWriteEnabled: false } : depth) : shadowDepth,
        multisample: msaa ?? undefined,
      });
    const surfaceModule = shaderModule(device, 'surface', common + lightWgsl + materialMapWgsl + surfaceWgsl);
    this.pipes.surface = pipe('surface', surfaceLayout, surfaceModule, 'vsMain', 'fsMain', list, ms);
    this.pipes.surfaceShadow = pipe('surface.shadow', surfaceLayout, surfaceModule, 'vsShadow', null, list, null);
    this.pipes.fine = pipe('fine', surfaceLayout, surfaceModule, 'vsFine', 'fsMain', list, ms);
    this.pipes.fineShadow = pipe('fine.shadow', surfaceLayout, surfaceModule, 'vsFineShadow', null, list, null);
    const fuzzModule = shaderModule(device, 'fuzz', common + lightWgsl + fuzzWgsl);
    this.pipes.edgeFuzz = pipe('edgeFuzz', fuzzLayout, fuzzModule, 'vsEdgeFuzz', 'fsFuzz', strip, ms, blended);
    const threadModule = shaderModule(device, 'thread', common + lightWgsl + threadGeomWgsl + threadWgsl);
    this.pipes.thread = pipe('thread', threadLayout, threadModule, 'vsThread', 'fsThread', strip, coverage);
    this.pipes.threadShadow = pipe('thread.shadow', threadLayout, threadModule, 'vsThreadShadow', null, strip, null);
    this.pipes.threadFuzz = pipe('threadFuzz', threadLayout, threadModule, 'vsThreadFuzz', 'fsThreadFuzz', strip, ms, blended);
    const tex = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: F, texture: { sampleType: 'float' } });
    this.floorLayout = device.createBindGroupLayout({
      label: 'floor',
      entries: [uni(0), tex(1), tex(2), tex(3), { binding: 4, visibility: F, sampler: { type: 'filtering' } }, tex(5), tex(6), { binding: 7, visibility: F, sampler: { type: 'filtering' } }, { binding: 11, visibility: F, sampler: { type: 'filtering' } }],
    });
    const floorModule = shaderModule(device, 'floor', common + lightWgsl + floorWgsl);
    this.pipes.floor = pipe('floor', this.floorLayout, floorModule, 'vsFloor', 'fsFloor', list, ms);
    const mountLayout = device.createBindGroupLayout({ label: 'mount', entries: [uni(0), ro(1), ...shadowEntries(2), { binding: 4, visibility: F, texture: { sampleType: 'depth' } }] });
    const mountModule = shaderModule(device, 'mount', common + lightWgsl + mountWgsl);
    this.pipes.mount = pipe('mount', mountLayout, mountModule, 'vsMount', 'fsMount', list, ms);
    this.pipes.mountShadow = pipe('mount.shadow', mountLayout, mountModule, 'vsMountShadow', null, list, null);
    this.mountCorners = createBufferWith(device, 'mountCorners', this.mountGeometry());

    const tileView = this.weaveTile.createView({ dimension: '2d-array' });
    const edgeView = this.edgeTex.createView();
    const shadowView = this.shadowTex.createView();
    const dummyView = this.dummyDepth.createView();
    const overView = this.overTex.createView();
    // the shaded passes read both shadow maps; the two shadow passes (each writing one) neither
    for (const [suffix, shadow, over, frame] of [
      ['', shadowView, overView, this.frameU],
      ['Shadow', dummyView, dummyView, this.frameU],
      ['Over', dummyView, dummyView, this.frameOverU],
    ] as const) {
      this.groups[`surface${suffix}`] = device.createBindGroup({
        layout: surfaceLayout,
        entries: [
          { binding: 0, resource: { buffer: frame } },
          { binding: 1, resource: { buffer: this.renderU } },
          { binding: 2, resource: { buffer: this.rsurf } },
          { binding: 3, resource: { buffer: this.cellFlags } },
          { binding: 4, resource: shadow },
          { binding: 5, resource: this.shadowSampler },
          { binding: 6, resource: tileView },
          { binding: 7, resource: this.tileSampler },
          { binding: 8, resource: { buffer: this.lists } },
          { binding: 9, resource: edgeView },
          { binding: 10, resource: this.edgeSampler },
          { binding: 11, resource: { buffer: solver.field } },
          { binding: 12, resource: { buffer: this.threadBits } },
          { binding: 13, resource: { buffer: solver.x } },
          { binding: 14, resource: { buffer: this.tframe } },
          { binding: 15, resource: { buffer: this.kappa } },
          { binding: 16, resource: over },
        ],
      });
      this.groups[`thread${suffix}`] = device.createBindGroup({
        layout: threadLayout,
        entries: [
          { binding: 0, resource: { buffer: frame } },
          { binding: 1, resource: { buffer: this.renderU } },
          { binding: 2, resource: { buffer: solver.x } },
          { binding: 3, resource: { buffer: solver.segs } },
          { binding: 4, resource: { buffer: this.tframe } },
          { binding: 5, resource: { buffer: this.nrm } },
          { binding: 6, resource: { buffer: this.runs } },
          { binding: 7, resource: { buffer: this.chunks } },
          { binding: 8, resource: { buffer: this.runGeo } },
          { binding: 11, resource: { buffer: this.tails } },
          { binding: 9, resource: shadow },
          { binding: 10, resource: this.shadowSampler },
          { binding: 12, resource: over },
        ],
      });
    }
    for (const [name, frame, shadow, over] of [
      ['mount', this.frameU, shadowView, overView],
      ['mountShadow', this.frameU, dummyView, dummyView],
      ['mountOver', this.frameOverU, dummyView, dummyView],
    ] as const) {
      this.groups[name] = device.createBindGroup({
        layout: this.pipes.mount.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: frame } },
          { binding: 1, resource: { buffer: this.mountCorners } },
          { binding: 2, resource: shadow },
          { binding: 3, resource: this.shadowSampler },
          { binding: 4, resource: over },
        ],
      });
    }
    this.groups.fuzz = device.createBindGroup({
      layout: fuzzLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameU } },
        { binding: 1, resource: { buffer: this.renderU } },
        { binding: 2, resource: { buffer: this.edgeList } },
        { binding: 3, resource: { buffer: this.edgeRoots } },
        { binding: 4, resource: { buffer: solver.field } },
        { binding: 9, resource: shadowView },
        { binding: 10, resource: this.shadowSampler },
        { binding: 11, resource: overView },
      ],
    });

    const postModule = shaderModule(device, 'post', common + postWgsl);
    this.postLayout = device.createBindGroupLayout({
      label: 'post',
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, texture: { sampleType: 'unfilterable-float' } },
        { binding: 2, visibility: F, buffer: { type: 'uniform' } },
      ],
    });
    this.postU = createBuffer(device, 'postU', 32, UNIFORM);
    this.pipes.post = device.createRenderPipeline({
      label: 'post',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.postLayout] }),
      vertex: { module: postModule, entryPoint: 'vsFull' },
      fragment: { module: postModule, entryPoint: 'fsTonemap', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  /** Shader prelude, frame uniforms and shadow maps for staged geometry (SceneExtra). */
  shared(): SharedResources {
    return {
      common: this.common,
      light: lightWgsl,
      frameU: this.frameU,
      frameOverU: this.frameOverU,
      shadowView: this.shadowTex.createView(),
      overView: this.overTex.createView(),
      dummyDepthView: this.dummyDepth.createView(),
      shadowSampler: this.shadowSampler,
      hdrFormat: HDR_FORMAT,
      samples: SAMPLES,
    };
  }

  /** Prefiltered weave tile per layer: every mip level averaged analytically (weave_tile.wgsl). */
  private buildWeaveTile(common: string): GPUTexture {
    const layers = this.scene.layers.length;
    const levels = Math.log2(TILE_SIZE) + 1;
    const tex = this.device.createTexture({
      label: 'weaveTile',
      size: [TILE_SIZE, TILE_SIZE, layers],
      format: 'rgba16float',
      mipLevelCount: levels,
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const module = shaderModule(this.device, 'weaveTile', common + weaveTileWgsl);
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

  /**
   * The mount (mount.wgsl): inner corners under the pinned collar (the second node in from each corner
   * of the first layer), just beneath the cloth, and outer corners far beyond the view.
   */
  private mountGeometry(): Float32Array {
    const L = this.scene.layers[0];
    const p = this.scene.position;
    const node = (i: number, j: number) => L.nodeOffset + j * L.nx + i;
    const inner = [node(1, 1), node(L.nx - 2, 1), node(L.nx - 2, L.ny - 2), node(1, L.ny - 2)];
    const z = p[3 * inner[0] + 2] - 0.5 * L.weave.thickness;
    const out = new Float32Array(32);
    inner.forEach((n, k) => out.set([p[3 * n], p[3 * n + 1], z, 0], 4 * k));
    const R = 1.5;
    [[-R, -R], [R, -R], [R, R], [-R, R]].forEach(([x, y], k) => out.set([x, y, z, 0], 16 + 4 * k));
    return out;
  }

  /** Coverage passes (coverage.wgsl) of each [shadow map, coverage texture]: level 0 from depth, then the mip chain. */
  private buildCoverage(maps: [GPUTexture, GPUTexture][]): void {
    const module = shaderModule(this.device, 'coverage', coverageWgsl);
    const C = GPUShaderStage.COMPUTE;
    const layout = this.device.createBindGroupLayout({
      label: 'coverage',
      entries: [
        { binding: 0, visibility: C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: C, texture: { sampleType: 'depth' } },
        { binding: 2, visibility: C, texture: { sampleType: 'unfilterable-float' } },
        { binding: 3, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
        { binding: 4, visibility: C, sampler: { type: 'non-filtering' } },
      ],
    });
    const gather = this.device.createSampler({ label: 'coverage.gather' });
    const pl = this.device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const fromDepth = this.device.createComputePipeline({ label: 'coverage.fromDepth', layout: pl, compute: { module, entryPoint: 'fromDepth' } });
    const down = this.device.createComputePipeline({ label: 'coverage.down', layout: pl, compute: { module, entryPoint: 'down' } });
    const blockU = createBufferWith(this.device, 'coverage.block', new Uint32Array([COV_BLOCK, 0, 0, 0]), UNIFORM);
    const unusedDepth = this.device.createTexture({ size: [1, 1], format: 'depth32float', usage: GPUTextureUsage.TEXTURE_BINDING });
    const unusedCov = this.device.createTexture({ size: [1, 1], format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING });
    for (const [depth, cov] of maps) {
      for (let k = 0; k < cov.mipLevelCount; k++) {
        const level = (m: number) => cov.createView({ baseMipLevel: m, mipLevelCount: 1 });
        const group = this.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: blockU } },
            { binding: 1, resource: (k === 0 ? depth : unusedDepth).createView() },
            { binding: 2, resource: k === 0 ? unusedCov.createView() : level(k - 1) },
            { binding: 3, resource: level(k) },
            { binding: 4, resource: gather },
          ],
        });
        this.covGroups.push({ pipe: k === 0 ? fromDepth : down, group, size: cov.width >> k });
      }
    }
  }

  /** Upload the floor's material set (mipmapped, roughness and AO packed); the floor is drawn from then on. */
  setFloorImages(images: FloorImages): void {
    const device = this.device;
    const levelsOf = (img: FloorImage) => Math.floor(Math.log2(Math.max(img.width, img.height))) + 1;
    const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
    const upload = (label: string, img: FloorImage, format: GPUTextureFormat, mipLevelCount: number) => {
      const t = device.createTexture({ label, size: [img.width, img.height], format, mipLevelCount, usage });
      if ('data' in img) device.queue.writeTexture({ texture: t }, img.data, { bytesPerRow: 4 * img.width }, [img.width, img.height]);
      else device.queue.copyExternalImageToTexture({ source: img }, { texture: t }, [img.width, img.height]);
      return t;
    };
    const base = upload('floor.base', images.base, 'rgba8unorm-srgb', levelsOf(images.base));
    const normal = upload('floor.normal', images.normal, 'rgba8unorm', levelsOf(images.normal));
    const rough = upload('floor.roughness', images.roughness, 'rgba8unorm', 1);
    const ao = upload('floor.ao', images.ao, 'rgba8unorm', 1);
    const ra = device.createTexture({ label: 'floor.ra', size: [rough.width, rough.height], format: 'rg8unorm', mipLevelCount: levelsOf(images.roughness), usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });

    const module = shaderModule(device, 'mipgen', mipgenWgsl);
    const layout = device.createBindGroupLayout({
      label: 'mipgen',
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    const pl = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipes = new Map<string, GPURenderPipeline>();
    const pipeFor = (format: GPUTextureFormat, entryPoint: string) => {
      const key = `${format}.${entryPoint}`;
      let p = pipes.get(key);
      if (!p) {
        p = device.createRenderPipeline({ label: `mipgen.${key}`, layout: pl, vertex: { module, entryPoint: 'vsFull' }, fragment: { module, entryPoint, targets: [{ format }] }, primitive: { topology: 'triangle-list' } });
        pipes.set(key, p);
      }
      return p;
    };
    const linear = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const enc = device.createCommandEncoder({ label: 'floor.mips' });
    const blit = (t: GPUTexture, level: number, entryPoint: string, a: GPUTextureView, b: GPUTextureView) => {
      const pass = enc.beginRenderPass({ colorAttachments: [{ view: t.createView({ baseMipLevel: level, mipLevelCount: 1 }), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
      pass.setPipeline(pipeFor(t.format, entryPoint));
      pass.setBindGroup(0, device.createBindGroup({ layout, entries: [{ binding: 0, resource: a }, { binding: 1, resource: b }, { binding: 2, resource: linear }] }));
      pass.draw(3);
      pass.end();
    };
    blit(ra, 0, 'fsPack', rough.createView(), ao.createView());
    for (const t of [base, normal, ra]) {
      for (let k = 1; k < t.mipLevelCount; k++) {
        const src = t.createView({ baseMipLevel: k - 1, mipLevelCount: 1 });
        blit(t, k, 'fsDown', src, src);
      }
    }
    device.queue.submit([enc.finish()]);
    void device.queue.onSubmittedWorkDone().then(() => {
      rough.destroy();
      ao.destroy();
    });
    const matSampler = device.createSampler({ addressModeU: 'repeat', addressModeV: 'repeat', magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 16 });
    const covSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear' });
    const detailSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', mipmapFilter: 'linear', maxAnisotropy: 16 });
    this.floorGroup = device.createBindGroup({
      label: 'floor',
      layout: this.floorLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameU } },
        { binding: 1, resource: base.createView() },
        { binding: 2, resource: normal.createView() },
        { binding: 3, resource: ra.createView() },
        { binding: 4, resource: matSampler },
        { binding: 5, resource: this.keyCov.createView() },
        { binding: 6, resource: this.overCov.createView() },
        { binding: 7, resource: covSampler },
        { binding: 11, resource: detailSampler },
      ],
    });
  }

  private renderParams(): ArrayBuffer {
    const buf = new ArrayBuffer(16 + 3 * LAYER_BYTES);
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
      const b = (16 + LAYER_BYTES * l) / 4;
      const m = L.weave;
      u[b] = L.nodeOffset;
      u[b + 1] = L.nx;
      u[b + 2] = L.ny;
      // yarns per span (warp column, pick row), twice: threads of a loose bundle are its yarns
      u[b + 3] = L.bundle[0] | (L.bundle[1] << 8) | (L.bundle[0] << 16) | (L.bundle[1] << 24);
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
      f.set([this.edgeRows[l] ?? 0, 0, this.edgeSize[0], this.edgeSize[1]], b + 24);
      const F = s.field[l];
      u.set([F.offset, F.K, F.M, 0], b + 28);
      // a freed yarn rounds up: radius from its linear density at loose packing; and its fibre count
      const looseR = (y: { linearDensity: number }) => Math.sqrt(y.linearDensity / (LOOSE_YARN_DENSITY * Math.PI));
      f.set([looseR(weft), looseR(warp), weft.linearDensity / COTTON_FIBRE_DENSITY, warp.linearDensity / COTTON_FIBRE_DENSITY], b + 32);
      cellOffset += (L.nx - 1) * (L.ny - 1);
      refinedOffset += rnx * ((L.ny - 1) * REFINE + 1);
    });
    return buf;
  }

  /** Rebuild field-derived state from scratch on the next frame (scene reset). */
  resetTopology(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.edgeSig);
    // connection slots and their hashes start over (the rebuild this triggers sets every node's entry)
    encoder.clearBuffer(this.kappa);
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
    this.postGroup = this.device.createBindGroup({
      layout: this.postLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frameU } },
        { binding: 1, resource: this.hdr.createView() },
        { binding: 2, resource: { buffer: this.postU } },
      ],
    });
  }

  private lightMatrix(target: Vec3): Mat4 {
    const key: Vec3 = [target[0] + this.keyOffset[0], target[1] + this.keyOffset[1], target[2] + this.keyOffset[2]];
    const dir = normalize(sub(key, target));
    const eye: Vec3 = [target[0] + dir[0] * 0.9, target[1] + dir[1] * 0.9, target[2] + dir[2] * 0.9];
    const view = lookAt(eye, target, Math.abs(dir[2]) > 0.95 ? [0, 1, 0] : [0, 0, 1]);
    const proj = orthographic(-SHADOW_HALF, SHADOW_HALF, -SHADOW_HALF, SHADOW_HALF, SHADOW_NEAR, SHADOW_FAR);
    return multiply(proj, view);
  }

  /** The overhead light's shadow projection: orthographic along its direction (it is distant). */
  private overMatrix(target: Vec3): Mat4 {
    const d = normalize(this.overDir);
    const eye: Vec3 = [target[0] + d[0], target[1] + d[1], target[2] + d[2]];
    const view = lookAt(eye, target, Math.abs(d[2]) > 0.95 ? [0, 1, 0] : [0, 0, 1]);
    return multiply(orthographic(-OVER_HALF, OVER_HALF, -OVER_HALF, OVER_HALF, OVER_NEAR, OVER_FAR), view);
  }


  /**
   * Encode the full frame for the current accepted state into `target`. `profile` (development)
   * receives begin/end timestamps of the prep, shadow, main and post passes (8 queries).
   */
  encode(encoder: GPUCommandEncoder, camera: CameraState, time: number, target: GPUTextureView, profile?: GPUQuerySet): void {
    const tw = (k: number) => (profile ? { querySet: profile, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } : undefined);
    if (!this.hdr || !this.msaaColor || !this.msaaDepth || !this.postGroup) return;
    const pe = this.post;
    this.device.queue.writeBuffer(this.postU, 0, new Float32Array([pe.grade, pe.grain, pe.vignette, time, pe.glitch, pe.glitchSeed, 0, 0]));
    if (this.freeze) {
      this.encodePost(encoder, target, tw(3));
      return;
    }
    const m = cameraMatrices(camera, this.width / this.height);
    const anchor = this.lightAnchor ?? camera.target;
    const key: Vec3 = [anchor[0] + this.keyOffset[0], anchor[1] + this.keyOffset[1], anchor[2] + this.keyOffset[2]];
    // aim the key slightly above the frame centre so illumination falls off toward the bottom
    const aim: Vec3 = [anchor[0] + this.keyAim[0], anchor[1] + this.keyAim[1], anchor[2] + this.keyAim[2]];
    const floor = this.floorGroup !== null && this.showFloor;
    const axis = normalize(sub(aim, key));
    const fu = new Float32Array(FRAME_BYTES / 4);
    fu.set(m.viewProj, 0);
    fu.set(this.lightMatrix(anchor), 16);
    fu.set([...m.eye, camera.fovY / this.height], 32);
    // keyPos relative intensity normalisation uses |keyPos| (see keyLight); positions are in world space
    fu.set([...key, this.keyIntensity], 36);
    fu.set([...axis, this.keyCone], 40);
    fu.set([...this.ambient, (2 * SHADOW_HALF) / SHADOW_SIZE], 44);
    fu.set([this.width, this.height, 1 / this.width, 1 / this.height], 48);
    const kd = Math.hypot(key[0] - anchor[0], key[1] - anchor[1], key[2] - anchor[2]);
    fu.set([this.exposure, time, kd, this.frameIndex++], 52);
    const over = this.overMatrix(anchor);
    const od = normalize(this.overDir);
    fu.set(over, 56);
    fu.set([...od, (2 * OVER_HALF) / OVER_SIZE], 72);
    fu.set([...this.overColor.map((c) => c * this.overIntensity), this.overAngle], 76);
    fu.set([this.floorZ, 1 / this.floorTile, floor ? 1 : 0, this.floorDetailTexels], 80);
    fu.set([KEY_ANGULAR_RADIUS, SHADOW_FAR - SHADOW_NEAR, 2 * SHADOW_HALF, SHADOW_SIZE / COV_BLOCK], 84);
    fu.set([this.overAngle, OVER_FAR - OVER_NEAR, 2 * OVER_HALF, OVER_SIZE / COV_BLOCK], 88);
    fu.set([this.keyInner, this.keyLeak, 0, 0], 92);
    fu.set([...this.keyColor, 0], 96);
    fu.set([...this.torchPos, this.torchIntensity], 100);
    fu.set([...this.torchAim, this.torchCone[0]], 104);
    fu.set([...this.torchColor, this.torchCone[1]], 108);
    this.device.queue.writeBuffer(this.frameU, 0, fu);
    // the overhead light's shadow pass: its projection in place of the key's, and the loose threads'
    // ribbons turned toward it (they face keyPos when drawn from the light)
    fu.set(over, 16);
    fu.set([anchor[0] + 10 * od[0], anchor[1] + 10 * od[1], anchor[2] + 10 * od[2]], 36);
    this.device.queue.writeBuffer(this.frameOverU, 0, fu);

    const N = this.scene.nodeCount;
    encoder.clearBuffer(this.lists, 0, 16);
    encoder.clearBuffer(this.threadCount, 8, 4);
    encoder.clearBuffer(this.tailCount);
    const cp = encoder.beginComputePass({ label: 'renderPrep', timestampWrites: tw(0) });
    const run = (name: string, groups: number) => {
      cp.setPipeline(this.cpipes[name]);
      cp.dispatchWorkgroups(groups);
    };
    const runIndirect = (name: string, args: GPUBuffer, offset: number) => {
      cp.setPipeline(this.cpipes[name]);
      cp.dispatchWorkgroupsIndirect(args, offset);
    };
    // field-derived state is rebuilt when the yarn field changed or a piece was retired (edgeArgs:
    // [0] crossings, [16] nodes, [32] cells, [48] single, all zero otherwise): loose threads, then
    // material connectivity and the free edges, then the edge field
    cp.setBindGroup(0, this.groups.threads);
    run('retireCount', Math.ceil(N / 128));
    cp.setBindGroup(0, this.edgeArgsGroup);
    run('edgeArgsMain', 1);
    cp.setBindGroup(0, this.groups.threads);
    for (const e of ['threadClear', 'threadPresence', 'threadAnchor', 'threadCloth', 'threadScan']) runIndirect(e, this.edgeArgs, 0);
    run('threadArgs', 1);
    cp.setBindGroup(0, this.groups.pieces);
    runIndirect('piecesClear', this.edgeArgs, 16);
    runIndirect('cellDirtyMain', this.edgeArgs, 32);
    runIndirect('nodeAlloc', this.edgeArgs, 16);
    runIndirect('piecesArgsMain', this.edgeArgs, 48);
    cp.setBindGroup(0, this.groups.piecesSolve);
    runIndirect('kappaSolve', this.piecesArgs, 0);
    cp.setBindGroup(0, this.groups.pieces);
    run('piecesArgsReset', 1);
    runIndirect('edgeListMain', this.edgeArgs, 0);
    runIndirect('edgeDrawArgs', this.edgeArgs, 48);
    this.edgeGroups.forEach((g, k) => {
      cp.setBindGroup(0, g);
      runIndirect(k === 0 ? 'seedInit' : k === this.edgeGroups.length - 1 ? 'edgeWrite' : 'seedStep', this.edgeArgs, 0);
    });
    // per frame: node frames, cell classes, the refined surface, the threads' attachment
    cp.setBindGroup(0, this.groups.refine);
    run('nodeFrame', Math.ceil(N / 128));
    run('classifyCells', Math.ceil(N / 128));
    run('args', 1);
    run('refine', Math.ceil(this.refinedCount / 128));
    cp.setBindGroup(0, this.groups.threadPrep);
    runIndirect('threadPrep', this.threadDraw, 32);
    cp.setBindGroup(0, this.groups.threadTail);
    runIndirect('threadTail', this.threadDraw, 32);
    cp.setBindGroup(0, this.groups.edgeRoots);
    runIndirect('edgeRootsMain', this.piecesArgs, 32);
    cp.end();

    const surfaceVerts = REFINE * REFINE * 6;
    const sp = encoder.beginRenderPass({
      label: 'shadow',
      timestampWrites: tw(1),
      colorAttachments: [],
      depthStencilAttachment: { view: this.shadowTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    sp.setBindGroup(0, this.groups.surfaceShadow);
    sp.setPipeline(this.pipes.surfaceShadow);
    sp.draw(surfaceVerts, this.cellCount);
    sp.setPipeline(this.pipes.fineShadow);
    sp.drawIndirect(this.drawArgs, 64);
    sp.setBindGroup(0, this.groups.threadShadow);
    sp.setPipeline(this.pipes.threadShadow);
    sp.drawIndirect(this.threadDraw, 0);
    sp.setBindGroup(0, this.groups.mountShadow);
    sp.setPipeline(this.pipes.mountShadow);
    sp.draw(24);
    for (const e of this.extras) e.drawShadow(sp, 'key');
    sp.end();

    const op = encoder.beginRenderPass({
      label: 'overShadow',
      colorAttachments: [],
      depthStencilAttachment: { view: this.overTex.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    op.setBindGroup(0, this.groups.surfaceOver);
    op.setPipeline(this.pipes.surfaceShadow);
    op.draw(surfaceVerts, this.cellCount);
    op.setPipeline(this.pipes.fineShadow);
    op.drawIndirect(this.drawArgs, 64);
    op.setBindGroup(0, this.groups.threadOver);
    op.setPipeline(this.pipes.threadShadow);
    op.drawIndirect(this.threadDraw, 0);
    op.setBindGroup(0, this.groups.mountOver);
    op.setPipeline(this.pipes.mountShadow);
    op.draw(24);
    for (const e of this.extras) e.drawShadow(op, 'over');
    op.end();

    if (floor) {
      // the floor's view of the cloth from each light
      const cc = encoder.beginComputePass({ label: 'floorCoverage' });
      for (const c of this.covGroups) {
        cc.setPipeline(c.pipe);
        cc.setBindGroup(0, c.group);
        cc.dispatchWorkgroups(Math.ceil(c.size / 8), Math.ceil(c.size / 8));
      }
      cc.end();
    }

    const mp = encoder.beginRenderPass({
      label: 'main',
      timestampWrites: tw(2),
      colorAttachments: [{ view: this.msaaColor.createView(), resolveTarget: this.hdr.createView(), clearValue: { r: this.background[0], g: this.background[1], b: this.background[2], a: 1 }, loadOp: 'clear', storeOp: 'discard' }],
      depthStencilAttachment: { view: this.msaaDepth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    mp.setBindGroup(0, this.groups.surface);
    mp.setPipeline(this.pipes.surface);
    mp.draw(surfaceVerts, this.cellCount);
    mp.setPipeline(this.pipes.fine);
    mp.drawIndirect(this.drawArgs, 64);
    mp.setBindGroup(0, this.groups.thread);
    mp.setPipeline(this.pipes.thread);
    mp.drawIndirect(this.threadDraw, 0);
    mp.setBindGroup(0, this.groups.mount);
    mp.setPipeline(this.pipes.mount);
    mp.draw(24);
    for (const e of this.extras) e.drawMain(mp);
    if (floor && this.floorGroup) {
      // after the opaque cloth (only what shows through it is shaded), before the blended fibre over it
      mp.setBindGroup(0, this.floorGroup);
      mp.setPipeline(this.pipes.floor);
      mp.draw(6);
    }
    mp.setBindGroup(0, this.groups.thread);
    mp.setPipeline(this.pipes.threadFuzz);
    mp.drawIndirect(this.threadDraw, 16);
    mp.setBindGroup(0, this.groups.fuzz);
    mp.setPipeline(this.pipes.edgeFuzz);
    mp.drawIndirect(this.piecesArgs, 16);
    mp.end();

    this.encodePost(encoder, target, tw(3));
  }

  private encodePost(encoder: GPUCommandEncoder, target: GPUTextureView, timestampWrites?: GPURenderPassTimestampWrites): void {
    const pp = encoder.beginRenderPass({
      label: 'post',
      timestampWrites,
      colorAttachments: [{ view: target, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pp.setPipeline(this.pipes.post);
    pp.setBindGroup(0, this.postGroup!);
    pp.draw(3);
    pp.end();
  }
}
