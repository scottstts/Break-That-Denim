import { acquireGpu, GpuUnavailableError } from '../gpu/device';
import { readBuffer } from '../gpu/util';
import { DEFAULT_CAMERA, NO_POST_EFFECTS, Renderer, cameraMatrices, type CameraState, type FloorImages } from '../rendering/renderer';
import { add, dot, normalize, scale, sub, transformPoint, type Vec3 } from '../rendering/math';
import { Interaction, MAX_GRIPS } from '../simulation/gpu/interaction';
import { Hand, PointerTrack } from '../simulation/hand';
import { GPU_DEFAULTS, GpuSolver, RELEASE_REFINEMENT } from '../simulation/gpu/solver';
import { loadDrape, settleDrape } from '../simulation/gpu/settle';
import { Topology } from '../simulation/gpu/topology';
import { compileScene, defaultSceneSpecs, type CompiledScene } from '../simulation/scene';
import { ALL_POINTERS, type DevCommand, type ExperienceName, type FromEngine, type PointerCommand, type ToEngine } from '../shared/messages';
import { BodyContact } from '../simulation/gpu/body';
import { fetchBeneathClip, fetchCreature } from '../stage/assets';
import { DESKTOP_PACE } from '../stage/beneath';
import type { Clip } from '../stage/clip';
import { CreatureRenderer } from '../stage/creatureRenderer';
import { Puppet } from '../stage/puppet';
import { Rig } from '../stage/rig';
import { Stage } from '../stage/stage';
import { loudnessAt } from '../audio/envelope';
import { ACOUSTIC_SLOT, ACOUSTIC_SLOTS, RECORD_FLOATS, SOURCES, decodeRecord, fracturePosition } from '../audio/record';
import { AcousticLog } from './acousticLog';

interface WorkerScope {
  postMessage(m: FromEngine): void;
  onmessage: ((ev: MessageEvent<ToEngine>) => void) | null;
  requestAnimationFrame(cb: (t: number) => void): number;
}
const scope = self as unknown as WorkerScope;
const send = (m: FromEngine) => scope.postMessage(m);
/** Loading progress, 0..1, by stage. */
const progress = (value: number) => send({ type: 'progress', value });

/** The floor's material set (public/texture), decoded as stored (no colour conversion); null if any is missing. */
async function loadFloorImages(): Promise<FloorImages | null> {
  const at = (name: string) =>
    fetch(new URL(`/texture/${name}`, self.location.href))
      .then((r) => (r.ok ? r.blob() : Promise.reject(new Error(name))))
      .then((b) => createImageBitmap(b, { colorSpaceConversion: 'none', premultiplyAlpha: 'none' }));
  try {
    const [base, normal, roughness, ao] = await Promise.all([at('floor_baseColor.jpg'), at('floor_normal.png'), at('floor_roughness.jpg'), at('floor_AO.jpg')]);
    return { base, normal, roughness, ao };
  } catch {
    return null;
  }
}

const H = GPU_DEFAULTS.h;
/** Upper bound on substeps per presented frame; a larger backlog is dropped, never replayed. */
const MAX_SUBSTEPS_PER_FRAME = 5;
/** Presentation interval while simulating (s): physics runs in natural time, frames at <= 30/s. */
const SIM_FRAME_INTERVAL = 1 / 30 - 0.002;
/** The cloth is at rest (simulation sleeps) below this nodal speed for REST_TIME seconds. */
const REST_SPEED = 0.003;
const REST_TIME = 0.4;
/** A staged experience's assets: the creature (GPU side and rig) and its authored clip. */
interface StagedAssets {
  renderer: CreatureRenderer;
  puppet: Puppet;
  clip: Clip;
}

/** Time over which the picture settles after an experience ends on its broken signal (ms). */
const REACQUIRE_MS = 600;
/** Wheel depth per unit deltaY while gripping (m). */
const WHEEL_DEPTH = 0.00006;

/** Shared clock of pointer events and frames (ms): both threads' performance.now() from one origin. */
const clock = (t: number) => performance.timeOrigin + t;

/** One finger's grip (interaction slot), its handle and the pointer driving it. */
interface GripState {
  slot: number;
  node: number;
  hand: Hand;
  track: PointerTrack;
  /** The drag plane: through the picked point, facing the camera (normal = camera forward). */
  plane: Vec3;
  normal: Vec3;
  /** Where the pointer's ray met the plane when it went down (drag origin). */
  downPoint: Vec3;
}

class Engine {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private scene!: CompiledScene;
  private solver!: GpuSolver;
  private renderer!: Renderer;
  private interaction!: Interaction;
  private topology!: Topology;
  /** A staged creature's body the cloth collides with (empty outside an experience). */
  private body!: BodyContact;
  /** The staged experience playing (free play when null), its creature once loaded, and its load. */
  private stage: Stage | null = null;
  private creatureLoad: Promise<StagedAssets> | null = null;
  /** Bumped whenever an experience is asked for or ended: a slower load does not start a stale one. */
  private experienceToken = 0;
  /** The staged laugh phrase as heard: the recording's loudness envelope, when `from` plays (ms since the epoch), the phrase. */
  private laugh: { envelope: Float32Array; start: number; from: number; to: number } | null = null;
  /** Simulated seconds per wall second while an experience plays (smoothed over about a second). */
  private pace = DESKTOP_PACE;
  /** When an experience ran to its end (ms, frame clock): the picture comes back through a reacquiring signal. */
  private reacquire = -Infinity;
  /** Ruptures + pull-outs at the last connectivity analysis. */
  private topologyEvents = 0;
  private camera: CameraState = { ...DEFAULT_CAMERA };
  private width = 1;
  private height = 1;
  private cssWidth = 1;
  private cssHeight = 1;
  private visible = true;
  private ready = false;
  private lastFrame = 0;
  private simBacklog = 0;
  /** Substeps still to run with the release refinement (solver.ts RELEASE_REFINEMENT). */
  private refineSteps = 0;
  /** Bumped by reset: a release pending its connectivity analysis is already done. */
  private releaseEpoch = 0;
  /** Grips by pointer id (one per finger, at most MAX_GRIPS). */
  private readonly grips = new Map<number, GripState>();
  /** Track of every pointer that is down, and the token of its pending pick. */
  private readonly down = new Map<number, PointerTrack>();
  private readonly picking = new Map<number, number>();
  private pickToken = 0;
  /** Acoustic records to the audio thread (audio/record.ts), once audio has started. */
  private audioPort: MessagePort | null = null;
  private acoustic: AcousticLog | null = null;
  /** Last pan of each sound source (kept while a source has no fracture). */
  private readonly lastPan = new Float64Array(SOURCES);
  private readonly anchors = new Float32Array(4 * SOURCES);
  private snapshot: { x: GPUBuffer; segs: GPUBuffer; tails: GPUBuffer; field: GPUBuffer } | null = null;
  /** GPU work of the previous frame still executing: never queue more than one frame ahead. */
  private inFlight = false;
  /** Simulation sleeps at rest; it wakes on interaction or reset. */
  private asleep = false;
  private restFor = 0;
  private speedProbe: GPUBuffer | null = null;
  private probeBusy = false;
  private probeSteps = 0;
  /** Something visible changed since the last presented frame. */
  private dirty = true;
  private lastPresent = 0;
  private statsTime = 0;
  private frameCount = 0;
  private substepsDone = 0;

  async init(canvas: OffscreenCanvas, width: number, height: number, cssWidth: number, cssHeight: number): Promise<void> {
    send({ type: 'status', state: 'loading' });
    // the settled drape downloads while the GPU state and pipelines are built
    const drape = fetch(new URL('/drape.bin', self.location.href))
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .catch(() => null);
    const floor = loadFloorImages();
    const gpu = await acquireGpu((navigator as Navigator & { gpu?: GPU }).gpu);
    progress(0.1);
    this.device = gpu.device;
    this.device.lost.then((info) => {
      send({ type: 'status', state: 'error', message: info.reason === 'destroyed' ? '' : 'The graphics device was lost. Reload to continue.' });
      this.ready = false;
    });
    this.device.addEventListener('uncapturederror', (e) => console.error('GPU', (e as GPUUncapturedErrorEvent).error.message));
    const context = canvas.getContext('webgpu');
    if (!context) throw new GpuUnavailableError('Could not create a WebGPU canvas context.');
    this.context = context;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: this.device, format, alphaMode: 'opaque' });
    this.width = width;
    this.height = height;
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    canvas.width = width;
    canvas.height = height;

    this.scene = compileScene(defaultSceneSpecs());
    progress(0.25);
    this.solver = new GpuSolver(this.device, this.scene, GPU_DEFAULTS);
    progress(0.5);
    this.interaction = new Interaction(this.device, this.solver, this.scene);
    this.solver.preSubstep = (pass) => this.interaction.encodeSubstep(pass);
    this.body = new BodyContact(this.device, this.solver, this.scene);
    this.solver.postCommit = (pass) => this.body.encodePierce(pass);
    // per Newton iteration, after the contact terms: a staged body's, then the grips'
    this.solver.preAssemble = (dispatch) => {
      this.body.encode(dispatch);
      this.interaction.encode(dispatch);
    };
    // pieces the fingers hold stay cloth: the anchors are whatever is held when an analysis applies
    this.topology = new Topology(this.device, this.solver, this.scene, () => this.interaction.coreNodes);
    this.renderer = new Renderer(this.device, format, this.solver, this.scene);
    this.renderer.resize(width, height);
    this.speedProbe = this.device.createBuffer({ label: 'speedProbe', size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    progress(0.7);

    const images = await floor;
    if (images) this.renderer.setFloorImages(images);
    await this.settle(await drape);
    progress(0.9);
    // authored defects could leave pieces disconnected from the start
    await this.topology.request();
    progress(1);
    this.ready = true;
    if (this.audioPort) this.startAcoustics();
    this.asleep = true;
    this.lastFrame = performance.now();
    send({ type: 'status', state: 'running' });
    send({ type: 'cursor', cursor: 'grab' });
    scope.requestAnimationFrame((t) => this.frame(t));
  }

  /** Load (or compute) the settled drape, then keep the validated state as the reset snapshot. */
  private async settle(drape: ArrayBuffer | null): Promise<void> {
    const loaded = drape !== null && loadDrape(this.solver, drape);
    if (!loaded) await settleDrape(this.device, this.solver);
    const copy = (src: GPUBuffer, label: string) => this.device.createBuffer({ label, size: src.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.snapshot = { x: copy(this.solver.x, 'snap.x'), segs: copy(this.solver.segs, 'snap.segs'), tails: copy(this.solver.tails, 'snap.tails'), field: copy(this.solver.field, 'snap.field') };
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.solver.x, 0, this.snapshot.x, 0, this.solver.x.size);
    enc.copyBufferToBuffer(this.solver.segs, 0, this.snapshot.segs, 0, this.solver.segs.size);
    enc.copyBufferToBuffer(this.solver.tails, 0, this.snapshot.tails, 0, this.solver.tails.size);
    enc.copyBufferToBuffer(this.solver.field, 0, this.snapshot.field, 0, this.solver.field.size);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  reset(): void {
    if (!this.ready || !this.snapshot) return;
    if (this.stage) {
      this.endExperience();
      return;
    }
    // a finished experience's sound may still be playing out
    send({ type: 'cue', cue: { kind: 'stop' } });
    this.restoreCloth();
  }

  /** The cloth back to its settled, intact state. */
  private restoreCloth(): void {
    if (!this.snapshot) return;
    this.releaseAll();
    this.releaseEpoch++;
    this.interaction.releaseAll();
    const enc = this.device.createCommandEncoder({ label: 'reset' });
    enc.copyBufferToBuffer(this.snapshot.x, 0, this.solver.x, 0, this.solver.x.size);
    enc.copyBufferToBuffer(this.snapshot.segs, 0, this.solver.segs, 0, this.solver.segs.size);
    enc.copyBufferToBuffer(this.snapshot.tails, 0, this.solver.tails, 0, this.solver.tails.size);
    enc.copyBufferToBuffer(this.snapshot.field, 0, this.solver.field, 0, this.solver.field.size);
    enc.clearBuffer(this.solver.v);
    enc.clearBuffer(this.solver.ext);
    enc.clearBuffer(this.solver.counters);
    enc.clearBuffer(this.solver.acousticSources);
    this.renderer.resetTopology(enc);
    this.device.queue.submit([enc.finish()]);
    this.topology.reset();
    void this.topology.request();
    this.topologyEvents = 0;
    this.simBacklog = 0;
    this.refineSteps = 0;
    this.asleep = true;
    this.dirty = true;
  }

  // ------------------------------------------------------------------ staged experiences

  /** The creature and its clip, loaded once (model, textures, performance) and kept for later runs. */
  private loadCreature(): Promise<StagedAssets> {
    this.creatureLoad ??= Promise.all([fetchCreature(self.location.href), fetchBeneathClip(self.location.href)]).then(([{ model, images }, clip]) => {
      const renderer = new CreatureRenderer(this.renderer, model, images);
      // the decoded images are released once their copies to the GPU have run
      void this.device.queue.onSubmittedWorkDone().then(() => {
        for (const img of Object.values(images)) if ('close' in img) img.close();
      });
      return { renderer, puppet: new Puppet(new Rig(model)), clip };
    });
    this.creatureLoad.catch(() => (this.creatureLoad = null));
    return this.creatureLoad;
  }

  heardLaugh(envelope: Float32Array, start: number, from: number, to: number): void {
    this.laugh = { envelope, start, from, to };
  }

  /** Loudness of the laugh as heard now (0 between phrases). */
  private laughLevel(): number {
    const l = this.laugh;
    if (!l) return 0;
    const t = l.from + (performance.timeOrigin + performance.now() - l.start) / 1000;
    return t >= l.from && t <= l.to ? loudnessAt(l.envelope, t) : 0;
  }

  /** A duration of the picture (simulated s) as heard at the pace it runs now (see DESKTOP_PACE). */
  private heard(duration: number): number {
    return (duration * DESKTOP_PACE) / Math.min(1.25, Math.max(0.1, this.pace));
  }

  prefetch(name: ExperienceName): void {
    if (this.ready && name === 'beneath') void this.loadCreature().catch(() => {});
  }

  /** Start (or restart) a staged experience on the intact cloth; null ends the one playing. */
  async experience(name: ExperienceName | null): Promise<void> {
    if (!this.ready) return;
    const token = ++this.experienceToken;
    if (name === null) {
      if (this.stage) this.endExperience();
      return;
    }
    if (this.stage) this.endExperience(false);
    send({ type: 'experience', name, state: 'loading' });
    let creature: StagedAssets;
    try {
      creature = await this.loadCreature();
    } catch (e) {
      console.error(e);
      if (token === this.experienceToken) send({ type: 'experience', name, state: 'failed' });
      return;
    }
    if (token !== this.experienceToken || !this.ready) return;
    this.releaseAll();
    this.restoreCloth();
    send({ type: 'cue', cue: { kind: 'stop' } });
    this.laugh = null;
    this.pace = DESKTOP_PACE;
    this.stage = new Stage({
      device: this.device,
      scene: this.scene,
      solver: this.solver,
      interaction: this.interaction,
      renderer: this.renderer,
      body: this.body,
      creature: creature.renderer,
      puppet: creature.puppet,
      clip: creature.clip,
      analyse: () => this.topology.request(),
      letGo: (slot) => {
        this.interaction.letGo(slot);
        const epoch = this.releaseEpoch;
        void this.topology.request().then(() => {
          if (epoch !== this.releaseEpoch) return;
          this.interaction.release(slot);
          this.refineSteps = Math.round(RELEASE_REFINEMENT.duration / H);
        });
      },
      cue: (e) => {
        if (e.kind === 'laugh') send({ type: 'cue', cue: { kind: 'laugh', from: e.from, to: e.to } });
        else if (e.kind === 'ambient') send({ type: 'cue', cue: { kind: 'ambient' } });
        else if (e.kind === 'ambientFade') send({ type: 'cue', cue: { kind: 'ambientFade', duration: this.heard(e.duration) } });
        else if (e.kind === 'glitch') send({ type: 'cue', cue: { kind: 'glitch', duration: this.heard(e.duration), level: e.level } });
      },
      held: (slot, held) => this.audioPort?.postMessage({ grip: slot, held }),
      laugh: () => this.laughLevel(),
    });
    this.wake();
    this.dirty = true;
    send({ type: 'experience', name, state: 'playing' });
    send({ type: 'cursor', cursor: 'default' });
  }

  /**
   * Back to free play: the creature leaves, the cloth is intact again, the camera the product's. An
   * experience that ran to its end (`now` given) leaves its sounds to finish; an interrupted one stops them.
   */
  private endExperience(announce = true, now = -Infinity): void {
    const stage = this.stage;
    if (!stage) return;
    this.reacquire = now;
    this.stage = null;
    stage.stop();
    if (now === -Infinity) {
      send({ type: 'cue', cue: { kind: 'stop' } });
      this.laugh = null;
    }
    for (let slot = 0; slot < MAX_GRIPS; slot++) this.audioPort?.postMessage({ grip: slot, held: false });
    this.restoreCloth();
    if (announce) {
      send({ type: 'experience', name: 'beneath', state: 'ended' });
      send({ type: 'cursor', cursor: 'grab' });
    }
  }

  /** After an experience's signal loss, free play comes back through a decoder settling (REACQUIRE_MS). */
  private reacquireSignal(now: number): void {
    const u = (now - this.reacquire) / REACQUIRE_MS;
    if (u < 0 || u >= 1) {
      if (this.renderer.post.glitch !== 0) this.renderer.post = { ...NO_POST_EFFECTS };
      return;
    }
    const g = (1 - u) * (1 - u);
    this.renderer.post = { ...NO_POST_EFFECTS, glitch: 0.7 * g, glitchSeed: Math.floor(now / 70) % 97 };
    this.dirty = true;
  }

  setSize(width: number, height: number, cssWidth: number, cssHeight: number): void {
    this.width = width;
    this.height = height;
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    const canvas = this.context.canvas as OffscreenCanvas;
    canvas.width = width;
    canvas.height = height;
    this.renderer?.resize(width, height);
    this.dirty = true;
  }

  private wake(): void {
    if (this.asleep) {
      this.asleep = false;
      this.restFor = 0;
      this.simBacklog = 0;
    }
  }

  private ray(cssX: number, cssY: number): { origin: Vec3; dir: Vec3 } {
    const m = cameraMatrices(this.camera, this.width / this.height);
    const nx = (cssX / this.cssWidth) * 2 - 1;
    const ny = 1 - (cssY / this.cssHeight) * 2;
    const a = transformPoint(m.invViewProj, [nx, ny, 0]);
    const b = transformPoint(m.invViewProj, [nx, ny, 1]);
    const p0: Vec3 = [a[0] / a[3], a[1] / a[3], a[2] / a[3]];
    const p1: Vec3 = [b[0] / b[3], b[1] / b[3], b[2] / b[3]];
    return { origin: p0, dir: normalize(sub(p1, p0)) };
  }

  pointer(c: PointerCommand): void {
    if (!this.ready || this.stage) return;
    const sample = { time: c.time, x: c.x, y: c.y };
    if (c.phase === 'cancel' && c.id === ALL_POINTERS) {
      this.releaseAll();
    } else if (c.phase === 'down') {
      // a release still waiting for its substep ends now: the finger is down again
      if ((this.grips.get(c.id)?.track.upTime ?? Infinity) < Infinity) this.endGrip(c.id);
      // one grip per finger, as many fingers as there are grip slots
      if (this.down.has(c.id) || this.down.size >= MAX_GRIPS) return;
      this.down.set(c.id, new PointerTrack(sample));
      void this.beginGrip(c.id, sample);
    } else if (c.phase === 'move') {
      this.down.get(c.id)?.add(sample);
    } else if (c.phase === 'up' && this.grips.has(c.id)) {
      // the grip ends at the substep that reaches the release in input time (stepHandles)
      this.down.get(c.id)?.add(sample);
      this.grips.get(c.id)!.track.upTime = sample.time;
    } else {
      this.down.delete(c.id);
      this.picking.delete(c.id);
      this.endGrip(c.id);
    }
  }

  private async beginGrip(id: number, at: { x: number; y: number }): Promise<void> {
    const token = ++this.pickToken;
    this.picking.set(id, token);
    const m = cameraMatrices(this.camera, this.width / this.height);
    const px = (at.x / this.cssWidth) * this.width;
    const py = (at.y / this.cssHeight) * this.height;
    // larger than the projected node spacing (~3.5 mm cells, ~1.1% of the view height)
    const radius = Math.max(8, 0.016 * this.height);
    const hit = await this.interaction.pickAt(m.viewProj, px, py, this.width, this.height, radius);
    if (this.picking.get(id) !== token) return;
    this.picking.delete(id);
    const track = this.down.get(id);
    if (!hit || !track) return;
    // a finger on material another finger already pinches holds nothing more
    if (this.interaction.holder(hit.node) >= 0) return;
    const slot = this.interaction.freeSlot();
    const p = hit.position as Vec3;
    if (slot < 0 || !this.interaction.attach(hit.node, p, slot, hit.window)) return;
    const forward = normalize(sub(this.camera.target, m.eye));
    const g: GripState = { slot, node: hit.node, hand: new Hand(p, scale(forward, -1)), track, plane: p, normal: forward, downPoint: p };
    g.downPoint = this.planePoint(g, track.first) ?? p;
    this.grips.set(id, g);
    this.audioPort?.postMessage({ grip: slot, held: true });
    this.wake();
    send({ type: 'cursor', cursor: 'grabbing' });
  }

  /** Where the pointer's ray at CSS (x, y) meets the grip's drag plane. */
  private planePoint(g: GripState, at: { x: number; y: number }): Vec3 | null {
    const { origin, dir } = this.ray(at.x, at.y);
    const denom = dot(dir, g.normal);
    if (Math.abs(denom) < 1e-6) return null;
    return add(origin, scale(dir, dot(sub(g.plane, origin), g.normal) / denom));
  }

  private endGrip(id: number): void {
    this.down.delete(id);
    const g = this.grips.get(id);
    if (!g) return;
    this.grips.delete(id);
    // a released finger is silent at once (only a held tear is heard)
    this.audioPort?.postMessage({ grip: g.slot, held: false });
    // pieces held only by this grip are freed first; the fingers open once that has applied (at most a
    // frame later), with the solve refined over the snap that follows
    this.interaction.letGo(g.slot);
    const epoch = this.releaseEpoch;
    void this.topology.request().then(() => {
      if (epoch !== this.releaseEpoch) return;
      this.interaction.release(g.slot);
      this.refineSteps = Math.round(RELEASE_REFINEMENT.duration / H);
    });
    if (this.grips.size === 0) send({ type: 'cursor', cursor: 'grab' });
  }

  /** Release every grip and forget every pointer (cancel, reset, hidden page). */
  private releaseAll(): void {
    this.pickToken++;
    this.picking.clear();
    for (const id of [...this.grips.keys()]) this.endGrip(id);
    this.down.clear();
  }

  wheel(deltaY: number): void {
    if (this.stage) return;
    if (this.grips.size > 0) {
      for (const g of this.grips.values()) g.hand.wheel(-deltaY * WHEEL_DEPTH);
    } else {
      this.camera.distance = Math.max(0.3, Math.min(1.15, this.camera.distance * Math.exp(deltaY * 0.0008)));
      this.dirty = true;
    }
  }

  /** Newton iterations the next substep must run at least (0: the schedule's own). */
  private substepIterations(): number {
    if (this.refineSteps <= 0) return 0;
    this.refineSteps--;
    return RELEASE_REFINEMENT.newtonIterations;
  }

  /**
   * Advance every grip's handle by one substep toward its pointer at input time `t` (shared clock);
   * a pointer that went up by then releases its grip before the substep.
   */
  private stepHandles(h: number, t: number): void {
    for (const [id, g] of [...this.grips]) {
      if (g.track.upTime <= t) {
        this.endGrip(id);
        continue;
      }
      const p = this.planePoint(g, g.track.at(t));
      g.track.prune(t);
      g.hand.step(h, p ? sub(p, g.downPoint) : [0, 0, 0]);
      this.interaction.setTarget(g.hand.target, g.slot);
      this.interaction.setPinch(g.hand.pinch, g.slot);
    }
  }


  /**
   * Sound sources follow the fingers: each emission belongs to the nearest held grip (common.wgsl
   * nearestSource), so two tears made by two hands are two sources. With no grip held the last
   * anchors stay, and a tear running on after the release keeps its source.
   */
  private updateAnchors(): void {
    if (this.stage) {
      const sources = this.stage.sources();
      if (sources.every((s) => !s)) return;
      this.anchors.fill(0);
      sources.forEach((s, slot) => s && this.anchors.set([...s, 1], 4 * slot));
      this.device.queue.writeBuffer(this.solver.acousticSources, 0, this.anchors);
      return;
    }
    if (this.grips.size === 0) return;
    this.anchors.fill(0);
    for (const g of this.grips.values()) this.anchors.set([...g.hand.target, 1], 4 * g.slot);
    this.device.queue.writeBuffer(this.solver.acousticSources, 0, this.anchors);
  }

  /** Audio has started: acoustic records go to `port` from now on. */
  setAudio(port: MessagePort): void {
    this.audioPort = port;
    if (this.ready) this.startAcoustics();
  }

  private startAcoustics(): void {
    if (this.acoustic) return;
    this.acoustic = new AcousticLog(this.device, this.solver.counters);
    // what accumulated before audio started is not heard
    const enc = this.device.createCommandEncoder({ label: 'acousticStart' });
    enc.clearBuffer(this.solver.counters, 4 * ACOUSTIC_SLOT, 4 * ACOUSTIC_SLOTS);
    this.device.queue.submit([enc.finish()]);
  }

  /** Decode a frame's acoustic records and send them to the audio thread. */
  private sendAcoustics(raw: Uint32Array, count: number, duration: number, viewProj: Float32Array): void {
    if (!this.audioPort) return;
    const out = new Float32Array(count * RECORD_FLOATS);
    for (let q = 0; q < count; q++) {
      for (let src = 0; src < SOURCES; src++) {
        const at = fracturePosition(raw, q * ACOUSTIC_SLOTS, src);
        if (!at) continue;
        // the source's fracture, horizontally on screen
        const c = transformPoint(viewProj, at);
        if (c[3] > 1e-6) this.lastPan[src] = Math.max(-1, Math.min(1, c[0] / c[3]));
      }
      decodeRecord(raw, q * ACOUSTIC_SLOTS, out, q * RECORD_FLOATS, duration, this.lastPan);
    }
    this.audioPort.postMessage(out, [out.buffer]);
  }

  /** Asynchronously read the largest nodal speed since the last probe (rest detection). */
  private probeRest(steps: number): void {
    if (this.probeBusy || !this.speedProbe) {
      this.probeSteps += steps;
      return;
    }
    const probe = this.speedProbe;
    const window = (this.probeSteps + steps) * H;
    this.probeSteps = 0;
    this.probeBusy = true;
    const enc = this.device.createCommandEncoder({ label: 'restProbe' });
    enc.copyBufferToBuffer(this.solver.counters, 0, probe, 0, 16);
    enc.clearBuffer(this.solver.counters, 8, 4);
    this.device.queue.submit([enc.finish()]);
    probe.mapAsync(GPUMapMode.READ).then(
      () => {
        const range = probe.getMappedRange();
        const speed = new Float32Array(range)[2];
        const c = new Uint32Array(range, 0, 2);
        const events = c[0] + c[1];
        probe.unmap();
        if (events !== this.topologyEvents) {
          this.topologyEvents = events;
          void this.topology.request();
        }
        this.probeBusy = false;
        if (this.grips.size > 0 || this.asleep || this.stage) {
          this.restFor = 0;
          return;
        }
        this.restFor = speed < REST_SPEED ? this.restFor + window : 0;
        if (this.restFor >= REST_TIME) this.asleep = true;
      },
      () => (this.probeBusy = false),
    );
  }

  private frame(now: number): void {
    if (!this.ready) return;
    scope.requestAnimationFrame((t) => this.frame(t));
    const wall = Math.max(0, (now - this.lastFrame) / 1000);
    const dt = Math.min(0.1, wall);
    // present at most every SIM_FRAME_INTERVAL while simulating, and never with a frame in flight
    if (this.inFlight || (!this.asleep && (now - this.lastPresent) / 1000 < SIM_FRAME_INTERVAL)) return;
    this.lastFrame = now;
    let steps = 0;
    if (this.visible && !this.asleep) {
      this.simBacklog += dt;
      steps = Math.floor(this.simBacklog / H);
      if (steps > MAX_SUBSTEPS_PER_FRAME) {
        steps = MAX_SUBSTEPS_PER_FRAME;
        this.simBacklog = 0;
      } else {
        this.simBacklog -= steps * H;
      }
    }
    // how fast the picture runs: simulated seconds per wall second, over about a second
    if (this.stage && wall > 0 && wall < 1) this.pace += (1 - Math.exp(-wall)) * ((steps * H) / wall - this.pace);
    if (steps === 0 && !this.dirty) return;
    this.dirty = false;
    this.lastPresent = now;
    if (steps > 0) this.updateAnchors();
    const acoustic = steps > 0 && this.acoustic?.begin() ? this.acoustic : null;
    // the frame's substeps follow the pointers through the wall interval it covers
    const end = clock(now);
    for (let k = 0; k < steps; k++) {
      // grip targets are queue writes; one submission per substep keeps each ordered with its work
      if (this.stage) this.stage.substep(H);
      else this.stepHandles(H, end - 1000 * dt * (1 - (k + 1) / steps));
      const se = this.device.createCommandEncoder({ label: 'substep' });
      this.solver.encodeSubstep(se, this.substepIterations());
      acoustic?.record(se);
      this.device.queue.submit([se.finish()]);
    }
    if (acoustic) {
      // read back as soon as the substeps finish (before the frame renders): the substeps are heard
      // over the wall time this frame presents them in
      const duration = dt / steps;
      const viewProj = cameraMatrices(this.stage ? this.stage.camera : this.camera, this.width / this.height).viewProj;
      void acoustic.end().then((r) => r && this.sendAcoustics(r.raw, r.count, duration, viewProj));
    }
    if (steps > 0) this.probeRest(steps);
    this.substepsDone += steps;
    const enc = this.device.createCommandEncoder({ label: 'frame' });
    const camera = this.stage ? this.stage.frame() : this.camera;
    if (!this.stage) this.reacquireSignal(now);
    this.renderer.encode(enc, camera, now / 1000, this.context.getCurrentTexture().createView());
    this.device.queue.submit([enc.finish()]);
    if (this.stage?.done) this.endExperience(true, now);
    this.inFlight = true;
    this.device.queue.onSubmittedWorkDone().then(() => (this.inFlight = false));
    this.frameCount++;
    if (now - this.statsTime > 1000) {
      const secs = (now - this.statsTime) / 1000;
      send({
        type: 'stats',
        stats: {
          frameMs: (secs * 1000) / Math.max(1, this.frameCount),
          substepsPerFrame: this.substepsDone / Math.max(1, this.frameCount),
          simWallRatio: (this.substepsDone * H) / secs,
          gpuMsPerSubstep: 0,
          gpuRenderMs: 0,
          brokenSegments: 0,
          asleep: this.asleep,
        },
      });
      this.frameCount = 0;
      this.substepsDone = 0;
      this.statsTime = now;
    }
  }

  setVisible(v: boolean): void {
    this.visible = v;
    if (!v) {
      this.releaseAll();
      this.simBacklog = 0;
    }
    this.dirty = true;
  }

  // ------------------------------------------------------------------ development driver

  private async runSubsteps(n: number): Promise<void> {
    for (let k = 0; k < n; k++) {
      this.stepHandles(H, clock(performance.now()));
      const se = this.device.createCommandEncoder({ label: 'devSubstep' });
      this.solver.encodeSubstep(se, this.substepIterations());
      this.device.queue.submit([se.finish()]);
      if ((k & 3) === 3) await this.device.queue.onSubmittedWorkDone();
    }
    await this.device.queue.onSubmittedWorkDone();
  }

  async dev(c: DevCommand): Promise<unknown> {
    if (!this.ready) return null;
    if (c.op === 'step') {
      await this.runSubsteps(c.substeps);
      const counters = new Uint32Array(await readBuffer(this.device, this.solver.counters, 8));
      return { ruptures: counters[0], pulledOut: counters[1] };
    }
    if (c.op === 'camera') {
      if (c.distance !== undefined) this.camera.distance = c.distance;
      if (c.pitch !== undefined) this.camera.pitch = c.pitch;
      if (c.yaw !== undefined) this.camera.yaw = c.yaw;
      if (c.target) this.camera.target = c.target;
      this.dirty = true;
      return null;
    }
    return null;
  }
}

const engine = new Engine();
scope.onmessage = (ev: MessageEvent<ToEngine>) => {
  const m = ev.data;
  switch (m.type) {
    case 'init':
      engine.init(m.canvas, m.width, m.height, m.cssWidth, m.cssHeight).catch((e: unknown) => {
        console.error(e);
        send({ type: 'status', state: 'error', message: e instanceof GpuUnavailableError ? e.message : 'The fabric could not be started on this device.' });
      });
      break;
    case 'resize':
      engine.setSize(m.width, m.height, m.cssWidth, m.cssHeight);
      break;
    case 'pointer':
      engine.pointer(m);
      break;
    case 'wheel':
      engine.wheel(m.deltaY);
      break;
    case 'reset':
      engine.reset();
      break;
    case 'experience':
      void engine.experience(m.name);
      break;
    case 'prefetch':
      engine.prefetch(m.name);
      break;
    case 'laugh':
      engine.heardLaugh(m.envelope, m.start, m.from, m.to);
      break;
    case 'visibility':
      engine.setVisible(m.visible);
      break;
    case 'audio':
      engine.setAudio(m.port);
      break;
    case 'dev':
      engine.dev(m).then((result) => send({ type: 'devDone', id: m.id, result }), (e: unknown) => send({ type: 'devDone', id: m.id, result: String(e) }));
      break;
  }
};
