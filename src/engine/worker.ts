import { acquireGpu, GpuUnavailableError } from '../gpu/device';
import { readBuffer } from '../gpu/util';
import { DEFAULT_CAMERA, Renderer, cameraMatrices, type CameraState } from '../rendering/renderer';
import { add, dot, normalize, scale, sub, transformPoint, type Vec3 } from '../rendering/math';
import { Interaction } from '../simulation/gpu/interaction';
import { GPU_DEFAULTS, GpuSolver } from '../simulation/gpu/solver';
import { loadDrape, settleDrape } from '../simulation/gpu/settle';
import { Topology } from '../simulation/gpu/topology';
import { compileScene, defaultSceneSpecs, type CompiledScene } from '../simulation/scene';
import type { DevCommand, FromEngine, PointerCommand, ToEngine } from '../shared/messages';

interface WorkerScope {
  postMessage(m: FromEngine): void;
  onmessage: ((ev: MessageEvent<ToEngine>) => void) | null;
  requestAnimationFrame(cb: (t: number) => void): number;
}
const scope = self as unknown as WorkerScope;
const send = (m: FromEngine) => scope.postMessage(m);

const H = GPU_DEFAULTS.h;
/** Upper bound on substeps per presented frame; a larger backlog is dropped, never replayed. */
const MAX_SUBSTEPS_PER_FRAME = 5;
/** Presentation interval while simulating (s): physics runs in natural time, frames at <= 30/s. */
const SIM_FRAME_INTERVAL = 1 / 30 - 0.002;
/** Grip handle: maximum speed (m/s) and engagement lift toward the camera (m). */
const HANDLE_MAX_SPEED = 2.5;
const ENGAGE_LIFT = 0.002;
/** The cloth is at rest (simulation sleeps) below this nodal speed for REST_TIME seconds. */
const REST_SPEED = 0.003;
const REST_TIME = 0.4;

interface GripState {
  node: number;
  plane: Vec3;
  normal: Vec3;
  handle: Vec3;
  target: Vec3;
  depth: number;
  engageTime: number;
}

class Engine {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private scene!: CompiledScene;
  private solver!: GpuSolver;
  private renderer!: Renderer;
  private interaction!: Interaction;
  private topology!: Topology;
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
  private grip: GripState | null = null;
  private pickToken = 0;
  private pointerDown = false;
  private latestPointer: PointerCommand | null = null;
  private snapshot: { x: GPUBuffer; segs: GPUBuffer; tails: GPUBuffer } | null = null;
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
    const gpu = await acquireGpu((navigator as Navigator & { gpu?: GPU }).gpu);
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
    this.solver = new GpuSolver(this.device, this.scene, GPU_DEFAULTS);
    this.interaction = new Interaction(this.device, this.solver, this.scene);
    this.solver.preSubstep = (pass) => this.interaction.encodeSubstep(pass);
    this.solver.preAssemble = (dispatch) => this.interaction.encode(dispatch);
    this.topology = new Topology(this.device, this.solver, this.scene);
    this.renderer = new Renderer(this.device, format, this.solver, this.scene);
    this.renderer.resize(width, height);
    this.speedProbe = this.device.createBuffer({ label: 'speedProbe', size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    await this.settle();
    // authored defects could leave pieces disconnected from the start
    await this.topology.request([]);
    this.ready = true;
    this.asleep = true;
    this.lastFrame = performance.now();
    send({ type: 'status', state: 'running' });
    send({ type: 'cursor', cursor: 'grab' });
    scope.requestAnimationFrame((t) => this.frame(t));
  }

  /** Load (or compute) the settled drape, then keep the validated state as the reset snapshot. */
  private async settle(): Promise<void> {
    let loaded = false;
    try {
      const r = await fetch(new URL('/drape.bin', self.location.href));
      if (r.ok) loaded = loadDrape(this.solver, await r.arrayBuffer());
    } catch {
      loaded = false;
    }
    if (!loaded) await settleDrape(this.device, this.solver);
    const copy = (src: GPUBuffer, label: string) => this.device.createBuffer({ label, size: src.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.snapshot = { x: copy(this.solver.x, 'snap.x'), segs: copy(this.solver.segs, 'snap.segs'), tails: copy(this.solver.tails, 'snap.tails') };
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.solver.x, 0, this.snapshot.x, 0, this.solver.x.size);
    enc.copyBufferToBuffer(this.solver.segs, 0, this.snapshot.segs, 0, this.solver.segs.size);
    enc.copyBufferToBuffer(this.solver.tails, 0, this.snapshot.tails, 0, this.solver.tails.size);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
  }

  reset(): void {
    if (!this.ready || !this.snapshot) return;
    this.endGrip();
    this.pickToken++;
    const enc = this.device.createCommandEncoder({ label: 'reset' });
    enc.copyBufferToBuffer(this.snapshot.x, 0, this.solver.x, 0, this.solver.x.size);
    enc.copyBufferToBuffer(this.snapshot.segs, 0, this.solver.segs, 0, this.solver.segs.size);
    enc.copyBufferToBuffer(this.snapshot.tails, 0, this.solver.tails, 0, this.solver.tails.size);
    enc.clearBuffer(this.solver.v);
    enc.clearBuffer(this.solver.ext);
    enc.clearBuffer(this.solver.counters);
    this.device.queue.submit([enc.finish()]);
    this.topology.reset();
    void this.topology.request([]);
    this.topologyEvents = 0;
    this.simBacklog = 0;
    this.asleep = true;
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
    if (!this.ready) return;
    if (c.phase === 'down') {
      this.pointerDown = true;
      this.latestPointer = c;
      void this.beginGrip(c);
    } else if (c.phase === 'move') {
      this.latestPointer = c;
      if (this.grip) this.updateGripTarget(c);
    } else {
      this.pointerDown = false;
      this.pickToken++;
      this.endGrip();
    }
  }

  private async beginGrip(c: PointerCommand): Promise<void> {
    const token = ++this.pickToken;
    const m = cameraMatrices(this.camera, this.width / this.height);
    const px = (c.x / this.cssWidth) * this.width;
    const py = (c.y / this.cssHeight) * this.height;
    // larger than the projected node spacing (~3.5 mm cells, ~1.1% of the view height)
    const radius = Math.max(8, 0.016 * this.height);
    const hit = await this.interaction.pickAt(m.viewProj, px, py, this.width, this.height, radius);
    if (token !== this.pickToken || !this.pointerDown || !hit) return;
    const forward = normalize(sub(this.camera.target, m.eye));
    const p = hit.position as Vec3;
    this.grip = { node: hit.node, plane: p, normal: forward, handle: p, target: p, depth: 0, engageTime: performance.now() };
    this.interaction.attach(hit.node, p);
    this.wake();
    send({ type: 'cursor', cursor: 'grabbing' });
    if (this.latestPointer) this.updateGripTarget(this.latestPointer);
  }

  private updateGripTarget(c: PointerCommand): void {
    const g = this.grip;
    if (!g) return;
    const { origin, dir } = this.ray(c.x, c.y);
    const denom = dot(dir, g.normal);
    if (Math.abs(denom) < 1e-6) return;
    const t = dot(sub(g.plane, origin), g.normal) / denom;
    g.target = add(origin, scale(dir, t));
  }

  private endGrip(): void {
    if (this.grip) {
      this.interaction.release();
      // pieces held only by the grip are free now
      void this.topology.request([]);
      this.grip = null;
      send({ type: 'cursor', cursor: 'grab' });
    }
  }

  wheel(deltaY: number): void {
    if (this.grip) {
      this.grip.depth = Math.max(-0.03, Math.min(0.1, this.grip.depth - deltaY * 0.00006));
    } else {
      this.camera.distance = Math.max(0.3, Math.min(1.15, this.camera.distance * Math.exp(deltaY * 0.0008)));
      this.dirty = true;
    }
  }

  /** Advance the grip handle along a bounded-speed path toward the pointer target (one substep). */
  private stepHandle(h: number): void {
    const g = this.grip;
    if (!g) return;
    const engage = Math.min(1, (performance.now() - g.engageTime) / 150);
    const lift = -(ENGAGE_LIFT * engage + g.depth);
    const goal = add(g.target, scale(g.normal, lift));
    const d = sub(goal, g.handle);
    const len = Math.hypot(d[0], d[1], d[2]);
    const maxStep = HANDLE_MAX_SPEED * h;
    g.handle = len <= maxStep ? goal : add(g.handle, scale(d, maxStep / len));
    this.interaction.setTarget(g.handle);
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
          void this.topology.request(this.interaction.coreNodes);
        }
        this.probeBusy = false;
        if (this.grip || this.asleep) {
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
    const dt = Math.min(0.1, Math.max(0, (now - this.lastFrame) / 1000));
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
    if (steps === 0 && !this.dirty) return;
    this.dirty = false;
    this.lastPresent = now;
    for (let k = 0; k < steps; k++) {
      // grip targets are queue writes; one submission per substep keeps each ordered with its work
      this.stepHandle(H);
      const se = this.device.createCommandEncoder({ label: 'substep' });
      this.solver.encodeSubstep(se);
      this.device.queue.submit([se.finish()]);
    }
    if (steps > 0) this.probeRest(steps);
    this.substepsDone += steps;
    const enc = this.device.createCommandEncoder({ label: 'frame' });
    this.renderer.encode(enc, this.camera, now / 1000, this.context.getCurrentTexture().createView());
    this.device.queue.submit([enc.finish()]);
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
      this.pickToken++;
      this.endGrip();
      this.simBacklog = 0;
    }
    this.dirty = true;
  }

  // ------------------------------------------------------------------ development driver

  private async runSubsteps(n: number): Promise<void> {
    for (let k = 0; k < n; k++) {
      this.stepHandle(H);
      const se = this.device.createCommandEncoder({ label: 'devSubstep' });
      this.solver.encodeSubstep(se);
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
    case 'visibility':
      engine.setVisible(m.visible);
      break;
    case 'dev':
      engine.dev(m).then((result) => send({ type: 'devDone', id: m.id, result }), (e: unknown) => send({ type: 'devDone', id: m.id, result: String(e) }));
      break;
  }
};
