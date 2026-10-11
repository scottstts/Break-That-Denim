import { DEFAULT_CAMERA, NO_POST_EFFECTS, type CameraState, type Renderer } from '../rendering/renderer';
import type { Vec3 } from '../rendering/math';
import type { BodyContact } from '../simulation/gpu/body';
import type { Interaction, PickResult } from '../simulation/gpu/interaction';
import type { GpuSolver } from '../simulation/gpu/solver';
import { hookPatch, type GripPatch } from '../simulation/grip';
import { materialAt, nodeAt } from '../simulation/materialPoint';
import type { CompiledScene } from '../simulation/scene';
import { stageEvents, type StageEvent } from './beneath';
import { CLAW_HOOK } from './claws';
import type { Clip, ClipSample } from './clip';
import type { CreatureRenderer } from './creatureRenderer';
import type { Puppet, Side } from './puppet';
import { apply4, invertRigid, v3, type V } from './rig';

/** What the stage drives: the engine's simulation, rendering and outputs. */
export interface StageHost {
  device: GPUDevice;
  scene: CompiledScene;
  solver: GpuSolver;
  interaction: Interaction;
  renderer: Renderer;
  body: BodyContact;
  creature: CreatureRenderer;
  puppet: Puppet;
  clip: Clip;
  /** Connectivity analysis now (pieces cut loose fall); resolves once applied. */
  analyse(): Promise<void>;
  /** Let a hold go as a released finger does (pieces only it held are freed first). */
  letGo(slot: number): void;
  /** A sound cue for the main thread, and holds starting or ending (the tear voice). */
  cue(e: StageEvent): void;
  held(slot: number, held: boolean): void;
  /** Loudness (0..1) of the staged laugh as heard now: it moves the jaw. */
  laugh(): number;
}

/** A hand's hold: fixed in the frame of the joint that is in the cloth; the nodes it holds. */
interface Hold {
  slot: number;
  joint: number;
  local: V;
  nodes: number[];
}

/** The cloth a hand is about to close on, and the correction that steers the hand onto it. */
interface Aim {
  hit: PickResult;
  offset: V;
  since: number;
  lead: number;
}

const SLOTS: Record<Side, number> = { R: 0, L: 1 };
const SIDES = ['L', 'R'] as const;
/** The plane the staged camera's target is put on (the cloth's, as the product camera's target). */
const TARGET_Z = DEFAULT_CAMERA.target[2];

/** The renderer's lighting and film as the product has them (restored when the stage ends). */
type Lighting = Pick<Renderer, 'keyIntensity' | 'overIntensity' | 'exposure' | 'keyOffset' | 'keyAim' | 'keyCone' | 'keyInner' | 'keyLeak' | 'keyColor' | 'ambient' | 'torchIntensity'>;

/**
 * One run of the staged experience: the clip's events (hooks, grabs, releases, sound cues) applied on
 * the simulation clock; per substep the creature as the clip poses it, the body the cloth collides with
 * (its claws pierce it) and its claws' holds; per frame the camera, lights and film as the clip has them.
 */
export class Stage {
  readonly duration: number;
  readonly events: StageEvent[];
  private readonly host: StageHost;
  private readonly clip: Clip;
  private t = 0;
  private next = 0;
  private readonly holds: Partial<Record<Side, Hold>> = {};
  private readonly aims: Partial<Record<Side, Aim>> = {};
  /** A hand's correction fading out after it let go (back onto its own path). */
  private readonly fading: Record<Side, { offset: V; since: number }> = { L: { offset: [0, 0, 0], since: -1 }, R: { offset: [0, 0, 0], since: -1 } };
  private readonly view: CameraState = { ...DEFAULT_CAMERA, target: [...DEFAULT_CAMERA.target], eye: [0, 0, 1], up: [0, 1, 0] };
  private readonly base: Lighting;
  private stopped = false;

  constructor(host: StageHost) {
    this.host = host;
    this.clip = host.clip;
    this.duration = this.clip.duration;
    this.events = stageEvents(this.clip);
    const r = host.renderer;
    this.base = {
      keyIntensity: r.keyIntensity,
      overIntensity: r.overIntensity,
      exposure: r.exposure,
      keyOffset: [...r.keyOffset],
      keyAim: [...r.keyAim],
      keyCone: r.keyCone,
      keyInner: r.keyInner,
      keyLeak: r.keyLeak,
      keyColor: [...r.keyColor],
      ambient: [...r.ambient],
      torchIntensity: r.torchIntensity,
    };
    // the experience plays over the void: nothing to land on
    r.showFloor = false;
    host.solver.setFloor(null);
    // the room's lights stay put while the camera moves; the bulb is placed relative to this anchor
    r.lightAnchor = [...DEFAULT_CAMERA.target];
    if (!r.extras.includes(host.creature)) r.extras.push(host.creature);
    host.creature.visible = true;
    host.puppet.bind(this.clip);
    this.poseAt(0);
  }

  /** The camera at the current time. */
  get camera(): CameraState {
    return this.view;
  }

  get time(): number {
    return this.t;
  }

  get done(): boolean {
    return this.t >= this.duration;
  }

  /** Simulated seconds advance by h: events up to then, the body's pose at the substep's end, hold targets. */
  substep(h: number): void {
    if (this.stopped) return;
    this.t += h;
    const ev = this.events;
    while (this.next < ev.length && ev[this.next].t <= this.t) this.apply(ev[this.next++]);
    this.poseAt(this.t);
    this.host.body.setPose(this.host.puppet.capsules(), h);
    for (const s of SIDES) {
      const hold = this.holds[s];
      if (hold) this.host.interaction.setTarget(this.holdPoint(hold), hold.slot);
    }
  }

  /** The frame at the current time: creature skinning, camera, lights and film; returns the camera. */
  frame(): CameraState {
    const s = this.clip.sample(this.t);
    this.placeCamera(s);
    const r = this.host.renderer;
    const b = this.base;
    // the bulb: a spot of radiant intensity bulbI; the renderer's key is irradiance at its anchor distance
    const anchor = r.lightAnchor!;
    r.keyOffset = [s.bulbPos[0] - anchor[0], s.bulbPos[1] - anchor[1], s.bulbPos[2] - anchor[2]];
    r.keyAim = [r.keyOffset[0] + s.bulbAim[0], r.keyOffset[1] + s.bulbAim[1], r.keyOffset[2] + s.bulbAim[2]];
    r.keyIntensity = s.bulbI / v3.dot(r.keyOffset, r.keyOffset);
    r.keyCone = s.bulbCone[0];
    r.keyInner = s.bulbCone[1];
    r.keyLeak = 0;
    r.keyColor = [...s.bulbColor];
    r.overIntensity = s.over;
    // the camera's light, on the same failing supply as the bulb
    r.torchPos = [...s.torchPos];
    r.torchAim = [...s.torchAim];
    r.torchCone = [s.torchCone[0], s.torchCone[1]];
    r.torchIntensity = s.torchI;
    r.torchColor = [...s.torchColor];
    r.ambient = [b.ambient[0] * s.ambient, b.ambient[1] * s.ambient, b.ambient[2] * s.ambient];
    r.exposure = b.exposure * s.exposure;
    this.host.creature.setFrame(this.host.puppet.rig.skinMatrices(), s.pitPos, s.pitI);
    // the corruption's look holds for a few frames, as a decoder's corrupted state does
    r.post = { grade: s.grade, grain: s.grain, vignette: s.vignette, glitch: s.glitch, glitchSeed: Math.floor(this.t * 9) % 97 };
    r.freeze = s.freeze;
    return this.view;
  }

  /** Where the sound of a hold comes from, per slot; null where nothing is held. */
  sources(): (V | null)[] {
    const out: (V | null)[] = [null, null, null, null];
    for (const s of SIDES) {
      const hold = this.holds[s];
      if (hold) out[hold.slot] = this.holdPoint(hold);
    }
    return out;
  }

  /** End the run: holds let go, the body leaves the cloth, the renderer is the product's again. */
  stop(): void {
    if (this.stopped) return;
    this.stopped = true;
    for (const s of SIDES) if (this.holds[s]) this.release(s);
    this.host.body.clear();
    const r = this.host.renderer;
    Object.assign(r, this.base);
    r.showFloor = true;
    this.host.solver.setFloor(r.floorZ);
    r.lightAnchor = null;
    r.post = { ...NO_POST_EFFECTS };
    r.freeze = false;
    this.host.creature.visible = false;
  }

  /** The clip's camera; its target is put on the cloth's plane along the line of sight. */
  private placeCamera(s: ClipSample): void {
    const v = this.view;
    const fwd = v3.norm(v3.sub(s.target, s.eye));
    const along = fwd[2] < -0.1 ? (s.eye[2] - TARGET_Z) / -fwd[2] : 1;
    v.eye = [...s.eye] as Vec3;
    v.up = [...s.up] as Vec3;
    v.target = [s.eye[0] + fwd[0] * along, s.eye[1] + fwd[1] * along, s.eye[2] + fwd[2] * along];
    v.distance = along;
    v.fovY = s.fovY;
  }

  private poseAt(t: number): void {
    const s = this.clip.sample(t);
    if (t === 0) this.placeCamera(s);
    const puppet = this.host.puppet;
    puppet.play(s, this.host.laugh());
    for (const side of SIDES) puppet.reach(side, this.correction(side, t));
  }

  /** A hand's correction onto the cloth it aims at, eased in over the aim's lead and out after release. */
  private correction(s: Side, t: number): V {
    const ease = (u: number) => {
      const k = Math.min(1, Math.max(0, u));
      return k * k * (3 - 2 * k);
    };
    const aim = this.aims[s];
    if (aim) return v3.scale(aim.offset, ease((t - aim.since) / (0.8 * aim.lead)));
    const f = this.fading[s];
    return f.since < 0 ? [0, 0, 0] : v3.scale(f.offset, 1 - ease((t - f.since) / 0.35));
  }

  private holdPoint(hold: Hold): V {
    return this.host.puppet.rig.point(hold.joint, hold.local);
  }

  private apply(e: StageEvent): void {
    switch (e.kind) {
      case 'hook':
        this.hook(e.side);
        break;
      case 'aim':
        void this.aim(e.side, e.radius, e.lead);
        break;
      case 'grab':
        this.grab(e.side);
        break;
      case 'release':
        if (this.holds[e.side]) this.release(e.side);
        else delete this.aims[e.side];
        break;
      default:
        this.host.cue(e);
    }
  }

  /** The claws hook the yarns at their tips (through the holes they pierced); the hold rides on the finger shafts. */
  private hook(s: Side): void {
    const { scene, interaction } = this.host;
    const puppet = this.host.puppet;
    if (this.holds[s]) return;
    const nodes = puppet.clawTips(s).map((p) => {
      const [u, v] = materialAt(scene, p[0], p[1]);
      return nodeAt(scene, u, v);
    });
    const patch = hookPatch(scene, nodes, CLAW_HOOK, interaction.isFree);
    if (!patch) return;
    // where the cloth is held: the claws' centre, on the cloth
    const c = puppet.clawCentre(s);
    const [u, v] = materialAt(scene, c[0], c[1]);
    this.attach(s, nodeAt(scene, u, v), patch, [c[0], c[1], 0], 'shaft');
  }

  /**
   * The hand finds the cloth nearest to where its claws will be when it closes (`lead` from now), and
   * is steered onto it meanwhile.
   */
  private async aim(s: Side, radius: number, lead: number): Promise<void> {
    const t0 = this.t;
    const puppet = this.host.puppet;
    puppet.play(this.clip.sample(t0 + lead));
    const planned = puppet.clawCentre(s);
    this.poseAt(t0);
    const hit = await this.host.interaction.nearest(planned, radius);
    if (this.stopped || !hit || this.holds[s]) return;
    this.aims[s] = { hit, offset: v3.sub(hit.position as V, planned), since: t0, lead };
  }

  /** The hand closes on the cloth it aimed at, where its claws are now. */
  private grab(s: Side): void {
    const aim = this.aims[s];
    if (!aim || this.holds[s]) return;
    const { scene, interaction } = this.host;
    const patch = hookPatch(scene, [aim.hit.node], CLAW_HOOK, interaction.isFree, aim.hit.window);
    if (!patch) return;
    this.attach(s, aim.hit.node, patch, this.host.puppet.clawCentre(s), 'claw');
  }

  /**
   * Hold `patch` from where it is: the handle starts at the cloth (`at`, taken as the picked node's place)
   * and from then on moves with the joint in the cloth; the held nodes ride on the claw (exempt from
   * its body's contact and piercing).
   */
  private attach(s: Side, picked: number, patch: GripPatch, at: V, part: 'shaft' | 'claw'): void {
    const slot = SLOTS[s];
    const rig = this.host.puppet.rig;
    const joint = this.host.puppet.holdJoint(s, part);
    const local = apply4(invertRigid(rig.world.subarray(16 * joint, 16 * joint + 16)), at);
    this.host.interaction.attachPatch(slot, picked, patch, at, CLAW_HOOK.maxLead);
    this.host.interaction.setTarget(at, slot);
    this.host.body.setExempt(patch.nodes, true);
    this.holds[s] = { slot, joint, local, nodes: patch.nodes };
    this.host.held(slot, true);
  }

  private release(s: Side): void {
    const hold = this.holds[s]!;
    delete this.holds[s];
    const aim = this.aims[s];
    delete this.aims[s];
    this.fading[s] = { offset: aim ? aim.offset : [0, 0, 0], since: this.t };
    this.host.body.setExempt(hold.nodes, false);
    this.host.held(hold.slot, false);
    this.host.letGo(hold.slot);
  }
}
