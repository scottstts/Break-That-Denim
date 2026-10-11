import type { BodyCapsule } from '../simulation/gpu/body';
import type { Clip, ClipSample } from './clip';
import { fitCapsules, v3, type BoneCapsule, type Rig, type V } from './rig';

export type Side = 'L' | 'R';

/** Joints of one hand's clawed fingers (base, tip segment), index to pinky. */
const FINGERS = ['finguer', 'middle', 'ring', 'pinky'] as const;

interface HandRig {
  arm: number;
  forearm: number;
  hand: number;
  fingers: [number, number][];
  /** Claw tips in each finger's tip-segment frame. */
  tips: V[];
}

/**
 * The staged creature in world space: its rig posed from an authored clip (Clip), a hand steered onto
 * cloth it closes on (reach), and what the cloth meets of it (capsules fitted to its skin, its claw
 * tips).
 */
export class Puppet {
  readonly rig: Rig;
  private readonly bones: BoneCapsule[];
  private readonly hands: Record<Side, HandRig>;
  /** Fitted capsules whose end is a claw's point (the tip segments): what pierces the cloth. */
  private readonly sharp: Set<number>;
  /** Rig joint of each clip joint, for the clip this puppet was bound to. */
  private clipJoints: Int32Array | null = null;
  private boundClip: Clip | null = null;

  constructor(rig: Rig) {
    this.rig = rig;
    const carried = rig.carriedVertices();
    // 3 mm of the model's units (cm, joint frames in m x 100) for the blend-skinned joints
    this.bones = fitCapsules(carried, 0.003);
    const handRig = (s: Side): HandRig => {
      const j = (n: string) => rig.joint(`${n}.${s}`);
      const fingers = FINGERS.map((f) => [j(f), rig.joint(`${f}.${s}.001`)] as [number, number]);
      // a claw tip: the tip segment's skin farthest along its length (+y, the bone axis)
      const tips = fingers.map(([, t]) => {
        const pts = carried[t];
        let best = pts[0];
        for (const p of pts) if (p[1] > best[1]) best = p;
        return best;
      });
      return { arm: j('Arm'), forearm: j('Forearm'), hand: j('Hand'), fingers, tips };
    };
    this.hands = { L: handRig('L'), R: handRig('R') };
    this.sharp = new Set();
    // a tip segment's capsule reaches its claw: what presses the cloth first is the claw's point
    for (const H of [this.hands.L, this.hands.R]) {
      H.fingers.forEach(([, tip], k) => {
        const c = this.bones.find((b) => b.joint === tip);
        if (!c) return;
        const dir = v3.norm(v3.sub(c.b, c.a));
        c.b = v3.sub(H.tips[k], v3.scale(dir, 0.8 * c.radius));
        this.sharp.add(tip);
      });
    }
  }

  /** Bind to a clip: its joints by name, the bind offsets it keeps, the skeleton's place in the world. */
  bind(clip: Clip): void {
    if (this.boundClip === clip) return;
    const rig = this.rig;
    this.clipJoints = Int32Array.from(clip.joints, (n) => rig.joint(n));
    if (this.clipJoints.length !== rig.count) throw new Error(`clip poses ${this.clipJoints.length} of ${rig.count} joints`);
    clip.bindT.forEach((t, k) => rig.localT.set(t, 3 * this.clipJoints![k]));
    rig.restT.set(rig.localT);
    rig.setSkeletonPlacement(clip.skeleton);
    this.boundClip = clip;
  }

  /** The creature as the clip has it at one sample. */
  play(s: ClipSample): void {
    const map = this.clipJoints!;
    const rig = this.rig;
    for (let k = 0; k < map.length; k++) rig.localR.set(s.rotations.subarray(4 * k, 4 * k + 4), 4 * map[k]);
    rig.localT.set(s.rootT, 3 * map[this.boundClip!.root]);
    rig.update();
  }

  /**
   * Steer a hand by `offset` (world) onto cloth it is about to close on: the arm re-solved (two-bone,
   * the elbow keeping its side), the hand's orientation and its fingers as the clip has them.
   */
  reach(s: Side, offset: V): void {
    if (offset[0] === 0 && offset[1] === 0 && offset[2] === 0) return;
    const rig = this.rig;
    const H = this.hands[s];
    const orient = rig.worldRotation(H.hand);
    const wrist = rig.position(H.hand);
    const elbow = rig.position(H.forearm);
    rig.twoBone(H.arm, H.forearm, H.hand, v3.add(wrist, offset), v3.add(elbow, offset));
    rig.setWorldRotation(H.hand, orient);
    rig.updateFrom(H.hand);
  }

  /** World positions of a hand's four claw tips (index to pinky). */
  clawTips(s: Side): V[] {
    const H = this.hands[s];
    return H.fingers.map(([, t], k) => this.rig.point(t, H.tips[k]));
  }

  /** Centroid of a hand's claw tips. */
  clawCentre(s: Side): V {
    const tips = this.clawTips(s);
    return v3.scale(tips.reduce((a, b) => v3.add(a, b)), 1 / tips.length);
  }

  /**
   * What the cloth collides with, in a fixed order: every fitted capsule in world space; the claw
   * segments are sharp at their end.
   */
  capsules(): BodyCapsule[] {
    return this.bones.map((c) => {
      const o = 16 * c.joint;
      const w = this.rig.world;
      const scale = Math.hypot(w[o], w[o + 1], w[o + 2]);
      return { a: this.rig.point(c.joint, c.a), b: this.rig.point(c.joint, c.b), radius: c.radius * scale, active: true, sharp: this.sharp.has(c.joint) };
    });
  }

  /** The joint each capsule of capsules() rides on (development). */
  capsuleJoints(): number[] {
    return this.bones.map((c) => c.joint);
  }

  /**
   * The joint a hold is fixed to: the middle finger's shaft where the claws hook through holes they
   * made, or its claw segment where they close on a rim.
   */
  holdJoint(s: Side, part: 'shaft' | 'claw'): number {
    const [base, tip] = this.hands[s].fingers[1];
    return part === 'shaft' ? base : tip;
  }

  /** Joint world position by name (development). */
  where(name: string): V {
    return this.rig.position(this.rig.joint(name));
  }
}
