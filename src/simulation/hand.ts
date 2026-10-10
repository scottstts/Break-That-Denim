import { GRIP, pinchScale } from './grip';

type Vec3 = [number, number, number];

/** How a finger's handle follows the pointer (SI). */
export const HAND = {
  /** Largest handle speed, m/s. */
  maxSpeed: 2.5,
  /**
   * Largest handle acceleration, m/s^2 (a quick flick of the wrist): a pointer that jumps or stops
   * dead does not jerk the cloth within one substep.
   */
  maxAccel: 60,
  /** The fingers lift the cloth toward the viewer by this much as they close (m). */
  engageLift: 0.002,
  /**
   * A hand that pinches cloth lifts what it pulls: the handle rises toward the viewer by this fraction
   * of its drag from the grab point, up to maxDragLift (the wheel adds depth on top).
   */
  liftPerDrag: 0.35,
  maxDragLift: 0.05,
  /** Wheel depth range (m, toward the viewer); the handle never goes behind the grab plane. */
  minDepth: -0.03,
  maxDepth: 0.1,
};

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

/**
 * One finger's handle, in simulation time. The hand starts exactly on the picked material (a grab
 * alone moves nothing); the requested drag (from the pointer, or a script) is lifted toward the viewer,
 * never pushed behind the grab plane, and followed with bounded speed and acceleration. The closing
 * pinch (grip.ts pinchScale) runs on the same clock as the mechanics, so engagement does not depend on
 * how substeps are batched into frames.
 */
export class Hand {
  /** World position of the picked material at the grab; targets are relative to it. */
  readonly anchor: Vec3;
  /** Unit direction toward the viewer: the visible side of the picked surface. */
  readonly toward: Vec3;
  /** Simulated time since the grab, s. */
  age = 0;
  /** Wheel depth toward the viewer, m. */
  depth = 0;
  private offset: Vec3 = [0, 0, 0];
  private vel: Vec3 = [0, 0, 0];

  constructor(anchor: Vec3, toward: Vec3) {
    this.anchor = [...anchor];
    const l = len(toward);
    this.toward = scale(toward, 1 / l);
  }

  /** Requested handle position (world), for Interaction.setTarget. */
  get target(): Vec3 {
    return add(this.anchor, this.offset);
  }

  /** Spread kept by the closing pinch now. */
  get pinch(): number {
    return pinchScale(this.age);
  }

  /** Engagement of the fingers, 0..1. */
  get engagement(): number {
    return Math.min(1, this.age / GRIP.engageTime);
  }

  wheel(dDepth: number): void {
    this.depth = Math.max(HAND.minDepth, Math.min(HAND.maxDepth, this.depth + dDepth));
  }

  /** Where the handle is asked to be for a requested drag (m, from the grab point), relative to the anchor. */
  goal(drag: Vec3): Vec3 {
    // the in-plane part of the drag sets the automatic lift (a component toward the viewer is already lift)
    const along = drag[0] * this.toward[0] + drag[1] * this.toward[1] + drag[2] * this.toward[2];
    const planar = len(sub(drag, scale(this.toward, along)));
    const auto = HAND.engageLift * this.engagement + Math.min(HAND.maxDragLift, HAND.liftPerDrag * planar);
    // the visible side only: the total offset toward the viewer never goes negative
    const lift = Math.max(-along, auto + this.depth);
    return add(drag, scale(this.toward, lift));
  }

  /** Advance one substep of length h toward the requested drag. */
  step(h: number, drag: Vec3): void {
    this.age += h;
    const d = sub(this.goal(drag), this.offset);
    const dist = len(d);
    // fastest approach that can still stop at the goal within the acceleration bound
    const speed = Math.min(HAND.maxSpeed, Math.sqrt(2 * HAND.maxAccel * dist), dist / h);
    const want: Vec3 = dist > 0 ? scale(d, speed / dist) : [0, 0, 0];
    const dv = sub(want, this.vel);
    const dvl = len(dv);
    const maxDv = HAND.maxAccel * h;
    this.vel = dvl > maxDv ? add(this.vel, scale(dv, maxDv / dvl)) : want;
    this.offset = add(this.offset, scale(this.vel, h));
  }
}

export interface PointerSample {
  /** Event time on the shared clock (ms). */
  time: number;
  x: number;
  y: number;
}

/**
 * A pointer's timestamped positions since it went down. The engine reads it at each substep's input
 * time, so a frame's substeps follow the pointer's path through that frame instead of all jumping to its
 * latest position, whatever the frame partition.
 */
export class PointerTrack {
  private readonly samples: PointerSample[];
  /** Input time at which the pointer went up (Infinity while down). */
  upTime = Infinity;

  constructor(down: PointerSample) {
    this.samples = [down];
  }

  get first(): PointerSample {
    return this.samples[0];
  }

  get last(): PointerSample {
    return this.samples[this.samples.length - 1];
  }

  add(s: PointerSample): void {
    // events arrive in order; a stale one changes nothing
    if (s.time < this.last.time) return;
    this.samples.push(s);
  }

  /** Position at input time `t`, linearly interpolated; held at the ends. */
  at(t: number): { x: number; y: number } {
    const s = this.samples;
    const end = t >= s[s.length - 1].time ? s[s.length - 1] : t <= s[0].time ? s[0] : null;
    if (end) return { x: end.x, y: end.y };
    let lo = 0;
    let hi = s.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (s[mid].time <= t) lo = mid;
      else hi = mid;
    }
    const a = s[lo];
    const b = s[hi];
    const w = b.time > a.time ? (t - a.time) / (b.time - a.time) : 1;
    return { x: a.x + (b.x - a.x) * w, y: a.y + (b.y - a.y) * w };
  }

  /** Forget samples no later read can need (all but the first and those after `t`'s bracket). */
  prune(t: number): void {
    const s = this.samples;
    let k = 1;
    while (k + 1 < s.length && s[k + 1].time <= t) k++;
    if (k > 1) s.splice(1, k - 1);
  }
}
