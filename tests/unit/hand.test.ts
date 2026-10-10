import { describe, expect, it } from 'vitest';
import { GRIP } from '../../src/simulation/grip';
import { HAND, Hand, PointerTrack } from '../../src/simulation/hand';

type V = [number, number, number];
const H = 1 / 120;
const up: V = [0, 0, 1];
const len = (a: V) => Math.hypot(a[0], a[1], a[2]);
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

describe('hand', () => {
  it('a grab alone moves nothing in the plane; the fingers only lift by the engagement lift', () => {
    const hand = new Hand([0.01, -0.02, 0.003], up);
    expect(hand.target).toEqual([0.01, -0.02, 0.003]);
    for (let k = 0; k < 60; k++) hand.step(H, [0, 0, 0]);
    const d = sub(hand.target, [0.01, -0.02, 0.003]);
    expect(Math.hypot(d[0], d[1])).toBeLessThan(1e-12);
    expect(d[2]).toBeCloseTo(HAND.engageLift, 9);
    expect(hand.age).toBeCloseTo(0.5, 12);
    expect(hand.pinch).toBeCloseTo(GRIP.pinch, 12);
  });

  it('never goes behind the grab plane, whatever the wheel asks', () => {
    const hand = new Hand([0, 0, 0], up);
    hand.wheel(-1);
    expect(hand.depth).toBe(HAND.minDepth);
    for (let k = 0; k < 240; k++) {
      hand.step(H, [0.01 * Math.sin(k / 20), 0, 0]);
      expect(hand.target[2]).toBeGreaterThanOrEqual(-1e-15);
    }
    // a requested drag away from the viewer is held at the plane
    for (let k = 0; k < 240; k++) hand.step(H, [0, 0, -0.02]);
    expect(hand.target[2]).toBeGreaterThanOrEqual(-1e-15);
  });

  it('lifts with the planar drag and follows within the speed and acceleration bounds', () => {
    const hand = new Hand([0, 0, 0], up);
    let prev: V = [...hand.target];
    let prevV = 0;
    let vPrev: V = [0, 0, 0];
    for (let k = 0; k < 120; k++) {
      // a pointer that jumps 10 cm at once
      hand.step(H, [0.1, 0, 0]);
      const v: V = sub(hand.target, prev).map((q) => q / H) as V;
      expect(len(v)).toBeLessThanOrEqual(HAND.maxSpeed * (1 + 1e-9));
      expect(len(sub(v, vPrev))).toBeLessThanOrEqual(HAND.maxAccel * H * (1 + 1e-9));
      prevV = len(v);
      vPrev = v;
      prev = [...hand.target];
    }
    expect(prevV).toBeLessThan(1e-6);
    expect(hand.target[0]).toBeCloseTo(0.1, 6);
    expect(hand.target[2]).toBeCloseTo(HAND.engageLift + Math.min(HAND.maxDragLift, HAND.liftPerDrag * 0.1), 6);
  });
});

describe('pointer track', () => {
  it('is read along its path at substep times, held at the ends, and pruned behind the reader', () => {
    const t = new PointerTrack({ time: 100, x: 0, y: 0 });
    t.add({ time: 110, x: 10, y: 0 });
    t.add({ time: 105, x: 99, y: 99 });
    t.add({ time: 120, x: 10, y: 20 });
    expect(t.at(95)).toEqual({ x: 0, y: 0 });
    expect(t.at(105)).toEqual({ x: 5, y: 0 });
    expect(t.at(115)).toEqual({ x: 10, y: 10 });
    expect(t.at(200)).toEqual({ x: 10, y: 20 });
    t.prune(115);
    expect(t.first).toEqual({ time: 100, x: 0, y: 0 });
    expect(t.at(115)).toEqual({ x: 10, y: 10 });
    expect(t.at(118)).toEqual({ x: 10, y: 16 });
  });
});
