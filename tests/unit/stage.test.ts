import { describe, expect, it } from 'vitest';
import { hookPatch } from '../../src/simulation/grip';
import { DENIM } from '../../src/simulation/material';
import { materialAt, nodeAt } from '../../src/simulation/materialPoint';
import { compileScene, defaultSceneSpecs } from '../../src/simulation/scene';
import { DESKTOP_PACE, LAUGH_AT, LAUGH_PIECES, stageEvents } from '../../src/stage/beneath';
import { Clip } from '../../src/stage/clip';
import { quat, v3, type V } from '../../src/stage/rig';
import { CLAW_HOOK } from '../../src/stage/claws';
import clipData from '../../public/stage/smilie.clip?inline';

const scene = compileScene(defaultSceneSpecs());
const L = scene.layers[0];

describe('material points', () => {
  it('maps a world point back to the material point the mount placed there', () => {
    const n = L.nodeOffset + 60 * L.nx + 90;
    const [u, v] = materialAt(scene, scene.position[3 * n], scene.position[3 * n + 1]);
    expect(u).toBeCloseTo(scene.restUV[2 * n], 9);
    expect(v).toBeCloseTo(scene.restUV[2 * n + 1], 9);
    expect(nodeAt(scene, u, v)).toBe(n);
  });
});

describe('claw hooks', () => {
  it('hold the union of a disc per claw, bounded to the hook force at the largest lead', () => {
    const claws = [0, 5, 10].map((d) => L.nodeOffset + 60 * L.nx + 80 + d);
    const p = hookPatch(scene, claws, CLAW_HOOK)!;
    expect(new Set(p.nodes).size).toBe(p.nodes.length);
    for (const c of claws) expect(p.nodes).toContain(c);
    const W = p.weights.reduce((a, b) => a + b, 0);
    expect(p.stiffness * CLAW_HOOK.maxLead * W).toBeCloseTo(CLAW_HOOK.maxForce, 6);
    // every member is within a claw's radius of some claw (material distance)
    for (const n of p.nodes) {
      const d = Math.min(...claws.map((c) => Math.hypot(scene.restUV[2 * n] - scene.restUV[2 * c], scene.restUV[2 * n + 1] - scene.restUV[2 * c + 1])));
      expect(d).toBeLessThan(CLAW_HOOK.radius);
    }
    // what another hold has is not taken
    const taken = new Set(p.nodes);
    expect(hookPatch(scene, claws, CLAW_HOOK, (n) => !taken.has(n))).toBeNull();
  });

  it('uses a material made of individual yarns (sanity of the coupon used above)', () => {
    expect(L.weave).toBe(DENIM);
  });
});

describe('rig rotations', () => {
  const close = (a: V, b: V) => a.forEach((x, k) => expect(x).toBeCloseTo(b[k], 9));
  it('shortest arc and frame alignment', () => {
    const a = v3.norm([0.3, -0.2, 0.9]);
    const b = v3.norm([-0.5, 0.7, 0.1]);
    close(quat.rotate(quat.fromTo(a, b), a), b);
    close(quat.rotate(quat.fromTo(a, v3.scale(a, -1)), a), v3.scale(a, -1));
    const f: V = [0, 0, 1];
    const u: V = [0, 1, 0];
    const F = v3.norm([1, -0.2, 0.3]);
    const U = v3.norm(v3.cross(v3.cross(F, [0, 0, 1]), F));
    const q = quat.basis(f, u, F, U);
    close(quat.rotate(q, f), F);
    close(quat.rotate(q, u), U);
  });
});

describe('Beneath clip', () => {
  // the clip as baked from Blender, inlined by the bundler as a data URL
  const clip = new Clip(Uint8Array.from(atob(clipData.slice(clipData.indexOf(',') + 1)), (c) => c.charCodeAt(0)).buffer);
  const events = stageEvents(clip);

  it('samples unit joint rotations and passes through its frames', () => {
    for (const t of [0, 0.37, 5.01, clip.duration, clip.duration + 1]) {
      const r = clip.sample(t).rotations;
      for (let j = 0; j < clip.joints.length; j++) expect(Math.hypot(r[4 * j], r[4 * j + 1], r[4 * j + 2], r[4 * j + 3])).toBeCloseTo(1, 9);
    }
    const a = [...clip.sample(100 / clip.fps).eye];
    const b = [...clip.sample(101 / clip.fps).eye];
    const m = clip.sample(100.5 / clip.fps).eye;
    m.forEach((x, k) => expect(x).toBeCloseTo((a[k] + b[k]) / 2, 9));
  });

  it('orders its events, ends every hold, and grabs only after aiming', () => {
    events.forEach((e, k) => k > 0 && expect(e.t).toBeGreaterThanOrEqual(events[k - 1].t));
    for (const side of ['L', 'R'] as const) {
      let holding = false;
      let aimed = -Infinity;
      for (const e of events) {
        if (!('side' in e) || e.side !== side) continue;
        if (e.kind === 'aim') {
          expect(holding).toBe(false);
          expect(e.lead).toBeGreaterThan(0);
          aimed = e.t + e.lead;
        }
        if (e.kind === 'hook' || e.kind === 'grab') {
          expect(holding).toBe(false);
          holding = true;
          if (e.kind === 'grab') expect(e.t).toBeCloseTo(aimed, 9);
        }
        if (e.kind === 'release') {
          expect(holding).toBe(true);
          holding = false;
        }
      }
      expect(holding).toBe(false);
    }
  });

  it('cues the laugh phrase by phrase where they fall on the desktop the laugh was cut on', () => {
    const laughs = events.filter((e) => e.kind === 'laugh');
    expect(laughs.map((e) => (e.kind === 'laugh' ? [e.from, e.to] : []))).toEqual(LAUGH_PIECES.map((p) => [...p]));
    expect(laughs[0].t).toBe(LAUGH_AT);
    // at that pace each phrase starts where the whole recording, started at LAUGH_AT, had it, and after
    // the one before it has ended
    laughs.forEach((e, k) => {
      if (e.kind !== 'laugh') return;
      expect((e.t - LAUGH_AT) / DESKTOP_PACE).toBeCloseTo(e.from, 9);
      const prev = laughs[k - 1];
      if (prev?.kind === 'laugh') expect(e.t).toBeGreaterThanOrEqual(prev.t + (prev.to - prev.from) * DESKTOP_PACE);
    });
  });

  it('starts the ambience with the scene and fades it out over the final breakup', () => {
    expect(events[0]).toEqual({ t: 0, kind: 'ambient' });
    const glitches = events.filter((e) => e.kind === 'glitch');
    const last = glitches[glitches.length - 1];
    const fade = events.filter((e) => e.kind === 'ambientFade');
    expect(fade).toEqual([{ t: last.t, kind: 'ambientFade', duration: last.kind === 'glitch' ? last.duration : NaN }]);
    expect(last.t + (last.kind === 'glitch' ? last.duration : 0)).toBeCloseTo(clip.duration, 1);
  });

  it('opens on a clean picture and ends on a frozen, broken one that is still the picture', () => {
    const s0 = clip.sample(0);
    expect(s0.glitch).toBe(0);
    expect(s0.freeze).toBe(false);
    const end = clip.sample(clip.duration);
    expect(end.freeze).toBe(true);
    expect(end.glitch).toBeGreaterThan(0);
    expect(end.exposure).toBeGreaterThan(0);
  });
});
