import type { Clip, ClipEvent } from './clip';
import type { Side } from './puppet';

/**
 * "Beneath": something under the cloth claws through it, tears the opening apart, rips strips off its
 * rim, rises into it in the dark, stares into the lens, hauls itself out and lunges; the picture
 * breaks up and freezes. The performance, camera, lights and film are authored in Blender
 * (blender/smilie) and baked into a clip (public/stage/smilie.clip); the cloth is never scripted:
 * claws pierce it and hold it through the mechanics (stage.ts). Times are seconds of simulated time
 * (the cloth and the creature share that clock).
 *
 * Sound and picture run on different clocks: the picture on simulated time (slower than real time
 * where the device cannot keep up), the recordings in real time. Each recording is therefore cued by
 * the picture: the laugh (public/exprerience_sounds/creepy_laugh.mp3, cut by the user to the run as
 * it plays on a desktop) in its four phrases (LAUGH_PIECES, split at its silences), each at the moment
 * of the picture it accompanies there; the ambience (ambiant.mp3) loops from the start and fades out
 * over the final breakup. The clip's laugh markers (an earlier recording's bursts) are not used.
 */

/** Something the stage does at a time (Stage applies them in order). */
export type StageEvent =
  /** A hand's claws hook the cloth they went through (at its claw tips). */
  | { t: number; kind: 'hook'; side: Side }
  /**
   * A hand finds the cloth it closes on at its next grab, `lead` seconds from now (nearest material to
   * where its claws will be, at most `radius` away), and is steered onto it meanwhile.
   */
  | { t: number; kind: 'aim'; side: Side; radius: number; lead: number }
  | { t: number; kind: 'grab'; side: Side }
  /** A hand lets go. */
  | { t: number; kind: 'release'; side: Side }
  /** A sound: a phrase [from, to] (s) of the laugh, the ambience (start, fade out), or signal noise. */
  | { t: number; kind: 'laugh'; from: number; to: number }
  | { t: number; kind: 'ambient' }
  | { t: number; kind: 'ambientFade'; duration: number }
  | { t: number; kind: 'glitch'; duration: number; level: number };

export const BENEATH_CLIP_URL = '/stage/smilie.clip';

/**
 * The picture's pace on the desktop the sound was laid out on: 13.8 s of simulated time in ~21.4 s of
 * wall time (the user's measured run, less the 0.6 s the picture takes to settle after it). Sound
 * lengths given in simulated time (glitch bursts, the ambience's fade) are played over
 * duration * DESKTOP_PACE / pace, the pace measured as it runs: as authored on that desktop, longer
 * where the picture runs slower.
 */
export const DESKTOP_PACE = 13.8 / 21.4;

/** When the laugh starts (s from the start). */
export const LAUGH_AT = 0.5;

/** The laugh's phrases in the recording (s), split at its silences below -40 dB. */
export const LAUGH_PIECES: readonly (readonly [number, number])[] = [
  [0, 4.71],
  [5.54, 9.83],
  [10.76, 16.01],
  [16.72, 20.11],
];

function side(e: ClipEvent): Side {
  const s = e.name.split('.')[1];
  if (s !== 'L' && s !== 'R') throw new Error(`clip event ${e.name} names no hand`);
  return s;
}

/** The stage's events: the clip's timeline markers and the laugh, in time order. */
export function stageEvents(clip: Clip): StageEvent[] {
  const markers = [...clip.events].sort((a, b) => a.t - b.t);
  const events = markers.flatMap((e, k): StageEvent[] => {
    const kind = e.name.split('.')[0];
    switch (kind) {
      case 'hook':
      case 'grab':
      case 'release':
        return [{ t: e.t, kind, side: side(e) }];
      case 'aim': {
        const s = side(e);
        const grab = markers.slice(k + 1).find((g) => g.name === `grab.${s}`);
        if (!grab || e.args.length !== 1) throw new Error(`clip aim at ${e.t} has no radius or grab after it`);
        return [{ t: e.t, kind, side: s, radius: e.args[0], lead: grab.t - e.t }];
      }
      case 'laugh':
        return [];
      case 'glitch':
        if (e.args.length !== 2) throw new Error(`clip glitch at ${e.t} needs duration and level`);
        return [{ t: e.t, kind, duration: e.args[0], level: e.args[1] }];
      default:
        throw new Error(`unknown clip event ${e.name}`);
    }
  });
  // each phrase where the picture is when it is heard on the desktop the laugh was cut on
  for (const [from, to] of LAUGH_PIECES) events.push({ t: LAUGH_AT + from * DESKTOP_PACE, kind: 'laugh', from, to });
  events.push({ t: 0, kind: 'ambient' });
  // the ambience fades out over the final breakup (the last glitch, which runs to the end)
  const breakup = events.reduce<StageEvent | null>((last, e) => (e.kind === 'glitch' && (!last || e.t >= last.t) ? e : last), null);
  if (breakup?.kind === 'glitch') events.push({ t: breakup.t, kind: 'ambientFade', duration: breakup.duration });
  return events.sort((a, b) => a.t - b.t);
}
