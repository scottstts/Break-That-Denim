# Staged experience "Beneath" (2026-10-11)

A scripted horror sequence on top of free play: a creature under the coupon claws through it, tears it
open, rips its rim, rises into the opening in the dark, stares, hauls itself out and lunges; the picture
ends frozen and glitched (still the picture: no snow, no black); free play resumes on intact cloth.

## Choices that matter beyond the code

- **Authored in Blender, baked to a clip.** The performance, camera, bulb, pit glow, film and event
  markers live in `blender/smilie/smilie.blend` (scripts there; approved by the user before baking).
  `export.py` bakes per frame into `public/stage/smilie.clip` (joint local rotations in glTF terms of
  `public/models/smilie.glb`, root translation, camera, lights, film; markers as events). The runtime
  only samples it on the simulation clock. Re-bake after any Blender change; the clip and the glb must
  come from the same rig.
- **Bind offsets are measured, not read from the glb.** For 13 joints the glb's node translations differ
  from its inverse bind matrices (the bind pose Blender imports). The bake stores the offsets as Blender
  has them and fails if any joint's offset moves during the clip.
- **The cloth is never faked.** Claws pierce through the mechanics: a node under a sharp capsule's point
  (past its end, within `CLAW_PIERCE.radius` of its axis) loaded past `CLAW_PIERCE.force` (4 N) has its
  crossings cut (body.wgsl `pierce`, after commit, before the damage update). Only the nail's point
  pierces: fingertip capsules are ~7 mm in radius, and testing the whole cap cut neighbouring nodes too
  (a blunt fingertip acting as a point). Hooks and rim grabs then hold through grips (`CLAW_HOOK`).
- **Held cloth rides on the claw.** Nodes in a claw's hold are exempt from the body's contact and
  piercing; on release they stay exempt until clear of the body (GPU-side), otherwise they were kicked
  out of the capsule (or re-pierced) the moment the claw let go.
- **Lighting follows the authored scene, with one physical substitution.** The bulb drives the renderer's
  key (position, aim, cone with inner edge, no spill, colour; `keyInner/keyLeak/keyColor` default to the
  product's exact look). The fill is the scene's World radiance (what lit the creature in Blender's
  preview), carried by the bulb as the light its lit cloth and room bounce: it follows the bulb's power
  and is gone in the blackouts. Measured against Blender renders of the same frames: face and shoulder
  levels match within a few sRGB steps. Blender's preview never showed the cloth shadowing the creature
  (the proxy is render-hidden), so moments inside the opening are darker in the runtime by design.
- **The laugh is one recording on the wall clock.** The user cut `creepy_laugh.mp3` (20.7 s) to the
  experience as it plays in the browser (~22 s wall time: the simulation runs below real time). It starts
  at `LAUGH_AT` (0.5 s) and plays whole at its own level; the natural end does not stop it (Reset or a
  restart does). The clip's laugh markers (bursts of the earlier recording) are not used.
- **Free play is untouched** when no experience runs: product key/fill values reproduce the previous
  shading exactly; the pierce pass and body terms dispatch nothing without a body.

## Open

- The jaw in the clip follows the earlier recording's bursts, not the new laugh (re-sync in Blender).
- The new laugh peaks at 0 dBFS; summed with the tear voice (limited at -3 dBFS) a run's mix peaks at
  +3 dBFS where tearing meets a loud passage.

- The creature is kinematic (unbounded force). Where the clip drives body parts through intact cloth the
  sheet fails everywhere at once (thread net) instead of cracking from the opening: briefly as the
  knuckles follow the claws (0.7 s), and fully at the lunge (11.45-11.7 s: ruptures 7k -> 15k, most of
  the coupon freed; the shoulders rise through cloth around a head-sized opening). Torso capsules move
  0.3 of their radius per substep there (`tools/stage/sweepprobe.ts`), so this is not tunnelling.
  Two-way coupling (the body yielding to cloth resistance) or a converged solve under fast kinematic
  loading is the mechanical fix; same family as the user's noted "exploding threads".
- Cuts made by piercing emit no fracture sound (only the field update's damage does).
- Not yet watched in a live browser (worker glue: fetch, `createImageBitmap`, menu, audio cues).
- The model is CC-BY-4.0 (Bento, Sketchfab): attribution is required wherever it ships.
