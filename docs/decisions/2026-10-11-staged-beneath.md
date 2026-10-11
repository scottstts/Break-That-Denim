# Staged experience "Beneath" (2026-10-11)

A scripted horror sequence on top of free play: a creature under the coupon claws through it, tears it
open, rises into the opening in the dark, pushes its head up through it and stares (the shock: the face
from ~4.8 s to the lunge), then lunges and the camera is pulled onto its face; the picture ends
frozen and glitched (still the picture: no snow, no black); free play resumes on intact cloth.

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
- **No body part goes fast through intact cloth.** With this engine a kinematic body driven quickly
  through intact denim fails it everywhere at once (a thread net); the same push done slowly tents or
  tears along cracks (the head pushing through: 0 ruptures; a 14 cm shoulder rise over 1.7 s: ~50, the
  sheet lifting into a mound). The choreography keeps to that: openings are made by claws (pierce, hook,
  pull), the head comes up through the opening it fits, its snap back is eased over 0.14 s (whipped in
  0.06 s it sometimes flung the rim hugging its head), and the climax moves the camera onto the face
  instead of the shoulders and arms through the sheet (they are wider and longer than the opening).
  Checked with `TRACE=1` (ruptures per 0.1 s): ~1.9k for the run, peaks ~570 at the tear-apart, none
  above 100 after it; before, ~16k with 4-12k spikes. The rim rips were dropped (user: performative,
  they barely changed the opening); the head still fits through without them.
- **The long arms need room.** With its shoulders just under the cloth, a hand at the cloth near its head
  is out of reach (creature.py reports claws missing by 23-46 cm) and the arm lands wherever the solve
  settles: hand keys there were the cause of nets, not the speed alone. Read the solve report after any
  hand change.
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
- **The camera carries a light.** A cool LED on the camera (authored in Blender as `Torch`, parented to
  the shot camera, 0.25 W/sr), on the bulb's failing supply but without its thermal lag (the driver cuts
  out), so blackouts and the strobe stay dark. It sits at the lens: nothing the camera sees is in its
  shadow, so it is unshadowed (no shadow map), and its inverse square makes the face brighter the closer
  it comes. It fills the eye sockets the bulb (behind the top edge) leaves black; on the cloth it adds ~5 %
  of the bulb's irradiance. Renderer `torch*` fields; zero in free play.
- **The jaw follows the laugh as heard.** The clip carries the jaw's keyed opening and how far the laugh
  may open it (`jawOpen`, `jawLaugh`, the opening axis in the header); the main thread measures the
  recording's loudness envelope (audio/envelope.ts) and when playback started, and the worker opens the
  jaw from that clock, so lip sync holds however far the simulation lags real time.
- **Glitches switch, not ease.** The film's glitch curve is constant-interpolated: a Bezier between the
  last burst and the breakup crept from 0 to 0.14 over the stare (the black macroblocks in the browser).
- **The laugh is one recording on the wall clock.** The user cut `creepy_laugh.mp3` (20.7 s) to the
  experience as it plays in the browser (~22 s wall time: the simulation runs below real time). It starts
  at `LAUGH_AT` (0.5 s) and plays whole at its own level; the natural end does not stop it (Reset or a
  restart does). The clip's laugh markers (bursts of the earlier recording) are not used.
- **The ambience runs under everything.** `ambiant.mp3` (25 s) starts with the stage's first substep at
  its own level and fades out linearly over the final breakup (the clip's last glitch, whose sound
  plays meanwhile), then stops; Reset or a restart stops it at once.
- **Free play is untouched** when no experience runs: product key/fill values reproduce the previous
  shading exactly; the pierce pass and body terms dispatch nothing without a body.

## Open

- The laugh (0 dBFS peak) and the ambience (-2.5 dBFS) play at their own levels (user decision); with
  the tear voice (limited at -3 dBFS) an offline mix of a run peaks at +4.3 dBFS, so loud moments clip
  at the output.
- Engine: fast kinematic loading through intact cloth bursts it into a thread net instead of cracks
  (not tunnelling: torso capsules move 0.3 of their radius per substep, `tools/stage/sweepprobe.ts`).
  The user will fix this in the engine; the choreography avoids it until then.
- Blender's preview has no laugh-driven jaw (the runtime adds it).
- Cuts made by piercing emit no fracture sound (only the field update's damage does).
- Not yet watched in a live browser (worker glue: fetch, `createImageBitmap`, menu, audio cues).
- The model is CC-BY-4.0 (Bento, Sketchfab): attribution is required wherever it ships.
