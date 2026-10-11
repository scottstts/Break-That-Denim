# Handoff (rolling)

Two-hand hypothesis verified (2026-10-11): nonblocking staged audit reproduces the floating 208-node
centre piece. It separates from mounts/both claw cores at 2.1667 s and all 208 nodes are correctly
detached with one valid GPU component ID by 2.1833 s; this stays true through 6.1 s while it floats.
No lingering anchor or missed classification found. New standalone
`tests/gpu/twohand-topology.ts`: 18/18 checks, including overlapping analysis/cuts/releases and zero
grip force/stiffness from lingering claw rim entries on detached material. Lint/typecheck and 67 unit
tests pass. Details/repro in the two-hand verification section of
`docs/decisions/2026-10-11-folded-fragment-contact.md`. No production fix or commit.

Follow-up investigation (2026-10-11, still unresolved): mapped the visible staged island near pixel
(437, 357) in `investigation-async2-6` to the same 208 detached mechanical nodes, then compared the
identical state with no body, a frozen body, and the animated body. With animated body/current
contacts it ends 10.15 mm above its start after 2 s, barely descending after body contact has ended.
With only its internal contacts ablated it clears the animated body and exits by 1 s. Thus staged
collisions contribute the lifting, while internal-contact resistance sustains the apparent hover.
Actual blend-skinned triangle distances also confirm capsule support off the visible skin, but
body-only support is transient in this replay. See the follow-up section in
`docs/decisions/2026-10-11-folded-fragment-contact.md`. Production code was not changed or committed.

Latest investigation (2026-10-11, floating scraps still unresolved):
`docs/decisions/2026-10-11-folded-fragment-contact.md`. A folded 208-node staged fragment with the
current rigid-mode change still floats after the creature, grips and all other material are removed.
Internal contact's diagonal stiffness and absolute displacement bounds suppress common motion:
1.44 mm fall in 0.1 s versus 51.36 mm when only that component's internal contacts are ablated.
`tools/studies/foldedfall.ts` reproduces and isolates both defects. No production fix applied in this
investigation. The earlier "no hovering scraps" conclusion below was too broad; collider refitting
alone does not resolve this failure.

Latest (2026-10-11, floating torn pieces): `docs/decisions/2026-10-11-detached-rigid-modes.md`
(uncommitted). Detached pieces hovered because PCG resolved ~4 % of their rigid motion (lace-like
pieces lose coarse support). Topology now labels pieces; assembly writes their rigid blocks (rigidK);
rigid_frame.wgsl builds E = Z^T A Z per piece; PCG starts from each piece's rigid response
(rigid_apply.wgsl). Verified headless: `npm run check` 67 unit, `npm run test:gpu` 26/26 + 19/19 + 7/7
(new comb / serpentine cut-outs). Beneath: no hovering scraps; sheet artifacts/captures/beneath-falling-pieces.png.
Not seen in a browser.
- Remind the user: floor collision for free play (falling pieces pass through the render-only floor).
- Still waiting on the user: re-fitting the creature's colliders (1-5 cm off the skin,
  tools/stage/colliderfit.ts); pieces resting on them still float.

Latest (2026-10-11, line-search stalls): `docs/decisions/2026-10-11-pinch-slip.md` (uncommitted).
The stalls were a grip defect, not a solver one: rim entries torn away from the pinch stayed sprung to
the hand from any distance. Pinch entries now slip out beyond a pad radius of the patch's own motion;
claws never slip. Rejections 0 in every gesture (haul 80, hold 82, haul@0.006,-0.004 26 before); tear-outs
tear less and settle after release. Beneath is bit-identical. setTarget's entry count fixed (it included
the strip between the pads). Verified headless: `npm run check` 67 unit, `npm run test:gpu` 26/26 +
19/19 + 5/5. Not seen in a browser.
- The previous handoff's "g.d > 0" lead was a probe artifact (PCG overwrites the rhs it read); LSPROBE
  now captures the rhs before PCG, and LSCHECK=1 adds the operator check.

Before (2026-10-11, open items of the assessment round): `docs/decisions/2026-10-11-parted-crossings-and-swept-body.md`.
A ruptured or slipped-out span's yarns are pulled out of the other side's crossings (both backends;
couple split into coupleU / coupleV; SEG_SLIPPED defers the slip case one substep). Body contact is
swept over the substep. maul thread ends > 5 mm off 41 -> 0 (worst 24.5 -> 2.3 mm). With the pinch
fix, parting does not change how much tears (the increase first measured was the grip defect).
- **For the user:** "Beneath" opens one clean hole around the mouth at 6 s (901 ruptures; previous
  build 938, with material bridging the hole): sheet `artifacts/captures/beneath-swept-parting.png`.
  The re-staging decision from the round below may be moot; it is theirs. Shipped drape still kept.

Latest (2026-10-11, assessment round): `docs/decisions/2026-10-11-detached-coherence.md` (acting on
artifacts/assessment; uncommitted). Detached pieces stay in the cloth solve (no more ballistic
nodes); active contact bounds in the Newton operator; cubic contact penalty with a per-iteration
depth trust region (free light bodies chattered forever); objective bending viscosity (both backends).
Verified headless: `npm run check` 65 unit, `npm run test:gpu` 26/26 + 13/13 + 5/5 (new
tests/gpu/detached.ts). maul: worst detached span 1.075x rest (was 84.6x), 0 stretched detached runs
(were 147). Not seen in a browser.
- **Decision for the user: "Beneath" tears less.** The lunge's extra tearing was mostly the bound clip
  stretching spans (old build: maul 581 ruptures at 2x2, 618 converged; new 285 / 229). The staged hole
  is now smaller and the upper face stays behind cloth at 6-9 s (938 vs 1866 ruptures by 13.7 s; sheet:
  run `tools/stage/beneath.ts` with FRAMES=6,9,12 in both builds). Re-staging (claw cuts, clip) is
  theirs to approve; physics was not bent back.
- Open: attached thread ends drawn 13-26 mm off their chain where a parted span's yarn field stays
  intact (decision record; `tools/studies/runends.ts <state>`). Shipped drape kept (re-settling with
  the new viscosity gives another wrinkle state; it is still an exact rest state).
- Tools: `tools/studies/restcontact.ts <state>` (does a dumped state come to rest),
  `releasepiece.ts`, session `FASTEST=n` (fastest nodes with state and contact bounds).

Latest (2026-10-11, round 2): "Beneath" re-staged around the head-through-the-hole stare (user: the
shock hinges on seeing the full face; no thread explosions anywhere). Blender scene re-animated and
saved (one .blend, the latest: no version copies kept); rim rips dropped, head snap eased (round 3);
runtime: camera light, laugh-driven jaw, clip v2. Decision record updated (same file). Verify with
`TRACE=1 node tools/gpu/run.mjs tools/stage/beneath.ts` (ruptures per 0.1 s), `LIT=1` + `CAM` for side
views, `tools/stage/clawprobe.ts`; review sheet `artifacts/captures/beneath-stare-review.png`. Re-bake:
background Blender runs animate.py, saves, runs export.py (no live Blender needed).

Latest (2026-10-11): "Beneath" now plays the Blender-approved clip (user approved the poses; the old
source model `smily_horror_monster.glb` was removed at their request; `public/models/smilie.glb` is the
model, gitignored). Decision record: `docs/decisions/2026-10-11-staged-beneath.md`.
- Bake: `blender/smilie/export.py` (inside Blender, or `Blender -b smilie.blend --python` a runpy wrapper;
  the file needs no save) -> `public/stage/smilie.clip`. `animate.py` no longer keys `ambient` (export
  derives the fill from the World and the bulb). The user's Blender MCP socket was closed after the bake.
- Runtime: `src/stage/clip.ts` (sampler), `beneath.ts` (events from markers), `stage.ts` (per substep:
  play + aim corrections + body pose + hold targets; per frame: camera, bulb as key, fill, pit, film),
  `puppet.ts` (bind/play/reach), body piercing `src/simulation/gpu/body.ts` + `body.wgsl` (solver
  `postCommit`), key light inner cone/spill/colour in the renderer, post without snow/blackout.
- Verified headless: `npm run check` (64 unit), `npm run test:gpu` 26/26 + 13/13 (new: pierce cuts
  exactly the crossings under the point; blunt and held nodes none), full runs 13.8 s, build ships clip
  and model. Pose 17 us/substep. Not watched in a browser yet (rule: headless only during build).
- Top open item: the lunge shreds the coupon (kinematic body through intact cloth; see the record's
  Open). Tools: `tools/stage/beneath.ts` (LIT=1 flat debug light, CAM debug camera), `clawprobe.ts`,
  `sweepprobe.ts`, `coverprobe.ts`, `posebench.ts`.

Latest: `docs/decisions/2026-10-11-staged-beneath.md`: staged horror experience "Beneath" (menu button
beside Reset; src/stage/*, body collider src/simulation/gpu/body.ts, claw puncture src/simulation/puncture.ts,
post grade/glitch in post.wgsl; uncommitted). Verified headless only: `npm run check` 66 unit,
`npm run test:gpu` 26/26 + 12/12, full runs with 0-5 line-search rejections. Not yet seen in a browser.
Reproduce: `node tools/gpu/run.mjs tools/stage/beneath.ts <name>` (FRAMES, VIDEO=1, AUDIO=f.json, CAM,
NOCREATURE, UNTIL), `tools/stage/audio.ts f.json out.wav`, `tools/stage/coverprobe.ts <t>`,
`tools/stage/posebench.ts`, `tools/stage/creaturestill.ts` (POSE, PLACE, SCALE). Model + laugh live in
public/models (gitignored) and public/exprerience_sounds.

Latest round: `docs/decisions/2026-10-10-stability-round.md` (acting on artifacts/assessment: neutral grab,
visible picking, connected patch, shared hand on the simulation clock, release refinement against the
numerical bounce, relative/swept contact bounds, contact across tears, deterministic contact; uncommitted).
Before: `docs/decisions/2026-10-10-floor-and-room-light.md` (render-only floor 12 cm below, mount board,
dim overhead lamp, 8K base/normal cropped to the visible floor).
Before: `docs/decisions/2026-10-10-tear-sound-sample.md` (tear sound: the recording's tearing passages
played back; the mechanics control only tearing on/off, loudness and place; fall-off on stop/release; uncommitted).
Before: `docs/decisions/2026-10-10-tear-sound-structure.md` (the procedural voice it replaced).
Before: `docs/decisions/2026-10-10-tear-sound.md` (records, transport, multi-touch, page UI).
Before: `docs/decisions/2026-10-10-folds-and-free-tails.md`, `docs/decisions/2026-10-10-edge-fibre-population.md`,
`docs/decisions/2026-10-10-interaction-round.md`.

## State

- Mechanics: unequal-span bending, gradient-consistent line search with rounding-bounded step choice,
  cancellation-free bending energy changes, force-derived free-end pull-out, wear-independent cohesive
  opening, debris bounded against the cloth, contact query culling.
- Scene: coupon stretched 0.2 % along the warp (user decision); `MECHANICS_VERSION` 10.
- Grip: two-pad pinch (fingertips close along the warp; the strip between folds freely; user decision,
  replaces the radial contraction), drag lift, bounded force; holds until release; torn-out pieces stay
  held. Pinch is the only loading (user decision): slit and eye openings are the correct result for it. Grabs pick
  only visible material, start exactly on it, never push behind the grab plane (src/simulation/hand.ts);
  4 Newton iterations for 0.1 s after a release (RELEASE_REFINEMENT). Releasing frees what only that
  grip held before the fingers open (Interaction.letGo; a crushed held piece was flung apart).
- Contact: relative (node and face corners) directional bounds against the first face a path crosses
  or the nearest start face; lips across tears collide inside the lattice exclusion window.
- Rendering: threads from the strand chains, split where the mechanics ruptured their bundle; flat clumps
  of freed yarns; free tails solved as heavy elastic rods resting on the cloth (`thread_tail.wgsl`); edge
  fibre from the fibre population (fans and rim hair, per fibre or closed form);
  rim/halo, footprint-filtered albedo, top key light. Colour, exposure and fill were measured against the
  reference and match; no shading change is justified.

## Verified

- Stability round: `npm run check` 53 unit, `npm run test:gpu` 26/26 + 12/12 (tests/gpu/interaction.ts).
  Release on intact cloth: no rebound (was 11 %, 2 rings). Sessions deterministic again; 0 post-release
  ruptures; maul 5 rejections. Substep +1.8 %. Not yet felt in the browser by the user.

- Tear sound: renders of haul / pluck / drag / rip sessions keep the recording's texture (fine texture
  distance 2.8, the recording's own tears 3.0-7.6); silent unless yarns part; falls off 60 dB in 0.3 s on
  stop or release (unit tests). Not yet confirmed by the user by ear.
- Multi-touch: `twohand.ts` apart/shear/overlap/adjacent/three clean; `npm run check` 45 unit.

- `npm run check` (33 unit), `npm run test:gpu` (26/26).
- Session `settle`: 0 line-search rejections (was 164), settles 0.73 s after release.
- `haul` (fast long drag, AUTOLIFT=1): grip held on worn and intact cloth; worn scrap stays in the hand.
- `maul` (four hard drags, AUTOLIFT=1): no non-finite values, 2 rejections in 6.5 s, no bridging threads,
  21 of 40k sheet spans through a cell. Without the slip release a 19 cm drag at full force tears out a
  large worn piece (freed 368 nodes), which falls when released.

## Reproduce

- Stability: `tools/studies/release.ts` (GRAB, PULL, SOLVER, ITERS/WINDOW/MAXITERS, TRACE, STRAIN),
  `tools/studies/stabilityAssessment.ts` (assessment probe; GRIP override), `tools/studies/benchmoving.ts`,
  `session.ts peel` (folded flap grabbed and lifted), `HAND` env override in session.ts; session `ASYNC=1`
  (connectivity analysis alongside the substeps, as the worker) with `FREED=1` (pieces freed since the
  release / per frame and their speeds); `tools/studies/freepiece.ts` (unanchored / hinged piece).

- Sound: `AUDIO=f.json node tools/gpu/run.mjs tools/gpu/scripts/session.ts rip` (or `twohand.ts`), then
  `tools/audio/render.ts f.json out.wav` (`VOICE` overrides TEAR_SAMPLE); compare with
  `tools/audio/spectrograms.mjs`, `texture.mjs`, `events.mjs`, `compare.mjs`, `analyze.mjs`.

- `node tools/gpu/run.mjs tools/gpu/scripts/session.ts <script> NAME 30` (`DUMP=1`, `SNAPS=t,...`,
  `FRAMES=dir`, `STATS=1`, `TRACK=n`, `LSPROBE=t`); `haul` with `GRAB`, `PULL`, `HOLD`, `AUTOLIFT=1`.
- `tools/studies/crossings.ts <state>`: spans through intact cells; `tools/studies/straightruns.ts <state>`:
  thread runs drawn without a carrying chain.
- `still.ts` with `CAMERA`, `LIGHT` (incl. `background`), `RENDERCOST=1`; `tools/compare/sbs.zsh`;
  `SCENE` / `DRAPE` overrides; `bench.ts`.
- `session.ts hold` (held pull, `GRAB`, `PULL`, `HOLD`; `DUMP=1`) + `tools/studies/folds.ts <state> <grab x,y>`:
  wrinkles and in-plane vs buckled gathering around a grip. `tools/studies/tails.ts <state>`: free tails
  through the cloth (solved vs the old procedural hang).
- `tools/compare/coverage.mjs`: fringe coverage by distance from the edge, from two stills over different
  backgrounds. `tools/compare/fuzzband.mjs`: loose-material band vs a reference frame (its edge length
  grows with fuzz, so compare coverage for our own changes).

## Open

- Stability round leftovers (decision record "Open"): grip force above 120 N at the iterate; no
  edge-edge/strand contact; torn cells have no collision face; rare crack-tip line-search rejections.
  Pad pinch not yet seen by the user in the browser.

0. Sound: verify by ear in the browser (reload: the worklet loads once per page). Loudness follows the
   mechanics' release rate (swells within a tear are the simulated cloth's); the slices come from 2 s of
   recording, so a long tear repeats material at random places. Mouse wheel depth applies to all grips.

1. Shape plan: measurement, rim volume (edge fibre) and fold diagnosis done. Folds buckle correctly and
   are converged in resolution; their depth follows the mount (taut along the warp) and the pull. Thread
   hairiness not pursued (reference single threads are as thin as ours).
2. Real time: ~0.93x on the M5. Edge fibre adds 1.2-3.6 ms to the render at default framing and ~12 ms at
   max zoom on heavily torn cloth (on-screen fringe fragments; explicit-geometry and culling alternatives
   measured/estimated as no better). Free tails add ~0.26 ms.
3. Threads inside openings are often single yarns in shadow (physically consistent, faint).
4. The reference framing camera sees past the coupon's top edge (product framing does not).
5. The surface rim band is still painted pale (`rimFibres`), in addition to the rim hair now drawn as
   fibre; the halo adds wear over 1 cm around openings (no measurable effect; not a tear mechanism).
6. Free tails are quasi-static (no swing of their own); they collide with intact cells only (not with
   threads or other tails), and not at all in a layered scene (no self-contact face hash).
