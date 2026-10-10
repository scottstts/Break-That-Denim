# Handoff (rolling)

Latest round: `docs/decisions/2026-10-10-floor-and-room-light.md` (render-only floor 12 cm below, mount board,
dim overhead lamp, 8K base/normal cropped to the visible floor; uncommitted).
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
- Grip: pinch gathering, drag lift, bounded force; holds until release; torn-out pieces stay held. Pinch
  is the only loading (user decision): slit and eye openings are the correct result for it.
- Rendering: threads from the strand chains, split where the mechanics ruptured their bundle; flat clumps
  of freed yarns; free tails solved as heavy elastic rods resting on the cloth (`thread_tail.wgsl`); edge
  fibre from the fibre population (fans and rim hair, per fibre or closed form);
  rim/halo, footprint-filtered albedo, top key light. Colour, exposure and fill were measured against the
  reference and match; no shading change is justified.

## Verified

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
