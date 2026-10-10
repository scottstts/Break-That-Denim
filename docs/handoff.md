# Handoff (rolling)

Latest round: `docs/decisions/2026-10-10-edge-fibre-population.md` (render; uncommitted, for the user
to inspect). Before: `docs/decisions/2026-10-10-interaction-round.md` (threads through cloth, restless
settling, grip slipping).

## State

- Mechanics: unequal-span bending, gradient-consistent line search with rounding-bounded step choice,
  cancellation-free bending energy changes, force-derived free-end pull-out, wear-independent cohesive
  opening, debris bounded against the cloth, contact query culling.
- Scene: coupon stretched 0.2 % along the warp (user decision); `MECHANICS_VERSION` 10.
- Grip: pinch gathering, drag lift, bounded force; holds until release; torn-out pieces stay held. Pinch
  is the only loading (user decision): slit and eye openings are the correct result for it.
- Rendering: threads from the strand chains, split where the mechanics ruptured their bundle; flat clumps
  of freed yarns; edge fibre from the fibre population (fans and rim hair, per fibre or closed form);
  rim/halo, footprint-filtered albedo, top key light. Colour, exposure and fill were measured against the
  reference and match; no shading change is justified.

## Verified

- `npm run check` (33 unit), `npm run test:gpu` (26/26).
- Session `settle`: 0 line-search rejections (was 164), settles 0.73 s after release.
- `haul` (fast long drag, AUTOLIFT=1): grip held on worn and intact cloth; worn scrap stays in the hand.
- `maul` (four hard drags, AUTOLIFT=1): no non-finite values, 2 rejections in 6.5 s, no bridging threads,
  21 of 40k sheet spans through a cell. Without the slip release a 19 cm drag at full force tears out a
  large worn piece (freed 368 nodes), which falls when released.

## Reproduce

- `node tools/gpu/run.mjs tools/gpu/scripts/session.ts <script> NAME 30` (`DUMP=1`, `SNAPS=t,...`,
  `FRAMES=dir`, `STATS=1`, `TRACK=n`, `LSPROBE=t`); `haul` with `GRAB`, `PULL`, `HOLD`, `AUTOLIFT=1`.
- `tools/studies/crossings.ts <state>`: spans through intact cells; `tools/studies/straightruns.ts <state>`:
  thread runs drawn without a carrying chain.
- `still.ts` with `CAMERA`, `LIGHT` (incl. `background`), `RENDERCOST=1`; `tools/compare/sbs.zsh`;
  `SCENE` / `DRAPE` overrides; `bench.ts`.
- `tools/compare/coverage.mjs`: fringe coverage by distance from the edge, from two stills over different
  backgrounds. `tools/compare/fuzzband.mjs`: loose-material band vs a reference frame (its edge length
  grows with fuzz, so compare coverage for our own changes).

## Open

1. Shape plan (agreed): measurement done; rim volume done (edge fibre); next the threads' hairiness
   envelope (same population model: Uster-type hairiness of a freed yarn), then fold depth (mechanics:
   does the cloth buckle under compression, is the lattice fine enough).
2. Real time: ~0.93x on the M5. Edge fibre adds 1.2-3.6 ms to the render at default framing and ~11 ms at
   max zoom on heavily torn cloth (per-fibre loop ~7 ms). Next: draw the sparse resolvable fibres as
   explicit geometry lit from their fan (compute pass), not per-pixel loops.
3. Threads inside openings are often single yarns in shadow (physically consistent, faint).
4. The reference framing camera sees past the coupon's top edge (product framing does not).
5. The surface rim band is still painted pale (`rimFibres`), in addition to the rim hair now drawn as
   fibre; the halo adds wear over 1 cm around openings (no measurable effect; not a tear mechanism).
6. One-ended threads beyond their chain hang procedurally toward -z without contact (978 such runs in
   `maul`, 34 longer than 5 mm).
