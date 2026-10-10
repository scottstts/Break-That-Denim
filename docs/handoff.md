# Handoff (rolling)

Latest round: `docs/decisions/2026-10-10-interaction-round.md` (user feedback: threads through cloth,
restless settling, grip slipping). Earlier: `docs/decisions/2026-10-10-assessment-response.md`.

## State

- Mechanics: unequal-span bending, gradient-consistent line search with rounding-bounded step choice,
  cancellation-free bending energy changes, force-derived free-end pull-out, wear-independent cohesive
  opening, debris bounded against the cloth, contact query culling.
- Scene: coupon stretched 0.2 % along the warp (user decision); `MECHANICS_VERSION` 10.
- Grip: pinch gathering, drag lift, bounded force; holds until release; torn-out pieces stay held.
- Rendering: threads from the strand chains, split where the mechanics ruptured their bundle; flat clumps
  of freed yarns, blended fibres, aggregate edge fringe, rim/halo, footprint-filtered albedo, top key light.

## Verified

- `npm run check` (33 unit), `npm run test:gpu` (26/26).
- Session `settle`: 0 line-search rejections (was 164), settles 0.73 s after release.
- `haul` (fast long drag, AUTOLIFT=1): grip held on worn and intact cloth; worn scrap stays in the hand.

## Reproduce

- `node tools/gpu/run.mjs tools/gpu/scripts/session.ts <script> NAME 30` (`DUMP=1`, `SNAPS=t,...`,
  `FRAMES=dir`, `STATS=1`, `TRACK=n`, `LSPROBE=t`); `haul` with `GRAB`, `PULL`, `HOLD`, `AUTOLIFT=1`.
- `tools/studies/crossings.ts <state>`: spans through intact cells; `tools/studies/straightruns.ts <state>`:
  thread runs drawn without a carrying chain.
- `still.ts` with `CAMERA`, `LIGHT`; `tools/compare/sbs.zsh`; `SCENE` / `DRAPE` overrides; `bench.ts`.

## Open

1. Tear morphology: grain cracks and eyes rather than the reference's lobes. Finer bundles (6 x 4) and
   out-of-plane plucks measured, no improvement; the reference's lobes grow under dome-like biaxial
   stretch that a single pinch does not apply.
2. Real time: ~0.93x on the M5.
3. Threads inside openings are often single yarns in shadow (physically consistent, faint).
4. The reference framing camera sees past the coupon's top edge (product framing does not).
5. Rim halo reads as a uniform pale glow at default framing (plausible abrasion up close).
