# Handoff (rolling)

Session acting on `artifacts/assessment/assessment.md`. Decisions and measurements:
`docs/decisions/2026-10-10-assessment-response.md`.

## State

- Mechanics: unequal-span bending, gradient-consistent line search with rounding-bounded step choice,
  force-derived free-end pull-out (capped at the yarn's peak), wear-independent cohesive opening,
  debris bounded against the cloth (lands, Coulomb friction), contact query culling.
- Scene: coupon stretched 0.2 % along the warp (user decision); `MECHANICS_VERSION` 10; drape regenerated.
- Grip: pinch gathering, drag lift, slip beyond the bounded force (worker.ts; mirrored in session.ts).
- Rendering: threads from the strand chains (no render-side simulation), flat clumps of freed yarns,
  blended fibres, aggregate edge fringe, rim/halo from the nearest opening, footprint-filtered albedo,
  top key light.

## Verified

- `npm run check` (32 unit), `npm run test:gpu` (26/26), `npm run build`.
- Substep 7.97 ms; product frame 36 ms at rest (sim/wall ~0.93). Render ~8 ms at 1789 x 1117.

## Reproduce

- `node tools/gpu/run.mjs tools/gpu/scripts/session.ts cells NAME 30` (`DUMP=1`, `SNAPS=t,...`,
  `FRAMES=dir` for clips); `drag` with `AUTOLIFT=1` emulates a mouse drag; `scrap` tests debris landing.
- `still.ts` with `CAMERA`, `LIGHT`; `tools/compare/sbs.zsh` for reference crops.
- `SCENE` / `DRAPE` overrides for material and mount experiments; `frameloop.ts`, `accuracy.ts`.

## Open

1. Tear morphology: grain cracks and large openings rather than the reference's cellular lobes.
   Measured causes and rejected levers are in the decision record; area failure of worn cells would
   need kinematics finer than the 9 x 6 bundle.
2. Real time: ~0.93x on the M5; every cheaper schedule measured costs trajectory accuracy.
3. Threads inside openings are often single yarns in shadow (physically consistent, faint).
4. The reference framing camera sees past the coupon's top edge (product framing does not).
