# Handoff (rolling)

Session acting on `artifacts/assessment/assessment.md`. Commits so far:

- `31783f2` mechanics corrections (bending of unequal spans, consistent line search, force-derived
  pull-out, wear-independent cohesive opening). `MECHANICS_VERSION` 10, drape regenerated.

## Verified

- `npm run check`: lint, typecheck, 32 unit tests.
- `npm run test:gpu`: 26/26 (torn fixture now perturbs rest lengths and cut masks, and compares rhs).
- `tools/studies/lsprobe.ts`: gradient and line-search energy agree to 7 digits per term; 118/120
  Newton iterations take the full step on the test coupon (before: ~half were uphill).
- Substep 8.19 ms (`bench.ts`), unchanged.

## In progress / next

1. Threads drawn from the mechanical chains (replace `thread_sim.wgsl` one-way dynamics).
2. Worn-material calibration against several gestures (holes now open larger but angular; pieces drop).
3. Rendering: surface saturation/contrast, rims, fray, bridges.
