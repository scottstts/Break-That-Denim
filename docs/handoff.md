# Handoff (rolling)

Session acting on `artifacts/assessment/assessment.md`. Decisions: `docs/decisions/2026-10-10-assessment-response.md`.

## Verified

- `npm run check` (32 unit), `npm run test:gpu` (26/26).
- `MECHANICS_VERSION` 10; `public/drape.bin` regenerated.

## Reproduce

- `node tools/gpu/run.mjs tools/gpu/scripts/session.ts cells NAME 30` with `DUMP=1 SNAPS=...`;
  `AUTOLIFT=1` with the `drag` script emulates a mouse drag.
- `still.ts` with `CAMERA`, `LIGHT` overrides; `tools/compare/sbs.zsh` for reference crops.
- `SCENE` / `DRAPE` env overrides for material experiments (drape.ts, render.ts).
- `frameloop.ts`: product frame wall time; `accuracy.ts conv 2x2-lean` for solver accuracy.

## Next

1. Real-time speed (sim/wall ~0.9): fused coarse V-cycle kernel.
2. Debris and thread contact (pieces fall through the cloth).
3. Rendering: rim texture, hole-edge translucent zones, thread fuzz at far view.
