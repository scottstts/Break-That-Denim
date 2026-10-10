# Handoff (rolling)

State on top of commit `6f13106 pass 5`. Nothing is committed: this includes round 6 and round 7 work. The design choices are in `docs/decisions/2026-10-10-mechanisms-round7.md` and `2026-10-10-threads-and-weakest-link.md`.

## Verified

- Lint, typecheck: clean.
- Unit tests: 28/28 passing.
- GPU tests: 25/25 passing. The coupon is now worn, so wear-dependent shear and bending are compared against the f64 reference.
- `public/drape.bin` regenerated (MECHANICS_VERSION 9).
- Substep 8.24 ms (`bench.ts`).

## Reproduce

- `node tools/gpu/run.mjs tools/gpu/scripts/session.ts cells NAME 60` runs pulls at each worn cell. Options:
  - `SNAPS=4.5`: a still under load at reference framing.
  - `DUMP=1`: dump the final state.
- `CAMERA='{"distance":0.42,"target":[0,0,-0.01]}' node tools/gpu/run.mjs tools/gpu/scripts/still.ts NAME OUT 1320 1100` renders a close-up.
- `wearmap.ts` shows the strength layout (now 39 cm wide).

## Next, in priority order

1. Ladder and bridge threads.
   - Today: straight, evenly spaced, thin, grey (a barcode).
   - Wanted: freed weft picks untwisted, clumped, sagging, paler.
   - Mechanism first: the picks of a bundle should not stay at woven spacing once no warp holds them.
2. Round lobes versus slits.
   - Single-pinch failure localises into one crack.
   - Investigate diffuse failure in worn interiors (strength disorder versus concentration) and finer mechanics in the worn zone (design: explicit yarn region).
3. Bundle-aligned warp cut windows.
   - Per-yarn strength scatter within a bundle is too correlated.
4. Edge lift and curl: a rest curvature of spans beside free edges, in both solvers.
5. Restlessness: self-contact chatter under the 2 × 2 schedule (round 6 notes).
