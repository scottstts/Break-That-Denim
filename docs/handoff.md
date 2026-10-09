# Handoff (2026-10-09, after feedback round 1)

Read with `docs/design.md`, `docs/decisions/2026-10-09-feedback-round1.md` and the earlier decision records. All work below is uncommitted; the user commits.

## State

Single warp-faced denim layer at reference scale: ~35 cm of cloth per view height, 0.62 x 0.40 m coupon, 9 x 6 bundled spans failing in strands, 21k nodes. Dark void behind holes, by user direction.

Checks: `npm run check` (lint, typecheck, 22 unit tests) and `npm run test:gpu` (16 GPU-vs-f64 checks) pass; `npm run build` is clean.

Cost on the M5: 6.65 ms per substep at h = 1/120, 5.8 ms per frame at 2 Mpx. Natural time holds at ~30 fps presentation with 4 substeps per frame.

Settled drape: `public/drape.bin`. It is validated by a scene + material + `MECHANICS_VERSION` hash, so regenerate it with `npm run gpu:drape` after any change to mechanics or scene. Tools refuse a stale drape.

## Tools

`tools/gpu/scripts/session.ts <script> [name] [every]` runs scripted gestures (`yank`, `tug`, `pull`, `open`, `intact`, `rest`) with the worker's handle logic, plus topology analysis and per-frame diagnostics. Output: `artifacts/captures/<name>.png` (contact sheet) and `.log`.

| Env | Effect |
| --- | --- |
| `SOLVER` | JSON solver overrides |
| `CAMERA` | JSON camera overrides |
| `SIZE` | Output size, `WxH` |
| `DUMP=1` | Writes the final x + segs state |
| `TRACK=n,..` | Per-node trajectories |
| `HUNT=t` | After time t, checks every substep for non-finite values and replays the failing substep stage by stage |

Reference frames are in `artifacts/reference/design-study/`. For side-by-side checks, render with `SIZE=1000x1230` and `hstack` against `frame_0.300.png`.

## Done this round (user feedback 1–4)

1. Too strong: removed the damage-commit gate (evidence in the decision record), weakened worn wefts, raised the grip to 150 N.
2. Rubbery: real denim bending stiffness, fold-stiffening bending, shear locking, realistic yarn compression.
3. Artifacts and black screen:
   - Root cause was an f32 pivot cancellation; fixed with a regularized pivot plus a NaN firewall.
   - Detached pieces become ballistic and retire.
   - The intact-surface control net uses only connected ring points.
   - The isolated-defect rectangles are gone: damaged cells are drawn by the woven surface shader with their missing yarns removed.
4. Weird folds: the razor crease was an in-plane shear band, fixed by shear locking and stiffer bending. The black tent side is now a dark-blue shadow (ambient fill).

Also: rescaled to the reference framing (weave density 3x finer on screen; warps vertical so wales rise right at ~57°), prefiltered weave tile, per-strand damage, quadratic debris drag, pick radius for the coarser node spacing.

## Known gaps, in priority order

1. **Hole quality.** Openings are black voids with fairly clean edges.
   - Missing: the reference's surviving pale weft bridges, dense fuzzy pale edge band, and cellular network of retained ligaments.
   - Tails/stubs exist but are ~1 px at this scale. The fuzz alpha builds up only near the edge.
   - Worn-through warp slots read as crisp "ladder" marks; they need soft, fibrous edges.
   - Ideas:
     - Make wefts at hole edges slip and pull out rather than break: check `slip` and tails pull-out at this bundle size.
     - Render the edge band from per-strand stubs with dense, longer fibre tufts.
     - Author wear as connected cells with explicit absent warp strands.
2. **Twill contrast at default zoom** is softer than the reference's crisp light ridges and dark grooves.
   - Tile mips blur ridges; the key light is not grazing enough.
   - Consider storing slope variance (LEAN-style) in the tile, and checking the key elevation against the reference.
3. **Rest drape composition.** Long parallel diagonal folds; the reference shows folds converging toward a point.
4. **No self-contact.** Folds and flaps can interpenetrate.
5. **Tension ridge.** A grip-to-mount ridge is one lattice line wide.
6. **Solver accuracy.** Release dynamics are still damped relative to a converged run (see the mechanics-schedule record).

## Pitfalls

- WGSL: mixing `||` and `&&` needs parentheses. Const vectors indexed at runtime must be copied to a `let` first.
- Overridable constants `FUZZ_PER_*` are duplicated in `refine.wgsl` and `fuzz.wgsl`; keep them equal.
- Session `far` counts nodes moved more than 0.3 m; gripped cloth legitimately moves 0.25 m at this scale.
- The worker is not exercised headless (probe, topology, reset paths); review those by reading.
