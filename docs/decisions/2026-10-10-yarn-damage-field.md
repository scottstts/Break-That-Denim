# Yarn damage field: failure at yarn resolution

Supersedes the span-level cohesive law, strands and failure sites of [feedback round 1](2026-10-09-feedback-round1.md) and [round 3](2026-10-10-tear-geometry-round3.md), and round 3's crack curves, crossing relaxation and crack-quad edge field.

## Why

Failure used to be decided per mechanical span (9 warps x 6 picks, about 3.4 mm). Every opening was therefore a union of lattice cells, and every crack ran along a lattice line. Better crack curves inside the cells only smoothed the outline of shapes that were still decided at the cell level.

Two things were tried first and dropped:

- **Relaxed crack crossings.** These smoothed the staircase outline but could not change which cells failed.
- **A nonlocal span damage driver** (3 x 3 weighted strain average). It produced parallel crack bands, not round openings.

## Model (`src/simulation/yarnField.ts`, `gpu/wgsl/yarnfield.wgsl`)

- **Field.** Per crossing of warp k and pick m, the field stores damage (u8, 255 = cut) and an abraded cross-section (u8) for each family. Per-yarn peak tension (f32) is compiled from:
  - the continuous wear field;
  - a correlated scatter;
  - the yarn's own slub noise along it.
- **Update** (after each substep):
  - The segment strain is interpolated bilinearly from the active parallel spans at their midpoints.
  - A per-yarn cohesive law (`axial.ts yarnDamage`) maps that strain to damage.
  - Only active spans whose yarn has not yet separated within them update.
- **Coupling.** The span law becomes `intact x elastic`, where `intact` is the mean over the span's yarns of `(1 - d_max)`. Each yarn's term is weighted by the harmonic mean of its cross-section along the span. A span separates only when every yarn crossing it is cut. Its tails split at the mean cut position.
- **Mirrors.** The f64 reference mirrors this exactly. The GPU correctness suite passes unchanged.
- **Removed.** The old envelope history (`maxExt`), strands and per-strand failure sites are gone.

## Material decisions

These were found by gesture studies (field maps plus renders). Use pulls at the worn cores, not the old far-reaching yank gestures.

- **Stiffness loss.** Abrasion removes fibres, so cross-section (stiffness) falls with strength.
  - Without this, worn cells were weak but stiff, and a crack tip always won: long straight grain tears with chunks dropping out.
  - With it, a worn cell is a compliant inclusion that takes up the pull and gives way as a region.
- **Grip cap: 120 N.**
  - Once notched, intact denim tongue-tears along the grain at about 50 to 100 N. That is real physics, and it cannot be tuned away while the hand delivers 160 to 300 N.
  - At 120 N, pulls open worn cores locally and intact cloth holds.
- **Softening opening.** Intact `softeningOpening` is 0.4 mm. The worn pull-out term is `0.6 mm x w^1.5` (longer pull-out, 1.2 to 2.5 mm, let worn regions stretch without opening).
  - Ductility now also comes from yarns failing one by one at scattered strengths.
- **Initial damage** follows the clip at t = 0.3 s: small ladder openings and pale worn areas.
  - Cores (w > `defectThreshold` 0.82, noisy edge) have lost all their warp.
  - Around them, single warp floats are worn through. These are 3-crossing runs on the face of the 3/1 twill, rising from `sparseFrom` 0.45.
  - Picks in worn cells keep 15 to 35 % (`wearStrengthLoss` 0.85 / 0.985).
- **Pinned to the drape.** `MECHANICS_VERSION` is 5, and the drape hash samples the initial field. Regenerate `public/drape.bin` whenever wear or material changes.

## Rendering from the field

- **Torn cells** (`torn.wgsl`) flood their crossing grid, bit-parallel per row, from each corner group through intact yarn links.
  - Only woven crossings (both yarns present) are flooded, plus 2 crossings of yarn end beyond them.
  - The 2-crossing fringe keeps a worn float part of the cloth. A longer run of lone yarn is a thread.
  - Each fragment keeps a sample only if its crossing carries the group's label. Labels are 4 bits per crossing in the `tornGeo` record.
- **Per-sample masking** (all cells): a separated yarn is gone, and the yarn below shows.
  - A warp gone over its pick shows the pick spread over its full pitch at rest. That pick is lit as exposed, not as a groove.
  - A partially damaged yarn frays through at random fibre positions.
- **Threads** (`yarn_geom.wgsl`) draw spans beside torn cells, per yarn, over runs of lone crossings.
  - Threads follow the node chain, cluster loosely, and sag by their own slack.
  - A run ending at the yarn's own cut has a free, drooping end.
  - The old tail pass and the node-level "owns cloth" gate are removed.
- **Edge field** (`edgefield.wgsl`) is a jump flood over the crossing grid, rebuilt only when the solver's field-event flag is set. Channels:
  - r: distance to a cut;
  - g, b: warp and pick presence, used by the filtered far-view weave;
  - a: density of cut segments within about 1.2 mm.
- **Rim** width, strength and lip slope scale with cut density, so isolated worn floats do not get a full rim.

## Tools

- `session.ts` scripts `pluck` (`GRAB`, `PULL` in 10 cm units), `worn` (one pull at each worn core) and `worn2` (a second round at the hole edges). Sessions with `DUMP=1` also dump the yarn field, and `still.ts` restores it.
- `tools/gpu/fieldmap.mjs <state> <out.png>` maps a dumped field: black = both yarns cut, cream = warp cut, cyan = pick cut, blue shades = warp damage.

## Measured cost (M5, 1789 x 1117)

- Substep: 8.0 ms, up from 7.6 ms (field update and coupling).
- Frame on a torn state: 13.4 ms against 11.3 ms before, on states that are not identical.
- The edge-field rebuild runs only on frames where the field changed.

## Open

- **Straight notches** where a detached chunk fell away. A span's yarns straddle two cells. When the cut lies in the departed cell, the surviving side's woven material ends at the cell edge. Fixing this needs ownership flooded across cells.
- Hole growth beyond the worn cores needs repeated pulls. Holes are smaller than the reference's cluster after one gesture.
- Threads are flat camera-facing ribbons with a fibre halo. They have no independent dynamics.
- The pick-fill under worn warps, and the far-view presence channels, are approximations of a loosened weave.
