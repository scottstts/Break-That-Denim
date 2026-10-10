# Round 6: loose threads, weakest-link yarn failure, worn cells

Supersedes the per-span thread pass of [the yarn damage field round](2026-10-10-yarn-damage-field.md) (`yarn_geom.wgsl`, `yarn.wgsl`, `classifySpans`), its per-crossing damage update, and round 3's rolled rim lip.

## Gaps found against the reference

Compared on `worn2`, with crops at matched scale and frame statistics:

1. **Threads.** They were drawn per lattice span, in halves, with all of a bundle's yarns spread across the span's width. Bridges read as parallel ribbon slabs and broke at every span boundary. A lone warp crossing a ladder of picks was drawn as cloth.
2. **Straight bundle cuts.** An overstretched span cut every yarn at every crossing in one substep, so openings grew in whole lattice spans.
3. **Departed pieces.** A piece that fell away took whole pick and warp bundles with it, leaving notches along lattice lines.
4. **Rim.** The rim was a dark, rolled lip. It read as a carved outline, not the reference's pale fibrous border.
5. **Hole size.** Worn cells failed only near the grip, so holes stayed small (still open, see below).

## Threads from the field (`threads.wgsl`, `thread_geom.wgsl`, `thread_prep.wgsl`, `thread.wgsl`)

- Classification per crossing and family:
  - **present**: uncut in the field, and the material is still in the scene;
  - **anchored**: present and part of a continuous piece of at least 3 crossings;
  - **cloth**: both yarns anchored at all 4 crossings of some 2 x 2 block around the crossing;
  - **lone**: anchored but not cloth.
- Interlacing is tested in 2D on purpose. A per-crossing "both yarns present" test made isolated surviving warps into strips of cloth.
- Every maximal lone run is one thread, drawn from half a crossing inside the cloth (held end) or to its cut (free end).
  - A run held at both ends needs at least `MIN_RUN` (4) crossings. Shorter runs are worn-through floats and stay on the surface.
  - A run free at both ends has fallen away. It is marked absent and not drawn.
- The surface and the edge field read the thread bits, so a yarn is drawn exactly once.
- **Geometry.** The bundle's centre line (Catmull-Rom over the lattice chain) is offset to the yarn's woven place. Away from the weave (within about 1 crossing) the yarns gather into one cord, or two in 1/4 of bundles, twisted around each other.
  - 8 % of yarns stray.
  - Each yarn sags by its own slack against the bundle's current arc length, computed once per run per frame (`threadPrep`).
- **Size.** Loose radius comes from linear density at 0.6 packing (`LayerR.loose`), not the flattened in-weave width.
- **Coverage.** Threads use alpha-to-coverage: true width over the ≥ 0.7 px ribbon.
- **Rebuild.** Extraction rebuilds only with the edge field (field event, or the retired-node count changed). Per frame, only `retireCount`, `threadArgs` and `threadPrep` run.

## Departed material

- The scene loses a yarn segment when the separated span's side that owns it is retired. Owners: the part before the yarn's first cut belongs to node a, the part after it to node b.
- A segment interlaced with a retained yarn of the other family stays present. This one-level retention stops a departing piece from carrying off whole bundles.
- `tornInfoMain` floods the groups of live corners before those of falling ones.
- **Remaining limit.** A departing piece still takes its own pick rows below the warp cuts. Its notch is irregular per warp but bounded by the pick-bundle line (see Open).

## Weakest-link yarn failure (mechanics, `yarnField.ts updateField` and `yarnfield.wgsl fieldUpdatePick/Warp`)

- A yarn's crossings within one span carry the same tension, so it softens at one crossing only: the one with the lowest peak strain.
- Its strain is interpolated once, at the middle of the span, at the yarn's own place across the bundle.
- Interpolating along the span (as before) put every yarn's cut at the same place, giving straight cuts across a bundle.
- **GPU work.** One invocation per (span, yarn), with up to 16 yarns per span. Two passes, one per family, so each byte of a field word has one writer.
- **Tests.** The GPU suite checks damage on a stretched worn coupon: 0 cut mismatches against the f64 reference. A unit test checks the single cut per yarn.

## Material (`scene.ts`)

- Worn strength ∝ area^1.5 (damaged fibres no longer run continuously), so a worn cell gives way as a region.
- Sharper cell boundaries (`smoothstep(0.03, 0.3, f2 - f1)`).
- Weft keeps less of its strength (`wearStrengthLoss` [0.95, 0.99], exponent 1.3).
- Fewer initial worn-through floats (0.3, from 0.6 wear). Defect cores from 0.86.
- Visible wear ∝ w^1.8, concentrated around the damage.
- `public/drape.bin` regenerated.

## Rendering

- The rim is a pale, dense fibre band, 0.8 to 2.8 mm wide, inside a 5 mm abraded halo. There is no normal tilt or lip occlusion.
- Edge distance is seeded at openings only (neither yarn in the cloth). Worn-through floats no longer get a rim.
- Warp floats vary in brightness per float (pebbly wales), with less long-streak mottling.

## Rejected experiments

- **Taut biaxial mount (−0.6 % to −1.5 %).** Holes opened larger, but drape folds disappeared, and at −1.5 % the rest state tore.
- **Larger cells (7.5 cm, larger zone).** Under the slack mount, pulls made grain slits instead of cell openings.
- **Grip radius 16 to 25 mm.** It made flaps and polygonal grain tears.

## Measured cost (M5, 1789 x 1117, `worn2`)

| Measure | Before | After |
| --- | --- | --- |
| Frame | 5.51 ms | 4.65 ms |
| Frame that rebuilds the field-derived state | | 6.03 ms |
| Substep | 7.97 ms | 7.87 ms |

## Open

- **Holes are still about 1 to 2 cm and cross-shaped**, against the reference's 5 to 8 cm round cells with thin ligaments. A lift tears out the patch under the grip as a plug, and its corners start straight grain slits. Worn interiors still localise failure at the grip boundary instead of failing throughout.
  - Candidates: grip weighting and plug release; a strain-rate or nonlocal process zone in the worn law; finer mechanics in the worn zone (the design's explicit yarn region).
- **Pick-bundle notches** remain where a piece departed (see Departed material).
- Threads still have no dynamics of their own: no swing or settling after a pull.
- Small rectangular worn-through gaps around holes look moth-eaten at macro range.

## Tools

- `tools/gpu/scripts/threadcheck.ts <state>`: GPU thread runs by ends and length, against a CPU run count.
- `tools/gpu/scripts/wearmap.ts [out]`: map of authored warp and pick strength, plus the weakest spots with session grab coordinates.
- `still.ts` with `REBUILD=1 RENDERCOST=1`: cost of a frame that rebuilds the field-derived state.
