# Round 3: torn geometry, rim, progressive failure

Supersedes parts of [feedback round 2](2026-10-10-feedback-round2.md): per-corner quarters with per-yarn clipping, explicit-span fuzz, uncorrelated strength scatter, uniform worn cells.

Scope (user direction): the tear itself is the target. The scene stays denim-only with a dark void behind; no backing layer.

## Why tears looked axis-aligned and paper-like

There were three causes:

- **Cell quarters.** Each live corner drew the quarter of its cell bounded by the cell midlines. Every free-edge segment was therefore a lattice line.
- **Per-yarn clipping.** Yarns were clipped at per-strand failure sites, which produced stair steps at the strand scale (1–2 mm).
- **Missing-yarn gaps.** Partially broken strands removed yarns from intact cells, which produced dark see-through specks.

## Torn cells: crack curves and corner groups

- A broken span is crossed by the crack at its failure site, drawn toward the middle: `crossingOf` = 0.5 + 0.7 (site − 0.5).
- The crack leaves each crossing at right angles to the cut yarns, so it is C1 across neighbouring cells.
- Crack shapes by number of broken edges:
  - Two broken edges: one cubic between their crossings.
  - Three or more: cubics to a junction.
  - One: a tip ending inside the cell.
- Corners joined by intact spans form a group. The group leader draws the whole cell on the group's own control net (`torn.wgsl`). Corners outside the group are rebuilt from group frames at rest length.
- Each group keeps the part of the cell on its side of the cracks. The free edge is the crack, so opening comes from the groups moving apart.
- Crack curves (per cell) and group boundary curves (per group) are computed once per frame in `tornGeo`. Fragments receive the boundary curves as flat varyings and stream the region polygon.
  - A first version built the polygon in private arrays per fragment. That cost about 7 ms at 1789×1117 and has been removed.
- Fray at the edge is per yarn. Yarns crossing the crack fringe past it by up to about 1 mm. Yarns parallel to it fall away by up to 0.75 mm.
- Partially broken spans in intact cells no longer remove yarns. They only add wear.

## Edge field and rim

- An r16float texture over the rest chart (0.3 mm texels) holds the distance to the nearest crack, up to 7 mm.
  - It is drawn by min-blending one quad per crack.
  - It is redrawn only when an order-independent signature of the torn set changes (`edgeArgs`, indirect counts are zero otherwise).
- Rim, lip and halo are shaded from this field everywhere, in intact cells and torn groups alike.
  - Mixing in the exact crack distance for shading produced seams along lattice lines.
  - The exact distance only decides coverage.
- A lone crack tip (one broken edge in its cell) stores its distance divided by 0.3 (`crackMaturity`). Fresh cuts get narrow rims instead of fuzzy blobs.
- Rim profile:
  - Width wanders between about 1 and 5.5 mm.
  - A rolled lip is applied as a normal tilt: height H·(x/x0)·exp(1 − x/x0), H ≤ 1 mm.
  - Fibre tufts are streaked across the edge.
  - Edge occlusion is kept light. Darker settings read as a carved outline at default framing.
- Refined vertices on a cell boundary take tangents from an adjacent surface cell. Tangents from the torn neighbour's naive net tilted boundary normals past vertical, which produced dark back-facing slabs.
- Edge fibres:
  - 96 per torn cell, rooted along cracks on each side's own surface.
  - Crimped (sinusoidal), mostly 0.25–1.75 mm, with 5% long drooping ends.
  - Straight fibres read as needles.
- Tails (loose yarn ends) start at the crack crossing and are drawn only beyond 0.6 mm. Each yarn of a bundle splays at its own angle, with paler untwisted ends.

## Strength field and progressive failure

These changes affect mechanics.

- **Correlated scatter.** Strength scatter is mostly spatially correlated (fbm at 11 mm, σ 0.2 + 0.12 w) with a small uncorrelated part (σ 0.07). Uncorrelated scatter hid the worn-cell structure and made lone weak spans fail away from the tear.
- **Radial worn cells.** Cells are weakest at their centres and recover to nearly intact ligaments. Cell size is 22 mm. Strength wear is 0.12 + 0.88 × interior. Failure grows outward along the contours and ligaments arrest it.
- **Separate visible wear.** Visible wear uses the cells only faintly (0.78 + 0.22 × interior). Showing them fully printed a honeycomb on the surface.
- **Weft vs warp.** Worn weft keeps more strength than worn warp: loss 0.75 vs 0.97 at full wear, as abrasion acts on warp floats.
- **Pull-out softening.** Worn yarns soften over up to an extra 3 mm (`3e-3 · w^1.5`), like pull-out.
  - Before this, failure was brittle: about 100 ruptures in 0.1 s, and the 300 N grip cut out a rectangular coupon (217 freed nodes).
  - Now holes open progressively, rounded, as several cells separated by ligaments.
  - Intact cloth stays brittle. It is not tearable by hand anyway.
- Grip limits are unchanged (300 N, 6 mm lead). Below about 100 N, worn cloth does not open at all.

## Test change

The torn-coupon fine-operator check excludes rows touched by spans within 1e-6 strain of the axial tension/compression kink, where the tangent jumps.

- Crossing slip relaxes spans next to boundaries to zero tension.
- f32 and f64, and even two f64 runs with different summation order, can take either branch.
- The new strength draw happened to produce such a span. The remaining rows agree to 1.8e-5.

## Open

- Bridges across openings are still rare. Surviving weft rows render as narrow pale strips but seldom persist long.
- The intact surface still lacks irregular cotton relief (yarn width, height and flattening variation).
- Loose material has no independent motion. Tails and fibres are shaped from their parent frame.
- After release, holes relax into narrower shapes than during the pull.

## Tools

- `tools/gpu/scripts/still.ts` renders a dumped session state (`DUMP=1`) from any `CAMERA`.
- With `RENDERCOST=1` it reports the minimum whole-frame GPU time over 15 frames. Pass boundaries overlap on Apple GPUs, so per-pass numbers are unreliable.
