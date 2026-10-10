# Round 7: mechanism corrections, reference scale, edge fibre

Supersedes the hardening-slip plasticity of the uncommitted round 6 work (`plasticity.ts`) and round 6's near-equal warp and weft abrasion loss.

## Removed: hardening slip

- The plastic kernel grew a span's rest length beyond yield and took no length from anywhere else. That created yarn out of nothing.
- Raising its range made worn cells bag and look pulled open. That effect came from the invented material, not from the cloth.
- Large stretch of woven cloth comes from three mechanisms, all now carrying it:
  - **Crossing slip** (`step.wgsl slip`), which is conservative. Its friction rises with the crossing yarns' tension (the wrap term), so it already hardens.
  - **Shear.** Wear-dependent; see below.
  - **Yarn drafting.** The worn yarns' long softening opening.
- `Seg` keeps 8 words. The two plastic words are `spare0` and `spare1` again.

## Reference scale

- The reference's own wales (5.3 px apart, 1.27 mm for this weave) put its frame at about 39 cm of cloth. Its lobes are therefore 8 to 10 cm across.
- Worn-zone radii are now (0.2, 0.14) m, cells 7.5 cm, feature scale 5 cm.
- The old layout was about half the reference's size.
- Gestures for this layout: the session script `cells` pulls at each cell's core.

## Material authoring

- **Abrasion loss, warp much greater than weft** (`wearStrengthLoss` [0.45, 0.9]).
  - On a warp-faced twill, abrasion acts on the warp floats of the face.
  - A worn region therefore loses its warps and leaves its weft bridging the opening: ladders open into harps of weft under load.
  - With near-equal loss both families fail together. No strands are left, and holes look eroded.
- **Worn-through cores** (initial defects) are compact.
  - Threshold 0.89, density 1, per-warp jitter 0.012, edge noise 0.04.
  - With scattered per-crossing draws, about 30 fragmented patches existed at rest. That was the main source of the "acid" look before any pull.
- **Sparse worn-through floats** are cut to a third (`pf` 0.02). They read as scratch-like dashes.

## Wear-dependent shear and bending (both solvers)

- **Jam angle** from yarn spacing: cos γ_lock = d/p.
  - Abrasion thins yarns, d ∝ sqrt(area). The cell's area is `cellArea`: the larger of the mean intact fractions of its u and v spans.
  - `SHEAR_COS0` is calibrated so that sound cloth keeps the old 17° jam (`SHEAR_JAM` 0.16).
  - Area 0.5 jams at about 47°.
  - The jam is computed from the current intact fractions, so cut yarns also loosen a cell. These are frozen during a solve, so the energy stays consistent.
- **Shear stiffness** scales with the crossing compaction normal force (`n0`, wear-reduced) over the intact value. `LayerMat.b.z` carries the intact bundled `crossingNormal0`.
- **Bending rigidity** of a triple scales with the mean intact fraction of its two spans: only continuous fibres bend.
- The GPU correctness coupon is now worn, so these terms are compared against the f64 reference.

## Friction-limited load sharing

- The overload of a yarn beside broken neighbours is at most `shareCap`:
  - (2 reach + 1) crossings, each passing μ (n0 + wrap T_cross), divided by the yarn's peak tension.
  - T_cross is the crossing yarn's tension at mid-span.
  - Beyond the cap, the tip yarns slip and share the load: the del zone.
- In practice the cap is about 0.05 to 0.2 for sound denim at low crossing tension, against an uncapped 2.8. It can be larger where the crossing yarns are highly tensioned.
- The geometric fraction stays 0.6.

## Rendering

- **Edge fibre** (`fuzz.wgsl`). The first 6 of 18 strands per edge crossing form one yarn-end brush: a core yarn that splays into crimped fibres beyond t0 (0.3 to 0.7).
  - Other fibre is gated by a tuft density that varies along the edge (noise at 5 mm and 1.6 mm), so edges alternate between tufted and bare stretches.
  - The previous uniform field of straight fibres read as fur or frost.
- **Surface rim band** width is 0.35 to 2.6 mm, from 1.6 to 4.8 mm. It is the yarns beside the cut that lost their crossings.
  - Its hairiness blur is reduced (1 + 2.5 rim).
  - The 6.5 mm abraded halo is unchanged.

## Evaluation

- `session.ts` `SNAPS=t1,t2` saves full-size stills at reference framing (`SNAPSIZE`, `SNAPCAMERA`). Compare the cloth while a pull is held: the reference is under continuous load.

## Open

- **Hole shape.** Openings are grain slits and eyes that partly close on release. Round lobes need the worn interiors to fail diffusely. Under one pinch, failure localises into one crack.
- **Ladders read as barcodes.** The weft picks in warp-cut regions are drawn straight, evenly spaced, thin and grey. Free picks should untwist, clump, sag and be paler.
- **Bundle-aligned windows.** Warp cuts still take whole 9-warp bundles together: the correlated scatter is similar across a bundle and the strain is per span.
- **No edge lift or curl.** No rest curvature has been added near free edges.
- **Restlessness** under the 2 × 2 schedule with self-contact is unchanged since round 6.
