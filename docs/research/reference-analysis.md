# Reference analysis

Primary artifact: [`refs/demo.mov`](../../refs/demo.mov). Overview: [`refs/demo_contact_sheet.png`](../../refs/demo_contact_sheet.png).

This note records observations useful for calibration, including the second inspection of native frames during the design review. It does not reconstruct unknown physical parameters or claim access to the original Houdini scene. See the [review judgments](../decisions/2026-10-09-design-review.md) for the distinction between observed features and authored choices.

## Media facts

Inspected with `ffprobe` and decoded frames:

| Property | Value |
| --- | --- |
| Duration | Approximately 4.665 s |
| Encoded image dimensions | 1320 × 1624 |
| Video codec | HEVC |
| Reported nominal frame rate | 60 fps |
| Reported average frame rate | Approximately 58.016 fps |
| Reported frame count | 272 |
| Color metadata | BT.709, full range |

The file is a screen recording. Encoded frame timing and compression are not direct measurements of the source simulator's timestep, framerate, resolution, or physical scale. Use decoded presentation timestamps when extracting comparisons.

The extraction tool's first run confirmed that some supplied contact-sheet labels do not match the corresponding video state: for example, the independently decoded frame at 2.105 s already has a substantial opening in the tan layer. The supplied sheet's 2.1 s image shows an earlier state. Keep the supplied sheet as a visual overview, but use decoded video PTS for timing or motion comparisons.

The initial reproducible extraction selected actual timestamps 0.305, 0.905, 1.505, 2.105, 2.705, 3.305, 3.905, and 4.521667 s for the requested 0.3 s–4.5 s sequence. The generated logs and source hash are in `artifacts/reference/design-study/` on the design workspace; they can be regenerated from the original video.

The reference slightly exceeds the runtime's 2-million-pixel budget. Preserve the source at native resolution for inspection; compare runtime output at its permitted resolution and matched image scale. Do not interpret video sharpening/compression artifacts as fibers to recreate.

## Visible features

### Intact blue material

- Dense directional structure is visible, including diagonal twill and a denim-like blue/pale yarn contrast.
- Light and dark variation follows the cloth's folds and yarn structure, with broader worn patches.
- The surface has a soft fibrous response rather than a glossy plastic or metallic lobe.
- Abraded pale regions coexist with blue retained yarn surfaces. Traditional surface-dyed warp/pale-weft denim is a useful construction hypothesis, not a recovered dye-depth measurement.
- Broad folds and finer wrinkles coexist. The surface does not behave like a smooth rubber membrane with a noise texture.

### Outer-layer damage

- Early openings are small and irregular, then grow and merge.
- Stronger retained strips form a sparse network around larger openings.
- Many pale threads remain across holes. Some are taut; others sag, curl, or move locally.
- Exposed thread orientation is related to the weave but is not a perfectly uniform parallel grid.
- Edges contain short fuzz, longer yarn ends, color variation, and visible thickness. A clean polygonal cut would miss the dominant cues.

### Tan layer and depth

- A distinct tan material is visible behind the blue openings, with its own surface structure and shape.
- The tan material later contains openings of its own. Its torn edge is thicker, denser, and more continuously fuzzy than the blue edge, while its surface retains directional structure. This supports testing brushed woven backing; appearance alone does not prove felt/nonwoven construction.
- Native frames at 3.305 and 3.905 s show blue textured material behind openings in the tan surface. Some late openings also reveal dark space. The previous two-sheet-plus-void description omitted an important visible depth surface.
- The clip does not establish whether that blue surface is an independent third sheet or a returning fold. The current design deliberately uses a third physical blue sheet; it does not claim to have recovered hidden connectivity.
- Darkening near the upper layer's edges varies with separation and light direction. A flat painted dark outline would not account for the pattern.
- The visible depth surfaces do not simply share identical normals and displacement.

### Motion

- Large-scale deformation changes relatively smoothly while local yarns and narrow strips move differently.
- The visible sequence includes progressive material loss as well as moving surviving strands.
- No clear on-screen interaction supplies a measurable pulling-force history. The cause of reference damage remains unknown.
- The user-facing product deliberately replaces any unknown time-driven damage authoring with mechanically driven pulling.

## Hypotheses to test, not facts

| Hypothesis | Evidence required |
| --- | --- |
| Warp-faced denim with surface-dyed blue warp and pale weft explains the blue material | Static weave/scattering/abrasion match under several light/view directions; test 3/1 twill first |
| A thicker brushed woven backing explains the tan material | Independently matched surface and fuzzy edge, plausible drape and rupture; do not assume a nonwoven law |
| Three independently tearable sheets provide a useful reconstruction | Match blue-through-tan depth, shadows, and later voids without contradictory off-crop mounting; hidden source connectivity remains unknown |
| Exposed pale strands correspond primarily to one yarn family | Close frame inspection and authored comparisons; do not force every strand into that interpretation |
| Extensive correlated wear can produce the cellular damaged network under pulling | Several repeatable mechanical pulls from the same worn asset reach reference-like openings and ligaments; distinguish initial defects from new failures |

## Calibration order

1. Fix image framing and an explicit provisional physical scale.
2. Match broad cloth shape and layer separation with static geometry.
3. Match light direction, source size, fill, and exposure.
4. Fit warp/weft pitch, yarn diameter, and weave pattern.
5. Fit dye distribution, scattering, and roughness under both frontal and grazing light.
6. Add explicit broken yarns and per-material frays, checking silhouettes, density, dye exposure, and subpixel coverage.
7. Match interlayer and thread shadows.
8. Introduce dynamic mechanics and test whether multiple plausible pulls reach the reference's structure without changing lighting to hide defects. Static similarity alone cannot establish this.

Repeat static material checks after physical parameters or scale change. A shader can look correct on one planar sample and fail on a tight fold.

## Comparison captures to retain

Keep an intact crop, an early damaged crop, a large outer opening with tan backing intact, blue material revealed through the tan opening, and a late opening to the void. Include native-resolution crops of bridges, denim abrasion, and both material edge profiles. Motion comparisons should contain a pull, failure, release, and settling. Compare natural timing as well as individual frames; low presentation cadence and slowed physical time are different behaviors.

Use `tools/reference/extract-frames.zsh` to produce repeatable native-resolution frames, timestamp logs, source metadata/hash, and a contact sheet. Generated files belong in ignored `artifacts/reference/`. The original `refs/` directory remains unchanged.

The source clip is already ignored by the repository. Future collaborators need an explicitly supplied local reference copy; do not assume that cloning the repository retrieves it. The extraction tool reports a missing reference directly.
