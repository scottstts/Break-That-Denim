# Acting on the technical assessment (artifacts/assessment)

## Representation: strand-resolution mechanics, no explicit real-pitch yarn patch

- The assessment recommends independent yarn mechanics in the damaged region. At real pitch the
  reference-scale worn zone (0.40 x 0.28 m) holds ~400k crossings; the solver costs ~0.4 us per node
  per substep (`bench.ts`: 21k nodes 8.2 ms, 47k nodes 18.4 ms), so it cannot run interactively.
- The reference's visible strands are spaced ~3-3.5 mm (ladders at 0.9 s: 12-15 px at 0.24 mm/px),
  the size of a 9 x 6 bundle span (3.42 x 3.48 mm). The lattice chains are therefore treated as the
  mechanical strands: threads are drawn from them, and nothing render-side moves on its own.
- Yarns inside a bundle still share kinematics; the fine field decides which yarns exist.

## Mechanics corrections

- **Gradient/energy mismatch along the mount.** The line-search energy skipped every term owned by a
  pinned node, the gradient did not. On the GPU-test coupon half of the late Newton directions were
  uphill and the old code took an unevaluated 1/8 step anyway (energy injection). Fixed on both
  backends; `tools/studies/lsprobe.ts` checks g.d against dE/dalpha per term.
- **Line search.** Four evaluated steps; a change is conclusive only beyond `LINE_NOISE` (4e-6) times
  the sum of its terms' magnitudes. Conclusive decrease: best step. All conclusive increases: no step.
  Inconclusive: full step. The same bound on both backends keeps them on the same branch.
- **Bending** with rest-length weights (laws.ts `bendGeometry`) and continuity from cut masks
  (`bendScale`).
- **Cohesive opening** no longer grows with wear (0.4 mm); the wear-scaled "pull-out" softening is gone.
- **Pull-out** of a cut end follows the shear-lag retraction (T eps - W) / f, with T capped at the
  yarn's own peak (the bundle strain near a crack includes its opening). Marks are monotone. Initial
  abrasion defects have no pull-out.

## Tear morphology

- The reference's holes grow radially from pale spots while the weave around them barely deforms
  (dense frames 0-1.6 s): time-driven erosion, which the design excludes. They are not a target.
- Pre-tension (mount -0.5 % / -0.3 %) gives no round lobes: folds vanish and more pieces drop. Rejected.
- Stronger warp/weft abrasion contrast (0.97 / 0.7) changes which cracks form, not their nature: strand
  damage-softening localizes into bundle rows (grain cracks), as woven fabric tears under a pinch.

## Rendering

- Threads: the render-side Verlet simulation is removed. Freed yarns cling side by side in one to three
  flat clumps per bundle within ~2.5 mm of the weave (fibre cohesion), at 0.4 packing.
- Edge fibres are alpha-blended without depth writes; alpha to coverage quantised sub-pixel fibres to
  0 or 1 of 4 samples (speckle). Cloth edge coverage uses ordered thresholds.
- Rim and halo take the character of the nearest opening (they vanished 1 mm from it) and scale with
  how open its neighbourhood is (pin-holes get a slight rim).
- Per-yarn albedo variation fades with resolvability in pixels (`resolved`): it aliased into vertical
  scratches at the default framing. The twill comes from the prefiltered tile.
- Key light grazes in from the top (reference); cone 0.9 so landscape frames are not vignetted.

## Grip

- Pinch: patch targets contract to 0.3 of their grab-time spread over 0.15 s.
- A 2D drag lifts the handle by 0.35 of the dragged distance (max 5 cm).
- Slip: the requested handle 4 cm beyond the force cap's reach for 0.3 s loses the grip. The coupon has
  only ~1-1.5 cm of in-plane slack, so without it the cursor ran away from cloth that could not follow.

## Measured

- Substep 8.2 ms; product frame (4 substeps + render) 37 ms at rest: sim/wall ~0.9.
- Options measured but not taken: one fewer multigrid level (-12 %, max error 2.8 -> 4.0 mm vs a
  converged solve), one coarse sweep (-4.5 %, 2.8 -> 3.1 mm), two line colours (unsafe on the GPU:
  same-colour lines two apart are coupled and would race). A fused single-workgroup cycle for the
  coarse levels (identical math) is estimated at ~10 %.
