# Single denim layer, bundled lattice, solver schedule, rendering split

Status: implemented for the first hands-on review. Evidence: `tools/studies/timestep.ts` (f64 reference), `tools/gpu/scripts/{grabstudy,accuracy,ringdown,profile}.ts` (Dawn/Metal, Apple M5).

## Scene

- One warp-faced denim layer (user direction). Behind holes is dark void. Cross-layer contact code stays but is skipped for single-layer scenes; self-contact does not exist yet.
- Coupon 0.13 x 0.19 m, warps near horizontal in view (twill lines rise to the right at ~33 deg, as in the reference). Camera shows ~10 cm across the view height (reference density).
- Mechanical spans bundle 3 warps x 2 picks (`bundleWeave`), giving near-square 1.14 x 1.16 mm cells and 18.9k nodes. Per-span quantities scale with yarn count; the crossing wrap force scales with the sliding family's count (`Yarn.yarnsPerSpan`); the shear spring is rescaled by cell shape. Rendering keeps individual yarns at true pitch. Consequence: a broken span releases 2-3 yarns together.
- The settled drape is precomputed (`npm run gpu:drape` -> `public/drape.bin`, validated by `sceneHash`); runtime settling is the fallback. Regenerate it whenever the scene or mechanics change equilibrium.

## Why per-step cost is latency-bound

At 19k nodes a substep is dominated by dependent-pass latency (~15-30 us per line-smoothing pass, 6 colors x 2 sweeps x levels), not bandwidth. Node count reduction alone does not help much; fewer passes, fewer V-cycles and fewer Newton iterations do.

## Accuracy findings that set the schedule

- From a settled drape with the app's grip, the over-resolved truth has 0 ruptures. Under-converged steps (fixed 2 Newton x 3 PCG at h = 1/120) committed 40 spurious ruptures at grab time. h = 1/60 tears spuriously even when converged; h = 1/120 converged does not.
- Projected Newton converges only linearly here (buckled drape, snap-through of folds), even with exact linear solves; the saturating (Huber) grip made it worse (isotropic majorizer stiffens the pull direction; exact Hessian overshoots).
- Unconverged steps also injected energy: the cloth "trembled" at rest (~1 mm/s floor) and the release snap-back was lost.

Decisions:

1. Grip is a quadratic spring per patch node (exact for Newton). Force is bounded by clamping the handle's lead over the patch centroid on the GPU once per substep (`gripLead`; `GRIP.maxForce` 40 N at `GRIP.maxLead` 2 mm). 40 N lets worn denim tear by hand; intact denim does not.
2. Newton step length by GPU line search over {1, 1/2, 1/4} (1/8 if none decreases) using per-term energy differences (`energy.wgsl`; axial terms by trapezoid of the damaged force law). The f64 reference mirrors it exactly (`ReferenceSolver.energyChange`).
3. Damage and slip advance only where both span ends' last Newton update is below `historyTolerance` (2 um): unconverged iterates never commit irreversible history.
4. Schedule: h = 1/120, 2 Newton x PCG(1 V-cycle), coarse Galerkin levels built once per substep and reused, 3 line colors, coarsest level <= 400 nodes with 2 sweeps. Optional GPU-gated extra Newton iterations exist (`newtonMax`, indirect dispatch, no readback) but are off by default.
5. Normal air drag (`airDrag` 1.5 N s/m^3, implicit) so a released coupon settles in about a second.

Known limitation: the production schedule is still far from converged dynamics after a release (peak position error 6-9 mm vs a converged run; motion settles sluggishly over 2-3 s). Every affordable schedule tested (including h = 1/240) is similar; closing this needs a stronger preconditioner, not more of the same iterations.

## Runtime

- The simulation sleeps when nothing is gripped and the largest nodal speed stays below 1.5 mm/s for 0.4 s (async readback of an atomic max); it starts asleep from the precomputed drape. No GPU work while asleep and unchanged.
- At most one frame of GPU work in flight; while simulating, frames are presented at <= 30/s with <= 4 substeps each (natural time when the GPU keeps up; otherwise bounded slow-down, never a queue).

## Rendering split

- Intact cells: bicubic (Catmull-Rom) surface, 2 x 2 sub-quads per cell; the fragment shader evaluates the woven yarn height field (3/1 twill, overlapping flattened warps, low weft, crimp) at the 4 MSAA positions with hairiness jitter (~45 um), outputs per-sample coverage (stretched weave opens), and lights once per pixel on averaged inputs (~9 ms/frame at 2 Mpx; the weave evaluation dominates).
- Cells missing any span: their surviving spans become explicit yarn ribbons (one per bundled yarn), plus render-only tails and fuzz fibres, from GPU-compacted lists and indirect draws.

## Tooling

- No browser automation (it froze the host). `tools/gpu/run.mjs` runs TypeScript entry points (Vite transforms) on Dawn/Metal in Node with throttled submissions. `npm run test:gpu` checks GPU kernels against the f64 reference; with a converged schedule trajectories agree to < 1 um over 1.6 mm of motion.
