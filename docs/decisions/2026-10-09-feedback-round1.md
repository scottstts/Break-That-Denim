# Feedback round 1: scale, robustness, tearing, fold mechanics

Supersedes parts of [mechanics-schedule](2026-10-09-mechanics-schedule.md): the history gate, the 3 x 2 bundling, the 10 cm framing and the 40 N grip.

## Reference scale: about 35 cm of cloth per view height

The reference's twill wales are ~5.5 px apart on its 1624 px native frames. Denim wales are ~1.3 mm apart (3/1 twill, 0.38 x 0.58 mm pitch), so the frame shows ~35 cm, and the 0.38 mm warps are unresolved (~1.7 px). The earlier 10 cm framing had matched the 5 px rib to the warp pitch. Other cues agree with 35 cm: the pale threads are 3–5 px (~0.5 mm yarns), and the folds span about a third of the frame, consistent with denim's ~3 cm bending length.

The wales rise to the right at ~59°, i.e. the warps run vertically in view.

Consequences:
- 0.62 x 0.40 m coupon.
- Spans bundle 9 warps x 6 picks (3.4 x 3.5 mm cells, 21k nodes, about the same cost as before).
- Grip radius 14 mm.
- Camera at 0.98 m.
- Gestures and limits in the worker scale with the framing.

## Failure in strands

A bundled span fails in up to 4 strands (warps: 3 x 3, picks: 3 x 2). Each strand has its own strength multiplier, correlated along its yarns, and its own failure site, packed into spare `Seg` words. All strands share the span's extension history.
- The span law is the mean of the per-strand damaged laws (`bundleResponse`), so partial failure redistributes load within the span.
- Worn-through warps are authored as absent strands, not broken spans.
- Strand masks live in `Seg.flags` bits 8–11 and drive rendering. A cell whose four spans all carry material is still drawn as woven surface with its missing yarns removed from the weave evaluation (`CellYarns`). Only cells with a fully separated span fall back to explicit yarns.

## Robustness

- **f32 pivot cancellation.** Rigid modes of detached pieces are held only by inertia (m/h² ~ 6e-3 against 9e5 axial stiffness), so line-solve pivots can cancel to exactly zero. One such NaN spread through the global PCG and line-search reductions and blacked out the cloth. Every smoother pivot is now shifted by 2e-6 of its scale, in the GPU and in the reference (`PIVOT_SHIFT`; `solvePenta` exact by default). Reductions and the Newton update ignore non-finite contributions.
- **Detached pieces are not simulated.** After rupture events or a release, a CPU BFS over packed connectivity, flooded from the mounts and the grip core, marks unreachable nodes as ballistic (`x.w = 1`, fixed in the solve, quadratic air drag). They are retired (`x.w = 2`, hidden) once they leave the scene.
  - Pieces under 24 nodes held only by the grip are released as well: a few nodes on one grip spring get energy-injecting directions from the global Newton step.
  - Results computed before a reset are discarded (epoch).
- **Intact-surface control net.** It uses only ring points connected through surviving spans; otherwise it extrapolates. Torn or falling neighbours had stretched intact cells into long bars.

## Damage commit gate removed

With the production schedule, ~90% of moving nodes exceed any useful Newton-update tolerance, so damage almost never committed while pulling. Ruptures on three gestures versus a converged run (h = 1/240, 5 Newton x 3 PCG):

| gesture | converged | ungated | 20 µm gate | 2 µm gate (old) |
| --- | --- | --- | --- | --- |
| intact pull | 0 | 0 | 0 | 0 |
| tug | 18 | 11 | 1 | 0 |
| yank | 33 | 35 | 28 | 16 |

Ungated tracks the converged run and adds no spurious ruptures; `historyTolerance` now defaults off.

## Fold mechanics

- **Bending** was 5–10x below denim (measured flexural rigidity ~1e-4 N·m per width). It is now 4.5e-8 / 4e-8 N·m² per warp / pick.
- **Fold-stiffening energy.** The quadratic |x_a − 2x_b + x_c|² saturates toward a 180° fold. It now goes through Ψ(q) = q / (1 − q/4L²), the discrete-elastic-rod tan²(θ/2) measure, with a C2 Taylor tail past s = 0.85. Its Hessian kb(Ψ′I + 2Ψ″rrᵀ) is majorized by a scalar to keep the stencil format.
- **Shear locking.** The razor "crease" in feedback was an in-plane shear band: one tensioned warp line slid ~45° relative to its neighbours, and the adjacent cell collapsed. Cell diagonals stiffen 40x past 16% compression, ramped over 8% (≈20° shear jamming, and no area collapse).
- **Yarn compression** is now half the decrimping stiffness (was ~5%), so in-plane compression buckles instead of gathering like rubber.

## Grip and strengths

- Grip: 150 N max force, 6 mm lead.
- Worn-zone strength loss: weft 0.85, warp 0.96.
- With weft retaining 45% of its strength, no force produced openings short of ripping the whole patch rim. At 70–90 N nothing visible tore.
- Intact bundles (51–99 N) do not fail around a 14 mm pinch; the worn zone opens.

## Appearance at this scale

The weave is prefiltered into a per-layer tile: one twill repeat, rgba16f, with each mip level averaged analytically from the weave model in `weave_tile.wgsl`. Above ~0.3–0.55 yarn per pixel, intact cells sample the tile with anisotropic filtering and skip the 4-sample analytic weave (frame 9.9 → 5.8 ms). Damaged cells keep per-sample coverage but shade through the same model.
