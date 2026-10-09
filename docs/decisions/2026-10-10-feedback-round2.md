# Feedback round 2: settling, release, tearing at the grip, torn edges, self-contact

Supersedes parts of [feedback round 1](2026-10-09-feedback-round1.md): the 1-PCG schedule, the 14 mm / 150 N grip, normal-only air drag, all-four-spans shear, and explicit span ribbons at hole edges.

## Restlessness, release lag, tears on release: under-convergence

With 2 Newton × 1 PCG, the inexact solves inject energy. After a release, the interior KE grew (30 → 70 µJ, about 10k nodes above 2 cm/s, never settling). The release snap was damped and lagged. The bounce afterwards committed damage, so tears appeared on release rather than while pulling.

A near-converged run (5 × 4) settles in about 1.2 s and shows no ruptures after release. A second PCG iteration recovers most of this; extra Newton iterations do not.

| Schedule | Cost (batched, M5) | After release |
| --- | --- | --- |
| 2 × 1 | 4.0 ms | KE grows, never settles |
| 2 × 2 | 6.9 ms (+0.6 ms self-contact) | ~1 µJ residual within 1 s |

- **What made 2 × 2 affordable.** The line smoother's reduced block-tridiagonal system was solved serially on one thread, and that dominated dispatch latency. It now uses odd-even cyclic reduction in 128-thread workgroups, which cut the V-cycle from 2.0 to 1.35 ms.
  - Parallel cyclic reduction was tried first and lost about 2 digits in f32 on stiff lines.
  - Cyclic reduction is Gaussian elimination in a symmetric order, so it keeps serial accuracy on these SPD systems.
- **Extrapolating the previous acceleration into the initial iterate** was tested and rejected: unstable, the sheet shattered.

Physical causes found by isolating the remaining motion:

- **Drag.** Air drag used a lattice normal even across tears. Hanging strips then swung as undamped conical pendulums. Drag is now built from connected spans only (`drag.wgsl`, mirrored in the reference):
  - normal drag for sheet nodes;
  - drag across the strand for one-family nodes;
  - isotropic drag for isolated nodes.
- **Shear.** A cell lost both shear diagonals when any one of its spans broke, so hole corners were free hinges. A diagonal now acts while either of its triangles is intact.
- **Bending damping.** Bending viscosity rose from 2e-3 to 2e-2 s, a damping ratio of about 0.3 at 5–10 Hz fold motion; fabric bending is friction-damped.
- **Prolongation.** Bilinear prolongation across a tear ties both sides to the same coarse unknowns. Weights are now masked to coarse nodes reachable through connected edges in the cell, then renormalized (GPU `prolongWeights`, reference `buildProlongation`). On a cut flap fixture, 2 × 2 with masking settles like a near-converged run without it.

Under-convergence also under-tears: 89 ruptures at 2 × 2 against 406 converged on the same pull. Strength and grip are calibrated against the production schedule.

## Tearing where the hand pulls

The old grip (14 mm, 150 N) spread load so thinly that weak cloth far along the load path failed first. Now it is 8 mm and 300 N. Ruptures land within 2 cm of the grip on worn, moderately worn, and intact cloth. An in-plane pull opens a slit behind the fingers that gapes as the flap follows the hand. The handle speed limit rose from 0.8 to 2.5 m/s.

Span-level log-normal strength scatter (σ 0.25 + 0.2 × wear, uncorrelated between neighbours) makes crack paths jagged. With the earlier ~12% scatter, a crack tip always broke the next span in line, so tears ran dead straight along the lattice. Strength is not part of the drape hash.

## Torn rendering

Cells with a separated span used to drop their surface. Their spans were then drawn as bundle ribbons, which read as rectangular slabs, and stubs stretched with the gap between the separated nodes.

Now each live corner of a torn cell draws its own quarter (`vsTorn`):

- The control net uses real neighbours where spans hold. Elsewhere it uses rest-length points from the node's tangent frame (`tframe`, wear packed in `w` to stay within 10 storage bindings).
- Yarns are clipped at their own failure sites, with recoil.
- Edge yarns are lost where no crossing yarn still reaches them (fraying).
- Wear rises near free edges.
- Tails draw only the remnant beyond the quarter, anchored in the same frame.

Span ribbons were removed. Edge fuzz is kept, but darker and shorter.

## Self-contact

Single-layer scenes use node-to-face self-contact on intact cells (`selfcontact.wgsl`):

- **Penalty and friction:** k = 500 N/m, contact distance 1 mm, lagged Coulomb friction.
- **Reactions:** scattered to the face's corners with fixed-point i32 atomics.
- **Hash:** faces hashed on 12 mm cells; each node searches the nearest 2 × 2 × 2 cells.
- **Bound:** the OGC-style bound is now directional and taken from the start of the substep. Only the approach toward the nearest face is limited, to 0.45 of the clearance above a 0.3 mm core; sliding stays free. Penalty alone could not resist the 300 N grip.
- **Exclusion:** faces within two lattice cells are ignored. Bounding against them throttled pulls and tore the sheet.

The cross-layer contact uses the same directional bound semantics. The f64 reference has no contact, so GPU tests run with `selfContact: false`.

## Known gaps

- Detached pieces become ballistic and do not collide. A scrap torn off above the sheet falls through.
- Edge–edge crossings are not handled.
- Hole outlines still show cell-scale (3.4 mm) steps.
- The frame budget is tight: 4 × 7.6 ms + ~5.4 ms render, about 28 fps at natural time.

## Tools

`session.ts` scripts:

| Script | What it does |
| --- | --- |
| `snap` | Pull, hold, release; reports ruptures before and after release, and settle time |
| `tear` | Fast drag from `GRAB`; reports rupture distances from the grip |
| `flap` | Fixed cut flap ring-down |
| `drop` | Flap folded over the sheet; self-contact check |

Session environment variables:

| Variable | Effect |
| --- | --- |
| `STATS=1` | KE split by interior, edge, and debris |
| `GRIP` | JSON grip overrides |
| `NOMASK=1` | Plain bilinear prolongation |
| `RENDERCOST=1` | Per-pass GPU time of the final frame |
| `WAIT` | Duration of `rest` |

`bench.ts` accepts `SOLVER` and times batches of four substeps. `contactprof.ts` profiles the self-contact stages.
