# G1 (partial): mechanics representation and solver structure

Status: CPU f64 reference evidence only. GPU cost is unmeasured; the GPU port mirrors this structure and will be measured against it.

## Representation: full yarn-level lattice, no shell

Design §3 prefers full yarn mechanics over the whole coupon if its cost passes, which removes shell/yarn coupling (U6) and promotion (U8). We start there.

Each layer is a woven crossing lattice in the sliding-crossing family (R2): one node per warp/weft crossing, shared by both yarns; yarn spans between crossings carry axial, bending and cohesive terms; cells carry a weak diagonal shear pair. Crimp is geometric (render offsets at crossings), and decrimping is represented by a C1 J-curve axial law. Yarn sliding is a friction-limited transfer of rest length between neighboring spans of the same yarn (capstan-like normal force from crossing-yarn tension), which conserves yarn material and turns free tails into pull-out: when a tail is consumed the yarn leaves the crossing and its last span becomes a tail of the next crossing.

Measured reference scale fixes the lattice: denim warp pitch 0.38 mm, weft 0.58 mm (rib period ~5 px, weft-bridge period ~12 px at ~13 px/mm on native frames), tan backing 0.5 mm. Three 0.11 × 0.10 m layers give ~143k nodes.

## Why plain block relaxation was rejected

Yarn-line block Gauss-Seidel (exact Newton solve along each yarn, zebra colors) resolves in-plane motion in 1–2 sweeps, because each displacement component is solved exactly along its own yarn family. Out-of-plane motion does not converge: cross-line bending couples lines into a plate (biharmonic) operator, and local stiffness exceeds per-node inertia by ~1e5 at h = 1/240 s. Measured on a 70×50 coupon: 8 sweeps per step leave 10–25% (max-norm) error in the step displacement; the result falls visibly in slow motion. More sweeps or smaller steps do not change the picture.

A rediscretized nonlinear multigrid (FAS/MG-OPT, coarse yarn = 2 fine yarns) gave descent directions but overshot when solved accurately: a buckled, compressed sheet looks compressed (soft) to the coarse chords. It would need line searches and coarse topology bookkeeping.

## Selected: projected Newton + PCG with Galerkin multigrid

- Fine operator: projected PSD Hessian of the incremental potential (secant tangent on the cohesive softening branch, clamped geometric terms).
- Coarse operators: P^T A P with bilinear prolongation over every second yarn; on the lattice this stays a radius-2 stencil of symmetric 3×3 blocks at every level. Holes/broken spans propagate automatically.
- Smoother: zebra (2 colors per family) exact block-tridiagonal yarn-line solves; same-pass couplings (±2 along a line, lines two apart) are majorized onto the diagonal with ||B||_F I. Forward order pre-smoothing and reversed post-smoothing make the V-cycle symmetric.
- Bilinear transfers are borderline for the 4th-order bending operator (two-grid rate ~0.47 even with an exact coarse solve; the slow error is the smooth out-of-plane bulge). PCG removes this: energy-norm error 2.7e-2, 4.5e-3, 1.3e-3 after 1, 2, 3 PCG(V-cycle) iterations (40×30 coupon, zebra smoother).
- Dynamics on the 70×50 coupon versus a fully converged solve: 2 Newton × 4 PCG gives ~2–4% max-norm error of the step displacement while moving; 3 × 3 gives ~2%.
- Inertial initial iterate (h v, gravity only in the objective): extrapolating gravity drops the interior against the pinned collar and creates a kinked boundary layer that Newton recovers slowly; with the cohesive law that previously committed spurious ruptures.

Contact is planned as two-sided node-to-segment penalty terms treated diagonally in the Newton operator (the opposite side detects its own contacts), with OGC-style per-node displacement bounds. This is a simplification of design §10 (no exact reaction coupling in the linear system) and must be revisited in the contact fixtures.

The IPC/ACCD versus AVBD/OGC race (U2) was not run; this structure is candidate A's Newton/PCG with a yarn-aware preconditioner and coarse correction, combined with OGC-style bounds instead of ACCD line-search filtering.
