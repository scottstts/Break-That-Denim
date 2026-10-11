# Floor contact (free play)

User request: long trailing flaps and torn-off pieces land and lie on the floor. Supersedes the
"no collision" decision in `2026-10-10-floor-and-room-light.md`.

## Decisions

- **One height, two users.** `src/simulation/floor.ts` FLOOR_Z (−0.12 m) is both the rendered floor
  (`Renderer.floorZ`) and the collision plane (`GpuSolver.setFloor`). Off by default in the solver (tests
  and the f64 reference comparison see no floor); the worker turns it on for free play. Stage turns it
  off while "Beneath" plays over the void and back on at its end. RETIRE_Z stays: with the floor on,
  nothing reaches it.
- **Law: the self-contact cubic against a rigid plane, in the node's own terms.** Assembled in
  `assemble.wgsl` (no extra dispatch, no ext slot), so it enters the rigid-mode blocks (rigidK) too.
  Contact distance = half the layer thickness (new `LayerMat.b.w`) + 0.3 mm rendered bulge. k = 1300
  N/m (one ply at half the compression of two: 4 × 500 × 0.65/1). μ = 0.35 (cotton on polished stone,
  the floor scan is terrazzo).
- **Hard bound in the step, as self-contact.** `step.wgsl floorBound` (projectStart and newtonApply):
  never below half the thickness, and no deeper per update than max(weight depth, 2 × current depth)
  (the trust region of the cubic). A falling node stops inelastically at its resting depth within one
  substep. The hard limit is an active bound in the operator (ACTIVE_BOUND), as cmeta bounds are.
- **Line search uses the penalty's quadratic model, not the exact cubic.** The iterate starts at h v
  (gravity only in the target), just outside the contact distance where the cubic has no curvature. The
  direction then runs ~0.66 mm into the floor, and the exact cubic rejected every step: pieces coasted
  on the floor. The other contacts use the same model + bound for the same reason.
- **Loose tails rest on it** (`thread_tail.wgsl`, `RenderU.floor` follows `solver.floorZ`).

## Found on the way: spinning detached pieces barely felt friction

A piece that landed spinning kept turning on the floor for ~0.85 s (Coulomb: ~0.02 s). Sliding was
right; spin was not. 8 Newton iterations gave the Coulomb result, so the 2-iteration solve was the
problem, not the law. Two linearization artifacts of rigid rotation, both fixed in the solver for all
detached pieces:

- **Start iterate (`rigid_spin.wgsl`).** A substep's velocity is its chord. Repeating the chord from
  the rotated position overshoots the circle and stretches spans by θ² (0.3 % at 10 rad/s). The
  geometric stiffness of that tension (~1500 × node inertia) resists any change of spin in the
  linearized system. The start becomes h v − 2(1 − cos θ) r⊥ about the piece's centroid: the continued
  rotation. ω = I⁻¹L per piece. The inertial target (pred) is unchanged. A tangent-velocity model
  (h v + (R − I − [θ]×) r) was tried first; it left half the stretch once velocities were chords.
- **Update path (`energy.wgsl steps`, `curveApply`; `rigid_spin.wgsl curveSet`).** The direction's
  rigid rotation w (rigid-mode start) is applied as R(αw), not the tangent αw × r. The tangent stretches
  spans by (αw)²/2, and the line search accepted only ~¼ steps. The line search evaluates the same
  curved path it applies. Per-node `curveRef` (w, arm) keeps the energy kernel at the 10 storage-buffer
  limit. It is zeroed when the last piece goes; curveSet rewrites all nodes while pieces exist.
- Measured on a 5 × 5 patch at 10 rad/s: 10 → 6.5 → 3.5 → 1.0 → 0.07 rad/s per substep (converged
  8-iteration solve: 6.1, 2.4, 0.15). Start fix alone: first substep only. Curve alone: ~1.3 rad/s per
  substep. Free-spin test now keeps 0.00 % strain (was 0.12 %), spin 8.43 of 10 rad/s.

## Effects

- Substep cost unchanged (benchmoving 8.81–8.88 ms with or without the floor, as before).
- `maul`: ruptures, rejections and the never-settling strand on the sheet are identical. The node that
  used to fall out of the scene stays.
- **For the user: "Beneath" tears less** (floor off there; the rotation fixes act on its scraps):
  936 ruptures / 225 freed by 12 s, previously 1042 / 338, with 2 rejections where there was 1. The
  face is still fully shown through one hole (`artifacts/captures/beneath-floor-compare.png`, previous
  left). The earlier record attributed the ~140 extra ruptures to scraps hitting the creature.
  Re-staging is the user's call.

## Open

- A near-vertical hanging flap renders with streaky weave at grazing angles (`trail` session).
  This comes from the renderer, not the floor.
- Floor friction is static-plane only (no rolling, no floor compliance). The hard bound is z-only
  and applied after the cmeta projections, so a node in a pile could be lifted slightly against a
  bound from a face above it.

## Reproduce

`tests/gpu/floor.ts` (in `npm run test:gpu`); `tools/studies/floorslide.ts` (V, OMEGA, NOCURVE, NOKR,
LSDUMP, ...); sessions `fall`, `trail` (NOFLOORCONTACT=1 for the old behaviour), then `still.ts
session-fall`; `tools/studies/benchmoving.ts` FLOOR=1.
