# Detached material keeps its mechanics; contact at the onset (acting on artifacts/assessment)

Assessment: a detached piece's nodes moved as independent ballistic particles while its spans stayed
intact, so the renderer drew them as threads stretched up to 85x their material (maul, 147 runs).

## Detached pieces stay in the cloth solve

- The stated reason for taking them out (f32 cannot resolve rigid modes next to yarn stiffness) does
  not hold here: an unanchored piece in the solve falls at the drag-limited rate (freepiece.ts), and
  hinged flaps always relied on the same modes. x.w = 1 now only means detached: no grip takes it, it
  is drawn as its own surface, it is retired out of the scene. Its spans load, damage and break; it
  collides as cloth (two-way, with friction) and with a staged body. The ballistic drag, commit clip
  and ad-hoc debris friction are gone (cloth drag and contact apply).
- Retirement is per node and removes its spans (SEG_EXISTS cleared, not SEG_BROKEN: no tear is
  counted or heard). A frozen retired node left connected anchored the rest of its piece below the
  scene. The multigrid fixed flag is not set for it (the step module is at the 10 storage-buffer limit):
  its row is the identity with nothing coupled, which the hierarchy tolerates.
- A pinch-crushed piece released now re-expands (5 % compression gone within a frame) and may tumble
  (0.06 m/s relative speed is rigid rotation); the interaction test measures span strain instead.

## Active contact bounds in the Newton operator

The OGC bound clips a node's step after the linear solve; nothing else knows. A pair landing across
a face edge had its span stretched 16 % in one substep by the clip and ruptured. A node at its bound
whose net force drives it on gets 100x its diagonal along the bound normal in the operator only
(active set; the line search keeps the true potential): the rest of the system takes the correction.

## Contact onset (pre-existing, exposed by free bodies)

A free thread lying on a hanging flap chattered at 1-4 cm/s indefinitely; the old build does the same
with the thread made cloth. Newton does not converge at contact onset: gravity alone moves a free node
h^2 g = 0.68 mm per substep against a 1 mm contact distance, and the penalty's stiffness is zero (or
jumps) where contact begins, so one iteration sank a light node 0.4 mm and the penalty kicked it back.
8 Newton iterations settle it; PCG iterations, h / 2, softer or stiffer k, K = 8 do not.

- Chosen: cubic penalty k s^3 / (3 delta) (C2, stiffening with compression; k = 500 gives ~4 kPa at
  0.3 mm) plus a per-iteration trust region: a step may go no deeper than max(depth carrying the
  node's weight, twice its depth now), applied by tightening the node's own bound for the rest of the
  substep. The model's curvature is then within 2x of the true one.
- Rejected: majorized stiffness over the whole bounded range (nodes near but not touching a face
  coast on inertia, ignoring gravity; worse); a 1D inertia+gravity+penalty "touchdown" start (ignores
  the support's own motion: a dropped scrap was launched 15 mm); quadratic law + onset truncation only
  (sustained 2-20 nJ residual in the fold state).
- Result from rest: fold state 0.2 mm/s (was 1-4 cm/s), strand on a drum 0.01 mm/s (was 28 mm/s),
  dropped scrap rests 0.9 mm above the sheet. Remaining sub-mm/s creep is the regularized friction.

## Bending viscosity is objective

The viscous term compared world-space bending vectors, so it damped rigid rotation (45 uJ for a 30 deg
turn of a bent triple). It now acts on the rate of |r| (both backends, Hessian majorizer unchanged).
Release recoil overshoot stays within the converged 6 x 4 solve's.

Re-settling the drape with it ends in a different wrinkle state (3.3 mm max, 0.34 mm mean). The shipped
drape is still an exact rest state (viscosity vanishes at rest) and was kept; MECHANICS_VERSION is
unchanged. Re-running gpu:drape would replace the approved look.

## Measured

maul (ASYNC=1, the assessment's observer): worst detached span 1.075x rest (was 84.6x), detached long
runs 0 (was 147), cloth 1.40x (1.42x). Gestures haul/snap/tear/rip/settle/maul/scrap/drop: wall time
equal, line-search rejections 2 (were 8), post-release ruptures 2 (were 24). Substep at rest +0.5 %.

The active bounds remove most of the tearing that only the clip produced: maul ruptures 581 (old, 2 x 2)
and 618 (old, converged 6 x 4) against 285 / 229 now; the old clip never reaches the constrained
solution at any iteration count. The staged "Beneath" lunge accordingly tears 938 spans instead of 1866
and its hole hides the upper face at 6-9 s; re-staging is the user's decision (physics not bent back).

## Open

- Attached runs' held ends drawn 13-26 mm off their chain, and the body's missing swept test: both
  addressed in 2026-10-11-parted-crossings-and-swept-body.md.
- Backward Euler damps rotation (a free piece loses 16 % of a 10 rad/s spin in 0.25 s; first order in h).
- Multi-layer contact (contact.wgsl, not the product scene) keeps the quadratic law.

## Reproduce

`npm run test:gpu` (tests/gpu/detached.ts: differential pull, edge landing, free tumble, strand at rest,
bending viscosity), `tools/studies/restcontact.ts <state>` (does a dumped state come to rest),
`tools/studies/releasepiece.ts`, `session.ts` with `FASTEST=n`, the assessment's observe.ts.
