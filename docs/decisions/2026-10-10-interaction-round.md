# Interaction round: restless settling, grip release, threads through cloth

## Restless settling: spurious line-search rejections

- After a tearing drag the line search rejected 164 of ~240 Newton iterations within a second of
  release (`rejected` in session logs). A rejected iteration takes no step: the substep keeps the
  inertial predictor `h v`, with no elastic, damping or gravity response. Flaps coasted and swung for
  seconds (the "flipping" after a stretch); the round-7 note on restlessness under 2 x 2 with
  self-contact was this.
- Cause: the bending energy change on the upper (quadratic) branch of `bendPsi` was `Psi(q1) - Psi(q0)`.
  On sharply folded triples (crumpled debris, a gathered pinch) that lost ~1e-7 Psi, a constant
  1e-3 J at every step length, far above the line-search rounding bound (which scales with the
  terms' changes, not their values). Debris triples, whose step is exactly zero, contributed most.
- `bendPsiDelta` is now a product with `dq` on both branches and across the joint (laws.ts, common.wgsl).
  Same drag: 0 rejections, settled 0.73 s after release instead of re-swinging until ~2 s.
- Probe: `LSPROBE=t` in session.ts runs each substep stage by stage from time t and prints g.d, the
  evaluated changes and tolerance per Newton iteration; on a conclusive increase it lists the nodes
  contributing most, split into axial / viscous / bending / shear.

## Grip

- No slip release. The grip holds until the button is released; force stays capped (120 N, 6 mm lead).
  The slip rule (3 cm beyond reach for 0.25 s) fired before a fast drag even finished; on intact
  cloth a long hold is stable (2 cm deflection, no ruptures, clean spring-back).
- A piece torn out inside the pinch stays held whatever its size (`MIN_HELD_PIECE` removed). Under 24
  nodes it used to become debris, so the hand suddenly held nothing. Its original reason (energy
  injection into a few nodes on one grip spring) did not reproduce after the line-search fix: the
  scrap hangs from the handle.
- Lone cloth nodes (all spans broken, not yet freed by the asynchronous topology pass) can reach a few
  m/s for a frame; they are single crossings and become debris at the next analysis.

## Threads through cloth: runs bridging ruptured spans

- A bundle span ruptures when each of its yarns separated somewhere within it, and its material goes to
  the two nodes as tails split at `failSite`. A yarn of that bundle can still be uncut along its own
  lone run (its cut lies just past where it is held), so the thread scan saw a run held at both ends
  that no chain carries. The renderer drew it straight between its ends: taut parallel bars across
  the opening, through any flap between (user screenshot).
- Runs held at both ends are now split at each non-holding span's failure site (`threads.wgsl
  emitSplit`); each part hangs from its own side. Runs with a free end were already hanging and are
  unchanged. Straight runs: 10 / 24 -> 0 on two torn states; run counts unchanged.
- Mechanically, spans through intact cells are rare (`tools/studies/crossings.ts`: 0-5 of ~41k sheet
  spans, 0-1 strand spans after full sessions). Segment-face contact was not needed for this.

## Measured and rejected

- **6 x 4 bundles** (2.3 mm cells, 47k nodes): 16.2 vs 7.8 ms per substep. Holes were more rectangular,
  and with strength calibrated for 9 x 6 the cloth broke far from the grip (434 spans 5-10 cm away vs
  ~9). Finer bundles do not by themselves make worn cells fail diffusely.
- **Out-of-plane pluck** of a worn core (5 cm toward the viewer): the cloth tents, nothing breaks within
  2 cm of the pinch, 19 spans break 2-10 cm away along the warp tension. A pinch cannot reproduce the
  reference's dome-like biaxial stretch, under which its lobes grow.
- **Ruptures after release** are not an under-convergence artifact: production gives 0 after release on
  `snap` / `rip`, a 5 x 4 solve gives 15 / 5. Production under-tears relative to 5 x 4 (116 vs 176 on
  `rip`), as recorded before.
