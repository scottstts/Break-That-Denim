# Stability round: acquisition, release, contact (acting on artifacts/assessment)

User symptoms: grabs that pull material up through the cloth from below, overreaction to a slight
grab, a strong bounce after release. Evidence: `tools/studies/stabilityAssessment.ts` (assessment
probe), `tools/studies/release.ts` (grab-stretch-release recoil), `tests/gpu/interaction.ts`.

## Acquisition

- **Neutral grab.** The GPU stores each grip's origin (the picked node's position when `gripInit`
  runs) and applies host targets relative to the host's attach anchor: handle = origin + target -
  anchor. A stationary pointer imposes nothing however stale the pick readback is (5 mm stale anchor:
  0.0001 mm motion). Median target jump at acquisition 1.36 mm -> 0; click transient (pointer case)
  9.1 -> 5.2 uJ peak KE.
- **Visible picking.** Ray against intact faces (two triangles) and against surviving spans not
  covered by an intact face (capsules of a quarter pitch); the nearest hit wins. Over no surface,
  the nearest crossing in the cursor disc that no intact face hides (0.5 mm margin). A material
  endpoint counts incoming spans. The old picker took any node within an NDC depth band (7.6 mm at
  default zoom), so it picked the far layer of a fold or the sheet under a flap.
- **Patch through current connectivity.** The pick reads back a (2R+1)^2 connectivity window; the
  patch keeps crossings reached from the picked node through surviving spans by a path of at most
  1.5 R (intact cloth: the old rest disc exactly). Grip kernels skip entries that stopped being cloth.
- **Topology anchors** are read when an analysis applies (anchor source), not when it was requested;
  coalesced requests resolve when the rerun completes.

## Hand (src/simulation/hand.ts, shared with the headless drivers)

- Engagement on the simulated clock; pointer samples (coalesced events, shared epoch clock) read at
  each substep's input time; pointer-up releases at the substep reaching its time.
- The handle never goes behind the grab plane (wheel depth included). Acceleration bounded at
  60 m/s^2 on top of 2.5 m/s.
- Session/twohand drivers now pick through the product picker and use the hand, so their gestures
  include the product's lift; results are not comparable with earlier session logs without
  `HAND='{"engageLift":0,"liftPerDrag":0,"maxAccel":1e9}'`.

## Release bounce: numerical, fixed by event-targeted Newton work

On intact cloth (3 cm pull, held point released 12 mm from rest), 2 x 2 regains KE after the snap
(42 -> 74 uJ at 70-100 ms), overshoots 8.7 % and rings twice; 6 x 4 at h and h/2 decays
monotonically with no overshoot. `RELEASE_REFINEMENT`: 4 Newton iterations for 0.1 s after any grip
ends removes the rebound on every loading measured (intact pull, 3 cm lift, weft pull, worn-zone
drag, click). Overshoot that remains under a lift also appears in 6 x 4 (gravity), so it is physical.

Rejected: 3 iterations (5 % rebound); a 0.05 s window (12 % overshoot); a 40 ms finger-opening ramp
(adds overshoot under 2 x 2, no gain under 6 x 4); gated iterations (the whole snap is unconverged:
saves ~10 %). Cost: 14.4 vs 7.9 ms per refined substep, ~80 ms of extra GPU per release; the snap
plays at ~0.55x for ~0.2 s. Post-release ruptures in sessions went to 0 (were 2-5).

## Held pieces flung on release (user report: a thread cluster jumps away)

- A small piece torn out inside the pinch is crushed in-plane toward the pinch point (nothing around
  it to buckle into: a 3 x 3 piece measured 3.9 mm across from ~7-10 mm). On release the worker freed
  it only when the asynchronous connectivity analysis applied, a frame or more later; until then it sat
  in the cloth solve with no anchor and its stored compression sprang it apart (haul with the
  worker's timing: 4 m/s, spread over 15 cm in one frame). The session harness awaited the analysis at
  release, which hid it (`ASYNC=1` now reproduces the worker).
- Now the fingers let go in two steps (`Interaction.letGo`): the grip keeps holding at its last target
  but stops anchoring, the analysis runs, and the springs are removed once it applied (at most a frame
  later), so the piece is already ballistic. Same haul: it stays together (3-4 mm) and falls the 5 cm
  it had been lifted.
- Not the cause: an unanchored piece at rest in the solve just falls (`tools/studies/freepiece.ts`);
  a pinched cluster still attached by a strand pops back at <= 0.16 m/s; pieces freed by ruptures leave
  the solve at the speed the hand gave them (0.4-1.2 m/s; 3.7 m/s once, ripped off by maul's fastest
  reversing yank).

## Pinch: two pads instead of a radial contraction (user decision)

- The radial pinch pulled every patch target toward the grab point to 0.3 of its spread: on intact
  cloth it mostly buckles (held span strain median -1.8 %, min -17 %), but a small torn piece with
  nothing around it is crushed in-plane.
- Now (`GRIP.pinchMode` 'pads'): the disc's halves either side of the grab line (rest coordinate
  along the closing axis, |u| >= half a pitch) are two pads; each slides rigidly toward the line by
  (1 - pinch) pitches over the engagement, along the axis taken on the GPU from the grab-time offsets
  (+ pad minus - pad). The strip between is not held; it buckles as the mechanics make it (on the
  hanging drape it sags 3 mm into a loop between the pads). Closing axis: u (warp, horizontal in view);
  v made a short, deep shadowed pocket that read like a hole. Without cloth on both sides (neither pad
  under a quarter of the other) the other axis, then the radial pinch.
- The strip is owned by its grip (no second finger takes it) but not an anchor: as an anchor, a strip
  crossing cut loose stayed cloth and fell 2.4 m during a hold.
- Measured vs radial: held patch min strain -5 % (was -17 %); crease release energy 8.3 vs 16.7 uJ
  (refined); freed held piece 7.3 mm across (radial: crushed to 3.9 mm); tear locality unchanged
  (ruptures within 5 cm of the grip, a few at 5-10 cm).

## Prolongation links only load-carrying spans

A span reaches intact ~ 0 (every yarn softened through) before it breaks (yarns separated); the
connectivity-aware prolongation still linked across it, tying a nearly free crack-tip node to the far
side's coarse unknowns (line-search probe: 68 mm directions, every step length conclusively uphill,
step rejected for the whole sheet). Links now need intact > `LINK_MIN_INTACT` (0.02, both backends).
Rejections in haul/snap/tear: 41 -> 23 summed over both pinch modes; 12 haul runs (6 grab points x 2
modes): rejections only at the default worn grab point, 0 elsewhere, either way. The remaining ones
are the same crack-tip pattern: slack, nearly cut spans let the Newton step overshoot into tension.

## Contact (selfcontact.wgsl)

- Faces hashed at their predicted centroid and, if different, their start centroid.
- A node scans the hash cells within half a cell of its whole predicted path (start to end, up to 4
  cells per axis): nearest face at the start, faces its path crosses (at start or predicted face
  positions), candidates near the end.
- The bound is against the first crossed face (its plane, on the start side, at any distance), else
  the nearest start face. Its four corners get a reciprocal bound (same share, toward the node): keys
  `(4095 - share/10 um) << 20 | node + 1`, atomicMax into a spare `pairs` word, resolved into a second
  `cmeta` vec4 per node. Node share + face share = 0.9 of the clearance above the core.
- Lattice exclusion only for faces linked to the node through surviving spans (L-shaped paths);
  the test runs only where a per-row damage mark (`damageRow`) shows a broken span or debris within 3
  cells, so intact cloth is unchanged and pays nothing.
- Ties (a closest point on a shared edge) go to the lower face index: with the reciprocal bound the
  winner matters, and hash fill order made sessions nondeterministic.
- The predicted step is made feasible before the first Newton iteration (`projectStart`).

Assessment fixtures: moving face -1.5 -> +0.96 mm, 20 mm predictor crossing -18.5 -> +0.96 mm.
Spans through cells after maul 23 -> 4-6 (18 of 23 were inside the old exclusion window).
Cost: +1.8 % per substep, rest and moving (`tools/studies/benchmoving.ts`).

Rejected: projecting the Newton direction onto the bounds before the line search (so the evaluated
step is the applied one). The line search is global: a non-descent projected direction rejects the
whole step and the sheet coasts; maul rejections 5-8 -> 42-43.

## Open

- Grip force at the iterate can exceed 120 N (bound applied at committed x only); a consistent
  saturating potential previously hurt convergence.
- No edge-edge or strand contact; torn cells lose their collision face; tiny scraps are ballistic.
- Bending viscosity damps rigid rotation of a bent chain (world-space curvature rate).
- Rare line-search rejections at crack tips (nearly free nodes whose slack spans let the Newton step
  overshoot into tension; the global line search then rejects every node's step). Cause found: a grip
  spring on a pinch rim torn off the patch (2026-10-11-pinch-slip.md).
- `session.ts` "settled" flag resets on any single 2 cm/s frame (debris blips): read the vmax trace.
