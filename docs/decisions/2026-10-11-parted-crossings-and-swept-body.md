# Parted crossings at rupture; swept body contact (the detached-coherence round's open items)

## A separated span's yarns part from the other side's weave

Thread ends were drawn 13-26 mm off their chain because the field kept crossings that tied two pieces
the lattice had already separated. A broken span's yarn belongs, up to its first separation, to the
owner a and beyond it to b (the tails already split its length this way, and threads.wgsl ownerState
draws it so). The span's first ceil(P / 2) crossings sit on a's cross yarns, the rest on b's. A stub
of b's yarn still woven into a's cross yarns (or the reverse) was the bridge.

- At rupture (coupleSpan / coupleSpans) such crossings get DAMAGE_PULLED: the stub's end slides out.
  The cross yarn has no free end nearby, so it stays. No new code: "pulled out" is exactly this
  state, and load sharing and pull-out already treat it so. Silent (no fracture emitted).
- Crossing slip that empties a tail (step.wgsl slip) also breaks a span. Its yarns left the crossings
  at one end, so the whole span belongs to the other end. The step module is at the storage-buffer
  limit and cannot bind the field, so slip sets SEG_SLIPPED with failSite 0/1 (the end), and the next
  coupling parts that end's half. That is one substep of lag. In maul, 3 of the 7 remaining off ends
  were this path.
- couple is split per family (coupleU, coupleV). A crossing's u32 holds both families' codes, and
  writing both in one dispatch would race.
- Authored defects are parted at compile too (25 spans, 56 crossings in the shipped scene). The drape
  is unaffected because broken spans carry no force.

Measured, maul (ASYNC=1, runends.ts): thread ends > 5 mm off their chain 41 -> 0, > 2 mm 77 -> 3; the
worst is 24.5 -> 2.3 mm (1-2 mm is the surface vs chain within a cell).

Consequence: none on how much tears. Measured with the pinch slip-out in place
(2026-10-11-pinch-slip.md), parting on / off: haul 30/28, snap 56/76, tear 41/34, rip 102/115, settle
70/72, maul 287/290. The +18-42 % first measured on tear-outs, and the line-search stalls, came from
the grip pulling torn-off cloth from far away (that record). Wall time is equal.

## Body contact is swept

bodyTerms tested only the end position: a 20 mm step through a 4 mm capsule got no force, and a limb
crossing a node left it on the far side, where the closest point pushed it the wrong way. Fingers
move up to 5x their radius per substep (sweepprobe.ts, 36 mm at 7 mm).

- Both motions are linear over the substep. Conservative advancement on the gap uses the largest
  speed of the node relative to the segment's points, max(|dp - da|, |dp - db|), not |dp| + |da|. A
  node riding on a limb then advances in one step, and only nodes within one relative step of a
  capsule loop. The loop runs at most 16 steps; a grazing path that has not converged by then falls
  back to the closest point.
- If the path enters the solid, the node meets the tangent plane where it first touched. The plane
  is carried to the end pose (the touch parameter s on the segment, the normal turned by the axis's
  shortest-arc rotation). Otherwise, or when the node starts inside, the end-pose closest point is
  used as before. Applying the start plane to every contact was not taken: a node sliding round a
  7 mm finger would be held to a stale plane.
- The bounds include the previous pose, and the cull tests the node's path box.

Measured: test:gpu body cases (through a still sphere -46.5 N, swept through by a limb -61.5 N, riding
and passing beside unchanged). Beneath (TRACE=1): ruptures by 6 s were 901 with both changes, 1034
with parting only, and 938 for the previous build; the final breakup tore 2 / 102 / 28. Without the
sweep, limbs that crossed the cloth pushed nodes out the far side and stretched spans round the limb.
At 6 s the lunge now opens one clean hole around the mouth, with no material bridging it (sheet
artifacts/captures/beneath-swept-parting.png: both | parting only | previous). Substeps took 20.6 s
against 21.8 s for the previous build.

## Open

- Slip-broken spans part one substep late (no field binding in the step module).
