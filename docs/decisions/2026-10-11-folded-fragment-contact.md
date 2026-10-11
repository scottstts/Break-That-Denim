# Investigation: folded detached fragments still float

The rigid-mode change does not finish the floating-scrap fix. With the current source, a staged
fragment can stay almost still because contact within that fragment resists its common motion.
This is a solver defect even with no creature, grips, mounted cloth, or other fragments present.
No production mechanics were changed during this investigation.

## Evidence

The staged headless run with asynchronous topology requests and two substeps per presented frame
produced a folded 208-node detached component around 6 s (first node 9543). This analysis cadence
produces a different tear graph from the previous four-substep captures; those captures did not
establish that other fragment shapes fall correctly. Saved state:
`artifacts/captures/investigation-async2-6.state`.

Replaying from zero velocity, with the creature and grips absent, keeps this component slow. It has
zero active span links to the mounted cloth. Its weight is 7.880 mN; its summed contact force is
zero to the printed precision. At 0.1 s, 32 nodes have a closest candidate face within contact reach;
all of those faces belong to the same component.

Controlled ablations, keeping gravity, damage, mechanical spans, air drag, the rigid-mode correction,
and contacts with other material unchanged:

| Replay | Fall after 0.1 s | Downward speed |
| --- | ---: | ---: |
| Current solver | 1.44 mm | 0.0164 m/s |
| Remove only internal contact bounds | 1.44 mm | 0.0164 m/s |
| Remove only internal contact terms | 5.67 mm | 0.0508 m/s |
| Remove both internal contact terms and bounds | 51.36 mm | 0.9320 m/s |

Retiring every node outside the component also leaves it stuck: 1.48 mm in 0.1 s, 3.40 mm in
0.25 s. This rules out hidden support by the creature or the main sheet for this case. Laying the
same isolated tear graph flat gives an initial downward step of 0.66 mm instead of 0.14 mm; it
subsequently buckles and develops internal contacts again. Shape and contact, rather than a mode's
gravity setting, determine whether the defect is exposed. The staged claws generate tightly folded
fragments; the existing compact-piece, strand, comb and serpentine checks do not cover this case.

## Mechanism

- `selfcontact.wgsl:522`: node-face contact contributes only diagonal Hessian blocks. Equal-and-opposite
  forces are scattered, but the off-diagonal coupling that would cancel stiffness on a common
  translation is missing. Internal contact therefore acts like an external restraint in the Newton
  operator. `assemble.wgsl:190` includes that Hessian in the detached piece's rigid block as well, so
  the new rigid-mode correction inherits the restraint.
- `selfcontact.wgsl:448` and `step.wgsl:61`: the contact clearance is divided into separate limits on
  each node's absolute displacement. These limits also restrict a folded piece translating together,
  even though its internal gaps would not change. Removing the contact terms alone still leaves
  this limitation; removing only the bounds still leaves the diagonal stiffness limitation.

There is no attractive force between the scrap and the sheet in this reproduction. The apparent
stickiness is numerical resistance to common motion. Oversized creature capsules remain a separate
possible source of visible gaps for scraps actually resting on the body; refitting them cannot fix
the isolated folded-fragment failure.

The repair needs relative node-face contact in both the solve and its motion limits, so internal
contact preserves a free component's common motion. Disabling self-contact or exempting all detached
material would hide the defect while allowing folded cloth and separate pieces to pass through each
other. A regression must drop an unsupported, folded, self-contacting component, alongside the
existing support and coherence contracts.

## Reproduce

```sh
zsh -ic 'ASYNC=1 SUBSTEPS=2 DUMPAT=6 FRAMES=6 UNTIL=6.1 node tools/gpu/run.mjs tools/stage/beneath.ts investigation-async2 1200 750'
zsh -ic 'node tools/gpu/run.mjs tools/studies/foldedfall.ts investigation-async2-6 11550'
```

The second command uses the saved component above. For a newly generated state, use a live detached
node from its large folded component (`DEBRIS=1 DEBRISAT=6` reports the components).
The probe compiles diagnostic shader ablations only; it does not alter production shaders.

## Follow-up: isolate what the staged scene adds

The visible surface was mapped back to its mechanical owners using the renderer's own material-map
shader (`tools/studies/visiblepieces.ts`). In the saved six-second capture, the island centred near
pixel (437, 357) at 1200 x 750 has 8,345 surface crossings carried exclusively by the 208-node
detached component above. Its nodes have no connection to the mounted sheet. This ties the numerical
probe to a visible staged scrap; it does not identify every scrap in the user's montage.

Matched two-second replays start from that same captured shape and zero velocity. The animated-body
cases play the creature's clip from 6 to 8 seconds, without grips, piercing, hand reach corrections,
or the audio-driven extra jaw opening. Only the presence/motion of the collision body and the
diagnostic internal-contact ablation vary:

| Body | Internal self-contact | Result |
| --- | --- | --- |
| Absent | Current | Falls only 37.84 mm in 2 s; still live |
| Animated | Current | Ends 10.15 mm **above** its start after 2 s; downward speed only 4.9 mm/s |
| Absent | Ablated | Falls out of the scene by 0.25 s |
| Animated | Ablated | Falls 135.94 mm by 0.5 s; out of the scene by 1 s |

In the animated/current case the body first pushes the fragment upward. At two seconds no node is
within 2 mm of an active capsule, the summed external vertical force is about 0.0001 mN, and internal
contacts remain. The continued suspension therefore does not require continuing creature support.
The stage provides the folded shape and moving collisions; the shared solver can retain the slow
motion even after those external contacts end. Freestyle working with other torn shapes does not
exercise this same state, and no stage-specific gravity or sleep switch was found.

The body also produces real invisible support. A frozen-body replay catches nodes about 1.5 mm from
the head/shoulder capsules while they are 8–35 mm from the actual blend-skinned mesh triangles.
This measurement uses all four skin weights and point-to-triangle distance, unlike the older
dominant-joint/nearest-vertex estimate. However, with internal contacts ablated, both frozen and
animated body replays clear the body by one second: oversized capsules alone do not explain the
persistent floating in this reproduction.

The visual probe also finds small surface islands whose owners still belong to mounted mechanical
cloth. Surface islands can be connected by rendered loose threads, so that observation alone does
not establish another floating-piece defect.

```sh
zsh -ic 'node tools/gpu/run.mjs tools/studies/visiblepieces.ts investigation-async2-6 6'
zsh -ic 'FALL_TIME=2 FALL_VARIANTS="production,no internal contact" node tools/gpu/run.mjs tools/studies/foldedfall.ts investigation-async2-6 11550'
zsh -ic 'FALL_TIME=2 FALL_VARIANTS="moving body,moving body no internal" node tools/gpu/run.mjs tools/studies/foldedfall.ts investigation-async2-6 11550'
```

This follow-up changes development probes and notes only. The uncommitted production attempt still
does not repair the relative-contact operator or bounds described above. Lint/typecheck and 67 unit
tests pass; the matched headless probes reproduce the unresolved behavior.

## Verification: two hands separating the centre piece

`tools/studies/topologyAudit.ts` independently floods the live active-span graph, without treating
the engine's detached flags as connectivity boundaries. It compares physical connections to mounts
and both current claw cores with `x.w`, and records the GPU solver's piece labels. The optional
`TOPOLOGY_AUDIT=1` in the staged runner submits readbacks without awaiting them between frames.

The nonblocking trace reproduced the floating 208-node component (seed 9543, watched node 11550):

- At 2.1500 s its material still belongs to the mounted component, which both hands hold.
- At 2.1667 s all 208 nodes form a separate component with no mount or claw-core connection; the
  asynchronous classification has not applied yet.
- At 2.1833 s all 208 nodes are detached and share one valid GPU piece ID.
- Across the next 236 classified samples through 6.1 s, the component remains detached, unanchored,
  and assigned one valid ID per sample, while it continues hovering. IDs can be renumbered when other
  pieces change; the component is never split across IDs or omitted from the solver's piece table.

Across all 366 frames there are no anchored nodes incorrectly marked detached, no unresolved
unanchored nodes at the end, and the largest observed classification delay is one presented frame
(16.7 ms of simulated time at two substeps/frame). The trace is
`artifacts/captures/twohand-audit-labels-topology.json`; the floating result is
`artifacts/captures/stage/twohand-audit-labels-6.00.png`.

`tests/gpu/twohand-topology.ts` additionally passes 18 checks covering left-then-right and reverse
cuts, both cuts before analysis, the second cut while readback is pending in either order, and both
hand releases during pending analysis. It checks both detachment flags and the GPU component table.
Both claws deliberately retain weak grip entries on the centre: after detachment those entries
produce zero force and zero stiffness, while the still-held sides retain their grip stiffness.

This rules out missed classification, lingering claw anchors, and omitted/split solver labels for
the reproduced floating centre piece. The two-handed tear changes its shape and contact history;
it does not leave this piece misidentified as attached. Production physics remains unchanged.

```sh
zsh -ic 'node tools/gpu/run.mjs tests/gpu/twohand-topology.ts'
zsh -ic 'ASYNC=1 SUBSTEPS=2 TOPOLOGY_AUDIT=1 AUDIT_NODE=11550 FRAMES=6 UNTIL=6.1 node tools/gpu/run.mjs tools/stage/beneath.ts twohand-audit-labels 1200 750'
```

Validation: lint and typecheck pass; all 67 unit tests and 52 existing GPU checks pass. Their passing
status does not cover the isolated folded-fragment failure demonstrated above.
