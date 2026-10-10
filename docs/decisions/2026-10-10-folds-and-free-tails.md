# Fold depth diagnosis, free thread tails, close-up cost

## Fold depth: no change

Census of a held pull (`session.ts hold`, `DUMP=1`; `tools/studies/folds.ts`). On rings around the grabbed material point it compares three lengths:

- rest circumference;
- 3D chord length;
- length projected on the ring's mean plane.

Results:

- **Gathering goes into buckling.** The in-plane share of the gathering is about 0 beyond 1 cm from the grip; the cloth does not squeeze like rubber.
- **Wrinkles are physical.** About 6 radial wrinkles, 2–5 cm wavelength, which matches tension-field (Cerda–Mahadevan) scaling for B ≈ 1.2e-4 N·m. Amplitude follows from the gathering: A ≈ (λ/π)·√Δ.
- **Resolution has converged.** 6 × 4 bundles (2.3 mm cells) give the same wrinkle count and RMS as 9 × 6, within 5%.
- **The cause is the loading.** Dragging along the warp, the mounted coupon (stretched 0.2% along the warp, the stiff decrimped direction) moves 4 mm in-plane under a 9.6 cm target and lifts instead. Across the slack weft, the same pull gathers 12–15%.

Fold depth is therefore set by the mount and the pull, not by the mechanics. Deeper folds would need a slacker mount, which is a scene decision.

## Free tails (`thread_tail.wgsl`)

A run held at one end, beyond the last span its chain holds, used to droop procedurally to −z within 1.5 mm, with no contact. A yarn's real bending length is far longer than that.

- **Stiffness.** Freed and untwisted, its fibres slide, so EI ≈ fibre count × fibre EI. A collapsed cotton ribbon of 20 × 7 µm with E = 8 GPa gives 4.6e-12 N·m² per fibre, so about 2e-9 N·m² for a warp. Bending length (EI/w)^(1/3) ≈ 1.4 cm: tails of a few mm stick out with slight droop, and only cm tails hang.
- **Shape.** Solved each frame as a clamped heavy cantilever.
  - Relaxed fixed point on the moment of the weight beyond each node, plus a set curl of 5–20 mm radius.
  - The load ramp scales with length over bending length; iteration stops once converged.
  - No state, so nothing to remap when runs are rebuilt after a rupture.
- **Contact.** Against the solver's self-contact face hash (`SelfContact.cells` / `items`, rebuilt every substep).
  - The faces within the tail's length of its root are gathered once per tail.
  - A segment that would cross a face or come within the yarn radius + 0.5 mm ends on it, on the side the tail came from, and the rest of the shape turns with it.
  - Faces within two cells of the root are skipped for the first 3 mm (the cloth the tail leaves).
- **Pipeline.** `threadPrep` stores each tail's root, direction and root node (`RUN_GEO` 9). The separate `threadTail` pass solves it, kept under the 10 storage-buffer limit. `runCenter` samples the solved points, so the thread, its fibres and its shadow all follow one shape.
- **Free tips.** They no longer droop procedurally; they only splay.
- **Stills.** `solver.encodeContactRefresh` builds the face hash for loaded states, so still renders show contact.

Measured with `tools/studies/tails.ts`:

| State | Free tails | Through cloth, old droop | Through cloth, solved |
|---|---|---|---|
| `s6-maul` | 953 | 5 | 0 |
| `s1-peel` | 224 | 3 | 0 |
| fresh `maul` | 1214 | 1 | 0 |

Cost is about 0.26 ms for 953 tails (about 0.2 ms of it is contact).

## Close-up cost: left as is

At max zoom on heavily torn cloth, the edge fibre costs 12.3 ms:

- vertex work for all strands: 2.2 ms;
- fragments: ~10 ms, of which the per-fibre loop is ~7 ms.

Rejected:

- **Explicit strands for resolvable fibres.** About 160 per side against 12 strands per crossing. That is an estimated 8–9 ms of vertex work with per-vertex shadowed lighting, or ~3 ms plus a compute lighting pass, against the ~7 ms it would replace.
- **Culling off-screen crossings.** Measured no gain.

The cost is the on-screen fringe fragments: the price of drawing the counted fibre population up close.
