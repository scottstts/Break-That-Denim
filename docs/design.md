# Fabric tearing: implementation design

Status: revised after independent technical review; feasibility gates have not yet been run. Numerical budgets below are proposed acceptance criteria, not measured performance. Research notes and source limitations are in [the source map](research/source-map.md); visual observations are in [the reference analysis](research/reference-analysis.md). The [review decision record](decisions/2026-10-09-design-review.md) explains accepted criticisms, qualifications, and retained decisions. This document is the current implementation specification.

## 1. Product and scope

Create a close-up, directly manipulable finite fabric coupon with three independently deformable, tearable layers: outer blue denim, a thicker tan brushed woven backing, and another blue denim sheet behind it. This is the authored scene decision for reproducing the observed depth, not a claim that the video proves three separate sheets. Pulling produces deformation, stress redistribution, yarn slippage, progressive rupture, frayed boundaries, hanging threads, and secondary motion. The target is the visual quality of `refs/demo.mov`, including in motion and at the closest supported camera distance.

The initial platform is desktop Chrome on a modern Apple silicon Mac. All interaction and simulation execute locally. The page is the fabric experience: a canvas, an unobtrusive reset control, appropriate pointer states, and essential loading/error feedback. Developer views are separate from the product.

New damage is caused by mechanical loading. The initial material contains an extensive authored worn region, partially abraded yarns, and explicit existing defects. There is no elapsed-time erosion, cursor proximity damage, or prescribed crack animation. Stored elastic energy may continue a tear after release. All visible cloth is eligible to tear under sufficient loading; the worn region determines likely initiation, not a hidden permission mask. Only the explicitly modeled mounting collar is outside the interactive material domain and camera range.

This is a specialized textile simulation and rendering system. The willingness to undertake Houdini-scale engineering applies to solving this problem properly; it does not imply building a node editor, general DCC application, arbitrary garment authoring system, or general multiphysics platform.

### Quality requirements that determine the architecture

- Surviving yarns across an opening transmit forces. Cutting their connectivity changes the motion of both sides.
- A hole is absent material in mechanics, collision, picking, shadows, and rendering.
- Yarn orientation, visible weave, material strength, fraying direction, and friction all refer to the same underlying textile.
- All three layers can slide, separate, touch, fold, and tear independently. Exposed backing is never a texture painted into a hole.
- Contact must preserve layer and yarn ordering through tight folds and fast pulls.
- Fine detail remains attached to its material through deformation, fracture, refinement, and camera motion.
- Visible geometry comes from accepted simulation states or a separately collision-validated interpolation between them. A responsive cursor cannot conceal seconds of queued physical input. Normal operation preserves wall-time gravity and recoil.

## 2. What can and cannot be inferred from the reference

The reference shows dense woven surfaces, irregular expanding openings, pale bridging threads, darker retained strips, fuzzy broken edges, soft directional illumination, and tan material at real depth. Native frames around 3.3–3.9 seconds also show blue material behind an opening in the tan surface. Its connectivity is not recoverable from the clip: another sheet and a returning fold are both possible. Three separate sheets provide a controllable initial reconstruction without inventing an unseen fold connection. Local yarn motion differs from broad cloth motion, justifying multiscale mechanics and appearance.

The clip does not establish its solver, actual textile construction, material dimensions, force history, render time, or whether its damage is mechanically generated. In particular, progressive disappearance in the reference does not prove a physically simulated fracture law. Its image quality is the benchmark; its unknown authoring technique is not an architectural assumption.

Use warp-faced denim with surface-dyed blue warp and pale weft as the blue-material hypothesis. The tan surface retains directional structure but has a thicker, denser fuzzy edge; use brushed woven backing initially. Do not infer a nonwoven felt constitutive law from edge appearance alone. The exact weave, material scale, boundary conditions, and mechanical parameters require calibration. Material-specific edge construction replaces the previous uniform thin-edge rule.

Use three comparison categories:

1. **Appearance:** matched framing, weave scale, edge detail, colors, shadows, and layer separation.
2. **Behavior:** folds, tension paths, pullout, fracture progression, recoil, and settling under repeatable loads.
3. **Robustness:** loads and viewpoints beyond the reference, including folded layers, opposing pulls, and near-total destruction.

## 3. Selected architecture

Use a hybrid, physically coupled representation:

| Region or feature | Mechanical representation | Visible representation |
| --- | --- | --- |
| Intact fabric away from damage | Fixed-resolution anisotropic thin shell | Geometric yarn reconstruction with footprint-filtered fiber response |
| Material approaching failure or significant yarn slip | Explicit woven yarn patch replacing shell material | Explicit yarn cores, plies, and selected fibers |
| Loose load-bearing strands | Extensible elastic rods with bending and contact | Curved yarn geometry and fiber detail |
| Small attached fuzz | No independent load-bearing degrees of freedom | Material-attached procedural fibers |
| Large detached threads and scraps | Dynamic rods or shell/yarn components | The same geometry and material as before detachment |

The shell is a homogenized model of the yarn material, not a separate stretchy sheet underneath it. In a promoted region, shell mass and elastic energy are removed as yarn mass and energy are introduced. Surviving yarn connectivity is authoritative there. Microfibers enrich appearance; they do not invent bridges or carry unaccounted forces.

Use a coupled implicit solve for shells, rods, grips, and contact. G1 compares two production candidates before committing: (A) incremental-potential contact with projected Newton, yarn-aware PCG, and CCD-filtered line search; (B) augmented vertex/block descent with Offset Geometric Contact (OGC) displacement bounds. Build a native double-precision reference and bounded WebGPU experiments, then retain one production solver. OGC is a contact method, not a replacement time integrator; AVBD is not automatically compatible with every contact model.

The branches share material definitions, fixtures, geometry kernels where applicable, and observables. Contact potentials, friction treatment, feasibility enforcement, and solver state differ and must be implemented explicitly. Matching residual numbers alone does not make their behavior equivalent. Neither browser performance nor coupling correctness is established by published native results.

The first integrated mechanical specimen has a **fixed explicit yarn region containing the worn area and an intact collar**, surrounded by a fixed shell mesh. Prove fracture, pullout, and appearance there before attempting dynamic promotion. Dynamic expansion remains required for the finished experience unless fully explicit simulation of the complete finite coupon proves cheaper. The fixed region is an implementation stage, not the final limit on where the user may tear.

### Why not simpler alternatives?

| Alternative | Decision |
| --- | --- |
| Textured spring grid with edges removed at a stretch threshold | Useful only as a disposable control experiment. Mesh orientation, solver iterations, and spring spacing would otherwise dictate tearing. It does not model pullout or surviving yarn mechanics. |
| Alpha-mask erosion plus decorative hair | Cannot satisfy mechanically driven damage or coherent picking/contact. |
| Full fiber-level dynamics | Unnecessary for the visible load paths and prohibitive in degree count. Fibers belong primarily to appearance. |
| Full yarn-level dynamics across all three complete sheets | Use small coupons as the reference and benchmark the complete asset early. Prefer this simpler production representation if its measured cost passes; otherwise require expansion-capable hybrid mechanics. |
| Shell fracture followed by randomly spawned strands | Risks disconnected force paths and artificial energy/mass creation. The material's yarn identities must exist before fracture. |
| Permanent fine geometry attached to a coarse unbroken collision sheet | Blocks holes and prevents true layer interaction. |
| XPBD as an unquestioned final solver | Small-step XPBD is a credible comparison, but approximate convergence must not become material softness or missed contact. Its suitability must be demonstrated on the same coupons. |
| CUDA, native ray tracing, or server-side simulation | Outside the chosen browser/Apple silicon execution model. Native implementations are research references, not runtime dependencies. |

Published yarn tearing, hybrid cloth, homogenization, and codimensional-contact methods support components of this direction. They do **not** establish the performance or correctness of this combination. See [sources R1–R7 and R18–R21](research/source-map.md).

## 4. Feasibility gates precede product construction

The following are implementation gates, not a sequence of increasingly elaborate demos. Each produces reusable kernels, fixtures, measurements, and a decision recorded beside its results.

| Gate | Required experiment | Exit condition | If it fails |
| --- | --- | --- | --- |
| G0: reference and scale | Static torn three-layer coupon, including blue-through-tan depth, denim abrasion, and thick tan frays | Weave, dye, edge construction, depth, and shadowing match at the target pixel budget, including moving camera/no-history views | Fix material, representation, or lighting before simulation |
| G1: browser mechanics/contact and capacity | Race both solvers and both crossing representations on shared fixtures; single-yarn rupture; synthetic full-asset counts and memory peaks on the actual Mac | Correct stiffness/contact, feasible free-end transition, and a credible measured route to wall-time motion at full supported capacity | Revise numerical method, representation, or scene construction now; do not defer the capacity contradiction to G6 |
| G2: material/interface equivalence | Full-yarn coupon, homogenized shell, then fixed mixed patch | Force, bending, rigid motion, transient response, and rendered intact appearance match across the interface | Correct homogenization/coupling; no texture or fuzz seam concealment |
| G4: physical fracture and reference reachability | Fixed explicit worn patch: notches, pullout, surviving bridges, compressed cuts, and several plausible grip paths | Resolution-stable fracture work; feasible free ends; user pulls reach cellular openings, pale bridges, and sequential layer exposure like the reference | Revise worn material, fracture/contact law, or grips; a static damaged render alone cannot pass |
| G3: representation change | After G4, expand into intact/stressed/bent/contacting material and merge patches | No pop, impulse, duplication, lost contact, artificial pinning, or visible boundary; full visible domain reachable | Improve conversion or select full-yarn mechanics; a fixed-region build cannot be released with unrestricted tearing |
| G5: complete local interaction | Three-layer tear with picking, grips, shadows, fibers, and certified temporal presentation | Reference quality, natural secondary motion, and feedback latency pass together | Diagnose by ablation; no feature completion claim |
| G6: full supported scene | Several damage regions, severe folds, detached pieces, near-total promotion/destruction, long sessions | Capacity, latency, recovery, and visual gates pass over the supported operating envelope | Revise the representation or the explicitly agreed scope; do not silently cap visible damage |

Gate IDs remain stable for evidence links; dependency order is **G0/G1 → G2 → G4 → G3 → G5 → G6**. G1 is a bounded comparison, not two complete product engines. First compare solver/contact branches on identical Cartesian-node coupons; separately compare general-contact rods and sliding-crossing rods. Test the winning combination together before selecting it. A speedup bought with penetration, excess stretch, over-damped recoil, or changed fracture work fails.

G0 and G1 inform each other: projected yarn scale fixes counts, and measured costs constrain feasible discretizations. G1 includes full-asset allocation and throughput stress cases even before complete fracture exists. Its result is an early feasibility decision, not a substitute for G6's actual destroyed scene. The single-yarn rupture test deliberately precedes the hybrid engine.

## 5. Physical scale, coordinates, and material identity

Author in SI units: meters, kilograms, seconds, newtons, and joules. Solver nondimensionalization is explicit and reversible. Render-world units never become implicit material parameters.

Each layer has a rest-material chart `(u, v)` measured in meters, a warp direction, a weft direction, and stable identifiers. The current shell embedding maps this material chart into world space. Yarn arc length is a separate material coordinate; sliding a yarn through the weave must not make its dye pattern remain fixed in the cloth chart.

Stable identity hierarchy:

```text
SceneSeed
  LayerId
    MaterialTileId
    YarnId (family + index)
      MaterialArcInterval
      FailureSiteId
      FiberSeed
    ShellElementId + generation
    ComponentId + topologyEpoch
```

Use immutable material identities and generation-checked storage handles. Buffer offsets are not identities. A triangle index can change after refinement without changing the material being grabbed. A split yarn creates child intervals with explicit parent provenance.

### Initial calibration envelope

These ranges establish useful experiments, not final material values:

| Quantity | Starting exploration range |
| --- | --- |
| Initial finite coupon, each of three layers | Provisional 0.10×0.125 m, plus a narrow modeled mounting collar; freeze dimensions after G0 |
| Supported framing | Within the coupon; approximately 0.06–0.10 m horizontal field for initial experiments |
| Warp/weft pitch | Approximately 0.3–0.7 mm; fit both directions separately |
| Yarn core diameter | Approximately 0.1–0.35 mm, consistent with the weave geometry |
| Cloth apparent thickness | Blue approximately 0.3–0.9 mm; tan may be thicker; fit separately and distinguish fuzz from core thickness |
| Intact shell edge length | Fixed 1–2 mm starting mesh, checked with a finer offline convergence mesh |
| Yarn activation tile | Initially 8×8 or 16×16 weave crossings; benchmark both |
| Primary grip radius | Approximately 3–6 mm in material space |

Select concrete scale, physical extent, camera range, and projected weave density together during G0/G2. The smaller provisional coupon replaces the previously unexamined large-sheet assumption. It is not permission to crop away reference content or enlarge yarns to reduce counts. If matching the image needs a larger coupon or finer pitch, its larger capacity goes back through G1 before further construction.

## 6. Material model and initial scene

### Material definition

A versioned material asset contains:

- Weave repeat and over/under order, warp/weft pitch, crimp geometry, yarn radii and linear densities.
- Axial stress/strain curves, bending and optional torsional response, unloading response, and damping.
- Yarn/yarn, cloth/cloth, and cross-layer friction coefficients, with a documented regularization velocity.
- Tensile initiation strength, fracture work per yarn, physical failure-site spacing, and correlated variation.
- A homogenized shell energy fitted to the same intact weave.
- Dye distribution, fiber/ply distributions, optical roughness, scattering parameters, and wear fields.

Keep optical radius, compressible outer contact radius, and dense mechanical core radius distinct but related. A fluffy halo does not require a correspondingly huge hard collision envelope.

### Material-specific construction

**Blue denim:** start with warp-faced 3/1 twill, indigo concentrated in the outer warp fibers and a pale core, plus predominantly undyed weft. Store dye penetration in yarn-local radial/material coordinates so sliding or rupture reveals the correct interior. Abrasion depends on exposed yarn surface, family, float length, and local wear history; it removes dye/fibers and changes effective cross-section consistently. Warp-biased surface exposure should produce preferential warp damage without declaring all weft immune. Pale bridges can include surviving weft and exposed warp cores; color alone is not a yarn-family label. Traditional denim construction supports this hypothesis [R22](research/source-map.md), but the reference does not identify a manufactured textile specification.

**Tan backing:** start with a denser brushed woven material, a greater apparent thickness, and longer/dense attached fibrillation. Keep a woven load-bearing skeleton until evidence requires another constitutive family. Its broken boundary may read as a continuous fuzzy band where closely spaced ends overlap; generate that band from surviving material and fiber coverage, not an arbitrary solid wall closing gaps. A bounded aggregate edge representation is allowed only if it preserves actual coverage, light transmission, silhouette, and contact clearance. G0 evaluates it separately from denim. If a felt/nonwoven skeleton is needed to match mechanics as well as appearance, that requires a separate material decision and fixtures.

### Wear

Generate wear in rest coordinates with fixed seeds and physical correlation lengths. Separate broad abrasion, local yarn strength, dye loss, and initial connectivity defects, while allowing them to share causes. For example, abrasion can expose pale fibers and reduce remaining cross-section. It does not automatically imply that every pale texel is a hole.

Use bounded distributions with an intact-strength floor outside deliberately authored defects. Avoid independent random thresholds per solver edge: refinement would then create a different material. Pre-existing broken yarn intervals and missing fragments are explicit initial topology.

Author an extensive worn zone at reference-relative scale: correlated warp abrasion, varied surviving weft, several incipient openings, and retained strong ligaments. Pre-existing broken intervals are present in the first mechanical snapshot; pre-damaged/slipping tiles therefore start explicit. Outside the zone, strength returns smoothly to the material baseline. Do not manufacture the whole final image at initialization: G4 must show additional mechanically generated openings and bridges under several plausible pulls, including pulls outside the most damaged region. Record initial versus newly broken material separately. Future crack paths and tear times are not stored.

### Three-layer setup

All layers extend beyond the supported visible crop and are mounted at explicit boundary regions. The mounts provide reaction forces; the cloth must not drift away as one rigid object when grabbed. Use distributed, calibrated boundary constraints instead of pinning isolated corner particles into star-shaped stress concentrations. The rear blue layer has its own material chart, wear, holes, and boundary conditions; it is not a copy of the front layer's moving geometry.

Prepare the drape offline or during asset construction by settling all three layers with gravity, mounting constraints, and friction. Store a validated initial snapshot. Interlayer gaps are fitted to the reference shadows and remain contact-feasible. The layers are not glued together unless a later scene explicitly introduces stitches.

The space behind the rear blue layer is dark and has real depth. Opening the tan layer can reveal blue before opening the rear sheet reveals the void. Do not add an invisible backing collider over the tearable region. Mounts and any scene support geometry are included in contact and shadow validation.

The resting scene must remain intact without user input. Any pre-tension is below the worn material's equilibrium failure threshold. Initial settling is completed before the user can grab it.

## 7. Intact shell mechanics

Use a triangular thin-shell discretization with material-space rest matrices, consistent area weights, surface mass density, and an anisotropic membrane energy. Derive the deformation gradient from the current triangle and its rest chart.

The energy responds independently to warp stretch, weft stretch, and shear, including coupling terms needed to match the yarn coupon. It must be invariant to world-space rigid motion. Compression permits buckling; it must not make the fabric behave like a volumetric rubber slab.

Fit the energy and its derivatives to periodic yarn-cell experiments: warp extension, weft extension, bias extension, biaxial loading, and unloading. Use HYLC [R18](research/source-map.md) as the homogenization starting point, not as evidence that a reversible fitted shell represents frictional hysteresis or damage. Begin with a smooth constitutive fit over a declared strain domain. Do not extrapolate an unconstrained table into unphysical response. Move into explicit yarn mechanics before irreversible slip or damage exceeds the shell model's valid domain.

Model bending independently from membrane extension using a consistent discrete-shell bending energy, calibrated against the woven coupon. Bending stiffness is not inferred from a solid plate's thickness-cubed law without validation; yarn rearrangement changes the effective response.

Plastic history, if needed to match residual creases, belongs in material space and must survive refinement. Do not add plasticity merely to mask numerical damping or insufficient convergence.

### Fixed baseline mesh

Compile a conforming shell mesh aligned to material-tile boundaries. Start with 1–2 mm spacing and compare a finer fixed mesh on sharp folds, contact, and grip support. Select resolution from force, curvature, and contact convergence, not a nominal vertex budget. Mesh diagonals are numerical choices, not fracture directions. Precompute boundary stencils and shell terms owned by each tile for replacement.

Runtime shell refinement/coarsening is deferred. Introduce it only if a converged fixed mesh cannot satisfy the measured budget; that decision requires conservative mass/history transfer and conformity tests. Offscreen material still transmits forces and receives the same mechanically justified resolution.

## 8. Explicit yarn mechanics and shell coupling

### Yarn model

Represent each activated yarn by an extensible rod centerline with rest arc lengths, axial response, bending, and a transported material frame. Include twist degrees of freedom only where their mechanical contribution is demonstrated; a visual ply twist is not by itself evidence that every rod needs a dynamic torsion variable.

Use physical yarn paths with crimp and the correct weave ordering. Do not weld warp and weft into shared no-slip particles. G1 compares Lagrangian rods with general contact against a sliding-crossing formulation based on [R2](research/source-map.md), on equal physical material and pullout fixtures. A sliding crossing must allow material transport, retain friction and changing arc lengths, and release when support disappears. Crossing proximity does not establish permanent textile connectivity.

Measure a mixed strategy in which intact supported crossings use the specialized formulation and crossings near fracture, unsupported loops, or a departing endpoint convert to general contact. General collision remains necessary for non-neighbor yarns, other layers, and folds. Before enabling a switch, demonstrate mass/momentum transfer, consistent contact/friction work, feasible geometry, and unchanged pullout resistance. If the switch fails these tests, keep general contact despite its cost or revise the representation. Crossing coordinates can change the mass operator and solver blocks; do not assume the Cartesian solver transfers unchanged.

Compile sufficient rod subdivision and physical cohesive sites for curvature/contact convergence in the supported scene. Runtime rod refinement is optional later work, not a baseline dependency. Rendering uses smoother curves with a measured deviation from collision centerlines. Long exposed threads remain mechanical even when visually subpixel.

### Domain replacement

The baseline hybrid method uses **disjoint material domains with a coupled boundary**, not two overlapping complete simulations. Every tile's mass and elastic energy have exactly one owner. Tile states are:

```text
SHELL → PROMOTION_PENDING → YARN
```

Promotion is irreversible for damaged or slipped material. Intact demotion is deferred until a separate reversible-transfer test exists. It is an optimization, not a prerequisite for correctness.

At an interface, use an intact transition collar with **constant linear constraints in physical Cartesian coordinates**. Normal-offset attachments are excluded from the baseline: they introduce curved CCD trajectories and extra inertia derivatives. However, projecting every crimped yarn endpoint onto the shell midsurface is also invalid; it flattens the weave and can alter bending and contact order.

The coupling experiment uses material-weighted averages over complete weave repeats in separated collar windows. Match their neutral-surface positions to shell barycentric samples: `A*x_yarn = B*x_shell`. Select constant weights with partition of unity and exact compatibility in the rest state. This makes the constraint invariant under common rigid translation/rotation without flattening each yarn. Use separated windows for each continuing yarn family so force pairs transmit bending moments; prohibit rank-deficient or redundant constraint rows. Window width and number are calibrated by G2, including transverse bending and shear. A single averaged position per continuation cannot pass moment-transfer tests.

Keep mass/elastic ownership disjoint through the collar; averaging constraints add neither a second material sheet nor penalty-spring energy. Assemble reciprocal reaction forces and constrained inertia. For the Cartesian reference, eliminate constraints into a sparse constant map `x = P*q` where practical, giving `M_q = Pᵀ*M*P`, or solve the equivalent constraint blocks exactly. Do not replace this generally non-diagonal mass with a diagonal approximation silently. Linear maps keep Cartesian search paths linear. Dense elimination that destroys locality is rejected on cost; coupled block solvers must enforce the same constraints.

For a selected sliding-crossing formulation, re-derive its kinematics, inertia, and admissible motion bounds rather than importing the Cartesian map by assumption. A nonlinear path must either be bounded conservatively or converted to Cartesian free nodes before using linear CCD. G1/G2 include this cost in the representation decision.

This averaged collar is an explicit **project hypothesis**, not a theorem furnished by the hybrid-cloth papers. Compare against full-yarn coupons for force, moments, rotation, transient reflection, and rendered crimp continuity. If it fails, broaden the explicit domain or use full-yarn mechanics; do not mask a clamp or assert that two anchors alone solve it. Dynamic conversion waits until the fixed collar passes.

### Activation criteria

Promote before a region can fail or exhibit unresolved slip. Trigger on predicted tensile utilization, grip proximity, neighboring active damage, and departure from the shell's calibrated strain/history domain. Initially investigate promotion near 50–70% of local initiation load; tune using error and lead-time measurements rather than visual preference.

Add a material-space collar beyond the predicted process zone and pullout distance for the next accepted step. Promotion is not a camera-based detail switch. In the fixed-patch G4 harness, approaching the boundary stops and labels the replay as outside that fixture's domain; enlarge the patch and rerun. It must not silently clamp a strand or report a finished product. In the product, expand before accepting a tear/slip that reaches the collar. If expansion cannot complete within the interaction budget, pause from the last valid state and fail the gate; never break shell elements and manufacture decorative yarns afterward.

### Promotion transaction

1. Reserve complete capacity for the new patch, its collar, contacts, render mappings, and rollback state.
2. Identify the exact material area and yarn intervals being transferred. Adjacent patches share existing yarn IDs; never create duplicate yarns on tile boundaries.
3. Activate precompiled deterministic rest-weave records. Initialize current shape from shell deformation plus a strain-indexed equilibrium crimp response [R19](research/source-map.md), then locally equilibrate with the shell boundary fixed. That lookup covers intact states only, not already slipped or damaged yarns.
4. Transfer velocity using a mass-weighted fit preserving total linear and angular momentum. Transfer compatible history; promote early enough that unrepresented irreversible yarn history does not already exist.
5. Remove the corresponding shell mass/energy and build coupled boundary operators. Record residual elastic-energy mismatch separately from physical damage dissipation.
6. Validate contact against every other layer and existing yarns, mass/length ownership, force continuity, and transfer tolerances. New geometry must be feasible before exposure to the runtime solver.
7. Publish the new topology and state atomically; otherwise retain the previous accepted state and revise the transaction.

A local equilibration is not permission to move visible material until a bad fit disappears. Transfer displacement and energy limits are in section 18. If contact prevents conversion, enlarge the patch, refine the source shell, or retry earlier in simulation time. No opaque one-frame crossfade conceals a mechanical discontinuity.

## 9. Progressive damage, fracture, and pullout

### Distinct mechanisms

- **Elastic deformation:** reversible yarn extension, decrimping, and cloth bending.
- **Sliding/pullout:** yarn material moves through crossings against friction. A yarn can emerge from a boundary without snapping.
- **Cohesive failure:** a finite amount of tensile work exhausts a yarn's load-bearing connection.
- **Fraying:** exposed yarns and fibers remain after adjacent support disappears; they inherit material identity and motion.

These mechanisms cannot all be represented by deleting a spring above one strain threshold.

### Failure law

Use irreversible tensile cohesive failure at material-defined potential failure sites along each yarn. The spacing and strength field are fixed in physical arc length, independently of rod subdivision. Additional quadrature or refinement does not create additional weak links.

For a yarn with effective cross-section `A_y`, define peak tensile force `T_peak` and fracture work `Gamma_y` in joules. A normalized softening law must satisfy:

```text
integral(T(opening), opening = 0 .. complete failure) = Gamma_y
Gamma_y = G_y * A_y    # when the material uses fracture energy G_y in J/m²
```

For a triangular cohesive law whose area includes its loading branch, the final opening is `2 * Gamma_y / T_peak`. If elastic loading is accounted for in another energy term, derive the remaining softening area accordingly; never count the same stored energy twice.

Track maximum tensile opening and irreversible damage `d ∈ [0,1]`. Unloading follows the damaged stiffness. Compression does not heal damage or produce tensile fracture. Begin without an arbitrary time-rate damage term; introduce rate dependence only if calibrated evidence requires it.

During each step, freeze damage for the mechanical solve, update trial history, and repeat to consistent force/damage state. If changes are too large or localization depends on the iteration order, reduce the timestep and resolve. Simultaneous independent failures may batch; failures coupled through the same load path require stress redistribution before further decisions.

Fracture occurs only in explicit yarn regions. The shell cannot make a final rupture decision from coarse strain and then ask the renderer to improvise its topology.

### Topology and energy at rupture

Rupture removes tensile/bending connectivity only after cohesive traction has fallen to zero. Remaining elastic energy, kinetic energy, dissipated fracture work, friction, and external grip work are logged. Do not add an artistic recoil impulse: recoil comes from released stored energy.

Model each failure site with a finite, geometrically resolved process segment. Its end cross-sections and mass fractions exist before full separation. This avoids creating two coincident, full-radius collision endpoints by duplicating a vertex at rupture. The residual fibers of this process segment become attached fray or free fragments according to the material asset; their mass and rest length stay in the material ledger.

The exact contact geometry of the weakening neck is prototyped on a single yarn in G1 and fully qualified in G4. It must provide a feasible transition from connected material to two finite-thickness free ends, including cuts under compression. If the ends cannot yet separate, they may have zero tensile traction while remaining in compressive contact. Do not push them apart, temporarily disable arbitrary contacts, or drop their thickness to zero to get past a solver error.

Connectivity updates invalidate affected bending elements, contact-neighbor classifications, component labels, boundary mappings, pick ownership, and temporal history. A hole opens because no surviving yarn spans that material region, not because a separate erosion texture has crossed a threshold.

### Pullout and free strands

Use the crossing/contact representation selected in G1. Track remaining embedded arc length and which crossings still support the yarn. Expand the explicit collar before a slipping material interval reaches a shell attachment; fixed anchors must not become artificial knots.

After a yarn exits a crossing, remove the crossing relationship while retaining ordinary collision. A released strand can collide with all cloth layers and other strands. Its endpoints are physical endpoints; drawing an interpolated line between unrelated torn edges is forbidden.

Tiny fuzz may be render-only when its inertia and transmitted forces are negligible. A bridge that visibly pulls two regions together, a long hanging thread, or a loop that can snag is mechanical. This classification is based on influence, not just screen size.

## 10. Contact and friction

Use one collision system for shell/shell, rod/shell, rod/rod, cross-layer, and support interactions. Necessary primitive cases include vertex/face and edge/edge for shells, segment/segment for rods, and rod/shell interior crossing cases. Rod endpoints alone are insufficient: a segment can pass through a face while both endpoints remain outside.

### Geometry and thickness

Maintain a collision mid-surface or midline plus material-dependent contact envelopes. Dense cores prevent crossing; a compressible outer range models the softer contact of a fibrous material. Render offsets must remain consistent with these envelopes at visible contact.

Fine discretization can create artificial repulsion between nearby parts of the same continuous material. For the IPC branch, investigate intrinsic-neighbor thickness filtering [R7](research/source-map.md) instead of broad neighbor exclusions. For OGC, use its own offset geometry and feasibility construction. Do not transplant one method's thickness/filtering rules into the other's proof. Both need explicit fracture-time reclassification tests, including new endpoints, concave folds, and tightly packed yarns. A suitable fracture-compatible thickness treatment is a G1/G4 decision, not already established by either paper.

Maintain distinct tests for membrane locking and contact locking. Increasing mesh resolution can help the former while worsening the latter under an incorrect thickness model.

### Broad phase

Use GPU-built/refitted bounding-volume hierarchies over shell faces/edges and rod segments, with swept bounds for motion and contact thickness. Keep material adjacency filtering separate from spatial proximity. Two points that were neighbors before tearing may no longer be excluded afterward.

Use count/scan/fill construction for candidate buffers, stable sorting where needed, deduplication, and an explicit overflow flag. Refit between topology changes; rebuild when measured tree quality degrades. Do not allocate an all-pairs contact matrix or rely on fixed per-cell occupancy limits.

A candidate set is valid only for its bounding geometry and displacement envelope. Newton search directions can leave the original predicted bounds. Refresh candidates or use conservative motion bounds before line search; discovering only endpoint proximity is insufficient.

### Narrow phase and acceptance

Every accepted displacement requires a conservative feasibility bound, including predictors, solver updates, initialization, and topology-related geometry changes. Degenerate triangles, nearly parallel edges, and tiny separations are normal test cases.

**IPC candidate:** start with Additive CCD (ACCD) from C-IPC for finite-thickness linear Cartesian trajectories. It provides the line-search feasibility bound before descent testing. Include point/triangle, edge/edge, and lower-dimensional degeneracies. Shell/rod mappings must provide conservative trajectories for the physical primitives; checking only the endpoints of a nonlinear crossing-coordinate path is invalid. The linear collar avoids normal-offset trajectories, but a selected specialized crossing model still needs its own proof or conservative subdivision.

**OGC candidate:** maintain per-vertex displacement bounds relative to the configuration used to construct the current feasible region. Restrict every update, including every block/color pass, by its cumulative displacement from that configuration. Refresh bounds/candidates before exhausting their validity and after topology changes. Restrict dependent physical nodes too: clamping independent coordinates alone is insufficient. Feasibility bounds replace repeated global CCD in this branch; they do not remove contact detection, force convergence, friction work, or validation. Test open/concave contact, compressed free ends, and bound-induced artificial damping. Use ACCD as an independent trajectory diagnostic and for presentation paths when needed, not as an unmeasured global pass after every OGC update.

**Finite precision is unresolved implementation work with a specific acceptance contract.** Both methods need lower bounds on separation and upper bounds on motion that account for `f32` error. Kernels must return a separation enclosure or `uncertain`, not an unqualified nearest-distance scalar. Nondimensionalize and center each query; propagate arithmetic/coordinate error through branch predicates, norms, and motion bounds. Widen bounds outward for supported WGSL rounding behavior. The guard is `epsilon_geom = epsilon_input + epsilon_arithmetic + epsilon_reconstruction`; each term must have a documented bound in physical units for that kernel. Advance only from a strictly positive lower-bound core gap, with a conservative fraction of permitted motion, and reject when progress becomes numerically indeterminate.

There is no defensible universal numerical epsilon to choose before the kernels, local coordinate range, and target compiler are known. G1 must produce those bounds, adversarial tests, and measured fallback frequency. A fixed fraction of yarn radius or a smaller line-search factor alone is not a precision proof. Ambiguous cases go to an audited native/WASM double-precision robust query or reject/retry; ordinary double precision alone also does not certify a geometric predicate. Post-step validation supplements conservative motion bounds; it cannot detect all tunneling by itself. If frequent CPU fallbacks miss latency, the GPU query is not ready for production.

Retain a double-buffered accepted/trial state. If contact validation, convergence, or capacity checks fail, the trial does not become visible. Reduce the timestep or resolve the failing patch. Record the cause to prevent silent perpetual retries.

### Friction

Use a regularized Coulomb friction model with persistent material contact identity and lagged normal forces in the nonlinear solve. Refresh friction data until the defined residual tolerance is met. Friction must dissipate work, oppose relative tangential motion, and distinguish sticking from sliding.

Cross-layer friction is calibrated separately from yarn crossings. Do not use global velocity damping to imitate interlayer friction; that would also suppress free-thread motion. Contact history is reset or remapped when its material pair ceases to exist.

## 11. Time integration and nonlinear solution

For fixed topology and damage, use an incremental potential of the form:

```text
Phi(q) = 0.5/h² * ||x(q) - x_pred||²_M
       + E_shell(q) + E_yarn(q) + E_cohesive(q)
       + E_contact(q) + E_grip(q) + D_friction(q)
```

`q` contains independent degrees of freedom; coupled samples use `x(q)`. This equation defines the physical objective, not the floating-point evaluation order. External gravity enters through the predictor or an equivalent potential, not both. Pins eliminate degrees of freedom where possible. Irreversible damage/history belong to an outer update. The inertial norm is over physical material samples; mapped inertia and its derivatives must match the selected coordinates.

### Numerical policy shared by the candidates

- Solve for increments from the accepted state: Cartesian `x = x_n + delta_x`, with inertial residual `delta_x - h*v_n - h²*a_ext`. For linear coupling use `delta_x = P*delta_q` and the mapped mass. Keep increments distinct from reconstructed positions throughout the solve. Evaluate local edge changes from old local edges plus displacement differences, avoiding subtraction of rounded world positions for tiny updates.
- Scale length, mass, force, and energy explicitly; center geometry queries locally. Scaling removes avoidable magnitude loss but cannot eliminate physical stiffness ratios.
- Compute line-search changes as stable **per-term energy differences**, then pairwise/compensated reductions. Use algebraic differences such as `2*a·d + d·d` for squared norms rather than subtracting two large totals. Contact/logarithmic terms need their own stable formulas and domain checks. An energy drop within its computed reduction/error bound is inconclusive; use residual/step criteria to determine convergence or retry. It is not evidence for arbitrary acceptance or perpetual backtracking.
- Define local energies once in a small, versioned expression/code-generation layer that emits C++ `f64` and WGSL `f32` energy/gradient/Hessian kernels. Include the exact branch and regularization policy. Verify derivatives independently with finite differences, rigid-motion identities, and work tests; shared generation can reproduce the same mistake in both backends. Robust geometry predicates receive separate audited implementations where symbolic differentiation is unsuitable.
- Keep original energies for diagnostics/descent even when search-direction Hessians are projected positive definite. Record degeneracy and regularization rather than hiding NaNs. Do not use `f16` mechanics.

### G1 solver comparison

**A: projected Newton/PCG.** Use matrix-free or element-block products with incidence gathers. Per-vertex block Jacobi is only a tiny-fixture baseline. The first serious preconditioner includes yarn-chain banded blocks for axial/bending couplings plus shell/contact-aware overlap or a coarse correction. Factor/solve each chosen chain block accurately, but do not call it an exact global solve: crossings, grips, shell interfaces, and contact couple chains. Measure factorization, update, reduction, and apply cost in WGSL. Compare chain blocks and additive Schwarz/coarse coupling on long taut yarns before extrapolating from a small patch. Use ACCD-filtered descent with the numerical policy above.

**B: augmented block descent with OGC.** Include constraint multipliers/penalty updates, a valid block/color schedule, and bounds on every physical update. Crossing variables and collar constraints may require blocks larger than a Cartesian vertex. Measure convergence to the same force, extension, damping, friction, and fracture-work tolerances as A. A short fixed sweep count that looks stable but changes the material fails. OGC and AVBD need an integration experiment; their papers do not guarantee this combination with our rods and cohesive law.

For scale: with `EA = 50–250 N`, segment length `0.3 mm`, node mass `9e-9 kg`, and `h = 1/240 s`, local axial stiffness divided by inertial stiffness is roughly `3.2e8–1.6e9`. These are stress-test inputs, not measured denim constants or a complete matrix condition number. They demonstrate why more iterations and diagonal scaling alone are not an adequate numerical plan.

Measure dependent dispatch latency, reductions, actual iteration distributions, broad-phase rebuilds, feasibility filters, and friction/history iterations. A hypothetical 8 dispatches × 5 microseconds × 100 PCG iterations is already 4 ms per Newton iteration before bandwidth costs; this is a warning calculation, not a measured Apple GPU timing or proof of inevitable IPC failure. Record residuals and physical error at equal cost for both candidates. Reaching an iteration limit never means convergence. Select one engine only after the combined representation/contact tests pass.

### Timestep policy

Compare `1/60`, `1/120`, `1/240`, and `1/480 s` on identical loads, fracture events, and free decay; start the reference at `1/240 s`. Choose the largest step satisfying temporal/mechanical error targets rather than precommitting to an unaffordable high rate. Smaller features and rapid rupture may require local episodes of smaller synchronized steps.

Adaptive stepping reacts to convergence, contact, fracture increments, and error estimates. It does not stretch `h` to match a delayed animation frame. Implicit stability does not remove temporal error or numerical damping.

Use one simulation clock for all layers and active yarns. Subcycling is deferred until a conservative, synchronized coupling scheme has been validated. Solving yarns several times against a frozen shell and then moving the shell once is not equivalent to two-way coupling.

Physical damping is calibrated against free decay. Numerical damping is measured separately using timestep sweeps. Air drag should depend on relative motion and exposed area; any simplified drag must have non-positive work in still air. No periodic wind or random motion is added to make a static cloth appear alive.

### Accepted-step sequence

1. Consume input into a bounded, continuous grip trajectory.
2. Check promotion needs and complete any required GPU topology transaction.
3. Predict motion; build swept contact candidates.
4. Solve mechanics/contact with trial damage frozen.
5. Update damage and friction history; resolve if required, or retry at smaller `h`.
6. Prepare fully failed connectivity changes and validate their contact geometry.
7. Validate residuals, finite values, material ledger, and collision state.
8. Commit state, topology epoch, history, and simulation time together.
9. Prepare rendering/picking from accepted state buffers; certify any interpolated presentation path separately.

## 12. Runtime ownership and WebGPU implementation

### Stack

- TypeScript and the existing Vite shell for boot, browser integration, and the minimal UI.
- Direct WebGPU and WGSL for compute, rendering, buffer ownership, and synchronization.
- A small C++ core built natively for reference tests and to single-threaded WebAssembly for asset compilation, checkpoint validation, and exceptional geometry queries.
- Reuse audited components from IPC Toolkit where practical for reference contact/CCD. It is a collection of functions, not a complete cloth solver. Its Emscripten build and transitive dependencies require a G1 build spike.

Choose direct WebGPU because simulation, derived geometry, and rendering need to share precisely controlled buffers. Do not introduce a second renderer-owned scene graph that forces copying all positions each frame. A general WebGPU engine can replace this decision only if the buffer and pass ownership contracts remain intact.

### Execution ownership

```text
Main thread
  pointer/keyboard events, resize, accessibility, essential status
             │ small ordered messages
             ▼
Engine worker
  simulation scheduler + native-reference/WASM utilities + GPU device/queue
             │
             ├─ accepted and trial mechanics buffers
             ├─ contact and solver passes
             ├─ render reconstruction and pick pass
             └─ OffscreenCanvas presentation
```

One worker owns the GPU device, queue, simulation, and render graph. One queue establishes ordering. Do not create separate devices for physics and rendering. Check worker canvas/device support at startup; a main-thread engine fallback may preserve functionality, but does not excuse missed latency gates.

Start without WebAssembly threads or SharedArrayBuffer. If profiling later justifies them, deploy COOP/COEP and validate every asset's isolation requirements. They are not assumed to be available merely because WebAssembly works.

### GPU topology, CPU durability

Compile the finite scene's material identities, rest weave, possible cohesive sites/free-end records, tile ownership, and bounded incidence templates before interaction. The GPU owns active topology and numerical state; the CPU receives an asynchronous event journal and checkpoints. Stable identities are preassigned, not invented from nondeterministic allocator order. Ordinary rupture must not require a GPU→CPU→GPU round trip.

Use deterministic per-tile spans or pre-reserved pages, activation masks, prefix sums, and compact active lists. Reserve the complete finite-asset topology capacity in G1's budget; contact candidates still need variable-sized scratch and overflow checks. An unrestricted GPU heap and procedural generation of every rest record are optional optimizations, not prerequisites. Rare capacity growth/recovery may involve CPU work and must pause safely; normal tear onset must fit pre-reserved capacity.

Each GPU transaction stages the base epoch, changed damage/connectivity, affected incidence, contact classifications/bounds, component/grip mappings, render history invalidations, and a conservation summary. Validate all safety-critical invariants and contact geometry **before** publishing accepted state and epoch. On failure, retain the previous state without partial mutation. Independent failures batch; causally coupled failures require a new mechanical redistribution. Breaking a pre-existing cohesive site can change a flag, but that flag is not the complete transaction.

Journal committed events through pooled asynchronous staging with sequence and epoch checks. CPU validation audits and enables recovery; it does not retroactively make an unsafe visible state acceptable. A checkpoint and its journal suffix must describe the same committed epochs. Bound the journal queue; overflow pauses or checkpoints rather than dropping events. Dynamic promotion adds initialization/equilibration/transfer validation to this same transaction protocol and remains an unsolved G3 gate.

### Buffers and numerical operations

- Use structure-of-arrays within a few storage arenas with explicit typed offsets/alignment. A page or field is not a separate binding. Budget each pass against `maxStorageBuffersPerShaderStage` (portable starting budget: eight), binding-size, alignment, and workgroup limits.
- An initial compute binding plan reserves arenas for immutable material/topology, accepted state, trial state, incidence/contact data, solver scratch, and transaction/indirect outputs: six storage bindings, leaving two for pass-specific needs. Large pages are processed in bounded batches with global reductions/halo gathers across batches; never require every page to be bound simultaneously. Validate mixed scalar layout against WGSL and generate matching host offsets.
- Maintain generation-checked page pools for nodes, elements, yarn intervals, contacts, and draw records. Reserve transaction scratch before starting a mutation.
- Use gathers or sorted segmented reductions for force/Hessian accumulation. Portable WGSL cannot be designed around floating-point atomic addition.
- Use integer atomics for flags and counters only where the algorithm tolerates ordering. A workgroup barrier does not synchronize the whole dispatch; global stages use separate dispatches.
- Keep reduction trees and topology-event sorting deterministic within a backend. Cross-device bitwise replay is not promised.
- Keep solver positions, damage, contact, and accumulated work in `f32` on GPU, with scaled units and compensated/structured reductions where needed. Do not introduce `f16` into mechanics for bandwidth savings without a separate accuracy case.
- Query actual features and limits. Each bound arena window fits the maximum storage binding size; total allocations also respect buffer-size and memory budgets. Timestamp queries and subgroup operations are optional accelerations, not correctness dependencies.
- Chunk long iterative work into bounded submissions. Use GPU convergence flags and indirect dispatch where useful; do not wait for a CPU scalar after every PCG iteration.
- Maintain accepted buffers independently of scratch that the next trial will overwrite. Rendering an in-progress nonlinear iterate is invalid.

Buffer mapping uses dedicated staging buffers and an asynchronous pool. A mapped GPU buffer is unavailable for normal GPU use; do not map a live simulation or draw buffer. Retire resources only after their queued use has completed. See [platform sources R13–R16](research/source-map.md).

## 13. Rendering architecture

### One material, several sampling scales

Visible geometry derives from the same layer, yarn, arc-length, and damage identities used by mechanics. The three appearance scales are:

1. Shell-scale shape and folds.
2. Yarn/ply-scale structure, gaps, cross-sections, and frayed boundaries.
3. Fiber-scale scattering and sparse explicit fibers.

Do not overlay unrelated noise textures to approximate these scales. Dye loss, exposed pale fibers, roughness, thickness, and edge fuzz should respond to coherent material causes.

### Intact regions

Use one geometric yarn reconstruction pipeline for intact shell-driven and explicitly simulated regions. Surface normals, warp/weft tangents, coverage, and scattering come from the same material definition. Do not build a second whole-cloth BRDF/geometry transition system unless measured geometry cost requires it. Keep footprint filtering, curve tessellation LOD, and unresolved-fiber aggregation: geometric yarns still become subpixel on foreshortened folds. A 0.45 mm pitch in a 0.10 m field spanning 1,100 pixels is about 5 pixels face-on but less than one pixel at 80° foreshortening.

Shell-driven yarns use a deformation-aware crimp reconstruction calibrated from the same unit-cell data as mechanics; mechanics-aware pattern deformation [R19](research/source-map.md) is the starting research method. Simple affine stretching of a rest weave stretches yarns when real cloth may instead decrimp or shear. Explicit regions use actual yarn positions and material arc lengths. The lookup is not used to erase damage or frictional slip.

G2 includes matched renders of an intact full-yarn coupon, shell reconstruction, and a fixed mixed patch under tension, shear, and bending. Measure crimp shape, silhouette, coverage, dye placement, shading, and shadows through the collar. Shared code alone does not ensure a seam-free image when its mechanical inputs differ. Any later aggregate appearance LOD needs coverage/scattering equivalence and temporal validation before adoption. Camera motion never changes mechanics.

Assign one primary visibility owner to each material region. In an explicit yarn domain, no underlying opaque shell is drawn or picked. Any aggregate distant appearance there is derived from surviving-yarn coverage and cannot close a physical hole. At a shell/yarn boundary, render reconstruction shares boundary samples and material charts; it must not leave a tiled gap or double-covered seam.

### Exposed yarns and frays

Reconstruct curves from accepted rod samples with bounded overshoot. Compute camera and shadow LOD separately from the same centerlines. Use compute-generated tube/ribbon geometry and indirect draws; do not depend on geometry or tessellation shader stages unavailable to the chosen WebGPU pipeline.

Include the curve-to-polyline deviation in conservative collision bounds, or refine rod samples until it is negligible relative to the available contact clearance. Smooth interpolation cannot be permitted to make a rendered yarn cross the backing while its coarser collision segments remain clear.

Resolve tube silhouettes when their width is visible. Use analytic footprint coverage for narrow opaque fibers, not a minimum one-pixel opaque width. Procedural ply/flyaway statistics follow the same material asset; fitting ideas from [R20](research/source-map.md) guide authoring but cannot recover microscopic parameters from the video.

Attached fibers inherit parent yarn deformation, sliding arc length, and transported frames. Fray length, curl, number, and color depend on yarn construction and abrasion history. Large free hairs that visibly strike another surface require mechanics or a contact-aware motion model; render-only strands cannot visibly spear through the backing.

Denim edges consist of actual remaining yarn ends and fibrillation; no solid extrusion around every hole. Tan edges use the denser brushed construction in section 6 and can present a continuous-looking fuzzy band. Actual voids must remain open. Intact hems or true cut edges may use an appropriate bounded closure. Match the blue and tan edge profiles independently.

### Fiber coverage experiment and acceptance

The first G0 candidate is **stochastic coverage with four-sample MSAA**, depth-tested against opaque yarn cores and cloth. Fibers are opaque microscopic geometry occupying partial pixel area, not translucent glass. Sample masks approximate analytic footprint coverage; patterns derive from stable fiber identity and a controlled sampling sequence, not reseeded geometry. Per-sample depth handles overlapping accepted fiber samples. Use the same coverage model for light transmittance, with independent light-space sampling.

Resolve color and retain enough coverage/depth/material information to reject history at disocclusion and topology changes. Do not average integer IDs; use explicit validity and contributor tests. Mixed samples from different depth layers cannot borrow a single front-layer motion vector indiscriminately. Start by rejecting history for ambiguous mixtures; measure the resulting noise before introducing more elaborate per-sample history. History storage, MSAA targets, and resolve cost belong to G0's resource budget.

This is a candidate, not a settled claim that four samples suffice. Test pale fuzz over dark holes, moving frays, immediate post-break frames, and no-history renders at the minimum presentation rate. Compare an order-independent weighted-blended implementation [R23](research/source-map.md) if stochastic noise or memory is unacceptable; its depth/color bias must be assessed against the same captures. Select and document one method before G0 exits. Do not retain an unspecified transparency placeholder or rely on temporal ghosts to hide inadequate coverage.

### Light transport and shadows

Use a broad, soft key light and restrained fill/environment lighting, initially matched to the reference. Fix exposure during material calibration. Brightness adaptation must not hide incorrect scattering or contact shadows.

The baseline shader combines diffuse fiber scattering, direction-dependent yarn response, and calibrated grazing sheen, with bounded reflected/transmitted energy. Two-sided orientation uses separate geometric and shading normals. Pale frays need broad fiber response, not metallic highlights.

Geometry casts shadows through actual holes. All cloth layers and resolved strands participate. For dense fibers, use coverage/transmittance-aware shadow sampling so a bundle does not become an opaque black ribbon. Begin with multiple stable samples of the broad light, shadow maps for macro geometry, and a compact light-space transmittance representation for fine yarn/fiber detail. Measure memory and light leaks before increasing sample count.

Ambient occlusion is a local visibility correction to indirect illumination. It cannot replace interlayer shadows, erase direct light, or create an arbitrary dark border around holes. Validate the frame without AO first.

Use an offline path-traced version of selected still scenes as a diagnostic for scattering and shadow error. This is development evidence, not prerecorded content used during interaction. Add a WebGPU software-BVH path-traced refinement mode only if raster lighting demonstrably cannot meet G0; it is not an unconditional second renderer to build.

### Render graph

```text
accepted snapshot or certified presentation interpolation
  → visible shell/yarn/fiber reconstruction + bounds + indirect draws
  → light visibility/transmittance passes
  → depth, stable material IDs, motion, normals
  → linear HDR material shading
  → indirect visibility correction
  → temporal resolve with coverage/topology validation
  → one tone map + one output color conversion
  → canvas
```

Small highlights must arise from the material and lighting. Bloom is disabled by default. Motion blur and depth of field remain disabled during validation and are not needed to conceal aliasing or bad contact.

### Temporal stability

History validates depth, material identity, layer, coverage, and topology epoch. Newly exposed backing, newly broken yarn ends, and disoccluded holes reject history. Retain valid history in unaffected regions rather than resetting the entire screen on every break.

Motion vectors use the same material point in consecutive presented states, including yarn sliding and any certified interpolation. New material points have invalid history rather than a made-up zero velocity. When simulation has not advanced, refine lighting/coverage only.

For smooth presentation, buffer accepted substeps and interpolate only within the same topology epoch along a separately certified Cartesian trajectory. Accepted endpoints do not prove their straight chord is collision-free: neither a sequence of Newton updates nor several substeps grants that result. Certify the actual displayed path with swept contact checks and the reconstruction error envelope. Interpolate positions consistently with bounded rod curves and regenerate normals; do not independently slerp frames into unchecked geometry. Include bounds for visible fuzz where it could visibly cross another layer.

Never interpolate across rupture, promotion, changed contact geometry, or component birth. Present the corresponding accepted state at the event boundary and reject affected history. If a chord cannot be certified, use finer accepted substep brackets or present accepted snapshots; do not extrapolate. Target a one-presentation-interval buffer, counted in input latency. Certifying presentation must be included in G1/G5 cost, not treated as free polish.

## 14. Interaction and camera

### Picking

Pick the visible surface at pointer-down using a stable material-ID/depth pass and coverage consistent with final rendering. Return layer, material coordinate or yarn arc location, topology epoch, and depth. Holes reveal deeper layers for picking. Fine fuzz is not normally a target; a substantial exposed thread can be. Picking a buffered presentation state returns material identity and remaps to the current accepted state before attaching.

Use asynchronous readback for the small pick result. Verify that the referenced epoch is still meaningful before attaching. If it is stale, remap by material identity or repick; do not attach to a recycled triangle index. No full position readback is needed for pointer movement.

### Grip mechanics

Attach a finite material patch, with smoothly weighted area, to a compliant virtual handle. A single pinned vertex creates singular stress and encourages needle-like tearing.

Pointer movement defines a target in a camera-facing plane through the initial grip. A small smooth lift can separate the grasped layer at engagement; wheel movement while dragging adjusts depth. Wheel movement without a grip changes camera distance within the calibrated close-up range. Keep the camera fixed during a primary drag.

Use an energy-based grip with a bounded force response and bounded target acceleration. Do not teleport cloth to the cursor. Express grip parameters in physical units and validate the work delivered to the fabric.

The grip remains attached to its original material subset as it tears. Material that leaves the grip loses its share of the coupling; do not renormalize all grip force onto a single surviving yarn. If the patch splits, each retained subset keeps its original weight. Releasing or losing the grip removes the handle without changing particle velocity discontinuously.

Pointer capture, pointer cancellation, leaving the window, focus loss, and pause all have explicit release behavior. A reset cancels pending picks and transactions using a scene-generation token. Touch is optional for the initial desktop target; its absence does not change mechanics.

### Bounded latency and natural timing

Sample input immediately but advance the handle on simulation time using a bounded trajectory. Coalesce superseded move samples while preserving press/release order. Do not execute a long historical pointer path after the user has already stopped.

Normal operation advances simulation at wall-time speed independently of presentation cadence. A 24 Hz image can still display natural-time physics; a quarter-speed simulation cannot. After a transient missed deadline, allow only a bounded catch-up window that preserves input ordering and the feedback gates. If that fails, release the grip and pause from the last accepted state with essential status feedback. On resume, rebase wall time; never replay seconds of buffered motion. Repeated pauses fail qualification. Do not enlarge timesteps beyond validated error bounds, rescale gravity, or silently substitute continuous slow motion. Slow motion is an explicit developer diagnostic only.

Keep one small reset button with keyboard focus and an accessible name. `Escape` releases a grip. A keyboard-operable grip/camera path can use the same handle API without adding a control panel. Cursor state indicates grab availability and active grip; no instructional overlays or diagnostic gauges are part of the normal view.

## 15. Performance and capacity design

Quality and latency are measured together. A 60 Hz canvas presenting an unchanged physics state is not 60 Hz interaction.

### Initial acceptance targets

Select and record an actual base Apple silicon machine with at least 16 GB unified memory during G1, plus a more capable comparison machine. Record chip, GPU cores, memory, OS, Chrome version, power mode, viewport, and thermal conditions. “Modern Mac” is not a reproducible benchmark specification.

| Metric | Proposed normal-operation target |
| --- | --- |
| Input event to first physically affected presented state | p95 ≤ 150 ms |
| Presentation of changing physical state during ordinary pulling | Target ≥ 24/s with bounded cadence; certified interpolation may smooth accepted substeps |
| Heavy fracture burst | p95 feedback ≤ 250 ms, followed by recovery; repeated stalls fail |
| Simulation-time / wall-time ratio during ordinary pulling | 1.0 over steady active windows; bounded transient lag within feedback targets, no accumulated drift |
| Frame presentation | Prefer 24–30/s or better when compute permits; no fixed 60 FPS promise |
| Total engine-owned GPU allocations | Initial planning target ≤ 768 MiB, including temporary peak allocations |
| CPU/WASM live data and staging | Initial planning target ≤ 384 MiB, excluding browser overhead |
| Load-to-interactive | Target ≤ 10 s warm-cache on the baseline; measure cold load separately |

These targets define meaningful interaction for engineering evaluation. If they prove incompatible with reference quality, present measured alternatives before changing the intended experience. Do not silently accept multi-second manipulation delays.

At wall-time speed, `h = 1/60, 1/120, 1/240, 1/480 s` gives gross per-substep budgets of `16.67, 8.33, 4.17, 2.08 ms`, respectively. Rendering, rejected trials, contact/promotion transactions, and presentation certification consume part of those budgets. For example, 8 ms of GPU rendering at 30 presentations/s consumes 240 ms of each second, leaving at most 3.17 ms per 240 Hz physics step before other costs. These are accounting examples, not chosen or measured pass timings. Report p50/p95/p99, thermal steady state, and burst debt. The new timing target is substantially harder than the previous quarter-speed allowance; G1 must confront it explicitly.

### The count model must precede optimization

For layer area `A` and pitches `p_u`, `p_v`, approximate crossing count is `A / (p_u * p_v)`. Count each material independently when pitches differ. At uniform 0.45 mm pitch:

| Extent per layer | Three-layer crossing count | Illustrative four rod nodes per crossing |
| --- | ---: | ---: |
| Fixed 0.06×0.06 m worn-region specimen | 53,333 | 213,333 |
| Provisional 0.10×0.125 m complete coupon | 185,185 | 740,741 |
| Previous large-sheet example, 0.18×0.24 m | 640,000 | 2,560,000 |

Four nodes per crossing is only a sizing example. The asset compiler must enumerate actual centerline subdivision, process segments, endpoint reserves, collars, and crossing variables. It emits `N_full`, every arena's maximum topology allocation, incidence sizes, and temporary coexistence during conversion. Fine tan pitch can dominate counts. G0 fixes the visible physical extent and weave density; G1 measures its full-capacity representation. A 200k-node benchmark is not a product ceiling and cannot establish full-scene feasibility.

Sweep actual active counts in G1/G3, initially around:

- The converged fixed shell mesh across three layers; include 10k–40k vertices as initial profiling sizes, not a resolution mandate.
- 20k, 80k, 200k explicit rod vertices, the fixed-patch actual count, and the compiled `N_full` (potentially well above 740k).
- Contact sets spanning sparse drape through tightly folded three-layer patches, with density and candidate counts recorded.
- Several independently active tear regions and the full-promotion worst case.

These are benchmark sizes, not arbitrary runtime clamps. Budget by allocated bytes, candidate contacts, solver incidence, and temporary peaks, not just rendered triangles.

At 2 million pixels, one single-sample `rgba16float` image uses approximately 16 MB before overhead; four-sample color uses about 64 MB, and four-sample `depth32float` about 32 MB. IDs, motion, histories, shadows, and transmittance add to this. Produce a resource-lifetime table including MSAA, topology reserves, staging, rollback, and reallocation overlap. GPU-owned topology moves latency off the tear path; it does not make the reserve memory free.

### Scheduling and optimization order

Profile CPU work, shader compilation, command encoding, GPU execution, queue wait, readback, topology work, and presentation separately. Timestamp queries supplement wall-clock measurements; their absence cannot disable profiling.

Optimize in this order:

1. Remove redundant candidate generation, full-buffer transfers, per-object JavaScript work, and avoidable synchronization.
2. Improve solver conditioning and physical-region activation.
3. Reuse acceleration structures and immutable material data with valid invalidation rules.
4. Cull/render by bounded projected contribution, preserving silhouettes, shadows, and mechanics.
5. Adjust presentation cadence or additional lighting samples while retaining the validated image quality.

No silent fallback reduces yarn count, solver convergence below acceptance, or physical collision thickness. If no combination meets reference quality and meaningful natural-time interaction, record the failed measurements and redesign before product construction. An explicit pause is preferable to invalid physics, but persistent pauses are a failed gate, not a shipping performance policy.

### Capacity and lifetime

Use explicit reservation and overflow handling for every variable-sized output. Pre-reserve finite topology according to the asset manifest. Contacts and other data-dependent outputs grow in bounded pages or reject/retry without truncation. If a safe growth exceeds the memory budget, fail the state transition visibly and record the gate failure; do not accept incomplete contact. Peak contact count is configuration-dependent even for fixed material counts and requires severe-fold stress fixtures.

Do not demote damaged material or delete quiet load-bearing threads to recover memory. Detached fragments may be retired only when they are outside the supported view and interaction domain, have no remaining connection, and cannot return to affect the scene. Their removal is journaled. Visible detached pieces remain present.

Estimate and allocate full promotion in G1; exercise actual near-total promotion and destruction in G6. Do not bound cumulative damage by grip force or an expected lifetime: an existing notch can propagate at lower load. The final finite asset must support every visible region and multiple separated tears within the declared physical operating envelope. If full capacity is infeasible, revise representation or return with a concrete scope tradeoff; do not quietly retain the fixed worn-region restriction.

### Drawing-buffer cap

The final canvas contains at most 2,000,000 drawing-buffer pixels. Compute:

```text
dpr = min(devicePixelRatio, 1.7, sqrt(2_000_000 / (cssWidth * cssHeight)))
```

Validate finite positive inputs, round final dimensions conservatively, and verify their product remains within the cap. DPR may be below one. Apply this on resize and display changes. Render targets follow the actual drawing-buffer dimensions; offscreen shadow maps have separate explicit memory budgets. Do not render a larger full-scene image and downsample merely to bypass the cap.

## 16. Data contracts and project layout

The architecture should expose narrow contracts rather than let rendering or input modify physical arrays directly.

| Contract | Essential content |
| --- | --- |
| `MaterialAsset` | Version, units, weave/yarn description, constitutive fit, failure law, optical parameters, provenance |
| `SceneAsset` | Three material instances, initial topology/snapshot, mounts, camera, lights, seeds, compiled full-capacity manifest |
| `InputCommand` | Sequence, scene generation, wall timestamp, grip/material reference, desired handle target |
| `TopologyTransaction` | GPU base epoch, reserved ranges, trial activations/connectivity, material ledger, contact changes, remapping, validation status, commit epoch |
| `AcceptedState` | Simulation time, topology epoch, GPU buffer handles, history validity, summary diagnostics |
| `PresentationBracket` | Two accepted times/epochs, certified motion envelope and path, material mapping, valid interpolation interval |
| `FailureEvent` | Yarn/site identity, damage, dissipated work, affected intervals, resulting connectivity |
| `ReplayRecord` | Versions, initial asset hash, inputs in simulation time, adaptive-step decisions, accepted topology events |
| `ValidationResult` | Scene/config hash, hardware, metrics, image/clip paths, failure reason, acceptance thresholds |

Proposed implementation layout:

```text
src/
  app/                 boot, input, minimal UI, capability/error handling
  engine/              worker, scheduler, transactions, resource lifecycle
  gpu/                 device, buffer pools, scans/sorts/reductions, timings
  simulation/
    shell/ yarn/ damage/ coupling/ contact/ solver/
  rendering/
    geometry/ materials/ lighting/ temporal/ passes/
  shared/              schemas, units, IDs, messages
native/
  core/                topology, material compilation, reference mechanics
  reference/           native executable and independent numerical checks
  wasm/                explicit ABI and build configuration
assets/
  materials/ scenes/   versioned authored inputs and compiled asset manifests
tests/
  unit/ numerical/ gpu/ topology/ interaction/ visual/ stress/
benchmarks/
  scenes/ configs/     repeatable workload definitions, not generated captures
tools/
  reference/           reproducible reference-frame extraction
  assets/              later material/scene compilation tools
  codegen/             shared local-energy definitions and C++/WGSL generation
docs/
  design.md
  research/            sources, observations, unresolved numerical risks
  decisions/           review judgments, later gate evidence and resulting changes
artifacts/             ignored generated captures, traces, benchmark results
```

Implement only the directories needed by the current milestone. Native/WebAssembly interfaces exchange coarse typed arrays or batched transactions, not thousands of per-node calls. ABI versions, units, endianness, and buffer ownership are explicit.

Compiled material assets include a reproducible source hash and calibration version. Avoid embedding opaque material constants inside shader source. Keep development diagnostics accessible through an internal route or test harness, not shipped as product panels.

## 17. Recovery, persistence, and reproducibility

### Replay

Record input trajectories in simulation time and retain timestep/topology decisions for diagnostic replay. Stable seeds and event ordering are mandatory. Cross-backend numerical divergence is assessed using tolerances and event envelopes, not a promise of identical floating-point trajectories.

For exact reproduction of an observed failure on the same backend, store accepted-state checkpoints including positions, velocities, rod frames, damage, friction history where required, topology, and material mappings. A seed plus pointer events is insufficient to reproduce a marginal fracture branch after changing the solver.

### Device loss and runtime failure

Handle device loss, allocation failure, invalid asset versions, and shader validation errors explicitly. Keep the last recoverable CPU checkpoint and event journal. Recreate the device and restore only from a consistent checkpoint; otherwise offer a reset with a concise status message. Never silently reconstruct an intact sheet over the user's torn state.

Do not read back the entire scene every frame to provide recovery. Create infrequent asynchronous checkpoints at safe boundaries and measure their memory/latency cost. Bound how much uncheckpointed interaction can be lost and report recovery from an older checkpoint honestly.

Pause on backgrounding and release active grips. On resume, continue from accepted simulation time; there is no minutes-long catch-up step. Reset restores the validated initial snapshot and reclaims or reuses all pools without leaving old pending callbacks active.

The initial product needs no backend service or account system. Persistent user saves, scene sharing, and exports are later features; internal replay and test capture are required from the first numerical milestone.

## 18. Validation and acceptance

### Numerical and mechanical fixtures

| Fixture | What it detects |
| --- | --- |
| Rigid translation/rotation of a shell and yarn patch | Non-objective energy, incorrectly transformed crimp, bad attachment derivatives |
| Warp/weft/bias/biaxial coupon | Isotropic shortcuts, wrong shear response, poor homogenization |
| Hanging strip and free-decay curve | Bending response, numerical versus physical damping |
| Single yarn loading/unloading/rupture | Cohesive work, irreversible history, resolution dependence |
| Yarn pulled through several crossings | Static/kinetic friction, pullout, false welded crossings |
| Opposing grips with one surviving bridge | Whether visible yarns actually transmit load |
| Shell/yarn interface under tension and bending | Artificial clamp, force loss, mass mismatch, reflected motion |
| Promotion during tension, rigid motion, and contact | Transfer impulses, intersections, history loss |
| Two-layer diagnostic and complete three-layer folds | Thickness, ordering, friction, contact overflow |
| Nearly parallel edges, degenerate faces, fast segment/face crossing | CCD and `f32` robustness |
| Severed yarn in compression | New-end contact feasibility and fracture-time filtering |
| Tear through a tile boundary and merge two yarn patches | Duplicate yarns, invalid connectivity, stale grips |
| Large detached scrap and looped thread | Component ownership, collisions, premature retirement |
| Long stiff yarn, loaded crossings, and coupled collar | Stiffness-ratio convergence, preconditioner scaling, block-descent material drift |
| Trial with tiny energy change on a large baseline | `f32` cancellation, reduction error, false line-search acceptance |
| Simultaneous GPU rupture, reset, and delayed journal readback | Partial commits, stale epochs, recovery gaps |
| Accepted endpoints with an intersecting interpolation chord | False presentation safety inferred from endpoint validity |

### Starting numerical thresholds

These thresholds are engineering targets to ratify with G1/G2 evidence. They must be scaled to scene units and must not be loosened simply to make a failing solver pass.

- Native double-precision derivatives agree with central differences over several perturbation scales; target relative error `1e-5` away from nonsmooth branch boundaries. Test branch boundaries separately.
- Conserved material mass through refinement/promotion/fracture has relative ledger error below `1e-6` on CPU. GPU reductions use a separately measured error budget, initially `1e-4`.
- Unforced representation changes preserve linear/angular momentum within `1e-4` relative error with an absolute floor for near-zero states.
- Promotion adds no unexplained positive energy above 1% of the local elastic-plus-kinetic energy scale, and moves visible geometry less than 0.25 pixel in matched views. Near-zero energy cases use a specified absolute tolerance.
- Full-yarn versus mixed coupons agree within 5% on force/displacement and bending response before fracture. Inspect transient motion as well as equilibrium.
- G2 renders have no visible intact collar seam under fixed and moving light/camera views; compare full-yarn and shell-driven crimp without fuzz or temporal accumulation.
- Halving mechanical element length and timestep changes peak coupon force and total fracture work by no more than 5% after calibration. Stochastic crack branches need statistical envelopes, not identical paths.
- No accepted core intersection in the contact fixtures. A distance tolerance is reported in physical units and relative to core radius; it is not permission for visible tunneling.
- Friction and physical damping have non-positive accumulated work in isolated tests. Energy release at fracture is explained by the material energy ledger.
- GPU versus native comparisons use state, force, energy, contact, and damage metrics. Passing an image comparison alone is insufficient.

Choose nonlinear convergence tolerances from these observables. Do not publish a universal residual number with no relationship to scene scale or visual/mechanical error.

### Visual contracts

Maintain fixed-camera captures at full supported resolution and close crops of:

1. Intact weave in flat light and grazing light.
2. A tight fold with both sides visible.
3. A small tear with several tensioned bridges.
4. A large opening exposing tan material.
5. Blue exposed through torn tan material, then all three layers open to a dark void, with clear depth and interlayer shadows.
6. A free frayed edge against a dark background.
7. The same regions during slow camera motion and after a pull is released.

Compare against the original clip at similar damage stages, not necessarily identical timestamps: this product's damage is input-driven. Match camera scale and illumination before judging material detail. Keep the original clip/contact sheet immutable.

Review silhouette, edge irregularity, weave continuity, density, color/scattering, shadow softness, thickness, interlayer depth, recoil, and settling separately. Pixel metrics help detect regression but cannot certify realism when tear topology differs.

Required ablations: unlit geometry, wireframe/material IDs, contact envelopes, damage/load paths, no fuzz, no AO, no temporal resolve, and flat reference lighting. Fuzz must improve an already coherent broken edge; temporal accumulation must not conceal disappearing threads.

### Mechanical reachability of the reference

G4 retains at least three different deterministic pull trajectories from the same worn initial asset: a local opening pull, a pull across several worn cells, and a pull starting in a less worn region. Include release/settling and a backing-layer grab after exposure. Compare the emergence of new holes, pale surviving bridges, strong ligaments, layer exposure, and frayed boundaries at matched image scale. Record handle displacement, force, external work, initial damage, newly dissipated fracture work, and newly disconnected material.

The mechanical parameters and wear asset are fixed across these trajectories; no trajectory-specific crack paths or threshold tweaks. A single lucky pull or a prebuilt late-stage hole does not pass. The reference supplies no force trace, so a numerical force limit cannot be recovered from it: G2/G4 must establish a coherent calibrated operating envelope for the virtual grip. A load outside that envelope is reported as such, not used to justify arbitrary damage. Subsequent G3 tests extend the less-worn-region pull beyond the original fixed patch.

### Operational tests

Exercise pointer cancellation, stale pick results, reset during topology work, resize/DPR changes, unsupported device features, allocation overflow, background/resume, and device loss. Run repeated destruction/reset cycles and a sustained manipulation session long enough to expose thermal throttling and memory growth; start with 20 minutes on the baseline machine.

The first implementation milestone must establish `lint`, `typecheck`, native numerical tests, browser GPU tests, and production build commands. Browser tests run against actual WebGPU hardware as well as ordinary CI where possible; software adapters are useful for correctness but cannot establish performance.

## 19. Implementation sequence and deliverables

### M0 — reproducible foundation

Establish versioned schemas, units, ID/generation rules, build scripts, native/WASM toolchain spike, GPU capability report, resource accounting, and deterministic fixtures. Add test/lint scripts and the accepted/trial buffer discipline before mechanics grow complex.

Deliver: a runnable harness, raw timing traces, fixture serialization, and reference captures. Exit: shader/ABI smoke tests, reset/lifecycle tests, typecheck/lint, and an actual hardware record.

### M1 — static appearance and contact/solver feasibility

Build G0's static three-layer coupon and G1's bounded solver/crossing experiments. Fit projected material scale, compile the complete finite-asset count manifest, and stress its allocation/throughput. Include finite-thickness single-yarn rupture, displacement-form `f32`, energy-difference tests, stiff chain preconditioning, OGC bounds, ACCD ambiguity handling, and supported WGSL binding layouts. Measure coverage/history candidates in moving views.

Deliver: static comparisons, native/GPU correctness results, full-size cost curves, selected solver/contact/crossing combination, selected fiber coverage method, and remaining explicit limitations. Exit: G0/G1. A partial experiment can reject a candidate but cannot qualify the whole product. If interaction at natural timing is implausible, revise now.

### M2 — intact material and fixed mixed patch

Calibrate unit cells, homogenized shell response, bending, friction, and damping. Test the fixed linear collar proposal against full-yarn coupons before treating it as selected. Build the fixed explicit worn patch, settled three-layer scene, finite-patch grips, and visible-material picking. Compare intact appearance across the collar using the shared reconstruction pipeline.

Deliver: G2 material/interface results, free-decay tests, layered folds, appearance comparisons, and grab/release replays. No dynamic conversion is needed to pass this milestone.

### M3 — progressive fracture in the fixed region

Implement cohesive failure, pullout, free-end contact, GPU topology transactions, component ownership, preserved grips, and material-derived edges. Add energy accounting and resolution/timestep sweeps. Complete the worn-scene reachability experiment with several pulls before investing in dynamic conversion. Boundary exhaustion stops the fixture explicitly.

Deliver: G4, surviving-bridge and reference-reachability replays, transaction latency, and checkpointed failures. Exit: progressive tearing already works in simple lighting and can produce the required structure.

### M4 — expansion across the visible domain

If full-yarn mechanics won G1 at complete capacity, validate the whole coupon directly and omit unnecessary promotion machinery. Otherwise implement conservative domain replacement, contact-aware initialization, patch merging, and GPU commit/rollback. Demonstrate expansion during tension, folds, contact, and pullout, including several separated tears.

Deliver: G3 or documented full-yarn equivalence, transfer ledger, promotion latency, and actual growth toward full capacity. Exit: every visible region remains mechanically reachable without artificial anchors or a damage-area cap.

### M5 — integrated reference quality

Integrate validated weave reconstruction, material-specific frays, accurate silhouettes, soft shadows/transmittance, and interlayer illumination. Fit the two material families across three layers. Implement certified temporal presentation and tune interaction without altering physics to compensate for rendering defects.

Deliver: G5 captures and motion comparisons, including ablations and honest end-to-end timings.

### M6 — capacity, recovery, and release qualification

Complete G6, worst-case promotion, severe folds, many fragments, long-session tests, reset/device-loss recovery, and minimal product controls. Optimize only with preserved numerical and visual contracts.

Deliver: the supported hardware/scene envelope, memory peaks, latency distributions, regression captures, and known limitations. Release is blocked by persistent contact artifacts, unsupported visible material states, or feedback below the agreed usability floor.

## 20. Decision status and unresolved implementation work

**Settled requirements:** three authored physical layers; two material families with different edges; mechanically driven new damage; all visible cloth tearable; stable material identity and conservation; accepted-state visibility; natural-time motion; the pixel cap and minimal reset UI. These do not depend on which numerical method wins.

**Starting implementation choices:** fixed shell mesh, fixed explicit worn-patch experiment, precompiled finite topology with GPU transactions, displacement-form `f32`, shared energy generation, one geometric yarn rendering pipeline, and early full-capacity measurement. Change them only for a documented correctness or cost reason. They are not evidence that the unresolved mechanics already works.

**Unresolved choices:** record each result with asset/kernel versions, hardware, raw measurements, failing fixtures, and the reason a candidate was accepted/rejected. Do not close a question with “looks fine” or a source citation. The following registry defines the useful work to do rather than forcing answers prematurely:

| ID / unresolved question | Experiment and evidence needed | Decision deadline / dependent work |
| --- | --- | --- |
| U1: exact denim/backing construction, scale, and finite extent | Native-frame matched static three-layer views, edge profiles, projected yarn counts, lighting sweeps; compile count manifest for each credible fit | G0; blocks frozen scene capacity and final material calibration |
| U2: IPC/PCG versus augmented block descent/OGC | Equal material/contact/rupture fixtures and full-size cost curves at equal physical error; native/GPU comparison including decay and dense contact | G1; blocks production solver and broad optimization |
| U3: general versus sliding-crossing yarns | Pullout, disappearing crossings, free ends, entanglement, layer contact, mass/work-preserving representation switch; test with selected solver | G1; blocks final yarn storage/kinematics |
| U4: finite-precision feasibility bounds | Derive kernel-specific error enclosures, adversarial and randomized geometry tests, supported compiler checks, measured ambiguous-query rate | G1; blocks any collision-safety claim or production acceptance |
| U5: finite-thickness rupture geometry | Connected→cohesive→separated single-yarn tests in tension and compression; mass, work, new-tip contact/filtering invariants | G1 prototype, G4 qualification; blocks cloth fracture |
| U6: fixed coupling and homogenization | Full-yarn versus averaged-linear-collar coupon: moments, rigid motion, membrane/bending response, waves, crimp and shadow seam | G2; blocks dynamic conversion. Prefer full-yarn if the interface cannot pass |
| U7: worn-material strength and fracture work | Several grip paths with fixed material, step/resolution sweeps, reference-shaped cellular openings and load-bearing bridges | G4; blocks claims that interaction reaches reference quality |
| U8: dynamic promotion and lead distance | Transfer into stressed/contacting regions before slip/failure, patch merging, rollback and latency; actual full-capacity growth | G3; blocks unrestricted hybrid release. Omit only if complete full-yarn mechanics qualifies |
| U9: fuzz coverage and lighting | Stochastic-MSAA first candidate versus weighted blending if needed; dark-hole frays, motion, disocclusion, no-history images, resource peaks | G0 choice, G5 integration; blocks final render graph/history allocation |
| U10: timestep and presentation buffering | Step sweeps against native reference, natural-speed recoil, certified interpolation costs, p95/p99 input latency and thermal throughput | G1 provisional, G5 final; blocks interaction qualification |
| U11: complete operational envelope | Actual converged asset counts, full promotion, compressed multilayer folds, many fragments, repeated destruction/reset, long sessions | G1 early rejection, G6 final; blocks release. No silent area limit |

The highest risks remain fracture-time contact, stiff `f32` mechanics, hybrid coupling/transfer, and total compute at natural timing. A credible implementation may reject proposed candidates. The purpose of these gates is to make that rejection early, reproducible, and useful without lowering the visual or interaction target by accident.
