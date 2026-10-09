# Research source map

Research date: 2026-10-09. These are primary papers, author project pages, official specifications, and implementation repositories. Search-engine publication/crawl dates are not treated as paper publication dates.

The [design](../design.md) makes project-specific decisions. A citation here supports a method or identifies a risk; it does not validate our eventual implementation, browser performance, or the integration of several methods. No third-party solver code or research assets have been copied into the project.

## Mechanics, fracture, and representation

### R1 — Simulation of Tearing Cloth with Frayed Edges

Metaaphanon, Bando, Chen, and Nishita, Pacific Graphics / Computer Graphics Forum, 2009.

[Author project page](https://web.media.mit.edu/~bandy/cloth/) · [Paper](https://web.media.mit.edu/~bandy/cloth/PG09cloth.pdf)

The work specifically addresses yarn movement, slippage, and fraying near a tear, with a transition from a continuum cloth representation to a local yarn representation. This is the closest conceptual precedent for treating fraying as material behavior instead of post-fracture decoration.

Use it when designing conversion and pullout tests. Do not infer modern contact guarantees or WebGPU performance from it. The project abstract and indexed paper excerpts were available during this review; a full equation-level implementation review is still required.

### R2 — Yarn-Level Simulation of Woven Cloth

Cirio, López-Moreno, Miraut, and Otaduy, ACM TOG 33(6), 2014.

[Institutional publication page](https://www.ccs.upm.es/research/publications/yarn-level-simulation-of-woven-cloth/) · [DOI](https://doi.org/10.1145/2661229.2661279) · [Author-hosted paper](https://filedn.eu/l8vDXhPLwQPyaW6XewVeXCk/publications/yarnlevel_woven_cloth/paper.pdf)

Uses a crossing/sliding discretization to reduce the cost of yarn contact. Relevant to crimp, nonlinear behavior, friction, snags, and tearing. Indexed paper material describes stress-based yarn splitting followed by relaxation, which reinforces the need to redistribute load after failure.

The proposal is not equivalent to rods with welded crossings. General-contact rods and specialized sliding coordinates are now coequal G1 candidates. The latter needs explicit free-end, mass/inertia, resampling, external-contact, and representation-switch tests; lower crossing cost does not remove these requirements.

### R3 — Mixing Yarns and Triangles in Cloth Simulation

Casafranca, Cirio, Rodríguez, Miguel, and Otaduy, Computer Graphics Forum 39(2), 2020.

[Publisher record](https://diglib.eg.org/items/31e33fda-4d0e-4596-99e4-27ebf8f1e9a5) · [Institutional paper](https://burjcdigital.urjc.es/server/api/core/bitstreams/ffc0c836-7d5f-4e71-8bec-5c518cc55907/content)

Combines triangle-scale deformation with yarn enrichment, explicitly addressing redundant kinematics, rank deficiency, and preconditioning. The indexed preconditioner section shows that naive mass-diagonal preconditioning can leave yarn motion lagging the triangle motion.

This is evidence that multiscale coupling needs numerical design, not just barycentric attachments. Our disjoint-domain replacement is a different formulation; its conversion, inertia, contact, and fracture behavior remain experimental. Do not cite this paper as proof of those extensions.

### R4 — Adaptive Tearing and Cracking of Thin Sheets

Pfaff, Narain, de Joya, and O'Brien, SIGGRAPH / ACM TOG, 2014.

[Author publication page](https://tobiaspfaff.com/publication/pfaff-2014-adaptive/) · [Paper](https://graphics.berkeley.edu/papers/Pfaff-ATC-2014-07/Pfaff-ATC-2014-07.pdf) · [ARCSim project](https://jamesobrien.com/resources/ARCSim/)

Relevant to adaptive resolution, stress redistribution, material-space defects, and reducing mesh dependence in thin-sheet fracture. It does not by itself produce woven yarn pullout. Our design uses yarn-defined fracture sites rather than allowing the shell triangulation to determine the final tear.

ARCSim is a reference implementation to study, not a dependency selected for copying. Its published usage conditions must be checked if code reuse is proposed.

### R5 — Discrete Elastic Rods

Bergou, Wardetzky, Robinson, Audoly, and Grinspun, SIGGRAPH / ACM TOG, 2008.

[Author project and paper links](https://www.cs.columbia.edu/cg/rods/index.html)

Foundation for rod centerlines, transported frames, bending, and twist. Use its mechanics and validation style when implementing exposed yarns. Textile yarns still need calibrated axial/bending properties, damage, contact, and optical construction; a circular elastic rod is not automatically a complete cotton-yarn material model.

## Contact and solver choices

### R6 — Incremental Potential Contact and Codimensional IPC

Li and collaborators, 2020; Li, Kaufman, and Jiang, 2021.

[IPC project](https://ipc-sim.github.io/) · [C-IPC project](https://ipc-sim.github.io/C-IPC/) · [C-IPC preprint](https://arxiv.org/abs/2012.04457) · [IPC technical supplement](https://ipc-sim.github.io/file/IPC-supplement-A-technical.pdf)

The relevant structure is an implicit energy solve with contact barriers, friction, and collision-filtered search steps. The codimensional extension treats shells and rods with thickness and supplies Additive CCD (ACCD), the initial CCD candidate. It is one branch of the early comparison, not a predetermined production winner.

The IPC supplement cautions that scaling a computed time of impact does not resolve all floating-point errors. Conservative separation queries and validation matter. Our changing topology, selected coupling/crossing coordinates, and WGSL arithmetic require separate validation; published guarantees cannot simply be inherited.

### R7 — Unlocking Thickness Modeling for Codimensional Contact Simulation

Gomez-Nogales, Chen, Martin, Garces, and Kaufman. Preprint posted 2025; the available manuscript carries 2026 venue metadata.

[Preprint record](https://arxiv.org/abs/2510.10256) · [Full HTML manuscript](https://arxiv.org/html/2510.10256v1)

Identifies contact locking when nearby intrinsic neighbors lie inside the nominal thickness envelope at fine discretization. Broadly culling these pairs can cause pull-through. The proposed filtering changes their effective contact thickness while retaining protection against intersection.

This directly changes our contact tests and neighbor classification. The paper also identifies limitations for spatially varying thickness and coupling transverse compression to other constitutive behavior. A dynamically torn, remeshed hybrid domain adds further problems; that extension must not be presented as already solved.

### R8 — StiffGIPC

Huang, Lu, Lin, Komura, and Li, ACM TOG, 2025.

[Paper](https://kemenghuang.github.io/img/stiff_gipc.pdf) · [Authors' repository](https://github.com/KemengHuang/Stiff-GIPC) · [Preprint record](https://arxiv.org/abs/2411.06224)

Useful for GPU preconditioning, stiffness disparity, assembly/reduction strategy, and realistic expectations of implicit contact cost. Its main experimental setup uses an RTX 4090; difficult examples still take seconds per simulated step. These are not predictions for our smaller scene, but they rule out assuming that “GPU IPC” implies interactive browser performance.

Use the paper's cost decomposition to structure our benchmarks. Do not transplant CUDA-specific assumptions, timing claims, or native memory behavior into WGSL.

### R9 — Small Steps in Physics Simulation

Macklin and collaborators, SCA, 2019.

[Authors' paper](https://matthias-research.github.io/pages/publications/smallsteps.pdf)

Shows the value of smaller timesteps in XPBD compared with spending the same work on more iterations of a large step. Useful as an alternative-solver experiment and as a warning against treating frame time as physics timestep.

It does not establish the accuracy of a particular textile fracture model, robust contact at arbitrary speed, or the quality of a solver stopped early. Compare measured material response and dissipation, not just whether the cloth remains numerically stable.

### R10 — Vertex Block Descent and Augmented Vertex Block Descent

Chen and collaborators, 2024; Giles, Diaz, and Yuksel, 2025.

[VBD paper](https://www.cemyuksel.com/research/papers/vbd-siggraph2024.pdf) · [AVBD paper](https://www.cemyuksel.com/research/papers/Augmented_VBD-SIGGRAPH25.pdf) · [AVBD author project](https://graphics.cs.utah.edu/research/projects/avbd/)

VBD provides a local optimization route to implicit dynamics, but the paper documents difficulty at high stiffness ratios. AVBD addresses constraints and stiffness disparity using an augmented formulation.

These are early G1 candidates alongside the global solver. Published demonstrations do not establish thin-yarn fracture quality, compatibility with our OGC/coupling configuration, or Apple GPU performance. Compare physical observables at matched accuracy; the iterative machinery and contact potentials need not be identical.

### R11 — IPC Toolkit

[Official repository](https://github.com/ipc-sim/ipc-toolkit)

Provides contact energy derivatives, distance queries, CCD, and friction-related building blocks. The repository explicitly states that it is not a complete simulator. Its stated license is MIT; check the exact chosen revision and transitive dependencies during adoption.

Candidate for the native reference implementation and selected WebAssembly geometry functions. No browser build or scene-scale performance has been verified in this design task. Pin a tested revision rather than depending on a moving branch.

[Official releases](https://github.com/ipc-sim/ipc-toolkit/releases) record OGC support in v1.5.0 and subsequent OGC feasibility fixes and API changes in v1.6.0. This makes shared native geometry infrastructure plausible, but is a reason to audit the chosen revision and regression cases, not to assume any port is correct. CPU LBVH performance reported there is not WGSL solver performance.

## Appearance

### R12 — Textile appearance and efficient fiber geometry

[Irawan and Marschner: Specular Reflection from Woven Cloth, 2012](https://www.cs.cornell.edu/~srm/publications/TOG12-cloth.html)

Supports relating directional reflectance and visible texture to fiber, yarn, and weave structure. Useful for a unified material definition across geometric and unresolved appearances.

[Wu and Yuksel: Real-time Fiber-level Cloth Rendering, 2017](https://people.csail.mit.edu/kuiwu/rtfr.html)

Supports on-demand fiber detail, aggregate yarn-core representation, and detail reduction based on image contribution. This is a rendering result, not evidence that all those fibers are mechanically simulated. WebGPU must use its own compute/indirect rendering path.

[Zhao, Jakob, Marschner, and Bala: Modeling and Rendering Fabrics at Micron-Resolution, 2012](https://research.cs.cornell.edu/ctcloth/)

Supports treating textile structure as volumetric at close range rather than assuming that a flat BRDF/normal map captures every effect. Useful as a reference for offline diagnostic renders, not a requirement to store a dense volume per frame.

[Li and collaborators: Fiber-level Woven Fabric Capture from a Single Photo, 2024](https://arxiv.org/abs/2409.06368)

Useful if authored material fitting cannot match real samples. It assumes a suitable microscope image and a capture/optimization pipeline; the supplied compressed video does not provide equivalent input.

## Browser and implementation constraints

### R13 — WGSL specification

[Official specification](https://www.w3.org/TR/WGSL/)

Use normative layout, synchronization, numeric, and atomic rules when designing kernels. The portable baseline has integer atomics, not floating-point atomic addition. Do not assume arbitrary scalar precision, CUDA warp behavior, geometry shaders, or dispatch-wide barriers.

The specification evolves. Pin the subset implemented by the actual target Chrome and validate every optional feature at runtime.

### R14 — Device limits and mapped buffers

[MDN: GPUSupportedLimits](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits) · [MDN: GPUBuffer.mapAsync](https://developer.mozilla.org/en-US/docs/Web/API/GPUBuffer/mapAsync)

Query and request actual supported limits; do not assume one large storage buffer can be bound everywhere. The documented default maximum storage binding size is 128 MiB and buffer size 256 MiB, but runtime values are authoritative. Mapping requires compatible buffer usage and excludes simultaneous normal GPU use, motivating separate staging pools.

### R15 — Worker canvas ownership

[MDN: OffscreenCanvas](https://developer.mozilla.org/en-US/docs/Web/API/OffscreenCanvas)

Supports moving rendering ownership away from the main input/UI thread. Test the complete WebGPU worker path, canvas transfer, resize, and recovery on the selected Chrome; an API's existence does not validate our lifecycle implementation.

### R16 — WebAssembly threading

[Official Emscripten pthread documentation](https://emscripten.org/docs/porting/pthreads.html)

Threaded WebAssembly has browser isolation/deployment requirements. Start with a single-threaded worker module. Introduce threads only after measuring a CPU bottleneck and validating headers/asset compatibility.

## Houdini context

### R17 — Vellum documentation

[Cloth simulation](https://www.sidefx.com/docs/houdini/vellum/clothsimulation.html) · [Constraint attributes](https://www.sidefx.com/docs/houdini/vellum/vellumattributes.html) · [Solver](https://www.sidefx.com/docs/houdini/nodes/dop/vellumsolver)

Useful examples of production concerns: stretch/bend separation, constraint break criteria, thickness, and difficulties when newly separated material enters collision handling. This documentation does not identify the method used in our reference clip. We have no original Houdini scene or solver settings.

## Sources added during design review

### R18 — Homogenized Yarn-Level Cloth

Sperl, Narain, and Wojtan, ACM TOG / SIGGRAPH, 2020.

[Author project, paper, supplement, fitting data/code](https://visualcomputing.ist.ac.at/publications/2020/HYLC/)

Derives a shell energy density from simulated yarn configurations. It is the concrete starting method for G2's intact constitutive fit. Its reversible energy model does not establish damaged/slipping history transfer or our shell/yarn boundary. Reproduce relevant unit-cell responses before extending the model.

### R19 — Mechanics-Aware Deformation of Yarn Pattern Geometry

Sperl, Narain, and Wojtan, ACM TOG / SIGGRAPH, 2021.

[Author project, paper, supplement, code](https://visualcomputing.ist.ac.at/publications/2021/MADYPG/)

Uses mesh strain and precomputed yarn configurations to reconstruct mechanics-aware detail. Relevant to intact shell-driven crimp and promotion initialization. It supplies appearance geometry, not fracture connectivity or a proof that our shell-driven and explicitly simulated regions will match at their interface.

### R20 — Fitting Procedural Yarn Models for Realistic Cloth Rendering

Zhao, Luan, and Bala, SIGGRAPH, 2016.

[Author-hosted paper](https://www.cs.cornell.edu/~kb/publications/SIG16ProceduralYarn.pdf) · [Institutional publication listing](https://rgb.cs.cornell.edu/author/shuang-zhao/)

Fits procedural fiber geometry and flyaway statistics to CT measurements. Useful for structured ply/fiber authoring instead of uniformly random fuzz. We have no CT scan of the reference textile, so fitting this material remains an authored calibration problem, not a recovered measurement.

### R21 — Offset Geometric Contact

Chen, Hsu, Liu, Macklin, Yang, and Yuksel, ACM TOG 44(4) / SIGGRAPH, 2025.

[Author project and paper/code links](https://ankachan.github.io/Projects/OGC/) · [Institutional paper](https://graphics.cs.utah.edu/research/projects/ogc/Offset_Geometric_Contact-SIGGRAPH2025.pdf)

Uses offset contact geometry and local displacement bounds to avoid repeated global CCD. Published examples include yarn-level interaction. This justifies an early competitor to IPC, not an assumption of free or universally robust contact. Our changed topology, `f32` kernels, mapping constraints, and friction require qualification; inspect concave/open boundaries and bound-induced damping.

The full PDF exceeded the research reader's size limit; the author page and indexed paper sections informed this architectural comparison. Read the full algorithm, limitations, supplements, and selected code before implementation. R11 offers an additional native implementation to audit; linked implementations can differ from the paper and each other.

### R22 — Denim construction and dye exposure

Cotton Incorporated, CottonWorks technical education.

[Denim construction](https://cottonworks.com/learning-hub/denim/denim-construction/) · [Denim basics](https://cottonworks.com/learning-hub/denim/denim-basics/)

Supports the traditional dyed-warp/undyed-weft construction and abrasion revealing a lighter yarn interior. Denim variants exist. Use this to author a specific ring-dyed warp model; do not infer from blue color that every thread, fiber, or worn patch has exactly that construction.

### R23 — Fiber coverage / transparency candidates

[Enderton, Sintorn, Shirley, and Luebke: Stochastic Transparency, I3D 2010](https://www.nvidia.com/docs/IO/88888/StochasticTransparency_I3D2010.pdf)

Samples subpixel coverage using multisample depth, avoiding fragment sorting at the cost of noise. Supports the first G0 coverage experiment. Four samples, temporal rejection at fracture, and this scene's pale-on-dark frays still require image and memory validation; the paper does not certify that operating point.

[McGuire and Bavoil: Weighted Blended Order-Independent Transparency, JCGT 2013](https://jcgt.org/published/0002/02/09/paper.pdf)

Alternative with bounded buffers and approximate depth weighting. Compare its color/occlusion bias against stochastic noise if necessary. Neither method licenses treating a physical hole as partially opaque or accumulating invalid temporal history.

## Research conclusions to preserve

1. Fraying, pullout, and bridging require material identity and yarn mechanics; fine noise alone cannot supply them.
2. Hybrid methods move cost into coupling, transfer, and conditioning. They are not a free visual-detail layer.
3. Contact thickness and mechanical resolution interact. Simply increasing resolution or excluding more neighbors can make correctness worse.
4. Numerical stability, mechanical accuracy, collision robustness, and interaction latency are different measurements.
5. No located source demonstrates this complete three-layer, dynamically tearing system at the proposed browser/Apple silicon operating point. The feasibility gates are necessary evidence, not optional polish.
6. Solver, contact, crossing coordinates, and coupling cannot be selected independently without a combined validation. Shared terminology or a common toolkit does not establish compatibility.

## Source-review limitations

Several PDF endpoints failed or exceeded the research reader's size limit. Project pages, publisher abstracts, and indexed excerpts were used only for claims supported there. R7's HTML manuscript, R8's parsed paper, and available solver/specification excerpts supported more detailed inspection. Review additions likewise do not imply a complete equation-level audit: R18/R19 project descriptions and indexed technical material, R20 indexed paper text, R21 project/indexed sections, and R23 paper summaries support the proposed experiments. Before implementing an algorithm, read its complete paper, supplement, and chosen code revision; record changed assumptions in `docs/decisions/`.

Avoid equating this bibliography with a dependency list. Algorithms, reference implementations, example assets, and production runtime dependencies have different adoption requirements.
