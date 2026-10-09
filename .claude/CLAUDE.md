# AGENTS.md

## Project Goal

Build an interactive web-based fabric tearing experience that reaches the visual realism of the Houdini reference in `refs/demo.mov`. `refs/demo_contact_sheet.png` is the contact sheet derived from this clip you can directly inspect.

The reference video is the primary quality target. Study it closely and use it as the benchmark for simulation behavior, tearing, material appearance, fine detail, lighting, and overall realism.

Your stacks should go lower level where needed, custom webgpu stack and/or wasm

## Priorities

1. **Visual quality is the highest priority.**
2. Physical behavior should look convincing at both large and small scales.
3. Preserve as much real-time interactivity as reasonably possible.
4. Performance optimizations must not noticeably reduce visual quality.
5. Prefer correctness and realism over shortcuts intended only to improve frame rate.

Real-time 60 FPS is not a requirement. Lower frame rates, variable frame rates, or heavier computation are acceptable if they materially improve the final result.

## UI

minimal UI, no panels, no descriptions, no additional elements, anything that is not strictly necessary should not be there

## Interaction

The fabric should be directly interactive.

At minimum, the user should be able to use the cursor to grab, pull, stretch, deform, and tear the material.

Interaction must remain physically coherent. Pulling and tearing should feel connected to the material rather than like scripted animation or a visual effect layered on top.

## Quality Bar

The target is not merely recognizable cloth tearing.

The result should reproduce the level of realism visible in `refs/demo.mov`, including:

- convincing fabric deformation and folding
- progressive tearing and failure
- irregular and believable torn boundaries
- exposed strands, threads, and fine frayed detail
- convincing material thickness
- realistic interaction between damaged and intact regions
- high-quality contact, self-contact, and layering
- convincing fabric surface structure
- realistic shading, shadows, highlights, and material response
- fine-scale detail that remains convincing in close-up views
- natural secondary motion after pulling or tearing

Avoid obvious game-like, procedural-looking, overly clean, rubbery, or synthetic results.

## Evaluation

Regularly compare the running result directly against `refs/demo.mov`.

Judge progress primarily by visual similarity and physical plausibility, not by implementation complexity, elegance, or benchmark performance.

Do not declare a feature complete merely because it technically works. It is complete only when its visible result meets the quality expected by the reference.

## Drawing-buffer and platform policy

The 2,000,000-pixel drawing-buffer limit is a hard cap. DPR may fall below 1
when the CSS viewport itself exceeds that budget; never restore a final
`Math.max(1, dpr)` floor, because that silently disables the cap on 4K desktop
viewports. Use the following base policy:

```typescript
const maxPixels = 2_000_000;
const width = Math.max(1, window.innerWidth);
const height = Math.max(1, window.innerHeight);

const dpr = Math.min(
  window.devicePixelRatio,
  1.7,
  Math.sqrt(maxPixels / (width * height))
);

renderer.setDrawingBufferSize(
  width,
  height,
  Number.isFinite(dpr) && dpr > 0 ? dpr : 1
);
```

## Decision Resolution

I laid the ground for the build out, that include docs, the reference material, and any other useful artifacts and materials:

- docs/design.md is the central design of the project. It is not beyond reproach, but it is very robust. so unless you have clear and concrete evidence that suggests otherwise, you should follow it
- others in docs/. useful to understand how the decisions were made, what study has been done for this project, details of the project (will have these docs during implementaton)
- refs/. This is the direct outcome i expect from this project
- others such as tools or other artifacts

When you have genuinely difficult decisions during implementation, you should consult these in this order: 1. reference clip. 2. design docs and other docs. Do not rush into decisions, make sure eveything you do is solid and concrete

## General Expectations

Make substantial changes when needed rather than protecting weak existing work.

Do not lower the target because an approach is difficult.

Keep the project stable and testable as it evolves. Avoid regressions in previously successful behavior or visual quality.

When quality and performance conflict, favor quality unless performance becomes so poor that meaningful interaction is no longer possible.

## Rules

- Do NOT use agent skills throughout
- run lint and typecheck as basic checks, tests as well
- When asked to write implementation documentations, do NOT include verbose and irrelevant things like broad project rules, what text was used, etc. The point of documentation for a specific session of implementation is to capture only design choices that were discussed or surfaced during coding beyond what code alone can tell that could potentially impact future implementations, not to repeat what the code or project rules already says
- everything you do you should have performance in mind, such as implementing new features, fixing issues, changing features, etc. That means NO sloppy solution that technically solves the problem at hand but leaves behind performance pitfalls and landmines