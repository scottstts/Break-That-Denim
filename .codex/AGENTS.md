## Project

This is a hyper-realistic fabric tear simulation experience

## UI

minimal UI, no panels, no descriptions, no additional elements, anything that is not strictly necessary should not be there

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

## Rules

- no live browser during build, use the existing headless setup if you need to see it
- Do NOT use agent skills throughout
- run lint and typecheck as basic checks, tests as well
- When asked to write implementation documentations, do NOT include verbose and irrelevant things like broad project rules, what text was used, etc. The point of documentation for a specific session of implementation is to capture only design choices that were discussed or surfaced during coding beyond what code alone can tell that could potentially impact future implementations, not to repeat what the code or project rules already says
- everything you do you should have performance in mind, such as implementing new features, fixing issues, changing features, etc. That means NO sloppy solution that technically solves the problem at hand but leaves behind performance pitfalls and landmines
- i want to stress this again: don't try to get close to the reference for the sake of getting close to that visual, and sacrifice the correctness of the engine. the reason why the reference looks so good is because it looks realistic and that realism comes from getting the underlying mechanisms correct. trying to get to a close visual match at the expense of messing up the underlying engine (like taking cheap shortcuts, etc) is counterproductive and you should NOT do it