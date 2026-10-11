# Floor, mount and overhead room light (render-only environment)

Follow-up (2026-10-11): the floor now collides in free play (user request); see
`2026-10-11-floor-contact.md`. The "no collision" bullet below is history.

## Decisions

- **Floor 12 cm below the mount, no collision (user decision).** Pulled flaps hang 3.5–6.4 cm
  (`s6-maul`, `t-maul`) and a grip pushes 3 cm, so the cloth never reaches it. A collision floor was
  started and removed at the user's request. Ballistic debris still falls to `RETIRE_Z` (−0.25 m) and so
  passes into the floor (hidden by depth, in shadow); this is the accepted cost of no collision.
- **Overhead lamp (user decision, "see how it looks").** With the key at ~26° elevation a hole's key light
  lands 2× the floor depth (~24 cm) away, under intact cloth; only a near-vertical source puts a lit patch
  inside the hole. Distant lamp tilted ~10° toward +y, angular radius 0.065 (≈30 cm fixture ≈2.3 m up),
  warm. `overIntensity` 0.1: the cloth is ~6 % brighter than the calibrated baseline (0.25 gave +15 % in
  darker regions). It lights and shadows the cloth too (surface, threads, edge fibre), so flaps under the
  sheet stay dark.
- **Mount board.** Without it the key reaches under the coupon's top edge and lights the floor beneath most
  of the cloth. Opaque dark board at the cloth plane (0.35 mm below), inner edge under the pinned collar,
  drawn in both shadow maps and the main view (seen only in the side strips that used to show the void).
  Floor points whose light ray leaves a shadow projection count as blocked by it.
- **Floor lighting from prefiltered coverage, not PCF.** Each shadow map is reduced to (coverage,
  coverage × depth) with box mips (`coverage.wgsl`). Penumbra radius = cloth height above the point along
  the light × source angular radius: the key (0.25 rad, implied by the cloth PCF's 3 texels over 2 mm)
  gives ~7 cm, the lamp ~8 mm, so the hole outline is soft but readable. Fill: cosine kernel of a parallel
  plane at the cloth's height, four equal-energy rings. No noise, ~0.07 ms with gathers (per-texel loads
  were 1.1 ms).
- **Key shadow near plane 0.6 → 0.3 m** to hold the mount where the floor's view of the key crosses it;
  the comparison bias is now in metres (`SHADOW_BIAS` 0.24 mm, as before), so cloth shadows are unchanged.
- **Floor texture tile centred on the view** (`uv + 0.5`). The set has a 2-px bake border and roughness/AO
  that do not wrap at low frequency (roughness 0.22 at the top rows, 0.17 at the bottom); one tile (2.5 m)
  covers far more than the visible floor (~0.72 × 0.5 m through the coupon), so seams never show.
  Normal map is +Y up (OpenGL), measured from AO-gradient correlation (±0.81).

- **Base colour and normal: centre crop of the 8K set** (3277 texels/m, about screen density), 2624 × 1968
  px = 0.80 × 0.60 m: the floor is only ever seen through the coupon (its outline projected to the floor,
  ±0.36 × ±0.27 m from the fixed camera, plus margin). Sampled clamped; `floorDetailTexels` (8192) maps
  them, so a crop must stay centred on the tile. Roughness and AO stay 2K over the whole tile. Originals
  and the unused 2K maps: `artifacts/floor-texture-original/`.

## Known limits

- A camera that sees more floor (wider framing, moving camera) needs a larger crop.
- Cost: 13.8 → 15.4–15.9 ms per frame at 1600×900 on `t-maul` (overhead shadow pass ≈0.7 ms, cloth overhead
  shading, floor ≈0.1 ms, coverage ≈0.07 ms).
- Not modelled: bounce from the lit floor onto the cloth underside; fill on the cloth is still the
  unoccluded hemisphere constant.
- Browser texture path (`createImageBitmap` + `copyExternalImageToTexture`) not exercised headless.

## Reproduce

`node tools/gpu/run.mjs tools/gpu/scripts/still.ts t-maul out 1600 900`; `FLOOR=none` skips the floor,
`LIGHT='{"overIntensity":0}'` removes the lamp (with both: matches the pre-floor cloth except edge-fibre
noise).
