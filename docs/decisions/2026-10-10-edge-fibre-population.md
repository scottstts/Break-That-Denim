# Edge fibre from the fibre population

Supersedes the edge-fibre part of round 7: 18 strands per crossing, the aggregate tuft ribbon (`vsEdgeTuft`) and handoff open item 6.

## Why

A render study measured the free edge's coverage over the void. Method: two stills over different backgrounds, `tools/compare/coverage.mjs`.

- **Before:** the fringe was 25% opaque at 0.9 mm from the edge and 6% at 1.9 mm.
- **Fibre counts imply:** a broken 84-tex warp holds about 470 fibres. With roughly half protruding, the fringe should be opaque at the root (optical depth about 10 along the normal) and thin out over a few millimetres.
- **Shading is not the cause.** Colour, exposure and fill already match the reference (render study, same date). The gap was the fringe's density.

## Population (`fuzz.wgsl`)

Per open side of an edge crossing:

- **Protruding fibres.** Fibres per yarn = linear density / 0.18 tex, passed in `LayerR.loose.zw`.
  - Sheath fibres slip out of the break (0.8) far more often than core fibres (0.3), because the twist presses the core hardest.
  - So a warp's protruding fibre is mostly its indigo sheath (dyed share 0.73).
- **Protrusion length.** Exponential with mean `FRAY_LEN` (1.5 mm), clamped at 6 mm.
  - This is the main free parameter.
  - It is not fitted to the reference, whose fringe forms under a different load.
- **Rim hair.** Hair of the loosened rim yarns, set by Uster-type hairiness (10 mm of fibre per mm of yarn, over a 1.5 mm band). On a warp, 85% of it is dyed, since hair is surface fibre.
- **Fibre albedos.** Single-scattering albedos derived from the woven material's own reflectances via R∞ = (1 − s) / (1 + s), s = √(1 − ω).
  - Calibrating them separately made the deep fringe khaki, because multiple scattering amplifies any tint.

## Representation

Twelve strands per crossing are dealt round its open sides, six per side, lowest first:

1. Rim-hair sheets (two).
2. The compact yarn end.
3. Fans of the end's fibre (three), each holding its share of the population.

How a sheet is evaluated:

- **Per-fibre evaluation.** A sheet holds n fibres. The fragment shader evaluates up to 48 of them: own root, divergence over a 20° fan, crimp, curl that grows with free length, and stratified end.
  - Only where fibres lie further apart than half a pixel.
  - Elsewhere it uses the closed-form mean of the same distribution: a trapezoid across the sheet, with Poisson overlap.
  - Rim hair over the pale cloth also uses the closed form until a fibre covers 6–12% of a pixel. Fans over the void do not: a few faint fibres there are streaks, not a veil.
- **Lighting.** Computed per vertex, once per albedo, with the key and shadow computed once. It is attenuated by the optical depth of the fringe above the strand: slot height times the column depth.
  - The path to the light is not scaled by 1/μ. Strands rise up to ~40°, so the fringe is about as thick as it reaches.
  - Multiple scattering decays by √(3(1 − ω)).
  - The strand's own shadow across its width uses the same density as its coverage.
- **Blending.** Strands are drawn back to front within a crossing (slot order). Alpha blending stays order-correct as long as the key and camera are on the same side; the light-side test flips the attenuation otherwise.

## Rejected

- **Opaque clumps** (coverage 1 − exp(−n d / w) over narrow wedges). They read as straw or rope.
- **Narrow parallel sheets.** They read as paper strips.
- **Sorting fibres by divergence** to cull the per-fibre loop. Curl moves a fibre as far sideways as the fan is wide, so nothing was culled and it ran slower.
- **Fibres in the key's shadow map** (stochastic coverage). No visible effect at the map's bias; image difference about 0.1% MAE.

## Cost

M5, 1600 × 1000, default camera, GPU frame:

| State | Before | After |
|---|---|---|
| `s5-rest` | 4.8 ms | 6.0 ms |
| `rr-haul` | 5.2 ms | 6.6 ms |
| `s1-peel` | 5.8 ms | 8.0 ms |
| `s6-maul` | 8.1 ms | 11.7 ms |
| `s6-maul`, max zoom (0.3) | ~10.6 ms | 22.2 ms |

The breakdown at max zoom: the per-fibre loop takes ~7 ms, the closed form ~3 ms.
