# Material torn from under a pad slips out of the pinch (the crack-tip line-search stalls)

## Cause

The stalls were not a solver defect. The earlier "g.d > 0" lead was a probe artifact: PCG overwrites
the level-0 right-hand side with its residual, and LSPROBE took g.d from that buffer. With the rhs read
before PCG (LSCHECK=1), g.d = -d.Ad: the direction descends, the PCG model is exact, and the operator
is symmetric to rounding and positive definite on 5 x 5 patches around the stalled rows.

The true energy along d was about 500x stiffer than the model. The stalled node (haul, 10285) had one
active span, compressed 14 % at intact 0.22 (the others broken). Its 6.7 N right-hand side was a grip
spring: a rim entry (weight 0.01) of the pinch, about 184 mm from its target. The cloth had torn
between the pinch's core and its rim, so the core left with the hand and the rim stayed on the sheet,
still sprung to the hand. The slack span's model stiffness let Newton aim the full distance (95 mm);
any fraction of it stretched the span into tension. All four step lengths were uphill, and the
global line search rejected the whole sheet's step every iteration for 0.4-0.7 s (haul 80, hold 82).

Probe of entries' deviation from where the rest of the patch holds them (a node's position less its
target offset, against the patch's weighted mean): at most 5.4 mm in intact pinches (haul, hold,
peel, maul, snap). In every tearing gesture, entries torn off reach 20-28 mm by 0.3 s and grow without
bound. The lead cap bounds only the centroid's force, so a split patch pulls its left-behind part
with kg w d from any distance.

## Rule

A pinch holds by friction under its pads. In gripLead (once per substep), an entry whose deviation
exceeds the pad radius (`GripPatch.slip`, GRIP.radius = 8 mm) is no longer under a pad: it slips out
for good (weight 0).

Claws do not slip (`slip` 0). Their yarns are caught on a claw and leave it only by tearing. A claw
patch also spans several claws whose relative motion the rigid grab-time offsets do not follow: in
Beneath, held core entries deviate up to 27 mm while hooked. With the pinch rule applied to claws,
the claws let go of what they were opening, and a bridge stayed across the hole at 3-6 s. Beneath is
now bit-identical to before this change.

The host's core list (detached-piece anchors) is not told about slips. Core entries never slipped in
the measured gestures; the rim tears first.

Also fixed: setTarget wrote the slot's entry count as the patch nodes plus the strip between the pads.
On a regrab with a smaller patch, stale entries of the previous grab were driven again.

## Measured (with / without the slip-out)

| Gesture | Line-search rejections | Ruptures | Settled after release |
|---|---|---|---|
| haul | 0 / 80 | 30 / 46 | never / never |
| haul at 0.006,-0.004 | 0 / 26 | 25 / 61 | 0.33 s / 0.80 s |
| hold | 0 / 82 | 30 / 35 | |
| snap | 0 / 0 | 56 / 71 | 1.20 s / never |
| tear | 0 / 0 | 41 / 65 | 0.47 s / never |
| rip | 0 / 0 | 102 / 121 | 2.37 s / never |
| settle | 0 / 0 | 70 / 66 | 1.53 s / never |
| maul | 0 / 0 | 287 / 283 | |

scrap, drop, peel and three intact haul grab points are identical. Wall time is equal. The tearing
that drops was the rim being pulled across the sheet after the pinch had torn loose.

## Reproduce

`LSPROBE=t LSCHECK=1` on session.ts: per Newton iteration, b.d, d.Ad and the model's energy change
against the evaluated one, the stalled rows' spans (strain now and at 1/8 step), the external
gradient and Hessian with pred - dx, operator symmetry, and patch eigenvalues. tests/gpu/interaction.ts:
pinch slip and claw hold.
