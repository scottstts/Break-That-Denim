# Detached pieces fall: their rigid modes start the Newton direction

Follow-up investigation: this correction does not fully resolve floating fragments. Folded pieces
with internal self-contact still lose common motion to diagonal contact stiffness and absolute
displacement bounds. The "no piece hovers" conclusion below describes the captured run, not every
staged tear state. See `2026-10-11-folded-fragment-contact.md` for the isolated reproduction.

User report: in Beneath, the scrap the first claw cuts tore off floated over the hole, drifted, and
stayed until the end. Free play looked fine.

## Cause

The solve, not the forces. For the stuck 92-node piece (Beneath dump at 3 s), the assembled system is
right: the net rhs equals its weight, and 1^T A 1 = 5.18 N/m against inertia alone 5.12. But two PCG
iterations gave a mean step of 0.027 mm against the exact 0.674 mm (4 %). A 5-node strip got 14 %. A
free piece's rigid modes have only inertia (~0.06 N/m per node) against yarn couplings near 1e6 N/m.
The prolongation reaches only coarse nodes linked through intact spans, so lace-like pieces keep
little coarse support, and the coarsest level is only smoothed. 8 PCG iterations or 40 coarse sweeps
change nothing.

Mounted cloth hides this because the mounts anchor those modes. A compact cut-out piece or a
one-wide strip falls normally. Combs and serpentines (inner spans broken, as tearing leaves them) fall
at 0.24 m/s after 0.25 s instead of ~1.45: tests/gpu/detached.ts.

Ruled out on the way: drag (terminal 2.1 m/s), gravity, a span or shear or bending link to the
cloth (bending weights sum to 2, shear needs two spans), the line search (full steps), contact (with
self-contact off the piece still fell at 0.6 m/s^2), Topology. The earlier "g.d > 0" lead was a probe
artifact (2026-10-11-pinch-slip.md).

## Fix

- Topology labels the detached pieces on every analysis (active-span components among detached, not
  retired nodes) and hands the solver a CSR (`setPieces`, only when it changed). A retired flag (bit 3)
  joined the packed connectivity for this.
- Per Newton iteration, assembly writes for detached nodes their external block (inertia, drag,
  external Hessian, active contact bounds) and the rotation stiffness of their own internal terms
  (rigidK). rigid_frame.wgsl builds each piece's E = Z^T A Z over its six rigid modes (translation,
  rotation about the mass centroid) and inverts it. Internal terms cancel exactly on translation, and
  on rotation they leave their tension's geometric stiffness: span and shear kPerp |d|^2 (I - e e^T),
  bending k (|r|^2 I - r r^T). Summing A itself would cancel 1e5-sized f32 terms.
- PCG starts from x0 = Z E^-1 Z^T b, with the residual b - A x0 (one matvec). PCG minimises the
  A-norm error over x0 + Krylov, so the direction is never worse than that start. Only while pieces
  exist.
- Rejected: adding Z E^-1 Z^T r after every V-cycle (additive coarse correction). A one-node piece
  resting on cloth, already resolved by its V-cycle, was counted twice; its direction turned into the
  contact (-0.14 mm against +0.05) and nodes were pumped off the sheet to 10-190 m/s.
- Rejected: E from the external blocks only. A spinning piece's spans, stretched by the predictor,
  resist rotation ~1e4 times more than its inertia, and the rotation correction swamped the step: the
  tumble test fell at 0.11 m/s.
- Measured: stuck piece 99 % of the exact step (was 4 %), strip 100 % (14 %), a one-node piece on cloth
  exact. tumble unchanged (1.44 m/s, spin 8.4 of 10).

## Effect

Beneath: no piece hovers in the hole. Scraps leave the scene within about 2 s, and what stays lies
on the sheet or on the creature's capsules (oversized; the collider re-fit awaits the user). Falling
pieces now hit the creature, so tearing by 12 s rises to ~1040 from ~900. Sheet:
artifacts/captures/beneath-falling-pieces.png (left fixed, right before; 2, 2.5, 3, 3.5, 4, 6 s).

Free play: ruptures and rejections unchanged; wall time equal (maul 19.0 s both). Released pieces now
keep falling until retired, so session "settled" flips both ways (haul, snap and rip settle sooner;
settle and maul still have a piece moving at the end).

## Open

- Pieces now fall through the render-only floor (the user asked to be reminded: floor collision for
  free play).
- Stage colliders stand 1-5 cm off the creature's skin (tools/stage/colliderfit.ts); pieces resting on
  them float. The proposed re-fit is waiting on the user.
- Pieces split without a new analysis keep one label until the next one (a valid, weaker coarse space).

## Reproduce

tests/gpu/detached.ts (comb, serpentine). Beneath DEBRIS=1 / DEBRISAT=t,.. / WATCH=n / DUMPAT=t;
tools/studies/restcontact.ts <state> with WATCH=n (NOPIECES=1 without the correction).
