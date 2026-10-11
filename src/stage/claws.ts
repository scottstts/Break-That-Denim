import type { HookProfile } from '../simulation/grip';

/**
 * A claw hooked through the weave holds the few yarns on its curve; the creature pulls far harder than
 * a pinching hand (GRIP.maxForce 120 N), so worn and intact denim alike tear from its claws.
 */
export const CLAW_HOOK: HookProfile = { radius: 0.0045, maxForce: 350, maxLead: 0.012 };
