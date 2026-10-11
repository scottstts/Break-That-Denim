/**
 * The floor beneath the coupon (m): a plane parallel to the mount, below everything a held flap hangs to
 * (pulled flaps reach ~6 cm, a grip pushes 3 cm). Long trailing flaps and torn-off pieces reach it.
 */
export const FLOOR_Z = -0.12;

/**
 * Contact law of cloth on the floor (polished terrazzo; SI). Same cubic penalty as self-contact
 * (selfcontact.wgsl): the cloth's fibres compress progressively, the floor is rigid.
 */
export const FLOOR_CONTACT = {
  /**
   * Contact distance beyond the cloth's half thickness (m): the bulge of the rendered bicubic surface
   * past the lattice nodes (SELF_CONTACT.distance less a full thickness). The distance from a node is
   * half the layer's thickness plus this; the node's step never takes it closer than the half thickness
   * (step.wgsl floorBound), so the cloth's underside never passes through the floor.
   */
  bulge: 0.3e-3,
  /**
   * Penalty stiffness k (N/m) of the cubic law: one ply against a rigid plane presses as two plies
   * (SELF_CONTACT, 500 N/m over 1 mm) do at half their compression, so k = 4 x 500 x 0.65 / 1.
   */
  stiffness: 1300,
  /** Cotton denim on polished stone. */
  friction: 0.35,
  /** Speed below which friction is viscous (m/s). */
  frictionSpeed: 0.002,
};
