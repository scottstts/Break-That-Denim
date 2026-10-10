import { DENIM, FAMILY_U, FAMILY_V, bundleWeave, crossingMass, type WeaveMaterial } from './material';
import { fbm, noise1, rand01, smoothstep } from '../shared/noise';
import { DAMAGE_CUT, MAX_BUNDLE, areaCode, coupleSpans, fieldLayers, pullOut, type FieldLayer } from './yarnField';

/**
 * Scene asset compilation: three independent woven lattices with explicit identities, mounts,
 * correlated wear, per-segment cohesive strength, and pre-existing defects.
 *
 * Indexing (per layer, local): node n = j * nx + i, i = v-yarn (warp end) index, j = u-yarn (pick) index.
 * Global node index = layer.nodeOffset + n. Each node owns segment 2*n + 0 (to (i+1, j), u-family)
 * and 2*n + 1 (to (i, j+1), v-family). Tails are per node and side: [u-, u+, v-, v+].
 */

/** Seed of the shared initial drape perturbation field. */
const DRAPE_SEED = 101;

export const SEG_EXISTS = 1;
export const SEG_BROKEN = 2;
/** Broken before interaction (authored defect), kept for the material ledger. */
export const SEG_INITIAL_DEFECT = 4;

export interface WearSpec {
  /** World-rest-plane center of the worn zone (m). */
  center: [number, number];
  /** Major-axis direction (unnormalized). */
  axis: [number, number];
  /** Semi-axes (m). */
  radii: [number, number];
  /** Maximum wear in the zone (0..1). */
  amount: number;
  /** Cellular opening scale (m); 0 disables the cellular term. */
  cellSize: number;
  /** Length scale (m) of the wear field's internal structure (domain warp, fine variation). */
  featureScale: number;
  /** Wear above which warp floats are worn through (separated) before interaction. */
  defectThreshold: number;
  /** Fraction of warp segments beyond the threshold that are worn through. */
  defectDensity: number;
  /** Wear from which single warp floats start to be worn through (rising toward defectThreshold). */
  sparseFrom: number;
}

export interface LayerSpec {
  /** Woven material of the individual yarns. */
  material: WeaveMaterial;
  /** Yarns per mechanical lattice span: [warps per column (u), picks per row (v)]; default [1, 1]. */
  bundle?: [number, number];
  /** Material extent (m). */
  sizeU: number;
  sizeV: number;
  /** Mount plane height (m). */
  z: number;
  /** Rotation of material axes in world XY (rad). */
  angle: number;
  /** Mount mapping: compression along u, v and shear (dimensionless). */
  compressU: number;
  compressV: number;
  shear: number;
  collarRows: number;
  wear: WearSpec;
  /** Strength multipliers at full wear for [u-family, v-family]. */
  wearStrengthLoss: [number, number];
  seed: number;
  /** Initial out-of-plane perturbation amplitude (m) that seeds buckling. */
  perturbation: number;
}

export interface LayerInfo {
  index: number;
  /** Mechanical (bundled) material of the lattice spans. */
  material: WeaveMaterial;
  /** Individual-yarn material (rendering, yarn identities). */
  weave: WeaveMaterial;
  /** Yarns per lattice span: [warps per column (u), picks per row (v)]. */
  bundle: [number, number];
  nx: number;
  ny: number;
  nodeOffset: number;
  nodeCount: number;
  /** World mapping of material coords for mounts: p = origin + A * (u, v). Row-major 2x2. */
  mapA: [number, number, number, number];
  mapOrigin: [number, number, number];
  restCenter: [number, number];
}

export interface CompiledScene {
  version: number;
  layers: LayerInfo[];
  nodeCount: number;
  /** Material (rest) coordinates per node, m. */
  restUV: Float64Array;
  position: Float64Array;
  mass: Float64Array;
  pinned: Uint8Array;
  /** Visible abrasion per node, 0..1 (drives optics and fuzz; strength uses the structural wear). */
  wear: Float32Array;
  /** Structural wear per node, 0..1 (strength, extensibility). */
  structuralWear: Float32Array;
  restLength: Float64Array;
  /** Softening opening of the span's yarns past their peak (m), see yarnField.ts. */
  softening: Float64Array;
  segFlags: Uint16Array;
  /** Fraction of the span's yarn load capacity that is intact (yarn damage field). */
  intact: Float64Array;
  /** Per span: bit q = bundled yarn q has separated within the span. */
  cutMask: Uint32Array;
  /** Where a separated span split into tails (fraction from the owning node). */
  failSite: Float32Array;
  /** Yarn damage field layout and initial state (yarnField.ts). */
  field: FieldLayer[];
  /** Per crossing and family (2 c + f): the yarn segment's peak tension, N. */
  fieldStrength: Float32Array;
  /** Per crossing and family: initial damage code (255 = worn through before interaction). */
  fieldDamage: Uint8Array;
  /** Per crossing and family: cross-section code of the abraded yarn segment (yarnField.ts areaCode). */
  fieldArea: Uint8Array;
  /** Per-node crossing normal force at zero tension (wear-reduced), N. */
  crossingNormal0: Float64Array;
  tails: Float64Array;
}

/**
 * Version of the mechanical model (laws and their constants outside the material assets). Bump it
 * whenever equilibrium shapes change, so stale precomputed drapes are rejected.
 */
export const MECHANICS_VERSION = 9;

/** Content hash of a compiled scene's initial state and mechanics (validates precomputed drapes). */
export function sceneHash(s: CompiledScene): number {
  let h = 0x811c9dc5;
  const mix = (v: number) => (h = Math.imul(h ^ v, 0x01000193) >>> 0);
  mix(MECHANICS_VERSION);
  for (const L of s.layers) {
    const json = JSON.stringify(L.material);
    for (let i = 0; i < json.length; i++) mix(json.charCodeAt(i));
  }
  // initial worn-through yarns change the rest state (they are not part of segFlags until a span
  // has lost all its yarns)
  for (let i = 0; i < s.fieldDamage.length; i += 64) h = Math.imul(h ^ s.fieldDamage[i] ^ (s.fieldArea[i] << 16) ^ (i << 8), 0x01000193) >>> 0;
  const f = new Float32Array(s.position);
  const u = new Uint32Array(f.buffer);
  for (let i = 0; i < u.length; i++) h = Math.imul(h ^ u[i], 0x01000193) >>> 0;
  for (let i = 0; i < s.segFlags.length; i++) h = Math.imul(h ^ s.segFlags[i], 0x01000193) >>> 0;
  return h >>> 0;
}

/** Worley F1/F2 distances for a jittered grid (cell units). */
function worley(seed: number, x: number, y: number): [number, number] {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  let f1 = 1e9;
  let f2 = 1e9;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const cx = ix + dx;
      const cy = iy + dy;
      const px = cx + 0.15 + 0.7 * rand01(seed, cx, cy);
      const py = cy + 0.15 + 0.7 * rand01(seed + 17, cx, cy);
      const d = Math.hypot(px - x, py - y);
      if (d < f1) {
        f2 = f1;
        f1 = d;
      } else if (d < f2) {
        f2 = d;
      }
    }
  }
  return [f1, f2];
}

/**
 * Wear in [0, 1] at world-rest-plane point (x, y): [structural, visible]. Both share the worn zone's
 * envelope and fine variation; the cellular structure (worn cells, retained ligaments) governs the
 * yarns' remaining strength fully, but shows on the surface only partly, as the abraded zone is
 * diffusely pale rather than drawn in cells, and most where the wear is deepest.
 */
export function wearAt(spec: WearSpec, seed: number, x: number, y: number): [number, number] {
  if (spec.amount <= 0) return [0, 0];
  const al = Math.hypot(spec.axis[0], spec.axis[1]) || 1;
  const ax = spec.axis[0] / al;
  const ay = spec.axis[1] / al;
  const dx = x - spec.center[0];
  const dy = y - spec.center[1];
  const a = (dx * ax + dy * ay) / spec.radii[0];
  const b = (-dx * ay + dy * ax) / spec.radii[1];
  const S = spec.featureScale;
  const warp = 0.32 * fbm(seed + 3, x / (1.4 * S), y / (1.4 * S), 4);
  const r = Math.hypot(a, b) + warp;
  let w = 1 - smoothstep(0.4, 1.1, r);
  let cells = 1;
  if (spec.cellSize > 0) {
    const jx = x + 0.4 * S * fbm(seed + 11, x / S, y / S, 3);
    const jy = y + 0.4 * S * fbm(seed + 12, x / S, y / S, 3);
    const [f1, f2] = worley(seed + 5, jx / spec.cellSize, jy / spec.cellSize);
    // Worn hardest at cell centres, recovering toward the boundaries between neighbouring cells
    // (retained ligaments, nearly intact). Under load a cell then fails from its middle outward along
    // these contours, into a rounded opening that the ligaments around it arrest.
    const interior = Math.pow(smoothstep(0.03, 0.3, f2 - f1), 0.7);
    cells = interior;
  }
  // Fine correlated abrasion variation.
  w *= (0.82 + 0.18 * fbm(seed + 7, x / (0.3 * S), y / (0.3 * S), 3)) * spec.amount;
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  // visible abrasion is concentrated where the cloth is most worn (around the damage), not spread
  // over the whole zone the structural wear weakens
  return [clamp(w * (0.12 + 0.88 * cells)), clamp(Math.pow(w, 1.8) * (0.65 + 0.35 * cells))];
}

export function compileLayer(spec: LayerSpec, index: number, nodeOffset: number): { info: LayerInfo } {
  const bundle = spec.bundle ?? [1, 1];
  const m = bundleWeave(spec.material, bundle[0], bundle[1]);
  const nx = Math.floor(spec.sizeU / m.pitchU) + 1;
  const ny = Math.floor(spec.sizeV / m.pitchV) + 1;
  const c = Math.cos(spec.angle);
  const s = Math.sin(spec.angle);
  // A = R(angle) * [[1 - cu, shear], [0, 1 - cv]]
  const k00 = 1 - spec.compressU;
  const k01 = spec.shear;
  const k11 = 1 - spec.compressV;
  const mapA: [number, number, number, number] = [c * k00, c * k01 - s * k11, s * k00, s * k01 + c * k11];
  const restCenter: [number, number] = [((nx - 1) * m.pitchU) / 2, ((ny - 1) * m.pitchV) / 2];
  const mapOrigin: [number, number, number] = [0, 0, spec.z];
  return {
    info: { index, material: m, weave: spec.material, bundle, nx, ny, nodeOffset, nodeCount: nx * ny, mapA, mapOrigin, restCenter },
  };
}

export function compileScene(specs: LayerSpec[]): CompiledScene {
  const layers: LayerInfo[] = [];
  let offset = 0;
  for (let l = 0; l < specs.length; l++) {
    const { info } = compileLayer(specs[l], l, offset);
    layers.push(info);
    offset += info.nodeCount;
  }
  const N = offset;
  const fl = fieldLayers(layers);
  const scene: CompiledScene = {
    version: 1,
    layers,
    nodeCount: N,
    restUV: new Float64Array(2 * N),
    position: new Float64Array(3 * N),
    mass: new Float64Array(N),
    pinned: new Uint8Array(N),
    wear: new Float32Array(N),
    structuralWear: new Float32Array(N),
    restLength: new Float64Array(2 * N),
    softening: new Float64Array(2 * N),
    segFlags: new Uint16Array(2 * N),
    intact: new Float64Array(2 * N),
    cutMask: new Uint32Array(2 * N),
    failSite: new Float32Array(2 * N),
    field: fl.layers,
    fieldStrength: new Float32Array(2 * fl.count),
    fieldDamage: new Uint8Array(2 * fl.count),
    fieldArea: new Uint8Array(2 * fl.count).fill(255),
    crossingNormal0: new Float64Array(N),
    tails: new Float64Array(4 * N),
  };

  for (let l = 0; l < specs.length; l++) {
    fillLayer(scene, specs[l], layers[l]);
    fillField(scene, specs[l], layers[l], fl.layers[l]);
  }
  // the worn-through yarns' ends have recoiled as they would have when they broke
  while (pullOut(fl.layers, scene.fieldArea, scene.fieldDamage) > 0);
  // span state from the initial field; spans that lost every yarn start separated
  const active = (q: number) => (scene.segFlags[q] & (SEG_EXISTS | SEG_BROKEN)) === SEG_EXISTS;
  const before = Uint16Array.from(scene.segFlags);
  coupleSpans(layers, fl.layers, scene.fieldArea, scene.fieldDamage, scene, active, SEG_BROKEN);
  for (let q = 0; q < scene.segFlags.length; q++) if ((scene.segFlags[q] & SEG_BROKEN) && !(before[q] & SEG_BROKEN)) scene.segFlags[q] |= SEG_INITIAL_DEFECT;
  return scene;
}

/** Rest-plane world XY of material point (u, v) under the undeformed rotation (no mount compression). */
export function restPlanePoint(spec: LayerSpec, info: LayerInfo, u: number, v: number): [number, number] {
  const c = Math.cos(spec.angle);
  const s = Math.sin(spec.angle);
  const du = u - info.restCenter[0];
  const dv = v - info.restCenter[1];
  return [c * du - s * dv, s * du + c * dv];
}

function fillLayer(scene: CompiledScene, spec: LayerSpec, info: LayerInfo): void {
  const m = info.material;
  const { nx, ny, nodeOffset, mapA, mapOrigin, restCenter } = info;
  const mass = crossingMass(m);
  const seed = spec.seed;
  const yarnU = m.yarns[FAMILY_U];
  const yarnV = m.yarns[FAMILY_V];
  const cr = spec.collarRows;

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const n = nodeOffset + j * nx + i;
      const u = i * m.pitchU;
      const v = j * m.pitchV;
      scene.restUV[2 * n] = u;
      scene.restUV[2 * n + 1] = v;
      const du = u - restCenter[0];
      const dv = v - restCenter[1];
      const px = mapOrigin[0] + mapA[0] * du + mapA[1] * dv;
      const py = mapOrigin[1] + mapA[2] * du + mapA[3] * dv;
      const pinned = i < cr || j < cr || i >= nx - cr || j >= ny - cr;
      // Smooth out-of-plane seed for buckling. It is one world-space field shared by every layer (same
      // seed, radial fade) so the stacked plies start parallel at their contact separation.
      const fadeIn = 1 - smoothstep(0.16, 0.24, Math.hypot(px, py));
      const pz = pinned ? 0 : spec.perturbation * fadeIn * fbm(DRAPE_SEED, px / 0.04, py / 0.04, 3);
      scene.position[3 * n] = px;
      scene.position[3 * n + 1] = py;
      scene.position[3 * n + 2] = mapOrigin[2] + pz;
      scene.mass[n] = mass;
      scene.pinned[n] = pinned ? 1 : 0;

      const [rx, ry] = restPlanePoint(spec, info, u, v);
      const [w, wVisible] = wearAt(spec.wear, seed, rx, ry);
      scene.wear[n] = wVisible;
      scene.structuralWear[n] = w;
      scene.crossingNormal0[n] = m.crossingNormal0 * (1 - 0.7 * w);

      for (let f = 0; f < 2; f++) {
        const sIdx = 2 * n + f;
        const exists = f === FAMILY_U ? i < nx - 1 : j < ny - 1;
        if (!exists) continue;
        const yarn = f === FAMILY_U ? yarnU : yarnV;
        scene.segFlags[sIdx] = SEG_EXISTS;
        scene.restLength[sIdx] = f === FAMILY_U ? m.pitchU : m.pitchV;
        scene.intact[sIdx] = 1;
        // Worn yarns lose their grip in a loosened weave: past the peak they slide out of it over
        // millimetres while still carrying a falling load (pull-out) instead of snapping, so damaged
        // cloth thins, stretches and bridges an opening before it lets go. Intact cloth stays brittle.
        scene.softening[sIdx] = yarn.softeningOpening * (1 + 2.5 * w) + 0.6e-3 * Math.pow(w, 1.5);
        scene.failSite[sIdx] = 0.5;
      }
    }
  }
}

/** Grid step (m) of the wear and scatter fields sampled by the yarn field. */
const FIELD_GRID = 1.0e-3;

/**
 * Per-yarn peak tensions and initial damage of the layer's crossings (yarnField.ts). Strength falls
 * with structural wear (warps more: abrasion acts on the warp floats of a warp-faced twill), varies
 * with a spatially correlated scatter (abrasion and yarn irregularity vary smoothly over 5-15 mm, so a
 * failure front wanders along weak valleys instead of breaking yarns at random) and along each yarn
 * (slubs, thin places over a few mm). In the most worn cloth short runs of warp are worn through.
 */
function fillField(scene: CompiledScene, spec: LayerSpec, info: LayerInfo, fl: FieldLayer): void {
  if (fl.bu > MAX_BUNDLE || fl.bv > MAX_BUNDLE) throw new Error('bundle too large for the cut masks');
  const weave = info.weave;
  const seed = spec.seed;
  const pu = weave.pitchU;
  const pv = weave.pitchV;
  // wear and correlated scatter on a regular grid over the layer's rest chart (bilinear lookups)
  const u0 = -fl.bu * pu;
  const v0 = -fl.bv * pv;
  const gw = Math.ceil((fl.K + 2 * fl.bu) * pu / FIELD_GRID) + 2;
  const gh = Math.ceil((fl.M + 2 * fl.bv) * pv / FIELD_GRID) + 2;
  const gWear = new Float32Array(gw * gh);
  const gCorr = new Float32Array(gw * gh);
  const gEdgeNoise = new Float32Array(gw * gh);
  for (let y = 0; y < gh; y++) {
    for (let x = 0; x < gw; x++) {
      const [rx, ry] = restPlanePoint(spec, info, u0 + x * FIELD_GRID, v0 + y * FIELD_GRID);
      gWear[y * gw + x] = wearAt(spec.wear, seed, rx, ry)[0];
      gCorr[y * gw + x] = fbm(seed + 503, rx / 0.011, ry / 0.011, 3);
      gEdgeNoise[y * gw + x] = fbm(seed + 991, rx / 0.004, ry / 0.004, 2);
    }
  }
  const sample = (g: Float32Array, u: number, v: number) => {
    const fx = Math.min(gw - 1.001, Math.max(0, (u - u0) / FIELD_GRID));
    const fy = Math.min(gh - 1.001, Math.max(0, (v - v0) / FIELD_GRID));
    const x = Math.floor(fx);
    const y = Math.floor(fy);
    const tx = fx - x;
    const ty = fy - y;
    const a = g[y * gw + x] + (g[y * gw + x + 1] - g[y * gw + x]) * tx;
    const b = g[(y + 1) * gw + x] + (g[(y + 1) * gw + x + 1] - g[(y + 1) * gw + x]) * tx;
    return a + (b - a) * ty;
  };
  const thr = spec.wear.defectThreshold;
  for (let mm = 0; mm < fl.M; mm++) {
    const v = (mm - (fl.bv - 1) / 2) * pv;
    for (let k = 0; k < fl.K; k++) {
      const u = (k - (fl.bu - 1) / 2) * pu;
      const c = fl.offset + mm * fl.K + k;
      const w = sample(gWear, u, v);
      const corr = sample(gCorr, u, v);
      for (let f = 0; f < 2; f++) {
        const loss = spec.wearStrengthLoss[f];
        const wf = f === FAMILY_U ? Math.pow(w, 1.3) : Math.pow(w, 1.25);
        const yarnIndex = f === FAMILY_V ? k : mm;
        const along = f === FAMILY_V ? v : u;
        const slub = 1 + 0.25 * noise1(seed + 4099 + 13 * yarnIndex + 7 * f, along / 0.0025);
        const z = (rand01(seed + 503, c, f) + rand01(seed + 617, c, f) - 1) * 2.45;
        const scatter = Math.exp((0.2 + 0.12 * w) * corr + (0.06 + 0.22 * w) * z);
        // abrasion removes fibres: cross-section, stiffness and strength fall together, so a worn yarn
        // still fails at about the strain of a sound one. It acts on the warp floats of the face (the
        // weft lies beneath them), so worn warp loses far more than worn weft: a worn cell gives way
        // in its warps first and leaves its pale weft bridging the opening. The loosened weave turns
        // further before jamming (laws.ts shearJam) and its yarns slip through their crossings.
        // Scatter and slubs are local weak or strong places along the yarn
        const area = Math.max(0.02, 1 - loss * wf);
        scene.fieldArea[2 * c + f] = areaCode(area);
        scene.fieldStrength[2 * c + f] = weave.yarns[f].strength * Math.max(0.005, area * scatter * slub);
      }
      // the most worn cores have lost their warp floats altogether: small rounded ladder openings
      // whose picks are exposed before any pull; their outline wanders with the wear and per warp
      const edge = thr + 0.04 * sample(gEdgeNoise, u, v) + 0.012 * (rand01(seed + 990, k) - 0.5);
      if (w > edge && rand01(seed + 991, k, mm) < spec.wear.defectDensity) scene.fieldDamage[2 * c + FAMILY_V] = DAMAGE_CUT;
      // around them, single warp floats are worn through: abrasion takes the float (the warp's run
      // over twillWarpOver picks on the face), exposing the pale pick beneath; ever more of them
      // toward the core
      const R = weave.twillRepeat;
      const float = Math.floor((mm - (k % R) + 4 * R) / R);
      const pf = 0.02 * Math.pow(smoothstep(spec.wear.sparseFrom, thr, w), 2);
      const onFace = (((mm - k) % R) + R) % R < weave.twillWarpOver;
      if (pf > 0 && onFace && rand01(seed + 997, k, float) < pf) scene.fieldDamage[2 * c + FAMILY_V] = DAMAGE_CUT;
    }
  }
}

/**
 * The authored coupon: one warp-faced denim layer mounted at its border, at the reference's scale
 * (about 35 cm of cloth across the view height: its twill wales are ~5.5 px apart on 1624 px native
 * frames, i.e. 1.3 mm wales at ~0.23 mm per pixel). Mechanical spans bundle 9 warps x 6 picks
 * (near-square 3.4 x 3.5 mm cells) that fail in strands of 3 warps / 2 picks. Warps run close to
 * vertical in view, which puts the wales at ~57 degrees rising to the right as in the reference. The coupon
 * extends beyond the supported framing on every side so no mount is ever visible.
 */
export function defaultSceneSpecs(): LayerSpec[] {
  return [
    {
      material: DENIM,
      bundle: [9, 6],
      sizeU: 0.62,
      sizeV: 0.4,
      z: 0,
      angle: 0.06,
      compressU: 0.025,
      compressV: 0.015,
      shear: 0.06,
      collarRows: 2,
      wear: {
        center: [0.01, -0.006],
        axis: [1, -1.15],
        radii: [0.2, 0.14],
        amount: 1,
        cellSize: 0.075,
        featureScale: 0.05,
        defectThreshold: 0.89,
        defectDensity: 1,
        sparseFrom: 0.6,
      },
      wearStrengthLoss: [0.45, 0.9],
      seed: 1001,
      perturbation: 2.5e-3,
    },
  ];
}
