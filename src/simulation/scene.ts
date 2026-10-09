import { BACKING, DENIM, FAMILY_U, FAMILY_V, crossingMass, type WeaveMaterial } from './material';
import { fbm, noise1, rand01, smoothstep } from '../shared/noise';

/**
 * Scene asset compilation: three independent woven lattices with explicit identities, mounts,
 * correlated wear, per-segment cohesive strength, and pre-existing defects.
 *
 * Indexing (per layer, local): node n = j * nx + i, i = v-yarn (warp end) index, j = u-yarn (pick) index.
 * Global node index = layer.nodeOffset + n. Each node owns segment 2*n + 0 (to (i+1, j), u-family)
 * and 2*n + 1 (to (i, j+1), v-family). Tails are per node and side: [u-, u+, v-, v+].
 */

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
  /** Wear above which pre-existing broken v-yarn (warp) spans may appear. */
  defectThreshold: number;
  /** Fraction of eligible spans that start broken. */
  defectDensity: number;
}

export interface LayerSpec {
  material: WeaveMaterial;
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
  material: WeaveMaterial;
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
  /** Abrasion/wear per node, 0..1 (drives optics and fuzz). */
  wear: Float32Array;
  restLength: Float64Array;
  peakTension: Float64Array;
  softening: Float64Array;
  maxExt: Float64Array;
  segFlags: Uint8Array;
  /** Position of the failure site within the span (fraction from the owning node). */
  failSite: Float32Array;
  /** Per-node crossing normal force at zero tension (wear-reduced), N. */
  crossingNormal0: Float64Array;
  tails: Float64Array;
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

/** Wear in [0, 1] at world-rest-plane point (x, y). */
export function wearAt(spec: WearSpec, seed: number, x: number, y: number): number {
  if (spec.amount <= 0) return 0;
  const al = Math.hypot(spec.axis[0], spec.axis[1]) || 1;
  const ax = spec.axis[0] / al;
  const ay = spec.axis[1] / al;
  const dx = x - spec.center[0];
  const dy = y - spec.center[1];
  const a = (dx * ax + dy * ay) / spec.radii[0];
  const b = (-dx * ay + dy * ax) / spec.radii[1];
  const warp = 0.32 * fbm(seed + 3, x / 0.014, y / 0.014, 4);
  const r = Math.hypot(a, b) + warp;
  let w = 1 - smoothstep(0.25, 1.05, r);
  if (spec.cellSize > 0) {
    const jx = x + 0.004 * fbm(seed + 11, x / 0.01, y / 0.01, 3);
    const jy = y + 0.004 * fbm(seed + 12, x / 0.01, y / 0.01, 3);
    const [f1, f2] = worley(seed + 5, jx / spec.cellSize, jy / spec.cellSize);
    // 1 at cell interiors, small near the boundaries between neighboring cells (retained ligaments).
    const ligament = smoothstep(0.0, 0.38, f2 - f1);
    w *= 0.5 + 0.5 * ligament;
  }
  // Fine correlated abrasion variation.
  w *= 0.82 + 0.18 * fbm(seed + 7, x / 0.003, y / 0.003, 3);
  return Math.min(1, Math.max(0, w * spec.amount));
}

export function compileLayer(spec: LayerSpec, index: number, nodeOffset: number): { info: LayerInfo } {
  const m = spec.material;
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
    info: { index, material: m, nx, ny, nodeOffset, nodeCount: nx * ny, mapA, mapOrigin, restCenter },
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
  const scene: CompiledScene = {
    version: 1,
    layers,
    nodeCount: N,
    restUV: new Float64Array(2 * N),
    position: new Float64Array(3 * N),
    mass: new Float64Array(N),
    pinned: new Uint8Array(N),
    wear: new Float32Array(N),
    restLength: new Float64Array(2 * N),
    peakTension: new Float64Array(2 * N),
    softening: new Float64Array(2 * N),
    maxExt: new Float64Array(2 * N),
    segFlags: new Uint8Array(2 * N),
    failSite: new Float32Array(2 * N),
    crossingNormal0: new Float64Array(N),
    tails: new Float64Array(4 * N),
  };

  for (let l = 0; l < specs.length; l++) {
    fillLayer(scene, specs[l], layers[l]);
  }
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
  const m = spec.material;
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
      // Smooth perturbation seeds buckling; it vanishes at the collar.
      const edge = Math.min(i, j, nx - 1 - i, ny - 1 - j) - cr;
      const fadeIn = smoothstep(0, 24, edge);
      const pz = pinned ? 0 : spec.perturbation * fadeIn * fbm(seed + 101, px / 0.012, py / 0.012, 3);
      scene.position[3 * n] = px;
      scene.position[3 * n + 1] = py;
      scene.position[3 * n + 2] = mapOrigin[2] + pz;
      scene.mass[n] = mass;
      scene.pinned[n] = pinned ? 1 : 0;

      const [rx, ry] = restPlanePoint(spec, info, u, v);
      const w = wearAt(spec.wear, seed, rx, ry);
      scene.wear[n] = w;
      scene.crossingNormal0[n] = m.crossingNormal0 * (1 - 0.7 * w);

      for (let f = 0; f < 2; f++) {
        const sIdx = 2 * n + f;
        const exists = f === FAMILY_U ? i < nx - 1 : j < ny - 1;
        if (!exists) continue;
        const yarn = f === FAMILY_U ? yarnU : yarnV;
        const yarnIndex = f === FAMILY_U ? j : i;
        const along = f === FAMILY_U ? u : v;
        // Correlated per-yarn variation along physical arc length (slubs / twist irregularity).
        const yarnVar = 0.12 * noise1(seed + 31 * (f + 1) + yarnIndex * 7, along / 0.004);
        const loss = spec.wearStrengthLoss[f];
        const wf = f === FAMILY_U ? Math.pow(w, 1.6) : Math.pow(w, 1.25);
        const factor = Math.max(0.04, (1 - loss * wf) * (1 + yarnVar));
        scene.segFlags[sIdx] = SEG_EXISTS;
        scene.restLength[sIdx] = f === FAMILY_U ? m.pitchU : m.pitchV;
        scene.peakTension[sIdx] = yarn.strength * factor;
        scene.softening[sIdx] = yarn.softeningOpening * (1 + 2.5 * w);
        scene.failSite[sIdx] = 0.25 + 0.5 * rand01(seed + 77, sIdx, f);
        // Authored pre-existing defects: broken warp spans in the most worn material.
        if (f === FAMILY_V && w > spec.wear.defectThreshold) {
          const p = (w - spec.wear.defectThreshold) / (1 - spec.wear.defectThreshold);
          if (rand01(seed + 991, i, j) < spec.wear.defectDensity * p) {
            breakSegmentInitial(scene, info, n, f);
          }
        }
      }
    }
  }
}

/** Break a span at its failure site and split its rest length into two tails (ledger-preserving). */
function breakSegmentInitial(scene: CompiledScene, info: LayerInfo, n: number, f: number): void {
  const sIdx = 2 * n + f;
  const L = scene.restLength[sIdx];
  const phi = scene.failSite[sIdx];
  const other = f === FAMILY_U ? n + 1 : n + info.nx;
  scene.segFlags[sIdx] |= SEG_BROKEN | SEG_INITIAL_DEFECT;
  // Owner node keeps the near part on its '+' side; the far node keeps the rest on its '-' side.
  scene.tails[4 * n + 2 * f + 1] += L * phi;
  scene.tails[4 * other + 2 * f] += L * (1 - phi);
}

/** The authored three-layer coupon (provisional G0 framing). */
export function defaultSceneSpecs(): LayerSpec[] {
  const frontWear: WearSpec = {
    center: [0.003, -0.002],
    axis: [1, -1.15],
    radii: [0.03, 0.019],
    amount: 1,
    cellSize: 0.0135,
    defectThreshold: 0.8,
    defectDensity: 0.55,
  };
  return [
    {
      material: DENIM,
      sizeU: 0.11,
      sizeV: 0.1,
      z: 0,
      angle: -0.49,
      compressU: 0.022,
      compressV: 0.012,
      shear: 0.03,
      collarRows: 2,
      wear: frontWear,
      wearStrengthLoss: [0.55, 0.93],
      seed: 1001,
      perturbation: 0.6e-3,
    },
    {
      material: BACKING,
      sizeU: 0.11,
      sizeV: 0.1,
      z: -0.85e-3,
      angle: 0.3,
      compressU: 0.02,
      compressV: 0.016,
      shear: 0.02,
      collarRows: 2,
      wear: {
        center: [0.006, -0.004],
        axis: [1, -0.6],
        radii: [0.02, 0.014],
        amount: 0.75,
        cellSize: 0,
        defectThreshold: 2,
        defectDensity: 0,
      },
      wearStrengthLoss: [0.65, 0.65],
      seed: 2002,
      perturbation: 0.5e-3,
    },
    {
      material: DENIM,
      sizeU: 0.11,
      sizeV: 0.1,
      z: -1.7e-3,
      angle: -0.62,
      compressU: 0.018,
      compressV: 0.018,
      shear: 0.01,
      collarRows: 2,
      wear: {
        center: [0.008, -0.006],
        axis: [1, 0.3],
        radii: [0.016, 0.012],
        amount: 0.45,
        cellSize: 0,
        defectThreshold: 2,
        defectDensity: 0,
      },
      wearStrengthLoss: [0.4, 0.7],
      seed: 3003,
      perturbation: 0.5e-3,
    },
  ];
}
