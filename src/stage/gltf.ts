/**
 * Minimal binary glTF (GLB) reader for one skinned mesh: node hierarchy, the skin, the mesh's first
 * primitive and its material's embedded images. Only what the staged creature needs.
 */

export interface GltfNode {
  name: string;
  parent: number;
  children: number[];
  translation: [number, number, number];
  rotation: [number, number, number, number];
  scale: [number, number, number];
  /** Column-major local matrix when the node gives one instead of TRS. */
  matrix: Float64Array | null;
}

export interface GltfImage {
  bytes: Uint8Array;
  mime: string;
}

export interface GltfSkinnedMesh {
  nodes: GltfNode[];
  /** Root nodes of the default scene. */
  roots: number[];
  /** Node index of each skin joint, in skin order. */
  joints: number[];
  /** Column-major inverse bind matrices, 16 per joint. */
  inverseBind: Float32Array;
  position: Float32Array;
  normal: Float32Array;
  /** xyz tangent, w handedness. */
  tangent: Float32Array;
  uv: Float32Array;
  /** Four skin joints (skin order) and weights per vertex. */
  jointIndex: Uint16Array;
  weight: Float32Array;
  index: Uint32Array;
  baseColor: GltfImage | null;
  normalMap: GltfImage | null;
  /** Occlusion (r), roughness (g), metallic (b). */
  orm: GltfImage | null;
}

interface Accessor {
  bufferView: number;
  byteOffset?: number;
  componentType: number;
  count: number;
  type: string;
  normalized?: boolean;
}
interface BufferView {
  byteOffset?: number;
  byteLength: number;
  byteStride?: number;
}
interface Json {
  scene?: number;
  scenes: { nodes: number[] }[];
  nodes: { name?: string; children?: number[]; translation?: number[]; rotation?: number[]; scale?: number[]; matrix?: number[]; mesh?: number; skin?: number }[];
  meshes: { primitives: { attributes: Record<string, number>; indices: number; material?: number }[] }[];
  skins: { joints: number[]; inverseBindMatrices: number }[];
  accessors: Accessor[];
  bufferViews: BufferView[];
  materials?: {
    normalTexture?: { index: number };
    occlusionTexture?: { index: number };
    pbrMetallicRoughness?: { baseColorTexture?: { index: number }; metallicRoughnessTexture?: { index: number } };
  }[];
  textures?: { source: number }[];
  images?: { bufferView: number; mimeType: string }[];
}

const COMPONENTS: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

export function parseGlb(data: ArrayBuffer): GltfSkinnedMesh {
  const dv = new DataView(data);
  if (dv.getUint32(0, true) !== 0x46546c67) throw new Error('not a GLB file');
  const jsonLength = dv.getUint32(12, true);
  const json = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 20, jsonLength))) as Json;
  const binStart = 20 + jsonLength + 8;
  const bin = new Uint8Array(data, binStart, dv.getUint32(20 + jsonLength, true));

  const read = (index: number): Float32Array | Uint16Array | Uint32Array => {
    const a = json.accessors[index];
    const view = json.bufferViews[a.bufferView];
    const n = COMPONENTS[a.type];
    const size = a.componentType === 5126 || a.componentType === 5125 ? 4 : a.componentType === 5123 ? 2 : 1;
    const stride = view.byteStride ?? n * size;
    const base = bin.byteOffset + (view.byteOffset ?? 0) + (a.byteOffset ?? 0);
    const src = new DataView(bin.buffer);
    const out = a.componentType === 5126 ? new Float32Array(a.count * n) : a.componentType === 5125 ? new Uint32Array(a.count * n) : new Uint16Array(a.count * n);
    for (let k = 0; k < a.count; k++) {
      for (let c = 0; c < n; c++) {
        const at = base + k * stride + c * size;
        if (a.componentType === 5126) out[k * n + c] = src.getFloat32(at, true);
        else if (a.componentType === 5125) out[k * n + c] = src.getUint32(at, true);
        else if (a.componentType === 5123) out[k * n + c] = src.getUint16(at, true);
        else out[k * n + c] = src.getUint8(at);
      }
    }
    return out;
  };

  const nodes: GltfNode[] = json.nodes.map((n) => ({
    name: n.name ?? '',
    parent: -1,
    children: n.children ?? [],
    translation: (n.translation ?? [0, 0, 0]) as [number, number, number],
    rotation: (n.rotation ?? [0, 0, 0, 1]) as [number, number, number, number],
    scale: (n.scale ?? [1, 1, 1]) as [number, number, number],
    matrix: n.matrix ? Float64Array.from(n.matrix) : null,
  }));
  nodes.forEach((n, i) => n.children.forEach((c) => (nodes[c].parent = i)));

  const meshNode = json.nodes.find((n) => n.mesh !== undefined && n.skin !== undefined);
  if (!meshNode) throw new Error('no skinned mesh');
  const prim = json.meshes[meshNode.mesh!].primitives[0];
  const skin = json.skins[meshNode.skin!];
  const at = prim.attributes;
  const vertexCount = json.accessors[at.POSITION].count;
  const weightsRaw = read(at.WEIGHTS_0);
  const weight = weightsRaw instanceof Float32Array ? weightsRaw : Float32Array.from(weightsRaw, (v) => v / (json.accessors[at.WEIGHTS_0].componentType === 5121 ? 255 : 65535));
  const jointsRaw = read(at.JOINTS_0);

  const image = (texture: number | undefined): GltfImage | null => {
    if (texture === undefined || !json.textures || !json.images) return null;
    const img = json.images[json.textures[texture].source];
    const view = json.bufferViews[img.bufferView];
    return { bytes: new Uint8Array(bin.buffer, bin.byteOffset + (view.byteOffset ?? 0), view.byteLength), mime: img.mimeType };
  };
  const mat = prim.material !== undefined ? json.materials?.[prim.material] : undefined;

  return {
    nodes,
    roots: json.scenes[json.scene ?? 0].nodes,
    joints: skin.joints,
    inverseBind: read(skin.inverseBindMatrices) as Float32Array,
    position: read(at.POSITION) as Float32Array,
    normal: read(at.NORMAL) as Float32Array,
    tangent: at.TANGENT !== undefined ? (read(at.TANGENT) as Float32Array) : new Float32Array(4 * vertexCount),
    uv: read(at.TEXCOORD_0) as Float32Array,
    jointIndex: jointsRaw instanceof Uint16Array ? jointsRaw : Uint16Array.from(jointsRaw),
    weight,
    index: Uint32Array.from(read(prim.indices)),
    baseColor: image(mat?.pbrMetallicRoughness?.baseColorTexture?.index),
    normalMap: image(mat?.normalTexture?.index),
    orm: image(mat?.pbrMetallicRoughness?.metallicRoughnessTexture?.index ?? mat?.occlusionTexture?.index),
  };
}
