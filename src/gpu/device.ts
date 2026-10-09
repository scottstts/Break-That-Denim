/** WebGPU device acquisition with explicit, validated limits. */

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  timestamps: boolean;
}

export class GpuUnavailableError extends Error {}

/** Limits the engine relies on; requested explicitly and checked against the adapter. */
const REQUIRED_LIMITS: Partial<Record<keyof GPUSupportedLimits, number>> = {
  maxStorageBuffersPerShaderStage: 10,
  maxComputeWorkgroupStorageSize: 32768,
  maxComputeInvocationsPerWorkgroup: 256,
  maxComputeWorkgroupSizeX: 256,
  maxStorageBufferBindingSize: 256 * 1024 * 1024,
  maxBufferSize: 512 * 1024 * 1024,
  maxComputeWorkgroupsPerDimension: 65535,
};

export async function acquireGpu(gpu: GPU | undefined): Promise<GpuContext> {
  if (!gpu) throw new GpuUnavailableError('WebGPU is not available in this browser.');
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new GpuUnavailableError('No WebGPU adapter.');
  const requiredLimits: Record<string, number> = {};
  for (const [k, v] of Object.entries(REQUIRED_LIMITS)) {
    const have = adapter.limits[k as keyof GPUSupportedLimits] as number;
    if (have < (v as number)) throw new GpuUnavailableError(`GPU limit ${k}=${have} is below the required ${v}.`);
    requiredLimits[k] = v as number;
  }
  const timestamps = adapter.features.has('timestamp-query');
  const device = await adapter.requestDevice({
    requiredLimits,
    requiredFeatures: timestamps ? ['timestamp-query'] : [],
  });
  return { adapter, device, timestamps };
}
