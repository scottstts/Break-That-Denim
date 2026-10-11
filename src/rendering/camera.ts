import { invert, lookAt, multiply, perspective, type Mat4, type Vec3 } from './math';

export interface CameraState {
  target: Vec3;
  distance: number;
  /** Tilt from the cloth normal toward -y (rad). */
  pitch: number;
  /** Rotation about the cloth normal (rad). */
  yaw: number;
  fovY: number;
  /** A staged camera's own placement: the eye (replacing distance, pitch and yaw) and up direction. */
  eye?: Vec3;
  up?: Vec3;
}

/** About 35 cm of cloth across the view height: the projected weave density of the reference. */
export const DEFAULT_CAMERA: CameraState = {
  target: [0.0, 0.0, -0.01],
  distance: 0.98,
  pitch: 0.16,
  yaw: 0,
  fovY: (20 * Math.PI) / 180,
};

export interface ViewMatrices {
  eye: Vec3;
  view: Mat4;
  proj: Mat4;
  viewProj: Mat4;
  invViewProj: Mat4;
}

export function cameraMatrices(c: CameraState, aspect: number): ViewMatrices {
  const dir: Vec3 = [Math.sin(c.yaw) * Math.sin(c.pitch), -Math.cos(c.yaw) * Math.sin(c.pitch), Math.cos(c.pitch)];
  const eye: Vec3 = c.eye ?? [c.target[0] + dir[0] * c.distance, c.target[1] + dir[1] * c.distance, c.target[2] + dir[2] * c.distance];
  const up: Vec3 = c.up ?? [-Math.sin(c.yaw), Math.cos(c.yaw), 0];
  const view = lookAt(eye, c.target, up);
  const proj = perspective(c.fovY, aspect, 0.05, 3.0);
  const viewProj = multiply(proj, view);
  return { eye, view, proj, viewProj, invViewProj: invert(viewProj) };
}
