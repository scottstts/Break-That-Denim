/** Main thread → engine worker. Small ordered messages only; no bulk state crosses this boundary. */
export type ToEngine =
  | { type: 'init'; canvas: OffscreenCanvas; width: number; height: number; cssWidth: number; cssHeight: number }
  | { type: 'resize'; width: number; height: number; cssWidth: number; cssHeight: number }
  | PointerCommand
  | { type: 'wheel'; seq: number; deltaY: number; time: number }
  | { type: 'reset' }
  | { type: 'visibility'; visible: boolean }
  /** Port to the audio thread's tear voice (audio/worklet.ts): acoustic records go there. */
  | { type: 'audio'; port: MessagePort }
  | DevCommand;

/** Development-only commands (never sent by the product UI). */
export type DevCommand =
  | { type: 'dev'; op: 'step'; substeps: number; id: number }
  | { type: 'dev'; op: 'camera'; distance?: number; pitch?: number; yaw?: number; target?: [number, number, number]; id: number };

/** Pointer coordinates are in CSS pixels relative to the canvas, with the CSS size alongside. */
export interface PointerCommand {
  type: 'pointer';
  phase: 'down' | 'move' | 'up' | 'cancel';
  /** Pointer (finger) identity; ALL_POINTERS with 'cancel' releases every grip. */
  id: number;
  seq: number;
  /** Event time on the shared clock: performance.timeOrigin + performance.now() at the event (ms). */
  time: number;
  x: number;
  y: number;
}

/** Engine worker → main thread. */
export type FromEngine =
  | { type: 'ready' }
  | { type: 'status'; state: 'loading' | 'running' | 'paused' | 'error'; message?: string }
  /** Loading progress by stage, 0..1. */
  | { type: 'progress'; value: number }
  | { type: 'cursor'; cursor: 'grab' | 'grabbing' | 'default' }
  | { type: 'stats'; stats: EngineStats }
  | { type: 'devDone'; id: number; result?: unknown };

export interface EngineStats {
  frameMs: number;
  substepsPerFrame: number;
  simWallRatio: number;
  gpuMsPerSubstep: number;
  gpuRenderMs: number;
  brokenSegments: number;
  /** Simulation is asleep (cloth at rest, no interaction). */
  asleep: boolean;
}

/** PointerCommand.id addressing every pointer. */
export const ALL_POINTERS = -1;
