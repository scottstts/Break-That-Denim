/** Main thread → engine worker. Small ordered messages only; no bulk state crosses this boundary. */
export type ToEngine =
  | { type: 'init'; canvas: OffscreenCanvas; width: number; height: number; cssWidth: number; cssHeight: number }
  | { type: 'resize'; width: number; height: number; cssWidth: number; cssHeight: number }
  | PointerCommand
  | { type: 'wheel'; seq: number; deltaY: number; time: number }
  | { type: 'reset' }
  /** Start a staged experience (restarts it if playing), or end it (null): back to free play. */
  | { type: 'experience'; name: ExperienceName | null }
  /** Fetch an experience's assets ahead of a likely start (the menu opened). */
  | { type: 'prefetch'; name: ExperienceName }
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
  | { type: 'devDone'; id: number; result?: unknown }
  | { type: 'experience'; name: ExperienceName; state: 'loading' | 'playing' | 'ended' | 'failed' }
  /** A sound the staged experience plays now (main-thread audio). */
  | { type: 'cue'; cue: SoundCue };

/** Staged experiences (src/stage). */
export type ExperienceName = 'beneath';

export type SoundCue =
  /** The laugh recording, whole. */
  | { kind: 'laugh' }
  /** Signal-noise burst of the camera breaking up, `duration` s at `level` (0..1). */
  | { kind: 'glitch'; duration: number; level: number }
  /** Stop every staged sound (the experience ended or was interrupted). */
  | { kind: 'stop' };

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
