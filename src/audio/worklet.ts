/**
 * Audio-thread side of the tear sound: the engine worker posts acoustic records (record.ts) straight
 * to this processor over a MessagePort; the tear voice renders them (stereo output, audio.ts).
 */
import { TearVoice } from './tearVoice';

// AudioWorkletGlobalScope (not part of the DOM library typings)
declare const sampleRate: number;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
}
declare function registerProcessor(name: string, ctor: new () => AudioWorkletProcessor & { process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean }): void;

class TearProcessor extends AudioWorkletProcessor {
  private readonly voice = new TearVoice(sampleRate, (Math.random() * 4294967296) >>> 0);

  constructor() {
    super();
    // the node's own port hands over the engine's port once
    this.port.onmessage = (e: MessageEvent<{ records: MessagePort }>) => {
      e.data.records.onmessage = (m: MessageEvent<Float32Array>) => this.voice.push(m.data);
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const out = outputs[0];
    if (out.length === 2) this.voice.process(out[0], out[1]);
    return true;
  }
}

registerProcessor('tear-voice', TearProcessor);
