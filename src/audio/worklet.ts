/**
 * Audio-thread side of the tear sound: the main thread hands over the prepared recording, the engine
 * worker posts acoustic records (record.ts) and grip changes straight to this processor over a
 * MessagePort; the tear voice renders them (stereo output, audio.ts).
 */
import { TearVoice, type TearSource } from './tearVoice';

// AudioWorkletGlobalScope (not part of the DOM library typings)
declare const sampleRate: number;
declare class AudioWorkletProcessor {
  readonly port: MessagePort;
}
declare function registerProcessor(name: string, ctor: new () => AudioWorkletProcessor & { process(inputs: Float32Array[][], outputs: Float32Array[][]): boolean }): void;

class TearProcessor extends AudioWorkletProcessor {
  private readonly voice = new TearVoice(sampleRate, (Math.random() * 4294967296) >>> 0, false);

  constructor() {
    super();
    // the node's own port hands over the engine's port once; on it come acoustic records and, in
    // order with them, each finger's grab and release
    this.port.onmessage = (e: MessageEvent<{ records: MessagePort } | { source: TearSource }>) => {
      if ('source' in e.data) {
        this.voice.setSource(e.data.source);
        return;
      }
      e.data.records.onmessage = (m: MessageEvent<Float32Array | { grip: number; held: boolean }>) => {
        const d = m.data;
        if (d instanceof Float32Array) this.voice.push(d);
        else this.voice.grip(d.grip, d.held);
      };
    };
  }

  process(_inputs: Float32Array[][], outputs: Float32Array[][]): boolean {
    const out = outputs[0];
    if (out.length === 2) this.voice.process(out[0], out[1]);
    return true;
  }
}

registerProcessor('tear-voice', TearProcessor);
