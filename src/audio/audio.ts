import workletUrl from './worklet.ts?worker&url';
import recordingUrl from '../../refs/fabric_tear_sound_sample.mp3?url';
import { prepareSource } from './tearVoice';

/**
 * Main-thread setup of the tear sound. Browsers start audio only from a user gesture, so the context
 * is created (or resumed) on the first pointer interaction; once its processor runs, `connect`
 * receives a MessagePort for the engine worker, which then sends its acoustic records directly to the
 * audio thread (the main thread never handles them). The recorded tear that carries the sound is
 * fetched at once, decoded at the context's rate once audio starts, prepared (tearVoice.ts
 * prepareSource) and handed to the processor.
 */
export class TearAudio {
  private context: AudioContext | null = null;
  private starting: Promise<void> | null = null;
  private readonly connect: (port: MessagePort) => void;
  private readonly recording: Promise<ArrayBuffer | null>;

  constructor(connect: (port: MessagePort) => void) {
    this.connect = connect;
    this.recording = fetch(recordingUrl)
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .catch(() => null);
  }

  /** Call from a user gesture (pointer down / up): starts or resumes audio. */
  unlock(): void {
    if (this.context) {
      if (this.context.state !== 'running') void this.context.resume().catch(() => {});
      return;
    }
    if (this.starting || typeof AudioContext === 'undefined' || typeof AudioWorkletNode === 'undefined') return;
    const context = new AudioContext({ latencyHint: 'interactive' });
    this.context = context;
    // resume within the gesture; the module loads asynchronously
    void context.resume().catch(() => {});
    this.starting = context.audioWorklet
      .addModule(workletUrl)
      .then(() => {
        const node = new AudioWorkletNode(context, 'tear-voice', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [2] });
        node.connect(context.destination);
        const channel = new MessageChannel();
        node.port.postMessage({ records: channel.port1 }, [channel.port1]);
        this.connect(channel.port2);
        void this.recording.then(async (bytes) => {
          if (!bytes) throw new Error('recording unavailable');
          const buffer = await context.decodeAudioData(bytes);
          const mono = new Float32Array(buffer.length);
          for (let c = 0; c < buffer.numberOfChannels; c++) {
            const ch = buffer.getChannelData(c);
            for (let i = 0; i < mono.length; i++) mono[i] += ch[i] / buffer.numberOfChannels;
          }
          const source = prepareSource(mono, context.sampleRate);
          node.port.postMessage({ source }, [source.pcm.buffer]);
        }).catch((e: unknown) => console.warn('Tear sound unavailable:', e));
      })
      .catch((e: unknown) => console.warn('Tear sound unavailable:', e));
  }

  /** Pause output while the page is hidden (the engine stops simulating then too). */
  setVisible(visible: boolean): void {
    if (!this.context) return;
    if (visible) void this.context.resume().catch(() => {});
    else void this.context.suspend().catch(() => {});
  }
}
