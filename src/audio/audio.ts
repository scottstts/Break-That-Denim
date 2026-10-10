import workletUrl from './worklet.ts?worker&url';

/**
 * Main-thread setup of the tear sound. Browsers start audio only from a user gesture, so the context
 * is created (or resumed) on the first pointer interaction; once its processor runs, `connect`
 * receives a MessagePort for the engine worker, which then sends its acoustic records directly to the
 * audio thread (the main thread never handles them).
 */
export class TearAudio {
  private context: AudioContext | null = null;
  private starting: Promise<void> | null = null;
  private readonly connect: (port: MessagePort) => void;

  constructor(connect: (port: MessagePort) => void) {
    this.connect = connect;
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
