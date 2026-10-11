import workletUrl from './worklet.ts?worker&url';
import type { SoundCue } from '../shared/messages';
import { glitchNoise } from './glitch';
import { prepareSource } from './tearVoice';

const recordingUrl = `${import.meta.env.BASE_URL}fabric_tear_sound_sample.mp3`;
const laughUrl = `${import.meta.env.BASE_URL}exprerience_sounds/creepy_laugh.mp3`;
/** Fades at the edges of a signal-noise burst (s). */
const EDGE = 0.015;

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
  /** Staged sounds: the laugh (fetched when an experience is in view, decoded once audio runs) and what plays. */
  private laughBytes: Promise<ArrayBuffer | null> | null = null;
  private laugh: Promise<AudioBuffer | null> | null = null;
  private readonly playing = new Set<AudioScheduledSourceNode>();

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

  /** Fetch the staged sounds ahead of use (an experience is about to be chosen); decoded once audio runs. */
  prefetchStaged(): void {
    this.laughBytes ??= fetch(laughUrl)
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .catch(() => null);
    if (this.context) void this.laughBuffer(this.context);
  }

  private laughBuffer(context: AudioContext): Promise<AudioBuffer | null> {
    this.laughBytes ??= fetch(laughUrl)
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .catch(() => null);
    this.laugh ??= this.laughBytes.then((b) => (b ? context.decodeAudioData(b) : null)).catch(() => null);
    return this.laugh;
  }

  /** Play (or stop) a staged experience's sound now. */
  cue(c: SoundCue): void {
    const context = this.context;
    if (!context) return;
    if (c.kind === 'stop') {
      const t = context.currentTime;
      for (const s of this.playing) {
        try {
          s.stop(t + 0.05);
        } catch {
          // already stopped
        }
      }
      this.playing.clear();
      return;
    }
    if (c.kind === 'laugh') {
      // the recording as it is: whole, at its own level
      void this.laughBuffer(context).then((buffer) => {
        if (!buffer) return;
        const src = context.createBufferSource();
        src.buffer = buffer;
        src.connect(context.destination);
        src.start();
        this.track(src);
      });
      return;
    }
    const n = Math.max(1, Math.round(c.duration * context.sampleRate));
    const buffer = context.createBuffer(2, n, context.sampleRate);
    glitchNoise(context.sampleRate, c.level, buffer.getChannelData(0), buffer.getChannelData(1));
    this.play(context, buffer, 0, c.duration, 1);
  }

  private play(context: AudioContext, buffer: AudioBuffer, offset: number, duration: number, level: number): void {
    const t = context.currentTime;
    const src = context.createBufferSource();
    src.buffer = buffer;
    const gain = context.createGain();
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(level, t + EDGE);
    gain.gain.setValueAtTime(level, t + duration - EDGE);
    gain.gain.linearRampToValueAtTime(0, t + duration);
    src.connect(gain).connect(context.destination);
    src.start(t, offset, duration);
    this.track(src);
  }

  private track(src: AudioScheduledSourceNode): void {
    this.playing.add(src);
    src.onended = () => this.playing.delete(src);
  }

  /** Pause output while the page is hidden (the engine stops simulating then too). */
  setVisible(visible: boolean): void {
    if (!this.context) return;
    if (visible) void this.context.resume().catch(() => {});
    else void this.context.suspend().catch(() => {});
  }
}

