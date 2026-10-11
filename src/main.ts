import './style.css';
import { TearAudio } from './audio/audio';
import { computeDrawingBufferSize } from './shared/drawingBuffer';
import { ALL_POINTERS, type ExperienceName, type FromEngine, type ToEngine } from './shared/messages';

const canvas = document.createElement('canvas');
canvas.className = 'stage';
document.body.prepend(canvas);

const status = document.querySelector<HTMLElement>('.status')!;
const reset = document.querySelector<HTMLButtonElement>('.reset')!;
const loader = document.querySelector<HTMLElement>('.loader')!;
const menu = document.querySelector<HTMLButtonElement>('.menu')!;
const list = document.querySelector<HTMLElement>('.experiences')!;
const items = [...list.querySelectorAll<HTMLButtonElement>('.experience')];

// the wordmark appears once its typefaces are in (no fallback flash), or with the fallback after a wait
void Promise.race([
  Promise.all([document.fonts.load('400 48px Anton'), document.fonts.load('600 12px Oswald')]),
  new Promise((r) => setTimeout(r, 2500)),
])
  .catch(() => {})
  .then(() => document.body.classList.add('fonts'));

function fail(message: string): void {
  status.textContent = message;
  status.classList.add('visible');
  document.body.classList.add('failed');
}

if (!('transferControlToOffscreen' in canvas) || !navigator.gpu) {
  fail('This experience needs a browser with WebGPU.');
  throw new Error('WebGPU or OffscreenCanvas unavailable');
}

const worker = new Worker(new URL('./engine/worker.ts', import.meta.url), { type: 'module' });
const post = (m: ToEngine, transfer: Transferable[] = []) => worker.postMessage(m, transfer);

// tear sound: the engine sends its acoustic records straight to the audio thread
const audio = new TearAudio((port) => post({ type: 'audio', port }, [port]));

function size() {
  const cssWidth = Math.max(1, window.innerWidth);
  const cssHeight = Math.max(1, window.innerHeight);
  const { width, height } = computeDrawingBufferSize(cssWidth, cssHeight, window.devicePixelRatio);
  return { width, height, cssWidth, cssHeight };
}

const offscreen = canvas.transferControlToOffscreen();
post({ type: 'init', canvas: offscreen, ...size() }, [offscreen]);

let resizeQueued = false;
function onResize() {
  if (resizeQueued) return;
  resizeQueued = true;
  requestAnimationFrame(() => {
    resizeQueued = false;
    post({ type: 'resize', ...size() });
  });
}
window.addEventListener('resize', onResize);
matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener('change', onResize);

// every pointer (mouse, pen, each finger) drives its own grip; event times go on the clock both
// threads share (performance.timeOrigin + their own performance.now())
let seq = 0;
const clock = (t: number) => performance.timeOrigin + t;
function pointer(phase: 'down' | 'move' | 'up' | 'cancel', e: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  post({ type: 'pointer', phase, id: e.pointerId, seq: seq++, time: clock(e.timeStamp), x: e.clientX - r.left, y: e.clientY - r.top });
}
function cancelAll(time: number) {
  post({ type: 'pointer', phase: 'cancel', id: ALL_POINTERS, seq: seq++, time: clock(time), x: 0, y: 0 });
}
canvas.addEventListener('pointerdown', (e) => {
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  audio.unlock();
  canvas.setPointerCapture(e.pointerId);
  pointer('down', e);
});
canvas.addEventListener('pointermove', (e) => {
  // the engine follows the pointer's path at substep resolution: coalesced samples carry it
  const samples = e.getCoalescedEvents?.() ?? [];
  if (samples.length === 0) pointer('move', e);
  for (const s of samples) pointer('move', s);
});
canvas.addEventListener('pointerup', (e) => {
  // iOS counts the end of a touch, not its start, as the gesture that may start audio
  audio.unlock();
  pointer('up', e);
});
canvas.addEventListener('pointercancel', (e) => pointer('cancel', e));
canvas.addEventListener('lostpointercapture', (e) => pointer('cancel', e));
canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    post({ type: 'wheel', seq: seq++, deltaY: e.deltaY, time: clock(e.timeStamp) });
  },
  { passive: false },
);
window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (menuOpen()) {
    openMenu(false);
    menu.focus();
  }
  cancelAll(e.timeStamp);
});
window.addEventListener('blur', () => cancelAll(performance.now()));
document.addEventListener('visibilitychange', () => {
  const visible = document.visibilityState === 'visible';
  post({ type: 'visibility', visible });
  audio.setVisible(visible);
});
// staged experiences: the menu lists them; choosing one plays it on intact cloth (the reset button
// ends it); assets are fetched as soon as the menu opens
const menuOpen = () => !list.hasAttribute('hidden');
function openMenu(open: boolean) {
  list.hidden = !open;
  menu.setAttribute('aria-expanded', String(open));
  if (!open) return;
  audio.prefetchStaged();
  for (const item of items) post({ type: 'prefetch', name: item.dataset.experience as ExperienceName });
}
menu.addEventListener('click', () => {
  audio.unlock();
  openMenu(!menuOpen());
});
for (const item of items) {
  item.addEventListener('click', () => {
    audio.unlock();
    audio.prefetchStaged();
    post({ type: 'experience', name: item.dataset.experience as ExperienceName });
    openMenu(false);
  });
}
document.addEventListener('pointerdown', (e) => {
  if (menuOpen() && !(e.target instanceof Node && (list.contains(e.target) || menu.contains(e.target)))) openMenu(false);
});
reset.addEventListener('click', () => {
  audio.unlock();
  openMenu(false);
  post({ type: 'reset' });
  reset.classList.remove('spin');
  void reset.offsetWidth;
  reset.classList.add('spin');
});

const pendingDev = new Map<number, (r: unknown) => void>();
let markReady: () => void = () => {};
const engineReady = new Promise<void>((r) => (markReady = r));
let devId = 1;
if (import.meta.env.DEV) {
  // Development driver (not part of the product): step physics, move the camera.
  const call = (cmd: Record<string, unknown>) =>
    new Promise((resolve) => {
      const id = devId++;
      pendingDev.set(id, resolve);
      post({ type: 'dev', id, ...cmd } as ToEngine);
    });
  (window as unknown as { __fabric: unknown }).__fabric = {
    ready: () => engineReady,
    step: (substeps: number) => call({ op: 'step', substeps }),
    camera: (c: Record<string, unknown>) => call({ op: 'camera', ...c }),
  };
}

worker.onmessage = (ev: MessageEvent<FromEngine>) => {
  const m = ev.data;
  if (m.type === 'devDone') {
    pendingDev.get(m.id)?.(m.result);
    pendingDev.delete(m.id);
  } else if (m.type === 'progress') {
    loader.style.setProperty('--p', String(m.value));
    loader.setAttribute('aria-valuenow', String(Math.round(100 * m.value)));
  } else if (m.type === 'status') {
    if (m.state === 'error') fail(m.message ?? '');
    else {
      status.textContent = m.state === 'paused' ? (m.message ?? '') : '';
      status.classList.toggle('visible', m.state === 'paused');
    }
    document.body.classList.toggle('loading', m.state === 'loading');
    if (m.state === 'running') markReady();
  } else if (m.type === 'experience') {
    const item = items.find((i) => i.dataset.experience === m.name);
    if (m.state === 'loading') item?.setAttribute('aria-busy', 'true');
    else item?.removeAttribute('aria-busy');
    document.body.classList.toggle('staged', m.state === 'loading' || m.state === 'playing');
    if (m.state === 'failed') {
      status.textContent = 'That experience could not be loaded.';
      status.classList.add('visible');
      setTimeout(() => status.classList.remove('visible'), 2600);
    }
  } else if (m.type === 'cue') {
    audio.cue(m.cue);
  } else if (m.type === 'cursor') {
    canvas.style.cursor = m.cursor;
  } else if (m.type === 'stats') {
    (window as unknown as { __fabricStats?: unknown }).__fabricStats = m.stats;
  }
};
worker.onerror = (e) => {
  fail('The fabric engine stopped unexpectedly.');
  console.error(e);
};
