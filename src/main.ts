import './style.css';
import { computeDrawingBufferSize } from './shared/drawingBuffer';
import type { FromEngine, ToEngine } from './shared/messages';

const canvas = document.createElement('canvas');
canvas.className = 'stage';
document.body.appendChild(canvas);

const status = document.createElement('div');
status.className = 'status';
status.setAttribute('role', 'status');
status.setAttribute('aria-live', 'polite');
document.body.appendChild(status);

const reset = document.createElement('button');
reset.className = 'reset';
reset.type = 'button';
reset.setAttribute('aria-label', 'Reset fabric');
reset.title = 'Reset';
reset.innerHTML =
  '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path d="M12 5V2L7 6l5 4V7a5 5 0 1 1-5 5H5a7 7 0 1 0 7-7z" fill="currentColor"/></svg>';
document.body.appendChild(reset);

if (!('transferControlToOffscreen' in canvas) || !navigator.gpu) {
  status.textContent = 'This experience needs a browser with WebGPU.';
  status.classList.add('visible');
  throw new Error('WebGPU or OffscreenCanvas unavailable');
}

const worker = new Worker(new URL('./engine/worker.ts', import.meta.url), { type: 'module' });
const post = (m: ToEngine, transfer: Transferable[] = []) => worker.postMessage(m, transfer);

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

let seq = 0;
function pointer(phase: 'down' | 'move' | 'up' | 'cancel', e: PointerEvent) {
  const r = canvas.getBoundingClientRect();
  post({ type: 'pointer', phase, seq: seq++, time: e.timeStamp, x: e.clientX - r.left, y: e.clientY - r.top });
}
canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  canvas.setPointerCapture(e.pointerId);
  pointer('down', e);
});
canvas.addEventListener('pointermove', (e) => {
  // Coalesced samples are superseded by the latest; only the newest position matters.
  pointer('move', e);
});
canvas.addEventListener('pointerup', (e) => pointer('up', e));
canvas.addEventListener('pointercancel', (e) => pointer('cancel', e));
canvas.addEventListener('lostpointercapture', (e) => pointer('cancel', e));
canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    post({ type: 'wheel', seq: seq++, deltaY: e.deltaY, time: e.timeStamp });
  },
  { passive: false },
);
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') post({ type: 'pointer', phase: 'cancel', seq: seq++, time: e.timeStamp, x: 0, y: 0 });
});
window.addEventListener('blur', () => post({ type: 'pointer', phase: 'cancel', seq: seq++, time: performance.now(), x: 0, y: 0 }));
document.addEventListener('visibilitychange', () => post({ type: 'visibility', visible: document.visibilityState === 'visible' }));
reset.addEventListener('click', () => post({ type: 'reset' }));

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
  } else if (m.type === 'status') {
    status.textContent = m.state === 'loading' ? '' : (m.message ?? '');
    status.classList.toggle('visible', m.state === 'error' || m.state === 'paused');
    document.body.classList.toggle('loading', m.state === 'loading');
    if (m.state === 'running') markReady();
  } else if (m.type === 'cursor') {
    canvas.style.cursor = m.cursor;
  } else if (m.type === 'stats') {
    (window as unknown as { __fabricStats?: unknown }).__fabricStats = m.stats;
  }
};
worker.onerror = (e) => {
  status.textContent = 'The fabric engine stopped unexpectedly.';
  status.classList.add('visible');
  console.error(e);
};
