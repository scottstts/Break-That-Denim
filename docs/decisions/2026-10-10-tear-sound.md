# Tear sound, multi-touch, page UI (2026-10-10)

The sound model below was replaced the same day: `2026-10-10-tear-sound-structure.md`. Records,
transport, multi-touch and UI still hold.

## Sound: what the recording is

`refs/fabric_tear_sound_sample.mp3` (three tears, 24 kHz mp3) is not used at runtime; it is the
measurement target. Looked at as a waveform (`tools/audio/compare.mjs`, scales 1 s / 200 ms / 40 ms /
10 ms) a tear is a **continuous, zero-mean, band-limited oscillation** (centroid ~2 kHz in loud
passages, ~1.2 kHz 25 dB down) in which short wavelets of one to two cycles overlap densely, some
standing out: per 50 ms window median kurtosis 5.1, crest 14.3 dB (`tools/audio/analyze.mjs`).

Two earlier versions were rejected by listening ("microphone being scratched", "mic glitch"); the
waveform comparison shows why, and summary statistics (grain counts, long-term spectrum, segment
kurtosis) had not:

- random independent clicks with heavy-tailed sizes and Poisson bursts per substep: clumpy loudness at
  2-30 ms;
- one coherent pulse per yarn (~1 k/s): every pulse a step with a slow low-frequency tail and flat
  silence between. Low bands summed coherently make steps; sparse pulses make gaps.

Judge the voice by side-by-side waveforms at all four scales and the 50 ms kurtosis/crest, not by
spectra alone.

## Sound: model

- GPU accumulates, per substep and per sound source, the strain energy released by yarn fibre
  fracture (J/m, from the yarn damage field: damage increment x energy density at the crossing's peak
  strain), pull-out count and friction work, and crossing-slip work x slip speed (rubbing noise grows
  with speed: creep under a held pull, ~1-2 mm/s, is near silent; active slip runs 5-20 mm/s). Slots
  live in the solver's `counters` buffer (no new storage binding: yarnfield and step are at the
  10-buffer limit); the source anchors are a uniform. Measured cost: none at bench resolution.
- `AcousticLog` copies each substep's block and clears it; one non-blocking readback per frame, mapped
  as soon as the substeps finish (before the render). Records go worker -> AudioWorklet over a
  MessagePort (main thread not involved).
- Voice (`src/audio/tearVoice.ts`): released energy flows through a two-stage reservoir (8 ms each)
  and fires a Poisson process of events. This de-quantizes the lattice (failure resolved per 8.3 ms
  substep and per 9 x 6 yarn bundle; single substeps reach 50-100 yarn equivalents). A fracture event
  is a fibre bundle of 32 fibres x an avalanche size from P(s) ~ s^-5/2 (fibre bundle model, global
  load sharing) on [1, 30]; the bundle size was set by matching the 50 ms kurtosis/crest (8 fibres:
  too Gaussian, 96: too spiky). Each event is one wavelet: an impulse into one band of a third-octave
  resonator bank (Q 3), the band drawn from a per-mechanism distribution fitted to the recording
  (`tools/audio/fit.ts`: fracture to loud passages, friction to the quiet stretching before them,
  ~1 dB rms). Band responses have unit energy, so loudness follows released energy.
- Absolute level: `gain` puts a hard rip at ~-23 dBFS RMS (recording -24). Limiter + soft clip guard
  multi-hand rips.
- Jitter buffer: 30 ms preroll after silence, catch-up at 1.25x / 2x beyond 80 / 160 ms queued. The
  reservoir bridges late batches (offline jitter test up to 60 ms adds only a few dips).

## Multi-touch

- Up to 4 grips (`MAX_GRIPS`), one per pointer, each its own slot: own handle, bounded lead/force
  (120 N each), pinch, drag lift. Patches are disjoint (a node belongs to one grip), so the per-node
  `ext` writes keep a single writer. A finger on material another finger already holds takes nothing;
  partly overlapping pinches keep only free nodes (force bound unchanged).
- `tools/gpu/scripts/twohand.ts` (apart / shear / overlap / adjacent / three): no non-finite values,
  no line-search rejections; pulling apart fails the cloth between the hands.
- Sound with several hands: each emission belongs to the nearest held grip (anchors updated per
  frame; kept after release so a tear running on keeps its source). Each source is its own stream with
  its own pan. One stream for all made two tears sound like one doubled sound.

## UI

Static markup in `index.html` (loader shows before scripts run): wordmark "Break That Denim" (Anton /
Oswald from Google Fonts, shown when loaded or after 2.5 s), torn along the N's diagonal with the
topstitch broken under it; loader = a double-row topstitch sewing itself with real staged progress
(worker `progress` messages; the drape downloads in parallel with pipeline creation); reset = stitched
pill. Safe-area insets; checked headless at 1440x900 and 390x844.

Audio starts on the first pointer gesture (iOS: also on pointerup); suspended while the page is hidden.
