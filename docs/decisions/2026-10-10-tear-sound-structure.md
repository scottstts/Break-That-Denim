# Tear sound: structure of the emission (2026-10-10, second round)

Replaced by `2026-10-10-tear-sound-sample.md` (the recording carries the sound). What follows is the
procedural voice it replaced and what was measured on the way; the measurement tools remain.

Supersedes the event model of `2026-10-10-tear-sound.md` (transport, multi-touch and jitter buffer
there still hold). The user heard the previous voice as "a microphone covered by a plastic bag", then
as "rumbles, with additional noises; the sample is a clear tear with some reverb", then as "tearing a
plastic sheet", then as "crumpling a plastic bag".

**Only tearing is heard (user decision).** The recording is evidence for the tear sound only. Yarn slip,
pull-out, settling, bounce and fibres giving in a stretched yarn that does not part make no sound; the
GPU no longer accumulates slip or pull-out work. Fracture sounds only while yarns part at the source
(a cut within `maxDelay`); energy released otherwise is dropped. Sound ends within ~0.3 s of the last
cut (crackle drain plus the room's decay).

**Silent once the finger lets go (user decision).** The worker posts `{ grip, held }` on the record port
at grab and release (ordered with the records). A released source is cleared at once; with no finger
holding, the output (room tail included) fades out over 15 ms and everything queued is dropped. Records
of a source not held are ignored, so a tear running on after release, bounce or settling is never heard.

## What was wrong (measured against refs/fabric_tear_sound_sample.mp3)

- **Crinkle.** Each event went into one random third-octave band, and events were coarse (32-960
  fibres), the coarseness tuned to reach the recording's spikiness. A band with 1 % of the spectrum got
  1 % of the events: the top octaves were a sparse scatter of isolated clicks (band kurtosis 6.4-11 kHz
  15-25 vs 5.5). Full-band kurtosis and long-term spectrum matched, which is why earlier checks passed.
- **No rasp.** The recording's envelope is modulated at 130-330 Hz (one yarn after the next at the tear
  front); a Poisson process has no such rhythm.
- **Rumble and extra noises.** Slip and pull-out events used a spectrum fitted to the recording's quiet
  pre-tear passages, which are hand-handling rumble (63-126 Hz dominant), so they peaked at 126-200 Hz.
  Yarn ruptures still owed after the cuts stopped were spaced out by a decaying rate: stray clicks.
- **Sizzle (plastic).** Above 11 kHz the voice carried -19 to -24 dB per band re 1-2 kHz (resonator
  skirts falling only 6 dB/octave); the recording has nothing there (its format ends at 12 kHz) and is
  already falling 6 dB per half octave at 8-11 kHz. Every metric, fit and picture had stopped at 10-12 kHz.
  A 6th-order Butterworth at 11 kHz (`TOP_ROLLOFF`) continues the measured fall; the spectrum is fitted
  through it.
- **Wide, flickering image (crumpled bag).** Every analysis had downmixed to mono. The recording is a
  point source: left/right correlation 0.97, level difference per 5 ms sd 0.33 dB. The voice was 0.77
  (dips to 0.42) and sd 1.1-1.8 dB: each event had its own pan jitter, the room had separate tails per
  ear, and each source's pan jumped every substep with the scatter of its fracture positions. Now every
  event of a source sits at the source's place, the source glides (60 ms) over half the screen width,
  and the room is one tail fed by the mid signal and added to both ears: 0.998 / 0.33 dB on a drag.
  Event-level distributions (`tools/audio/events.mjs`: prominence, amplitude spread, spacing, duration,
  tonality) already matched the recording in mono, so they could not explain what was heard.
- **Dry.** The recording carries a small room: after a tear stops it decays ~1 dB per 5 ms (RT60
  ~0.28 s) from ~10 dB under the tear.

## Model now (src/audio/tearVoice.ts, room.ts)

- Fibre avalanches from one fibre to a whole yarn (`yarnFibres` 400 = DENIM mean, s^-5/2): tens of
  thousands of events/s, every band densely filled. No fitted event size.
- Yarn ruptures: `rupture` 0.5 of the fracture energy is banked and spent as bursts, one per yarn cut,
  at each front's pace. The lattice parts a span of 6-9 yarns within a substep (pluck: 4-10 cuts every
  40-60 ms), so cuts are credits spent by a gamma renewal process (`regularity` 4) at the front's cut
  rate averaged over `rateTime`; the rate is held once cuts stop. Energy with no yarn to part (damage
  without separation) is dropped after `maxDelay`. A burst is capped at one yarn's share.
- One radiation spectrum (`SPECTRUM`) for all mechanisms; band Q 2 (Q 3 leaves the top octaves sparse);
  refit with `tools/audio/fit.ts` (1 dB rms).
- Room: 16-line FDN with input allpass diffusion, RT60 0.28 s, wet 0.1 of the direct energy.

GPU (`common.wgsl` ACOUSTIC, stride 8): +0 fracture energy, +1..4 fracture-weighted position, +5 yarns
cut, +6..7 a 64-bit hashed mask of the 1 cm cells they were cut in (popcount = fronts). Measured but
not kept: the energy of the final cut increment (depends on the substep: the softening law releases most energy before the cut, 26 % at the cut in a
rip), and the count of softening yarns (Little's law: softening lasts 20-60 ms per yarn vs 2-3 ms
between cuts at a front, so the progressive release is correctly a Poisson crackle).

## How it was judged

- Matched condition: synthetic records of one front in intact denim lumped as the lattice lumps it
  (`tools/audio/front.mjs`). Score over 50 ms kurtosis/crest, envelope modulation 16-1000 Hz, envelope
  autocorrelation, per-octave band kurtosis, cross-band fine-envelope coherence. The recording's own
  three tears sit 1.8-2.8 from their pooled metrics; old voice 7.0, new 2.4 (200 yarns/s, one front).
  Denser conditions (400/s x 2, 600/s x 3) fit the recording worse with every event model, so the
  recording is a single front near 200 yarns/s; fast multi-front rips in the app are denser by physics.
- Rejected by that test: a broadband coherent click per event or per rupture (helps the dense rip,
  breaks the matched case: 2.5 -> 6-12).
- Then side-by-side spectrograms with the recording on real sessions (haul, rip): that is what found
  the rumble, the stray clicks and the missing room; the scores had not.
- Hearing-based check: `tools/audio/texture.mjs` (McDermott & Simoncelli texture statistics on an ERB
  filterbank; "fine" variants normalise each envelope by its 50 ms mean so the hand's slow loudness
  swings do not count). The recording's three tears differ from each other by 3.0-7.6 on its fine
  distance; the voice scores 2.6 (haul session) and 3.1 (single front). Q, collapse time and rupture
  share move it little (3.1-4.0); this metric is not a reason to tune further.

Cost: 1.2 % of a core on average, worst 128-sample block 0.42 ms (rip, M5).
