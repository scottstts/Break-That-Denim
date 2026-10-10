# Tear sound: the recording carries it (2026-10-10, third round)

Replaces the procedural voices of `2026-10-10-tear-sound.md` and `2026-10-10-tear-sound-structure.md`
(user decision, after six rounds heard as mic noise, plastic bag, rumble, plastic sheet, crumpled bag;
the last two with treble spreading everywhere from the simulation modulating every event). Those
notes still hold for the records' transport, multi-touch sources and the measurement tools.

## Model (src/audio/tearVoice.ts)

- Content: refs/fabric_tear_sound_sample.mp3 itself, its three tearing passages only (0.58-1.13,
  2.40-2.75, 3.87-5.02 s; the quiet stretches before them are hands handling cloth). Mono, high-passed
  at 70 Hz, slow loudness (60 ms RMS) divided out; played as 100 ms slices from random places with
  half-overlapping power-complementary (sine) windows, at original speed and pitch. The rasp, events,
  treble and room are the recording's.
- What the mechanics control, per finger, and nothing else: tearing (a yarn cut within 0.1 s, while
  held), loudness (amplitude sqrt(release rate / 60 J/m/s), averaged over 80 ms, at most 2x the
  recording: 60 J/m/s is a hand tear of intact denim at ~200 yarns/s x 0.3 J/m) and place (pan from the
  fracture position, half the screen width, 60 ms glide).
- Stop (tear stops or finger released): fall-off of 60 dB in 0.3 s, the recording's own tail rate.
  Records of a released finger are ignored; time since the last cut runs on audio time (records stop
  when the cloth rests). Onset 20 ms.
- Loading: the main thread fetches the mp3 at start, decodes it at the context's rate when audio starts,
  prepares it (`prepareSource`) and transfers it to the worklet; silent until then.

GPU records shrink to fracture energy, its position and yarns cut (`ACOUSTIC_STRIDE` 6). The synthetic
room, top roll-off, resonator bank and their fit tool are gone.

Cost: preparation ~6 ms once; 6 us mean per 128-sample block (0.2 % of a core).
