/**
 * A camera's signal breaking up: digital noise held over random short runs (a decoder repeating
 * corrupt samples), gated in bursts, with mains hum from the interference and dropouts to silence;
 * `level` sets how much of it there is (0..1).
 */
export function glitchNoise(rate: number, level: number, left: Float32Array, right: Float32Array): void {
  const n = left.length;
  let seed = 0x9e3779b9;
  const rand = () => {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    return (seed >>> 0) / 4294967296;
  };
  for (const out of [left, right]) {
    let held = 0;
    let hold = 0;
    let gate = 1;
    let gateLeft = 0;
    for (let i = 0; i < n; i++) {
      if (gateLeft-- <= 0) {
        // bursts and dropouts of 10-120 ms; denser as the level rises
        gate = rand() < 0.35 + 0.55 * level ? 0.5 + 0.5 * rand() : 0;
        gateLeft = Math.round(rate * (0.01 + 0.11 * rand()));
      }
      if (hold-- <= 0) {
        held = (rand() * 2 - 1) * (0.4 + 0.6 * rand());
        hold = Math.floor(rand() * rand() * 40 * (1 - 0.6 * level)) + 1;
      }
      const hum = 0.25 * Math.sign(Math.sin((2 * Math.PI * 50 * i) / rate)) * (0.5 + 0.5 * level);
      // the sound fades as the signal is lost at the end of a long breakup
      const tail = Math.min(1, (n - i) / (0.05 * rate));
      out[i] = 0.22 * level * gate * (held + hum) * tail;
    }
  }
}
