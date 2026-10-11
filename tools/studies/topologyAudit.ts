/** Independent graph audit for a live staged run; deliberately ignores x.w when finding anchors. */
import { readBuffer } from '../../src/gpu/util';
import type { GpuSolver } from '../../src/simulation/gpu/solver';
import type { Interaction } from '../../src/simulation/gpu/interaction';

export class TopologyAudit {
  readonly rows: object[] = [];
  private readonly waiting = new Map<number, number>();
  private previous = '';
  private maxDelay = 0;
  private maxWrong = 0;
  private readonly pending: Promise<void>[] = [];
  private readonly solver: GpuSolver;
  private readonly interaction: Interaction;
  private readonly watch: number;

  constructor(solver: GpuSolver, interaction: Interaction, watch: number) {
    this.solver = solver;
    this.interaction = interaction;
    this.watch = watch;
  }

  record(t: number): void { this.pending.push(this.sample(t)); }
  async finish(): Promise<void> { await Promise.all(this.pending); }

  async sample(t: number): Promise<void> {
    const { solver, interaction } = this;
    const anchors = new Set(interaction.coreNodes);
    const cores = [new Set(interaction.core(0).filter((n) => anchors.has(n))), new Set(interaction.core(1).filter((n) => anchors.has(n)))];
    const patches = [0, 1].map((slot) => new Set(interaction.core(slot)));
    const pieceOf = (solver as unknown as { pieceOf: GPUBuffer }).pieceOf;
    const [xb, sb, pb] = await Promise.all([solver.x, solver.segs, pieceOf].map((b) => readBuffer(solver.device, b)));
    const x = new Float32Array(xb), segs = new Uint32Array(sb);
    const labels = new Uint32Array(pb);
    const nx = solver.scene.layers[0].nx;
    const seen = new Uint8Array(solver.N);
    const unanchored = new Set<number>();
    const pieces: object[] = [];
    let watched: object = {};
    let wrongDetached = 0;
    for (let seed = 0; seed < solver.N; seed++) {
      if (seen[seed] || x[4 * seed + 3] > 1.5) continue;
      const ns = [seed];
      seen[seed] = 1;
      for (let at = 0; at < ns.length; at++) {
        const n = ns[at], i = n % nx, j = Math.floor(n / nx);
        const edges = [[i + 1 < nx ? n + 1 : -1, 2 * n], [i > 0 ? n - 1 : -1, 2 * (n - 1)], [n + nx < solver.N ? n + nx : -1, 2 * n + 1], [j > 0 ? n - nx : -1, 2 * (n - nx) + 1]];
        for (const [m, s] of edges) {
          if (m < 0 || seen[m] || x[4 * m + 3] > 1.5 || (segs[8 * s + 5] & 3) !== 1) continue;
          seen[m] = 1;
          ns.push(m);
        }
      }
      const mounted = ns.some((n) => solver.scene.pinned[n]);
      const hands = cores.map((c) => ns.filter((n) => c.has(n)));
      const held = hands.some((c) => c.length > 0);
      const detached = ns.filter((n) => x[4 * n + 3] > 0.5).length;
      if (!mounted && !held) for (const n of ns) unanchored.add(n);
      else wrongDetached += detached;
      const row = { seed, size: ns.length, mounted, hands, savedCores: patches.map((c) => ns.filter((n) => c.has(n))), detached, solverPieceIds: [...new Set(ns.filter((n) => x[4 * n + 3] > 0.5).map((n) => labels[n]))], xyz: [0, 1, 2].map((a) => ns.reduce((s, n) => s + x[4 * n + a], 0) / ns.length) };
      if (ns.includes(this.watch)) watched = row;
      if (!mounted && ns.length >= 3) pieces.push(row);
    }
    const late: number[] = [];
    for (const n of unanchored) {
      if (x[4 * n + 3] > 0.5) continue;
      late.push(n);
      if (!this.waiting.has(n)) this.waiting.set(n, t);
    }
    let delay = 0;
    for (const [n, since] of this.waiting) {
      this.maxDelay = Math.max(this.maxDelay, t - since);
      if (!unanchored.has(n) || x[4 * n + 3] > 0.5) this.waiting.delete(n);
      else delay = Math.max(delay, t - since);
    }
    this.maxWrong = Math.max(this.maxWrong, wrongDetached);
    const row = { t, watched, pieces, late, delay, wrongDetached };
    this.rows.push(row);
    const key = JSON.stringify({ watched: { ...watched, xyz: undefined }, late, wrongDetached });
    if (key !== this.previous) {
      console.log('topology audit', JSON.stringify({ t, watched, lateCount: late.length, delay, wrongDetached }));
      this.previous = key;
    }
  }

  summary(): object {
    return { samples: this.rows.length, maxClassificationDelaySeconds: this.maxDelay, maxAnchoredNodesIncorrectlyDetached: this.maxWrong, unresolvedNodes: [...this.waiting.keys()] };
  }
}
