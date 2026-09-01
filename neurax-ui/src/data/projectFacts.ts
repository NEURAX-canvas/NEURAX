/**
 * Figures the landing page states about NEURAX.
 *
 * Every number here is checked against the repository by
 * `projectFacts.test.ts`, because a landing page is the one surface where a
 * wrong number is a claim rather than a bug. The previous copy advertised "680+
 * configurable block types" and broke that down per family in the FAQ; the
 * catalogue holds 193. It also carried market statistics — a $2M average cost
 * for a failed training run, 73% of models needing redesign, a 5x production
 * overrun — with no source behind any of them.
 *
 * If a number cannot be derived from the repository or attributed to a source,
 * it does not belong on this page.
 */

/** Blocks and macro-blocks in the architecture catalogue, per family. */
export const CATALOGUE = {
  cnn: { blocks: 20, macroBlocks: 3 },
  diffusion: { blocks: 16, macroBlocks: 3 },
  gan: { blocks: 13, macroBlocks: 2 },
  gnn: { blocks: 17, macroBlocks: 3 },
  moe: { blocks: 19, macroBlocks: 2 },
  rnn: { blocks: 14, macroBlocks: 3 },
  ssm: { blocks: 13, macroBlocks: 2 },
  transformer: { blocks: 17, macroBlocks: 3 },
} as const;

export const FAMILY_COUNT = Object.keys(CATALOGUE).length;

export const BLOCK_COUNT = Object.values(CATALOGUE).reduce(
  (total, family) => total + family.blocks + family.macroBlocks,
  0,
);

/** Metrics returned by a single `/analyze` call. */
export const METRIC_COUNT = 66;

/** Passes in the IR pipeline, from architecture through to the report. */
export const IR_PASS_COUNT = 10;

/** Reference architectures shipped as ready-to-load presets. */
export const PRESET_COUNT = 30;

/**
 * Headline latency.
 *
 * The published benchmark is an 8B-parameter architecture analysed end to end;
 * the CLI reports well under this on the shipped examples.
 */
export const ANALYSIS_BUDGET_MS = 50;

/**
 * Export targets.
 *
 * A design leaving NEURAX is either archived for re-import, handed to the
 * compiler, or committed somewhere — nothing else is offered, because nothing
 * else could be offered truthfully.
 *
 * The framework emitters that used to sit here produced a class whose
 * `__init__` was empty and whose `forward` was a chain of `x2 = x1`: for
 * LLaMA 2 7B, an identity function carrying the model's name. The GitHub
 * target used to push exactly that into users' repositories, and now pushes
 * the same two files this list names.
 */
export const EXPORT_FORMATS = [
  'JSON — architecture and analysis, re-importable',
  'NEURAX IR — the exact topology the compiler analyses',
  'GitHub — the same two files, committed to a repository',
] as const;

/** The four figures shown under the hero. */
export const HERO_STATS = [
  {
    value: String(FAMILY_COUNT),
    label: 'Architecture families',
    detail: 'Transformer, CNN, MoE, SSM, diffusion, GAN, GNN, RNN',
  },
  {
    value: String(BLOCK_COUNT),
    label: 'Catalogue blocks',
    detail: 'Individual blocks and macro-blocks across every family',
  },
  {
    value: String(METRIC_COUNT),
    label: 'Metrics per analysis',
    detail: 'Parameters, FLOPs, memory, latency, cost, energy and carbon',
  },
  {
    value: `<${ANALYSIS_BUDGET_MS}ms`,
    label: 'End-to-end analysis',
    detail: `${IR_PASS_COUNT}-pass IR pipeline, no GPU required`,
  },
] as const;
