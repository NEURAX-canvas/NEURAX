/**
 * Does the compiler actually understand an imported model?
 *
 * The unit tests beside this file check that the importer emits a graph whose
 * arithmetic works out. That is necessary but not sufficient: a graph can be
 * internally consistent and still be something the compiler reads differently
 * than intended — a block whose parameters it takes under other names, a
 * feed-forward it counts as two matrices where the design means three, experts
 * attached where they are counted once instead of once per layer. Every one of
 * those bugs has happened in this repository, and none of them is visible
 * without asking the compiler itself.
 *
 * So this test runs the real pipeline end to end: a real `config.json` from the
 * Hub → the importer → `compileToNeuraxIR` → the running `neurax-service` →
 * a parameter count compared against the published one. Nothing here is
 * simulated.
 *
 * It needs the service running, and skips itself when it is not, so that
 * `vitest` on a laptop with no backend stays green:
 *
 *     cargo run -p neurax-service          # or the desktop app
 *     NEURAX_API=http://127.0.0.1:9098 npx vitest run *.integration.test.ts
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { parseHuggingFaceConfig } from './huggingfaceImporter';
import { compileToNeuraxIR } from './neuraxCompiler';
import { normalizeBlockParams } from './blockDefaults';
import { CanvasNode } from '@/types/architecture.ts';

const API = process.env.NEURAX_API ?? 'http://127.0.0.1:9098';

let serviceUp = false;

beforeAll(async () => {
  try {
    const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2000) });
    serviceUp = res.ok;
  } catch {
    serviceUp = false;
  }
  if (!serviceUp) {
    console.warn(`[skip] no neurax-service at ${API}; run it to exercise these tests`);
  }
});

/**
 * Configs kept whole, as they appear on the Hub, with their published counts.
 *
 * The tolerance is 5 %: the compiler models biases, norms and tied embeddings
 * in its own way, and a published "7B" is itself a rounded number. What this
 * catches is not a percent of drift but the order-of-magnitude class of error —
 * the +122 % and −96 % kind that shipped here before.
 */
const MODELS = [
  {
    name: 'LLaMA-2 7B',
    published: 6.74e9,
    config: {
      architectures: ['LlamaForCausalLM'],
      model_type: 'llama',
      hidden_size: 4096,
      intermediate_size: 11008,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 32,
      max_position_embeddings: 4096,
      vocab_size: 32000,
      rms_norm_eps: 1e-5,
      rope_theta: 10000.0,
      hidden_act: 'silu',
      tie_word_embeddings: false,
    },
  },
  {
    name: 'Mistral 7B',
    published: 7.24e9,
    config: {
      architectures: ['MistralForCausalLM'],
      model_type: 'mistral',
      hidden_size: 4096,
      intermediate_size: 14336,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      max_position_embeddings: 32768,
      vocab_size: 32000,
      rms_norm_eps: 1e-5,
      rope_theta: 10000.0,
      hidden_act: 'silu',
      tie_word_embeddings: false,
    },
  },
  {
    name: 'Qwen2 7B',
    published: 7.62e9,
    config: {
      architectures: ['Qwen2ForCausalLM'],
      model_type: 'qwen2',
      hidden_size: 3584,
      intermediate_size: 18944,
      num_hidden_layers: 28,
      num_attention_heads: 28,
      num_key_value_heads: 4,
      max_position_embeddings: 32768,
      vocab_size: 152064,
      rms_norm_eps: 1e-6,
      rope_theta: 1000000.0,
      hidden_act: 'silu',
      tie_word_embeddings: false,
    },
  },
  {
    name: 'GPT-2',
    published: 124e6,
    config: {
      architectures: ['GPT2LMHeadModel'],
      model_type: 'gpt2',
      n_embd: 768,
      n_head: 12,
      n_layer: 12,
      n_positions: 1024,
      n_inner: null,
      vocab_size: 50257,
      layer_norm_epsilon: 1e-5,
      activation_function: 'gelu_new',
    },
  },
];

/**
 * Mixtral, kept out of the list above on purpose. See the characterisation test
 * at the bottom of this file for why.
 */
const MIXTRAL = {
  name: 'Mixtral 8x7B',
  published: 46.7e9,
  config: {
      architectures: ['MixtralForCausalLM'],
      model_type: 'mixtral',
      hidden_size: 4096,
      intermediate_size: 14336,
      num_hidden_layers: 32,
      num_attention_heads: 32,
      num_key_value_heads: 8,
      num_local_experts: 8,
      num_experts_per_tok: 2,
      max_position_embeddings: 32768,
      vocab_size: 32000,
      rms_norm_eps: 1e-5,
      rope_theta: 1000000.0,
      hidden_act: 'silu',
      tie_word_embeddings: false,
  },
};

/**
 * Mirror what the page does between import and analysis.
 *
 * The page hydrates imported blocks against their schema before compiling —
 * templates state `hidden_size` where a block's schema says `d_model`, and
 * without the alias pass the width is ignored and the block falls back to its
 * default. Skipping it here would test a path the application never takes.
 */
function hydrate(nodes: CanvasNode[]): CanvasNode[] {
  return nodes.map((node) => ({
    ...node,
    params: normalizeBlockParams(node.type, node.params) as CanvasNode['params'],
  }));
}

async function analyseThroughService(configJson: string, name: string) {
  const imported = parseHuggingFaceConfig(configJson);
  expect(imported.error, `${name} failed to import`).toBeUndefined();

  const ir = compileToNeuraxIR(hydrate(imported.nodes), imported.connections, {
    modelName: name,
    family: imported.family,
    ...imported.hardwareConfig,
  });

  const res = await fetch(`${API}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ topology: ir }),
  });

  // Read the body once: a response body can only be consumed a single time, so
  // it cannot be read both for an assertion message and for the result.
  const body = await res.text();
  expect(res.ok, `${name}: /analyze returned ${res.status} — ${body.slice(0, 400)}`).toBe(true);

  const { report } = JSON.parse(body) as { report: Record<string, unknown> };
  return report;
}

describe.skipIf(!process.env.NEURAX_API && false)('the compiler understands an imported model', () => {
  for (const { name, published, config } of MODELS) {
    it(`${name} analyses to its published parameter count`, async () => {
      if (!serviceUp) return;

      const report = await analyseThroughService(JSON.stringify(config), name);
      const metrics = report.metrics as Record<string, unknown> | undefined;
      const total = metrics?.total_parameters;

      expect(typeof total, `${name}: no total_parameters in the report`).toBe('number');

      const counted = total as number;
      const error = Math.abs(counted - published) / published;

      expect(
        error,
        `${name}: compiler says ${(counted / 1e9).toFixed(3)} B, published ${(published / 1e9).toFixed(3)} B (${(error * 100).toFixed(1)} % off)`,
      ).toBeLessThan(0.05);
    }, 30000);
  }

  it('reads the layer count from the stack, not from the block list', async () => {
    if (!serviceUp) return;

    // The graph has about fifteen blocks and describes a thirty-two layer
    // model. A compiler that counted blocks would say fifteen — and every
    // memory and cost figure downstream would be wrong by the same factor.
    const report = await analyseThroughService(JSON.stringify(MODELS[0].config), 'LLaMA-2 7B');
    const metrics = report.metrics as Record<string, unknown>;
    expect(metrics.num_layers).toBe(32);
  }, 30000);

  it('produces memory figures consistent with the parameter count', async () => {
    if (!serviceUp) return;

    const report = await analyseThroughService(JSON.stringify(MODELS[0].config), 'LLaMA-2 7B');
    const metrics = report.metrics as Record<string, unknown>;

    const params = metrics.total_parameters as number;
    const weights = metrics.parameter_memory_bytes as number;

    // Whatever precision was assumed, the weights cannot be fewer than one byte
    // per parameter nor more than four. This is the check that catches a
    // parameter count and a memory figure computed from different models.
    expect(weights).toBeGreaterThanOrEqual(params);
    expect(weights).toBeLessThanOrEqual(params * 4);
  }, 30000);

  it('scales the cost of a mixture of experts by its expert count', async () => {
    if (!serviceUp) return;

    // Mixtral and Mistral share every attention and embedding dimension; the
    // only difference is eight experts where there was one feed-forward. If the
    // compiler read the experts as a single block, the two would come back the
    // same size — which is exactly the defect that once put Mixtral out by
    // +122 %.
    const dense = await analyseThroughService(JSON.stringify(MODELS[1].config), 'Mistral 7B');
    const routed = await analyseThroughService(JSON.stringify(MIXTRAL.config), MIXTRAL.name);

    const denseParams = (dense.metrics as Record<string, unknown>).total_parameters as number;
    const routedParams = (routed.metrics as Record<string, unknown>).total_parameters as number;

    expect(routedParams).toBeGreaterThan(denseParams * 4);
  }, 60000);
});

/**
 * A known gap, recorded rather than hidden.
 *
 * A routed model imported from its config comes out about 22 % below its
 * published size — Mixtral 8x7B analyses to roughly 36 B against a published
 * 46.7 B. This is **not** an importer defect, and the evidence is that the
 * built-in `tpl-mixtral-8x7b` template, hydrated and compiled through exactly
 * the same path, returns the identical figure. Both send the compiler the same
 * graph; the shortfall is in how the mixture-of-experts blocks are accounted
 * for once they reach it.
 *
 * The cause is visible in the emitted IR: `noisy_topk_router` and
 * `expert_combine` each lower to a *full* `moe` layer, the second with no
 * parameters at all, so it falls back to the schema defaults of 64 experts at
 * width 768. The expert widths that result do not add up to the model's.
 *
 * Fixing that is a change to the compiler's MoE accounting, not to this
 * importer, so it is out of scope here. What is in scope is not pretending the
 * number is right. This test pins the current behaviour: when the compiler is
 * fixed it will fail, which is the point — it forces the tolerance to be
 * tightened rather than letting a silent improvement go unnoticed, and stops
 * anyone reading the suite above and concluding routed models are verified.
 */
describe('known gap: routed models under-count', () => {
  it('Mixtral 8x7B is short of its published size by a documented margin', async () => {
    if (!serviceUp) return;

    const report = await analyseThroughService(JSON.stringify(MIXTRAL.config), MIXTRAL.name);
    const counted = (report.metrics as Record<string, unknown>).total_parameters as number;
    const shortfall = (MIXTRAL.published - counted) / MIXTRAL.published;

    expect(
      shortfall,
      `Mixtral now analyses to ${(counted / 1e9).toFixed(2)} B — if the MoE accounting has ` +
        'been fixed, move this model back into the verified list above and delete this test.',
    ).toBeGreaterThan(0.1);

    // Still the right order of magnitude: this is an accounting gap, not the
    // compiler failing to see the experts at all.
    expect(counted).toBeGreaterThan(30e9);
  }, 30000);
});
