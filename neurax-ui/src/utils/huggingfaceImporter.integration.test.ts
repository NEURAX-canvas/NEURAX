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
  {
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
  },
  {
    // DeepSeek's own paper cites ~16.4B total, ~2.8B active — this list only
    // checks `total_parameters`; active-parameter accuracy is checked
    // separately below.
    name: 'DeepSeek-MoE-16B',
    published: 16.4e9,
    config: {
      architectures: ['DeepseekForCausalLM'],
      model_type: 'deepseek',
      hidden_size: 2048,
      intermediate_size: 10944,
      moe_intermediate_size: 1408,
      num_hidden_layers: 28,
      num_attention_heads: 16,
      num_key_value_heads: 16,
      n_routed_experts: 64,
      n_shared_experts: 2,
      num_experts_per_tok: 6,
      max_position_embeddings: 4096,
      vocab_size: 102400,
      rms_norm_eps: 1e-6,
      rope_theta: 10000,
      hidden_act: 'silu',
      tie_word_embeddings: false,
    },
  },
];

const MIXTRAL = MODELS.find((m) => m.name === 'Mixtral 8x7B')!;

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
 * The gap this test file used to record rather than hide.
 *
 * A routed model imported from its config used to come out wrong by 20-180%
 * — Mixtral 8x7B analysed to roughly 36B against a published 46.7B;
 * DeepSeek-MoE-16B to roughly 46B against a published 16.4B. The cause: a
 * block diagram draws an MoE layer as separate router / experts / combine
 * nodes, and the compiler read every one of them as if it alone were a
 * complete MoE layer — the router's `hidden × num_experts` gating matrix
 * costed as `num_experts` full experts, and the per-layer repeat count
 * diluted across however many same-typed nodes one logical layer used.
 *
 * Fixed by giving the router, the combine step, and a DeepSeek-style shared
 * expert their own parser types (`moe_router` / `moe_combine` /
 * `moe_shared_expert`) with their own real, much smaller formulas — see
 * `neurax-ir/src/architecture/mod.rs`. Both models are back in the verified
 * list above; this file now also checks the metric that gap-fix made
 * possible to compute honestly: how many of those parameters a token
 * actually touches.
 */
describe('mixture-of-experts: active parameters, not just total', () => {
  it('Mixtral 8x7B: active parameters are a small, correct fraction of the total', async () => {
    if (!serviceUp) return;

    const report = await analyseThroughService(JSON.stringify(MIXTRAL.config), MIXTRAL.name);
    const metrics = report.metrics as Record<string, unknown>;
    const total = metrics.total_parameters as number;
    const active = metrics.active_parameters as number;

    // Published: ~12.9B active out of ~46.7B total — a token reaches 2 of 8
    // experts, not all of them.
    const published = 12.9e9;
    const error = Math.abs(active - published) / published;
    expect(
      error,
      `Mixtral active params: compiler says ${(active / 1e9).toFixed(2)}B, published ~${(published / 1e9).toFixed(1)}B`,
    ).toBeLessThan(0.1);

    // The metric's entire reason to exist: active must be well below total,
    // not a rounding difference of it.
    expect(active).toBeLessThan(total * 0.5);
  }, 30000);

  it("DeepSeek-MoE-16B: active parameters include its always-on shared experts", async () => {
    if (!serviceUp) return;

    const deepseek = MODELS.find((m) => m.name === 'DeepSeek-MoE-16B')!;
    const report = await analyseThroughService(JSON.stringify(deepseek.config), deepseek.name);
    const metrics = report.metrics as Record<string, unknown>;
    const total = metrics.total_parameters as number;
    const active = metrics.active_parameters as number;

    // Published: ~2.8B active — 6 of 64 routed experts, plus 2 shared
    // experts that run on every token regardless of routing.
    const published = 2.8e9;
    const error = Math.abs(active - published) / published;
    expect(
      error,
      `DeepSeek active params: compiler says ${(active / 1e9).toFixed(2)}B, published ~${(published / 1e9).toFixed(1)}B`,
    ).toBeLessThan(0.15);

    expect(active).toBeLessThan(total * 0.5);
  }, 30000);
});
