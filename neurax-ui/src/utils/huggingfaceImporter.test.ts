/**
 * An imported model must be the model, not something shaped like it.
 *
 * The repository has been burned by this exact class of defect before: a
 * template that looked like LLaMA and reported numbers for a 28 %-smaller
 * model, because nothing compared a computed figure to a published one. An
 * importer is worse in that respect than a template — a template is wrong once
 * and can be fixed, whereas an importer is wrong for every config anyone ever
 * feeds it.
 *
 * So these tests do two separate things. They check that the shape facts read
 * out of a real `config.json` match the model card, and they independently
 * count the parameters implied by the *emitted graph* and compare that to the
 * published count. The second check is what catches a graph that carries the
 * right numbers in the wrong places — a gated feed-forward emitted as a
 * standard one, an untied LM head silently tied, experts attached where the
 * compiler will count them once instead of per layer.
 *
 * The configs below are trimmed copies of the real files on the Hub, keeping
 * every field this importer reads.
 */
import { describe, it, expect } from 'vitest';
import {
  parseHuggingFaceConfig,
  looksLikeHuggingFaceConfig,
  describeHuggingFaceConfig,
} from './huggingfaceImporter';
import { CanvasNode } from '@/types/architecture.ts';

// ── Real configs, trimmed to the fields that carry shape ────────────────────

const LLAMA2_7B = {
  architectures: ['LlamaForCausalLM'],
  _name_or_path: 'meta-llama/Llama-2-7b-hf',
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
  torch_dtype: 'float16',
};

const MISTRAL_7B = {
  architectures: ['MistralForCausalLM'],
  model_type: 'mistral',
  hidden_size: 4096,
  intermediate_size: 14336,
  num_hidden_layers: 32,
  num_attention_heads: 32,
  num_key_value_heads: 8,
  max_position_embeddings: 32768,
  sliding_window: 4096,
  vocab_size: 32000,
  rms_norm_eps: 1e-5,
  rope_theta: 10000.0,
  hidden_act: 'silu',
  tie_word_embeddings: false,
};

const MIXTRAL_8X7B = {
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
};

const QWEN2_7B = {
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
};

const GEMMA_2B = {
  architectures: ['GemmaForCausalLM'],
  model_type: 'gemma',
  hidden_size: 2048,
  intermediate_size: 16384,
  num_hidden_layers: 18,
  num_attention_heads: 8,
  num_key_value_heads: 1,
  head_dim: 256,
  max_position_embeddings: 8192,
  vocab_size: 256000,
  rms_norm_eps: 1e-6,
  rope_theta: 10000.0,
  hidden_act: 'gelu',
  tie_word_embeddings: true,
};

/** GPT-2: the awkward one — every field under a different name, `n_inner` null. */
const GPT2_BASE = {
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
};

const BERT_BASE = {
  architectures: ['BertForMaskedLM'],
  model_type: 'bert',
  hidden_size: 768,
  intermediate_size: 3072,
  num_hidden_layers: 12,
  num_attention_heads: 12,
  max_position_embeddings: 512,
  type_vocab_size: 2,
  vocab_size: 30522,
  layer_norm_eps: 1e-12,
  hidden_act: 'gelu',
};

/** A multimodal release, whose language model hides under `text_config`. */
const LLAVA_7B = {
  architectures: ['LlavaForConditionalGeneration'],
  _name_or_path: 'llava-hf/llava-1.5-7b-hf',
  model_type: 'llava',
  vision_config: { hidden_size: 1024, num_hidden_layers: 24, image_size: 336 },
  text_config: {
    model_type: 'llama',
    hidden_size: 4096,
    intermediate_size: 11008,
    num_hidden_layers: 32,
    num_attention_heads: 32,
    num_key_value_heads: 32,
    max_position_embeddings: 4096,
    vocab_size: 32064,
    rms_norm_eps: 1e-5,
    rope_theta: 10000.0,
    hidden_act: 'silu',
  },
};

// ── An independent parameter count, read off the emitted graph ──────────────

function node(nodes: CanvasNode[], type: string): CanvasNode | undefined {
  return nodes.find((n) => n.type === type);
}

function allOf(nodes: CanvasNode[], type: string): CanvasNode[] {
  return nodes.filter((n) => n.type === type);
}

function p(n: CanvasNode | undefined, key: string): number {
  const value = n?.params?.[key];
  return typeof value === 'number' ? value : 0;
}

/**
 * Parameters implied by the graph, counted from first principles.
 *
 * Written deliberately in terms of what the *nodes* say, not what the config
 * said, so that a graph which drops or mislabels a block fails here even though
 * the config was read correctly.
 */
function countParams(nodes: CanvasNode[]): number {
  const embedding = node(nodes, 'token_embedding');
  const width = p(embedding, 'hidden_size');
  const vocab = p(embedding, 'vocab_size');

  let total = vocab * width;

  const learnedPos = node(nodes, 'pos_absolute');
  if (learnedPos) total += p(learnedPos, 'max_length') * width;

  // Everything hanging off the stack is paid for once per layer.
  const stack = node(nodes, 'layer_stack');
  const layers = p(stack, 'num_layers');

  let perLayer = 0;

  const attention = node(nodes, 'gqa_attention') ?? node(nodes, 'mha_attention');
  if (attention) {
    const heads = p(attention, 'num_heads');
    const kvHeads = p(attention, 'num_kv_heads') || heads;
    const headDim = p(attention, 'head_dim') || width / heads;
    perLayer +=
      width * (heads * headDim) + // Q
      2 * width * (kvHeads * headDim) + // K, V
      (heads * headDim) * width; // O
  }

  const gated = node(nodes, 'ffn_gated');
  if (gated) perLayer += 3 * p(gated, 'hidden_size') * p(gated, 'intermediate_size');

  const standard = node(nodes, 'ffn_standard');
  if (standard) perLayer += 2 * p(standard, 'hidden_size') * p(standard, 'intermediate_size');

  const experts = node(nodes, 'moe_layer');
  if (experts) {
    const gatedExperts = experts.params?.activation === 'swiglu' ? 3 : 2;
    perLayer +=
      p(experts, 'num_experts') *
      gatedExperts *
      p(experts, 'hidden_size') *
      p(experts, 'intermediate_size');
  }

  const router = node(nodes, 'noisy_topk_router');
  if (router) perLayer += p(router, 'hidden_size') * p(router, 'num_experts');

  const shared = node(nodes, 'shared_expert');
  if (shared) {
    perLayer +=
      p(shared, 'num_experts') * 3 * p(shared, 'hidden_size') * p(shared, 'intermediate_size');
  }

  // Norms inside the block body; the final one sits on the trunk.
  const norms = [...allOf(nodes, 'rmsnorm'), ...allOf(nodes, 'layernorm')];
  const weightsPerNorm = allOf(nodes, 'rmsnorm').length > 0 ? 1 : 2;
  const bodyNorms = norms.filter((n) => n.name !== 'Final Norm');
  perLayer += bodyNorms.length * width * weightsPerNorm;

  total += perLayer * layers;
  total += width * weightsPerNorm; // final norm

  const head = node(nodes, 'lm_head');
  if (head && head.params?.tie_weights !== true) total += vocab * width;

  return total;
}

/** Published counts, and the tolerance an accounting at this level allows. */
const PUBLISHED: Array<{ name: string; config: object; params: number; tolerance: number }> = [
  { name: 'LLaMA-2 7B', config: LLAMA2_7B, params: 6.74e9, tolerance: 0.02 },
  { name: 'Mistral 7B', config: MISTRAL_7B, params: 7.24e9, tolerance: 0.02 },
  { name: 'Mixtral 8x7B', config: MIXTRAL_8X7B, params: 46.7e9, tolerance: 0.02 },
  { name: 'Qwen2 7B', config: QWEN2_7B, params: 7.62e9, tolerance: 0.02 },
  { name: 'Gemma 2B', config: GEMMA_2B, params: 2.51e9, tolerance: 0.02 },
  { name: 'GPT-2', config: GPT2_BASE, params: 124e6, tolerance: 0.02 },
  // BERT's published 110 M includes a pooler and token-type embeddings, which
  // this graph does not model; 5 % is the honest allowance for that.
  { name: 'BERT-base', config: BERT_BASE, params: 110e6, tolerance: 0.05 },
];

describe('HuggingFace config import', () => {
  describe('reproduces published parameter counts', () => {
    for (const { name, config, params, tolerance } of PUBLISHED) {
      it(name, () => {
        const result = parseHuggingFaceConfig(JSON.stringify(config));
        expect(result.error, `${name} failed to import`).toBeUndefined();

        const counted = countParams(result.nodes);
        const error = Math.abs(counted - params) / params;
        expect(
          error,
          `${name}: graph implies ${(counted / 1e9).toFixed(3)} B, published ${(params / 1e9).toFixed(3)} B (${(error * 100).toFixed(1)} % off)`,
        ).toBeLessThan(tolerance);
      });
    }
  });

  describe('reads the shape facts', () => {
    it('resolves grouped-query attention', () => {
      const { nodes } = parseHuggingFaceConfig(JSON.stringify(MISTRAL_7B));
      const attention = node(nodes, 'gqa_attention');
      expect(attention).toBeDefined();
      expect(attention!.params.num_heads).toBe(32);
      expect(attention!.params.num_kv_heads).toBe(8);
      expect(attention!.params.head_dim).toBe(128);
    });

    it('picks RMSNorm and rotary positions for a LLaMA-shaped config', () => {
      const { nodes } = parseHuggingFaceConfig(JSON.stringify(LLAMA2_7B));
      expect(node(nodes, 'rmsnorm')).toBeDefined();
      expect(node(nodes, 'layernorm')).toBeUndefined();
      expect(node(nodes, 'pos_rope')).toBeDefined();
      expect(node(nodes, 'ffn_gated')).toBeDefined();
      expect(node(nodes, 'ffn_standard')).toBeUndefined();
    });

    it('picks LayerNorm and learned positions for GPT-2', () => {
      const { nodes } = parseHuggingFaceConfig(JSON.stringify(GPT2_BASE));
      expect(node(nodes, 'layernorm')).toBeDefined();
      expect(node(nodes, 'rmsnorm')).toBeUndefined();
      expect(node(nodes, 'pos_absolute')).toBeDefined();
      expect(node(nodes, 'ffn_standard')).toBeDefined();
      expect(node(nodes, 'mha_attention')).toBeDefined();
    });

    it('reads GPT-2 aliases, and treats a null n_inner as 4x the width', () => {
      const { nodes, hardwareConfig } = parseHuggingFaceConfig(JSON.stringify(GPT2_BASE));
      expect(hardwareConfig?.hiddenDim).toBe(768);
      expect(hardwareConfig?.numLayers).toBe(12);
      expect(hardwareConfig?.numHeads).toBe(12);
      expect(p(node(nodes, 'ffn_standard'), 'intermediate_size')).toBe(3072);
    });

    it('ties GPT-2 embeddings by default, and leaves LLaMA untied when stated', () => {
      const gpt2 = parseHuggingFaceConfig(JSON.stringify(GPT2_BASE));
      expect(node(gpt2.nodes, 'lm_head')!.params.tie_weights).toBe(true);

      const llama = parseHuggingFaceConfig(JSON.stringify(LLAMA2_7B));
      expect(node(llama.nodes, 'lm_head')!.params.tie_weights).toBe(false);
    });

    it('gives an encoder a classification head rather than an LM head', () => {
      const { nodes, family } = parseHuggingFaceConfig(JSON.stringify(BERT_BASE));
      expect(node(nodes, 'classification_head')).toBeDefined();
      expect(node(nodes, 'lm_head')).toBeUndefined();
      expect(family).toBe('transformer');
      const attention = node(nodes, 'mha_attention');
      expect(attention!.params.causal).toBe(false);
    });

    it('reads a mixture of experts as the moe family', () => {
      const { nodes, family, hardwareConfig } = parseHuggingFaceConfig(
        JSON.stringify(MIXTRAL_8X7B),
      );
      expect(family).toBe('moe');
      expect(hardwareConfig?.numExperts).toBe(8);
      expect(hardwareConfig?.topK).toBe(2);

      const experts = node(nodes, 'moe_layer');
      expect(experts!.params.num_experts).toBe(8);
      // Mixtral states no `moe_intermediate_size`; the experts are the width of
      // the ordinary `intermediate_size`. The compiler reads `intermediate_size`
      // on every FFN-shaped block, not an expert-specific alias — see the note
      // on this key in huggingfaceImporter.ts.
      expect(experts!.params.intermediate_size).toBe(14336);
      // The compiler's active-parameters metric reads `top_k` off this node in
      // isolation, with no view of the router beside it — duplicated here for
      // the same reason `intermediate_size` is.
      expect(experts!.params.top_k).toBe(2);
      expect(node(nodes, 'ffn_gated'), 'a routed model has no dense FFN').toBeUndefined();
    });

    it('reads DeepSeek-style shared experts alongside the routed ones', () => {
      const deepseek = {
        model_type: 'deepseek_v2',
        hidden_size: 2048,
        intermediate_size: 10944,
        moe_intermediate_size: 1408,
        num_hidden_layers: 27,
        num_attention_heads: 16,
        num_key_value_heads: 16,
        n_routed_experts: 64,
        n_shared_experts: 2,
        num_experts_per_tok: 6,
        vocab_size: 102400,
        max_position_embeddings: 4096,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        hidden_act: 'silu',
      };
      const { nodes, hardwareConfig } = parseHuggingFaceConfig(JSON.stringify(deepseek));
      expect(p(node(nodes, 'moe_layer'), 'num_experts')).toBe(64);
      expect(p(node(nodes, 'moe_layer'), 'intermediate_size')).toBe(1408);
      expect(p(node(nodes, 'moe_layer'), 'top_k')).toBe(6);
      expect(p(node(nodes, 'shared_expert'), 'num_experts')).toBe(2);
      expect(p(node(nodes, 'noisy_topk_router'), 'top_k')).toBe(6);
      // No first_k_dense_replace stated — every layer routes, so there is no
      // separate dense-FFN block to represent.
      expect(node(nodes, 'ffn_gated'), 'no dense layers were declared').toBeUndefined();
      expect(hardwareConfig?.numDenseLayers).toBeUndefined();
    });

    it("reads DeepSeek's first_k_dense_replace as a separate dense-FFN block, at the model's real dense width", () => {
      // Real DeepSeek-MoE-16B: layer 0 is a plain dense FFN (width 10944,
      // not the 1408 the routed experts use), the other 27 route.
      const deepseek = {
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
        first_k_dense_replace: 1,
        vocab_size: 102400,
        max_position_embeddings: 4096,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        hidden_act: 'silu',
      };
      const { nodes, hardwareConfig: hw } = parseHuggingFaceConfig(JSON.stringify(deepseek));
      const denseFfn = node(nodes, 'ffn_gated');
      expect(denseFfn, 'a dense-FFN block for the first_k_dense_replace layers').toBeDefined();
      expect(p(denseFfn, 'intermediate_size')).toBe(10944);
      expect(p(denseFfn, 'hidden_size')).toBe(2048);
      // Still there, at the routed (much narrower) width — the two blocks
      // are not the same node, and neither took over the other's role.
      expect(p(node(nodes, 'moe_layer'), 'intermediate_size')).toBe(1408);
      expect(hw?.numDenseLayers).toBe(1);
    });

    it('reads the text tower of a multimodal config', () => {
      const result = parseHuggingFaceConfig(JSON.stringify(LLAVA_7B));
      expect(result.error).toBeUndefined();
      expect(p(node(result.nodes, 'token_embedding'), 'hidden_size')).toBe(4096);
      expect(p(node(result.nodes, 'layer_stack'), 'num_layers')).toBe(32);
      // 32064, the text tower's vocabulary — not the vision tower's 1024 width.
      expect(p(node(result.nodes, 'token_embedding'), 'vocab_size')).toBe(32064);
      expect(result.notes.some((n) => n.includes('text_config'))).toBe(true);
    });
  });

  describe('emits a graph the compiler can read', () => {
    it('closes both branches back onto the layer stack', () => {
      const { nodes, connections } = parseHuggingFaceConfig(JSON.stringify(LLAMA2_7B));
      const stack = node(nodes, 'layer_stack')!;
      const intoStack = connections.filter((c) => c.to === stack.id);
      // The positional block feeds it, and both residual branches close onto it.
      expect(intoStack.length).toBe(3);
      const residuals = allOf(nodes, 'residual_add').map((n) => n.id);
      for (const residual of residuals) {
        expect(connections.some((c) => c.from === residual && c.to === stack.id)).toBe(true);
      }
    });

    it('gives every node a unique id and every connection real endpoints', () => {
      for (const { name, config } of PUBLISHED) {
        const { nodes, connections } = parseHuggingFaceConfig(JSON.stringify(config));
        const ids = new Set(nodes.map((n) => n.id));
        expect(ids.size, `${name} has duplicate node ids`).toBe(nodes.length);

        const connIds = new Set(connections.map((c) => c.id));
        expect(connIds.size, `${name} has duplicate connection ids`).toBe(connections.length);

        for (const c of connections) {
          expect(ids.has(c.from), `${name}: connection from a missing node`).toBe(true);
          expect(ids.has(c.to), `${name}: connection to a missing node`).toBe(true);
        }
      }
    });

    it('runs from an input to an output', () => {
      const { nodes } = parseHuggingFaceConfig(JSON.stringify(LLAMA2_7B));
      expect(allOf(nodes, 'input')).toHaveLength(1);
      expect(allOf(nodes, 'output')).toHaveLength(1);
    });
  });

  describe('recognises what it is given', () => {
    it('accepts real configs', () => {
      for (const { name, config } of PUBLISHED) {
        expect(looksLikeHuggingFaceConfig(config), `${name} not recognised`).toBe(true);
      }
      expect(looksLikeHuggingFaceConfig(LLAVA_7B)).toBe(true);
    });

    it('does not mistake a NEURAX export for a config', () => {
      const neurax = {
        schema_version: '1.0',
        model: { name: 'x', type: 'transformer', layers: [{ id: 'a', type: 'dense' }] },
      };
      expect(looksLikeHuggingFaceConfig(neurax)).toBe(false);
    });

    it('rejects things that are not configs at all', () => {
      expect(looksLikeHuggingFaceConfig(null)).toBe(false);
      expect(looksLikeHuggingFaceConfig([1, 2, 3])).toBe(false);
      expect(looksLikeHuggingFaceConfig({ foo: 'bar' })).toBe(false);
      expect(looksLikeHuggingFaceConfig('a string')).toBe(false);
    });

    it('names the architecture for the dialog', () => {
      expect(describeHuggingFaceConfig(JSON.stringify(LLAMA2_7B))).toBe('LlamaForCausalLM');
      expect(describeHuggingFaceConfig('not json')).toBeNull();
    });
  });

  describe('fails in a way the user can act on', () => {
    it('reports invalid JSON rather than throwing', () => {
      const result = parseHuggingFaceConfig('{ definitely not json');
      expect(result.error).toMatch(/not valid json/i);
      expect(result.nodes).toEqual([]);
    });

    it('says which field is missing', () => {
      const result = parseHuggingFaceConfig(JSON.stringify({ model_type: 'llama' }));
      expect(result.error).toMatch(/hidden size/i);
    });

    it('reports a config with a width but no depth', () => {
      const result = parseHuggingFaceConfig(
        JSON.stringify({ model_type: 'llama', hidden_size: 4096 }),
      );
      expect(result.error).toMatch(/layer count/i);
    });

    it('explains an abbreviated multimodal text_config instead of the generic "no hidden size" error', () => {
      // A real config.json from llava-hf/llava-1.5-7b-hf: its `text_config`
      // only overrides what differs from lmsys/vicuna-7b-v1.5, relying on
      // that base checkpoint's own defaults for hidden_size and friends —
      // fields this importer has no way to resolve from a name string alone.
      // Falling through to the top level (which has no hidden_size either,
      // for an entirely unrelated reason) used to give the same unhelpful
      // "no hidden size in this config" message as a config missing the
      // field outright.
      const llava = {
        architectures: ['LlavaForConditionalGeneration'],
        model_type: 'llava',
        text_config: {
          _name_or_path: 'lmsys/vicuna-7b-v1.5',
          architectures: ['LlamaForCausalLM'],
          max_position_embeddings: 4096,
          model_type: 'llama',
          rms_norm_eps: 1e-5,
          vocab_size: 32064,
        },
        vision_config: { hidden_size: 1024, model_type: 'clip_vision_model' },
      };
      const result = parseHuggingFaceConfig(JSON.stringify(llava));
      expect(result.error).toMatch(/text_config/);
      expect(result.error).toMatch(/lmsys\/vicuna-7b-v1\.5/);
      expect(result.error).not.toMatch(/^no hidden size/i);
    });

    it('still gives the generic message when a text_config key is absent entirely', () => {
      // No text_config at all — the ordinary "missing field" error still
      // applies; only a *present but incomplete* one gets the specific one.
      const result = parseHuggingFaceConfig(JSON.stringify({ model_type: 'llava' }));
      expect(result.error).toMatch(/hidden size/i);
      expect(result.error).not.toMatch(/text_config/);
    });

    it('records the defaults it had to invent', () => {
      const sparse = {
        model_type: 'llama',
        hidden_size: 4096,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        rms_norm_eps: 1e-5,
      };
      const result = parseHuggingFaceConfig(JSON.stringify(sparse));
      expect(result.error).toBeUndefined();
      expect(result.assumptions.length).toBeGreaterThan(0);
      expect(result.assumptions.join(' ')).toMatch(/vocabulary/i);
    });

    it('says so when it had to guess the feed-forward from the activation', () => {
      const unknown = {
        model_type: 'some_new_llm',
        hidden_size: 4096,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        intermediate_size: 11008,
        vocab_size: 32000,
        hidden_act: 'silu',
      };
      const result = parseHuggingFaceConfig(JSON.stringify(unknown));
      expect(node(result.nodes, 'ffn_gated')).toBeDefined();
      expect(result.notes.join(' ')).toMatch(/unrecognised model_type/i);
    });
  });
});

// ── Mamba / Mamba-2 — real configs fetched live from the Hub ────────────────

const MAMBA_2_8B = {
  architectures: ['MambaForCausalLM'],
  model_type: 'mamba',
  hidden_size: 2560,
  num_hidden_layers: 64,
  vocab_size: 50280,
  state_size: 16,
  expand: 2,
  conv_kernel: 4,
  layer_norm_epsilon: 1e-5,
};

const MAMBA_130M = {
  architectures: ['MambaForCausalLM'],
  _name_or_path: 'state-spaces/mamba-130m-hf',
  model_type: 'mamba',
  hidden_size: 768,
  num_hidden_layers: 24,
  vocab_size: 50280,
  state_size: 16,
  expand: 2,
  conv_kernel: 4,
};

const MAMBA2_CODESTRAL_7B = {
  architectures: ['Mamba2ForCausalLM'],
  _name_or_path: 'mistralai/Mamba-Codestral-7B-v0.1',
  model_type: 'mamba2',
  hidden_size: 4096,
  num_hidden_layers: 64,
  vocab_size: 32768,
  state_size: 128,
  expand: 2,
  conv_kernel: 4,
  num_heads: 128,
  head_dim: 64,
  tie_word_embeddings: false,
};

/**
 * Independent of `countParams` above (attention-shaped, doesn't apply here)
 * and of `neurax_formulas::ssm::mamba_params` (the Rust implementation this
 * checks against) — reimplemented from the same published block structure
 * (in_proj, conv1d, the A/B/C/D SSM parameters, out_proj) rather than
 * imported from either, so a bug shared between the graph and one formula
 * still has a chance of being caught here.
 */
function countMambaParams(nodes: CanvasNode[]): number {
  const embedding = node(nodes, 'token_embedding');
  const width = p(embedding, 'hidden_size');
  const vocab = p(embedding, 'vocab_size');

  const block = node(nodes, 'mamba_block');
  const stateDim = p(block, 'state_dim');
  const expansion = p(block, 'expansion_factor');
  const dInner = width * expansion;

  const inProj = width * (dInner * 2);
  const conv1d = dInner * 4;
  const ssm = dInner * stateDim * 3 + dInner;
  const outProj = dInner * width;
  const perBlock = inProj + conv1d + ssm + outProj;

  const stack = node(nodes, 'layer_stack');
  const layers = p(stack, 'num_layers');

  // One RMSNorm inside the block body, one on the trunk after the stack.
  const bodyNormWeights = width;
  const finalNormWeights = width;

  let total = vocab * width; // embedding
  total += layers * (perBlock + bodyNormWeights);
  total += finalNormWeights;

  const head = node(nodes, 'lm_head');
  if (head && head.params?.tie_weights !== true) total += vocab * width;

  return total;
}

describe('Mamba / Mamba-2 import', () => {
  it('reproduces the parameter count already verified against the published 2.8B Mamba (±10%, matching neurax-core/tests/published_model_accuracy.rs)', () => {
    const result = parseHuggingFaceConfig(JSON.stringify(MAMBA_2_8B));
    expect(result.error).toBeUndefined();
    expect(result.family).toBe('ssm');

    const counted = countMambaParams(result.nodes);
    const published = 2.8e9;
    const error = Math.abs(counted - published) / published;
    expect(
      error,
      `graph implies ${(counted / 1e9).toFixed(3)} B, published ${(published / 1e9).toFixed(1)} B (${(error * 100).toFixed(1)} % off)`,
    ).toBeLessThan(0.1);
  });

  it('reads the real 24-layer shape of Mamba-130M, not a guessed one', () => {
    // The reference template this importer replaces stated 12 layers for
    // this size — half the real config's 24. A wrong layer count is not
    // recoverable by any downstream formula; catching it here is cheaper
    // than catching it in a parameter-count assertion three steps later.
    const result = parseHuggingFaceConfig(JSON.stringify(MAMBA_130M));
    expect(result.error).toBeUndefined();
    const stack = node(result.nodes, 'layer_stack');
    expect(p(stack, 'num_layers')).toBe(24);
    expect(result.modelName).toBe('state-spaces/mamba-130m-hf');
  });

  it('carries exactly one parameter-bearing SSM node per layer, not several', () => {
    // The bug this graph shape replaces: two nodes each independently
    // carrying the block's full formula (double-counted) plus a third
    // carrying none of it. One node, one formula, once.
    const result = parseHuggingFaceConfig(JSON.stringify(MAMBA_2_8B));
    expect(allOf(result.nodes, 'mamba_block')).toHaveLength(1);
    expect(allOf(result.nodes, 'state_space')).toHaveLength(0);
  });

  it('imports Mamba-2 (SSD) and notes the approximation honestly', () => {
    const result = parseHuggingFaceConfig(JSON.stringify(MAMBA2_CODESTRAL_7B));
    expect(result.error).toBeUndefined();
    expect(result.family).toBe('ssm');
    const block = node(result.nodes, 'mamba_block');
    expect(p(block, 'state_dim')).toBe(128);
    expect(result.notes.join(' ')).toMatch(/approximation/i);

    const head = node(result.nodes, 'lm_head');
    expect(head?.params?.tie_weights).toBe(false);
  });

  it('is additive: importing a Mamba config never touches the transformer path', () => {
    const result = parseHuggingFaceConfig(JSON.stringify(LLAMA2_7B));
    expect(result.family).toBe('transformer');
    expect(node(result.nodes, 'mamba_block')).toBeUndefined();
  });
});

describe('honest refusal of families NEURAX does not import', () => {
  // ResNet/RegNet/ConvNeXt/EfficientNet moved out of this list once real,
  // live-verified support was added — see the "CNN import" describe block
  // below for their (now positive) coverage. MobileNet and ConvNeXt V2
  // stay refused: no live config was read for either while building the
  // CNN importer.
  const cases: Array<{ name: string; config: object; match: RegExp }> = [
    { name: 'MobileNet', config: { model_type: 'mobilenet_v2', num_channels: 3 }, match: /CNN \(MobileNet\)/ },
    { name: 'ViT', config: { model_type: 'vit', image_size: 224 }, match: /vision transformer/ },
    { name: 'Whisper', config: { model_type: 'whisper', num_mel_bins: 80 }, match: /speech \(Whisper\)/ },
    { name: 'RWKV', config: { model_type: 'rwkv', vocab_size: 50277 }, match: /RWKV/ },
  ];

  for (const { name, config, match } of cases) {
    it(`names ${name} rather than reporting a missing field`, () => {
      const result = parseHuggingFaceConfig(JSON.stringify(config));
      expect(result.error).toBeDefined();
      expect(result.error).toMatch(match);
      expect(result.error).toMatch(/NEURAX imports transformer and MoE language models, Mamba\/Mamba-2, and ResNet\/RegNet\/ConvNeXt\/EfficientNet CNNs/);
    });
  }

  it('names a diffusers pipeline component NEURAX still refuses (the VAE, not the UNet)', () => {
    const vae = {
      _class_name: 'AutoencoderKL',
      _diffusers_version: '0.21.0',
      in_channels: 3,
      out_channels: 3,
    };
    const result = parseHuggingFaceConfig(JSON.stringify(vae));
    expect(result.error).toMatch(/AutoencoderKL/);
    expect(result.error).toMatch(/diffusers pipeline component/);
  });

  it('still falls through to the generic error for a truly unlisted family', () => {
    const result = parseHuggingFaceConfig(JSON.stringify({ model_type: 'some_future_cnn' }));
    expect(result.error).toMatch(/no hidden size/i);
  });
});

describe('CNN import — real configs fetched live from the Hub', () => {
  // microsoft/resnet-50, fetched live while building this importer.
  const RESNET50 = {
    architectures: ['ResNetForImageClassification'],
    depths: [3, 4, 6, 3],
    downsample_in_first_stage: false,
    embedding_size: 64,
    hidden_act: 'relu',
    hidden_sizes: [256, 512, 1024, 2048],
    model_type: 'resnet',
    num_channels: 3,
  };

  // facebook/convnext-tiny-224, fetched live.
  const CONVNEXT_TINY = {
    architectures: ['ConvNextForImageClassification'],
    depths: [3, 3, 9, 3],
    hidden_act: 'gelu',
    hidden_sizes: [96, 192, 384, 768],
    model_type: 'convnext',
    num_channels: 3,
    patch_size: 4,
  };

  // google/efficientnet-b0, fetched live.
  const EFFICIENTNET_B0 = {
    model_type: 'efficientnet',
    num_channels: 3,
    hidden_act: 'swish',
    in_channels: [32, 16, 24, 40, 80, 112, 192],
    out_channels: [16, 24, 40, 80, 112, 192, 320],
    kernel_sizes: [3, 3, 5, 3, 5, 5, 3],
    expand_ratios: [1, 6, 6, 6, 6, 6, 6],
    num_block_repeats: [1, 2, 2, 3, 3, 4, 1],
    strides: [1, 2, 2, 2, 1, 2, 1],
    squeeze_expansion_ratio: 0.25,
  };

  it('imports ResNet-50 as 4 bottleneck stages, no unresolved layer types', () => {
    const result = parseHuggingFaceConfig(JSON.stringify(RESNET50), 'resnet-50');
    expect(result.error).toBeUndefined();
    expect(result.family).toBe('cnn');
    const stages = result.nodes.filter((n) => n.type === 'bottleneck_block');
    expect(stages.map((s) => s.params.blocks)).toEqual([3, 4, 6, 3]);
    expect(stages.map((s) => s.params.planes)).toEqual([256, 512, 1024, 2048]);
  });

  it('imports ConvNeXt-Tiny as 4 convnext_block stages', () => {
    const result = parseHuggingFaceConfig(JSON.stringify(CONVNEXT_TINY), 'convnext-tiny');
    expect(result.error).toBeUndefined();
    const stages = result.nodes.filter((n) => n.type === 'convnext_block');
    expect(stages.map((s) => s.params.dim)).toEqual([96, 192, 384, 768]);
    expect(stages.map((s) => s.params.num_blocks)).toEqual([3, 3, 9, 3]);
  });

  it('imports EfficientNet-B0 as 7 mbconv_block stages with real per-stage shapes', () => {
    const result = parseHuggingFaceConfig(JSON.stringify(EFFICIENTNET_B0), 'efficientnet-b0');
    expect(result.error).toBeUndefined();
    const stages = result.nodes.filter((n) => n.type === 'mbconv_block');
    expect(stages).toHaveLength(7);
    expect(stages[0].params).toMatchObject({ in_channels: 32, out_channels: 16, expand_ratio: 1, num_blocks: 1 });
    expect(stages[6].params).toMatchObject({ in_channels: 192, out_channels: 320, expand_ratio: 6, num_blocks: 1 });
  });

  it('a config missing the stage arrays reports a real error, not a wrong shape', () => {
    const result = parseHuggingFaceConfig(JSON.stringify({ model_type: 'resnet', num_channels: 3 }));
    expect(result.error).toMatch(/depths\/hidden_sizes/);
  });
});

describe('Diffusion (UNet) import — real config fetched live from the Hub', () => {
  // runwayml/stable-diffusion-v1-5's unet/config.json, fetched live while
  // building this importer.
  const SD15_UNET = {
    _class_name: 'UNet2DConditionModel',
    _diffusers_version: '0.6.0',
    act_fn: 'silu',
    attention_head_dim: 8,
    block_out_channels: [320, 640, 1280, 1280],
    center_input_sample: false,
    cross_attention_dim: 768,
    down_block_types: ['CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'],
    in_channels: 4,
    layers_per_block: 2,
    out_channels: 4,
    sample_size: 64,
    up_block_types: ['UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'],
  };

  it('imports the real SD 1.5 UNet as encoder/mid/decoder with the real channel schedule', () => {
    const result = parseHuggingFaceConfig(JSON.stringify(SD15_UNET), 'sd-1.5-unet');
    expect(result.error).toBeUndefined();
    expect(result.family).toBe('diffusion');

    const encoder = result.nodes.find((n) => n.type === 'unet_encoder')!;
    expect(encoder.params.channels).toEqual([320, 640, 1280, 1280]);
    expect(encoder.params.num_res_blocks).toBe(2);
    expect(encoder.params.cross_attention_dim).toBe(768);

    const decoder = result.nodes.find((n) => n.type === 'unet_decoder')!;
    expect(decoder.params.channels).toEqual([1280, 1280, 640, 320]);

    const mid = result.nodes.find((n) => n.type === 'unet_mid')!;
    expect(mid.params.channels).toBe(1280);
  });

  it('is honest that only the UNet is counted, not the full pipeline', () => {
    const result = parseHuggingFaceConfig(JSON.stringify(SD15_UNET), 'sd-1.5-unet');
    expect(result.notes.join(' ')).toMatch(/VAE and text encoder are separate/);
  });

  it('still refuses a UNet2DModel/UNet2DConditionModel config missing block_out_channels', () => {
    const result = parseHuggingFaceConfig(JSON.stringify({ _class_name: 'UNet2DConditionModel' }));
    expect(result.error).toMatch(/block_out_channels/);
  });
});
