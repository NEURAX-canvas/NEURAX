/**
 * Reading a HuggingFace `config.json` into a NEURAX design.
 *
 * This is the door into the tool for a model that already exists. Almost every
 * open-weights model on the Hub ships a `config.json` that fully determines its
 * shape — widths, depth, head counts, expert counts — and until now the only
 * way to get one of those models onto the canvas was to rebuild it block by
 * block, or to hope a reference template happened to match.
 *
 * What is produced here is deliberately the *same* node graph the reference
 * templates in `data/modelTemplates.ts` use: input → embedding → positional →
 * `layer_stack` with the block body hanging off it → final norm → head. That
 * matters more than it looks. The templates are the shape the compiler has been
 * measured against (`modelTemplates.accuracy.test.ts`, and the Rust-side
 * `published_model_accuracy.rs`), so emitting the same shape means an imported
 * LLaMA is analysed by exactly the paths a template LLaMA is. A different but
 * "equivalent" graph would be a second, unmeasured encoding of the same model.
 *
 * The mapping is a best effort over a format that is only loosely a standard:
 * `config.json` is whatever the modelling code for that architecture happens to
 * read. Rather than pretend to a catalogue of every architecture, this reads
 * the fields that are near-universal, keys the handful of real structural
 * choices (gated vs standard FFN, RMS vs LayerNorm, RoPE vs learned positions)
 * off evidence in the config itself, and reports what it concluded so the user
 * can see whether it guessed right.
 */

import { CanvasNode, Connection, LayerType } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { HardwareConfig } from '@/contexts/HardwareContext.tsx';
import { ImportResult } from '@/utils/architectureImporter.ts';

/** A `config.json` as parsed — every field optional, because every field is. */
type RawConfig = Record<string, unknown>;

/** The shape facts an importable config yields, once the aliases are resolved. */
interface ModelShape {
  modelType: string;
  architecture: string | null;
  name: string;

  hiddenSize: number;
  numLayers: number;
  numHeads: number;
  numKvHeads: number;
  headDim: number;
  intermediateSize: number;
  vocabSize: number;
  maxPositions: number;

  normEps: number;
  ropeTheta: number | null;
  activation: string;
  tieWordEmbeddings: boolean;

  /** Present only for mixture-of-experts configs. */
  moe: {
    numExperts: number;
    topK: number;
    expertIntermediateSize: number;
    sharedExperts: number;
  } | null;

  /** Encoder models get a classification head, decoders an LM head. */
  causal: boolean;
}

/** The structural choices, derived from the config rather than hard-coded. */
interface Traits {
  norm: Extract<LayerType, 'rmsnorm' | 'layernorm'>;
  positional: Extract<LayerType, 'pos_rope' | 'pos_absolute'>;
  ffn: Extract<LayerType, 'ffn_gated' | 'ffn_standard'>;
  attention: Extract<LayerType, 'gqa_attention' | 'mha_attention'>;
  head: Extract<LayerType, 'lm_head' | 'classification_head'>;
}

/** Outcome of an import, with the reasoning the UI shows back to the user. */
export interface HuggingFaceImportResult extends ImportResult {
  /** Human-readable lines describing what was read and what was inferred. */
  notes: string[];
  /** Fields the config did not supply, which therefore fell back to a default. */
  assumptions: string[];
}

// ── Reading loosely-specified fields ────────────────────────────────────────

/**
 * First numeric value among a set of alias keys.
 *
 * `config.json` names the same quantity differently per architecture family —
 * `hidden_size` in LLaMA, `n_embd` in GPT-2, `d_model` in T5 — and a config may
 * carry a key explicitly set to `null` (GPT-2's `n_inner`), which must be
 * treated as absent rather than as zero.
 */
function num(config: RawConfig, ...keys: string[]): number | undefined {
  for (const key of keys) {
    const value = config[key];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return undefined;
}

function str(config: RawConfig, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = config[key];
    if (typeof value === 'string' && value.length > 0) return value;
  }
  return undefined;
}

function bool(config: RawConfig, ...keys: string[]): boolean | undefined {
  for (const key of keys) {
    const value = config[key];
    if (typeof value === 'boolean') return value;
  }
  return undefined;
}

/**
 * The sub-config that actually describes the language model.
 *
 * Multimodal releases (LLaVA, Gemma 3, Qwen-VL, Idefics) put the text tower
 * under `text_config` and leave the top level holding only wiring. Reading the
 * top level there yields a model with no layers at all.
 */
function textConfig(config: RawConfig): { config: RawConfig; nested: boolean } {
  for (const key of ['text_config', 'llm_config', 'language_config']) {
    const nested = config[key];
    if (
      nested &&
      typeof nested === 'object' &&
      !Array.isArray(nested) &&
      num(nested as RawConfig, 'hidden_size', 'n_embd', 'd_model') !== undefined
    ) {
      return { config: nested as RawConfig, nested: true };
    }
  }
  return { config, nested: false };
}

/** Whether this JSON is a HuggingFace config rather than a NEURAX design. */
export function looksLikeHuggingFaceConfig(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const config = value as RawConfig;

  // A NEURAX export always carries `model.layers`; never confuse the two.
  const model = config.model;
  if (model && typeof model === 'object' && Array.isArray((model as RawConfig).layers)) {
    return false;
  }

  if (typeof config.model_type === 'string') return true;
  if (Array.isArray(config.architectures)) return true;

  // Some configs (older GPT-2 forks, hand-written ones) carry neither, but no
  // real config lacks both a width and a depth.
  const { config: text } = textConfig(config);
  const hasWidth = num(text, 'hidden_size', 'n_embd', 'd_model') !== undefined;
  const hasDepth = num(text, 'num_hidden_layers', 'n_layer', 'num_layers') !== undefined;
  return hasWidth && hasDepth;
}

// ── Interpreting the shape ──────────────────────────────────────────────────

/** Architectures whose head count is fixed, not the config's business. */
const ENCODER_TYPES = new Set([
  'bert', 'roberta', 'distilbert', 'albert', 'electra', 'deberta', 'deberta-v2',
  'xlm-roberta', 'camembert', 'mpnet',
]);

/** Activations that only ever appear on a gated (SwiGLU/GeGLU) feed-forward. */
const GATED_ACTIVATIONS = new Set(['silu', 'swish', 'gelu_pytorch_tanh']);

/**
 * Architectures with a gated feed-forward despite a non-gated activation name,
 * or with a gated one despite carrying a LayerNorm.
 *
 * The heuristic below gets the modern LLaMA-shaped families right from the
 * activation alone; this table covers the ones where the config gives no
 * usable signal.
 */
const GATED_FFN_TYPES = new Set([
  'llama', 'mistral', 'mixtral', 'qwen2', 'qwen3', 'qwen2_moe', 'qwen3_moe',
  'gemma', 'gemma2', 'gemma3', 'gemma3_text', 'phi3', 'olmo', 'olmo2',
  'starcoder2', 'stablelm', 'deepseek', 'deepseek_v2', 'deepseek_v3',
  'granite', 'granitemoe', 'internlm2', 'baichuan', 'yi', 'cohere', 'cohere2',
  'glm', 'glm4', 'exaone', 'minicpm', 'nemotron', 'zamba',
]);

/** Architectures known to use a standard two-matrix feed-forward. */
const STANDARD_FFN_TYPES = new Set([
  'gpt2', 'gpt_neo', 'gpt_neox', 'gptj', 'bloom', 'opt', 'falcon', 'mpt',
  'codegen', 'phi', 'bert', 'roberta', 'distilbert', 'albert', 'electra',
  'xlm-roberta', 'deberta', 'deberta-v2',
]);

function deriveTraits(shape: ModelShape, config: RawConfig): { traits: Traits; notes: string[] } {
  const notes: string[] = [];

  // RMSNorm is announced by the epsilon field the modelling code reads; a
  // config carrying `rms_norm_eps` is not using LayerNorm.
  const hasRmsEps = num(config, 'rms_norm_eps') !== undefined;
  const norm: Traits['norm'] = hasRmsEps ? 'rmsnorm' : 'layernorm';

  // RoPE leaves two possible traces: an explicit theta, or a scaling config.
  const hasRope =
    num(config, 'rope_theta', 'rotary_emb_base') !== undefined ||
    config.rope_scaling != null ||
    bool(config, 'rotary') === true ||
    num(config, 'rotary_dim', 'rotary_pct') !== undefined;
  const positional: Traits['positional'] = hasRope ? 'pos_rope' : 'pos_absolute';

  // Gating: the explicit tables first, then the activation as evidence.
  let ffn: Traits['ffn'];
  if (GATED_FFN_TYPES.has(shape.modelType)) {
    ffn = 'ffn_gated';
  } else if (STANDARD_FFN_TYPES.has(shape.modelType)) {
    ffn = 'ffn_standard';
  } else if (GATED_ACTIVATIONS.has(shape.activation)) {
    ffn = 'ffn_gated';
    notes.push(
      `Unrecognised model_type "${shape.modelType}"; the ${shape.activation} activation indicates a gated feed-forward.`,
    );
  } else {
    ffn = 'ffn_standard';
    notes.push(
      `Unrecognised model_type "${shape.modelType}"; assumed a standard feed-forward from the ${shape.activation} activation.`,
    );
  }

  // A config that states a KV head count is from a family that supports
  // grouped-query attention, whether or not this checkpoint groups anything.
  const attention: Traits['attention'] =
    num(config, 'num_key_value_heads', 'num_kv_heads', 'n_kv_heads') !== undefined
      ? 'gqa_attention'
      : 'mha_attention';

  const head: Traits['head'] = shape.causal ? 'lm_head' : 'classification_head';

  return { traits: { norm, positional, ffn, attention, head }, notes };
}

function readShape(
  config: RawConfig,
  fallbackName: string,
): { shape: ModelShape; assumptions: string[] } | { error: string } {
  const assumptions: string[] = [];

  const hiddenSize = num(config, 'hidden_size', 'n_embd', 'd_model', 'dim', 'hidden_dim');
  if (hiddenSize === undefined || hiddenSize <= 0) {
    return {
      error:
        'No hidden size in this config (looked for hidden_size, n_embd, d_model). ' +
        'If this is a multimodal model, its text tower may be under a key this importer does not know.',
    };
  }

  const numLayers = num(config, 'num_hidden_layers', 'n_layer', 'num_layers', 'n_layers');
  if (numLayers === undefined || numLayers <= 0) {
    return {
      error:
        'No layer count in this config (looked for num_hidden_layers, n_layer, num_layers).',
    };
  }

  let numHeads = num(config, 'num_attention_heads', 'n_head', 'num_heads', 'n_heads');
  if (numHeads === undefined || numHeads <= 0) {
    // Every attention model has heads; a config omitting the count is unusual
    // enough to say so rather than silently picking one.
    numHeads = Math.max(1, Math.round(hiddenSize / 64));
    assumptions.push(`Head count absent; assumed ${numHeads} (one head per 64 channels).`);
  }

  const numKvHeads =
    num(config, 'num_key_value_heads', 'num_kv_heads', 'n_kv_heads') ?? numHeads;

  const headDim = num(config, 'head_dim') ?? Math.floor(hiddenSize / numHeads);

  const modelType = (str(config, 'model_type') ?? '').toLowerCase();
  const architectures = Array.isArray(config.architectures)
    ? (config.architectures as unknown[]).filter((a): a is string => typeof a === 'string')
    : [];
  const architecture = architectures[0] ?? null;

  // MoE, under each family's spelling of the same three numbers.
  const numExperts = num(
    config,
    'num_local_experts', 'num_experts', 'n_routed_experts', 'moe_num_experts', 'num_expert',
  );
  const moe =
    numExperts !== undefined && numExperts > 1
      ? {
          numExperts,
          topK: num(config, 'num_experts_per_tok', 'top_k', 'moe_top_k', 'num_selected_experts') ?? 2,
          expertIntermediateSize:
            num(config, 'moe_intermediate_size', 'expert_intermediate_size', 'ffn_dim') ??
            num(config, 'intermediate_size') ??
            hiddenSize * 4,
          sharedExperts: num(config, 'n_shared_experts', 'num_shared_experts') ?? 0,
        }
      : null;

  if (moe && num(config, 'num_experts_per_tok', 'top_k', 'moe_top_k', 'num_selected_experts') === undefined) {
    assumptions.push('Router top-k absent; assumed top-2, the usual choice.');
  }

  let intermediateSize = num(config, 'intermediate_size', 'n_inner', 'ffn_hidden_size', 'ffn_dim');
  if (intermediateSize === undefined || intermediateSize <= 0) {
    // GPT-2 writes `"n_inner": null` and means 4× the width.
    intermediateSize = hiddenSize * 4;
    assumptions.push(`Feed-forward width absent; assumed 4× hidden size (${intermediateSize}).`);
  }

  let vocabSize = num(config, 'vocab_size', 'padded_vocab_size');
  if (vocabSize === undefined || vocabSize <= 0) {
    vocabSize = 32000;
    assumptions.push('Vocabulary size absent; assumed 32000.');
  }

  let maxPositions = num(config, 'max_position_embeddings', 'n_positions', 'max_seq_len', 'seq_length');
  if (maxPositions === undefined || maxPositions <= 0) {
    maxPositions = 2048;
    assumptions.push('Context length absent; assumed 2048.');
  }

  const normEps =
    num(config, 'rms_norm_eps', 'layer_norm_epsilon', 'layer_norm_eps', 'norm_eps') ?? 1e-5;

  const ropeTheta = num(config, 'rope_theta', 'rotary_emb_base') ?? null;

  const activation = (
    str(config, 'hidden_act', 'activation_function', 'hidden_activation', 'activation') ?? 'gelu'
  ).toLowerCase();

  // Weight tying defaults to true in HuggingFace's base config, and the small
  // models that rely on it (GPT-2, Gemma) do not restate it. Getting this wrong
  // misses or double-counts a vocab × width matrix, which on a small model is
  // most of the parameters.
  const tieWordEmbeddings = bool(config, 'tie_word_embeddings', 'tie_weights') ?? true;

  // Decided from the model type first, and only then from the class name.
  //
  // A bare `…Model` suffix is not evidence of an encoder: `LlamaModel`,
  // `MistralModel` and `Qwen2Model` are the base checkpoints of decoders,
  // published without their LM head. Treating them as encoders would give them
  // a classification head and mark their attention non-causal. Only the task
  // suffixes that genuinely mean an encoder are read as one.
  const isEncoder =
    ENCODER_TYPES.has(modelType) ||
    (architecture !== null &&
      /For(MaskedLM|SequenceClassification|TokenClassification|QuestionAnswering|MultipleChoice)$/.test(
        architecture,
      ));

  const name =
    str(config, '_name_or_path', 'name_or_path') ??
    architecture ??
    (modelType ? modelType : fallbackName);

  return {
    shape: {
      modelType: modelType || 'unknown',
      architecture,
      name,
      hiddenSize,
      numLayers,
      numHeads,
      numKvHeads,
      headDim,
      intermediateSize,
      vocabSize,
      maxPositions,
      normEps,
      ropeTheta,
      activation,
      tieWordEmbeddings,
      moe,
      causal: !isEncoder,
    },
    assumptions,
  };
}

// ── Building the graph ──────────────────────────────────────────────────────

/**
 * Lay out the design the way the reference templates do.
 *
 * The trunk runs left to right along y = 140; the repeated block body sits
 * above (attention) and below (feed-forward) it, and both branches close back
 * onto the `layer_stack` node. This is not decoration — the compiler reads the
 * stack's `num_layers` and the bodies hanging off it, so a body that is not
 * connected back to the stack is counted once instead of `num_layers` times.
 */
function buildGraph(
  shape: ModelShape,
  traits: Traits,
): { nodes: CanvasNode[]; connections: Connection[] } {
  const nodes: CanvasNode[] = [];
  const connections: Connection[] = [];
  let nodeSeq = 0;
  let connSeq = 0;

  const add = (
    type: LayerType,
    name: string,
    x: number,
    y: number,
    params: Record<string, unknown>,
  ): string => {
    const id = `hf-n${++nodeSeq}`;
    nodes.push({ id, type, name, x, y, params: params as CanvasNode['params'] });
    return id;
  };

  const link = (from: string, to: string) => {
    connections.push({ id: `hf-c${++connSeq}`, from, to });
  };

  const width = shape.hiddenSize;
  const TRUNK = 140;
  const ATTN = 60;
  const FFN = 220;

  // ── Trunk: input → embedding → positional ──
  const input = add('input', 'Input Tokens', 50, TRUNK, {
    sequence_length: shape.maxPositions,
  });

  const embedding = add('token_embedding', 'Token Embedding', 250, TRUNK, {
    vocab_size: shape.vocabSize,
    hidden_size: width,
    ...(traits.positional === 'pos_absolute'
      ? { max_position_embeddings: shape.maxPositions }
      : {}),
  });
  link(input, embedding);

  const positional =
    traits.positional === 'pos_rope'
      ? add('pos_rope', 'RoPE', 450, TRUNK, {
          hidden_size: width,
          theta: shape.ropeTheta ?? 10000.0,
          max_length: shape.maxPositions,
        })
      : add('pos_absolute', 'Learned Positional Embedding', 450, TRUNK, {
          max_length: shape.maxPositions,
          hidden_size: width,
        });
  link(embedding, positional);

  const stackLabel = shape.moe
    ? `${shape.numLayers}× MoE Decoder Block`
    : `${shape.numLayers}× ${shape.causal ? 'Decoder' : 'Encoder'} Block`;
  const stack = add('layer_stack', stackLabel, 650, TRUNK, {
    num_layers: shape.numLayers,
  });
  link(positional, stack);

  // ── Attention branch ──
  const preAttnNorm = add(traits.norm, 'Pre-Attention Norm', 650, ATTN, {
    hidden_size: width,
    eps: shape.normEps,
  });
  const attention = add(
    traits.attention,
    shape.numKvHeads < shape.numHeads
      ? `GQA (${shape.numHeads} heads, ${shape.numKvHeads} KV)`
      : `Self-Attention (${shape.numHeads} heads)`,
    850,
    ATTN,
    {
      hidden_size: width,
      num_heads: shape.numHeads,
      num_kv_heads: shape.numKvHeads,
      head_dim: shape.headDim,
      causal: shape.causal,
    },
  );
  const attnResidual = add('residual_add', 'Residual Add', 1050, ATTN, {});
  link(preAttnNorm, attention);
  link(attention, attnResidual);
  link(attnResidual, stack);

  // ── Feed-forward branch, routed or dense ──
  const preFfnNorm = add(traits.norm, shape.moe ? 'Pre-MoE Norm' : 'Pre-FFN Norm', 650, FFN, {
    hidden_size: width,
    eps: shape.normEps,
  });

  let ffnTail: string;
  if (shape.moe) {
    const router = add('noisy_topk_router', `Top-${shape.moe.topK} Router`, 850, FFN, {
      num_experts: shape.moe.numExperts,
      top_k: shape.moe.topK,
      hidden_size: width,
    });
    const experts = add('moe_layer', `${shape.moe.numExperts}× Experts`, 1050, FFN, {
      num_experts: shape.moe.numExperts,
      // Duplicated from the router node above: the compiler's active-params
      // metric (how many parameters a token actually touches, not how many
      // the model owns) reads `top_k` off this node specifically, since it
      // computes each node's contribution in isolation with no view of the
      // router beside it. Left absent, it silently fell back to a default
      // of 2 — correct for Mixtral only because Mixtral's real top-k
      // happens to also be 2; DeepSeek's real 6 was invisible to it.
      top_k: shape.moe.topK,
      // The compiler reads `intermediate_size` on every FFN-shaped block —
      // an expert-specific alias here read as absent and silently fell back
      // to 4× hidden size, which is close enough to Mixtral's real ratio to
      // hide the bug and wildly wrong for DeepSeek's much narrower experts.
      intermediate_size: shape.moe.expertIntermediateSize,
      hidden_size: width,
      activation: traits.ffn === 'ffn_gated' ? 'swiglu' : shape.activation,
    });
    const combine = add('expert_combine', 'Expert Combine', 1250, FFN, {});
    link(preFfnNorm, router);
    link(router, experts);
    link(experts, combine);
    ffnTail = combine;

    if (shape.moe.sharedExperts > 0) {
      // DeepSeek-style always-on experts run beside the routed ones and add
      // real parameters; omitting them understates the model.
      const shared = add(
        'shared_expert',
        `${shape.moe.sharedExperts}× Shared Expert`,
        1050,
        FFN + 80,
        {
          num_experts: shape.moe.sharedExperts,
          intermediate_size: shape.moe.expertIntermediateSize,
          hidden_size: width,
        },
      );
      link(preFfnNorm, shared);
      link(shared, combine);
    }
  } else {
    ffnTail = add(
      traits.ffn,
      traits.ffn === 'ffn_gated' ? 'Gated FFN (SwiGLU)' : 'FFN (2× Linear)',
      850,
      FFN,
      {
        hidden_size: width,
        intermediate_size: shape.intermediateSize,
        activation: shape.activation,
      },
    );
    link(preFfnNorm, ffnTail);
  }

  const ffnResidual = add('residual_add', 'Residual Add', 1450, FFN, {});
  link(ffnTail, ffnResidual);
  link(ffnResidual, stack);

  // ── Trunk again: final norm → head → output ──
  const finalNorm = add(traits.norm, 'Final Norm', 1650, TRUNK, {
    hidden_size: width,
    eps: shape.normEps,
  });
  link(stack, finalNorm);

  const head =
    traits.head === 'lm_head'
      ? add('lm_head', shape.tieWordEmbeddings ? 'LM Head (tied)' : 'LM Head', 1850, TRUNK, {
          vocab_size: shape.vocabSize,
          hidden_size: width,
          tie_weights: shape.tieWordEmbeddings,
          // The output projection's shape, stated explicitly.
          //
          // The compiler lowers `lm_head` to a dense layer and sizes it from
          // `in_features` × `out_features`. Without them it has only a vocab
          // size and a width, and counts nothing — which on a model with a
          // large vocabulary is most of the difference between right and
          // wrong: Qwen2 7B came out at 7.09 B instead of 7.62 B, exactly one
          // missing 152064 × 3584 matrix. Untied heads are omitted, because a
          // tied head genuinely has no weights of its own.
          ...(shape.tieWordEmbeddings
            ? {}
            : { in_features: width, out_features: shape.vocabSize }),
        })
      : add('classification_head', 'Classification Head', 1850, TRUNK, {
          num_labels: 2,
          hidden_size: width,
        });
  link(finalNorm, head);

  const output = add('output', 'Output', 2050, TRUNK, {});
  link(head, output);

  return { nodes, connections };
}

function buildHardwareConfig(shape: ModelShape): Partial<HardwareConfig> {
  const config: Partial<HardwareConfig> = {
    hiddenDim: shape.hiddenSize,
    numLayers: shape.numLayers,
    numHeads: shape.numHeads,
    kvHeads: shape.numKvHeads,
    headDim: shape.headDim,
    ffnDim: shape.intermediateSize,
    vocabSize: shape.vocabSize,
    seqLen: shape.maxPositions,
    maxSeqLen: shape.maxPositions,
    activation: shape.activation,
  };

  if (shape.ropeTheta !== null) config.ropeTheta = shape.ropeTheta;
  if (shape.moe) {
    config.numExperts = shape.moe.numExperts;
    config.topK = shape.moe.topK;
  }

  return config;
}

// ── Entry point ─────────────────────────────────────────────────────────────

/**
 * Turn a HuggingFace `config.json` into a NEURAX design.
 *
 * Errors are returned rather than thrown, matching `parseArchitectureJSON`, so
 * the import dialog can show them beside a preview instead of unmounting.
 */
export function parseHuggingFaceConfig(
  jsonString: string,
  fallbackName = 'Imported Model',
): HuggingFaceImportResult {
  const empty = { nodes: [], connections: [], notes: [], assumptions: [] };

  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonString);
  } catch (err) {
    return {
      ...empty,
      modelName: fallbackName,
      error: err instanceof Error ? `Not valid JSON: ${err.message}` : 'Not valid JSON',
    };
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { ...empty, modelName: fallbackName, error: 'A config.json must be a JSON object.' };
  }

  const top = parsed as RawConfig;
  const { config, nested } = textConfig(top);

  const read = readShape(config, fallbackName);
  if ('error' in read) {
    return { ...empty, modelName: fallbackName, error: read.error };
  }

  const { shape, assumptions } = read;
  const { traits, notes } = deriveTraits(shape, config);
  const { nodes, connections } = buildGraph(shape, traits);

  // The top level names a multimodal model better than its text tower does.
  const displayName = nested
    ? str(top, '_name_or_path') ?? shape.name
    : shape.name;

  const summary: string[] = [];
  if (nested) {
    summary.push('Read the text tower from `text_config`; vision or audio towers are not imported.');
  }
  summary.push(
    `${shape.numLayers} layers, width ${shape.hiddenSize}, ${shape.numHeads} heads` +
      (shape.numKvHeads < shape.numHeads ? ` (${shape.numKvHeads} KV, grouped-query)` : '') +
      `, vocabulary ${shape.vocabSize.toLocaleString('en-US')}.`,
  );
  if (shape.moe) {
    summary.push(
      `Mixture of experts: ${shape.moe.numExperts} experts, top-${shape.moe.topK}, ` +
        `expert width ${shape.moe.expertIntermediateSize}` +
        (shape.moe.sharedExperts > 0 ? `, ${shape.moe.sharedExperts} shared` : '') +
        '.',
    );
  }
  summary.push(
    `${traits.norm === 'rmsnorm' ? 'RMSNorm' : 'LayerNorm'}, ` +
      `${traits.positional === 'pos_rope' ? 'rotary positions' : 'learned positions'}, ` +
      `${traits.ffn === 'ffn_gated' ? 'gated' : 'standard'} feed-forward.`,
  );
  if (shape.tieWordEmbeddings && shape.causal) {
    summary.push('Input and output embeddings are tied.');
  }

  return {
    nodes,
    connections,
    modelName: displayName,
    family: (shape.moe ? 'moe' : 'transformer') as ArchitectureFamily,
    hardwareConfig: buildHardwareConfig(shape),
    notes: [...summary, ...notes],
    assumptions,
  };
}

/** A short label for the detected architecture, for the dialog header. */
export function describeHuggingFaceConfig(jsonString: string): string | null {
  try {
    const parsed = JSON.parse(jsonString) as RawConfig;
    const { config } = textConfig(parsed);
    const type = str(config, 'model_type') ?? str(parsed, 'model_type');
    const arch = Array.isArray(parsed.architectures) ? parsed.architectures[0] : null;
    if (typeof arch === 'string') return arch;
    return type ?? null;
  } catch {
    return null;
  }
}
