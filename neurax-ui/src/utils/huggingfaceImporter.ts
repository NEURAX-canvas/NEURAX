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
    /** DeepSeek-style: the model's first N layers are plain dense
     * feed-forward, not routed — `first_k_dense_replace` in the config. 0
     * for a model that routes from the first layer, like Mixtral. */
    firstKDenseReplace: number;
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
 * A `text_config` present but missing the width fields that would let it be
 * read on its own — real ones on the Hub (LLaVA-1.5, and this pattern is
 * common across vision-language releases) often only override the fields
 * that differ from a named base checkpoint, leaving `hidden_size` and
 * friends to that checkpoint's own defaults. NEURAX has no way to resolve
 * "whatever lmsys/vicuna-7b-v1.5 defaults to" from the string alone —
 * guessing a size from a model name would be exactly the kind of invented
 * number this importer refuses to produce elsewhere.
 */
interface IncompleteNestedConfig {
  key: string;
  baseModel: string | null;
}

/**
 * The sub-config that actually describes the language model.
 *
 * Multimodal releases (LLaVA, Gemma 3, Qwen-VL, Idefics) put the text tower
 * under `text_config` and leave the top level holding only wiring. Reading the
 * top level there yields a model with no layers at all.
 */
function textConfig(
  config: RawConfig,
): { config: RawConfig; nested: boolean; incomplete?: IncompleteNestedConfig } {
  let incomplete: IncompleteNestedConfig | undefined;
  for (const key of ['text_config', 'llm_config', 'language_config']) {
    const nested = config[key];
    if (!nested || typeof nested !== 'object' || Array.isArray(nested)) continue;
    const nestedConfig = nested as RawConfig;
    if (num(nestedConfig, 'hidden_size', 'n_embd', 'd_model') !== undefined) {
      return { config: nestedConfig, nested: true };
    }
    // A real nested config, just one this importer cannot resolve on its
    // own — remember it so the caller can explain why, rather than fail
    // with the same generic "no hidden size" error the top level would give
    // for an entirely unrelated reason.
    if (!incomplete) {
      incomplete = {
        key,
        baseModel: str(nestedConfig, '_name_or_path', 'name_or_path') ?? null,
      };
    }
  }
  return { config, nested: false, incomplete };
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

/**
 * `model_type` values NEURAX recognises but does not import, mapped to a
 * human family name for the error message.
 *
 * NEURAX imports three families from HuggingFace: transformer (decoder and
 * encoder, including mixture-of-experts) and Mamba/Mamba-2. Everything
 * below is a real, common family on the Hub that this importer would
 * otherwise misreport as "no hidden size in this config" — true, but not
 * the actual reason, and not actionable the way naming the real family is.
 * Not exhaustive: an unlisted, genuinely unsupported config still reaches
 * the generic field-not-found error, just without a name for what it is.
 */
const KNOWN_UNSUPPORTED_MODEL_TYPES: Record<string, string> = {
  convnextv2: 'CNN (ConvNeXt V2)',
  mobilenet_v1: 'CNN (MobileNet)',
  mobilenet_v2: 'CNN (MobileNet)',
  mobilevit: 'CNN (MobileViT)',
  vit: 'vision transformer (ViT)',
  deit: 'vision transformer (DeiT)',
  beit: 'vision transformer (BEiT)',
  swin: 'vision transformer (Swin)',
  swinv2: 'vision transformer (Swin V2)',
  detr: 'object-detection (DETR)',
  yolos: 'object-detection (YOLOS)',
  sam: 'segmentation (SAM)',
  dpt: 'depth-estimation (DPT)',
  clip: 'vision-language (CLIP)',
  blip: 'vision-language (BLIP)',
  'blip-2': 'vision-language (BLIP-2)',
  git: 'vision-language (GIT)',
  whisper: 'speech (Whisper)',
  wav2vec2: 'speech (Wav2Vec2)',
  hubert: 'speech (HuBERT)',
  speecht5: 'speech (SpeechT5)',
  vits: 'text-to-speech (VITS)',
  rwkv: 'RWKV',
  rwkv5: 'RWKV',
  rwkv6: 'RWKV',
};

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
          firstKDenseReplace: num(config, 'first_k_dense_replace') ?? 0,
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

// ── Mamba / Mamba-2 (state-space models) ────────────────────────────────────
//
// A different family entirely — no attention, no heads, no `num_attention_
// heads` field to even look for — so this is a parallel path, not a branch
// of the transformer one above. Both `state-spaces/mamba-*-hf` (`model_type:
// "mamba"`) and `state-spaces/mamba2-*-hf`-style checkpoints (`model_type:
// "mamba2"`) were fetched live from the Hub and read field-by-field against
// this importer while writing it, the same way the transformer path's
// aliases were derived from real configs rather than the spec.

const MAMBA_MODEL_TYPES = new Set(['mamba', 'mamba2']);

// ── CNN import ───────────────────────────────────────────────────────────
//
// A fourth family, alongside transformer/MoE and Mamba/Mamba-2. ResNet,
// RegNet, ConvNeXt and EfficientNet all ship a `config.json` on the Hub —
// fetched live (`microsoft/resnet-50`, `facebook/regnet-y-040`,
// `facebook/convnext-tiny-224`, `google/efficientnet-b0`) and read
// field-by-field while writing this, the same discipline as the transformer
// and Mamba paths. The graph this emits is deliberately the exact node
// shape `data/modelTemplates.ts`'s own ResNet-50/EfficientNet-B0/ConvNeXt-
// Tiny reference templates already use (`bottleneck_block` with
// `{planes, blocks, stride, expansion}`, `mbconv_block` with per-stage
// `{in_channels, out_channels, kernel_size, expand_ratio, se_ratio,
// num_blocks}`, `convnext_block` with `{dim, num_blocks, kernel_size,
// downsample}`) — the shapes the compiler's accuracy tests already measure,
// not a new, unmeasured encoding.
//
// Deliberately not attempted: MobileNet, MobileViT, ConvNeXt V2 — no live
// config was read for these while writing this importer, so they stay in
// `KNOWN_UNSUPPORTED_MODEL_TYPES` rather than guess at a schema.

const CNN_MODEL_TYPES = new Set(['resnet', 'regnet', 'convnext', 'efficientnet']);

interface CnnStage {
  planes: number;
  blocks: number;
  stride: number;
  // EfficientNet (MBConv) only — one stage per entry in its per-stage arrays.
  inChannels?: number;
  kernelSize?: number;
  expandRatio?: number;
  seRatio?: number;
}

interface CnnShape {
  modelType: 'resnet' | 'regnet' | 'convnext' | 'efficientnet';
  name: string;
  numChannels: number;
  stemOutChannels: number;
  stages: CnnStage[];
  numLabels: number;
  activation: string;
}

function readCnnShape(
  config: RawConfig,
  fallbackName: string,
): { shape: CnnShape; assumptions: string[] } | { error: string } {
  const assumptions: string[] = [];
  const modelType = (str(config, 'model_type') ?? '') as CnnShape['modelType'];
  const numChannels = num(config, 'num_channels') ?? 3;
  const activation = str(config, 'hidden_act') ?? 'relu';
  const numLabels = (() => {
    const id2label = config.id2label;
    if (id2label && typeof id2label === 'object') {
      return Object.keys(id2label as Record<string, unknown>).length;
    }
    assumptions.push('Class count absent; assumed 1000 (ImageNet-1k, the near-universal default).');
    return 1000;
  })();

  const name = str(config, '_name_or_path', 'name_or_path') ?? fallbackName;

  if (modelType === 'resnet' || modelType === 'regnet') {
    const depths = config.depths;
    const hiddenSizes = config.hidden_sizes;
    if (!Array.isArray(depths) || !Array.isArray(hiddenSizes) || depths.length === 0) {
      return { error: `No stage depths/hidden_sizes in this ${modelType} config.` };
    }
    const stemOutChannels = num(config, 'embedding_size') ?? 64;
    const stages: CnnStage[] = depths.map((blocks, i) => ({
      planes: Number(hiddenSizes[i]),
      blocks: Number(blocks),
      stride: i === 0 ? 1 : 2,
    }));
    return { shape: { modelType, name, numChannels, stemOutChannels, stages, numLabels, activation }, assumptions };
  }

  if (modelType === 'convnext') {
    const depths = config.depths;
    const hiddenSizes = config.hidden_sizes;
    if (!Array.isArray(depths) || !Array.isArray(hiddenSizes) || depths.length === 0) {
      return { error: 'No stage depths/hidden_sizes in this ConvNeXt config.' };
    }
    const patchSize = num(config, 'patch_size') ?? 4;
    const stages: CnnStage[] = depths.map((blocks, i) => ({
      planes: Number(hiddenSizes[i]),
      blocks: Number(blocks),
      stride: i === 0 ? 1 : 2,
      kernelSize: 7,
    }));
    return {
      shape: { modelType, name, numChannels, stemOutChannels: patchSize, stages, numLabels, activation: str(config, 'hidden_act') ?? 'gelu' },
      assumptions,
    };
  }

  // EfficientNet — every stage is its own entry across seven parallel arrays,
  // not a `depths`/`hidden_sizes` pair the way the others are.
  const inChannels = config.in_channels;
  const outChannels = config.out_channels;
  const kernelSizes = config.kernel_sizes;
  const expandRatios = config.expand_ratios;
  const numBlockRepeats = config.num_block_repeats;
  const strides = config.strides;
  if (
    !Array.isArray(inChannels) || !Array.isArray(outChannels) ||
    !Array.isArray(kernelSizes) || !Array.isArray(expandRatios) ||
    !Array.isArray(numBlockRepeats) || !Array.isArray(strides) ||
    inChannels.length === 0
  ) {
    return { error: 'EfficientNet config is missing one of its per-stage arrays (in_channels, out_channels, kernel_sizes, expand_ratios, num_block_repeats, strides).' };
  }
  const seRatio = num(config, 'squeeze_expansion_ratio') ?? 0.25;
  const stemOutChannels = Number(inChannels[0]);
  const stages: CnnStage[] = inChannels.map((_, i) => ({
    planes: Number(outChannels[i]),
    blocks: Number(numBlockRepeats[i]),
    stride: Number(strides[i]),
    inChannels: Number(inChannels[i]),
    kernelSize: Number(kernelSizes[i]),
    expandRatio: Number(expandRatios[i]),
    seRatio,
  }));
  return {
    shape: { modelType, name, numChannels, stemOutChannels, stages, numLabels, activation: str(config, 'hidden_act') ?? 'swish' },
    assumptions,
  };
}

function buildCnnGraph(shape: CnnShape): { nodes: CanvasNode[]; connections: Connection[] } {
  const nodes: CanvasNode[] = [];
  const connections: Connection[] = [];
  let nodeSeq = 0;
  let connSeq = 0;
  const add = (type: LayerType, name: string, x: number, params: Record<string, unknown>): string => {
    const id = `hf-n${++nodeSeq}`;
    nodes.push({ id, type, name, x, y: 140, params: params as CanvasNode['params'] });
    return id;
  };
  const link = (from: string, to: string) => connections.push({ id: `hf-c${++connSeq}`, from, to });

  let x = 50;
  const input = add('input', 'Input Image', x, {});
  x += 200;

  let prev = input;
  if (shape.modelType === 'convnext') {
    const stem = add('stem_block', `Stem: ${shape.stemOutChannels}, patch ${shape.stages[0]?.kernelSize ?? 4}`, x, {
      out_channels: shape.stemOutChannels,
      kernel_size: 4,
      stride: 4,
    });
    link(prev, stem);
    prev = stem;
    x += 200;
  } else {
    const stem = add('stem_block', `Stem: 7×7 Conv (${shape.stemOutChannels})`, x, {
      out_channels: shape.stemOutChannels,
      kernel_size: 7,
      stride: 2,
    });
    link(prev, stem);
    x += 200;
    const pool = add('max_pool', '3×3 Max Pool (stride 2)', x, { kernel_size: 3, stride: 2, padding: 1 });
    link(stem, pool);
    prev = pool;
    x += 200;
  }

  for (const [i, stage] of shape.stages.entries()) {
    let stageNode: string;
    if (shape.modelType === 'resnet' || shape.modelType === 'regnet') {
      stageNode = add(
        'bottleneck_block',
        `Stage ${i + 1}: ${stage.blocks}× Bottleneck (${stage.planes})`,
        x,
        { planes: stage.planes, blocks: stage.blocks, stride: stage.stride, expansion: 4 },
      );
    } else if (shape.modelType === 'convnext') {
      stageNode = add(
        'convnext_block',
        `Stage ${i + 1}: ${stage.blocks}× ConvNeXt Block (${stage.planes})`,
        x,
        { dim: stage.planes, num_blocks: stage.blocks, kernel_size: 7, downsample: i > 0 },
      );
    } else {
      stageNode = add(
        'mbconv_block',
        `MBConv${stage.expandRatio} (${stage.planes}), k${stage.kernelSize}×${stage.kernelSize} ×${stage.blocks}`,
        x,
        {
          in_channels: stage.inChannels, out_channels: stage.planes, kernel_size: stage.kernelSize,
          expand_ratio: stage.expandRatio, se_ratio: stage.seRatio, stride: stage.stride, num_blocks: stage.blocks,
        },
      );
    }
    link(prev, stageNode);
    prev = stageNode;
    x += 200;
  }

  const pool = add('global_pool', 'Global Average Pool', x, {});
  link(prev, pool);
  x += 200;

  const head = add('classification_head', `Classifier (${shape.numLabels})`, x, { num_labels: shape.numLabels });
  link(pool, head);
  x += 200;

  const output = add('output', 'Output', x, {});
  link(head, output);

  return { nodes, connections };
}

function buildCnnHardwareConfig(shape: CnnShape): Partial<HardwareConfig> {
  return {
    inChannels: shape.numChannels,
    numClasses: shape.numLabels,
    convActivation: shape.activation,
  };
}

function parseCnnConfig(config: RawConfig, fallbackName: string): HuggingFaceImportResult {
  const empty = { nodes: [], connections: [], notes: [], assumptions: [] };
  const read = readCnnShape(config, fallbackName);
  if ('error' in read) {
    return { ...empty, modelName: fallbackName, error: read.error };
  }
  const { shape, assumptions } = read;
  const { nodes, connections } = buildCnnGraph(shape);

  const totalBlocks = shape.stages.reduce((sum, s) => sum + s.blocks, 0);
  const notes: string[] = [
    `${shape.modelType.toUpperCase()}: ${shape.stages.length} stages, ${totalBlocks} blocks total, ` +
      `${shape.numChannels} input channels, ${shape.numLabels.toLocaleString('en-US')} classes.`,
  ];

  return {
    nodes,
    connections,
    modelName: shape.name,
    family: 'cnn' as ArchitectureFamily,
    hardwareConfig: buildCnnHardwareConfig(shape),
    notes,
    assumptions,
  };
}

// ── Diffusion (UNet) import ─────────────────────────────────────────────
//
// A diffusion pipeline is several separate models (UNet, VAE, text
// encoder), each its own config file under a subfolder — `model_index.json`
// names them, `unet/config.json` holds the one that actually does the
// denoising and carries the overwhelming majority of the pipeline's
// parameters. This reads that file — real, live-verified against
// `runwayml/stable-diffusion-v1-5`'s `unet/config.json` (`block_out_channels:
// [320,640,1280,1280]`, `layers_per_block: 2`, `cross_attention_dim: 768`) —
// on its own, the same way a language model's `config.json` is read on its
// own without also fetching its tokenizer. VAE and text-encoder parameter
// counts aren't included: not a rounding error (the UNet dominates a real
// pipeline's training-time cost, which is what NEURAX exists to estimate)
// but a real, stated gap, not silently absorbed into the UNet's number.

function readDiffusionUnetShape(
  config: RawConfig,
  fallbackName: string,
): { shape: DiffusionUnetShape; assumptions: string[] } | { error: string } {
  const assumptions: string[] = [];
  const blockOutChannels = config.block_out_channels;
  if (!Array.isArray(blockOutChannels) || blockOutChannels.length === 0) {
    return { error: 'No block_out_channels in this UNet config.' };
  }
  const inChannels = num(config, 'in_channels') ?? 4;
  const outChannels = num(config, 'out_channels') ?? inChannels;
  const layersPerBlock = num(config, 'layers_per_block') ?? 2;
  const crossAttentionDim = num(config, 'cross_attention_dim');
  const downBlockTypes = Array.isArray(config.down_block_types)
    ? (config.down_block_types as unknown[]).map(String)
    : [];
  const hasCrossAttention = downBlockTypes.some((t) => t.includes('CrossAttn')) || crossAttentionDim !== undefined;

  if (crossAttentionDim === undefined && hasCrossAttention) {
    assumptions.push('down_block_types names a cross-attention block but cross_attention_dim is absent; cross-attention width omitted from the block params.');
  }

  const name = str(config, '_name_or_path', 'name_or_path') ?? fallbackName;
  return {
    shape: {
      name,
      channels: blockOutChannels.map(Number),
      inChannels,
      outChannels,
      layersPerBlock,
      crossAttentionDim: crossAttentionDim ?? null,
      hasCrossAttention,
    },
    assumptions,
  };
}

interface DiffusionUnetShape {
  name: string;
  channels: number[];
  inChannels: number;
  outChannels: number;
  layersPerBlock: number;
  crossAttentionDim: number | null;
  hasCrossAttention: boolean;
}

function buildDiffusionUnetGraph(shape: DiffusionUnetShape): { nodes: CanvasNode[]; connections: Connection[] } {
  const nodes: CanvasNode[] = [];
  const connections: Connection[] = [];
  let nodeSeq = 0;
  let connSeq = 0;
  const add = (type: LayerType, name: string, x: number, params: Record<string, unknown>): string => {
    const id = `hf-n${++nodeSeq}`;
    nodes.push({ id, type, name, x, y: 140, params: params as CanvasNode['params'] });
    return id;
  };
  const link = (from: string, to: string) => connections.push({ id: `hf-c${++connSeq}`, from, to });

  const input = add('input', 'Noised Latent', 50, {});
  const crossAttnParams = shape.hasCrossAttention && shape.crossAttentionDim
    ? { cross_attention_dim: shape.crossAttentionDim }
    : {};

  const encoder = add('unet_encoder', `UNet Encoder (${shape.channels.length}× down)`, 250, {
    in_channels: shape.inChannels,
    channels: shape.channels,
    num_res_blocks: shape.layersPerBlock,
    ...crossAttnParams,
  });
  link(input, encoder);

  const mid = add('unet_mid', 'UNet Middle', 450, {
    channels: shape.channels[shape.channels.length - 1],
    with_attention: true,
    ...crossAttnParams,
  });
  link(encoder, mid);

  const decoder = add('unet_decoder', `UNet Decoder (${shape.channels.length}× up)`, 650, {
    out_channels: shape.outChannels,
    channels: [...shape.channels].reverse(),
    num_res_blocks: shape.layersPerBlock + 1,
    ...crossAttnParams,
  });
  link(mid, decoder);

  const output = add('output', 'Predicted Noise', 850, {});
  link(decoder, output);

  return { nodes, connections };
}

function buildDiffusionUnetHardwareConfig(shape: DiffusionUnetShape): Partial<HardwareConfig> {
  return {
    inChannels: shape.inChannels,
    outChannels: shape.outChannels,
    modelChannels: shape.channels[0],
    numResBlocks: shape.layersPerBlock,
    channelMult: shape.channels.map((c) => c / shape.channels[0]).join(','),
    ...(shape.crossAttentionDim ? { crossAttentionDim: shape.crossAttentionDim } : {}),
  };
}

function parseDiffusionUnetConfig(config: RawConfig, fallbackName: string): HuggingFaceImportResult {
  const empty = { nodes: [], connections: [], notes: [], assumptions: [] };
  const read = readDiffusionUnetShape(config, fallbackName);
  if ('error' in read) {
    return { ...empty, modelName: fallbackName, error: read.error };
  }
  const { shape, assumptions } = read;
  const { nodes, connections } = buildDiffusionUnetGraph(shape);

  const notes: string[] = [
    `UNet: ${shape.channels.length} resolution levels (${shape.channels.join(' → ')} channels), ` +
      `${shape.layersPerBlock} residual blocks per level` +
      (shape.hasCrossAttention && shape.crossAttentionDim ? `, cross-attention width ${shape.crossAttentionDim}` : '') +
      '.',
    'Only the UNet (the denoising network) is imported — the VAE and text encoder are separate models in ' +
      'the pipeline, not read here, so this parameter count is the UNet\'s alone, not the full pipeline\'s.',
  ];

  return {
    nodes,
    connections,
    modelName: shape.name,
    family: 'diffusion' as ArchitectureFamily,
    hardwareConfig: buildDiffusionUnetHardwareConfig(shape),
    notes,
    assumptions,
  };
}

interface MambaShape {
  modelType: 'mamba' | 'mamba2';
  name: string;
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  stateDim: number;
  expansionFactor: number;
  convKernel: number;
  normEps: number;
  tieWordEmbeddings: boolean;
  /** Mamba-2 only — absent for Mamba-1, where the block has no head structure. */
  numHeads: number | null;
  headDim: number | null;
}

function readMambaShape(
  config: RawConfig,
  fallbackName: string,
): { shape: MambaShape; assumptions: string[] } | { error: string } {
  const assumptions: string[] = [];
  const modelType = (str(config, 'model_type') ?? '').toLowerCase() as 'mamba' | 'mamba2';

  const hiddenSize = num(config, 'hidden_size', 'd_model');
  if (hiddenSize === undefined || hiddenSize <= 0) {
    return {
      error: 'No hidden size in this Mamba config (looked for hidden_size, d_model).',
    };
  }

  const numLayers = num(config, 'num_hidden_layers', 'n_layer');
  if (numLayers === undefined || numLayers <= 0) {
    return {
      error: 'No layer count in this Mamba config (looked for num_hidden_layers, n_layer).',
    };
  }

  let vocabSize = num(config, 'vocab_size', 'padded_vocab_size');
  if (vocabSize === undefined || vocabSize <= 0) {
    vocabSize = 50280;
    assumptions.push('Vocabulary size absent; assumed 50280 (state-spaces/mamba’s own default).');
  }

  let stateDim = num(config, 'state_size', 'd_state');
  if (stateDim === undefined || stateDim <= 0) {
    stateDim = 16;
    assumptions.push('SSM state dimension absent; assumed 16, the published default for every Mamba/Mamba-2 size.');
  }

  let expansionFactor = num(config, 'expand');
  if (expansionFactor === undefined || expansionFactor <= 0) {
    expansionFactor = 2;
    assumptions.push('Expansion factor absent; assumed 2, the published default.');
  }

  const convKernel = num(config, 'conv_kernel', 'd_conv') ?? 4;
  const normEps = num(config, 'layer_norm_epsilon', 'norm_eps') ?? 1e-5;
  const tieWordEmbeddings = bool(config, 'tie_word_embeddings', 'tie_embeddings') ?? true;

  // Mamba-2 only: the SSD algorithm organises the state into heads, the way
  // attention organises width into heads. Reading these is descriptive here
  // (the notes explain what makes this a Mamba-2, not a Mamba-1, block) —
  // `neurax_formulas::ssm::mamba_params` doesn't yet have a head-aware
  // variant of the formula, so the parameter count uses the same formula as
  // Mamba-1's. Real Mamba-2 checkpoints are close to it (the head structure
  // reshapes the same total state, it doesn't add a separate parameter
  // tensor per head) but this is a real, stated approximation, not a
  // verified-exact one the way the transformer path's numbers are.
  const numHeads = num(config, 'num_heads') ?? null;
  const headDim = num(config, 'head_dim') ?? null;

  const name = str(config, '_name_or_path', 'name_or_path') ?? fallbackName;

  return {
    shape: {
      modelType,
      name,
      hiddenSize,
      numLayers,
      vocabSize,
      stateDim,
      expansionFactor,
      convKernel,
      normEps,
      tieWordEmbeddings,
      numHeads,
      headDim,
    },
    assumptions,
  };
}

/**
 * Lay out a Mamba design as one real block, not its internal sub-steps.
 *
 * The reference templates used to decompose a Mamba block into its visual
 * sub-components (input projection, causal conv, the selective-scan step,
 * output projection) on the canvas. It read well and computed wrong: two of
 * those sub-nodes each independently carried the *whole* block's parameter
 * formula (double-counting it), and the step that is conceptually the core
 * of the block — the selective scan itself — carried none of its own,
 * because nothing in the compiler's layer-type resolution recognised it and
 * it fell back to a zero-parameter default. A model imported this way
 * reported roughly double the block's real parameter count. One node per
 * layer, carrying the one real formula (`neurax_formulas::ssm::mamba_params`,
 * verified against Mamba-2.8B's published size — see
 * `neurax-core/tests/published_model_accuracy.rs`), is correct where the
 * six-node version was not, even though it shows less of the block's
 * internal structure on the canvas.
 */
function buildMambaGraph(shape: MambaShape): { nodes: CanvasNode[]; connections: Connection[] } {
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
  const BODY = 60;

  const input = add('input', 'Input Tokens', 50, TRUNK, {});
  const embedding = add('token_embedding', 'Token Embedding', 250, TRUNK, {
    vocab_size: shape.vocabSize,
    hidden_size: width,
  });
  link(input, embedding);

  const stack = add(
    'layer_stack',
    `${shape.numLayers}× ${shape.modelType === 'mamba2' ? 'Mamba-2' : 'Mamba'} Block`,
    650,
    TRUNK,
    { num_layers: shape.numLayers },
  );
  link(embedding, stack);

  const preNorm = add('rmsnorm', 'Pre-Block RMSNorm', 650, BODY, {
    hidden_size: width,
    eps: shape.normEps,
  });
  const block = add(
    'mamba_block',
    shape.modelType === 'mamba2'
      ? `Mamba-2 (SSD, state ${shape.stateDim})`
      : `Mamba (selective SSM, state ${shape.stateDim})`,
    850,
    BODY,
    {
      hidden_size: width,
      state_dim: shape.stateDim,
      expansion_factor: shape.expansionFactor,
      conv_kernel_size: shape.convKernel,
    },
  );
  const residual = add('residual_add', 'Residual Add', 1050, BODY, {});
  link(preNorm, block);
  link(block, residual);
  link(residual, stack);

  const finalNorm = add('rmsnorm', 'Final RMSNorm', 1650, TRUNK, {
    hidden_size: width,
    eps: shape.normEps,
  });
  link(stack, finalNorm);

  const head = add('lm_head', shape.tieWordEmbeddings ? 'LM Head (tied)' : 'LM Head', 1850, TRUNK, {
    vocab_size: shape.vocabSize,
    hidden_size: width,
    tie_weights: shape.tieWordEmbeddings,
    ...(shape.tieWordEmbeddings ? {} : { in_features: width, out_features: shape.vocabSize }),
  });
  link(finalNorm, head);

  const output = add('output', 'Output', 2050, TRUNK, {});
  link(head, output);

  return { nodes, connections };
}

function buildMambaHardwareConfig(shape: MambaShape): Partial<HardwareConfig> {
  return {
    hiddenDim: shape.hiddenSize,
    numLayers: shape.numLayers,
    vocabSize: shape.vocabSize,
    dState: shape.stateDim,
  };
}

function parseMambaConfig(config: RawConfig, fallbackName: string): HuggingFaceImportResult {
  const empty = { nodes: [], connections: [], notes: [], assumptions: [] };
  const read = readMambaShape(config, fallbackName);
  if ('error' in read) {
    return { ...empty, modelName: fallbackName, error: read.error };
  }

  const { shape, assumptions } = read;
  const { nodes, connections } = buildMambaGraph(shape);

  const notes: string[] = [
    `${shape.numLayers} layers, width ${shape.hiddenSize}, SSM state ${shape.stateDim}, ` +
      `expansion ×${shape.expansionFactor}, vocabulary ${shape.vocabSize.toLocaleString('en-US')}.`,
  ];
  if (shape.modelType === 'mamba2') {
    notes.push(
      `Mamba-2 (SSD)${shape.numHeads ? `: ${shape.numHeads} heads` : ''}${shape.headDim ? ` × ${shape.headDim}` : ''}. ` +
        'Parameter count uses the same formula as Mamba-1 — an approximation for the head-structured SSD algorithm, not a verified-exact one.',
    );
  }
  if (shape.tieWordEmbeddings) {
    notes.push('Input and output embeddings are tied.');
  }

  return {
    nodes,
    connections,
    modelName: shape.name,
    family: 'ssm' as ArchitectureFamily,
    hardwareConfig: buildMambaHardwareConfig(shape),
    notes,
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

  // Usually one tail feeding the residual join — two only for a
  // DeepSeek-style model, where the routed path and the dense-replacement
  // path are two different representative blocks, never both active on the
  // same layer, that still both need to reach the same join point.
  const ffnTails: string[] = [];
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
    ffnTails.push(combine);

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

    if (shape.moe.firstKDenseReplace > 0) {
      // DeepSeek-style: the first `firstKDenseReplace` layers use a plain
      // feed-forward, not routing at all — no router, no experts, no
      // combine step for those layers. A separate representative node for
      // this real (not routed) FFN, at the model's real dense width, lets
      // the compiler scale it against exactly that many layers instead of
      // silently treating every layer as routed from the first one.
      const denseFfn = add(
        traits.ffn,
        `Dense FFN (first ${shape.moe.firstKDenseReplace} layer${shape.moe.firstKDenseReplace === 1 ? '' : 's'})`,
        1050,
        FFN - 80,
        {
          hidden_size: width,
          intermediate_size: shape.intermediateSize,
          activation: shape.activation,
        },
      );
      link(preFfnNorm, denseFfn);
      ffnTails.push(denseFfn);
    }
  } else {
    const ffnTail = add(
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
    ffnTails.push(ffnTail);
  }

  const ffnResidual = add('residual_add', 'Residual Add', 1450, FFN, {});
  for (const tail of ffnTails) {
    link(tail, ffnResidual);
  }
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
    if (shape.moe.firstKDenseReplace > 0) {
      config.numDenseLayers = shape.moe.firstKDenseReplace;
    }
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

  // Diffusers pipeline configs (UNet2DConditionModel, AutoencoderKL, ...)
  // don't carry `model_type` at all — `_class_name` is their signature
  // instead, and reading past it into the generic "no hidden size" error
  // below would blame a field name mismatch for a family this importer
  // was never going to read.
  const diffusersClass = str(top, '_class_name');
  if (diffusersClass === 'UNet2DConditionModel' || diffusersClass === 'UNet2DModel') {
    return parseDiffusionUnetConfig(top, fallbackName);
  }
  if (diffusersClass) {
    return {
      ...empty,
      modelName: fallbackName,
      error:
        `"${diffusersClass}" is a diffusers pipeline component NEURAX doesn't read — only the UNet ` +
        '(UNet2DConditionModel/UNet2DModel) is imported today, not the VAE, text encoder, or scheduler. ' +
        'Fetch the pipeline\'s unet/config.json specifically, or build the rest on the canvas from the Diffusion family.',
    };
  }

  const topModelType = (str(top, 'model_type') ?? '').toLowerCase();
  if (MAMBA_MODEL_TYPES.has(topModelType)) {
    return parseMambaConfig(top, fallbackName);
  }
  if (CNN_MODEL_TYPES.has(topModelType)) {
    return parseCnnConfig(top, fallbackName);
  }

  const { config, nested, incomplete } = textConfig(top);

  const nestedModelType = (str(config, 'model_type') ?? '').toLowerCase();
  if (MAMBA_MODEL_TYPES.has(nestedModelType)) {
    return parseMambaConfig(config, fallbackName);
  }
  if (CNN_MODEL_TYPES.has(nestedModelType)) {
    return parseCnnConfig(config, fallbackName);
  }

  const unsupportedFamily = KNOWN_UNSUPPORTED_MODEL_TYPES[nestedModelType || topModelType];
  if (unsupportedFamily) {
    return {
      ...empty,
      modelName: fallbackName,
      error:
        `This is a ${unsupportedFamily} model ("model_type": "${nestedModelType || topModelType}"). ` +
        'NEURAX imports transformer and MoE language models, Mamba/Mamba-2, and ResNet/RegNet/ConvNeXt/EfficientNet CNNs from HuggingFace — ' +
        `${unsupportedFamily} architectures aren't read from the Hub yet.`,
    };
  }

  const read = readShape(config, fallbackName);
  if ('error' in read) {
    if (incomplete) {
      const base = incomplete.baseModel;
      return {
        ...empty,
        modelName: fallbackName,
        error:
          `This is a multimodal model whose "${incomplete.key}" only overrides some fields` +
          (base ? ` — the rest come from ${base}'s own defaults, which NEURAX has no way to look up.` : ', relying on defaults from a base checkpoint NEURAX has no way to look up.') +
          ` Import that base model's own config.json directly (${base ? `search the Hub for "${base}"` : 'find it under "architectures" or "_name_or_path" in this file'}), or fill in hidden_size, num_hidden_layers, num_attention_heads and intermediate_size under "${incomplete.key}" yourself.`,
      };
    }
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
