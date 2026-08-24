import { CanvasNode, Connection } from '@/types/architecture.ts';
import { normalizeBlockParams } from '@/utils/blockDefaults.ts';
import { InitializationRecord } from '@/utils/neuraxFile.ts';

export type InitializationMethod = 
  | 'xavier_uniform'
  | 'xavier_normal'
  | 'he_uniform'
  | 'he_normal'
  | 'lsuv'
  | 'orthogonal'
  | 'sparse'
  | 'delta_orthogonal';

export interface InitializationConfig {
  method: InitializationMethod;
  gain?: number;
  sparsity?: number;
  mode?: 'fan_in' | 'fan_out' | 'fan_avg';
}

/**
 * A layer's initialisation, as facts about it rather than the sampled numbers
 * themselves.
 *
 * Used to also carry the actual generated `weights: number[][]` matrix and a
 * `bias` vector. Nothing in the application ever read them — the panel's own
 * Gradient Flow Score is computed from the closed-form `variance` below, not
 * by inspecting the samples, and the `.neurax` export deliberately stores this
 * same recipe rather than the arrays (see `buildInitializationRecord`). What
 * they cost to generate stopped being affordable once fan-in/fan-out reflect
 * a real model's width rather than a 512 fallback: a real attention layer's
 * matrix is tens of millions of floats, and `orthogonal` computed it through
 * a Gram–Schmidt pass whose cost is cubic in the matrix dimensions — the
 * combination is what made initialising a real design hang.
 */
export interface LayerWeights {
  layerId: string;
  layerName: string;
  layerType: string;
  shape: number[];
  initMethod: InitializationMethod;
  variance: number;
  fanIn: number;
  fanOut: number;
}

/**
 * What this panel can honestly say about the initialisation it just computed.
 *
 * Used to carry four more fields — epochs saved, compute hours saved, dataset
 * efficiency and a convergence-speed multiplier — each read from a fixed table
 * keyed only by which method was picked (LSUV always "1.6x", sparse always
 * "1.2x"), identical for every architecture and unrelated to the model on the
 * canvas. The first three were removed from the UI in an earlier pass; the
 * fourth, `convergenceSpeedBoost`, kept being rendered as a "Convergence
 * Boost" meter beside the real Gradient Flow Score, with nothing to tell them
 * apart. All four are gone. What remains is grounded in the weights this
 * panel actually generated.
 */
export interface SustainabilityMetrics {
  gradientFlowScore: number; // 0-100, from the real variance of the generated weights
  memoryOptimization: number; // percentage, from the sparsity actually configured
}

export interface InitializedArchitecture {
  modelName: string;
  layers: LayerWeights[];
  connections: Connection[];
  config: InitializationConfig;
  metrics: SustainabilityMetrics;
}

/**
 * The variance a layer's weights would have under the chosen method — the
 * same closed-form result the published formula for each gives, without
 * sampling the matrix that formula describes.
 *
 * This used to build the actual `rows × cols` array: draw every element,
 * and — for `orthogonal` and `delta_orthogonal` — run a Gram–Schmidt pass
 * over it whose cost is cubic in the matrix dimensions. Nothing in the
 * application ever read the array; see the note on `LayerWeights`. On the
 * flat 512-wide default every real block used to fall back to, generating
 * and orthogonalising a 512×512 matrix was slow but survivable. Once
 * fan-in/fan-out reflect a real model's width, the same matrix for one
 * attention layer is tens of millions of elements, and the cubic pass over it
 * is what made initialising a real design hang.
 */
function layerVariance(
  fanIn: number,
  fanOut: number,
  gain: number,
  sparsity: number,
  method: InitializationMethod,
): number {
  switch (method) {
    // Xavier's uniform bound is a = gain·√(6/(fanIn+fanOut)), chosen so that
    // Var(U(-a,a)) = a²/3 lands on exactly the same target variance as the
    // normal variant — that's the point of the derivation, not a difference
    // between them. Both branches report that one shared target.
    case 'xavier_uniform':
    case 'xavier_normal':
      return (2 * gain * gain) / (fanIn + fanOut);
    // Same relationship for He: a = gain·√(6/fanIn) makes the uniform
    // variant match the normal variant's 2·gain²/fanIn target.
    case 'he_uniform':
    case 'he_normal':
      return (2 * gain * gain) / fanIn;
    case 'sparse':
      return ((2 * gain * gain) / (fanIn + fanOut)) * (1 - sparsity);
    case 'delta_orthogonal':
      // Delta-orthogonal places an orthogonal matrix at the kernel's centre
      // tap and zeroes the rest (Xiao et al., "Dynamical Isometry and a Mean
      // Field Theory of CNNs") — the orthogonal block itself is what
      // propagates the signal, so the *effective* per-element variance
      // stays at the orthogonal case, not `rows`. Returning `rows` here (a
      // real model's width, e.g. 4096) drove `calculateSustainabilityMetrics`'s
      // `exp(-|avg - 1|)` score to zero for any design containing one.
      return 1.0;
    case 'orthogonal':
    case 'lsuv':
      // Both target unit variance by construction.
      return 1.0;
    default:
      return 1.0;
  }
}

/**
 * Fan-in, fan-out and weight shape for a block, from its real parameters.
 *
 * This used to switch on four literal type strings — `dense`, `conv2d`,
 * `attention`, `transformer` — that no reference template or import has
 * emitted since the block catalogue moved to specific names like
 * `gqa_attention`, `ffn_gated` and `rmsnorm`. Every real block fell through to
 * a flat 512×512 default: the Xavier/He formulas below are computed exactly
 * right, for a layer at a width nothing on the canvas actually has.
 *
 * Matched by substring against the type name instead of an exact list, so a
 * new attention or conv variant added to the catalogue resolves correctly
 * without a change here — the same class of drift is what broke the literal
 * switch the first time.
 */
function resolveLayerDims(node: CanvasNode): { fanIn: number; fanOut: number; shape: number[] } {
  const p = normalizeBlockParams(node.type, node.params ?? {}) as Record<string, unknown>;
  const num = (value: unknown, fallback: number): number =>
    typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : fallback;

  if (node.type.includes('conv')) {
    const kernel = num(p.kernel_size, 3);
    const inChannels = num(p.in_channels, 3);
    const outChannels = num(p.out_channels ?? p.filters, 64);
    return {
      fanIn: inChannels * kernel * kernel,
      fanOut: outChannels * kernel * kernel,
      shape: [outChannels, inChannels, kernel, kernel],
    };
  }

  // Norm layers hold one value per channel — no fan-in/fan-out matmul. Their
  // own schema key is `normalized_shape`, not `d_model` — layernorm/rmsnorm
  // never declare a `d_model` field for `normalizeBlockParams` to alias onto,
  // so reading `d_model` here always missed and fell back to 512.
  if (node.type.includes('norm')) {
    const width = num(p.normalized_shape, num(p.d_model, 512));
    return { fanIn: width, fanOut: width, shape: [width] };
  }

  const dModel = num(p.d_model, 512);

  // Attention blocks project to a combined Q/K/V width.
  if (node.type.includes('attention')) {
    return { fanIn: dModel, fanOut: dModel * 3, shape: [dModel, dModel * 3] };
  }

  // Everything else is a dense projection. FFN blocks widen to `d_ff`; heads
  // project onto a vocabulary or a class count; anything left maps the
  // model's width onto itself.
  const fanOut = num(p.d_ff, num(p.vocab_size, num(p.num_classes, num(p.num_labels, dModel))));
  return { fanIn: dModel, fanOut, shape: [dModel, fanOut] };
}

// Initialize weights for a single layer
function initializeLayerWeights(
  node: CanvasNode,
  config: InitializationConfig
): LayerWeights {
  const { fanIn, fanOut, shape } = resolveLayerDims(node);
  const gain = config.gain || 1.0;
  const sparsity = config.sparsity || 0.9;

  return {
    layerId: node.id,
    layerName: node.name,
    layerType: node.type,
    shape,
    initMethod: config.method,
    variance: layerVariance(fanIn, fanOut, gain, sparsity, config.method),
    fanIn,
    fanOut,
  };
}

function calculateSustainabilityMetrics(
  layers: LayerWeights[],
  config: InitializationConfig
): SustainabilityMetrics {
  // Gradient flow score, from variance preservation: unit variance across
  // layers is the actual goal Xavier/He initialisation targets, and this
  // scores how close the real, computed variance of these layers came to it.
  // An average over no layers is NaN, which reached the UI as "NaN/100".
  const avgVariance = layers.length
    ? layers.reduce((sum, l) => sum + l.variance, 0) / layers.length
    : 0;
  const gradientFlowScore = layers.length
    ? Math.min(100, Math.round(100 * Math.exp(-Math.abs(avgVariance - 1))))
    : 0;

  // Memory saved by sparse initialisation: the sparsity the user configured
  // is, by construction, the fraction of weights that are zero.
  const memoryOptimization =
    config.method === 'sparse' ? Math.round((config.sparsity || 0.9) * 100) : 0;

  return { gradientFlowScore, memoryOptimization };
}

// Main function to initialize architecture
/**
 * Canvas block types that hold trainable weights, grouped by how they are
 * initialised.
 *
 * The list used to name six generic types — dense, conv2d, attention,
 * transformer, layernorm, batchnorm — none of which any reference template
 * uses. A LLaMA or BERT template placed `token_embedding`, `gqa_attention`,
 * `ffn_gated` and `rmsnorm`, matched nothing, and the panel reported zero
 * layers, zero weights and a NaN gradient-flow score.
 */
const TRAINABLE_BLOCK_KINDS: Record<string, 'dense' | 'conv2d' | 'attention' | 'norm'> = {
  dense: 'dense',
  linear: 'dense',
  lm_head: 'dense',
  classification_head: 'dense',
  token_embedding: 'dense',
  embedding: 'dense',
  ffn_standard: 'dense',
  ffn_gated: 'dense',
  moe_layer: 'dense',
  conv2d: 'conv2d',
  conv1d: 'conv2d',
  depthwise_conv2d: 'conv2d',
  conv_transpose2d: 'conv2d',
  attention: 'attention',
  mha_attention: 'attention',
  gqa_attention: 'attention',
  mqa_attention: 'attention',
  cross_attention: 'attention',
  transformer: 'attention',
  layernorm: 'norm',
  rmsnorm: 'norm',
  batchnorm: 'norm',
  groupnorm: 'norm',
  instancenorm: 'norm',
};

/** True when this block carries weights worth initialising. */
export function isTrainableBlock(blockType: string): boolean {
  return blockType in TRAINABLE_BLOCK_KINDS;
}

export function initializeArchitecture(
  nodes: CanvasNode[],
  connections: Connection[],
  config: InitializationConfig,
  modelName: string = 'GreenAIModel'
): InitializedArchitecture {
  // Filter layers that have trainable weights
  const trainableLayers = nodes.filter((n) => isTrainableBlock(n.type));
  
  // Initialize each layer
  const layers = trainableLayers.map(node => initializeLayerWeights(node, config));
  
  // Calculate sustainability metrics
  const metrics = calculateSustainabilityMetrics(layers, config);
  
  return {
    modelName,
    layers,
    connections,
    config,
    metrics,
  };
}

/**
 * Export the initialisation this panel computed, as data — not as a Python
 * script pretending to be one.
 *
 * The previous export produced a `.py` file that claimed to build a
 * "pre-initialized model": a PyTorch constructor whose `__init__` was blocks
 * of comments for any real block type, a forward pass calling every layer
 * with the wrong number of arguments, a dummy input hardcoded to an RGB image
 * shape regardless of the architecture, and — even where a layer happened to
 * be defined — the actual computed weight values were never written into the
 * file, only their shape as a descriptive string, so PyTorch's own generic
 * initialiser silently overwrote them on load. It could not have run, for any
 * model built since the block catalogue moved past `dense`/`conv2d`/
 * `attention`/`transformer` as literal type names.
 *
 * This is what the panel can say honestly instead: the method and its
 * configuration, and the real per-layer shape, fan-in/fan-out and variance it
 * already computes correctly. That's the recipe a training script needs to
 * reproduce the same initialisation deterministically — not the megabytes of
 * random floats a fresh run regenerates identically from the same recipe
 * anyway. It becomes the `initialization` section of a `.neurax` file, so
 * opening the design back in NEURAX shows the exact setup without recomputing,
 * the same way the compiler's last analysis already does.
 */
export function buildInitializationRecord(
  architecture: InitializedArchitecture,
  hyperparams: HyperparameterConfig,
): InitializationRecord {
  return {
    method: architecture.config.method,
    gain: architecture.config.gain,
    sparsity: architecture.config.sparsity,
    hyperparameters: {
      learningRate: hyperparams.learningRate,
      dropout: hyperparams.dropout,
      weightDecay: hyperparams.weightDecay,
      warmupSteps: hyperparams.warmupSteps,
      optimizer: hyperparams.optimizer,
      gradientClipping: hyperparams.gradientClipping,
    },
    layers: architecture.layers.map((layer) => ({
      layerId: layer.layerId,
      layerName: layer.layerName,
      layerType: layer.layerType,
      shape: layer.shape,
      fanIn: layer.fanIn,
      fanOut: layer.fanOut,
      variance: layer.variance,
    })),
  };
}

// Hyperparameter recommendation types and logic
export interface HyperparameterConfig {
  learningRate: number;
  dropout: number;
  weightDecay: number;
  warmupSteps: number;
  optimizer: 'Adam' | 'AdamW' | 'SGD';
  gradientClipping: number;
  /**
   * MoE only: coefficient on the auxiliary load-balancing loss,
   * `L_aux = coefficient × Σ f_i·P_i` (f_i = each expert's share of routed
   * tokens, P_i = the router's average probability for it). 0.01 is the
   * paper's own value — Fedus, Zoph & Shazeer 2021, "Switch Transformers:
   * Scaling to Trillion Parameter Models with Simple and Efficient
   * Sparsity". Undefined for non-MoE architectures.
   */
  routerAuxLossCoefficient?: number;
  /**
   * Diffusion only: EMA decay applied to the model weights during training.
   * Standard since DDPM (Ho, Jain & Abbeel 2020) and still the value that
   * wins comparative studies (0.99 / 0.999 / 0.9999) for long runs across
   * DDPM/ADM/Stable-Diffusion-style training scripts. Undefined otherwise.
   */
  emaDecay?: number;
  /**
   * GAN only: the discriminator's own learning rate, separate from
   * `learningRate` (the generator's). Training G and D at different rates
   * is a real, peer-reviewed technique — TTUR, Heusel et al. 2017
   * (arXiv:1706.08500) — but the paper proves the *technique* converges,
   * not one fixed ratio, so this defaults equal to `learningRate` rather
   * than assuming a multiplier NEURAX has no basis for. Undefined for
   * non-GAN architectures.
   */
  discriminatorLearningRate?: number;
}

/**
 * Whether any node's type names the given family — `hasAttention`,
 * `hasConv`, `hasRnn` and so on.
 *
 * These checks used to compare `node.type` for exact equality against a
 * handful of literal strings — `'attention'`, `'transformer'`, `'lstm'`,
 * `'gru'` — none of which any reference template or import has emitted since
 * the block catalogue moved to specific names (`gqa_attention`, `lstm_cell`,
 * `bilstm`). Every real design answered "no" to all of them, so a 7-billion
 * parameter transformer got the same recommendation as an empty canvas.
 * Matched by substring instead, the same fix applied to `resolveLayerDims`.
 */
function hasBlockFamily(nodes: CanvasNode[], substring: string): boolean {
  return nodes.some((n) => n.type.includes(substring));
}

/**
 * Parameter count implied by a node's real, resolved shape.
 *
 * Mirrors the product-of-dimensions convention `ProductionWorkspace` already
 * uses for its own "Weights" stat card, so the two don't silently disagree —
 * and reuses `resolveLayerDims` rather than re-deriving widths a third way.
 */
function estimateNodeParams(node: CanvasNode): number {
  if (!isTrainableBlock(node.type)) return 0;
  const { shape } = resolveLayerDims(node);
  return shape.reduce((total, dim) => total * dim, 1);
}

export function getRecommendedHyperparams(
  nodes: CanvasNode[],
  _connections: Connection[],
): HyperparameterConfig {
  const hasAttention = hasBlockFamily(nodes, 'attention');
  const hasConv = hasBlockFamily(nodes, 'conv');
  const hasDense = nodes.some((n) => n.type === 'dense' || n.type === 'linear');
  const hasNorm = hasBlockFamily(nodes, 'norm');

  const hasMoE = hasBlockFamily(nodes, 'moe');
  const hasSSM = ['mamba', 's4', 'rwkv', 'retention'].some((s) => hasBlockFamily(nodes, s));
  const hasDiffusion = ['unet', 'vae_', 'noise_scheduler', 'diffusion'].some((s) =>
    hasBlockFamily(nodes, s),
  );
  const hasGAN = ['generator', 'discriminator', 'style', 'adain'].some((s) =>
    hasBlockFamily(nodes, s),
  );
  const hasGNN = ['gcn', 'graph_conv', 'graph_att', 'message_passing'].some((s) =>
    hasBlockFamily(nodes, s),
  );

  const totalParams = nodes.reduce((sum, n) => sum + estimateNodeParams(n), 0) || 1_000_000;

  // Learning rate: smaller for larger models. This generic, size-scaled
  // default is overridden below for families with their own literature-
  // reported rate instead of a rule of thumb.
  let learningRate = 0.001;
  if (totalParams > 100_000_000) learningRate = 0.0001;
  else if (totalParams > 10_000_000) learningRate = 0.0003;
  else if (totalParams > 1_000_000) learningRate = 0.0005;

  // Dropout: architecture-dependent
  let dropout = 0.1;
  if (hasDense && !hasConv && !hasAttention) dropout = 0.2;
  else if (hasConv && !hasAttention) dropout = 0.05;

  // Weight decay
  const weightDecay = hasNorm ? 0.01 : 0.0;

  // Warmup steps based on depth
  const warmupSteps = Math.max(100, Math.round(nodes.length * 50));

  // Optimizer
  const optimizer: HyperparameterConfig['optimizer'] = hasAttention ? 'AdamW' : 'Adam';

  // Gradient clipping
  const gradientClipping = 1.0;

  // ── Family-specific overrides ──────────────────────────────────────
  //
  // Each below replaces the generic, size-scaled numbers above with a real,
  // citable recipe for that family instead — see `HyperparameterConfig`'s
  // own field docs for the exact sources. Checked in this order because a
  // design can incidentally match more than one generic flag (a diffusion
  // UNet's ResBlocks also set `hasConv`) — the most specific family match
  // wins over a generic fallback, not the other way round.

  if (hasGNN) {
    // Kipf & Welling 2017, "Semi-Supervised Classification with Graph
    // Convolutional Networks" (ICLR) — reported directly from their Cora
    // experiments (a 2-layer, 16-hidden-unit GCN trained with Adam). GNNs
    // in practice stay shallow and narrow compared to a deep transformer,
    // so the size-scaled rate above would recommend a value an order of
    // magnitude too small.
    return {
      learningRate: 0.01,
      dropout: 0.5,
      weightDecay: 5e-4,
      warmupSteps: 0,
      optimizer: 'Adam',
      gradientClipping,
    };
  }

  if (hasGAN) {
    // DCGAN, Radford, Metz & Chintala 2015 — the paper states explicitly
    // that Adam's default β1=0.9 caused training oscillation, and settled
    // on lr=0.0002, β1=0.5. Still the de facto standard starting point
    // across GAN variants, not just DCGAN.
    //
    // `discriminatorLearningRate` starts equal to the generator's rather
    // than assuming a ratio between them: TTUR (Heusel et al. 2017,
    // arXiv:1706.08500) is real, peer-reviewed justification for training G
    // and D at different rates, but the paper proves the *technique*
    // converges, not one fixed multiplier — NEURAX has no basis to invent
    // one, so this is left for the user to actually tune apart.
    return {
      learningRate: 0.0002,
      discriminatorLearningRate: 0.0002,
      dropout: 0,
      weightDecay: 0,
      warmupSteps: 0,
      optimizer: 'Adam',
      gradientClipping,
    };
  }

  if (hasDiffusion) {
    // EMA decay 0.9999 is standard since DDPM (Ho, Jain & Abbeel 2020) and
    // still what comparative studies (0.99 / 0.999 / 0.9999) favour for
    // long runs, across DDPM/ADM/Stable-Diffusion-style training scripts.
    // Diffusion UNets typically run with little to no dropout.
    return {
      learningRate,
      emaDecay: 0.9999,
      dropout: 0,
      weightDecay,
      warmupSteps,
      optimizer: 'AdamW',
      gradientClipping,
    };
  }

  if (hasSSM) {
    // The official Mamba implementation (state-spaces/mamba) marks the
    // SSM-internal parameters (A_log, D, ...) with a `_no_weight_decay`
    // attribute — excluded from weight decay by convention in the
    // reference implementation. Underdocumented even there (see the repo's
    // issue #798) rather than a rule stated in the paper text itself, so
    // it's cited as the reference implementation's practice, not a formula
    // from the paper.
    return {
      learningRate,
      dropout,
      weightDecay: 0,
      warmupSteps,
      optimizer: 'AdamW',
      gradientClipping,
    };
  }

  if (hasMoE) {
    // Fedus, Zoph & Shazeer 2021, "Switch Transformers: Scaling to
    // Trillion Parameter Models with Simple and Efficient Sparsity" — the
    // paper's own coefficient on the auxiliary load-balancing loss,
    // L_aux = 0.01 × Σ f_i·P_i (f_i = each expert's share of routed
    // tokens, P_i = the router's average probability for it).
    return {
      learningRate,
      routerAuxLossCoefficient: 0.01,
      dropout,
      weightDecay,
      warmupSteps,
      optimizer: 'AdamW',
      gradientClipping,
    };
  }

  return { learningRate, dropout, weightDecay, warmupSteps, optimizer, gradientClipping };
}

// Get recommended initialization method based on architecture
export function getRecommendedInit(nodes: CanvasNode[]): InitializationMethod {
  const hasRelu =
    nodes.some((n) => n.type === 'relu') ||
    nodes.some((n) => n.params?.activation === 'relu');
  const hasAttention = hasBlockFamily(nodes, 'attention');
  const hasRnn = hasBlockFamily(nodes, 'lstm') || hasBlockFamily(nodes, 'gru');

  if (hasRnn) {
    return 'orthogonal';
  } else if (hasAttention) {
    return 'xavier_normal';
  } else if (hasRelu) {
    return 'he_normal';
  } else {
    return 'xavier_uniform';
  }
}

export const INITIALIZATION_METHODS: {
  id: InitializationMethod;
  name: string;
  description: string;
  bestFor: string;
  formula: string;
}[] = [
  {
    id: 'xavier_uniform',
    name: 'Xavier/Glorot Uniform',
    description: 'Uniform distribution maintaining variance across layers',
    bestFor: 'Tanh, Sigmoid activations',
    formula: 'U(-√(6/(n_in + n_out)), √(6/(n_in + n_out)))',
  },
  {
    id: 'xavier_normal',
    name: 'Xavier/Glorot Normal',
    description: 'Normal distribution optimal for symmetric activations',
    bestFor: 'Transformers, Attention layers',
    formula: 'N(0, √(2/(n_in + n_out)))',
  },
  {
    id: 'he_uniform',
    name: 'He/Kaiming Uniform',
    description: 'Designed for ReLU non-linearity, uniform variant',
    bestFor: 'ReLU, Leaky ReLU networks',
    formula: 'U(-√(6/n_in), √(6/n_in))',
  },
  {
    id: 'he_normal',
    name: 'He/Kaiming Normal',
    description: 'Optimal for deep networks with ReLU activation',
    bestFor: 'Deep CNNs, ResNets',
    formula: 'N(0, √(2/n_in))',
  },
  {
    id: 'orthogonal',
    name: 'Orthogonal',
    description: 'Preserves gradient magnitude through layers',
    bestFor: 'RNNs, LSTMs, very deep networks',
    formula: 'QR decomposition',
  },
  {
    id: 'lsuv',
    name: 'LSUV',
    description: 'Layer-Sequential Unit-Variance for stable training',
    bestFor: 'Very deep networks, BatchNorm-free architectures',
    formula: 'Orthogonal + variance normalization',
  },
  {
    id: 'sparse',
    name: 'Sparse',
    description: 'Initialize with mostly zero weights for efficiency',
    bestFor: 'Sparse architectures, pruning-friendly models',
    formula: '90% zeros, 10% Xavier',
  },
  {
    id: 'delta_orthogonal',
    name: 'Delta-Orthogonal',
    description: 'Orthogonal with gradient flow preservation',
    bestFor: 'Recurrent networks, sequence models',
    formula: 'Orthogonal × √n',
  },
];
