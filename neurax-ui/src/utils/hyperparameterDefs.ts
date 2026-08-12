/**
 * hyperparameterDefs.ts
 *
 * Définit la liste exhaustive des hyperparamètres optimisables
 * pour chaque famille d'architecture NEURAX.
 *
 * Chaque définition précise :
 *  - le type (int, float, categorical, bool)
 *  - les bornes min/max et le pas
 *  - si l'échelle est logarithmique (LR, etc.)
 *  - les dépendances/contraintes entre paramètres
 *  - la priorité dans l'optimisation
 */

import { ArchitectureFamily } from '@/types/plugins';
import { HardwareConfig, MANDATORY_FIELDS } from '@/contexts/HardwareContext';

// ─── Types ───────────────────────────────────────────────────────────

export type ParamType = 'int' | 'float' | 'categorical' | 'bool';

/**
 * Key a hyperparameter writes to.
 *
 * Built-ins name a `HardwareConfig` field; the plain-string case covers
 * user-defined parameters, which ride in the model's `global_params`.
 */
export type HyperparamKey = keyof HardwareConfig | (string & {});

export interface SearchRange {
  min: number;
  max: number;
  /** Step for grid/coarse search (default 1 for int, dynamic for float) */
  step?: number;
  /** Use log scale (e.g. learning rate) */
  logScale?: boolean;
  /** Number of grid points (overrides step) */
  gridPoints?: number;
}

export interface CategoricalOption {
  value: string | boolean | number;
  label: string;
}

export interface ParamConstraint {
  /** Description of the constraint for UI display */
  description: string;
  /** Validation function */
  validate: (config: Record<string, unknown>) => boolean;
  /** Auto-fix function that returns corrected config */
  fix?: (config: Record<string, unknown>) => Record<string, unknown>;
}

export interface HyperparameterDef {
  /**
   * Config key this parameter writes to.
   *
   * Built-ins name a `HardwareConfig` field; the plain-string case covers
   * user-defined parameters, which are forwarded into the model's
   * `global_params` so a config can carry as many as an architecture needs.
   */
  key: HyperparamKey;
  /** Display label */
  label: string;
  /** Short description / tooltip */
  description: string;
  /** Value type */
  type: ParamType;
  /** Default value */
  defaultValue: number | string | boolean;

  // Numeric ranges
  range?: SearchRange;

  // Categorical options
  options?: CategoricalOption[];

  // Dependencies
  /** Only show when parentKey has parentValue */
  parentKey?: keyof HardwareConfig;
  parentValue?: unknown;

  // Constraints
  constraints?: ParamConstraint[];

  /** Derived from other fields (auto-computed, not directly optimizable) */
  isDerived?: boolean;
  /** Derivation formula as string (shown in UI) */
  derivedFormula?: string;

  /**
   * Whether the family can be compiled without this parameter.
   *
   * Required parameters have no defensible default for the family — a
   * Transformer without a hidden size is not a model — and the UI blocks
   * analysis until they are set. Optional ones fall back to `defaultValue`.
   */
  required?: boolean;

  /** True for parameters the user added rather than the built-in catalogue. */
  isCustom?: boolean;

  // Optimization priority
  priority: 'critical' | 'high' | 'medium' | 'low';

  /** Group/category for UI organisation */
  group: 'capacity' | 'compute' | 'memory' | 'training' | 'regularization' | 'data' | 'hardware' | 'architecture';
}

export interface FamilyHyperparameterDefs {
  family: ArchitectureFamily;
  params: HyperparameterDef[];
  /** Cross-parameter constraints */
  globalConstraints?: ParamConstraint[];
  /** Recommended default search space (overrides individual ranges) */
  defaultSearchSpace?: Partial<Record<HyperparamKey, SearchRange>>;
}

// ─── Shared params (used inline in family defs) ───────────────────

// These are defined directly per-family to allow customization

// ─── Per-family param constraints ───────────────────────────────────

const TRANSFORMER_CONSTRAINTS: ParamConstraint[] = [
  {
    description: 'num_heads must divide hiddenDim (d_model % num_heads === 0)',
    validate: (c) => {
      const h = c.hiddenDim as number;
      const n = c.numHeads as number;
      return !h || !n || h % n === 0;
    },
    fix: (c) => {
      const h = c.hiddenDim as number;
      const n = c.numHeads as number;
      if (h && n && h % n !== 0) {
        return { ...c, numHeads: Math.max(1, Math.round(h / 64)) };
      }
      return c;
    },
  },
  {
    description: 'kvHeads must divide numHeads (GQA constraint)',
    validate: (c) => {
      const n = c.numHeads as number;
      const kv = c.kvHeads as number;
      return !n || !kv || n % kv === 0;
    },
    fix: (c) => {
      const n = c.numHeads as number;
      const kv = c.kvHeads as number;
      if (n && kv && n % kv !== 0) {
        return { ...c, kvHeads: Math.max(1, Math.round(n / 4)) };
      }
      return c;
    },
  },
  {
    description: 'headDim * numHeads should be close to hiddenDim',
    validate: (c) => {
      const hd = c.hiddenDim as number;
      const nh = c.numHeads as number;
      const hdim = c.headDim as number;
      if (hd && nh && hdim) {
        const ratio = (nh * hdim) / hd;
        return ratio > 0.8 && ratio < 1.25;
      }
      return true;
    },
  },
  {
    description: 'ffnDim typically = 4 * hiddenDim',
    validate: () => true, // Not a hard constraint, recommended
  },
];

const MOE_CONSTRAINTS: ParamConstraint[] = [
  {
    description: 'topK must be <= numExperts',
    validate: (c) => {
      const e = c.numExperts as number;
      const k = c.topK as number;
      return !e || !k || k <= e;
    },
    fix: (c) => {
      const e = c.numExperts as number;
      const k = c.topK as number;
      if (e && k && k > e) return { ...c, topK: Math.max(1, Math.round(e / 4)) };
      return c;
    },
  },
  ...TRANSFORMER_CONSTRAINTS,
];

const CNN_CONSTRAINTS: ParamConstraint[] = [
  {
    description: 'imgHeight and imgWidth should be powers of 2 (for downsampling stages)',
    validate: (c) => {
      const h = c.imgHeight as number;
      const w = c.imgWidth as number;
      const isPowerOf2 = (n: number) => n > 0 && (n & (n - 1)) === 0;
      return !h || !w || (isPowerOf2(h) && isPowerOf2(w));
    },
  },
];

const SSM_CONSTRAINTS: ParamConstraint[] = [
  {
    description: 'dState should be a power of 2 (for efficient scan)',
    validate: (c) => {
      const d = c.dState as number;
      return !d || (d > 0 && (d & (d - 1)) === 0);
    },
  },
  {
    description: 'expandFactor * hiddenDim should fit in memory',
    validate: () => true,
  },
];

const DIFFUSION_CONSTRAINTS: ParamConstraint[] = [
  {
    description: 'imgHeight and imgWidth should be divisible by 8 (for U-Net stages)',
    validate: (c) => {
      const h = c.imgHeight as number;
      const w = c.imgWidth as number;
      return (!h || h % 8 === 0) && (!w || w % 8 === 0);
    },
  },
];

const GNN_CONSTRAINTS: ParamConstraint[] = [
  {
    description: 'nodeFeatDim should be a power of 2',
    validate: (c) => {
      const d = c.nodeFeatDim as number;
      return !d || (d > 0 && (d & (d - 1)) === 0);
    },
  },
];

// ─── Per-family definitions ─────────────────────────────────────────

// ─── Shared parameter blocks ────────────────────────────────────────
//
// Optimisation, regularisation and parallelism settings apply to every family.
// Defining them once keeps the ranges and descriptions consistent, and every
// family below splices them in, so the picker offers the same training controls
// regardless of the architecture being designed.

/** Optimisation settings: how the weights are updated. */
export const TRAINING_PARAMS: HyperparameterDef[] = [
  {
    key: 'learningRate',
    label: 'Learning Rate',
    description:
      'Optimizer step size. The single most sensitive training hyperparameter: too high diverges, too low stalls. Transformers are typically trained between 1e-5 and 3e-4, scaling down as the model grows.',
    type: 'float',
    defaultValue: 3e-4,
    range: { min: 1e-6, max: 1e-2, logScale: true },
    required: true,
    priority: 'critical',
    group: 'training',
  },
  {
    key: 'batchSize',
    label: 'Batch Size',
    description:
      'Samples per optimizer step. Drives activation memory linearly and sets the gradient noise scale.',
    type: 'int',
    defaultValue: 8,
    range: { min: 1, max: 4096, step: 1 },
    required: true,
    priority: 'critical',
    group: 'training',
  },
  {
    key: 'optimizer',
    label: 'Optimizer',
    description:
      'Update rule. AdamW is the Transformer standard; it keeps two moment estimates per parameter, so optimizer state costs about 8 bytes per parameter in fp32.',
    type: 'categorical',
    defaultValue: 'adamw',
    options: [
      { value: 'adamw', label: 'AdamW (decoupled weight decay)' },
      { value: 'adam', label: 'Adam' },
      { value: 'sgd', label: 'SGD (+ momentum)' },
      { value: 'adafactor', label: 'Adafactor (memory-efficient)' },
      { value: 'lion', label: 'Lion (sign-based)' },
    ],
    priority: 'high',
    group: 'training',
  },
  {
    key: 'numEpochs',
    label: 'Epochs',
    description:
      'Passes over the dataset. Combined with dataset size it determines the step budget, and therefore the training time, cost and carbon totals.',
    type: 'float',
    defaultValue: 1,
    range: { min: 0.01, max: 1000, step: 0.01 },
    priority: 'high',
    group: 'training',
  },
  {
    key: 'maxSteps',
    label: 'Max Steps',
    description:
      'Explicit optimizer-step budget. Leave at 0 to derive it from epochs and dataset size.',
    type: 'int',
    defaultValue: 0,
    range: { min: 0, max: 10_000_000, step: 1 },
    priority: 'medium',
    group: 'training',
  },
  {
    key: 'warmupSteps',
    label: 'Warmup Steps',
    description:
      'Steps spent ramping the learning rate from near zero. Transformers are unstable without warmup; 1-5% of total steps is typical.',
    type: 'int',
    defaultValue: 2000,
    range: { min: 0, max: 100_000, step: 100 },
    priority: 'high',
    group: 'training',
  },
  {
    key: 'lrScheduler',
    label: 'LR Schedule',
    description: 'How the learning rate decays after warmup.',
    type: 'categorical',
    defaultValue: 'cosine',
    options: [
      { value: 'cosine', label: 'Cosine decay' },
      { value: 'linear', label: 'Linear decay' },
      { value: 'constant', label: 'Constant' },
      { value: 'inverse_sqrt', label: 'Inverse square root' },
    ],
    priority: 'medium',
    group: 'training',
  },
];

/** Regularisation: what keeps the model from overfitting. */
export const REGULARIZATION_PARAMS: HyperparameterDef[] = [
  {
    key: 'dropout',
    label: 'Dropout',
    description:
      'Probability of zeroing a unit during training. Large pretraining runs often set this to 0 and rely on data scale instead.',
    type: 'float',
    defaultValue: 0.0,
    range: { min: 0, max: 0.5, step: 0.05 },
    priority: 'medium',
    group: 'regularization',
  },
  {
    key: 'attnDrop',
    label: 'Attention Dropout',
    description: 'Dropout applied to the attention probabilities specifically.',
    type: 'float',
    defaultValue: 0.0,
    range: { min: 0, max: 0.5, step: 0.05 },
    priority: 'low',
    group: 'regularization',
  },
  {
    key: 'weightDecay',
    label: 'Weight Decay',
    description:
      'L2-style penalty applied by the optimizer. 0.1 is the common choice for large language models.',
    type: 'float',
    defaultValue: 0.1,
    range: { min: 0, max: 1, step: 0.01 },
    priority: 'medium',
    group: 'regularization',
  },
  {
    key: 'earlyStoppingPatience',
    label: 'Early Stopping Patience',
    description:
      'Epochs to wait for validation improvement before stopping. 0 disables early stopping.',
    type: 'int',
    defaultValue: 0,
    range: { min: 0, max: 100, step: 1 },
    priority: 'low',
    group: 'regularization',
  },
];

/** Parallelism and memory strategy for large-scale training. */
export const PARALLELISM_PARAMS: HyperparameterDef[] = [
  {
    key: 'tensorParallel',
    label: 'Tensor Parallel (TP)',
    description:
      'Shards individual weight matrices across GPUs. Cuts per-GPU parameter and activation memory, at the cost of an all-reduce inside every layer, so it is normally kept within one node.',
    type: 'int',
    defaultValue: 1,
    range: { min: 1, max: 64, step: 1 },
    priority: 'high',
    group: 'hardware',
  },
  {
    key: 'pipelineParallel',
    label: 'Pipeline Parallel (PP)',
    description:
      'Splits the layer stack into stages on different GPUs. Cheap on bandwidth but introduces pipeline bubbles unless micro-batches are numerous.',
    type: 'int',
    defaultValue: 1,
    range: { min: 1, max: 64, step: 1 },
    priority: 'high',
    group: 'hardware',
  },
  {
    key: 'expertParallel',
    label: 'Expert Parallel (EP)',
    description: 'Distributes Mixture-of-Experts experts across GPUs.',
    type: 'int',
    defaultValue: 1,
    range: { min: 1, max: 64, step: 1 },
    priority: 'medium',
    group: 'hardware',
  },
  {
    key: 'microBatchSize',
    label: 'Micro Batch Size',
    description:
      'Per-device batch processed before gradients accumulate. Global batch = micro batch x accumulation x data-parallel degree.',
    type: 'int',
    defaultValue: 1,
    range: { min: 1, max: 1024, step: 1 },
    priority: 'medium',
    group: 'hardware',
  },
  {
    key: 'gradAccumSteps',
    label: 'Gradient Accumulation',
    description:
      'Micro-batches accumulated per optimizer step. Buys a large effective batch on limited memory, at proportionally more time per step.',
    type: 'int',
    defaultValue: 1,
    range: { min: 1, max: 1024, step: 1 },
    priority: 'medium',
    group: 'hardware',
  },
  {
    key: 'gradientCheckpointing',
    label: 'Activation Checkpointing',
    description:
      'Recomputes activations during the backward pass instead of storing them. Cuts activation memory to roughly sqrt(L) layers for about 30% more compute.',
    type: 'bool',
    defaultValue: false,
    priority: 'high',
    group: 'memory',
  },
  {
    key: 'zeroStage',
    label: 'ZeRO Stage',
    description:
      'Partitions optimizer state (1), gradients (2), then parameters (3) across data-parallel ranks.',
    type: 'categorical',
    defaultValue: 0,
    options: [
      { value: 0, label: 'Disabled' },
      { value: 1, label: 'Stage 1 - optimizer state' },
      { value: 2, label: 'Stage 2 - + gradients' },
      { value: 3, label: 'Stage 3 - + parameters' },
    ],
    priority: 'high',
    group: 'memory',
  },
];

/** Applied to every family, so training controls are always available. */
export const UNIVERSAL_PARAMS: HyperparameterDef[] = [
  ...TRAINING_PARAMS,
  ...REGULARIZATION_PARAMS,
  ...PARALLELISM_PARAMS,
];

export const FAMILY_HYPERPARAM_DEFS: Record<ArchitectureFamily, FamilyHyperparameterDefs> = {
  transformer: {
    family: 'transformer',
    globalConstraints: TRANSFORMER_CONSTRAINTS,
    params: [
      {
        key: 'hiddenDim',
        label: 'd_model (Hidden Dim)',
        description: 'Core dimension of the transformer through all layers',
        type: 'int',
        defaultValue: 768,
        range: { min: 64, max: 16384, step: 64, gridPoints: 8 },
        priority: 'critical',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of the transformer stack',
        type: 'int',
        defaultValue: 12,
        range: { min: 1, max: 256, step: 1, gridPoints: 8 },
        priority: 'critical',
        group: 'capacity',
      },
      {
        key: 'numHeads',
        label: 'Number of Heads',
        description: 'Parallel attention heads for multi-head attention',
        type: 'int',
        defaultValue: 12,
        range: { min: 1, max: 128, step: 1, gridPoints: 6 },
        constraints: [
          {
            description: 'Must divide hiddenDim evenly',
            validate: (c) => {
              const h = c.hiddenDim as number;
              const n = c.numHeads as number;
              return !h || !n || h % n === 0;
            },
          },
        ],
        priority: 'high',
        group: 'architecture',
      },
      {
        key: 'headDim',
        label: 'Head Dimension',
        description: 'Dimension per attention head (auto = hiddenDim / numHeads)',
        type: 'int',
        defaultValue: 64,
        range: { min: 16, max: 256, step: 16, gridPoints: 5 },
        isDerived: true,
        derivedFormula: 'hiddenDim / numHeads',
        priority: 'medium',
        group: 'architecture',
      },
      {
        key: 'ffnDim',
        label: 'FFN Intermediate Dim',
        description: 'Hidden dimension of the feed-forward network (typically 4× d_model)',
        type: 'int',
        defaultValue: 3072,
        range: { min: 128, max: 65536, step: 64, gridPoints: 8 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'seqLen',
        label: 'Sequence Length',
        description: 'Maximum sequence length / context window',
        type: 'int',
        defaultValue: 2048,
        range: { min: 128, max: 262144, logScale: true, gridPoints: 7 },
        priority: 'critical',
        group: 'memory',
      },
      {
        key: 'vocabSize',
        label: 'Vocabulary Size',
        description: 'Size of the token vocabulary',
        type: 'int',
        defaultValue: 32000,
        range: { min: 1000, max: 512000, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'kvHeads',
        label: 'KV Heads (GQA)',
        description: 'Number of key/value heads for grouped-query attention',
        type: 'int',
        defaultValue: 0,
        isDerived: true,
        derivedFormula: 'numHeads (MHA) or numHeads/4 (GQA)',
        priority: 'low',
        group: 'architecture',
      },
      {
        key: 'useBias',
        label: 'Use Bias',
        description: 'Enable bias terms in linear projections',
        type: 'bool',
        defaultValue: false,
        priority: 'low',
        group: 'architecture',
      },
      {
        key: 'useFlash',
        label: 'Flash Attention',
        description: 'Use Flash Attention algorithm for faster attention',
        type: 'bool',
        defaultValue: true,
        priority: 'low',
        group: 'compute',
      },
      {
        key: 'activation',
        label: 'Activation Function',
        description: 'Nonlinearity in the FFN',
        type: 'categorical',
        defaultValue: 'gelu',
        options: [
          { value: 'gelu', label: 'GELU' },
          { value: 'relu', label: 'ReLU' },
          { value: 'silu', label: 'SiLU/Swish' },
        ],
        priority: 'medium',
        group: 'architecture',
      },
      {
        key: 'useCache',
        label: 'KV Cache',
        description: 'Cached key/value tensors for autoregressive generation',
        type: 'bool',
        defaultValue: true,
        priority: 'low',
        group: 'memory',
      },
      {
        key: 'ropeTheta',
        label: 'RoPE θ (Theta)',
        description: 'Base frequency for Rotary Position Embedding',
        type: 'int',
        defaultValue: 10000,
        range: { min: 1000, max: 10000000, logScale: true, gridPoints: 5 },
        priority: 'low',
        group: 'architecture',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      hiddenDim: { min: 256, max: 4096, step: 128 },
      numLayers: { min: 4, max: 48 },
      numHeads: { min: 4, max: 32 },
      ffnDim: { min: 1024, max: 16384, step: 256 },
      seqLen: { min: 512, max: 32768, logScale: true },
      learningRate: { min: 1e-5, max: 0.01, logScale: true },
      batchSize: { min: 8, max: 1024, logScale: true },
    },
  },

  moe: {
    family: 'moe',
    globalConstraints: MOE_CONSTRAINTS,
    params: [
      {
        key: 'hiddenDim',
        label: 'd_model (Hidden Dim)',
        description: 'Core model dimension',
        type: 'int',
        defaultValue: 768,
        range: { min: 64, max: 16384, step: 64, gridPoints: 8 },
        priority: 'critical',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of the MoE transformer',
        type: 'int',
        defaultValue: 12,
        range: { min: 1, max: 128, step: 1, gridPoints: 6 },
        priority: 'critical',
        group: 'capacity',
      },
      {
        key: 'numHeads',
        label: 'Number of Heads',
        description: 'Parallel attention heads',
        type: 'int',
        defaultValue: 12,
        range: { min: 1, max: 64, step: 1, gridPoints: 5 },
        priority: 'high',
        group: 'architecture',
      },
      {
        key: 'ffnDim',
        label: 'Expert FFN Dim',
        description: 'Hidden dimension of each expert feed-forward network',
        type: 'int',
        defaultValue: 2048,
        range: { min: 128, max: 32768, step: 64, gridPoints: 7 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'numExperts',
        label: 'Number of Experts',
        description: 'Count of expert sub-networks in the MoE layer',
        type: 'int',
        defaultValue: 8,
        range: { min: 2, max: 1024, logScale: true, gridPoints: 6 },
        priority: 'critical',
        group: 'architecture',
      },
      {
        key: 'topK',
        label: 'Top-K Experts',
        description: 'Number of experts activated per token',
        type: 'int',
        defaultValue: 2,
        range: { min: 1, max: 16, step: 1 },
        priority: 'high',
        group: 'architecture',
      },
      {
        key: 'seqLen',
        label: 'Sequence Length',
        description: 'Maximum context length',
        type: 'int',
        defaultValue: 2048,
        range: { min: 128, max: 131072, logScale: true, gridPoints: 6 },
        priority: 'critical',
        group: 'memory',
      },
      {
        key: 'vocabSize',
        label: 'Vocabulary Size',
        description: 'Size of token vocabulary',
        type: 'int',
        defaultValue: 32000,
        range: { min: 1000, max: 256000, logScale: true, gridPoints: 4 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'dropout',
        label: 'Dropout Rate',
        description: 'Random neuron dropout for regularization',
        type: 'float',
        defaultValue: 0.1,
        range: { min: 0, max: 0.5, step: 0.05 },
        priority: 'medium',
        group: 'regularization',
      },
      {
        key: 'expertCapacity',
        label: 'Expert Capacity',
        description: 'Max tokens per expert (fraction of total)',
        type: 'float',
        defaultValue: 1.25,
        range: { min: 0.5, max: 4.0, step: 0.25 },
        priority: 'low',
        group: 'architecture',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      hiddenDim: { min: 256, max: 4096, step: 128 },
      numLayers: { min: 4, max: 32 },
      numExperts: { min: 4, max: 128, logScale: true },
      topK: { min: 1, max: 8 },
      seqLen: { min: 512, max: 16384, logScale: true },
    },
  },

  cnn: {
    family: 'cnn',
    globalConstraints: CNN_CONSTRAINTS,
    params: [
      {
        key: 'inChannels',
        label: 'Input Channels',
        description: 'Number of input image channels (3 for RGB)',
        type: 'int',
        defaultValue: 3,
        range: { min: 1, max: 256, step: 1 },
        priority: 'critical',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of the CNN',
        type: 'int',
        defaultValue: 50,
        range: { min: 1, max: 500, step: 1, gridPoints: 7 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'hiddenDim',
        label: 'Base Channel Count',
        description: 'Base number of channels after first conv layer',
        type: 'int',
        defaultValue: 64,
        range: { min: 16, max: 1024, step: 16, gridPoints: 6 },
        priority: 'high',
        group: 'capacity',
        parentKey: 'inChannels',
      },
      {
        key: 'numClasses',
        label: 'Number of Classes',
        description: 'Output classes for classification',
        type: 'int',
        defaultValue: 1000,
        range: { min: 2, max: 100000, logScale: true, gridPoints: 4 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'imgHeight',
        label: 'Image Height',
        description: 'Height of input image in pixels',
        type: 'int',
        defaultValue: 224,
        range: { min: 8, max: 4096, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'imgWidth',
        label: 'Image Width',
        description: 'Width of input image in pixels',
        type: 'int',
        defaultValue: 224,
        range: { min: 8, max: 4096, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'normType',
        label: 'Normalization Type',
        description: 'Type of normalization layer',
        type: 'categorical',
        defaultValue: 'batch_norm',
        options: [
          { value: 'batch_norm', label: 'BatchNorm' },
          { value: 'layer_norm', label: 'LayerNorm' },
          { value: 'none', label: 'None' },
        ],
        priority: 'medium',
        group: 'architecture',
      },
      {
        key: 'convActivation',
        label: 'Convolution Activation',
        description: 'Activation function after convolutions',
        type: 'categorical',
        defaultValue: 'relu',
        options: [
          { value: 'relu', label: 'ReLU' },
          { value: 'silu', label: 'SiLU' },
          { value: 'gelu', label: 'GELU' },
        ],
        priority: 'low',
        group: 'architecture',
      },
      {
        key: 'poolType',
        label: 'Pooling Type',
        description: 'Downsampling method',
        type: 'categorical',
        defaultValue: 'avg',
        options: [
          { value: 'avg', label: 'Average Pooling' },
          { value: 'max', label: 'Max Pooling' },
        ],
        priority: 'low',
        group: 'architecture',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      numLayers: { min: 10, max: 200, step: 10 },
      hiddenDim: { min: 32, max: 512, step: 32 },
      imgHeight: { min: 32, max: 1024, logScale: true },
      batchSize: { min: 16, max: 512, logScale: true },
    },
  },

  diffusion: {
    family: 'diffusion',
    globalConstraints: DIFFUSION_CONSTRAINTS,
    params: [
      {
        key: 'inChannels',
        label: 'Input Channels',
        description: 'Channels of input image / latent',
        type: 'int',
        defaultValue: 3,
        range: { min: 1, max: 16, step: 1 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'numDenoisingSteps',
        label: 'Denoising Steps',
        description: 'Number of diffusion denoising steps',
        type: 'int',
        defaultValue: 50,
        range: { min: 10, max: 5000, logScale: true, gridPoints: 6 },
        priority: 'critical',
        group: 'compute',
      },
      {
        key: 'guidanceScale',
        label: 'Guidance Scale (CFG)',
        description: 'Classifier-free guidance weight',
        type: 'float',
        defaultValue: 7.5,
        range: { min: 1.0, max: 20.0, step: 0.5 },
        priority: 'high',
        group: 'training',
      },
      {
        key: 'modelChannels',
        label: 'Base Model Channels',
        description: 'Base channel count in the U-Net',
        type: 'int',
        defaultValue: 128,
        range: { min: 32, max: 512, step: 32, gridPoints: 5 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'numResBlocks',
        label: 'ResBlocks per Stage',
        description: 'Number of residual blocks per U-Net stage',
        type: 'int',
        defaultValue: 2,
        range: { min: 1, max: 8, step: 1 },
        priority: 'medium',
        group: 'capacity',
      },
      {
        key: 'imgHeight',
        label: 'Image Height',
        description: 'Output image height',
        type: 'int',
        defaultValue: 512,
        range: { min: 16, max: 4096, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'imgWidth',
        label: 'Image Width',
        description: 'Output image width',
        type: 'int',
        defaultValue: 512,
        range: { min: 16, max: 4096, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'dropout',
        label: 'Dropout Rate',
        description: 'Dropout for regularization in U-Net',
        type: 'float',
        defaultValue: 0.1,
        range: { min: 0, max: 0.5, step: 0.05 },
        priority: 'low',
        group: 'regularization',
      },
      {
        key: 'outChannels',
        label: 'Output Channels',
        description: 'Channels of generated output',
        type: 'int',
        defaultValue: 3,
        range: { min: 1, max: 16, step: 1 },
        priority: 'low',
        group: 'capacity',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      numDenoisingSteps: { min: 20, max: 1000, logScale: true },
      guidanceScale: { min: 2.0, max: 15.0, step: 0.5 },
      modelChannels: { min: 64, max: 320, step: 64 },
      batchSize: { min: 1, max: 64, logScale: true },
    },
  },

  ssm: {
    family: 'ssm',
    globalConstraints: SSM_CONSTRAINTS,
    params: [
      {
        key: 'hiddenDim',
        label: 'Model Dimension (d)',
        description: 'Core hidden dimension of the SSM',
        type: 'int',
        defaultValue: 768,
        range: { min: 64, max: 8192, step: 64, gridPoints: 7 },
        priority: 'critical',
        group: 'capacity',
      },
      {
        key: 'dState',
        label: 'State Dimension (d_state)',
        description: 'Internal state size of the SSM (power of 2 recommended)',
        type: 'int',
        defaultValue: 16,
        range: { min: 2, max: 256, step: 2, gridPoints: 7 },
        priority: 'critical',
        group: 'architecture',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of the SSM stack',
        type: 'int',
        defaultValue: 24,
        range: { min: 1, max: 128, step: 1, gridPoints: 6 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'dtRank',
        label: 'DT Rank',
        description: 'Rank of the delta_t projection matrix',
        type: 'int',
        defaultValue: 48,
        range: { min: 8, max: 256, step: 8, gridPoints: 5 },
        priority: 'medium',
        group: 'architecture',
      },
      {
        key: 'expandFactor',
        label: 'Expand Factor',
        description: 'Expansion factor for SSM inner dimension',
        type: 'int',
        defaultValue: 2,
        range: { min: 1, max: 8, step: 1 },
        priority: 'medium',
        group: 'capacity',
      },
      {
        key: 'convKernel',
        label: 'Conv Kernel Size',
        description: 'Kernel size of the causal convolution in Mamba',
        type: 'int',
        defaultValue: 4,
        range: { min: 2, max: 16, step: 1 },
        priority: 'low',
        group: 'architecture',
      },
      {
        key: 'seqLen',
        label: 'Sequence Length',
        description: 'Maximum sequence length',
        type: 'int',
        defaultValue: 2048,
        range: { min: 128, max: 131072, logScale: true, gridPoints: 6 },
        priority: 'high',
        group: 'memory',
      },
      {
        key: 'vocabSize',
        label: 'Vocabulary Size',
        description: 'Size of token vocabulary',
        type: 'int',
        defaultValue: 32000,
        range: { min: 1000, max: 256000, logScale: true, gridPoints: 4 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'dropout',
        label: 'Dropout Rate',
        description: 'Dropout for regularization',
        type: 'float',
        defaultValue: 0.1,
        range: { min: 0, max: 0.5, step: 0.05 },
        priority: 'low',
        group: 'regularization',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      hiddenDim: { min: 256, max: 4096, step: 128 },
      dState: { min: 8, max: 128, step: 8 },
      numLayers: { min: 4, max: 48 },
      seqLen: { min: 512, max: 16384, logScale: true },
    },
  },

  gnn: {
    family: 'gnn',
    globalConstraints: GNN_CONSTRAINTS,
    params: [
      {
        key: 'nodeFeatDim',
        label: 'Node Feature Dimension',
        description: 'Dimension of node feature vectors',
        type: 'int',
        defaultValue: 16,
        range: { min: 8, max: 1024, step: 8, gridPoints: 7 },
        priority: 'critical',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of the GNN message passing',
        type: 'int',
        defaultValue: 3,
        range: { min: 1, max: 100, step: 1, gridPoints: 5 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'numNodes',
        label: 'Number of Nodes',
        description: 'Number of nodes in the graph',
        type: 'int',
        defaultValue: 10000,
        range: { min: 10, max: 10000000, logScale: true, gridPoints: 6 },
        priority: 'high',
        group: 'data',
      },
      {
        key: 'numEdges',
        label: 'Number of Edges',
        description: 'Number of edges in the graph',
        type: 'int',
        defaultValue: 50000,
        range: { min: 10, max: 100000000, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'outDim',
        label: 'Output Dimension',
        description: 'Output feature dimension',
        type: 'int',
        defaultValue: 32,
        range: { min: 8, max: 1024, step: 8, gridPoints: 5 },
        priority: 'medium',
        group: 'capacity',
      },
      {
        key: 'edgeFeatDim',
        label: 'Edge Feature Dimension',
        description: 'Dimension of edge feature vectors',
        type: 'int',
        defaultValue: 8,
        range: { min: 2, max: 512, step: 2, gridPoints: 5 },
        priority: 'medium',
        group: 'capacity',
      },
      {
        key: 'aggrType',
        label: 'Aggregation Type',
        description: 'Neighbor message aggregation method',
        type: 'categorical',
        defaultValue: 'mean',
        options: [
          { value: 'mean', label: 'Mean' },
          { value: 'sum', label: 'Sum' },
          { value: 'max', label: 'Max' },
        ],
        priority: 'medium',
        group: 'architecture',
      },
      {
        key: 'dropout',
        label: 'Dropout Rate',
        description: 'Dropout for GNN regularization',
        type: 'float',
        defaultValue: 0.1,
        range: { min: 0, max: 0.5, step: 0.05 },
        priority: 'low',
        group: 'regularization',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      nodeFeatDim: { min: 16, max: 256, step: 16 },
      numLayers: { min: 2, max: 20 },
      numNodes: { min: 100, max: 100000, logScale: true },
    },
  },

  rnn: {
    family: 'rnn',
    params: [
      {
        key: 'hiddenSize',
        label: 'Hidden Size',
        description: 'Hidden state dimension of the RNN cell',
        type: 'int',
        defaultValue: 512,
        range: { min: 32, max: 4096, step: 32, gridPoints: 7 },
        priority: 'critical',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of stacked RNN layers',
        type: 'int',
        defaultValue: 2,
        range: { min: 1, max: 50, step: 1, gridPoints: 5 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'seqLen',
        label: 'Sequence Length',
        description: 'Maximum input sequence length',
        type: 'int',
        defaultValue: 512,
        range: { min: 16, max: 65536, logScale: true, gridPoints: 6 },
        priority: 'high',
        group: 'memory',
      },
      {
        key: 'vocabSize',
        label: 'Vocabulary Size',
        description: 'Size of token vocabulary',
        type: 'int',
        defaultValue: 32000,
        range: { min: 1000, max: 256000, logScale: true, gridPoints: 4 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'isBidirectional',
        label: 'Bidirectional',
        description: 'Process sequence in both directions',
        type: 'bool',
        defaultValue: false,
        priority: 'medium',
        group: 'architecture',
      },
      {
        key: 'dropout',
        label: 'Dropout Rate',
        description: 'Dropout between RNN layers',
        type: 'float',
        defaultValue: 0.1,
        range: { min: 0, max: 0.5, step: 0.05 },
        priority: 'low',
        group: 'regularization',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      hiddenSize: { min: 64, max: 2048, step: 64 },
      numLayers: { min: 1, max: 10 },
      seqLen: { min: 32, max: 8192, logScale: true },
    },
  },

  gan: {
    family: 'gan',
    params: [
      {
        key: 'hiddenDim',
        label: 'Hidden/Base Dim',
        description: 'Base feature dimension for GAN layers',
        type: 'int',
        defaultValue: 256,
        range: { min: 16, max: 4096, step: 16, gridPoints: 6 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of generator/discriminator',
        type: 'int',
        defaultValue: 4,
        range: { min: 1, max: 20, step: 1 },
        priority: 'medium',
        group: 'capacity',
      },
      {
        key: 'imgHeight',
        label: 'Image Height',
        description: 'Output/generated image height',
        type: 'int',
        defaultValue: 64,
        range: { min: 8, max: 2048, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'imgWidth',
        label: 'Image Width',
        description: 'Output/generated image width',
        type: 'int',
        defaultValue: 64,
        range: { min: 8, max: 2048, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'inChannels',
        label: 'Channels',
        description: 'Number of image channels',
        type: 'int',
        defaultValue: 3,
        range: { min: 1, max: 16, step: 1 },
        priority: 'medium',
        group: 'data',
      },
      {
        key: 'learningRate',
        label: 'Learning Rate',
        description: 'Learning rate for GAN training',
        type: 'float',
        defaultValue: 0.0002,
        range: { min: 1e-6, max: 0.01, logScale: true, gridPoints: 6 },
        priority: 'critical',
        group: 'training',
      },
      {
        key: 'batchSize',
        label: 'Batch Size',
        description: 'Training batch size',
        type: 'int',
        defaultValue: 64,
        range: { min: 1, max: 512, logScale: true, gridPoints: 5 },
        priority: 'high',
        group: 'memory',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      hiddenDim: { min: 32, max: 1024, step: 32 },
      learningRate: { min: 1e-5, max: 0.005, logScale: true },
      batchSize: { min: 8, max: 256, logScale: true },
    },
  },

  snn: {
    family: 'snn',
    params: [
      {
        key: 'hiddenDim',
        label: 'Hidden Dimension',
        description: 'Core dimension of the SNN layers',
        type: 'int',
        defaultValue: 256,
        range: { min: 32, max: 4096, step: 32, gridPoints: 6 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of the spiking network',
        type: 'int',
        defaultValue: 4,
        range: { min: 1, max: 50, step: 1, gridPoints: 5 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'timesteps',
        label: 'Timesteps',
        description: 'Number of simulation timesteps for spike propagation',
        type: 'int',
        defaultValue: 100,
        range: { min: 10, max: 1000, logScale: true, gridPoints: 5 },
        priority: 'critical',
        group: 'compute',
      },
      {
        key: 'dState',
        label: 'State Dimension (d_state)',
        description: 'Internal neuron state dimension',
        type: 'int',
        defaultValue: 16,
        range: { min: 2, max: 128, step: 2, gridPoints: 5 },
        priority: 'medium',
        group: 'architecture',
      },
      {
        key: 'spikeRate',
        label: 'Spike Rate',
        description: 'Target firing rate for spike regularization',
        type: 'float',
        defaultValue: 0.05,
        range: { min: 0.01, max: 0.5, step: 0.01 },
        priority: 'medium',
        group: 'training',
      },
      {
        key: 'dropout',
        label: 'Dropout Rate',
        description: 'Dropout for SNN regularization',
        type: 'float',
        defaultValue: 0.1,
        range: { min: 0, max: 0.5, step: 0.05 },
        priority: 'low',
        group: 'regularization',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      hiddenDim: { min: 64, max: 1024, step: 64 },
      timesteps: { min: 20, max: 500, logScale: true },
      numLayers: { min: 2, max: 20 },
    },
  },

  rl: {
    family: 'rl',
    params: [
      {
        key: 'hiddenDim',
        label: 'Hidden Dimension',
        description: 'Hidden dimension of policy/value networks',
        type: 'int',
        defaultValue: 256,
        range: { min: 32, max: 2048, step: 32, gridPoints: 5 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description: 'Depth of RL network',
        type: 'int',
        defaultValue: 2,
        range: { min: 1, max: 20, step: 1 },
        priority: 'medium',
        group: 'capacity',
      },
      {
        key: 'actionDim',
        label: 'Action Dimension',
        description: 'Number of possible actions',
        type: 'int',
        defaultValue: 18,
        range: { min: 1, max: 10000, logScale: true, gridPoints: 5 },
        priority: 'high',
        group: 'data',
      },
      {
        key: 'stateDim',
        label: 'State Dimension',
        description: 'Dimension of the state observation',
        type: 'int',
        defaultValue: 128,
        range: { min: 4, max: 4096, step: 4, gridPoints: 6 },
        priority: 'high',
        group: 'data',
      },
      {
        key: 'learningRate',
        label: 'Learning Rate',
        description: 'Learning rate for RL policy update',
        type: 'float',
        defaultValue: 0.0003,
        range: { min: 1e-6, max: 0.01, logScale: true, gridPoints: 6 },
        priority: 'critical',
        group: 'training',
      },
      {
        key: 'batchSize',
        label: 'Batch Size',
        description: 'Experience replay batch size',
        type: 'int',
        defaultValue: 64,
        range: { min: 1, max: 4096, logScale: true, gridPoints: 5 },
        priority: 'medium',
        group: 'memory',
      },
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      hiddenDim: { min: 64, max: 1024, step: 64 },
      stateDim: { min: 8, max: 1024, step: 8 },
      learningRate: { min: 1e-5, max: 0.005, logScale: true },
    },
  },

  // experimental shares transformer params
  experimental: {
    family: 'experimental',
    globalConstraints: TRANSFORMER_CONSTRAINTS,
    params: [
      {
        key: 'hiddenDim',
        label: 'Hidden Dimension',
        description:
          'Width of the residual stream. Drives parameter count quadratically through the projection matrices, so it is the strongest single lever on model size.',
        type: 'int',
        defaultValue: 768,
        range: { min: 64, max: 16384, step: 64, gridPoints: 6 },
        priority: 'high',
        group: 'capacity',
      },
      {
        key: 'numLayers',
        label: 'Number of Layers',
        description:
          'Depth of the block stack. Parameters and FLOPs scale linearly with it, while deeper stacks are harder to keep numerically stable.',
        type: 'int',
        defaultValue: 6,
        range: { min: 1, max: 128, step: 1, gridPoints: 5 },
        priority: 'high',
        group: 'capacity',
      },
      // batchSize and learningRate come from UNIVERSAL_PARAMS, which documents
      // them properly; redefining them here only shadowed the better entries.
          ...UNIVERSAL_PARAMS,
    ],
    defaultSearchSpace: {
      hiddenDim: { min: 128, max: 2048, step: 128 },
      numLayers: { min: 2, max: 24 },
    },
  },
};

// ─── Helper functions ───────────────────────────────────────────────

/** Get optimizable params for a given family */
export function getParamsForFamily(family: ArchitectureFamily): HyperparameterDef[] {
  const defs = FAMILY_HYPERPARAM_DEFS[family];
  if (!defs) return [];

  // `required` is derived from MANDATORY_FIELDS rather than restated on each
  // definition, so the panel's badges and the validation that gates analysis
  // can never disagree about which parameters a family actually needs.
  const mandatory = new Set<string>([
    ...(MANDATORY_FIELDS.common ?? []),
    ...(MANDATORY_FIELDS[family] ?? []),
  ] as string[]);

  // Families splice in the universal training block, and some also define a
  // parameter of their own with the same key (MoE tunes its own dropout range).
  // Keep the first occurrence — the family-specific one, which is listed before
  // the universal block — so the panel never renders a key twice.
  const seen = new Set<string>();
  return defs.params
    .filter((p) => !p.isDerived)
    .filter((p) => {
      const key = p.key as string;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    })
    .map((p) => (mandatory.has(p.key as string) ? { ...p, required: true } : p));
}

/** Parameters a family cannot be compiled without. */
export function getRequiredParamsForFamily(family: ArchitectureFamily): HyperparameterDef[] {
  return getParamsForFamily(family).filter((p) => p.required);
}

/** Parameters that fall back to a documented default when left unset. */
export function getOptionalParamsForFamily(family: ArchitectureFamily): HyperparameterDef[] {
  return getParamsForFamily(family).filter((p) => !p.required);
}

/** Get family constraints */
export function getConstraintsForFamily(family: ArchitectureFamily): ParamConstraint[] {
  return FAMILY_HYPERPARAM_DEFS[family]?.globalConstraints ?? [];
}

/** Get default search space for a family */
export function getDefaultSearchSpace(
  family: ArchitectureFamily,
): Partial<Record<HyperparamKey, SearchRange>> {
  return FAMILY_HYPERPARAM_DEFS[family]?.defaultSearchSpace ?? {};
}

/** Validate a full config against all family constraints */
export function validateConfig(
  config: Partial<HardwareConfig>,
  family: ArchitectureFamily,
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const defs = FAMILY_HYPERPARAM_DEFS[family];
  if (!defs) return { valid: true, errors: [] };

  for (const constraint of defs.globalConstraints ?? []) {
    if (!constraint.validate(config as Record<string, unknown>)) {
      errors.push(constraint.description);
    }
  }

  for (const param of defs.params) {
    for (const constraint of param.constraints ?? []) {
      if (!constraint.validate(config as Record<string, unknown>)) {
        errors.push(`${param.label}: ${constraint.description}`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

/** Auto-fix a config to satisfy constraints */
export function autoFixConfig(
  config: Partial<HardwareConfig>,
  family: ArchitectureFamily,
): Partial<HardwareConfig> {
  let fixed = { ...config } as Record<string, unknown>;
  const defs = FAMILY_HYPERPARAM_DEFS[family];
  if (!defs) return config;

  for (const constraint of defs.globalConstraints ?? []) {
    if (!constraint.validate(fixed) && constraint.fix) {
      fixed = constraint.fix(fixed);
    }
  }

  for (const param of defs.params) {
    for (const constraint of param.constraints ?? []) {
      if (!constraint.validate(fixed) && constraint.fix) {
        fixed = constraint.fix(fixed);
      }
    }
  }

  return fixed as Partial<HardwareConfig>;
}
