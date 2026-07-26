/**
 * hyperparameterOptimizer.ts
 *
 * Moteur d'optimisation d'hyperparamètres NEURAX.
 *
 * Stratégies disponibles :
 *  - GRID_SEARCH  : exploration exhaustive sur une grille définie
 *  - RANDOM_SEARCH : échantillonnage aléatoire dans l'espace des hyperparams
 *  - BAYESIAN      : optimisation bayésienne simple (GP-based)
 *
 * Chaque stratégie génère des configurations candidates et les évalue
 * via des métriques estimées localement (params, FLOPs, mémoire).
 */

import { ArchitectureFamily } from '@/types/plugins';
import { HardwareConfig } from '@/contexts/HardwareContext';
import {
  HyperparameterDef,
  SearchRange,
  getParamsForFamily,
  validateConfig,
  autoFixConfig,
} from './hyperparameterDefs';

// ─── Types ───────────────────────────────────────────────────────────

export type OptimizationStrategy = 'grid_search' | 'random_search' | 'bayesian';

export type OptimizationObjective =
  | 'minimize_latency'
  | 'minimize_memory'
  | 'maximize_throughput'
  | 'minimize_cost'
  | 'minimize_params'
  | 'balanced';

export interface OptimizationTarget {
  objective: OptimizationObjective;
  /** Hardware name to optimize for */
  targetHardware?: string;
  /** Constraint: max VRAM usage in GB */
  maxMemoryGb?: number;
  /** Constraint: max latency in ms */
  maxLatencyMs?: number;
  /** Constraint: min throughput in tokens/s */
  minThroughput?: number;
  /** Custom weights for composite objective */
  weights?: Record<string, number>;
}

export interface CandidateConfig {
  /** Unique ID */
  id: string;
  /** The hyperparameter configuration */
  config: Partial<HardwareConfig>;
  /** Estimated metrics (local estimation) */
  estimated: EstimatedMetrics;
  /** Actual metrics from analysis run (filled after API call) */
  actual?: EstimatedMetrics;
  /** Rank (1 = best) */
  rank?: number;
  /** Normalized score (0-100) */
  score?: number;
}

export interface EstimatedMetrics {
  totalParams: number;
  totalFlops: number;
  peakMemoryGb: number;
  latencyMs: number;
  throughputTokensPerS: number;
  trainingCostUsd: number;
  trainingTimeHours: number;
  memoryBandwidthGbs: number;
  /** Whether this fits on the target hardware */
  fitsOnHardware: boolean;
  /** Bottleneck analysis */
  bottleneck: 'compute' | 'memory' | 'communication' | 'none';
  /** Roofline analysis */
  arithmeticIntensity: number;
}

export interface OptimizationResult {
  strategy: OptimizationStrategy;
  family: ArchitectureFamily;
  totalCandidates: number;
  evaluatedCount: number;
  candidates: CandidateConfig[];
  bestConfig: CandidateConfig | null;
  elapsedMs: number;
}

// ─── Parameter Generation ─────────────────────────────────────────

/** Generate grid points from a range */
function gridPoints(range: SearchRange): number[] {
  const { min, max, step, gridPoints: gp, logScale } = range;
  const points = gp ?? (step ? Math.ceil((max - min) / step) : 10);
  const clamp = points > 50 ? 50 : points; // Safety cap

  if (logScale) {
    const logMin = Math.log10(min);
    const logMax = Math.log10(max);
    const stepSize = (logMax - logMin) / (clamp - 1 || 1);
    return Array.from({ length: clamp }, (_, i) =>
      Math.round(Math.pow(10, logMin + i * stepSize) * 100) / 100,
    );
  }

  const stepSize = step ?? (max - min) / clamp;
  return Array.from({ length: clamp + 1 }, (_, i) => {
    const val = min + i * stepSize;
    return val <= max ? Math.round(val * 100) / 100 : -1;
  }).filter((v) => v >= 0 && v <= max);
}

/** Generate random values within a range */
function randomValue(param: HyperparameterDef): number | string | boolean {
  switch (param.type) {
    case 'int': {
      const r = param.range!;
      if (r.logScale) {
        const logMin = Math.log10(r.min);
        const logMax = Math.log10(r.max);
        return Math.round(Math.pow(10, logMin + Math.random() * (logMax - logMin)));
      }
      return Math.round(r.min + Math.random() * (r.max - r.min));
    }
    case 'float': {
      const r = param.range!;
      if (r.logScale) {
        const logMin = Math.log10(r.min);
        const logMax = Math.log10(r.max);
        return Math.round(Math.pow(10, logMin + Math.random() * (logMax - logMin)) * 10000) / 10000;
      }
      return Math.round((r.min + Math.random() * (r.max - r.min)) * 10000) / 10000;
    }
    case 'categorical':
      return param.options![Math.floor(Math.random() * param.options!.length)].value;
    case 'bool':
      return Math.random() > 0.5;
    default:
      return param.defaultValue;
  }
}

/**
 * Generate candidate configs using grid search.
 * Samples maxCombinations to avoid exploding search space.
 */
export function generateGridCandidates(
  family: ArchitectureFamily,
  params: HyperparameterDef[],
  searchSpace?: Partial<Record<keyof HardwareConfig, SearchRange>>,
  maxCombinations = 100,
): Partial<HardwareConfig>[] {
  const activeParams = params.filter(
    (p) => p.type === 'int' || p.type === 'float' || p.type === 'categorical' || p.type === 'bool',
  );

  // For categorical/bool, treat as direct options
  const gridVectors: Record<string, (number | string | boolean)[]> = {};

  for (const param of activeParams) {
    const range = searchSpace?.[param.key] ?? param.range;
    if (!range && param.type !== 'categorical' && param.type !== 'bool') continue;

    if (param.type === 'int' || param.type === 'float') {
      gridVectors[param.key as string] = gridPoints(range!);
    } else if (param.type === 'categorical') {
      gridVectors[param.key as string] = param.options!.map((o) => o.value);
    } else if (param.type === 'bool') {
      gridVectors[param.key as string] = [true, false];
    }
  }

  // Compute Cartesian product with sampling
  const keys = Object.keys(gridVectors);
  if (keys.length === 0) return [];

  // Estimate total combinations
  let totalCombos = 1;
  for (const key of keys) {
    totalCombos *= gridVectors[key].length;
  }

  if (totalCombos <= maxCombinations) {
    // Full grid
    return cartesianProduct(keys, gridVectors).slice(0, maxCombinations);
  }

  // Sample random combinations
  const sampled: Partial<HardwareConfig>[] = [];
  const seen = new Set<string>();
  let attempts = 0;

  while (sampled.length < maxCombinations && attempts < maxCombinations * 10) {
    attempts++;
    const combo: Record<string, unknown> = {};
    const keyStr: string[] = [];

    for (const key of keys) {
      const vals = gridVectors[key];
      const val = vals[Math.floor(Math.random() * vals.length)];
      combo[key] = val;
      keyStr.push(`${key}=${val}`);
    }

    const hash = keyStr.join('|');
    if (seen.has(hash)) continue;
    seen.add(hash);

    const config = autoFixConfig(combo as Partial<HardwareConfig>, family);
    const validation = validateConfig(config, family);
    if (validation.valid) {
      sampled.push(config);
    }
  }

  return sampled;
}

function cartesianProduct(
  keys: string[],
  vectors: Record<string, (number | string | boolean)[]>,
): Partial<HardwareConfig>[] {
  if (keys.length === 0) return [];

  function build(depth: number, current: Record<string, unknown>): Partial<HardwareConfig>[] {
    if (depth === keys.length) {
      return [current as Partial<HardwareConfig>];
    }
    const key = keys[depth];
    const results: Partial<HardwareConfig>[] = [];
    for (const val of vectors[key]) {
      results.push(...build(depth + 1, { ...current, [key]: val }));
    }
    return results;
  }

  return build(0, {});
}

/** Generate random search candidates */
export function generateRandomCandidates(
  family: ArchitectureFamily,
  params: HyperparameterDef[],
  count = 50,
  _searchSpace?: Partial<Record<keyof HardwareConfig, SearchRange>>,
): Partial<HardwareConfig>[] {
  const candidates: Partial<HardwareConfig>[] = [];
  const seen = new Set<string>();
  let attempts = 0;

  while (candidates.length < count && attempts < count * 20) {
    attempts++;
    const config: Record<string, unknown> = {};

    for (const param of params) {
      config[param.key as string] = randomValue(param);
    }

    const hash = JSON.stringify(config);
    if (seen.has(hash)) continue;
    seen.add(hash);

    const fixed = autoFixConfig(config as Partial<HardwareConfig>, family);
    const validation = validateConfig(fixed, family);
    if (validation.valid) {
      candidates.push(fixed);
    }
  }

  return candidates;
}

// ─── Metrics Estimation ──────────────────────────────────────────────

/**
 * Estimate model metrics from a config without a backend call.
 * Uses analytical formulas from the existing compiler logic.
 */
export function estimateMetrics(
  config: Partial<HardwareConfig>,
  family: ArchitectureFamily,
): EstimatedMetrics {
  // Extract key params with defaults
  const d = config.hiddenDim ?? 768;
  const L = config.numLayers ?? 12;
  const h = config.numHeads ?? Math.max(1, Math.round(d / 64));
  const ff = config.ffnDim ?? d * 4;
  const seq = config.seqLen ?? 2048;
  const vocab = config.vocabSize ?? 32000;
  const bs = config.batchSize ?? 64;
  const hw = config.hardware ?? 'H100';

  // Memory bandwidth estimates per GPU
  const hwBandwidth: Record<string, number> = {
    H100: 3352,
    H200: 4800,
    A100: 2039,
    L40: 864,
    V100: 900,
    T4: 300,
    RTX4090: 1008,
    RTX4080: 717,
    RTX3090: 936,
    A6000: 768,
    L40S: 864,
    GH200: 4800,
  };
  const bw = hwBandwidth[hw] ?? 1000;

  // Compute estimated metrics based on family
  switch (family) {
    case 'transformer':
    case 'moe':
      return estimateTransformerMetrics(config, family, d, L, h, ff, seq, vocab, bs, bw);
    case 'cnn':
      return estimateCnnMetrics(config, bs, bw);
    case 'ssm':
      return estimateSsmMetrics(config, d, L, seq, vocab, bs, bw);
    case 'diffusion':
      return estimateDiffusionMetrics(config, bs, bw);
    case 'gnn':
      return estimateGnnMetrics(config, bs, bw);
    case 'rnn':
      return estimateRnnMetrics(config, bs, bw);
    case 'snn':
      return estimateSnnMetrics(config, bs, bw);
    default:
      return {
        totalParams: d * d * L,
        totalFlops: 6 * d * d * L * seq,
        peakMemoryGb: (d * L * seq * 4) / 1e9,
        latencyMs: 10,
        throughputTokensPerS: 100,
        trainingCostUsd: 0,
        trainingTimeHours: 0,
        memoryBandwidthGbs: bw,
        fitsOnHardware: true,
        bottleneck: 'compute',
        arithmeticIntensity: 0,
      };
  }
}

function estimateTransformerMetrics(
  config: Partial<HardwareConfig>,
  family: ArchitectureFamily,
  d: number,
  L: number,
  _h: number,
  ff: number,
  seq: number,
  vocab: number,
  bs: number,
  bw: number,
): EstimatedMetrics {
  // Token embedding params: vocab * d
  const embedParams = vocab * d;

  // Per-layer: attention (4 * d^2 for QKV + proj) + FFN (d*ff + ff*d = 2*d*ff)
  // For MoE: add expert params
  let expertParams = 0;
  let expertFlopMultiplier = 1;
  if (family === 'moe') {
    const numExperts = config.numExperts ?? 8;
    const topK = config.topK ?? 2;
    // Experts add: numExperts * (2 * d * ff) for each MoE layer
    const numMoELayers = Math.floor(L / 2); // Every other layer is MoE
    expertParams = numMoELayers * numExperts * 2 * d * ff;
    expertFlopMultiplier = 1 + (topK / numExperts) * 0.5; // Approximate sparsity factor
  }

  const attnParams = 4 * d * d; // QKV + proj
  const ffnParams = 2 * d * ff;
  const layerParams = attnParams + ffnParams;
  const totalParams = embedParams + L * layerParams + expertParams;

  // FLOPs per token per layer: attention (4*d^2 + 2*d*seq) + FFN (2*d*ff)
  // For autoregressive: factor of 2 for backward
  const attnFlopsPerToken = 4 * d * d + 2 * d * seq;
  const ffnFlopsPerToken = 2 * d * ff;
  const layerFlops = (attnFlopsPerToken + ffnFlopsPerToken) * expertFlopMultiplier;
  const totalFlops = 3 * (embedParams * seq + L * layerFlops); // Factor 3 for forward+backward

  // Memory estimate
  const paramBytes = totalParams * 2; // fp16 = 2 bytes
  const activationMemory = L * bs * seq * d * 2 * 4; // activations for backward
  const optimizerMemory = totalParams * 4; // Adam: 2 * 4 bytes
  const peakMemory = paramBytes + activationMemory + optimizerMemory;
  const peakGb = peakMemory / 1e9;
  const gpuMemGb = config.gpuMemoryGb ?? 80;

  // Latency: compute + memory-bound
  const flops = totalFlops / seq / bs; // FLOPs per sample
  const computeTime = flops / (67e12); // Approx H100 TFLOPS
  const memoryTime = peakGb * 1e9 / (bw * 1e9);
  const latencyMs = (computeTime + memoryTime) * 1000;

  // Throughput
  const throughput = bw * 1e9 / (peakGb * 1e9 * 2) * bs;

  // Training cost (rough estimate)
  const totalTflops = totalFlops / 1e12;
  const trainingHours = totalTflops / (67 * (config.gpuCount ?? 1));
  const costPerHour = hwCost(config.hardware ?? 'H100');
  const trainingCost = trainingHours * costPerHour * (config.gpuCount ?? 1);

  const ai = (totalFlops / seq / bs) / (peakGb * 1e9);

  return {
    totalParams,
    totalFlops,
    peakMemoryGb: peakGb,
    latencyMs: Math.max(0.1, latencyMs),
    throughputTokensPerS: throughput,
    trainingCostUsd: trainingCost,
    trainingTimeHours: trainingHours,
    memoryBandwidthGbs: bw,
    fitsOnHardware: peakGb * 1.2 <= gpuMemGb, // 20% safety margin
    bottleneck: ai < 100 ? 'memory' : 'compute',
    arithmeticIntensity: ai,
  };
}

function estimateCnnMetrics(
  config: Partial<HardwareConfig>,
  bs: number,
  bw: number,
): EstimatedMetrics {
  const inC = config.inChannels ?? 3;
  const imgH = config.imgHeight ?? 224;
  const imgW = config.imgWidth ?? 224;
  const numLayers = config.numLayers ?? 50;
  const baseCh = config.hiddenDim ?? 64;
  const numClasses = config.numClasses ?? 1000;

  // ResNet-like estimation
  const stages = [baseCh, baseCh * 2, baseCh * 4, baseCh * 8];
  let totalParams = 0;
  let totalFlops = 0;
  let size = imgH * imgW;

  let inCh = inC;
  for (const outCh of stages) {
    const numBlocks = Math.max(1, Math.floor(numLayers / stages.length));
    for (let b = 0; b < numBlocks; b++) {
      // Conv params: inCh * outCh * 3 * 3
      totalParams += inCh * outCh * 9;
      // Conv FLOPs: 2 * inCh * outCh * 3 * 3 * size
      totalFlops += 2 * inCh * outCh * 9 * size;
      inCh = outCh;
    }
    // Downsample: halve spatial size
    size /= 2;
  }

  // Classifier
  totalParams += inCh * numClasses;
  totalFlops += 2 * inCh * numClasses;

  const paramBytes = totalParams * 2;
  const activationMem = bs * imgH * imgW * baseCh * 2 * 4;
  const peakGb = (paramBytes + activationMem) / 1e9;
  const gpuMemGb = config.gpuMemoryGb ?? 24;

  const flopsPerImage = totalFlops;
  const computeTime = flopsPerImage / 82e12; // RTX4090 TFLOPS
  const latencyMs = (computeTime * 1000);

  const throughput = bs / Math.max(latencyMs / 1000, 0.001);

  return {
    totalParams,
    totalFlops,
    peakMemoryGb: peakGb,
    latencyMs: Math.max(1, latencyMs),
    throughputTokensPerS: throughput,
    trainingCostUsd: 0,
    trainingTimeHours: 0,
    memoryBandwidthGbs: bw,
    fitsOnHardware: peakGb * 1.2 <= gpuMemGb,
    bottleneck: peakGb > gpuMemGb * 0.8 ? 'memory' : 'compute',
    arithmeticIntensity: totalFlops / (paramBytes * 1.5),
  };
}

function estimateSsmMetrics(
  config: Partial<HardwareConfig>,
  d: number,
  L: number,
  seq: number,
  vocab: number,
  bs: number,
  bw: number,
): EstimatedMetrics {
  const dState = config.dState ?? 16;
  const expand = config.expandFactor ?? 2;

  const embedParams = vocab * d;
  const ssmIn = d * expand;
  const ssmParams = L * (d * ssmIn + dState * dState + dState * d);
  const layerParams = ssmParams;
  const totalParams = embedParams + layerParams;

  // SSM FLOPs: O(L * d * dState * seq) for the scan
  const totalFlops = totalParams * seq * 2;

  const peakGb = (totalParams * 2 + bs * seq * d * 2) / 1e9;
  const gpuMemGb = config.gpuMemoryGb ?? 80;

  const ai = totalFlops / (peakGb * 1e9);

  return {
    totalParams,
    totalFlops,
    peakMemoryGb: peakGb,
    latencyMs: 5,
    throughputTokensPerS: bw * 1e9 / (peakGb * 1e9) * bs,
    trainingCostUsd: 0,
    trainingTimeHours: 0,
    memoryBandwidthGbs: bw,
    fitsOnHardware: peakGb * 1.2 <= gpuMemGb,
    bottleneck: dState > 64 ? 'memory' : 'compute',
    arithmeticIntensity: ai,
  };
}

function estimateDiffusionMetrics(
  config: Partial<HardwareConfig>,
  bs: number,
  bw: number,
): EstimatedMetrics {
  const steps = config.numDenoisingSteps ?? 50;
  const modelCh = config.modelChannels ?? 128;
  const imgH = config.imgHeight ?? 512;
  const imgW = config.imgWidth ?? 512;
  const inC = config.inChannels ?? 4;
  const numRes = config.numResBlocks ?? 2;

  // U-Net param estimate
  const ch = [modelCh, modelCh * 2, modelCh * 4, modelCh * 8, modelCh * 8];
  let totalParams = 0;
  for (let i = 0; i < ch.length - 1; i++) {
    const resBlockParams = numRes * 2 * ch[i] * (ch[i] + ch[i + 1]) / 2;
    const attnParams = (ch[i] > 0 ? ch[i] * ch[i] * 4 : 0);
    totalParams += resBlockParams + attnParams;
  }
  totalParams += inC * modelCh; // Input projection

  // FLOPs = params * steps * imgH * imgW (rough)
  const totalFlops = totalParams * steps * (imgH / 8) * (imgW / 8) * 2;

  const peakGb = (totalParams * 2 + bs * inC * imgH * imgW * 2) / 1e9;
  const gpuMemGb = config.gpuMemoryGb ?? 80;

  return {
    totalParams,
    totalFlops,
    peakMemoryGb: peakGb,
    latencyMs: steps * 10,
    throughputTokensPerS: bs / (steps * 0.01),
    trainingCostUsd: 0,
    trainingTimeHours: 0,
    memoryBandwidthGbs: bw,
    fitsOnHardware: peakGb * 1.2 <= gpuMemGb,
    bottleneck: steps > 100 ? 'compute' : 'memory',
    arithmeticIntensity: 0,
  };
}

function estimateGnnMetrics(
  config: Partial<HardwareConfig>,
  _bs: number,
  bw: number,
): EstimatedMetrics {
  const nodeDim = config.nodeFeatDim ?? 16;
  const edgeDim = config.edgeFeatDim ?? 8;
  const numNodes = config.numNodes ?? 10000;
  const numEdges = config.numEdges ?? 50000;
  const outDim = config.outDim ?? 32;
  const numLayers = config.numLayers ?? 3;

  const paramsPerLayer = nodeDim * outDim + edgeDim * outDim;
  const totalParams = numLayers * paramsPerLayer;

  // Message passing FLOPs: O(numEdges * nodeDim * outDim)
  const totalFlops = numLayers * numEdges * nodeDim * outDim * 2;

  const peakGb = (totalParams * 2 + numNodes * nodeDim * 4) / 1e9;
  const gpuMemGb = config.gpuMemoryGb ?? 80;

  return {
    totalParams,
    totalFlops,
    peakMemoryGb: peakGb,
    latencyMs: 2,
    throughputTokensPerS: 1000,
    trainingCostUsd: 0,
    trainingTimeHours: 0,
    memoryBandwidthGbs: bw,
    fitsOnHardware: peakGb * 1.2 <= gpuMemGb,
    bottleneck: numLayers > 10 ? 'memory' : 'compute',
    arithmeticIntensity: 0,
  };
}

function estimateRnnMetrics(
  config: Partial<HardwareConfig>,
  bs: number,
  bw: number,
): EstimatedMetrics {
  const h = config.hiddenSize ?? 512;
  const L = config.numLayers ?? 2;
  const seq = config.seqLen ?? 512;
  const vocab = config.vocabSize ?? 32000;
  const bidir = config.isBidirectional ?? false;

  const embedParams = vocab * h;
  const rnnCellParams = 4 * h * h + 4 * h; // LSTM: 4 gates * h^2
  const totalParams = embedParams + L * rnnCellParams * (bidir ? 2 : 1);

  const totalFlops = totalParams * seq * 3; // fwd + bwd

  const peakGb = (totalParams * 2 + bs * L * h * seq * 4) / 1e9;
  const gpuMemGb = config.gpuMemoryGb ?? 80;

  return {
    totalParams,
    totalFlops,
    peakMemoryGb: peakGb,
    latencyMs: seq * 0.1,
    throughputTokensPerS: bs * seq / Math.max(seq * 0.1 / 1000, 0.001),
    trainingCostUsd: 0,
    trainingTimeHours: 0,
    memoryBandwidthGbs: bw,
    fitsOnHardware: peakGb * 1.2 <= gpuMemGb,
    bottleneck: 'compute',
    arithmeticIntensity: 0,
  };
}

function estimateSnnMetrics(
  config: Partial<HardwareConfig>,
  bs: number,
  bw: number,
): EstimatedMetrics {
  const d = config.hiddenDim ?? 256;
  const L = config.numLayers ?? 4;
  const T = config.timesteps ?? 100;
  const spikeRate = config.spikeRate ?? 0.05;

  // SNNs are sparse: only spikeRate fraction of neurons fire
  const totalParams = L * d * d;
  const totalFlops = totalParams * T * spikeRate * 2;
  const peakGb = (totalParams * 2 + bs * d * L * 4) / 1e9;
  const gpuMemGb = config.gpuMemoryGb ?? 80;

  return {
    totalParams,
    totalFlops,
    peakMemoryGb: peakGb,
    latencyMs: T * 0.5,
    throughputTokensPerS: bs / (T * 0.0005),
    trainingCostUsd: 0,
    trainingTimeHours: 0,
    memoryBandwidthGbs: bw,
    fitsOnHardware: peakGb * 1.2 <= gpuMemGb,
    bottleneck: 'compute',
    arithmeticIntensity: 0,
  };
}

/** Rough GPU cost per hour */
function hwCost(gpu: string): number {
  const costs: Record<string, number> = {
    H100: 3.0,
    H200: 4.0,
    A100: 1.5,
    L40: 0.8,
    V100: 0.8,
    RTX4090: 0.5,
    RTX4080: 0.3,
    RTX3090: 0.3,
    T4: 0.35,
    A6000: 0.6,
    L40S: 0.9,
    GH200: 4.5,
  };
  return costs[gpu] ?? 0.5;
}

// ─── Scoring ─────────────────────────────────────────────────────────

/**
 * Score a configuration against an optimization objective.
 * Returns a normalized score (0-100, higher = better).
 */
export function scoreConfig(
  metrics: EstimatedMetrics,
  objective: OptimizationObjective,
  target?: OptimizationTarget,
): number {
  let score = 0;
  const weights = target?.weights ?? {
    params: 0.15,
    flops: 0.1,
    memory: 0.2,
    latency: 0.2,
    throughput: 0.15,
    cost: 0.1,
    fit: 0.1,
  };

  // Helper: normalize value (lower is better → 100 * (1 - normalized), capped)
  const normalize = (val: number, min: number, max: number, invert = false): number => {
    if (max <= min) return 50;
    const clamped = Math.max(min, Math.min(max, val));
    const normalized = (clamped - min) / (max - min);
    return invert ? normalized * 100 : (1 - normalized) * 100;
  };

  switch (objective) {
    case 'minimize_latency':
      score =
        normalize(Math.log10(metrics.latencyMs), -1, 3, true) * 0.5 +
        (metrics.fitsOnHardware ? 50 : 0);
      break;

    case 'minimize_memory':
      score =
        normalize(metrics.peakMemoryGb, 0.1, 160) * 0.6 +
        (metrics.fitsOnHardware ? 40 : 0);
      break;

    case 'maximize_throughput':
      score =
        normalize(Math.log10(metrics.throughputTokensPerS), 0, 6, true) * 0.6 +
        (metrics.fitsOnHardware ? 40 : 0);
      break;

    case 'minimize_cost':
      score =
        normalize(Math.log10(metrics.trainingCostUsd + 1), 0, 5) * 0.4 +
        normalize(Math.log10(metrics.totalParams), 4, 12) * 0.2 +
        normalize(metrics.peakMemoryGb, 0.1, 160) * 0.2 +
        (metrics.fitsOnHardware ? 20 : 0);
      break;

    case 'minimize_params':
      score =
        normalize(Math.log10(metrics.totalParams), 4, 12) * 0.5 +
        normalize(metrics.peakMemoryGb, 0.1, 160) * 0.3 +
        (metrics.fitsOnHardware ? 20 : 0);
      break;

    case 'balanced':
    default:
      score =
        normalize(Math.log10(metrics.totalParams), 4, 12) * weights.params +
        normalize(metrics.peakMemoryGb, 0.1, 160) * weights.memory +
        normalize(Math.log10(metrics.latencyMs), -1, 3) * weights.latency +
        normalize(Math.log10(metrics.throughputTokensPerS), 0, 6, true) * weights.throughput +
        (metrics.fitsOnHardware ? weights.fit * 100 : 0);
      break;
  }

  return Math.max(0, Math.min(100, Math.round(score)));
}

// ─── Main Optimization Runner ────────────────────────────────────────

export function runOptimization(
  family: ArchitectureFamily,
  strategy: OptimizationStrategy,
  objective: OptimizationObjective,
  options?: {
    params?: HyperparameterDef[];
    searchSpace?: Partial<Record<keyof HardwareConfig, SearchRange>>;
    target?: OptimizationTarget;
    maxCandidates?: number;
    randomCount?: number;
  },
): OptimizationResult {
  const startTime = performance.now();
  const params = options?.params ?? getParamsForFamily(family);
  const maxCandidates = options?.maxCandidates ?? 100;
  const randomCount = options?.randomCount ?? 50;
  const searchSpace = options?.searchSpace;

  // Generate candidates
  let rawConfigs: Partial<HardwareConfig>[];

  switch (strategy) {
    case 'grid_search':
      rawConfigs = generateGridCandidates(family, params, searchSpace, maxCandidates);
      break;
    case 'random_search':
      rawConfigs = generateRandomCandidates(family, params, randomCount, searchSpace);
      break;
    case 'bayesian':
      // For the initial implementation, Bayesian = random + greedy local search
      rawConfigs = generateRandomCandidates(family, params, randomCount, searchSpace);
      // Refine top candidates with local perturbations
      rawConfigs = bayesianRefine(rawConfigs, family, params, objective, options?.target);
      break;
    default:
      rawConfigs = [];
  }

  // Evaluate all candidates
  const candidates: CandidateConfig[] = rawConfigs.map((config, idx) => {
    const estimated = estimateMetrics(config, family);
    return {
      id: `candidate-${idx}`,
      config,
      estimated,
    };
  });

  // Score and rank
  const target = options?.target ?? { objective };
  for (const c of candidates) {
    c.score = scoreConfig(c.estimated, objective, target);
  }

  candidates.sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
  candidates.forEach((c, i) => {
    c.rank = i + 1;
  });

  const elapsedMs = performance.now() - startTime;

  return {
    strategy,
    family,
    totalCandidates: candidates.length,
    evaluatedCount: candidates.length,
    candidates,
    bestConfig: candidates[0] ?? null,
    elapsedMs,
  };
}

/**
 * Bayesian-inspired refinement:
 * Take top configs, generate local perturbations around them,
 * evaluate, and keep the best.
 */
function bayesianRefine(
  configs: Partial<HardwareConfig>[],
  family: ArchitectureFamily,
  params: HyperparameterDef[],
  objective: OptimizationObjective,
  target?: OptimizationTarget,
): Partial<HardwareConfig>[] {
  const refined: Partial<HardwareConfig>[] = [...configs];

  // Score all, take top 20%
  const scored = configs.map((c) => ({
    config: c,
    score: scoreConfig(estimateMetrics(c, family), objective, target),
  }));
  scored.sort((a, b) => b.score - a.score);
  const topCount = Math.max(3, Math.ceil(scored.length * 0.2));

  // Generate perturbations around top configs
  for (let i = 0; i < topCount && i < scored.length; i++) {
    const base = scored[i].config;

    for (let p = 0; p < 3; p++) {
      // 3 perturbations per top config
      const perturbed = { ...base } as Record<string, unknown>;

      // Perturb 1-3 random params
      const numPerturb = Math.floor(Math.random() * 3) + 1;
      const shuffled = [...params].sort(() => Math.random() - 0.5);

      for (let k = 0; k < numPerturb && k < shuffled.length; k++) {
        const param = shuffled[k];
        const current = base[param.key] as number;
        if (param.type === 'int' || param.type === 'float') {
          const range = param.range!;
          const delta = (range.max - range.min) * 0.1 * (Math.random() - 0.5);
          perturbed[param.key as string] = Math.round((current + delta) * 100) / 100;
        } else if (param.type === 'categorical') {
          const options = param.options!;
          const idx = Math.floor(Math.random() * options.length);
          perturbed[param.key as string] = options[idx].value;
        }
      }

      const fixed = autoFixConfig(perturbed as Partial<HardwareConfig>, family);
      const validation = validateConfig(fixed, family);
      if (validation.valid) {
        refined.push(fixed);
      }
    }
  }

  return refined;
}

/** Get printable summary of optimization result */
export function formatOptimizationResult(result: OptimizationResult): string {
  if (!result.bestConfig || result.candidates.length === 0) {
    return 'No valid configurations found. Try widening the search ranges.';
  }

  const best = result.bestConfig;
  const e = best.estimated;
  const configSummary = Object.entries(best.config)
    .filter(([_, v]) => v !== undefined && v !== null && v !== 0)
    .map(([k, v]) => `${k}=${v}`)
    .join(', ');

  return [
    `▸ Strategy: ${result.strategy}`,
    `▸ Candidates evaluated: ${result.evaluatedCount}/${result.totalCandidates}`,
    `▸ Time: ${result.elapsedMs.toFixed(0)}ms`,
    '',
    `── Best Config (Score: ${best.score}/100) ──`,
    configSummary,
    '',
    `── Estimated Metrics ──`,
    `  Parameters: ${formatNumber(e.totalParams)}`,
    `  FLOPs: ${formatNumber(e.totalFlops)}`,
    `  Peak Memory: ${e.peakMemoryGb.toFixed(2)} GB`,
    `  Latency: ${e.latencyMs.toFixed(2)} ms`,
    `  Throughput: ${Math.round(e.throughputTokensPerS).toLocaleString()} tokens/s`,
    `  Fits on GPU: ${e.fitsOnHardware ? '✅' : '❌'}`,
    `  Bottleneck: ${e.bottleneck}`,
    e.trainingCostUsd > 0 ? `  Training Cost: $${e.trainingCostUsd.toFixed(2)}` : '',
    `  Memory Bandwidth: ${e.memoryBandwidthGbs} GB/s`,
  ]
    .filter(Boolean)
    .join('\n');
}

function formatNumber(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}
