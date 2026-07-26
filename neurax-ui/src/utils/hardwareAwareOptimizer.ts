/**
 * hardwareAwareOptimizer.ts
 *
 * Module d'optimisation "Hardware-Aware" pour NEURAX.
 *
 * Analyse les specs d'un GPU cible et recommande :
 *  - La taille max de modèle qui tient en mémoire
 *  - La précision optimale (fp32/fp16/int8) selon le HW
 *  - Les hyperparamètres idéaux pour saturer le GPU
 *  - Le batch size optimal (compute-bound vs memory-bound)
 */

import { ArchitectureFamily } from '@/types/plugins';
import { HardwareConfig } from '@/contexts/HardwareContext';
import { estimateMetrics, EstimatedMetrics, OptimizationObjective } from './hyperparameterOptimizer';

// ─── GPU Database (frontend-side) ────────────────────────────────────

export interface GpuSpecs {
  name: string;
  memoryGb: number;
  memoryBandwidthGbs: number;
  tflopsFp32: number;
  tflopsFp16: number;
  tflopsBf16: number;
  tflopsInt8: number;
  tensorCores: boolean;
  nvlink: boolean;
  nvlinkBandwidthGbs: number;
  tdpWatts: number;
  launchYear: number;
  l2CacheMb: number;
  numSms: number;
}

/**
 * GPU specifications database (mirrors neurax-hardware-db/src/gpu.rs)
 * Used client-side for quick hardware-aware recommendations.
 */
export const GPU_DATABASE: Record<string, GpuSpecs> = {
  'H100-SXM': {
    name: 'NVIDIA H100 SXM',
    memoryGb: 80,
    memoryBandwidthGbs: 3352,
    tflopsFp32: 67.0,
    tflopsFp16: 1979.0,
    tflopsBf16: 1979.0,
    tflopsInt8: 3958.0,
    tensorCores: true,
    nvlink: true,
    nvlinkBandwidthGbs: 900,
    tdpWatts: 700,
    launchYear: 2022,
    l2CacheMb: 50,
    numSms: 132,
  },
  'H100-PCIe': {
    name: 'NVIDIA H100 PCIe',
    memoryGb: 80,
    memoryBandwidthGbs: 2000,
    tflopsFp32: 51.0,
    tflopsFp16: 1513.0,
    tflopsBf16: 1513.0,
    tflopsInt8: 3026.0,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 350,
    launchYear: 2022,
    l2CacheMb: 50,
    numSms: 114,
  },
  'H200': {
    name: 'NVIDIA H200',
    memoryGb: 141,
    memoryBandwidthGbs: 4800,
    tflopsFp32: 67.0,
    tflopsFp16: 1979.0,
    tflopsBf16: 1979.0,
    tflopsInt8: 3958.0,
    tensorCores: true,
    nvlink: true,
    nvlinkBandwidthGbs: 900,
    tdpWatts: 700,
    launchYear: 2024,
    l2CacheMb: 50,
    numSms: 132,
  },
  'GH200': {
    name: 'NVIDIA GH200 Grace Hopper',
    memoryGb: 141,
    memoryBandwidthGbs: 4800,
    tflopsFp32: 67.0,
    tflopsFp16: 1979.0,
    tflopsBf16: 1979.0,
    tflopsInt8: 3958.0,
    tensorCores: true,
    nvlink: true,
    nvlinkBandwidthGbs: 900,
    tdpWatts: 700,
    launchYear: 2024,
    l2CacheMb: 50,
    numSms: 132,
  },
  'A100-SXM': {
    name: 'NVIDIA A100 SXM',
    memoryGb: 80,
    memoryBandwidthGbs: 2039,
    tflopsFp32: 19.5,
    tflopsFp16: 312.0,
    tflopsBf16: 312.0,
    tflopsInt8: 624.0,
    tensorCores: true,
    nvlink: true,
    nvlinkBandwidthGbs: 600,
    tdpWatts: 400,
    launchYear: 2020,
    l2CacheMb: 40,
    numSms: 108,
  },
  'A100-PCIe': {
    name: 'NVIDIA A100 PCIe',
    memoryGb: 40,
    memoryBandwidthGbs: 1555,
    tflopsFp32: 19.5,
    tflopsFp16: 312.0,
    tflopsBf16: 312.0,
    tflopsInt8: 624.0,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 250,
    launchYear: 2020,
    l2CacheMb: 40,
    numSms: 108,
  },
  'L40': {
    name: 'NVIDIA L40',
    memoryGb: 48,
    memoryBandwidthGbs: 864,
    tflopsFp32: 91.6,
    tflopsFp16: 181.0,
    tflopsBf16: 181.0,
    tflopsInt8: 724.0,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 300,
    launchYear: 2022,
    l2CacheMb: 96,
    numSms: 118,
  },
  'L40S': {
    name: 'NVIDIA L40S',
    memoryGb: 48,
    memoryBandwidthGbs: 864,
    tflopsFp32: 91.6,
    tflopsFp16: 362.0,
    tflopsBf16: 362.0,
    tflopsInt8: 1448.0,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 350,
    launchYear: 2023,
    l2CacheMb: 96,
    numSms: 118,
  },
  'V100': {
    name: 'NVIDIA V100',
    memoryGb: 32,
    memoryBandwidthGbs: 900,
    tflopsFp32: 15.7,
    tflopsFp16: 125.0,
    tflopsBf16: 0,
    tflopsInt8: 0,
    tensorCores: true,
    nvlink: true,
    nvlinkBandwidthGbs: 300,
    tdpWatts: 300,
    launchYear: 2017,
    l2CacheMb: 6,
    numSms: 80,
  },
  'RTX4090': {
    name: 'NVIDIA RTX 4090',
    memoryGb: 24,
    memoryBandwidthGbs: 1008,
    tflopsFp32: 82.6,
    tflopsFp16: 165.2,
    tflopsBf16: 165.2,
    tflopsInt8: 660.8,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 450,
    launchYear: 2022,
    l2CacheMb: 72,
    numSms: 128,
  },
  'RTX4080': {
    name: 'NVIDIA RTX 4080',
    memoryGb: 16,
    memoryBandwidthGbs: 717,
    tflopsFp32: 48.7,
    tflopsFp16: 97.4,
    tflopsBf16: 97.4,
    tflopsInt8: 389.0,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 320,
    launchYear: 2022,
    l2CacheMb: 64,
    numSms: 76,
  },
  'RTX3090': {
    name: 'NVIDIA RTX 3090',
    memoryGb: 24,
    memoryBandwidthGbs: 936,
    tflopsFp32: 35.6,
    tflopsFp16: 71.0,
    tflopsBf16: 71.0,
    tflopsInt8: 284.0,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 350,
    launchYear: 2020,
    l2CacheMb: 6,
    numSms: 82,
  },
  'T4': {
    name: 'NVIDIA T4',
    memoryGb: 16,
    memoryBandwidthGbs: 300,
    tflopsFp32: 8.1,
    tflopsFp16: 65.0,
    tflopsBf16: 0,
    tflopsInt8: 130.0,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 70,
    launchYear: 2018,
    l2CacheMb: 4,
    numSms: 40,
  },
  'A6000': {
    name: 'NVIDIA RTX A6000',
    memoryGb: 48,
    memoryBandwidthGbs: 768,
    tflopsFp32: 38.7,
    tflopsFp16: 77.4,
    tflopsBf16: 77.4,
    tflopsInt8: 309.6,
    tensorCores: true,
    nvlink: false,
    nvlinkBandwidthGbs: 0,
    tdpWatts: 300,
    launchYear: 2021,
    l2CacheMb: 6,
    numSms: 84,
  },
};

// ─── Hardware Capability Analysis ────────────────────────────────────

export interface HardwareCapability {
  /** Max model parameters that fit in memory (fp16) */
  maxParamsFp16: number;
  /** Max model parameters (int8 quantized) */
  maxParamsInt8: number;
  /** Max sequence length for given hidden dim */
  maxSeqLen: (hiddenDim: number, numLayers: number) => number;
  /** Recommended precision for this GPU */
  recommendedPrecision: string;
  /** Optimal batch size for compute saturation */
  optimalBatchSize: number;
  /** Arithmetic intensity threshold (FLOPs/byte) */
  ridgePoint: number;
  /** Efficiency factor based on launch year */
  efficiencyFactor: number;
  /** Recommended use case */
  bestFor: string;
  /** Memory bottleneck analysis */
  memoryAnalysis: {
    totalVramGb: number;
    osReservedGb: number;
    availableForModelGb: number;
    overheadFactor: number;
  };
}

/**
 * Analyze hardware capabilities for a given GPU.
 */
export function analyzeHardware(gpuName: string): HardwareCapability | null {
  const gpu = GPU_DATABASE[gpuName] ?? GPU_DATABASE['RTX4090'];
  if (!gpu) return null;

  // Reserve ~2GB for OS + framework overhead
  const osReserved = 2;
  const overheadFactor = 1.15; // 15% overhead for activations, buffers
  const availableGb = gpu.memoryGb - osReserved;

  // fp16 params: 2 bytes per param + overhead
  const maxParamsFp16 = Math.floor((availableGb * 1e9) / (2 * overheadFactor));
  const maxParamsInt8 = Math.floor((availableGb * 1e9) / (1 * overheadFactor));

  // Ridge point: FP32 TFLOPS / BW
  const ridgePoint = (gpu.tflopsFp16 * 1e12) / (gpu.memoryBandwidthGbs * 1e9);

  // Efficiency factor (newer GPUs have higher utilization)
  const efficiency = Math.min(0.65, 0.4 + (gpu.launchYear - 2017) * 0.03);

  // Recommended precision based on HW capabilities
  let recommendedPrecision = 'fp32';
  if (gpu.tflopsFp16 > 0) recommendedPrecision = gpu.tflopsBf16 > 0 ? 'bf16' : 'fp16';
  if (gpu.tflopsInt8 > 0 && gpu.memoryGb <= 24) recommendedPrecision = 'int8';

  // Optimal batch size (rule of thumb)
  const optimalBatchSize = Math.max(1, Math.min(512, Math.round(gpu.memoryGb * 2)));

  // Best use description
  let bestFor = 'General-purpose training and inference';
  if (gpu.memoryGb >= 80 && gpu.nvlink) {
    bestFor = 'Large-scale foundation model training';
  } else if (gpu.memoryGb >= 48) {
    bestFor = 'Large model fine-tuning and inference';
  } else if (gpu.memoryGb >= 24) {
    bestFor = 'Production inference and small-to-medium training';
  } else {
    bestFor = 'Edge inference, prototyping, and small models';
  }

  // Max sequence length estimator
  const maxSeqLen = (hiddenDim: number, numLayers: number): number => {
    const bytesPerToken = hiddenDim * numLayers * 2 * 2; // fp16 * 2 for KV cache
    const maxTokens = Math.floor((availableGb * 1e9) / bytesPerToken);
    return Math.max(128, Math.min(262144, maxTokens));
  };

  return {
    maxParamsFp16,
    maxParamsInt8,
    maxSeqLen,
    recommendedPrecision,
    optimalBatchSize,
    ridgePoint,
    efficiencyFactor: efficiency,
    bestFor,
    memoryAnalysis: {
      totalVramGb: gpu.memoryGb,
      osReservedGb: osReserved,
      availableForModelGb: availableGb,
      overheadFactor,
    },
  };
}

// ─── Hardware-Aware Recommendations ──────────────────────────────────

export interface HardwareRecommendation {
  gpu: string;
  capability: HardwareCapability;

  /** Recommended hyperparameters */
  recommended: {
    precision: string;
    batchSize: number;
    hiddenDim: number;
    numLayers: number;
    seqLen: number;
    ffnDim: number;
  };

  /** Analysis of what fits */
  fitAnalysis: {
    estimatedParams: number;
    estimatedMemoryGb: number;
    fits: boolean;
    memoryHeadroomGb: number;
    utilizationPct: number;
  };

  /** Bottleneck analysis */
  bottleneck: string;

  /** Strategy for this HW */
  strategy: string;
}

/**
 * Generate hardware-aware recommendations for a model family.
 */
export function recommendForHardware(
  family: ArchitectureFamily,
  targetGpu: string,
  _objective: OptimizationObjective = 'balanced',
): HardwareRecommendation[] {
  // Get all matching GPUs (or just the one specified)
  const gpuNames = targetGpu
    ? [targetGpu]
    : Object.keys(GPU_DATABASE).filter((g) =>
        g.toLowerCase().includes(targetGpu?.toLowerCase() ?? ''),
      );

  if (gpuNames.length === 0) return [];

  const recommendations: HardwareRecommendation[] = [];

  for (const gpuName of gpuNames) {
    const cap = analyzeHardware(gpuName);
    if (!cap) continue;

    // Estimate model capacity
    const paramsBudget = cap.maxParamsFp16;
    const hiddenDim = familyToRecommendedDim(family, paramsBudget);
    const numLayers = familyToRecommendedLayers(family, paramsBudget, hiddenDim);
    const seqLen = Math.min(8192, cap.maxSeqLen(hiddenDim, numLayers));
    const ffnDim = hiddenDim * 4;
    const batchSize = cap.optimalBatchSize;

    // Build recommended config
    const config: Partial<HardwareConfig> = {
      hardware: gpuName,
      precision: cap.recommendedPrecision as HardwareConfig['precision'],
      batchSize,
      hiddenDim,
      numLayers,
      ffnDim,
      seqLen,
    };

    if (family === 'transformer' || family === 'moe') {
      config.numHeads = Math.max(4, Math.round(hiddenDim / 64));
      config.vocabSize = 32000;
    }
    if (family === 'moe') {
      config.numExperts = Math.max(2, Math.round(Math.sqrt(paramsBudget / 1e6)));
      config.topK = 2;
    }
    if (family === 'ssm') {
      config.dState = 16;
      config.expandFactor = 2;
    }
    if (family === 'cnn') {
      config.imgHeight = 224;
      config.imgWidth = 224;
      config.inChannels = 3;
    }
    if (family === 'diffusion') {
      config.numDenoisingSteps = 50;
      config.modelChannels = Math.min(256, Math.round(hiddenDim / 2));
    }

    // Estimate metrics for this config
    const metrics = estimateMetrics(config, family);

    const fitPct = (metrics.peakMemoryGb / cap.memoryAnalysis.availableForModelGb) * 100;

    // Determine bottleneck
    let bottleneck: string;
    if (!metrics.fitsOnHardware) {
      bottleneck = `OUT OF MEMORY (${metrics.peakMemoryGb.toFixed(1)} GB needed, ${cap.memoryAnalysis.availableForModelGb.toFixed(1)} GB available)`;
    } else if (metrics.arithmeticIntensity < cap.ridgePoint / 10) {
      bottleneck = 'Memory-bound — increase batch size or sequence length';
    } else if (metrics.arithmeticIntensity > cap.ridgePoint * 10) {
      bottleneck = 'Compute-bound — increase model size to saturate GPU';
    } else {
      bottleneck = 'Balanced — good utilization of GPU resources';
    }

    recommendations.push({
      gpu: gpuName,
      capability: cap,
      recommended: {
        precision: cap.recommendedPrecision,
        batchSize,
        hiddenDim,
        numLayers,
        seqLen,
        ffnDim,
      },
      fitAnalysis: {
        estimatedParams: metrics.totalParams,
        estimatedMemoryGb: metrics.peakMemoryGb,
        fits: metrics.fitsOnHardware,
        memoryHeadroomGb: cap.memoryAnalysis.availableForModelGb - metrics.peakMemoryGb,
        utilizationPct: Math.round(fitPct),
      },
      bottleneck,
      strategy: deriveStrategy(family, metrics, cap),
    });
  }

  return recommendations;
}

/**
 * Find the best GPU for a given model config.
 */
export function findBestGpu(
  family: ArchitectureFamily,
  config: Partial<HardwareConfig>,
): { gpu: string; score: number }[] {
  const metrics = estimateMetrics(config, family);
  const scored: { gpu: string; score: number }[] = [];

  for (const [gpuName, gpu] of Object.entries(GPU_DATABASE)) {
    const availableGb = gpu.memoryGb - 2;
    const fits = metrics.peakMemoryGb * 1.2 <= availableGb;

    // Score: prefer best fit, then cheapest, then fastest
    const fitScore = fits ? 40 : 0;
    const memScore = Math.max(0, 30 - Math.abs(metrics.peakMemoryGb - availableGb * 0.6) * 2);
    const bwScore = (gpu.memoryBandwidthGbs / 4800) * 15;
    const tfScore = (gpu.tflopsFp16 / 1979) * 15;

    scored.push({
      gpu: gpuName,
      score: fitScore + memScore + bwScore + tfScore,
    });
  }

  return scored.sort((a, b) => b.score - a.score).slice(0, 5);
}

// ─── Helper Functions ────────────────────────────────────────────────

function familyToRecommendedDim(family: ArchitectureFamily, paramsBudget: number): number {
  // Estimate hidden dim from param budget
  const budgetB = paramsBudget / 1e9;

  switch (family) {
    case 'transformer':
      // ~12 * d^2 for a 12-layer transformer (rough)
      return Math.min(16384, Math.max(64, Math.round(Math.sqrt(budgetB * 1e9 / 12 / 4) / 64) * 64));
    case 'moe':
      return Math.min(8192, Math.max(64, Math.round(Math.sqrt(budgetB * 1e9 / 8 / 8) / 64) * 64));
    case 'ssm':
      return Math.min(8192, Math.max(64, Math.round(Math.sqrt(budgetB * 1e9 / 24) / 64) * 64));
    case 'cnn':
      return Math.min(512, Math.max(16, Math.round(budgetB * 10) / 16) * 16);
    case 'diffusion':
      return Math.min(256, Math.max(32, Math.round(budgetB * 5) / 32) * 32);
    case 'gnn':
      return Math.min(256, Math.max(8, Math.round(budgetB * 20) / 8) * 8);
    case 'rnn':
      return Math.min(2048, Math.max(32, Math.round(Math.sqrt(budgetB * 1e9 / 4) / 32) * 32));
    default:
      return 768;
  }
}

function familyToRecommendedLayers(
  family: ArchitectureFamily,
  paramsBudget: number,
  hiddenDim: number,
): number {
  if (hiddenDim <= 0) return 6;

  const paramsPerLayer = 12 * hiddenDim * hiddenDim; // rough estimate
  const maxLayers = Math.floor(paramsBudget / paramsPerLayer);

  switch (family) {
    case 'transformer':
      return Math.min(256, Math.max(1, maxLayers));
    case 'moe':
      return Math.min(128, Math.max(1, Math.floor(maxLayers / 2)));
    case 'ssm':
      return Math.min(128, Math.max(1, maxLayers));
    case 'cnn':
      return Math.min(500, Math.max(1, Math.round(budgetScale(paramsBudget, 1e6, 5e8, 10, 200))));
    case 'gnn':
      return Math.min(100, Math.max(1, Math.round(budgetScale(paramsBudget, 1e5, 1e8, 2, 20))));
    case 'rnn':
      return Math.min(50, Math.max(1, Math.round(budgetScale(paramsBudget, 1e6, 1e9, 1, 20))));
    default:
      return Math.min(48, Math.max(1, maxLayers));
  }
}

function budgetScale(
  budget: number,
  minBudget: number,
  maxBudget: number,
  minVal: number,
  maxVal: number,
): number {
  if (budget <= minBudget) return minVal;
  if (budget >= maxBudget) return maxVal;
  const ratio = (budget - minBudget) / (maxBudget - minBudget);
  return minVal + ratio * (maxVal - minVal);
}

function deriveStrategy(
  family: ArchitectureFamily,
  metrics: EstimatedMetrics,
  cap: HardwareCapability,
): string {
  const utilPct = (metrics.peakMemoryGb / cap.memoryAnalysis.availableForModelGb) * 100;

  if (utilPct > 90) {
    return 'Reduce model size (hiddenDim or numLayers) or use quantization (INT8/INT4)';
  }
  if (utilPct < 30) {
    return 'GPU is underutilized — consider increasing model size or batch size significantly';
  }
  if (metrics.bottleneck === 'memory') {
    return 'Memory-bound — increase batch size, sequence length, or use Flash Attention';
  }
  if (metrics.bottleneck === 'compute') {
    return 'Compute-bound — model is large enough to saturate the GPU, good utilization';
  }

  return `Good fit for ${family} models — ${Math.round(utilPct)}% VRAM utilization`;
}

/** Get all available GPUs for hardware selection */
export function getAllGpus(): string[] {
  return Object.keys(GPU_DATABASE);
}

/** Get GPU display name */
export function getGpuDisplayName(gpuName: string): string {
  return GPU_DATABASE[gpuName]?.name ?? gpuName;
}
