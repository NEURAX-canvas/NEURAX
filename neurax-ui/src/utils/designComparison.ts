/**
 * Holding two designs side by side.
 *
 * Design work is comparative. Nobody asks "is 47 GB of VRAM good"; they ask
 * "is 32 layers at width 4096 better than 48 at 3072, for what it costs". Until
 * now the only way to answer that in NEURAX was to analyse one, write the
 * numbers down, edit the design, analyse again, and compare by memory — which
 * is both tedious and how people end up comparing a number from one precision
 * against a number from another without noticing.
 *
 * This module is deliberately only arithmetic over two `AnalysisResult`s. It
 * computes no metrics of its own: every figure shown in a comparison is one the
 * compiler produced, so a comparison can never disagree with the analysis it is
 * built from.
 */

import { AnalysisResult } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { HardwareConfig } from '@/contexts/HardwareContext.tsx';

/** One design, captured at a moment, with the analysis that described it. */
export interface DesignVariant {
  id: string;
  name: string;
  capturedAt: string;
  architecture: ArchitectureFamily;
  blockCount: number;
  connectionCount: number;
  analysis: AnalysisResult;
  /** Kept so a comparison can say the two were measured under the same terms. */
  hardware: Partial<HardwareConfig>;
}

/** How a metric should read, and which direction is an improvement. */
type Unit = 'count' | 'bytes' | 'flops' | 'ms' | 'usd' | 'hours' | 'kwh' | 'kg' | 'ratio' | 'percent';

/** Whether a smaller number is better, a larger one, or neither. */
type Better = 'lower' | 'higher' | 'neutral';

interface MetricSpec {
  key: string;
  label: string;
  group: string;
  unit: Unit;
  better: Better;
  read: (analysis: AnalysisResult) => number | null | undefined;
}

/**
 * The metrics a comparison shows.
 *
 * A deliberately short list out of the sixty-six the compiler produces. A
 * comparison that shows everything shows nothing — these are the figures that
 * decide whether a design is viable: does it fit, what does it cost, how fast
 * is it, and what does it emit.
 */
const METRICS: MetricSpec[] = [
  { key: 'totalParams', label: 'Parameters', group: 'Model', unit: 'count', better: 'neutral', read: (a) => a.totalParams },
  { key: 'numLayers', label: 'Layers', group: 'Model', unit: 'count', better: 'neutral', read: (a) => a.numLayers },

  { key: 'peakVramBytes', label: 'Peak VRAM', group: 'Memory', unit: 'bytes', better: 'lower', read: (a) => a.peakVramBytes },
  { key: 'parameterMemoryBytes', label: 'Weights', group: 'Memory', unit: 'bytes', better: 'lower', read: (a) => a.parameterMemoryBytes },
  { key: 'activationMemoryBytes', label: 'Activations', group: 'Memory', unit: 'bytes', better: 'lower', read: (a) => a.activationMemoryBytes },
  { key: 'maxBatchSizeFit', label: 'Max batch that fits', group: 'Memory', unit: 'count', better: 'higher', read: (a) => a.maxBatchSizeFit },

  { key: 'totalFlops', label: 'Total FLOPs', group: 'Compute', unit: 'flops', better: 'lower', read: (a) => a.totalFlops },
  { key: 'flopsPerToken', label: 'FLOPs per token', group: 'Compute', unit: 'flops', better: 'lower', read: (a) => a.flopsPerToken },
  { key: 'arithmeticIntensity', label: 'Arithmetic intensity', group: 'Compute', unit: 'ratio', better: 'higher', read: (a) => a.arithmeticIntensity },

  { key: 'latencyMs', label: 'Latency', group: 'Performance', unit: 'ms', better: 'lower', read: (a) => a.latencyMs },
  { key: 'throughputTokensPerS', label: 'Throughput', group: 'Performance', unit: 'count', better: 'higher', read: (a) => a.throughputTokensPerS },
  { key: 'gpuUtilization', label: 'GPU utilisation', group: 'Performance', unit: 'percent', better: 'higher', read: (a) => a.gpuUtilization },

  { key: 'trainingCostUsd', label: 'Training cost', group: 'Cost', unit: 'usd', better: 'lower', read: (a) => a.trainingCostUsd },
  { key: 'trainingTimeHours', label: 'Training time', group: 'Cost', unit: 'hours', better: 'lower', read: (a) => a.trainingTimeHours },
  { key: 'costPerMillionTokensUsd', label: 'Cost per M tokens', group: 'Cost', unit: 'usd', better: 'lower', read: (a) => a.costPerMillionTokensUsd },
  { key: 'energyKwh', label: 'Energy', group: 'Cost', unit: 'kwh', better: 'lower', read: (a) => a.energyKwh },
  { key: 'co2Kg', label: 'CO₂', group: 'Cost', unit: 'kg', better: 'lower', read: (a) => a.co2Kg },
];

/** One metric, on both sides. */
export interface MetricComparison {
  key: string;
  label: string;
  group: string;
  unit: Unit;
  baseline: number | null;
  candidate: number | null;
  /** `candidate - baseline`, or null when either side is missing. */
  delta: number | null;
  /** Relative change as a fraction; null when the baseline is zero or missing. */
  ratio: number | null;
  /**
   * Whether the change is an improvement. `neutral` covers both metrics with no
   * preferred direction (parameter count) and changes too small to call.
   */
  verdict: 'better' | 'worse' | 'neutral';
}

export interface ComparisonReport {
  baseline: DesignVariant;
  candidate: DesignVariant;
  metrics: MetricComparison[];
  /**
   * Differences in the terms under which the two were analysed. A comparison
   * across different hardware or precision is not wrong, but it is answering a
   * different question, and reading it as an architecture comparison is a
   * mistake worth interrupting.
   */
  incomparable: string[];
}

/**
 * A change smaller than this is noise, not a result.
 *
 * The analytical pipeline is deterministic, so an identical design gives an
 * identical number — but two designs differing in a rounding-level way should
 * not be presented as one beating the other.
 */
const SIGNIFICANT = 0.005; // 0.5 %

function compareMetric(spec: MetricSpec, baseline: AnalysisResult, candidate: AnalysisResult): MetricComparison {
  const a = spec.read(baseline);
  const b = spec.read(candidate);

  const left = typeof a === 'number' && Number.isFinite(a) ? a : null;
  const right = typeof b === 'number' && Number.isFinite(b) ? b : null;

  const delta = left !== null && right !== null ? right - left : null;
  const ratio = left !== null && right !== null && left !== 0 ? (right - left) / Math.abs(left) : null;

  let verdict: MetricComparison['verdict'] = 'neutral';
  if (spec.better !== 'neutral' && delta !== null && ratio !== null && Math.abs(ratio) >= SIGNIFICANT) {
    const improved = spec.better === 'lower' ? delta < 0 : delta > 0;
    verdict = improved ? 'better' : 'worse';
  }

  return {
    key: spec.key,
    label: spec.label,
    group: spec.group,
    unit: spec.unit,
    baseline: left,
    candidate: right,
    delta,
    ratio,
    verdict,
  };
}

/** Settings that must match for a comparison to be about the architecture. */
const SHARED_TERMS: Array<{ key: keyof HardwareConfig; label: string }> = [
  { key: 'hardware', label: 'GPU' },
  { key: 'gpuCount', label: 'GPU count' },
  { key: 'precision', label: 'Precision' },
  { key: 'batchSize', label: 'Batch size' },
  { key: 'seqLen', label: 'Sequence length' },
];

function findIncomparable(baseline: DesignVariant, candidate: DesignVariant): string[] {
  const notes: string[] = [];

  for (const { key, label } of SHARED_TERMS) {
    const a = baseline.hardware[key];
    const b = candidate.hardware[key];
    if (a === undefined || b === undefined) continue;
    if (a !== b) notes.push(`${label}: ${String(a)} → ${String(b)}`);
  }

  if (baseline.architecture !== candidate.architecture) {
    notes.push(`Architecture family: ${baseline.architecture} → ${candidate.architecture}`);
  }

  return notes;
}

/** Compare two captured designs. */
export function compareDesigns(baseline: DesignVariant, candidate: DesignVariant): ComparisonReport {
  return {
    baseline,
    candidate,
    metrics: METRICS.map((spec) => compareMetric(spec, baseline.analysis, candidate.analysis)),
    incomparable: findIncomparable(baseline, candidate),
  };
}

/** The groups present in a report, in the order the metrics declare them. */
export function comparisonGroups(report: ComparisonReport): string[] {
  const seen: string[] = [];
  for (const metric of report.metrics) {
    if (!seen.includes(metric.group)) seen.push(metric.group);
  }
  return seen;
}

// ── Formatting ──────────────────────────────────────────────────────────────

function humanCount(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e12) return `${(value / 1e12).toFixed(2)} T`;
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)} B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(2)} M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(2)} K`;
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
}

function humanBytes(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1024 ** 4) return `${(value / 1024 ** 4).toFixed(2)} TB`;
  if (abs >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  if (abs >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  if (abs >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value.toFixed(0)} B`;
}

function humanFlops(value: number): string {
  const abs = Math.abs(value);
  if (abs >= 1e18) return `${(value / 1e18).toFixed(2)} EFLOP`;
  if (abs >= 1e15) return `${(value / 1e15).toFixed(2)} PFLOP`;
  if (abs >= 1e12) return `${(value / 1e12).toFixed(2)} TFLOP`;
  if (abs >= 1e9) return `${(value / 1e9).toFixed(2)} GFLOP`;
  return humanCount(value);
}

/** Render a metric value in its unit. `null` becomes an em dash, not a zero. */
export function formatMetric(value: number | null, unit: Unit): string {
  if (value === null) return '—';

  switch (unit) {
    case 'bytes':
      return humanBytes(value);
    case 'flops':
      return humanFlops(value);
    case 'ms':
      return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(2)} ms`;
    case 'usd':
      return value >= 1000
        ? `$${humanCount(value)}`
        : `$${value.toFixed(value < 1 ? 4 : 2)}`;
    case 'hours':
      return value >= 24 ? `${(value / 24).toFixed(1)} d` : `${value.toFixed(1)} h`;
    case 'kwh':
      return `${humanCount(value)} kWh`;
    case 'kg':
      return value >= 1000 ? `${(value / 1000).toFixed(2)} t` : `${value.toFixed(1)} kg`;
    case 'percent':
      // The compiler reports utilisation as a fraction in some fields and a
      // percentage in others; anything at or below 1 is read as a fraction.
      return `${(value <= 1 ? value * 100 : value).toFixed(1)} %`;
    case 'ratio':
      return value.toFixed(2);
    case 'count':
    default:
      return humanCount(value);
  }
}

/** Render the change as a signed percentage, or an em dash when there is none. */
export function formatDelta(metric: MetricComparison): string {
  if (metric.ratio === null) {
    // A metric that went from nothing to something has no percentage, but the
    // change is still worth stating.
    if (metric.baseline === null && metric.candidate !== null) return 'new';
    if (metric.baseline !== null && metric.candidate === null) return 'gone';

    // Both sides are present but the baseline was zero, so there is no
    // percentage to give. An em dash here would read as "no change", when in
    // fact something went from nothing to a real figure — a cost of $0 rising
    // to $500 is the most significant change on the table, not the least.
    if (metric.delta !== null && metric.delta !== 0) {
      return `${metric.delta > 0 ? '+' : '−'}${formatMetric(Math.abs(metric.delta), metric.unit)}`;
    }
    return '—';
  }

  if (Math.abs(metric.ratio) < SIGNIFICANT) return '=';

  const pct = metric.ratio * 100;
  const sign = pct > 0 ? '+' : '';
  return `${sign}${Math.abs(pct) >= 100 ? pct.toFixed(0) : pct.toFixed(1)} %`;
}
