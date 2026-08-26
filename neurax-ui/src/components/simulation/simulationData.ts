import { AnalysisResult, CanvasNode, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';

const BYTES_IN_MB = 1024 ** 2;

/**
 * Categorical palette for the analysis charts.
 *
 * Validated with the palette checker in both light and dark mode: every step
 * sits inside the lightness band, clears the chroma floor (so none reads as
 * gray), holds at least 3:1 against the chart surface, and keeps adjacent pairs
 * separable under protanopia, deuteranopia and tritanopia.
 *
 * The previous values failed three of those checks — cyan was too light, slate
 * read as gray, and five of the nine fell below 3:1 contrast.
 *
 * `CATEGORICAL_ORDER` is the assignment order for series. Use it in sequence and
 * never cycle it: a tenth series folds into "Other" or becomes a small multiple,
 * because a generated tenth hue cannot be checked for separation.
 */
export const SIMULATION_COLORS = {
  blue: '#0284c7',
  amber: '#d97706',
  violet: '#7c3aed',
  green: '#059669',
  red: '#dc2626',
  cyan: '#0891b2',
  pink: '#db2777',
  indigo: '#4f46e5',
  teal: '#0d9488',
} as const;

/** Fixed assignment order — adjacent pairs are the ones the checker validated. */
export const CATEGORICAL_ORDER = [
  SIMULATION_COLORS.blue,
  SIMULATION_COLORS.amber,
  SIMULATION_COLORS.violet,
  SIMULATION_COLORS.green,
  SIMULATION_COLORS.red,
  SIMULATION_COLORS.cyan,
  SIMULATION_COLORS.pink,
  SIMULATION_COLORS.indigo,
  SIMULATION_COLORS.teal,
] as const;

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function formatBytes(bytes: number, digits = 1): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : digits)} ${units[unit]}`;
}

export function formatCompactNumber(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '0';
  const abs = Math.abs(value);
  // E/P tiers matter here specifically for FLOPs — a 175B-parameter model's
  // training compute is already petascale, and this is the formatter charts
  // reach for on FLOPs axes. Without them, a value like 2.4e15 rendered as
  // "2400000.0T", not fixed by any amount of digit-rounding.
  if (abs >= 1e18) return `${(value / 1e18).toFixed(digits)}E`;
  if (abs >= 1e15) return `${(value / 1e15).toFixed(digits)}P`;
  if (abs >= 1e12) return `${(value / 1e12).toFixed(digits)}T`;
  if (abs >= 1e9) return `${(value / 1e9).toFixed(digits)}B`;
  if (abs >= 1e6) return `${(value / 1e6).toFixed(digits)}M`;
  if (abs >= 1e3) return `${(value / 1e3).toFixed(digits)}K`;
  return `${value.toFixed(abs >= 10 ? 0 : digits)}`;
}

export function formatPercent(ratio: number, digits = 0): string {
  return `${(ratio * 100).toFixed(digits)}%`;
}

export function hasAnalysisReportData(analysis?: AnalysisResult | null): analysis is AnalysisResult {
  if (!analysis) return false;

  return (
    analysis.totalParams > 0
    || analysis.numLayers > 0
    || analysis.totalOperations > 0
    || analysis.totalTensorCount > 0
    || analysis.peakVramBytes > 0
    || analysis.latencyMs !== null
    || analysis.diagnosticCount !== undefined && analysis.diagnosticCount > 0
    || (analysis.diagnostics?.length ?? 0) > 0
    || (analysis.reportWarnings?.length ?? 0) > 0
    || Object.keys(analysis.opsDistribution ?? {}).length > 0
  );
}

export function hasCompilationProgress(analysis?: AnalysisResult | null): boolean {
  if (!analysis?.compilation) return false;
  const { current_phase, total_progress, phase_timeline } = analysis.compilation;
  return Boolean(
    (typeof current_phase === 'string' && current_phase.trim().length > 0)
    || typeof total_progress === 'number'
    || (phase_timeline?.length ?? 0) > 0,
  );
}

export function hasPhaseTimeline(analysis?: AnalysisResult | null): boolean {
  return (analysis?.compilation?.phase_timeline?.length ?? 0) > 0;
}

export function hasLivePartialMetrics(analysis?: AnalysisResult | null): boolean {
  return (analysis?.live_trace?.partial_metrics?.length ?? 0) > 0;
}

export function hasLiveThroughputTrace(analysis?: AnalysisResult | null): boolean {
  return (analysis?.live_trace?.throughput_trace?.length ?? 0) > 0;
}

export function hasMemoryHeatmap(analysis?: AnalysisResult | null): boolean {
  return (analysis?.memory_heatmap?.length ?? 0) > 0 || (analysis?.live_trace?.memory_heatmap?.length ?? 0) > 0;
}

export function hasMemoryLiveness(analysis?: AnalysisResult | null): boolean {
  return (analysis?.memory_liveness?.length ?? 0) > 0 || (analysis?.live_trace?.memory_liveness?.length ?? 0) > 0;
}

export function hasGradientMemoryBreakdown(analysis?: AnalysisResult | null): boolean {
  return (analysis?.gradient_memory_breakdown?.length ?? 0) > 0 || (analysis?.live_trace?.gradient_memory_breakdown?.length ?? 0) > 0;
}

export function hasKvCacheScaling(analysis?: AnalysisResult | null): boolean {
  return (analysis?.kv_cache_scaling?.length ?? 0) > 0 || (analysis?.live_trace?.kv_cache_scaling?.length ?? 0) > 0;
}

export function hasPerLayerRows(perLayer?: ArrayLike<unknown> | null): boolean {
  return (perLayer?.length ?? 0) > 0;
}

export function hasPerLayerLatencyMap(analysis?: AnalysisResult | null): boolean {
  return Object.keys(analysis?.perLayerLatency ?? {}).length > 0;
}

export function hasPerLayerVramMap(analysis?: AnalysisResult | null): boolean {
  return Object.keys(analysis?.perLayerVram ?? {}).length > 0;
}

export function normalizeSeverity(value?: string): 'critical' | 'warning' | 'info' | 'hint' {
  const normalized = (value ?? '').toLowerCase();
  if (normalized === 'critical' || normalized === 'error') return 'critical';
  if (normalized === 'warning') return 'warning';
  if (normalized === 'hint') return 'hint';
  return 'info';
}

export function severityColor(severity: string): string {
  switch (normalizeSeverity(severity)) {
    case 'critical':
      return SIMULATION_COLORS.red;
    case 'warning':
      return SIMULATION_COLORS.amber;
    case 'hint':
      return SIMULATION_COLORS.teal;
    default:
      return SIMULATION_COLORS.blue;
  }
}

export function severityWeight(severity: string): number {
  switch (normalizeSeverity(severity)) {
    case 'critical':
      return 3;
    case 'warning':
      return 2;
    case 'hint':
      return 1;
    default:
      return 1;
  }
}

export function normalizePhaseStatus(status?: string): 'completed' | 'inprogress' | 'pending' | 'failed' {
  const normalized = (status ?? '').toLowerCase();
  if (normalized.includes('complete')) return 'completed';
  if (normalized.includes('progress') || normalized.includes('running')) return 'inprogress';
  if (normalized.includes('fail') || normalized.includes('error')) return 'failed';
  return 'pending';
}

export function parseFlopsValue(value: string | number | undefined): number {
  if (value === undefined || value === null) return 0;
  if (typeof value === 'number') return value;
  const match = value.trim().match(/^(-?\d+(\.\d+)?)\s*([A-Za-z]+)?$/);
  if (!match) return 0;
  let numeric = parseFloat(match[1]);
  const unit = match[3]?.toUpperCase() ?? '';
  if (unit.startsWith('TFLOP') || unit === 'T') numeric *= 1e12;
  else if (unit.startsWith('GFLOP') || unit === 'G') numeric *= 1e9;
  else if (unit.startsWith('MFLOP') || unit === 'M') numeric *= 1e6;
  else if (unit === 'K') numeric *= 1e3;
  return numeric;
}

export interface DerivedLayerMetric {
  key: string;
  name: string;
  kind: string;
  flops: number;
  params: number;
  share: number;
  latencyMs: number;
  weightsMb: number;
  activationsMb: number;
  gradientsMb: number;
  optimizerMb: number;
  forwardMb: number;
  backwardMb: number;
  memoryMb: number;
}

function classifyLayerKind(name: string): string {
  const value = name.toLowerCase();
  if (value.includes('attn') || value.includes('attention')) return 'attention';
  if (value.includes('ffn') || value.includes('mlp') || value.includes('expert')) return 'ffn';
  if (value.includes('embed')) return 'embedding';
  if (value.includes('norm')) return 'norm';
  if (value.includes('conv') || value.includes('unet')) return 'conv';
  if (value.includes('head') || value.includes('classifier')) return 'head';
  return 'other';
}

export function buildDerivedLayerMetrics(
  analysis: AnalysisResult,
  perLayer: PerLayerBreakdownRow[] = [],
): DerivedLayerMetric[] {
  const rawRows = perLayer.length > 0
    ? perLayer
    : Object.entries(analysis.opsDistribution ?? {})
      .map(([name, count]) => ({
        id: name,
        name,
        params: Math.round((analysis.totalParams || 0) * (count / Math.max(1, Object.values(analysis.opsDistribution).reduce((sum, value) => sum + value, 0)))),
        flops: ((analysis.totalFlops || 0) * (count / Math.max(1, Object.values(analysis.opsDistribution).reduce((sum, value) => sum + value, 0)))).toString(),
      }));

  const totalFlops = rawRows.reduce((sum, row) => sum + parseFlopsValue(row.flops), 0) || 1;
  const totalParams = rawRows.reduce((sum, row) => sum + (row.params ?? 0), 0) || 1;
  const parameterMb = analysis.parameterMemoryBytes / BYTES_IN_MB;
  const activationMb = analysis.activationMemoryBytes / BYTES_IN_MB;
  const gradientMb = analysis.gradientMemoryBytes / BYTES_IN_MB;
  const optimizerMb = analysis.optimizerStateBytes / BYTES_IN_MB;
  const gradientBreakdown = new Map(
    (analysis.gradient_memory_breakdown ?? analysis.live_trace?.gradient_memory_breakdown ?? []).map((entry) => [
      entry.name.toLowerCase(),
      {
        forwardMb: entry.forward / BYTES_IN_MB,
        backwardMb: entry.backward / BYTES_IN_MB,
      },
    ]),
  );

  return rawRows
    .map((row, index) => {
      const name = row.name ?? row.id ?? `Layer ${index + 1}`;
      const key = row.id ?? name;
      const flops = parseFlopsValue(row.flops);
      const params = row.params ?? 0;
      const flopsShare = flops / totalFlops;
      const paramShare = params / totalParams;
      const grad = gradientBreakdown.get(name.toLowerCase()) ?? gradientBreakdown.get(key.toLowerCase());
      const weights = Math.max(parameterMb * paramShare, (params * 2) / BYTES_IN_MB);
      const activations = Math.max(activationMb * flopsShare, 0);
      const gradients = Math.max(gradientMb * paramShare, 0);
      const optimizer = Math.max(optimizerMb * paramShare, 0);
      const forwardMb = grad?.forwardMb ?? (weights * 0.35 + activations * 0.65);
      const backwardMb = grad?.backwardMb ?? (gradients * 0.7 + optimizer * 0.6);
      const memoryMb = weights + activations + gradients + optimizer;

      return {
        key,
        name,
        kind: classifyLayerKind(name),
        // Raw FLOPs, not pre-divided to a fixed "G" unit — the one caller
        // of this (OptimizationCharts' Bottleneck Pareto) formats with
        // formatCompactNumber, which scales itself; pre-dividing here is
        // what produced axis ticks like "2400000G" for a GPT-scale layer.
        flops,
        params,
        share: flopsShare,
        latencyMs: (analysis.latencyMs ?? 0) * flopsShare,
        weightsMb: weights,
        activationsMb: activations,
        gradientsMb: gradients,
        optimizerMb: optimizer,
        forwardMb,
        backwardMb,
        memoryMb,
      };
    })
    .sort((a, b) => b.flops - a.flops)
    .slice(0, 10);
}

// `deriveOptimizationOpportunities` and `deriveFusionCandidates` used to
// live here — both scored their entries with UI-invented numbers (a
// priority-to-percentage lookup table, and pattern-matched fusion
// "difficulty" dressed up next to a score) rather than anything the
// compiler measured. Removed along with the Optimization tab cards that
// rendered them: the only real content in either — the recommendation
// text and priority label, or the fusion pattern name — isn't chart-shaped
// data, so it doesn't belong in a charts-only Simulation tab.

function bucketCategory(category?: string, code?: string): 'shape' | 'memory' | 'parallel' | 'op' | 'config' | 'general' {
  const value = `${category ?? ''} ${code ?? ''}`.toLowerCase();
  if (value.includes('shape') || value.includes('e002') || value.includes('w002')) return 'shape';
  if (value.includes('memory') || value.includes('e001') || value.includes('w005')) return 'memory';
  if (value.includes('parallel') || value.includes('w006')) return 'parallel';
  if (value.includes('unsupported') || value.includes('custom') || value.includes('op')) return 'op';
  if (value.includes('config')) return 'config';
  return 'general';
}

export function deriveIssueSummary(
  analysis: AnalysisResult,
  warnings: Warning[] = [],
): Array<{ severity: string; count: number; fill: string }> {
  const diagnostics = analysis.diagnostics ?? [];
  const counts = {
    critical: 0,
    warning: 0,
    info: 0,
    hint: 0,
  };

  diagnostics.forEach((diagnostic) => {
    counts[normalizeSeverity(diagnostic.severity)] += 1;
  });
  warnings.forEach((warning) => {
    if (warning.type === 'error') counts.critical += 1;
    else if (warning.type === 'warning') counts.warning += 1;
    else counts.info += 1;
  });

  const entries = [
    { severity: 'Critical', count: counts.critical, fill: SIMULATION_COLORS.red },
    { severity: 'Warning', count: counts.warning, fill: SIMULATION_COLORS.amber },
    { severity: 'Info', count: counts.info, fill: SIMULATION_COLORS.blue },
    { severity: 'Hint', count: counts.hint, fill: SIMULATION_COLORS.green },
  ].filter((entry) => entry.count > 0);

  // Always return at least one entry so the chart never disappears
  if (entries.length === 0) {
    entries.push({ severity: 'All Clear', count: 0, fill: SIMULATION_COLORS.green });
  }

  return entries;
}

export function deriveDiagnosticsByLayer(
  analysis: AnalysisResult,
  warnings: Warning[] = [],
  perLayer: PerLayerBreakdownRow[] = [],
  nodes: CanvasNode[] = [],
): Array<Record<string, string | number>> {
  const labelById = new Map(nodes.map((node) => [node.id, node.name || node.type]));
  const rows = new Map<string, Record<string, string | number>>();

  const ensureRow = (key: string) => {
    if (!rows.has(key)) {
      rows.set(key, {
        layer: key,
        shape: 0,
        memory: 0,
        parallel: 0,
        op: 0,
        config: 0,
        general: 0,
      });
    }
    return rows.get(key)!;
  };

  (analysis.diagnostics ?? []).forEach((diagnostic) => {
    const rowKey = diagnostic.layer_id ? (labelById.get(diagnostic.layer_id) ?? diagnostic.layer_id) : 'Global';
    const row = ensureRow(rowKey);
    const bucket = bucketCategory(diagnostic.category, diagnostic.code);
    row[bucket] = (row[bucket] as number) + severityWeight(diagnostic.severity);
  });

  warnings.forEach((warning) => {
    const rowKey = warning.nodeId ? (labelById.get(warning.nodeId) ?? warning.nodeId) : 'Global';
    const row = ensureRow(rowKey);
    row.general = (row.general as number) + severityWeight(warning.type);
  });

  if (rows.size === 0) {
    // Always return at least one row so the chart never disappears
    if (perLayer.length > 0) {
      perLayer.slice(0, 5).forEach((row) => ensureRow(row.name));
    } else {
      ensureRow('All layers — no diagnostics');
    }
  }

  return Array.from(rows.values()).slice(0, 8);
}

/**
 * Used to also report "Diagnostics" and "Memory Fit" bars — computed from
 * arbitrary UI-side weights (critical*0.2 + warning*0.08, and a flat +0.35
 * offset on the VRAM ratio) that trace to nothing the compiler emits. Only
 * the three bars below come directly from real compiler fields:
 * `tensorResolutionRatio`, the ratio of resolved to total tensors, and
 * `confidenceScore` itself.
 */
export function deriveConfidenceBars(
  analysis: AnalysisResult,
): Array<{ label: string; value: number; fill: string }> {
  const tensorConfidence = analysis.totalTensorCount > 0
    ? (1 - (analysis.unresolvedDimCount / analysis.totalTensorCount))
    : analysis.confidenceScore;

  return [
    { label: 'Shape', value: clamp(analysis.tensorResolutionRatio * 100, 0, 100), fill: SIMULATION_COLORS.green },
    { label: 'Concrete Dims', value: clamp(tensorConfidence * 100, 0, 100), fill: SIMULATION_COLORS.teal },
    { label: 'Overall', value: clamp(analysis.confidenceScore * 100, 0, 100), fill: SIMULATION_COLORS.indigo },
  ];
}

export function deriveUnsupportedOps(analysis: AnalysisResult): Array<{
  name: string;
  detail: string;
  count: number;
  severity: string;
}> {
  const items: Array<{ name: string; detail: string; count: number; severity: string }> = [];
  const customWarnings = (analysis.reportWarnings ?? []).filter((warning) => /custom|unsupported|estimated flops/i.test(warning));
  customWarnings.forEach((warning) => {
    items.push({
      name: 'custom_ops',
      detail: warning,
      count: analysis.customLayerCount ?? 1,
      severity: 'warning',
    });
  });

  (analysis.diagnostics ?? [])
    .filter((diagnostic) => {
      const code = diagnostic.code?.toLowerCase() ?? '';
      return code === 'e003' || code === 'e004' || code === 'w001';
    })
    .forEach((diagnostic) => {
      items.push({
        name: diagnostic.code?.toLowerCase() ?? diagnostic.category,
        detail: diagnostic.message,
        count: 1,
        severity: diagnostic.severity,
      });
    });

  if (items.length === 0) {
    items.push({
      name: 'supported_ops',
      detail: 'No unsupported or fallback operations were reported by the compiler.',
      count: 0,
      severity: 'info',
    });
  }

  return items;
}

export function deriveResolutionDistribution(
  analysis: AnalysisResult,
  _warnings: Warning[] = [],
): Array<{ name: string; value: number; fill: string }> {
  if (analysis.totalTensorCount > 0) {
    const total = Math.max(Math.round(analysis.totalTensorCount || 0), 0);
    const certain = Math.min(total, Math.max(0, Math.round(total * clamp(analysis.tensorResolutionRatio, 0, 1))));

    let remaining = Math.max(0, total - certain);
    const unknown = Math.min(remaining, Math.max(analysis.customLayerCount ?? 0, 0));
    remaining -= unknown;

    const ambiguous = analysis.unresolvedDimCount > 0 ? remaining : 0;
    const inferred = Math.max(0, remaining - ambiguous);

    const entries = [
      { name: 'Certain', value: certain, fill: SIMULATION_COLORS.green },
      { name: 'Inferred', value: inferred, fill: SIMULATION_COLORS.blue },
      { name: 'Ambiguous', value: ambiguous, fill: SIMULATION_COLORS.amber },
      { name: 'Unknown', value: unknown, fill: SIMULATION_COLORS.red },
    ].filter((entry) => entry.value > 0);

    if (entries.length > 0) return entries;
  }

  // Fallback: always show at least one entry
  return [{ name: 'Certain', value: 100, fill: SIMULATION_COLORS.green }];
}

// `derivePenaltyWaterfall` used to live here — only its first and last
// steps (100, and `confidenceScore * 100`) were real; the two steps in
// between multiplied by flat constants (0.6, 0.8) that don't trace to
// anything the compiler computes, an invented breakdown of a real number.
// `deriveFusionCandidates` used to live here too, alongside it — pattern-
// matched fusion "difficulty" from node-type-name string matching, not a
// compiler measurement, self-disclosed as such in its own UI copy. Both
// removed along with the Diagnostics/Optimization cards that rendered them.
