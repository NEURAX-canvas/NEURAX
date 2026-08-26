import { useState, useCallback } from 'react';
import { GitCompare, Plus, Trash2, Play, Loader2, Cpu, Zap, HardDrive, DollarSign, Thermometer, Clock, TrendingUp, AlertTriangle } from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { AnalysisResult } from '@/types/architecture.ts';
import { compareAnalyses, listHardware, CompareHardwareConfig, CompareResultItem, HardwareDetail } from '@/services/neuraxApi.ts';
import { formatBytes, formatCompactNumber } from '../simulationData.ts';
import {
  ChartCard,
  ChartContainer,
  ChartErrorBoundary,
  EmptyChartState,
  chartTooltipStyle,
  CHART_MARGINS,
} from '../shared';

interface ComparisonChartsProps {
  analysis?: AnalysisResult;
  topology?: Record<string, unknown>;
}

const PRECISION_OPTIONS = ['fp32', 'fp16', 'bf16', 'int8', 'fp8'];
const DEFAULT_GPUS = ['H100-SXM', 'A100-SXM', 'A100-PCIe', 'RTX4090', 'H200', 'GH200'];

interface ComparisonConfig {
  id: string;
  hardware: string;
  precision: string;
  batchSize: number;
  gpuCount: number;
}

function formatNumber(n: number | null | undefined, decimals = 2): string {
  if (n === null || n === undefined) return '—';
  return formatCompactNumber(n, decimals);
}

const COMPARISON_COLORS = ['hsl(var(--chart-1))', 'hsl(var(--chart-3))', 'hsl(var(--chart-4))', 'hsl(var(--chart-5))', 'hsl(var(--chart-2))'];

/**
 * One metric, one real horizontal bar chart. Used to be four almost-
 * identical blocks of hand-rolled CSS divs with an inline `width: pct%` —
 * a real chart type (recharts BarChart) does the same ranking-by-one-
 * metric job with an actual scale and tooltip, for the same reason the
 * Per-Layer and Optimization tabs moved off CSS bars.
 */
function ComparisonBar({
  title,
  items,
  formatValue,
}: {
  title: string;
  items: Array<{ label: string; value: number | null | undefined }>;
  formatValue: (value: number) => string;
}) {
  const data = items
    .filter((item): item is { label: string; value: number } => item.value !== null && item.value !== undefined)
    .map((item) => ({ name: item.label, value: item.value }));

  if (data.length === 0) {
    return (
      <ChartCard title={title}>
        <EmptyChartState icon={TrendingUp} title="No data" description="Run the comparison to see this metric." />
      </ChartCard>
    );
  }

  return (
    <ChartCard title={title}>
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={CHART_MARGINS.barHorizontal}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
            <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={formatValue} />
            <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} width={110} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [formatValue(value), title]} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((entry, idx) => (
                <Cell key={entry.name} fill={COMPARISON_COLORS[idx % COMPARISON_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

function extractMetric(report: Record<string, unknown>, key: string): unknown {
  const keys = key.split('.');
  let val: unknown = report;
  for (const k of keys) {
    if (val && typeof val === 'object') {
      val = (val as Record<string, unknown>)[k];
    } else {
      return undefined;
    }
  }
  return val;
}

export function ComparisonCharts({ analysis, topology }: ComparisonChartsProps) {
  const [configs, setConfigs] = useState<ComparisonConfig[]>([
    { id: '1', hardware: 'H100-SXM', precision: 'fp16', batchSize: analysis?.selectedBatchSize ?? 64, gpuCount: 1 },
    { id: '2', hardware: 'A100-SXM', precision: 'fp16', batchSize: analysis?.selectedBatchSize ?? 64, gpuCount: 1 },
  ]);
  const [results, setResults] = useState<CompareResultItem[]>([]);
  const [isComparing, setIsComparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hardwareList, setHardwareList] = useState<HardwareDetail[]>([]);
  const [hardwareLoaded, setHardwareLoaded] = useState(false);

  const loadHardware = useCallback(async () => {
    if (hardwareLoaded) return;
    try {
      const hw = await listHardware();
      setHardwareList(hw);
      setHardwareLoaded(true);
    } catch {
      setHardwareLoaded(true);
    }
  }, [hardwareLoaded]);

  if (!hardwareLoaded) {
    void loadHardware();
  }

  const gpuOptions = hardwareList.length > 0
    ? hardwareList.map(h => h.name)
    : DEFAULT_GPUS;

  const addConfig = () => {
    const id = String(Date.now());
    setConfigs(prev => [...prev, {
      id,
      hardware: gpuOptions[Math.min(prev.length, gpuOptions.length - 1)] || gpuOptions[0],
      precision: 'fp16',
      batchSize: analysis?.selectedBatchSize ?? 64,
      gpuCount: 1,
    }]);
  };

  const removeConfig = (id: string) => {
    setConfigs(prev => prev.length > 1 ? prev.filter(c => c.id !== id) : prev);
  };

  const updateConfig = (id: string, field: keyof ComparisonConfig, value: string | number) => {
    setConfigs(prev => prev.map(c => c.id === id ? { ...c, [field]: value } : c));
  };

  const runComparison = async () => {
    if (!topology) {
      setError('No topology available. Run an analysis first.');
      return;
    }

    setIsComparing(true);
    setError(null);

    try {
      const compareConfigs: CompareHardwareConfig[] = configs.map(c => ({
        hardware: c.hardware,
        precision: c.precision,
        batch_size: c.batchSize,
        gpu_count: c.gpuCount,
      }));

      const response = await compareAnalyses({
        topology,
        configs: compareConfigs,
      });

      setResults(response.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Comparison failed');
    } finally {
      setIsComparing(false);
    }
  };

  if (!analysis || analysis.totalParams === 0) {
    return (
      <EmptyChartState
        icon={GitCompare}
        title="No comparison data available"
        description="Run analysis to compare deployment trade-offs."
      />
    );
  }

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <GitCompare className="w-5 h-5 text-primary" />
            Comparison — Benchmarks & Variants
          </h2>
        </div>

        {/* Current config summary */}
        <ChartCard title="Current Configuration" size="wide" className="!min-h-0">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="rounded-lg border border-border/60 bg-secondary/20 px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Current device</div>
              <div className="mt-1 text-sm font-semibold">{analysis.gpuName || 'Unknown GPU'}</div>
            </div>
            <div className="rounded-lg border border-border/60 bg-secondary/20 px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Current precision</div>
              <div className="mt-1 text-sm font-semibold uppercase">{analysis.selectedPrecision || 'N/A'}</div>
            </div>
            <div className="rounded-lg border border-border/60 bg-secondary/20 px-4 py-3">
              <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Batch size</div>
              <div className="mt-1 text-sm font-semibold">{analysis.selectedBatchSize ?? 'N/A'}</div>
            </div>
          </div>
        </ChartCard>

        {/* Configuration cards */}
        <ChartCard title="Hardware Configurations to Compare" size="wide" className="!min-h-0" action={
          <button
            onClick={addConfig}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-primary/10 text-primary rounded-md hover:bg-primary/20 transition-colors"
          >
            <Plus className="w-3 h-3" />
            Add Config
          </button>
        }>
          <div className="space-y-3">
            {configs.map((config) => (
              <div key={config.id} className="flex items-center gap-3 p-3 rounded-lg border border-border/40 bg-secondary/10">
                <select
                  value={config.hardware}
                  onChange={(e) => updateConfig(config.id, 'hardware', e.target.value)}
                  className="flex-1 min-w-0 px-2 py-1.5 text-xs bg-background border border-border rounded-md"
                >
                  {gpuOptions.map(gpu => (
                    <option key={gpu} value={gpu}>{gpu}</option>
                  ))}
                </select>
                <select
                  value={config.precision}
                  onChange={(e) => updateConfig(config.id, 'precision', e.target.value)}
                  className="w-20 px-2 py-1.5 text-xs bg-background border border-border rounded-md"
                >
                  {PRECISION_OPTIONS.map(p => (
                    <option key={p} value={p}>{p.toUpperCase()}</option>
                  ))}
                </select>
                <div className="flex items-center gap-1">
                  <span className="text-[11px] text-muted-foreground">BS:</span>
                  <input
                    type="number"
                    value={config.batchSize}
                    onChange={(e) => updateConfig(config.id, 'batchSize', parseInt(e.target.value) || 1)}
                    className="w-16 px-2 py-1.5 text-xs bg-background border border-border rounded-md"
                    min={1}
                    max={4096}
                  />
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-[11px] text-muted-foreground">×GPU:</span>
                  <input
                    type="number"
                    value={config.gpuCount}
                    onChange={(e) => updateConfig(config.id, 'gpuCount', parseInt(e.target.value) || 1)}
                    className="w-14 px-2 py-1.5 text-xs bg-background border border-border rounded-md"
                    min={1}
                    max={256}
                  />
                </div>
                <button
                  onClick={() => removeConfig(config.id)}
                  className="p-1 text-muted-foreground hover:text-destructive transition-colors"
                  disabled={configs.length <= 1}
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-3 mt-4">
            <button
              onClick={runComparison}
              disabled={isComparing || configs.length === 0}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isComparing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              {isComparing ? 'Comparing...' : 'Run Comparison'}
            </button>
            {error && (
              <div className="flex items-center gap-1 text-xs text-destructive">
                <AlertTriangle className="w-3 h-3" />
                {error}
              </div>
            )}
          </div>
        </ChartCard>

        {/* Results table */}
        {results.length > 0 && (
          <ChartCard title="Comparison Results" size="wide" className="!min-h-0 overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-2 text-muted-foreground font-medium">Config</th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">
                    <div className="flex items-center justify-end gap-1"><Cpu className="w-3 h-3" /> Params</div>
                  </th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">
                    <div className="flex items-center justify-end gap-1"><Zap className="w-3 h-3" /> FLOPs</div>
                  </th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">
                    <div className="flex items-center justify-end gap-1"><HardDrive className="w-3 h-3" /> Peak VRAM</div>
                  </th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">
                    <div className="flex items-center justify-end gap-1"><Clock className="w-3 h-3" /> Latency</div>
                  </th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">
                    <div className="flex items-center justify-end gap-1"><TrendingUp className="w-3 h-3" /> Throughput</div>
                  </th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">
                    <div className="flex items-center justify-end gap-1"><DollarSign className="w-3 h-3" /> Cost/train</div>
                  </th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">
                    <div className="flex items-center justify-end gap-1"><Thermometer className="w-3 h-3" /> CO₂</div>
                  </th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">Max Batch</th>
                  <th className="text-right py-2 px-2 text-muted-foreground font-medium">GPU Util</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-border/50 bg-primary/5">
                  <td className="py-2 px-2 font-medium">
                    <div className="text-xs">{analysis.gpuName || 'Current'}</div>
                    <div className="text-[10px] text-muted-foreground">{analysis.selectedPrecision?.toUpperCase()} × {analysis.selectedBatchSize}</div>
                  </td>
                  <td className="text-right py-2 px-2">{formatNumber(analysis.totalParams)}</td>
                  <td className="text-right py-2 px-2">{formatNumber(analysis.totalFlops)}</td>
                  <td className="text-right py-2 px-2">{formatBytes(analysis.peakVramBytes)}</td>
                  <td className="text-right py-2 px-2">{analysis.latencyMs !== null ? `${analysis.latencyMs.toFixed(1)}ms` : '—'}</td>
                  <td className="text-right py-2 px-2">{formatNumber(analysis.throughputTokensPerS)}/s</td>
                  <td className="text-right py-2 px-2">${analysis.trainingCostUsd.toFixed(2)}</td>
                  <td className="text-right py-2 px-2">{analysis.co2Kg.toFixed(2)}kg</td>
                  <td className="text-right py-2 px-2">{analysis.maxBatchSizeFit}</td>
                  <td className="text-right py-2 px-2">{analysis.gpuUtilization !== null ? `${(analysis.gpuUtilization * 100).toFixed(1)}%` : '—'}</td>
                </tr>
                {results.map((result, idx) => {
                  const report = result.report as Record<string, unknown> | undefined;

                  // The backend report is flat under `metrics` — there is no
                  // `performance`/`memory`/`cost` nesting. Those paths never
                  // matched anything, so every field below `total_parameters`
                  // and `total_flops` silently resolved to `undefined` and
                  // rendered as "—" for every comparison row.
                  const totalParams = extractMetric(report ?? {}, 'metrics.total_parameters') as number | undefined;
                  const totalFlops = extractMetric(report ?? {}, 'metrics.total_flops') as number | undefined;
                  const peakVram = extractMetric(report ?? {}, 'metrics.peak_vram_bytes') as number | undefined;
                  const latency = extractMetric(report ?? {}, 'metrics.latency_ms') as number | null | undefined;
                  const throughput = extractMetric(report ?? {}, 'metrics.throughput_tokens_per_s') as number | undefined;
                  const trainingCost = extractMetric(report ?? {}, 'metrics.training_cost_usd') as number | undefined;
                  const co2 = extractMetric(report ?? {}, 'metrics.co2_kg') as number | undefined;
                  const maxBatch = extractMetric(report ?? {}, 'metrics.max_batch_size_fit') as number | undefined;
                  const gpuUtil = extractMetric(report ?? {}, 'metrics.gpu_utilization') as number | null | undefined;

                  return (
                    <tr key={idx} className="border-b border-border/30 hover:bg-secondary/10 transition-colors">
                      <td className="py-2 px-2 font-medium">
                        <div className="text-xs">{result.label}</div>
                        {result.error && (
                          <div className="text-[10px] text-destructive flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" /> {result.error}
                          </div>
                        )}
                      </td>
                      <td className="text-right py-2 px-2">{totalParams ? formatNumber(totalParams) : '—'}</td>
                      <td className="text-right py-2 px-2">{totalFlops ? formatNumber(totalFlops) : '—'}</td>
                      <td className="text-right py-2 px-2">{peakVram ? formatBytes(peakVram) : '—'}</td>
                      <td className="text-right py-2 px-2">{latency !== null && latency !== undefined ? `${latency.toFixed(1)}ms` : '—'}</td>
                      <td className="text-right py-2 px-2">{throughput ? `${formatNumber(throughput)}/s` : '—'}</td>
                      <td className="text-right py-2 px-2">{trainingCost ? `$${trainingCost.toFixed(2)}` : '—'}</td>
                      <td className="text-right py-2 px-2">{co2 ? `${co2.toFixed(2)}kg` : '—'}</td>
                      <td className="text-right py-2 px-2">{maxBatch ?? '—'}</td>
                      <td className="text-right py-2 px-2">{gpuUtil !== null && gpuUtil !== undefined ? `${(gpuUtil * 100).toFixed(1)}%` : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </ChartCard>
        )}

        {/* Visual comparison — one real chart per metric. Four separate
            charts, not one: throughput, latency, cost and VRAM are four
            incompatible units, and combining them on a shared axis is
            exactly the mistake already fixed elsewhere in this tab. */}
        {results.length > 0 && results.some(r => r.report) && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ComparisonBar
              title="Throughput (tokens/s)"
              items={[
                { label: `${analysis.gpuName} (${analysis.selectedPrecision})`, value: analysis.throughputTokensPerS },
                ...results.filter(r => r.report).map(r => ({
                  label: r.label,
                  value: extractMetric(r.report ?? {}, 'metrics.throughput_tokens_per_s') as number | undefined,
                })),
              ]}
              formatValue={(v) => `${formatNumber(v)}/s`}
            />
            <ComparisonBar
              title="Latency (ms)"
              items={[
                { label: `${analysis.gpuName} (${analysis.selectedPrecision})`, value: analysis.latencyMs },
                ...results.filter(r => r.report).map(r => ({
                  label: r.label,
                  value: extractMetric(r.report ?? {}, 'metrics.latency_ms') as number | null | undefined,
                })),
              ]}
              formatValue={(v) => `${v.toFixed(1)}ms`}
            />
            <ComparisonBar
              title="Training Cost (USD)"
              items={[
                { label: `${analysis.gpuName} (${analysis.selectedPrecision})`, value: analysis.trainingCostUsd },
                ...results.filter(r => r.report).map(r => ({
                  label: r.label,
                  value: extractMetric(r.report ?? {}, 'metrics.training_cost_usd') as number | undefined,
                })),
              ]}
              formatValue={(v) => `$${v.toFixed(2)}`}
            />
            <ComparisonBar
              title="Peak VRAM"
              items={[
                { label: `${analysis.gpuName} (${analysis.selectedPrecision})`, value: analysis.peakVramBytes },
                ...results.filter(r => r.report).map(r => ({
                  label: r.label,
                  value: extractMetric(r.report ?? {}, 'metrics.peak_vram_bytes') as number | undefined,
                })),
              ]}
              formatValue={(v) => formatBytes(v)}
            />
          </div>
        )}
      </div>
    </ChartErrorBoundary>
  );
}
