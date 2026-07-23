import { Bug, Info, Cpu, Server } from 'lucide-react';
import { AnalysisResult, CanvasNode, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';
import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  deriveConfidenceBars,
  deriveDiagnosticsByLayer,
  deriveIssueSummary,
  derivePenaltyWaterfall,
  deriveResolutionDistribution,
  deriveUnsupportedOps,
  severityColor,
} from '../simulationData.ts';
import {
  ChartCard,
  ChartContainer,
  chartTooltipStyle,
  CHART_MARGINS,
  ChartErrorBoundary,
  EmptyChartState,
} from '../shared';


interface DebuggingChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  warnings?: Warning[];
  nodes?: CanvasNode[];
}

const CHART_COLORS = [
  'hsl(var(--chart-4))',
  'hsl(var(--chart-3))',
  'hsl(var(--chart-1))',
  'hsl(var(--chart-2))',
  'hsl(var(--chart-5))',
  'hsl(var(--chart-6))',
];

function cellTone(value: number): string {
  if (value >= 3) return 'bg-red-500/80';
  if (value >= 2) return 'bg-amber-500/80';
  if (value >= 1) return 'bg-blue-500/70';
  return 'bg-secondary/30';
}

export function DebuggingCharts({
  analysis,
  perLayer = [],
  warnings = [],
  nodes = [],
}: DebuggingChartsProps) {
  if (!analysis) return null;

  // ── Derived data (always returns at least one item per function) ──
  const severityRows = deriveIssueSummary(analysis, warnings);
  const diagnosticsByLayer = deriveDiagnosticsByLayer(analysis, warnings, perLayer, nodes);
  const confidenceRows = deriveConfidenceBars(analysis, warnings);
  const rawOps = Object.keys(analysis.opsDistribution ?? {}).length > 0
    ? Object.entries(analysis.opsDistribution).map(([name, value]) => ({ name, value }))
    : [];
  const hasOpData = rawOps.length > 0;
  const opRows = hasOpData ? rawOps : [];
  const unsupportedRows = deriveUnsupportedOps(analysis);
  const resolutionRows = deriveResolutionDistribution(analysis, warnings);
  const waterfallRows = derivePenaltyWaterfall(analysis);
  const hasBlocking = warnings.some((warning) => warning.type === 'error');

  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Bug className="w-5 h-5 text-red-400" />
            Debugging — Diagnostics & Confidence
          </h2>
          <div className="flex items-center gap-2 text-[10px] text-muted-foreground bg-secondary/50 px-2 py-1 rounded-md">
            <Info className="w-3 h-3" />
            Direct view into compiler diagnostics, shape confidence, and fallback penalties
          </div>
        </div>

        {/* ── Row 1: 8.1 · 8.2 · 8.3 ── */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* ── 8.1 Diagnostic Severity ── */}
          <ChartCard title="8.1 — Diagnostic Severity">
            <ChartContainer minH={256}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={severityRows}
                    dataKey="count"
                    innerRadius={60}
                    outerRadius={90}
                    paddingAngle={3}
                  >
                    {severityRows.map((entry) => (
                      <Cell key={entry.severity} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [`${value}`, 'Count']} />
                </PieChart>
              </ResponsiveContainer>
            </ChartContainer>
            <div className="mt-3 grid grid-cols-2 gap-2">
              {severityRows.map((entry) => (
                <div key={entry.severity} className="flex items-center justify-between rounded-md bg-secondary/20 px-2 py-1">
                  <span className="text-[11px]" style={{ color: entry.fill }}>{entry.severity}</span>
                  <span className="text-[11px] font-mono font-semibold">{entry.count}</span>
                </div>
              ))}
            </div>
          </ChartCard>

          {/* ── 8.2 Diagnostics by Layer ── */}
          <ChartCard title="8.2 — Diagnostics by Layer">
            <div className="space-y-2">
              <div className="grid grid-cols-[1.2fr_repeat(6,minmax(0,1fr))] gap-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                <span>Layer</span>
                <span className="text-center">Shape</span>
                <span className="text-center">Memory</span>
                <span className="text-center">Parallel</span>
                <span className="text-center">Op</span>
                <span className="text-center">Config</span>
                <span className="text-center">General</span>
              </div>
              {diagnosticsByLayer.map((row) => (
                <div key={String(row.layer)} className="grid grid-cols-[1.2fr_repeat(6,minmax(0,1fr))] gap-1 items-center">
                  <span className="text-[11px] text-muted-foreground truncate pr-2">{String(row.layer)}</span>
                  {['shape', 'memory', 'parallel', 'op', 'config', 'general'].map((bucket) => {
                    const value = Number(row[bucket] ?? 0);
                    return (
                      <div key={bucket} className={`h-6 rounded ${cellTone(value)} flex items-center justify-center text-[11px] font-semibold text-white`}>
                        {value > 0 ? value : ''}
                      </div>
                    );
                  })}
                </div>
              ))}
            </div>
          </ChartCard>

          {/* ── 8.3 Shape Confidence ── */}
          <ChartCard title="8.3 — Shape Confidence">
            <ChartContainer minH={256}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={confidenceRows} margin={CHART_MARGINS.bar}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis dataKey="label" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}%`} />
                  <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [`${value.toFixed(0)}%`, 'Confidence']} />
                  <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                    {confidenceRows.map((entry) => (
                      <Cell key={entry.label} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>
        </div>

        {/* ── 8.4 Parallelism Efficiency (full width) ── */}
        <ChartCard title="8.4 — Parallelism Efficiency">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="rounded-lg bg-secondary/20 p-3">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-primary" />
                <span className="text-[11px] font-medium text-muted-foreground">Data Parallel</span>
              </div>
              <div className="flex items-end gap-2">
                <span className="text-2xl font-bold font-mono">{(analysis.dataParallelEfficiency * 100).toFixed(0)}%</span>
                <span className="text-[11px] text-muted-foreground pb-1">efficiency</span>
              </div>
              {(analysis.communicationOverhead ?? 0) > 0 && (
                <div className="mt-2">
                  <div className="flex justify-between text-[10px] text-muted-foreground mb-1">
                    <span>Communication Overhead</span>
                    <span>{(analysis.communicationOverhead * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <div className="h-full bg-orange-500 rounded-full transition-all" style={{ width: `${Math.min(analysis.communicationOverhead * 100, 100)}%` }} />
                  </div>
                </div>
              )}
            </div>
            <div className="rounded-lg bg-secondary/20 p-3">
              <div className="flex items-center gap-2 mb-2">
                <Server className="w-4 h-4 text-primary" />
                <span className="text-[11px] font-medium text-muted-foreground">Pipeline Parallel</span>
              </div>
              <div className="text-2xl font-bold font-mono">{analysis.pipelineStages || '—'}</div>
              <div className="text-[11px] text-muted-foreground mt-1">pipeline stages</div>
            </div>
            <div className="rounded-lg bg-secondary/20 p-3">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-purple-400" />
                <span className="text-[11px] font-medium text-muted-foreground">Tensor Parallel</span>
              </div>
              <div className="text-2xl font-bold font-mono">{analysis.tensorParallelDegree || '—'}</div>
              <div className="text-[11px] text-muted-foreground mt-1">TP degree</div>
            </div>
            <div className="rounded-lg bg-secondary/20 p-3">
              <div className="flex items-center gap-2 mb-2">
                <Cpu className="w-4 h-4 text-emerald-400" />
                <span className="text-[11px] font-medium text-muted-foreground">Optimal GPUs</span>
              </div>
              <div className="text-2xl font-bold font-mono">{analysis.optimalGpuCount}</div>
              <div className="text-[11px] text-muted-foreground mt-1">recommended</div>
            </div>
          </div>
          {(analysis.gpuName) && (
            <div className="mt-4 rounded-lg bg-secondary/20 px-3 py-2">
              <div className="flex flex-wrap gap-4 text-[11px]">
                <span><span className="text-muted-foreground">Interconnect:</span> {analysis.interconnect || 'N/A'}</span>
                {analysis.interconnectBandwidthGbs ? <span><span className="text-muted-foreground">IC Bandwidth:</span> {analysis.interconnectBandwidthGbs} GB/s</span> : null}
                <span><span className="text-muted-foreground">Tensor Cores:</span> {analysis.tensorCoreUtilization ? `${(analysis.tensorCoreUtilization * 100).toFixed(0)}%` : 'N/A'}</span>
                {analysis.effectiveTflops ? <span><span className="text-muted-foreground">Effective TFLOPs:</span> {analysis.effectiveTflops.toFixed(1)}</span> : null}
                {analysis.samplesPerS ? <span><span className="text-muted-foreground">Samples/s:</span> {analysis.samplesPerS.toFixed(1)}</span> : null}
              </div>
            </div>
          )}
        </ChartCard>

        {/* ── Row 2: 8.5 · 8.6 · 8.7 ── */}
        <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
          {/* ── 8.5 OpKind Distribution ── */}
          <ChartCard title="8.5 — OpKind Distribution">
            <ChartContainer minH={256}>
              {hasOpData ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={opRows.slice(0, 6)}
                      dataKey="value"
                      outerRadius={88}
                      paddingAngle={2}
                    >
                      {opRows.slice(0, 6).map((entry, index) => (
                        <Cell key={entry.name} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [`${value}`, 'Ops']} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChartState
                  icon={Bug}
                  title="No op distribution data"
                  description="Run analysis to see operation kind distribution."
                />
              )}
            </ChartContainer>
          </ChartCard>

          {/* ── 8.6 Unsupported Ops / Fallbacks ── */}
          <ChartCard title="8.6 — Unsupported Ops / Fallbacks">
            <div className="space-y-3 max-h-64 overflow-auto pr-1 scrollbar-thin">
              {unsupportedRows.map((item) => (
                <div key={`${item.name}:${item.detail}`} className="rounded-lg bg-secondary/20 px-3 py-2">
                  <div className="flex items-center justify-between gap-2 text-xs">
                    <span className="font-semibold">{item.name}</span>
                    <span className="rounded px-2 py-0.5 text-[10px]" style={{ color: severityColor(item.severity), backgroundColor: `${severityColor(item.severity)}22` }}>
                      {item.severity}
                    </span>
                  </div>
                  <div className="mt-1 text-[11px] text-muted-foreground">{item.detail}</div>
                  <div className="mt-1 text-[11px] text-muted-foreground">Occurrences: {item.count}</div>
                </div>
              ))}
            </div>
          </ChartCard>

          {/* ── 8.7 Resolution Distribution ── */}
          <ChartCard title="8.7 — Resolution Distribution">
            <ChartContainer minH={256}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={resolutionRows} margin={CHART_MARGINS.bar}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [`${value}`, 'Count']} />
                  <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                    {resolutionRows.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>
        </div>

        {/* ── 8.8 Penalty Impact Waterfall (full width) ── */}
        <ChartCard title="8.8 — Penalty Impact Waterfall">
          <div className="grid grid-cols-5 gap-2">
            {waterfallRows.map((row, index) => {
              const isInitial = index === 0;
              const isFinal = index === waterfallRows.length - 1;
              const fill = isInitial || isFinal
                ? 'hsl(var(--chart-2))'
                : row.delta < 0
                  ? 'hsl(var(--chart-3))'
                  : 'hsl(var(--chart-1))';
              return (
                <div key={row.label} className="rounded-lg overflow-hidden border border-border/50 bg-secondary/10">
                  <div
                    className="h-32 flex items-end justify-center pb-3 text-sm font-bold font-mono text-white"
                    style={{ backgroundColor: fill.replace(')', 'cc)'), minHeight: `${Math.max(64, row.value * 1.2)}px` }}
                  >
                    {row.value.toFixed(0)}
                  </div>
                  <div className="px-3 py-2 text-center">
                    <div className="text-xs font-semibold">{row.label}</div>
                    {!isInitial && (
                      <div className="text-[11px] text-muted-foreground">
                        {row.delta >= 0 ? '+' : ''}
                        {row.delta.toFixed(0)}
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
          <div className={`mt-4 rounded-md px-3 py-2 text-[11px] ${hasBlocking ? 'bg-red-500/10 text-red-200' : 'bg-emerald-500/10 text-emerald-200'}`}>
            {hasBlocking
              ? 'Blocking issues remain. Resolve the critical diagnostics before trusting downstream performance projections.'
              : 'No blocking diagnostics were reported. Remaining penalties mostly come from confidence loss, unresolved dimensions, or custom-op estimation.'}
          </div>
        </ChartCard>
      </div>
    </ChartErrorBoundary>
  );
}
