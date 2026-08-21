import { HardDrive } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  AreaChart, Area,
  LineChart, Line,
} from 'recharts';
import { formatBytes } from '../simulationData.ts';
import {
  ChartCard,
  ChartContainer,
  DonutRing,
  StatCard,
  chartTooltipStyle,
  chartActiveDot,
  CHART_MARGINS,
  ChartErrorBoundary,
  EmptyChartState,
} from '../shared';


interface MemoryChartsProps {
  analysis?: AnalysisResult;
}

const MEMORY_COLORS = {
  activations: 'hsl(var(--chart-1))',
  weights: 'hsl(var(--chart-3))',
  temp: 'hsl(var(--chart-4))',
  forward: 'hsl(var(--chart-1))',
  backward: 'hsl(var(--chart-4))',
};

/**
 * Bytes per parameter at each precision NEURAX can analyse — the exact
 * values `neurax-formulas::dtype_bytes` uses, so this matches what a full
 * analysis at that precision would report, not a separate estimate.
 *
 * INT4 is listed at 1 byte/parameter, the same as INT8: the compiler does
 * not yet model sub-byte packing (two 4-bit values sharing one byte), so
 * this shows the same conservative, honest number the rest of the app
 * would if INT4 were selected — not the ~0.5 bytes/parameter true 4-bit
 * packing would achieve. Real quantized runtimes (llama.cpp, bitsandbytes)
 * do pack INT4, so treat this row as an upper bound, not GGUF-file-size
 * parity.
 */
const PRECISION_BYTES: Array<{ id: string; label: string; bytes: number }> = [
  { id: 'fp32', label: 'FP32', bytes: 4 },
  { id: 'fp16', label: 'FP16', bytes: 2 },
  { id: 'bf16', label: 'BF16', bytes: 2 },
  { id: 'int8', label: 'INT8', bytes: 1 },
  { id: 'int4', label: 'INT4', bytes: 1 },
];

/** Weight memory at every precision NEURAX supports, for a real parameter
 * count — pulled out of the component so it's testable without rendering
 * a chart. */
export function computePrecisionMemory(
  totalParams: number,
): Array<{ id: string; label: string; bytes: number }> {
  if (totalParams <= 0) return [];
  return PRECISION_BYTES.map(({ id, label, bytes }) => ({
    id,
    label,
    bytes: totalParams * bytes,
  }));
}

export function MemoryCharts({ analysis }: MemoryChartsProps) {
  if (!analysis || analysis.peakVramBytes === 0) {
    return (
      <EmptyChartState
        icon={HardDrive}
        title="No memory analysis available"
        description="Run analysis to see VRAM breakdown."
      />
    );
  }

  // ─── Data sources ────────────────────────────────────────────────────────

  const rawHeatmap = analysis.memory_heatmap || analysis.live_trace?.memory_heatmap || [];
  const hasRawHeatmap = rawHeatmap.length > 0;
  const gradientSource = analysis.gradient_memory_breakdown || analysis.live_trace?.gradient_memory_breakdown || [];

  // Derive heatmap from gradient_memory_breakdown when no raw heatmap
  const heatmapData = hasRawHeatmap
    ? rawHeatmap
    : gradientSource.length > 0
      ? gradientSource.map((g) => ({
        layer: g.name,
        timeline: [g.forward > g.backward ? 2 : g.backward > 0 ? 1 : 0].concat(
          Array.from({ length: 19 }, (_, j) => (g.backward > 0 && j < 15 ? 2 : j < 18 ? 1 : 0)),
        ),
      }))
      : [];

  const rawLiveness = analysis.memory_liveness || analysis.live_trace?.memory_liveness || [];
  const hasRawLiveness = rawLiveness.length > 0;
  const livenessData = hasRawLiveness
    ? rawLiveness.map(d => ({ step: d.step, vram: d.value / (1024 ** 2) }))
    : [];

  const rawGradient = gradientSource;
  const hasRawGradient = rawGradient.length > 0;
  const gradientChartData = hasRawGradient
    ? rawGradient.map(d => ({
      name: d.name,
      forward: d.forward / (1024 ** 2),
      backward: d.backward / (1024 ** 2),
    }))
    : [];

  const rawKv = analysis.kv_cache_scaling || analysis.live_trace?.kv_cache_scaling || [];
  const hasRawKv = rawKv.length > 0;
  const kvData = hasRawKv
    ? rawKv.map(d => ({ seq: d.seq, value: d.value / (1024 ** 2) }))
    : [];

  // 4.3 Donut
  const donutData = [
    { name: 'Activations', value: analysis.activationMemoryBytes, color: MEMORY_COLORS.activations },
    { name: 'Weights', value: analysis.parameterMemoryBytes, color: MEMORY_COLORS.weights },
    { name: 'Temp Buffers', value: Math.max(0, analysis.peakVramBytes - analysis.activationMemoryBytes - analysis.parameterMemoryBytes), color: MEMORY_COLORS.temp },
  ].filter(d => d.value > 0);

  // 4.3b Weight memory at every precision NEURAX supports, from the one
  // real number that doesn't change with precision: total parameters.
  // Re-running a full analysis at each precision would also be correct, but
  // asks the question three more times to answer it three more ways —
  // parameter count times bytes-per-parameter is the same arithmetic the
  // compiler itself does for whichever single precision is selected.
  //
  // Not marked against "the precision this analysis used": FP16 and BF16
  // are both 2 bytes/parameter, INT8 and INT4 both 1 (see PRECISION_BYTES),
  // so which one was actually selected can't be told apart from the
  // resulting byte count alone — showing a specific row as "active" would
  // be a guess dressed up as a fact for exactly the pairs a reader would
  // most want distinguished.
  const precisionData = computePrecisionMemory(analysis.totalParams);

  const supportsHeatmap = heatmapData.length > 0;

  return (
    <ChartErrorBoundary>
      <div className="space-y-6 pb-12">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <HardDrive className="w-5 h-5 text-primary" />
            Memory — VRAM Deep Dive
          </h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ── 4.1 Memory Heatmap ── */}
          <ChartCard
            title="Memory Heatmap (Timeline)"
            badge={
              hasRawHeatmap
                ? { text: 'live', variant: 'live' }
                : supportsHeatmap
                  ? { text: 'derived', variant: 'derived' }
                  : undefined
            }
          >
            {supportsHeatmap ? (
              <div className="overflow-x-auto">
                <div className="min-w-[400px] space-y-1">
                  {(heatmapData as Array<{ layer: string; timeline: number[] }>).map((layer, idx) => (
                    <div key={idx} className="flex items-center gap-2">
                      <div className="w-24 shrink-0 text-[10px] text-muted-foreground truncate" title={layer.layer}>
                        {layer.layer}
                      </div>
                      <div className="flex-1 flex gap-0.5 h-3">
                        {(layer.timeline || []).map((active, stepIdx) => (
                          <div
                            key={stepIdx}
                            className={`flex-1 rounded-sm transition-all duration-300 ${
                              active > 0
                                ? stepIdx < 5
                                  ? 'bg-green-500/60'
                                  : stepIdx < 12
                                    ? 'bg-yellow-500/60'
                                    : 'bg-orange-500/60'
                                : 'bg-white/5'
                            }`}
                          />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-3 flex justify-between text-[9px] text-muted-foreground uppercase tracking-widest">
                  <span>Layer 0</span>
                  <span>Step T</span>
                  <span>Layer N</span>
                </div>
              </div>
            ) : (
              <div className="h-32 flex items-center justify-center text-[11px] text-muted-foreground">
                <p>No heatmap data — run streaming analysis for live memory trace</p>
              </div>
            )}
          </ChartCard>

          {/* ── 4.2 VRAM Liveness ── */}
          <ChartCard
            title="VRAM Liveness"
            badge={hasRawLiveness ? { text: 'live', variant: 'live' } : undefined}
          >
            {hasRawLiveness ? (
              <ChartContainer>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={livenessData} margin={CHART_MARGINS.area}>
                    <defs>
                      <linearGradient id="memVram" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={MEMORY_COLORS.activations} stopOpacity={0.3} />
                        <stop offset="95%" stopColor={MEMORY_COLORS.activations} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                    <XAxis dataKey="step" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(val) => `${val}M`} />
                    <Tooltip contentStyle={chartTooltipStyle()} formatter={(val: number) => [`${val.toFixed(2)} MB`, 'VRAM']} />
                    <Area
                      type="monotone"
                      dataKey="vram"
                      stroke={MEMORY_COLORS.activations}
                      fillOpacity={1}
                      fill="url(#memVram)"
                      strokeWidth={2}
                      animationDuration={1500}
                      dot={false}
                      activeDot={chartActiveDot(MEMORY_COLORS.activations)}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartContainer>
            ) : (
              <EmptyChartState
                icon={HardDrive}
                title="No liveness data"
                description="Run streaming analysis to see VRAM liveness over time."
              />
            )}
          </ChartCard>

          {/* ── 4.3 Peak VRAM Breakdown ── */}
          <ChartCard title="Peak VRAM Breakdown">
            <div className="h-56 relative flex items-center justify-center">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={donutData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                    animationDuration={1000}
                  >
                    {donutData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={chartTooltipStyle()} formatter={(val: number) => formatBytes(val)} />
                </PieChart>
              </ResponsiveContainer>
              <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                <span className="text-[10px] text-muted-foreground uppercase">Peak</span>
                <span className="text-xl font-bold font-mono">{formatBytes(analysis.peakVramBytes)}</span>
              </div>
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2">
              {donutData.map((d, i) => (
                <div key={i} className="flex flex-col items-center gap-1">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                    <span className="text-[10px] text-muted-foreground">{d.name}</span>
                  </div>
                  <span className="text-xs font-mono font-bold">{(d.value / analysis.peakVramBytes * 100).toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </ChartCard>

          {/* ── 4.3b Weight memory at every precision ── */}
          {precisionData.length > 0 && (
            <ChartCard
              title="Weight Memory by Precision"
              badge={{ text: 'derived', variant: 'derived' }}
            >
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={precisionData} layout="vertical" margin={CHART_MARGINS.barHorizontal}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} opacity={0.2} />
                    <XAxis type="number" tickFormatter={(v) => formatBytes(v)} fontSize={10} />
                    <YAxis type="category" dataKey="label" width={40} fontSize={11} />
                    <Tooltip
                      contentStyle={chartTooltipStyle()}
                      formatter={(val: number) => formatBytes(val)}
                    />
                    <Bar dataKey="bytes" fill={MEMORY_COLORS.weights} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <p className="mt-2 text-[10px] text-muted-foreground leading-relaxed">
                {analysis.totalParams.toLocaleString('en-US')} parameters × bytes/parameter at
                each precision — weights only, not activations or KV-cache, which also shrink
                with precision but depend on the workload, not just the model. INT4 shown at 1
                byte/parameter: NEURAX does not yet model true 4-bit packing, so treat it as an
                upper bound, not GGUF-file-size parity.
              </p>
            </ChartCard>
          )}

          {/* ── 4.4 Memory Fragmentation & OOM Risk ── */}
          <ChartCard title="Memory Fragmentation & OOM Risk">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="text-[10px] text-muted-foreground uppercase">Fragmentation</div>
                <div className="relative h-32 flex items-center justify-center">
                  <DonutRing
                    value={Math.min(analysis.memoryFragmentationPct ?? 0, 100)}
                    size={112}
                    strokeWidth={10}
                    color={
                      (analysis.memoryFragmentationPct ?? 0) > 30
                        ? 'hsl(var(--chart-4))'
                        : (analysis.memoryFragmentationPct ?? 0) > 10
                          ? 'hsl(var(--chart-3))'
                          : 'hsl(var(--chart-2))'
                    }
                    centerLabel={`${(analysis.memoryFragmentationPct ?? 0).toFixed(0)}%`}
                    centerSublabel="fragmented"
                  />
                </div>
              </div>
              <div className="space-y-3">
                <div className="text-[10px] text-muted-foreground uppercase">OOM Risk</div>
                <StatCard
                  label="OOM Risk"
                  value={analysis.oomRisk || 'low'}
                  variant={
                    analysis.oomRisk === 'high'
                      ? 'danger'
                      : analysis.oomRisk === 'medium'
                        ? 'warning'
                        : 'success'
                  }
                />
                <div className="space-y-1">
                  <div className="flex justify-between text-[10px]">
                    <span className="text-muted-foreground">Max Batch Fit</span>
                    <span className="font-mono font-semibold">{analysis.maxBatchSizeFit || '—'}</span>
                  </div>
                  <div className="flex justify-between text-[10px]">
                    <span className="text-muted-foreground">Peak / GPU</span>
                    <span className="font-mono font-semibold">
                      {(analysis.peakVramBytes / 1e9).toFixed(2)} GB / {analysis.gpuMemoryGb?.toFixed(1)} GB
                    </span>
                  </div>
                  <div className="flex justify-between text-[10px]">
                    <span className="text-muted-foreground">Utilization</span>
                    <span className="font-mono font-semibold">
                      {analysis.gpuMemoryGb > 0
                        ? `${((analysis.peakVramBytes / 1e9 / analysis.gpuMemoryGb) * 100).toFixed(0)}%`
                        : '—'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </ChartCard>

          {/* ── 4.5 Gradient Memory ── */}
          <ChartCard
            title="Gradient Memory (Training)"
            badge={hasRawGradient ? { text: 'live', variant: 'live' } : undefined}
          >
            {hasRawGradient ? (
              <ChartContainer>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={gradientChartData} margin={{ ...CHART_MARGINS.bar, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                    <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} tick={{ fill: 'hsl(var(--muted-foreground))' }} interval={0} angle={-25} textAnchor="end" height={72} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(val) => `${val}M`} />
                    <Tooltip contentStyle={chartTooltipStyle()} />
                    <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                    <Bar dataKey="forward" name="Forward" stackId="a" fill={MEMORY_COLORS.forward} radius={[0, 0, 0, 0]} />
                    <Bar dataKey="backward" name="Backward" stackId="a" fill={MEMORY_COLORS.backward} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartContainer>
            ) : (
              <EmptyChartState
                icon={HardDrive}
                title="No gradient memory data"
                description="Run training analysis to see gradient memory breakdown."
              />
            )}
          </ChartCard>

          {/* ── 4.6 KV Cache Growth ── */}
          <ChartCard
            title="KV Cache Growth (LLM)"
            badge={hasRawKv ? { text: 'live', variant: 'live' } : undefined}
            className="lg:col-span-2"
          >
            {hasRawKv ? (
              <ChartContainer minH={256}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={kvData} margin={{ ...CHART_MARGINS.line, left: 20, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis
                      dataKey="seq"
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={11}
                      tickLine={false}
                      axisLine={false}
                      label={{
                        value: 'Sequence Length',
                        position: 'bottom',
                        offset: 0,
                        fontSize: 11,
                        fill: 'hsl(var(--muted-foreground))',
                      }}
                    />
                    <YAxis
                      stroke="hsl(var(--muted-foreground))"
                      fontSize={11}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(val) => `${val}M`}
                      label={{
                        value: 'Cache Size (MB)',
                        angle: -90,
                        position: 'insideLeft',
                        offset: 0,
                        fontSize: 11,
                        fill: 'hsl(var(--muted-foreground))',
                      }}
                    />
                    <Tooltip contentStyle={chartTooltipStyle()} formatter={(val: number) => [`${val.toFixed(2)} MB`, 'Cache Size']} />
                    <Line
                      type="stepAfter"
                      dataKey="value"
                      stroke={MEMORY_COLORS.activations}
                      strokeWidth={3}
                      dot={{ r: 4, fill: MEMORY_COLORS.activations, strokeWidth: 0 }}
                      activeDot={{ ...chartActiveDot(MEMORY_COLORS.activations), r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </ChartContainer>
            ) : (
              <EmptyChartState
                icon={HardDrive}
                title="No KV cache data"
                description="Run LLM analysis to see KV cache scaling projections."
              />
            )}
          </ChartCard>
        </div>
      </div>
    </ChartErrorBoundary>
  );
}
