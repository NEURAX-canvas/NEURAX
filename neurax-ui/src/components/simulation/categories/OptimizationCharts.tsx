import { Target, Info } from 'lucide-react';
import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  Bar, CartesianGrid, Cell, ComposedChart, Legend, Line,
  Pie, PieChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  buildDerivedLayerMetrics,
  formatCompactNumber,
} from '../simulationData.ts';
import {
  ChartCard,
  ChartContainer,
  ChartLegend,
  chartTooltipStyle,
  chartActiveDot,
  CHART_MARGINS,
  ChartErrorBoundary,
  EmptyChartState,
} from '../shared';


interface OptimizationChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

function buildRooflineRows(analysis: AnalysisResult) {
  const intensity = Math.max(analysis.arithmeticIntensity, 0.1);
  const crossover = analysis.gpuBandwidthGbs > 0 ? (analysis.gpuTflops / analysis.gpuBandwidthGbs) * 1000 : intensity;
  const maxIntensity = Math.max(intensity * 4, crossover * 2, 16);

  // Generate log-spaced intensity values for smooth curves on log-log axes
  const numSteps = 30;
  const logMin = Math.log10(0.05);
  const logMax = Math.log10(maxIntensity);
  const step = (logMax - logMin) / (numSteps - 1);
  const intensities = Array.from({ length: numSteps }, (_, i) =>
    Math.round(10 ** (logMin + i * step) * 100) / 100
  );

  return intensities.map((x) => ({
    intensity: x,
    memoryRoof: (x * analysis.gpuBandwidthGbs) / 1000,
    computeRoof: analysis.gpuTflops,
  }));
}

export function OptimizationCharts({
  analysis,
  perLayer = [],
}: OptimizationChartsProps) {
  if (!analysis || analysis.totalFlops === 0) {
    return (
      <EmptyChartState
        icon={Target}
        title="No optimization data available"
        description="Run analysis to see roofline and bottlenecks."
      />
    );
  }

  // Achieved throughput is measured from the analysis's own latency. Without it
  // there is nothing to report, so leave it null rather than inventing a figure
  // from an assumed utilisation.
  const achievedTflops = analysis.latencyMs
    ? (analysis.totalFlops / 1e12) / (analysis.latencyMs / 1000)
    : null;
  const rooflineRows = buildRooflineRows(analysis);
  const layerRows = buildDerivedLayerMetrics(analysis, perLayer)
    .sort((a, b) => b.flops - a.flops)
    .map((row, index, list) => {
      const cumulative = list
        .slice(0, index + 1)
        .reduce((sum, current) => sum + current.flops, 0);
      const total = list.reduce((sum, current) => sum + current.flops, 0) || 1;
      return {
        name: row.name,
        flops: row.flops,
        cumulativePct: (cumulative / total) * 100,
      };
    });
  const rooflineMix = (() => {
    // "Mixed" used to blend in `(1 - confidenceScore) * 30` — data-quality
    // (how resolved the tensor shapes are) and hardware-utilization
    // (compute- vs memory-bound) are unrelated axes, so a low-confidence
    // analysis inflated "Mixed" regardless of where the design actually
    // sits on the roofline. Driven only by proximity to the 50/50 point now.
    const compute = Math.max(0, Math.min(analysis.rooflinePosition, 1));
    const mixed = Math.min(compute, 1 - compute) * 40;
    const memory = Math.max(0, 100 - mixed - compute * 100);
    return [
      { name: 'Compute-bound', value: compute * 100, fill: 'hsl(var(--chart-2))' },
      { name: 'Memory-bound', value: memory, fill: 'hsl(var(--chart-4))' },
      { name: 'Mixed', value: mixed, fill: 'hsl(var(--chart-3))' },
    ].filter((entry) => entry.value > 0);
  })();
  return (
    <ChartErrorBoundary>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            Optimization — Roofline & Bottlenecks
          </h2>
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground bg-secondary/50 px-2 py-1 rounded-md">
            <Info className="w-3 h-3" />
            Compiler-backed hardware ceilings and bottlenecks
          </div>
        </div>

        {/* `wide` and `square` cards never share a CSS grid row: a `wide`
            card's real height comes from its ChartContainer's `minH`, a
            `square` card's comes from `aspect-ratio` — mixed in one grid
            row, the default `align-items: stretch` forces the square card
            to the wide card's height while its width stays column-narrow,
            and Recharts' ResponsiveContainer measures that mismatched box
            and renders a cropped, off-scale chart. Each size gets its own
            row so its own sizing rule is the only one in play. */}
        <div className="space-y-6">
          {/* 6.1 Roofline Model — the one real chart this tab keeps: it's the
              only view that needs a log-log axis to be read correctly. */}
          <ChartCard title="Roofline Model" size="wide">
            <ChartContainer minH={340}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={rooflineRows} margin={CHART_MARGINS.composed}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis
                    dataKey="intensity"
                    scale="log"
                    type="number"
                    domain={['auto', 'auto']}
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) => `${value.toFixed(1)}`}
                    label={{
                      value: 'Arithmetic Intensity (FLOP/B) — log scale',
                      position: 'insideBottom',
                      offset: -4,
                      fill: 'hsl(var(--muted-foreground))',
                      fontSize: 11,
                    }}
                  />
                  <YAxis
                    scale="log"
                    type="number"
                    domain={['auto', 'auto']}
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(value) => `${value.toFixed(1)}`}
                    label={{
                      value: 'TFLOP/s — log scale',
                      angle: -90,
                      position: 'insideLeft',
                      fill: 'hsl(var(--muted-foreground))',
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    contentStyle={chartTooltipStyle()}
                    formatter={(value: number, name: string) => [`${value.toFixed(2)} TFLOP/s`, name === 'memoryRoof' ? 'Memory roof' : 'Compute roof']}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line type="monotone" dataKey="memoryRoof" name="Memory roof" stroke="hsl(var(--chart-4))" strokeWidth={2} dot={false} activeDot={chartActiveDot('hsl(var(--chart-4))')} />
                  <Line type="monotone" dataKey="computeRoof" name="Compute roof" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} activeDot={chartActiveDot('hsl(var(--chart-2))')} />
                  {achievedTflops !== null && (
                    <ReferenceDot
                      x={Math.max(analysis.arithmeticIntensity, 0.1)}
                      y={achievedTflops}
                      r={6}
                      fill="hsl(var(--chart-1))"
                      stroke="white"
                      label={{ value: 'Current model', position: 'top', fill: 'hsl(var(--chart-1))', fontSize: 11 }}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          {/* 6.2 Bottleneck Pareto — real per-layer FLOPs (bars, left axis)
              against the cumulative share they add up to (line, right
              axis): the 80/20 question a Per-Layer ranking alone can't
              answer, since it doesn't show where the cumulative curve
              bends. Needs both a categorical axis with real layer names
              and a second numeric axis, so it keeps the wide exception. */}
          <ChartCard title="Bottleneck Pareto (80/20)" size="wide">
            <ChartContainer minH={340}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={layerRows.slice(0, 8)} margin={{ ...CHART_MARGINS.composed, bottom: 36 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
                  <XAxis
                    dataKey="name"
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={11}
                    tickLine={false}
                    axisLine={false}
                    angle={-25}
                    textAnchor="end"
                    interval={0}
                    height={72}
                    tick={{ fontSize: 11 }}
                  />
                  <YAxis yAxisId="left" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(value: number) => formatCompactNumber(value)} />
                  <YAxis yAxisId="right" orientation="right" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                  <Tooltip
                    contentStyle={chartTooltipStyle()}
                    formatter={(value: number, name: string) => [
                      name === 'cumulativePct' ? `${value.toFixed(0)}%` : `${formatCompactNumber(value)}FLOPs`,
                      name === 'cumulativePct' ? 'Cumulative' : 'Layer FLOPs',
                    ]}
                  />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar yAxisId="left" dataKey="flops" name="Layer FLOPs" fill="hsl(var(--chart-1))" radius={[3, 3, 0, 0]} />
                  <Line yAxisId="right" type="monotone" dataKey="cumulativePct" name="Cumulative %" stroke="hsl(var(--chart-3))" strokeWidth={2} dot={{ r: 3, fill: 'hsl(var(--chart-3))' }} activeDot={{ ...chartActiveDot('hsl(var(--chart-3))'), r: 6 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          {/* 6.3 Compute vs Memory Bound — real, three-category share of
              `rooflinePosition`; a small donut with a legend is the
              standard shape for a whole split into few named parts. Kept
              to its own square footprint rather than stretched to match
              the wide charts above it. */}
          <div className="max-w-sm">
          <ChartCard title="Compute vs Memory Bound">
            <div className="h-full flex flex-col">
              <ChartContainer className="flex-1 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={rooflineMix}
                      dataKey="value"
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={85}
                      paddingAngle={3}
                    >
                      {rooflineMix.map((entry) => (
                        <Cell key={entry.name} fill={entry.fill} />
                      ))}
                    </Pie>
                    <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [`${value.toFixed(1)}%`, 'Share']} />
                  </PieChart>
                </ResponsiveContainer>
              </ChartContainer>
              <ChartLegend
                entries={rooflineMix.map((d) => ({ name: d.name, value: d.value, color: d.fill, formattedValue: `${d.value.toFixed(1)}%` }))}
                showPercent={false}
              />
            </div>
          </ChartCard>
          </div>
        </div>
      </div>
    </ChartErrorBoundary>
  );
}
