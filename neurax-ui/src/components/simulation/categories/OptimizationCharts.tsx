import { Target, Info } from 'lucide-react';
import { AnalysisResult, CanvasNode, Connection, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  Bar, CartesianGrid, Cell, ComposedChart, Legend, Line,
  Pie, PieChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import {
  buildDerivedLayerMetrics,
  deriveFusionCandidates,
  deriveOptimizationOpportunities,
  formatPercent,
} from '../simulationData.ts';
import {
  ChartCard,
  ChartContainer,
  chartTooltipStyle,
  chartActiveDot,
  CHART_MARGINS,
  ChartErrorBoundary,
  EmptyChartState,
} from '../shared';


interface OptimizationChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  nodes?: CanvasNode[];
  connections?: Connection[];
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
  nodes = [],
  connections = [],
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
    const compute = Math.max(0, Math.min(analysis.rooflinePosition, 1));
    const mixed = Math.min(18, (1 - analysis.confidenceScore) * 30 + Math.min(compute, 1 - compute) * 20);
    const memory = Math.max(0, 100 - mixed - compute * 100);
    return [
      { name: 'Compute-bound', value: compute * 100, fill: 'hsl(var(--chart-2))' },
      { name: 'Memory-bound', value: memory, fill: 'hsl(var(--chart-4))' },
      { name: 'Mixed', value: mixed, fill: 'hsl(var(--chart-3))' },
    ].filter((entry) => entry.value > 0);
  })();
  const opportunities = deriveOptimizationOpportunities(analysis);
  const fusionCandidates = deriveFusionCandidates(nodes, connections, perLayer);

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
            Compiler-backed hardware ceilings, bottlenecks, and fusion opportunities
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          {/* 6.1 Roofline Model */}
          <ChartCard title="Roofline Model" className="xl:col-span-2">
            <ChartContainer minH={320}>
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

          {/* 6.2 Bottleneck Pareto */}
          <ChartCard title="Bottleneck Pareto (80/20)">
            <ChartContainer minH={288}>
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={layerRows} margin={{ ...CHART_MARGINS.composed, bottom: 36 }}>
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
                  <YAxis yAxisId="left" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(value) => `${value.toFixed(0)}G`} />
                  <YAxis yAxisId="right" orientation="right" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                  <Tooltip
                    contentStyle={chartTooltipStyle()}
                    formatter={(value: number, name: string) => [
                      name === 'cumulativePct' ? `${value.toFixed(0)}%` : `${value.toFixed(2)} GFLOPs`,
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

          {/* 6.3 Compute vs Memory Bound */}
          <ChartCard title="Compute vs Memory Bound">
            <ChartContainer minH={288}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={rooflineMix}
                    dataKey="value"
                    innerRadius={62}
                    outerRadius={92}
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
            <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
              {rooflineMix.map((entry) => (
                <div key={entry.name} className="text-center">
                  <div className="font-semibold" style={{ color: entry.fill }}>{entry.name}</div>
                  <div>{entry.value.toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </ChartCard>

          {/* 6.4 Optimization Opportunities */}
          <ChartCard title="Optimization Opportunities">
            <div className="space-y-3">
              {opportunities.map((entry) => (
                <div key={entry.title}>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[11px]">{entry.title}</span>
                    <span className="text-[11px] font-mono">{entry.score.toFixed(0)}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-secondary overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${entry.score}%`,
                        backgroundColor:
                          entry.priority === 'high'
                            ? 'hsl(var(--chart-2))'
                            : entry.priority === 'medium'
                              ? 'hsl(var(--chart-3))'
                              : 'hsl(var(--chart-1))',
                      }}
                    />
                  </div>
                  <div className="text-[11px] text-muted-foreground mt-1">{entry.description}</div>
                </div>
              ))}
            </div>
          </ChartCard>

          {/* 6.5 Layer Fusion Candidates */}
          <ChartCard title="Layer Fusion Candidates">
            <div className="space-y-3">
              {fusionCandidates.map((candidate) => (
                <div key={candidate.label} className="rounded-lg bg-secondary/20 px-3 py-2">
                  <div className="flex items-center justify-between gap-3 text-xs">
                    <span className="font-medium">{candidate.label}</span>
                    <span className="text-[11px] font-mono" style={{ color: 'hsl(var(--chart-2))' }}>
                      {candidate.gainPct.toFixed(0)}%
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 rounded-full bg-secondary overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${candidate.gainPct}%`, backgroundColor: 'hsl(var(--chart-2))' }}
                    />
                  </div>
                  <div className="mt-1 text-[11px] text-muted-foreground">
                    {candidate.difficulty} integration • estimated uplift from reduced kernel launch / memory traffic
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-4 rounded-md border border-border/50 bg-secondary/10 px-3 py-2 text-[11px] text-muted-foreground">
              Roofline position: {formatPercent(analysis.rooflinePosition, 0)} toward compute-bound. Peak utilization is approximately{' '}
              {formatPercent((achievedTflops || 0) / Math.max(analysis.gpuTflops || 1, 1), 0)} of hardware peak.
            </div>
          </ChartCard>
        </div>
      </div>
    </ChartErrorBoundary>
  );
}
