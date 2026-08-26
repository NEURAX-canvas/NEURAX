import { Layers } from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import { hasPerLayerRows, parseFlopsValue, formatCompactNumber, formatBytes, SIMULATION_COLORS } from '../simulationData.ts';
import { ChartCard, ChartContainer, ChartErrorBoundary, EmptyChartState, chartTooltipStyle, CHART_MARGINS } from '../shared';

interface PerLayerChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

const BAR_COLOR = SIMULATION_COLORS.blue;
const TOP_N = 8;

/**
 * Used to be a table (11+ rows, 6 columns) before that: 6 separate bar
 * charts, one per metric, all plotting the same handful of layer names.
 * Neither survives the "only what needs a graph" bar this pass sets: a
 * table isn't a chart at all, and six charts of the same rows were the
 * exact repetition this tab is meant to avoid. Two rankings — which layers
 * dominate FLOPs, which dominate VRAM — are genuinely different questions
 * (an embedding table can lead on VRAM while contributing almost no FLOPs)
 * and each one is a real, single-metric ranking a bar chart is the right
 * shape for.
 */
function buildRows(analysis: AnalysisResult, perLayer: PerLayerBreakdownRow[]) {
  const vramMap = analysis.perLayerVram ?? {};
  const totalVramBytes = analysis.peakVramBytes || 1;
  const totalParams = analysis.totalParams || 1;
  const totalFlops = analysis.totalFlops || 1;

  return perLayer.map((row) => {
    const flops = parseFlopsValue(row.flops);
    const vramValue = vramMap[row.id] ?? vramMap[row.name];
    const vramBytes = typeof vramValue === 'number' && vramValue > 0
      ? vramValue
      : ((row.params ?? 0) / totalParams) * totalVramBytes;
    return {
      key: row.id ?? row.name,
      name: row.name.length > 18 ? `${row.name.slice(0, 16)}…` : row.name,
      flops,
      flopsSharePct: (flops / totalFlops) * 100,
      vramBytes,
    };
  });
}

function topByFlops(rows: ReturnType<typeof buildRows>) {
  const sorted = [...rows].sort((a, b) => b.flops - a.flops);
  return sorted.slice(0, TOP_N);
}

function topByVram(rows: ReturnType<typeof buildRows>) {
  const sorted = [...rows].sort((a, b) => b.vramBytes - a.vramBytes);
  return sorted.slice(0, TOP_N);
}

export function PerLayerCharts({ analysis, perLayer }: PerLayerChartsProps) {
  const hasData = analysis && hasPerLayerRows(perLayer);

  if (!hasData) {
    return (
      <EmptyChartState
        icon={Layers}
        title="No per-layer data available"
        description="Run analysis to see individual layer metrics."
      />
    );
  }

  const rows = buildRows(analysis!, perLayer ?? []);
  const flopsRows = topByFlops(rows);
  const vramRows = topByVram(rows);

  return (
    <ChartErrorBoundary>
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-primary" />
          <h2 className="text-base font-semibold">Per Layer Breakdown</h2>
          <span className="text-xs text-muted-foreground">— top {Math.min(TOP_N, rows.length)} of {rows.length} layers</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ChartCard title="Layers by FLOPs">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={flopsRows} layout="vertical" margin={CHART_MARGINS.barHorizontal}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v: number) => formatCompactNumber(v)} />
                  <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} width={110} />
                  <Tooltip
                    contentStyle={chartTooltipStyle()}
                    formatter={(value: number, _name, props) => [`${formatCompactNumber(value)}FLOPs (${props.payload.flopsSharePct.toFixed(1)}%)`, 'FLOPs']}
                  />
                  <Bar dataKey="flops" radius={[0, 4, 4, 0]}>
                    {flopsRows.map((row) => (
                      <Cell key={row.key} fill={BAR_COLOR} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          <ChartCard title="Layers by VRAM">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={vramRows} layout="vertical" margin={CHART_MARGINS.barHorizontal}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" horizontal={false} />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v: number) => formatBytes(v)} />
                  <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} width={110} />
                  <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [formatBytes(value), 'VRAM']} />
                  <Bar dataKey="vramBytes" radius={[0, 4, 4, 0]}>
                    {vramRows.map((row) => (
                      <Cell key={row.key} fill="hsl(var(--chart-3))" />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>
        </div>
      </div>
    </ChartErrorBoundary>
  );
}
