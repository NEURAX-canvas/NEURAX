import { Layers } from 'lucide-react';
import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, LineChart, Line, Cell,
} from 'recharts';
import { hasPerLayerRows, parseFlopsValue } from '../simulationData.ts';
import {
  ChartCard,
  ChartContainer,
  chartTooltipStyle,
  CHART_MARGINS,
  ChartErrorBoundary,
  EmptyChartState,
} from '../shared';


interface PerLayerChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Build a per-layer dataset from perLayer rows */
function buildLayerData(perLayer: PerLayerBreakdownRow[]) {
  return perLayer.map((row) => ({
    key: row.id ?? row.name,
    name: row.name,
    flops: parseFloat((parseFlopsValue(row.flops) / 1e9).toFixed(2)),
    params: row.params ?? 0,
  }));
}

/** Build VRAM data, deriving from params ratio when no per-layer VRAM map */
function buildVramData(analysis: AnalysisResult, perLayer: PerLayerBreakdownRow[]) {
  const vramMap = analysis.perLayerVram ?? {};
  const totalVramBytes = analysis.peakVramBytes || 1;
  const totalParams = analysis.totalParams || 1;

  return perLayer.map((row) => {
    const value = vramMap[row.id] ?? vramMap[row.name];
    if (typeof value === 'number' && value > 0) {
      return { name: row.name, vramMb: value / 1e6 };
    }
    const paramRatio = (row.params ?? 0) / totalParams;
    return { name: row.name, vramMb: parseFloat(((paramRatio * totalVramBytes) / 1e6).toFixed(2)) };
  });
}

/** Build latency data, deriving from FLOPs/throughput when no per-layer latency map */
function buildLatencyData(analysis: AnalysisResult, perLayer: PerLayerBreakdownRow[]) {
  const latencyMap = analysis.perLayerLatency ?? {};
  const totalFlops = analysis.totalFlops || 1;
  const throughputMs = analysis.latencyMs || 100;

  return perLayer.map((row) => {
    const value = latencyMap[row.id] ?? latencyMap[row.name];
    if (typeof value === 'number' && value > 0) {
      return { name: row.name, latency: parseFloat(value.toFixed(2)) };
    }
    const rowFlops = parseFlopsValue(row.flops);
    const flopRatio = rowFlops / totalFlops;
    return { name: row.name, latency: parseFloat((flopRatio * throughputMs).toFixed(2)) };
  });
}

// ─── 3.1 FLOPs per Layer ─────────────────────────────────────────────────────

function FlopsPerLayer({ data }: { data: ReturnType<typeof buildLayerData> }) {
  return (
    <ChartCard title="3.1 — FLOPs per Layer" badge={{ text: 'live', variant: 'live' }}>
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={CHART_MARGINS.bar}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis
              dataKey="name"
              stroke="hsl(var(--muted-foreground))"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              angle={-25}
              textAnchor="end"
              height={72}
              tick={{ fontSize: 11 }}
              interval={0}
            />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={v => `${v}G`} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(v: number) => [`${v} GFLOPs`, 'Compute']} />
            <Bar dataKey="flops" fill="hsl(var(--chart-1))" radius={[3, 3, 0, 0]} barSize={24} />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

// ─── 3.2 VRAM per Layer ──────────────────────────────────────────────────────

function VramByLayer({ data, hasLiveMap }: { data: ReturnType<typeof buildVramData>; hasLiveMap: boolean }) {
  return (
    <ChartCard
      title="3.2 — VRAM per Layer"
      badge={hasLiveMap ? { text: 'live', variant: 'live' } : { text: 'derived', variant: 'derived' }}
    >
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={CHART_MARGINS.area}>
            <defs>
              <linearGradient id="plVram" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.7} />
                <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0.2} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis
              dataKey="name"
              stroke="hsl(var(--muted-foreground))"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              angle={-25}
              textAnchor="end"
              height={72}
              tick={{ fontSize: 11 }}
              interval={0}
            />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={v => `${v}MB`} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(v: number) => [`${v} MB`, 'VRAM']} />
            <Area type="monotone" dataKey="vramMb" name="VRAM" stroke="hsl(var(--chart-1))" fill="url(#plVram)" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

// ─── 3.3 Latency per Layer ──────────────────────────────────────────────────

function LatencyPerLayer({ data, hasLiveMap }: { data: ReturnType<typeof buildLatencyData>; hasLiveMap: boolean }) {
  return (
    <ChartCard
      title="3.3 — Latency per Layer"
      badge={hasLiveMap ? { text: 'live', variant: 'live' } : { text: 'derived', variant: 'derived' }}
    >
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={CHART_MARGINS.line}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis
              dataKey="name"
              stroke="hsl(var(--muted-foreground))"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              angle={-25}
              textAnchor="end"
              height={72}
              tick={{ fontSize: 11 }}
              interval={0}
            />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={v => `${v}ms`} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(v: number) => [`${v} ms`, 'Latency']} />
            <Line type="monotone" dataKey="latency" stroke="hsl(var(--chart-3))" strokeWidth={2} dot={{ r: 3, fill: 'hsl(var(--chart-3))', stroke: 'none' }} />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

// ─── 3.4 Parameters per Layer ─────────────────────────────────────────────

function ParamsPerLayer({ data }: { data: ReturnType<typeof buildLayerData> }) {
  return (
    <ChartCard title="3.4 — Parameters per Layer" badge={{ text: 'live', variant: 'live' }}>
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={CHART_MARGINS.bar}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis
              dataKey="name"
              stroke="hsl(var(--muted-foreground))"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              angle={-25}
              textAnchor="end"
              height={72}
              tick={{ fontSize: 11 }}
              interval={0}
            />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} tickFormatter={v => `${(v / 1e6).toFixed(1)}M`} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(v: number) => [`${v.toLocaleString()} params`, 'Parameters']} />
            <Bar dataKey="params" fill="hsl(var(--chart-3))" radius={[3, 3, 0, 0]} barSize={24} />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

// ─── 3.5 Compute-Memory Ratio ───────────────────────────────────────

function ComputeMemoryRatio({ merged }: { merged: { name: string; flops: number; vramMb: number; ratio: number }[] }) {
  return (
    <ChartCard title="3.5 — Compute-Memory Ratio (FLOPs/MB)" badge={{ text: 'derived', variant: 'derived' }}>
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={merged} margin={CHART_MARGINS.bar}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis
              dataKey="name"
              stroke="hsl(var(--muted-foreground))"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              angle={-25}
              textAnchor="end"
              height={72}
              tick={{ fontSize: 11 }}
              interval={0}
            />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(v: number, n: string) => [v.toFixed(2), n === 'flops' ? 'GFLOPs' : n === 'vramMb' ? 'MB' : 'Ratio']} />
            <Bar dataKey="flops" name="GFLOPs" fill="hsl(var(--chart-1))" radius={[3, 3, 0, 0]} barSize={12} />
            <Bar dataKey="ratio" name="FLOPs/MB" fill="hsl(var(--chart-3))" radius={[3, 3, 0, 0]} barSize={12} />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

// ─── 3.6 Layer Score (Efficiency Index) ─────────────────────────────
// (3.6 renamed — 3.6 Bandwidth per Layer removed as it was duplicate of 3.2)

function LayerScore({ data }: { data: ReturnType<typeof buildLayerData> }) {
  const scored = data.map(d => {
    const maxP = Math.max(...data.map(x => x.params), 1);
    const maxF = Math.max(...data.map(x => x.flops), 1);
    const efficiency = parseFloat((((d.flops / maxF) / (d.params / maxP + 0.01)) * 100).toFixed(1));
    return { name: d.name, score: Math.min(efficiency, 100) };
  });

  return (
    <ChartCard title="3.6 — Layer Efficiency Score" badge={{ text: 'derived', variant: 'derived' }}>
      <ChartContainer>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={scored} margin={CHART_MARGINS.bar}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" vertical={false} />
            <XAxis
              dataKey="name"
              stroke="hsl(var(--muted-foreground))"
              fontSize={11}
              tickLine={false}
              axisLine={false}
              angle={-25}
              textAnchor="end"
              height={72}
              tick={{ fontSize: 11 }}
              interval={0}
            />
            <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={v => `${v}%`} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(v: number) => [`${v}%`, 'Efficiency']} />
            <Bar dataKey="score" fill="hsl(var(--chart-2))" radius={[3, 3, 0, 0]} barSize={24}>
              {scored.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={
                    entry.score >= 50
                      ? 'hsl(var(--chart-2))'
                      : entry.score >= 25
                        ? 'hsl(var(--chart-3))'
                        : 'hsl(var(--chart-4))'
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

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

  const data = buildLayerData(perLayer ?? []);
  const vramData = buildVramData(analysis!, perLayer ?? []);
  const latencyData = buildLatencyData(analysis!, perLayer ?? []);

  const hasLiveVramMap = Object.keys(analysis?.perLayerVram ?? {}).length > 0;
  const hasLiveLatencyMap = Object.keys(analysis?.perLayerLatency ?? {}).length > 0;

  const merged = data.map(d => {
    const v = vramData.find(v => v.name === d.name);
    return {
      name: d.name,
      flops: d.flops,
      vramMb: v?.vramMb ?? 0,
      ratio: v?.vramMb ? parseFloat((d.flops / v.vramMb).toFixed(2)) : 0,
    };
  });

  return (
    <ChartErrorBoundary>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-primary" />
          <h2 className="text-base font-semibold">Per Layer Breakdown</h2>
        </div>

        {/* Row 1: FLOPs · VRAM */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <FlopsPerLayer data={data} />
          <VramByLayer data={vramData} hasLiveMap={hasLiveVramMap} />
        </div>

        {/* Row 2: Latency · Params */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <LatencyPerLayer data={latencyData} hasLiveMap={hasLiveLatencyMap} />
          <ParamsPerLayer data={data} />
        </div>

        {/* Row 3: Compute-Memory Ratio · Layer Score */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ComputeMemoryRatio merged={merged} />
          <LayerScore data={data} />
        </div>
      </div>
    </ChartErrorBoundary>
  );
}
