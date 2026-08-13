import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Label,
} from 'recharts';
import { BarChart3, PieChart as PieChartIcon, Cpu, Zap, DollarSign, Layers, Info } from 'lucide-react';
import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  SIMULATION_COLORS,
  formatBytes,
  formatCompactNumber,
} from '../simulationData.ts';
import {
  ChartCard,
  DonutRing,
  StatCard,
  ChartContainer,
  chartTooltipStyle,
  EmptyChartState,
  ChartErrorBoundary,
  CHART_MARGINS,
} from '../shared/index.ts';


interface Props {
  analysis?: AnalysisResult;
  /** Per-layer breakdown, which is where the real parameter counts live. */
  perLayer?: PerLayerBreakdownRow[];
}

function topOps(analysis: AnalysisResult, n: number = 6) {
  return Object.entries(analysis.opsDistribution ?? {})
    .sort(([, a], [, b]) => b - a)
    .slice(0, n)
    .map(([name, value]) => ({ name, value }));
}

const PIE_COLORS = [
  SIMULATION_COLORS.blue,
  SIMULATION_COLORS.amber,
  SIMULATION_COLORS.red,
  SIMULATION_COLORS.teal,
  SIMULATION_COLORS.violet,
  SIMULATION_COLORS.cyan,
];

/* ─── Model Size (Parameters) ─── */
//
// This charted `opsDistribution` — the operation-type histogram — under a title
// promising parameters, and wrote "ops" in the centre of a donut labelled
// "Model Size (Parameters)". The real per-layer counts arrive in `perLayer`.
function ModelSizeDonut({ analysis, perLayer }: { analysis: AnalysisResult; perLayer?: PerLayerBreakdownRow[] }) {
  const layers = (perLayer ?? [])
    .filter((row) => (row.params ?? 0) > 0)
    .sort((a, b) => (b.params ?? 0) - (a.params ?? 0));

  if (layers.length === 0) {
    return (
      <ChartCard title="Model Size (Parameters)">
        <EmptyChartState
          icon={PieChartIcon}
          title="No parameter data"
          description="Run analysis to see how parameters are distributed across layers."
        />
      </ChartCard>
    );
  }

  // Keep the largest contributors distinct and fold the tail into one slice, so
  // the palette is never cycled past the steps it was validated for.
  const TOP = 5;
  const head = layers.slice(0, TOP);
  const tail = layers.slice(TOP);
  const data = head.map((row, i) => ({
    name: row.name,
    value: row.params ?? 0,
    color: PIE_COLORS[i % PIE_COLORS.length],
  }));
  if (tail.length > 0) {
    data.push({
      name: `Other (${tail.length} layers)`,
      value: tail.reduce((sum, row) => sum + (row.params ?? 0), 0),
      color: PIE_COLORS[TOP % PIE_COLORS.length],
    });
  }

  const total = analysis.totalParams > 0
    ? analysis.totalParams
    : data.reduce((sum, d) => sum + d.value, 0);

  return (
    <ChartCard title="Model Size (Parameters)">
      <ChartContainer minH={200}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={80} paddingAngle={2} dataKey="value">
              <Label
                content={({ viewBox }) => {
                  if (!viewBox || !('cx' in viewBox)) return null;
                  const vb = viewBox as { cx?: number; cy?: number };
                  return (
                    <text x={vb.cx} y={vb.cy} textAnchor="middle" dominantBaseline="middle">
                      <tspan x={vb.cx} dy="-0.5em" className="fill-foreground text-lg font-bold font-mono">
                        {formatCompactNumber(total)}
                      </tspan>
                      <tspan x={vb.cx} dy="1.4em" className="fill-muted-foreground text-[9px] uppercase tracking-wider">
                        params
                      </tspan>
                    </text>
                  );
                }}
              />
              {data.map((entry, idx) => (
                <Cell key={idx} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={chartTooltipStyle()}
              formatter={(value: number) => [formatCompactNumber(value), 'Parameters']}
            />
          </PieChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

/* ─── 2.2 FLOPs by Op Type ─── */
function FlopsByOp({ analysis }: { analysis: AnalysisResult }) {
  const ops = topOps(analysis, 6);
  if (ops.length === 0) {
    return (
      <ChartCard title="FLOPs by Op Type">
        <EmptyChartState icon={BarChart3} title="No FLOPs data" description="Run analysis to see FLOPs distribution." />
      </ChartCard>
    );
  }
  const totalFlops = analysis.totalFlops || 1;
  const data = ops.map((op, i) => ({
    name: op.name.length > 14 ? op.name.slice(0, 12) + '…' : op.name,
    value: ((op.value / Object.values(analysis.opsDistribution ?? {}).reduce((a, b) => a + b, 0)) * totalFlops) / 1e9,
    color: PIE_COLORS[i % PIE_COLORS.length],
  }));
  return (
    <ChartCard title="FLOPs by Op Type">
      <ChartContainer minH={200}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={CHART_MARGINS.barHorizontal}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
            <XAxis type="number" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v >= 1 ? v.toFixed(0) : v.toFixed(1)}G`} />
            <YAxis type="category" dataKey="name" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} width={100} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [`${value >= 1 ? value.toFixed(0) : value.toFixed(1)} GFLOPs`, 'FLOPs']} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {data.map((entry, idx) => (
                <Cell key={idx} fill={entry.color} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

/* ─── 2.3 Latency Breakdown ─── */
function LatencyBreakdown({ analysis }: { analysis: AnalysisResult }) {
  const phases = analysis.compilation?.phase_timeline;
  if (!phases || phases.length === 0) {
    return (
      <ChartCard title="Latency Breakdown">
        <EmptyChartState icon={BarChart3} title="No phase timeline" description="Enable profiling during compilation." />
      </ChartCard>
    );
  }
  const data = phases.map((p) => ({ name: p.name, ms: p.duration_ms }));
  return (
    <ChartCard title="Latency Breakdown">
      <ChartContainer minH={200}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={CHART_MARGINS.bar}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="name" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v: number) => `${v.toFixed(0)}ms`} />
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [`${value.toFixed(1)}ms`, 'Duration']} />
            <Bar dataKey="ms" fill="var(--chart-3)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

/* ─── 2.4 Key Stats Strip ─── */
function KeyStatsStrip({ analysis }: { analysis: AnalysisResult }) {
  return (
    <ChartCard title="Key Stats">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard icon={<Cpu className="w-3 h-3" />} label="Parameters" value={formatCompactNumber(analysis.totalParams)} sublabel={`${analysis.numLayers} layers`} />
        <StatCard icon={<Zap className="w-3 h-3" />} label="FLOPs" value={formatCompactNumber(analysis.totalFlops)} sublabel={`${formatCompactNumber(analysis.totalOperations)} ops`} />
        <StatCard icon={<Layers className="w-3 h-3" />} label="VRAM" value={formatBytes(analysis.peakVramBytes)} sublabel={analysis.gpuMemoryGb ? `${analysis.gpuMemoryGb}GB GPU` : undefined} />
        <StatCard
          icon={<DollarSign className="w-3 h-3" />}
          label="Latency"
          value={analysis.latencyMs != null ? `${analysis.latencyMs.toFixed(1)}ms` : '—'}
          trend={analysis.totalParams > 1e9 ? 'down' : undefined}
          variant={analysis.latencyMs != null && analysis.latencyMs > 100 ? 'warning' : 'default'}
        />
      </div>
    </ChartCard>
  );
}

/* ─── 2.5 Confidence Score ─── */
function ConfidenceScore({ analysis }: { analysis: AnalysisResult }) {
  const score = Math.round((analysis.confidenceScore ?? 0) * 100);
  return (
    <ChartCard title="Confidence Score">
      <div className="flex items-center justify-center h-full">
        <DonutRing value={score} centerLabel={`${score}`} centerSublabel="/100" />
      </div>
    </ChartCard>
  );
}

/* ─── 2.6 Hardware Fit Score ─── */
function HardwareFitScore({ analysis }: { analysis: AnalysisResult }) {
  // Both halves of this score must come from the analysis. Substituting
  // constants for missing VRAM or utilisation produced a confident-looking
  // score that described nothing.
  const hasVram = (analysis.gpuMemoryGb ?? 0) > 0 && analysis.peakVramBytes > 0;
  const hasUtilisation = analysis.gpuUtilization != null;

  if (!hasVram || !hasUtilisation) {
    return (
      <ChartCard title="Hardware Fit Score">
        <EmptyChartState
          icon={Info}
          title="Not enough hardware data"
          description="Set a target GPU and run an analysis to score the fit."
        />
      </ChartCard>
    );
  }

  const vramRatio = analysis.peakVramBytes / (analysis.gpuMemoryGb! * 1024 ** 3);
  const fitScore = Math.round(Math.max(0, Math.min(100, (1 - vramRatio + 0.3) * 100)));
  const utilScore = Math.round(analysis.gpuUtilization! * 100);
  const overall = Math.round((fitScore + utilScore) / 2);

  return (
    <ChartCard title="Hardware Fit Score">
      <div className="flex flex-col items-center justify-center h-full gap-2">
        <DonutRing value={overall} centerLabel={`${overall}`} centerSublabel="/100" size={100} />
        <div className="flex gap-4 mt-1">
          <StatCard label="Fit" value={`${fitScore}%`} variant={fitScore > 70 ? 'success' : fitScore > 40 ? 'warning' : 'danger'} />
          <StatCard label="Util." value={`${utilScore}%`} variant={utilScore > 60 ? 'success' : 'info'} />
        </div>
      </div>
    </ChartCard>
  );
}

/* ─── 2.7 Cost Summary ─── */
function CostSummary({ analysis }: { analysis: AnalysisResult }) {
  const flopsCost = Math.max((analysis.totalFlops || 0) * 2e-11, 0);
  const memCost = Math.max((analysis.peakVramBytes || 0) * 5e-12, 0);
  const latCost = Math.max(((analysis.latencyMs ?? 0) / 1000) * 3e-10 * (analysis.totalFlops || 0), 0);
  const total = flopsCost + memCost + latCost || 1;

  const items = [
    { label: 'FLOPs', value: flopsCost, pct: flopsCost / total, color: 'var(--chart-1)' },
    { label: 'VRAM', value: memCost, pct: memCost / total, color: 'var(--chart-3)' },
    { label: 'Latency', value: latCost, pct: latCost / total, color: 'var(--chart-5)' },
  ];

  return (
    <ChartCard title="Cost Summary (Estimated)">
      <div className="flex flex-col gap-2 h-full justify-center">
        <div className="flex h-6 w-full rounded-full overflow-hidden bg-secondary border border-border">
          {items.map((item, idx) => (
            <div
              key={idx}
              className="h-full transition-all"
              style={{ width: `${item.pct * 100}%`, backgroundColor: item.color, minWidth: item.pct > 0.01 ? 4 : 0 }}
              title={`${item.label}: $${item.value.toFixed(6)}`}
            />
          ))}
        </div>
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1">
          {items.map((item, idx) => (
            <div key={idx} className="flex items-center gap-1.5 text-[10px]">
              <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
              <span className="text-muted-foreground">{item.label}</span>
              <span className="font-mono font-medium">${item.value.toFixed(4)}</span>
            </div>
          ))}
        </div>
        <div className="mt-1 pt-2 border-t border-border flex justify-between text-[11px]">
          <span className="text-muted-foreground">Estimated total</span>
          <span className="font-mono font-bold">${total.toFixed(4)}</span>
        </div>
      </div>
    </ChartCard>
  );
}

/* ─── 2.8 Dialect Distribution ─── */
function DialectDistribution({ analysis }: { analysis: AnalysisResult }) {
  const ops = topOps(analysis);
  if (ops.length === 0) {
    return (
      <ChartCard title="Dialect Distribution">
        <EmptyChartState icon={PieChartIcon} title="No ops data" description="Run analysis to see dialect distribution." />
      </ChartCard>
    );
  }
  const buckets: Record<string, number> = {};
  ops.forEach((op) => {
    const lc = op.name.toLowerCase();
    if (lc.includes('attn') || lc.includes('flash')) buckets['Attention'] = (buckets['Attention'] ?? 0) + op.value;
    else if (lc.includes('linear') || lc.includes('dense')) buckets['Dense'] = (buckets['Dense'] ?? 0) + op.value;
    else if (lc.includes('conv') || lc.includes('unet')) buckets['Conv'] = (buckets['Conv'] ?? 0) + op.value;
    else if (lc.includes('norm') || lc.includes('batchnorm') || lc.includes('layernorm')) buckets['Normalize'] = (buckets['Normalize'] ?? 0) + op.value;
    else if (lc.includes('embed')) buckets['Embed'] = (buckets['Embed'] ?? 0) + op.value;
    else buckets['Other'] = (buckets['Other'] ?? 0) + op.value;
  });
  const data = Object.entries(buckets)
    .sort(([, a], [, b]) => b - a)
    .map(([name, value], i) => ({ name, value, color: PIE_COLORS[i % PIE_COLORS.length] }));
  return (
    <ChartCard title="Dialect Distribution">
      <ChartContainer minH={200}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={data} cx="50%" cy="50%" innerRadius={50} outerRadius={75} paddingAngle={2} dataKey="value">
              {data.map((entry, idx) => (
                <Cell key={idx} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [formatCompactNumber(value), 'Count']} />
          </PieChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

/* ─── Grid Layout ─── */
function Row({ children, cols = 3 }: { children: React.ReactNode; cols?: number }) {
  const colClass = `grid grid-cols-1 lg:grid-cols-${cols} gap-6`;
  return <div className={colClass}>{children}</div>;
}

export function GlobalResultsCharts({ analysis, perLayer }: Props) {
  if (!analysis) return null;

  return (
    <ChartErrorBoundary name="Global Results">
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-indigo-500" />
          <h2 className="text-lg font-semibold">Global Results — Synthesis Overview</h2>
        </div>

        <Row cols={3}>
          <ModelSizeDonut analysis={analysis} perLayer={perLayer} />
          <FlopsByOp analysis={analysis} />
          <LatencyBreakdown analysis={analysis} />
        </Row>

        <KeyStatsStrip analysis={analysis} />

        <Row cols={3}>
          <ConfidenceScore analysis={analysis} />
          <HardwareFitScore analysis={analysis} />
          <CostSummary analysis={analysis} />
        </Row>

        <Row cols={3}>
          <DialectDistribution analysis={analysis} />
          <div /> {/* spacer */}
          <div /> {/* spacer */}
        </Row>
      </div>
    </ChartErrorBoundary>
  );
}
