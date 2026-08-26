import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
} from 'recharts';
import { BarChart3, PieChart as PieChartIcon, Cpu, Zap, DollarSign, Layers, Info, Activity } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  SIMULATION_COLORS,
  formatBytes,
  formatCompactNumber,
} from '../simulationData.ts';
import {
  ChartCard,
  ChartContainer,
  ChartLegend,
  DonutRing,
  StatCard,
  chartTooltipStyle,
  EmptyChartState,
  ChartErrorBoundary,
} from '../shared/index.ts';


interface Props {
  analysis?: AnalysisResult;
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

/* ─── Key Stats header ───
 *
 * Not a chart, deliberately: five single numbers with nothing to compare
 * them against have no distribution, trend, or proportion for a chart to
 * show — forcing one would be exactly the "does this metric actually need
 * a graph to be seen correctly" question the rest of this tab now enforces.
 * This stays as a plain, compact readout so a reader always knows which
 * model and hardware target the charts below are describing.
 */
function KeyStatsStrip({ analysis }: { analysis: AnalysisResult }) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
      <StatCard icon={<Cpu className="w-3 h-3" />} label="Parameters" value={formatCompactNumber(analysis.totalParams)} sublabel={`${analysis.numLayers} layers`} />
      <StatCard icon={<Zap className="w-3 h-3" />} label="FLOPs" value={formatCompactNumber(analysis.totalFlops)} sublabel={`${formatCompactNumber(analysis.totalOperations)} ops`} />
      <StatCard icon={<Layers className="w-3 h-3" />} label="VRAM" value={formatBytes(analysis.peakVramBytes)} sublabel={analysis.gpuMemoryGb ? `${analysis.gpuMemoryGb}GB GPU` : undefined} />
      <StatCard
        icon={<Activity className="w-3 h-3" />}
        label="Latency"
        value={analysis.latencyMs != null ? `${analysis.latencyMs.toFixed(1)}ms` : '—'}
        trend={analysis.totalParams > 1e9 ? 'down' : undefined}
        variant={analysis.latencyMs != null && analysis.latencyMs > 100 ? 'warning' : 'default'}
      />
      {analysis.trainingCostUsd > 0 && (
        <StatCard
          icon={<DollarSign className="w-3 h-3" />}
          label="Train Cost"
          value={`$${analysis.trainingCostUsd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
        />
      )}
    </div>
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

/* ─── Operation Distribution ───
 *
 * The only chart of its kind left in the whole Simulation module. It used to
 * exist three times — here as "FLOPs by Op Type" (a bar chart extrapolating
 * FLOPs share from op counts) and "Dialect Distribution" (a donut bucketing
 * the same op counts into semantic categories), and again in Diagnostics as
 * "OpKind Distribution" (an unbucketed top-6 donut of the same numbers). All
 * three read the same `analysis.opsDistribution` map. One legible chart,
 * bucketed into the categories that actually distinguish an architecture
 * (attention vs dense vs conv vs normalize vs embed), replaces all three.
 */
function OperationDistribution({ analysis }: { analysis: AnalysisResult }) {
  const ops = topOps(analysis, 20);
  if (ops.length === 0) {
    return (
      <ChartCard title="Operation Distribution">
        <EmptyChartState icon={PieChartIcon} title="No ops data" description="Run analysis to see the operation-type breakdown." />
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
  const total = Object.values(buckets).reduce((a, b) => a + b, 0) || 1;
  const data = Object.entries(buckets)
    .sort(([, a], [, b]) => b - a)
    .map(([name, value], i) => ({ name, value, color: PIE_COLORS[i % PIE_COLORS.length] }));
  return (
    <ChartCard title="Operation Distribution">
      <div className="h-full flex flex-col">
        <ChartContainer className="flex-1 min-h-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={2} dataKey="value">
                {data.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => [`${((value / total) * 100).toFixed(1)}%`, 'Share of ops']} />
            </PieChart>
          </ResponsiveContainer>
        </ChartContainer>
        <ChartLegend entries={data.map((d) => ({ ...d, formattedValue: `${((d.value / total) * 100).toFixed(1)}%` }))} showPercent={false} />
      </div>
    </ChartCard>
  );
}

/* ─── Grid Layout ───
 *
 * Tailwind's scanner only picks up class names that appear literally in
 * source — `grid-cols-${cols}` built at runtime never matches any literal
 * text it scans, so the responsive column class was silently never
 * generated and every "3-column" row rendered as a single column, full
 * width, one card per screen. This is what actually produced the
 * disproportionate, oversized cards.
 */
const ROW_COL_CLASSES: Record<number, string> = {
  2: 'grid grid-cols-1 lg:grid-cols-2 gap-6',
  3: 'grid grid-cols-1 lg:grid-cols-3 gap-6',
  4: 'grid grid-cols-1 lg:grid-cols-4 gap-6',
};
function Row({ children, cols = 3 }: { children: React.ReactNode; cols?: number }) {
  return <div className={ROW_COL_CLASSES[cols] ?? ROW_COL_CLASSES[3]}>{children}</div>;
}

export function GlobalResultsCharts({ analysis }: Props) {
  if (!analysis) return null;

  return (
    <ChartErrorBoundary name="Global Results">
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-indigo-500" />
          <h2 className="text-lg font-semibold">Global Results — Synthesis Overview</h2>
        </div>

        <KeyStatsStrip analysis={analysis} />

        <Row cols={3}>
          <ConfidenceScore analysis={analysis} />
          <HardwareFitScore analysis={analysis} />
          <OperationDistribution analysis={analysis} />
        </Row>
      </div>
    </ChartErrorBoundary>
  );
}
