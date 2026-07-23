import { useMemo } from 'react';
import { Zap, Info, AlertTriangle, CheckCircle2, Activity } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line,
} from 'recharts';
import {
  normalizePhaseStatus,
  normalizeSeverity,
} from '../simulationData.ts';
import {
  ChartCard,
  DonutRing,
  ChartContainer,
  chartTooltipStyle,
  EmptyChartState,
  ChartErrorBoundary,
  CHART_MARGINS,
} from '../shared/index.ts';


interface RealTimeChartsProps {
  analysis?: AnalysisResult;
}

function ProgressBar({ analysis }: { analysis: AnalysisResult }) {
  const { compilation, confidenceScore } = analysis;
  const progressVal = compilation?.total_progress ?? (confidenceScore > 0 ? confidenceScore * 0.9 : 0.5);
  const currentPhase = compilation?.current_phase ?? (confidenceScore > 0 ? 'Complete' : 'Pending');

  return (
    <ChartCard title="1.1 — Global Progress" badge={compilation?.current_phase ? { text: 'live', variant: 'live' } : undefined}>
      <div className="space-y-2">
        <div className="text-[11px] text-muted-foreground">
          Phase: <span className="text-foreground font-medium">{currentPhase}</span>
        </div>
        <div className="relative h-2.5 w-full bg-secondary rounded-full overflow-hidden">
          <div
            className="absolute top-0 left-0 h-full bg-[#3b82f6] transition-all duration-700 ease-out rounded-full"
            style={{ width: `${Math.round(progressVal * 100)}%` }}
          />
        </div>
        <div className="flex justify-between items-center">
          <span className="text-[10px] text-muted-foreground">Compilation progress</span>
          <span className="text-[11px] font-mono font-bold">{Math.round(progressVal * 100)}%</span>
        </div>
      </div>
    </ChartCard>
  );
}

function TimelinePhases({ analysis }: { analysis: AnalysisResult }) {
  const timelinePhases = analysis.compilation?.phase_timeline;
  const hasLive = timelinePhases && timelinePhases.length > 0;

  // Use real phases or show empty state
  const phases = hasLive ? timelinePhases : null;

  if (!phases) {
    return (
      <ChartCard title="1.2 — Timeline des Phases" badge={{ text: 'estimated', variant: 'derived' }}>
        <EmptyChartState
          icon={Activity}
          title="No compilation data yet"
          description="Run a compilation with profiling to see phase timing."
        />
      </ChartCard>
    );
  }

  const totalDuration = phases.reduce((acc, p) => acc + p.duration_ms, 0);

  return (
    <ChartCard title="1.2 — Timeline des Phases" badge={{ text: 'live', variant: 'live' }}>
      <div className="flex h-10 w-full rounded-md overflow-hidden bg-secondary/50 border border-border/50">
        {phases.map((phase, idx) => {
          const width = totalDuration > 0 ? (phase.duration_ms / totalDuration) * 100 : 10;
          const status = normalizePhaseStatus(phase.status);
          const color = status === 'completed'
            ? 'var(--chart-2)'
            : status === 'inprogress'
              ? 'var(--chart-1)'
              : status === 'failed'
                ? 'var(--chart-4)'
                : '#64748b';
          return (
            <div
              key={idx}
              className="h-full flex items-center justify-center text-[9px] font-bold text-white px-1 whitespace-nowrap overflow-hidden transition-all duration-300"
              style={{ width: `${width}%`, backgroundColor: color }}
              title={`${phase.name}: ${phase.duration_ms.toFixed(1)}ms`}
            >
              {phase.name}
            </div>
          );
        })}
      </div>
      <div className="mt-3 flex flex-wrap gap-3">
        {phases.map((phase, idx) => {
          const status = normalizePhaseStatus(phase.status);
          const colors: Record<string, string> = {
            completed: 'bg-[var(--chart-2)]',
            inprogress: 'bg-[var(--chart-1)]',
            failed: 'bg-[var(--chart-4)]',
            pending: 'bg-slate-500',
          };
          return (
            <div key={`legend-${idx}`} className="flex items-center gap-1.5">
              <div className={`w-1.5 h-1.5 rounded-full ${colors[status] ?? 'bg-slate-500'}`} />
              <span className="text-[10px] text-muted-foreground">
                {phase.name}
                <span className="ml-1 font-mono">({phase.duration_ms.toFixed(0)}ms)</span>
              </span>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}

function LiveDiagnosticsFeed({ analysis }: { analysis: AnalysisResult }) {
  const { diagnostics } = analysis;

  return (
    <ChartCard title="1.3 — Live Diagnostics Feed" className="max-h-[320px] overflow-hidden flex flex-col">
      <div className="flex-1 overflow-auto pr-1 space-y-2 scrollbar-thin">
        {diagnostics && diagnostics.length > 0 ? (
          diagnostics.map((diag, idx) => {
            const sev = normalizeSeverity(diag.severity);
            const severityStyles: Record<string, string> = {
              critical: 'bg-red-500/5 border-red-500/10 text-red-500',
              warning: 'bg-amber-500/5 border-amber-500/10 text-amber-500',
              hint: 'bg-emerald-500/5 border-emerald-500/10 text-emerald-400',
              info: 'bg-blue-500/5 border-blue-500/10 text-blue-400',
            };
            return (
              <div
                key={`${diag.code ?? diag.category}-${idx}`}
                className={`p-3 rounded-md border ${severityStyles[sev] ?? 'bg-secondary/50 border-border'}`}
              >
                <div className={`flex items-center gap-2 text-[10px] font-bold mb-1 ${severityStyles[sev] ?? ''}`}>
                  {sev === 'warning' || sev === 'critical' ? (
                    <AlertTriangle className="w-3 h-3 shrink-0" />
                  ) : (
                    <Info className="w-3 h-3 shrink-0" />
                  )}
                  {diag.category ?? diag.code}
                </div>
                <p className="text-[10px] text-muted-foreground leading-snug">{diag.message}</p>
                {diag.suggestion && (
                  <p className="text-[9px] text-primary/60 mt-1 italic">Suggestion: {diag.suggestion}</p>
                )}
              </div>
            );
          })
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground/50 space-y-2">
            <CheckCircle2 className="w-8 h-8" />
            <span className="text-[11px] font-medium">No issues detected</span>
            <span className="text-[10px] text-center">All diagnostics passed in the current pass.</span>
          </div>
        )}
      </div>
    </ChartCard>
  );
}

function PartialMetrics({ analysis }: { analysis: AnalysisResult }) {
  const cleanMetrics = analysis.live_trace?.partial_metrics;

  const data = useMemo(() => {
    if (cleanMetrics && cleanMetrics.length > 0) {
      return cleanMetrics.map(([time, value]) => ({ time, value }));
    }
    return null;
  }, [cleanMetrics]);

  if (!data) {
    return (
      <ChartCard title="1.4 — Partial Metrics" badge={{ text: 'no data', variant: 'info' }}>
        <EmptyChartState
          icon={Activity}
          title="No live metrics"
          description="Partial metrics appear during active compilation."
        />
      </ChartCard>
    );
  }

  return (
    <ChartCard title="1.4 — Partial Metrics" badge={{ text: 'live', variant: 'live' }}>
      <ChartContainer minH={192}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={CHART_MARGINS.area}>
            <defs>
              <linearGradient id="partialGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--chart-3)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="var(--chart-3)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="time" type="number" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} tickCount={5} />
            <Tooltip contentStyle={chartTooltipStyle()} />
            <Area
              type="monotone"
              dataKey="value"
              name="Activity"
              stroke="var(--chart-3)"
              fillOpacity={1}
              fill="url(#partialGrad)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

function ConfidenceScore({ analysis }: { analysis: AnalysisResult }) {
  const { confidenceScore } = analysis;
  const score = confidenceScore ?? 0;
  const isReliable = score > 0.8;

  return (
    <ChartCard title="1.5 — Confidence Score Live">
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <DonutRing
          value={Math.round(score * 100)}
          size={112}
          centerLabel={`${Math.round(score * 100)}`}
          centerSublabel="/100"
        />
        <div className="flex items-center gap-1.5 text-[11px] font-medium">
          {isReliable ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-[var(--chart-2)]" />
              <span style={{ color: 'var(--chart-2)' }}>Reliable</span>
            </>
          ) : (
            <>
              <AlertTriangle className="w-4 h-4 text-[var(--chart-3)]" />
              <span style={{ color: 'var(--chart-3)' }}>Approximated</span>
            </>
          )}
        </div>
        <span className="text-[10px] text-muted-foreground text-center">
          Confidence in synthesis results based on topology complexity.
        </span>
      </div>
    </ChartCard>
  );
}

function ThroughputChart({ analysis }: { analysis: AnalysisResult }) {
  const cleanTrace = analysis.live_trace?.throughput_trace;

  const data = useMemo(() => {
    if (cleanTrace && cleanTrace.length > 0) {
      return cleanTrace.map(([time, value]) => ({ time, value }));
    }
    return null;
  }, [cleanTrace]);

  if (!data) {
    return (
      <ChartCard title="1.6 — Throughput Instantané" badge={{ text: 'no data', variant: 'info' }}>
        <EmptyChartState
          icon={Activity}
          title="No throughput data"
          description="Throughput traces appear during active inference or training."
        />
      </ChartCard>
    );
  }

  return (
    <ChartCard title="1.6 — Throughput Instantané" badge={{ text: 'live', variant: 'live' }}>
      <ChartContainer minH={192}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={CHART_MARGINS.line}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
            <XAxis dataKey="time" type="number" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
            <YAxis stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} tickCount={3} />
            <Tooltip contentStyle={chartTooltipStyle()} />
            <Line
              type="monotone"
              dataKey="value"
              name="Tokens/sec"
              stroke="var(--chart-2)"
              dot={false}
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartContainer>
    </ChartCard>
  );
}

export function RealTimeCharts({ analysis }: RealTimeChartsProps) {
  if (!analysis) return null;

  return (
    <ChartErrorBoundary name="Real-Time">
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-amber-500 fill-amber-500" />
          <h2 className="text-lg font-semibold">Real-Time — Live Compilation</h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ProgressBar analysis={analysis} />
          <TimelinePhases analysis={analysis} />
          <LiveDiagnosticsFeed analysis={analysis} />
          <PartialMetrics analysis={analysis} />
          <ConfidenceScore analysis={analysis} />
          <ThroughputChart analysis={analysis} />
        </div>
      </div>
    </ChartErrorBoundary>
  );
}
