import { useMemo, useState, useEffect } from 'react';
import { Zap, Info, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line,
} from 'recharts';
import {
  hasCompilationProgress,
  hasLivePartialMetrics,
  hasLiveThroughputTrace,
  hasPhaseTimeline,
  normalizePhaseStatus,
  normalizeSeverity,
} from '../simulationData.ts';


interface RealTimeChartsProps {
  analysis?: AnalysisResult;
}

/** Delay rendering charts until after mount so Recharts can measure the container. */
function useIsMounted() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => { setMounted(true); }, []);
  return mounted;
}

export function RealTimeCharts({ analysis }: RealTimeChartsProps) {
  // ── Guard: nothing to render ──────────────────────────────────────
  if (!analysis) return null;

  const { compilation, live_trace, diagnostics, confidenceScore } = analysis;

  // ── Stable memoised data references ───────────────────────────────
  const partialMetricsData = useMemo(
    () => live_trace?.partial_metrics?.map(([time, value]) => ({ time, value })) ?? [],
    [live_trace?.partial_metrics],
  );
  const throughputData = useMemo(
    () => live_trace?.throughput_trace?.map(([time, value]) => ({ time, value })) ?? [],
    [live_trace?.throughput_trace],
  );

  const hasProgress = useMemo(() => hasCompilationProgress(analysis), [analysis]);
  const hasTimeline = useMemo(() => hasPhaseTimeline(analysis), [analysis]);
  const hasPartialMetrics = useMemo(() => hasLivePartialMetrics(analysis), [analysis]);
  const hasThroughputTrace = useMemo(() => hasLiveThroughputTrace(analysis), [analysis]);

  // ── Render charts only after mount (avoids Recharts 0×0 measure) ──
  const isMounted = useIsMounted();

  // ── Template ──────────────────────────────────────────────────────
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Zap className="w-5 h-5 text-amber-500 fill-amber-500" />
        <h2 className="text-lg font-semibold">Real-Time — Live Compilation</h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── 1.1 Global Progress ──────────────────────────────────── */}
        <div
          key="section-progress"
          className={`panel-section p-4 bg-card/30 border-primary/5 ${!hasTimeline ? 'lg:col-span-2' : ''}${!hasProgress ? ' hidden' : ''}`}
        >
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">1.1 — Global Progress</h3>
          </div>
          <div className="space-y-2">
            <div className="text-[10px] text-muted-foreground uppercase">
              Phase: <span className="text-foreground font-medium">{compilation?.current_phase || 'Idle'}</span>
            </div>
            <div className="relative h-2 w-full bg-secondary rounded-full overflow-hidden">
              <div
                className="absolute top-0 left-0 h-full bg-[#3b82f6] transition-all duration-500"
                style={{ width: `${(compilation?.total_progress || 0) * 100}%` }}
              />
            </div>
            <div className="text-[10px] text-right text-muted-foreground">
              {Math.round((compilation?.total_progress || 0) * 100)}%
            </div>
          </div>
        </div>

        {/* ── 1.2 Timeline des Phases ──────────────────────────────── */}
        <div
          key="section-timeline"
          className={`panel-section p-4 bg-card/30 border-primary/5 ${!hasProgress ? 'lg:col-span-2' : ''}${!hasTimeline ? ' hidden' : ''}`}
        >
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">1.2 — Timeline des Phases</h3>
          <div className="flex h-10 w-full rounded-md overflow-hidden bg-secondary/50 border border-border/50">
            {compilation?.phase_timeline.map((phase, idx) => {
              const totalDuration = compilation.phase_timeline.reduce((acc, p) => acc + p.duration_ms, 0);
              const width = totalDuration > 0 ? (phase.duration_ms / totalDuration) * 100 : 10;
              const status = normalizePhaseStatus(phase.status);
              const color = status === 'completed'
                ? '#10b981'
                : status === 'inprogress'
                  ? '#3b82f6'
                  : status === 'failed'
                    ? '#ef4444'
                    : '#64748b';

              return (
                <div
                  key={idx}
                  className="h-full flex items-center justify-center text-[8px] font-bold text-white px-1 whitespace-nowrap overflow-hidden transition-all duration-300"
                  style={{ width: `${width}%`, backgroundColor: color }}
                  title={`${phase.name}: ${phase.duration_ms.toFixed(1)}ms`}
                >
                  {phase.name}
                </div>
              );
            })}
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            {compilation?.phase_timeline.map((phase, idx) => {
              const status = normalizePhaseStatus(phase.status);
              return (
                <div key={`legend-${idx}`} className="flex items-center gap-1.5">
                  <div
                    className={`w-1.5 h-1.5 rounded-full ${
                      status === 'completed'
                        ? 'bg-[#10b981]'
                        : status === 'inprogress'
                          ? 'bg-[#3b82f6]'
                          : status === 'failed'
                            ? 'bg-red-500'
                            : 'bg-slate-500'
                    }`}
                  />
                  <span className="text-[9px] text-muted-foreground">
                    {phase.name} ({phase.duration_ms.toFixed(0)}ms)
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* ── 1.3 Live Diagnostics Feed ────────────────────────────── */}
        <div
          key="section-diagnostics"
          className="panel-section p-4 bg-card/30 border-primary/5 max-h-[320px] overflow-hidden flex flex-col"
        >
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">1.3 — Live Diagnostics Feed</h3>
          <div className="space-y-3 flex-1 overflow-auto pr-2 scrollbar-thin">
            {diagnostics && diagnostics.length > 0 ? (
              diagnostics.map((diag, idx) => {
                const sev = normalizeSeverity(diag.severity);
                return (
                  <div
                    key={`${diag.code ?? diag.category}-${idx}`}
                    className={`p-3 rounded-md border ${
                      sev === 'critical'
                        ? 'bg-red-500/5 border-red-500/10'
                        : sev === 'warning'
                          ? 'bg-amber-500/5 border-amber-500/10'
                          : sev === 'hint'
                            ? 'bg-emerald-500/5 border-emerald-500/10'
                            : 'bg-blue-500/5 border-blue-500/10'
                    }`}
                  >
                    <div
                      className={`flex items-center gap-2 text-[10px] font-bold mb-1 ${
                        sev === 'critical'
                          ? 'text-red-500'
                          : sev === 'warning'
                            ? 'text-amber-500'
                            : sev === 'hint'
                              ? 'text-emerald-400'
                              : 'text-blue-400'
                      }`}
                    >
                      {sev === 'warning' || sev === 'critical' ? (
                        <AlertTriangle className="w-3 h-3 shrink-0" />
                      ) : (
                        <Info className="w-3 h-3 shrink-0" />
                      )}
                      {diag.category}
                    </div>
                    <p className="text-[10px] text-muted-foreground leading-snug">{diag.message}</p>
                    {diag.suggestion && (
                      <p className="text-[9px] text-primary/60 mt-1 italic">Suggestion: {diag.suggestion}</p>
                    )}
                  </div>
                );
              })
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-muted-foreground opacity-50 space-y-2">
                <CheckCircle2 className="w-8 h-8" />
                <span className="text-[10px]">No issues detected in current pass.</span>
              </div>
            )}
          </div>
        </div>

        {/* ── 1.5 Confidence Score Live ────────────────────────────── */}
        <div
          key="section-confidence"
          className="panel-section p-4 bg-card/30 border-primary/5 flex flex-col items-center justify-center"
        >
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider self-start mb-6">1.5 — Confidence Score Live</h3>
          <div className="flex items-center gap-8">
            <div className="relative w-28 h-28">
              <svg className="w-full h-full transform -rotate-90">
                <circle
                  cx="56"
                  cy="56"
                  r="48"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  className="text-secondary"
                />
                <circle
                  cx="56"
                  cy="56"
                  r="48"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="transparent"
                  strokeDasharray={2 * Math.PI * 48}
                  strokeDashoffset={2 * Math.PI * 48 * (1 - (confidenceScore || 0))}
                  strokeLinecap="round"
                  className={
                    confidenceScore && confidenceScore > 0.8
                      ? 'text-[#10b981]'
                      : confidenceScore && confidenceScore > 0.5
                        ? 'text-amber-500'
                        : 'text-red-500'
                  }
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-3xl font-bold font-mono">{Math.round((confidenceScore || 0) * 100)}</div>
              </div>
            </div>
            <div className="space-y-1">
              <div
                className={`flex items-center gap-1.5 text-xs font-bold ${confidenceScore && confidenceScore > 0.8 ? 'text-[#10b981]' : 'text-amber-500'}`}
              >
                {confidenceScore && confidenceScore > 0.8 ? (
                  <CheckCircle2 className="w-4 h-4" />
                ) : (
                  <AlertTriangle className="w-4 h-4" />
                )}
                {confidenceScore && confidenceScore > 0.8 ? 'Reliable' : 'Approximated'}
              </div>
              <div className="text-[10px] text-muted-foreground">
                Confidence in synthesis results based on topology complexity.
              </div>
            </div>
          </div>
        </div>

        {/* ── 1.4 Partial Metrics (Live) ───────────────────────────── */}
        <div
          key="section-partial-metrics"
          className={`panel-section p-4 bg-card/30 border-primary/5 ${!hasThroughputTrace ? 'lg:col-span-2' : ''}${!hasPartialMetrics ? ' hidden' : ''}`}
        >
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">1.4 — Partial Metrics (Live)</h3>
          <div className="h-48 min-h-[192px] w-full">
            {isMounted ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={192}>
                <AreaChart data={partialMetricsData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorPartial" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                  <XAxis dataKey="time" type="number" stroke="#666" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#666" fontSize={10} tickLine={false} axisLine={false} tickCount={5} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111', border: '1px solid #333', fontSize: '10px' }}
                    itemStyle={{ color: '#f59e0b' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    name="Activity"
                    stroke="#f59e0b"
                    fillOpacity={1}
                    fill="url(#colorPartial)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full w-full flex items-center justify-center text-muted-foreground text-xs">
                Waiting for data…
              </div>
            )}
          </div>
        </div>

        {/* ── 1.6 Throughput Instantané ────────────────────────────── */}
        <div
          key="section-throughput"
          className={`panel-section p-4 bg-card/30 border-primary/5 ${!hasPartialMetrics ? 'lg:col-span-2' : ''}${!hasThroughputTrace ? ' hidden' : ''}`}
        >
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-4">1.6 — Throughput Instantané</h3>
          <div className="h-48 min-h-[192px] w-full">
            {isMounted ? (
              <ResponsiveContainer width="100%" height="100%" minWidth={200} minHeight={192}>
                <LineChart data={throughputData} margin={{ top: 10, right: 10, left: -30, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                  <XAxis dataKey="time" type="number" stroke="#666" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#666" fontSize={10} tickLine={false} axisLine={false} tickCount={3} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#111', border: '1px solid #333', fontSize: '10px' }}
                    itemStyle={{ color: '#10b981' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    name="Tokens/sec"
                    stroke="#10b981"
                    dot={false}
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full w-full flex items-center justify-center text-muted-foreground text-xs">
                Waiting for data…
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
