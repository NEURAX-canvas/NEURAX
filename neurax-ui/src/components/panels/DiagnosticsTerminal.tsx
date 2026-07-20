import { useState, useMemo, useRef, useEffect } from 'react';
import {
  AlertTriangle,
  XCircle,
  Info,
  Lightbulb,
  Terminal,
  ChevronUp,
  ChevronDown,
  GripHorizontal,
  RefreshCw,
  Bug,
  Layers3,
} from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import { Button } from '@/components/ui/button.tsx';
import { cn } from '@/lib/utils.ts';

// ─── Types ────────────────────────────────────────────────────────

interface DiagnosticsTerminalProps {
  analysis?: AnalysisResult;
  /** Height in pixels of the terminal panel */
  height: number;
  onHeightChange: (height: number) => void;
  /** Whether the terminal is maximized / hidden */
  minimized: boolean;
  onMinimizedChange: (minimized: boolean) => void;
}

type TerminalTab = 'diagnostics' | 'warnings' | 'recommendations' | 'compilation';

// ─── Helpers ──────────────────────────────────────────────────────

function formatTimestamp(): string {
  const now = new Date();
  return now.toLocaleTimeString('en-US', { hour12: false });
}

function severityIcon(severity: string) {
  const s = severity.toLowerCase();
  if (s === 'error' || s === 'critical') return XCircle;
  if (s === 'warning' || s === 'warn') return AlertTriangle;
  return Info;
}

function severityColor(severity: string): string {
  const s = severity.toLowerCase();
  if (s === 'error' || s === 'critical') return 'text-red-400';
  if (s === 'warning' || s === 'warn') return 'text-amber-400';
  return 'text-emerald-400';
}

function severityBg(severity: string): string {
  const s = severity.toLowerCase();
  if (s === 'error' || s === 'critical') return 'bg-red-950/30 border-red-900/40';
  if (s === 'warning' || s === 'warn') return 'bg-amber-950/30 border-amber-900/40';
  return 'bg-emerald-950/20 border-emerald-900/30';
}

// ─── Main Component ───────────────────────────────────────────────

export function DiagnosticsTerminal({
  analysis,
  height,
  onHeightChange,
  minimized,
  onMinimizedChange,
}: DiagnosticsTerminalProps) {
  const [activeTab, setActiveTab] = useState<TerminalTab>('diagnostics');
  const isResizing = useRef(false);
  const startY = useRef(0);
  const startHeight = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll when new content arrives
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [analysis, activeTab]);

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    startY.current = e.clientY;
    startHeight.current = height;

    const handleMouseMove = (ev: MouseEvent) => {
      if (!isResizing.current) return;
      const delta = ev.clientY - startY.current;
      const newHeight = Math.max(48, Math.min(500, startHeight.current + delta));
      onHeightChange(newHeight);
    };

    const handleMouseUp = () => {
      isResizing.current = false;
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const hasData = !!analysis;

  // ── Derived data ──────────────────────────────────────────────

  const diagnostics = useMemo(() => analysis?.diagnostics ?? [], [analysis]);
  const warnings = useMemo(() => analysis?.reportWarnings ?? [], [analysis]);
  const recommendations = useMemo(() => analysis?.recommendations ?? [], [analysis]);
  const compilationPhase = useMemo(() => analysis?.compilation?.current_phase ?? 'Idle', [analysis]);
  const phaseTimeline = useMemo(() => analysis?.compilation?.phase_timeline ?? [], [analysis]);

  const tabs: { id: TerminalTab; label: string; count: number; icon: React.ElementType }[] = [
    { id: 'diagnostics', label: 'Diagnostics', count: diagnostics.length, icon: Bug },
    { id: 'warnings', label: 'Warnings', count: warnings.length, icon: AlertTriangle },
    { id: 'recommendations', label: 'Recommendations', count: recommendations.length, icon: Lightbulb },
    { id: 'compilation', label: 'Compilation', count: phaseTimeline.length, icon: Layers3 },
  ];

  return (
    <div
      className={cn(
        'bg-[#0d1117] border-t border-border/70 flex flex-col',
        minimized ? 'h-0 overflow-hidden border-t-0' : '',
      )}
      style={minimized ? undefined : { height }}
    >
      {/* ── Resize Handle ── */}
      {!minimized && (
        <div
          className="h-2 cursor-row-resize flex items-center justify-center hover:bg-primary/5 transition-colors shrink-0 group"
          onMouseDown={handleResizeStart}
        >
          <GripHorizontal className="w-4 h-3 text-muted-foreground/30 group-hover:text-primary/50" />
        </div>
      )}

      {/* ── Terminal Header ── */}
      <div className="h-9 px-3 flex items-center justify-between bg-[#161b22] border-b border-border/60 shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Diagnostics Terminal
            </span>
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-0.5 ml-2">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  'flex items-center gap-1 px-2 py-1 rounded text-[10px] font-medium transition-colors',
                  activeTab === tab.id
                    ? 'bg-primary/15 text-primary border border-primary/25'
                    : 'text-muted-foreground/60 hover:text-muted-foreground hover:bg-secondary/20',
                )}
              >
                <tab.icon className="w-3 h-3" />
                {tab.label}
                {tab.count > 0 && (
                  <span className={cn(
                    'ml-0.5 px-1 rounded text-[9px] font-mono',
                    activeTab === tab.id ? 'bg-primary/20 text-primary' : 'bg-muted-foreground/10 text-muted-foreground/70',
                  )}>
                    {tab.count}
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {hasData && (
            <span className="text-[9px] text-emerald-500/60 font-mono">{formatTimestamp()}</span>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="w-5 h-5 text-muted-foreground/50 hover:text-muted-foreground"
            onClick={() => onMinimizedChange(!minimized)}
          >
            {minimized ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </Button>
        </div>
      </div>

      {/* ── Terminal Content ── */}
      {!minimized && (
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto scrollbar-thin p-2 font-mono text-[11px] leading-relaxed"
        >
          {!hasData && (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground/30">
              <Terminal className="w-8 h-8 mb-2" />
              <span className="text-[11px]">Run analysis to see diagnostics</span>
            </div>
          )}

          {hasData && activeTab === 'diagnostics' && (
            <div className="space-y-1">
              {diagnostics.length === 0 && (
                <div className="text-emerald-500/50 py-4 text-center">
                  <Info className="w-4 h-4 inline mr-1" />
                  No diagnostics — model is clean.
                </div>
              )}
              {diagnostics.map((d, i) => {
                const SevIcon = severityIcon(d.severity);
                const sevColor = severityColor(d.severity);
                const sevBg = severityBg(d.severity);
                return (
                  <div
                    key={i}
                    className={cn('rounded border px-3 py-2', sevBg)}
                  >
                    <div className="flex items-start gap-2">
                      <SevIcon className={cn('w-3.5 h-3.5 mt-0.5 shrink-0', sevColor)} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className={cn('text-[10px] uppercase font-bold', sevColor)}>
                            {d.severity}
                          </span>
                          {d.code && (
                            <span className="text-[9px] text-muted-foreground/50 font-mono">[{d.code}]</span>
                          )}
                          {d.category && (
                            <span className="text-[9px] text-muted-foreground/40">{d.category}</span>
                          )}
                        </div>
                        <div className="text-gray-300 mt-0.5">{d.message}</div>
                        {d.suggestion && (
                          <div className="text-emerald-400/70 text-[10px] mt-1">
                            💡 {d.suggestion}
                          </div>
                        )}
                        {d.layer_id && (
                          <div className="text-blue-400/50 text-[9px] mt-0.5">
                            layer: {d.layer_id}
                          </div>
                        )}
                        {d.precision_impact !== undefined && (
                          <div className="text-amber-400/50 text-[9px] mt-0.5">
                            precision impact: {d.precision_impact}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {hasData && activeTab === 'warnings' && (
            <div className="space-y-1">
              {warnings.length === 0 && (
                <div className="text-emerald-500/50 py-4 text-center">
                  <Info className="w-4 h-4 inline mr-1" />
                  No warnings — report is clean.
                </div>
              )}
              {warnings.map((w, i) => (
                <div
                  key={i}
                  className="rounded border border-amber-900/40 bg-amber-950/30 px-3 py-2 flex items-start gap-2"
                >
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
                  <span className="text-gray-300">{w}</span>
                </div>
              ))}
            </div>
          )}

          {hasData && activeTab === 'recommendations' && (
            <div className="space-y-1">
              {recommendations.length === 0 && (
                <div className="text-emerald-500/50 py-4 text-center">
                  <Lightbulb className="w-4 h-4 inline mr-1" />
                  No recommendations available.
                </div>
              )}
              {recommendations.map((rec, i) => {
                const prioColor = rec.priority === 'high'
                  ? 'text-orange-400 border-orange-900/40 bg-orange-950/20'
                  : rec.priority === 'medium'
                    ? 'text-yellow-400 border-yellow-900/40 bg-yellow-950/20'
                    : 'text-blue-400 border-blue-900/40 bg-blue-950/20';
                return (
                  <div
                    key={i}
                    className={cn('rounded border px-3 py-2', prioColor)}
                  >
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <Lightbulb className="w-3.5 h-3.5 shrink-0" />
                        <span className="text-gray-200 text-[11px] font-medium">{rec.title}</span>
                      </div>
                      <span className="text-[9px] uppercase font-bold opacity-70">{rec.priority}</span>
                    </div>
                    <div className="text-gray-400 text-[10px] mt-1">{rec.description}</div>
                    {rec.impact && (
                      <div className="text-emerald-400/60 text-[9px] mt-0.5">Impact: {rec.impact}</div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {hasData && activeTab === 'compilation' && (
            <div className="space-y-1">
              <div className="flex items-center gap-2 px-1 py-1 text-emerald-400/70 text-[10px]">
                <RefreshCw className="w-3 h-3" />
                <span>Current phase: <span className="text-emerald-300 font-semibold">{compilationPhase}</span></span>
              </div>
              {phaseTimeline.length === 0 && (
                <div className="text-muted-foreground/40 py-4 text-center text-[10px]">
                  No phase timeline data.
                </div>
              )}
              {phaseTimeline.map((phase, i) => {
                const statusColor = phase.status === 'completed'
                  ? 'text-emerald-400'
                  : phase.status === 'running'
                    ? 'text-blue-400'
                    : phase.status === 'error'
                      ? 'text-red-400'
                      : 'text-muted-foreground';
                const statusDot = phase.status === 'completed' ? '●'
                  : phase.status === 'running' ? '▶'
                    : phase.status === 'error' ? '✕'
                      : '○';
                return (
                  <div
                    key={i}
                    className="flex items-center gap-2 px-2 py-1 rounded hover:bg-secondary/10 transition-colors"
                  >
                    <span className={cn('text-[9px] font-mono', statusColor)}>{statusDot}</span>
                    <span className="text-gray-300 text-[10px] flex-1">{phase.name}</span>
                    <span className={cn('text-[9px] uppercase font-mono', statusColor)}>{phase.status}</span>
                    <span className="text-muted-foreground/40 text-[9px] font-mono">
                      {phase.duration_ms >= 1000
                        ? `${(phase.duration_ms / 1000).toFixed(1)}s`
                        : `${phase.duration_ms.toFixed(0)}ms`}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* Footer with live status */}
          {hasData && (
            <div className="sticky bottom-0 flex items-center gap-3 px-2 py-1.5 mt-2 border-t border-border/30 bg-[#0d1117] text-[9px] text-muted-foreground/40">
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500/60 inline-block" />
                NEURAX compiler
              </span>
              <span>
                {diagnostics.length} diagnostic{diagnostics.length !== 1 ? 's' : ''}
              </span>
              <span>
                {warnings.length} warning{warnings.length !== 1 ? 's' : ''}
              </span>
              {analysis?.confidenceScore && (
                <span className="ml-auto">
                  confidence: {(analysis.confidenceScore * 100).toFixed(0)}%
                </span>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default DiagnosticsTerminal;
