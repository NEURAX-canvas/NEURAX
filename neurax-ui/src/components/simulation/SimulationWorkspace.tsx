import { useState, lazy, Suspense } from 'react';
import {
  BarChart3, Layers, HardDrive,
  GitCompare, Target, TrendingUp, Bug,
} from 'lucide-react';
import { CanvasNode, Connection, AnalysisResult, PerLayerBreakdownRow, Warning } from '@/types/architecture.ts';
import { ChartSkeleton } from './shared';

const GlobalResultsCharts = lazy(() => import('./categories/GlobalResultsCharts.tsx').then(m => ({ default: m.GlobalResultsCharts })));
const PerLayerCharts = lazy(() => import('./categories/PerLayerCharts.tsx').then(m => ({ default: m.PerLayerCharts })));
const MemoryCharts = lazy(() => import('./categories/MemoryCharts.tsx').then(m => ({ default: m.MemoryCharts })));
const ComparisonCharts = lazy(() => import('./categories/ComparisonCharts.tsx').then(m => ({ default: m.ComparisonCharts })));
const OptimizationCharts = lazy(() => import('./categories/OptimizationCharts.tsx').then(m => ({ default: m.OptimizationCharts })));
const TrainingCharts = lazy(() => import('./categories/TrainingCharts.tsx').then(m => ({ default: m.TrainingCharts })));
const DebuggingCharts = lazy(() => import('./categories/DebuggingCharts.tsx').then(m => ({ default: m.DebuggingCharts })));

export type SimulationCategoryId =
  | 'overview' | 'perlayer' | 'memory' | 'training'
  | 'optimization' | 'comparison' | 'diagnostics';

/**
 * Analysis views, ordered as they are read rather than as they were written.
 *
 * The sequence follows the question a designer actually works through: what is
 * this model, where does its cost sit, will it fit, what will it cost to train,
 * how could it be faster, how does it compare, and what is wrong with it.
 *
 * `chartCount` is asserted against the number of cards each module renders, so
 * the badges cannot drift out of date the way they had (Comparison advertised
 * nine charts and rendered four).
 */
export const SIMULATION_CATEGORIES = [
  {
    id: 'overview',
    label: 'Overview',
    hint: 'Headline size, cost and hardware fit',
    icon: BarChart3,
    chartCount: 5,
  },
  {
    id: 'perlayer',
    label: 'Per Layer',
    hint: 'Where the parameters, FLOPs and latency sit',
    icon: Layers,
    chartCount: 2,
  },
  {
    id: 'memory',
    label: 'Memory',
    hint: 'VRAM over time, peak breakdown and OOM risk',
    icon: HardDrive,
    chartCount: 8,
  },
  {
    id: 'training',
    label: 'Training',
    hint: 'Time, cost and carbon for the training run',
    icon: TrendingUp,
    chartCount: 2,
  },
  {
    id: 'optimization',
    label: 'Optimization',
    hint: 'Roofline, bottlenecks and what to change first',
    icon: Target,
    chartCount: 3,
  },
  {
    id: 'comparison',
    label: 'Comparison',
    hint: 'This design against other hardware and precisions',
    icon: GitCompare,
    chartCount: 5,
  },
  {
    id: 'diagnostics',
    label: 'Diagnostics',
    hint: 'Warnings, unsupported ops and confidence',
    icon: Bug,
    chartCount: 6,
  },
] as const satisfies ReadonlyArray<{
  id: SimulationCategoryId;
  label: string;
  hint: string;
  icon: typeof Bug;
  chartCount: number;
}>;

interface SimulationWorkspaceProps {
  nodes: CanvasNode[];
  connections: Connection[];
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
  warnings?: Warning[];
  topology?: Record<string, unknown>;
}

export function SimulationWorkspace({
  nodes,
  analysis,
  perLayer,
  warnings,
  topology,
}: SimulationWorkspaceProps) {
  const [activeCategory, setActiveCategory] = useState<SimulationCategoryId>('overview');

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      <div className="border-b border-border bg-card px-4 py-2">
        <div
          className="flex items-center gap-1 overflow-x-auto scrollbar-thin"
          role="tablist"
          aria-label="Analysis views"
        >
          {SIMULATION_CATEGORIES.map((category) => {
            const Icon = category.icon;
            const isActive = activeCategory === category.id;
            return (
              <button
                key={category.id}
                type="button"
                role="tab"
                aria-selected={isActive}
                title={category.hint}
                onClick={() => setActiveCategory(category.id)}
                className={`flex items-center gap-1 sm:gap-2 px-2 sm:px-3 py-1.5 text-xs font-medium rounded-md transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                }`}
              >
                <Icon className="w-3.5 h-3.5 shrink-0" />
                <span className="hidden sm:inline">{category.label}</span>
                <span
                  className={`text-[10px] px-1 sm:px-1.5 py-0.5 rounded-full hidden sm:inline ${
                    isActive ? 'bg-primary-foreground/20' : 'bg-muted'
                  }`}
                >
                  {category.chartCount}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Only the active category is mounted, so its charts load on demand. */}
      <div className="flex-1 overflow-auto p-4 scrollbar-thin" role="tabpanel">
        <Suspense fallback={<ChartSkeleton variant="stats-grid" />}>
          {activeCategory === 'overview' && <GlobalResultsCharts analysis={analysis} />}
          {activeCategory === 'perlayer' && <PerLayerCharts analysis={analysis} perLayer={perLayer} />}
          {activeCategory === 'memory' && <MemoryCharts analysis={analysis} />}
          {activeCategory === 'training' && <TrainingCharts analysis={analysis} />}
          {activeCategory === 'optimization' && (
            <OptimizationCharts analysis={analysis} perLayer={perLayer} />
          )}
          {activeCategory === 'comparison' && (
            <ComparisonCharts analysis={analysis} topology={topology} />
          )}
          {activeCategory === 'diagnostics' && (
            <DebuggingCharts
              analysis={analysis}
              perLayer={perLayer}
              warnings={warnings}
              nodes={nodes}
            />
          )}
        </Suspense>
      </div>
    </div>
  );
}
