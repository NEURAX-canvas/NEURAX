import { type ReactNode } from 'react';
import { cn } from '@/lib/utils';

/**
 * Two sizes, not a free-form number. Every chart in Simulation is a
 * `square` tile by default — same width and height as every sibling in its
 * grid row, because a grid's columns are already equal width, so
 * `aspect-square` alone makes every tile the same size with no extra
 * bookkeeping. `wide` is the deliberate exception, reserved for charts a
 * square genuinely can't serve: a log-log Roofline needs room for its axis
 * labels, a per-layer bar chart needs room for names along one edge. Reach
 * for `wide` only when a real reading problem forces it, not by default.
 */
export type ChartCardSize = 'square' | 'wide';

const WIDE_MIN_HEIGHT = 400;

export interface ChartCardProps {
  title: string;
  badge?: {
    text: string;
    variant?: 'live' | 'derived' | 'warning' | 'info';
  };
  className?: string;
  size?: ChartCardSize;
  children: ReactNode;
  action?: ReactNode;
}

const badgeStyles: Record<string, string> = {
  live: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
  derived: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
  warning: 'bg-red-500/10 text-red-500 border-red-500/20',
  info: 'bg-blue-500/10 text-blue-500 border-blue-500/20',
};

export function ChartCard({ title, badge, className, size = 'square', children, action }: ChartCardProps) {
  return (
    <div
      className={cn(
        'panel-section p-4 bg-card/30 border-primary/5 flex flex-col overflow-hidden',
        size === 'square' ? 'aspect-square' : '',
        className,
      )}
      style={size === 'wide' ? { minHeight: WIDE_MIN_HEIGHT } : undefined}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
            {title}
          </h3>
          {badge && (
            <span
              className={cn(
                'text-[9px] font-mono px-1.5 py-0.5 rounded border',
                badgeStyles[badge.variant ?? 'info'],
              )}
            >
              ● {badge.text}
            </span>
          )}
        </div>
        {action && <div className="flex items-center gap-1">{action}</div>}
      </div>
      <div className="flex-1 min-h-0">
        {children}
      </div>
    </div>
  );
}
