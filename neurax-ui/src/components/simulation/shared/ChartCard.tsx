import { type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface ChartCardProps {
  title: string;
  badge?: {
    text: string;
    variant?: 'live' | 'derived' | 'warning' | 'info';
  };
  className?: string;
  minH?: number;
  children: ReactNode;
  action?: ReactNode;
}

const badgeStyles: Record<string, string> = {
  live: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20',
  derived: 'bg-amber-500/10 text-amber-500 border-amber-500/20',
  warning: 'bg-red-500/10 text-red-500 border-red-500/20',
  info: 'bg-blue-500/10 text-blue-500 border-blue-500/20',
};

export function ChartCard({ title, badge, className, minH = 224, children, action }: ChartCardProps) {
  return (
    <div
      className={cn(
        'panel-section p-4 bg-card/30 border-primary/5 flex flex-col overflow-hidden',
        className,
      )}
      style={{ minHeight: minH }}
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
