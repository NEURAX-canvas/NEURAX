import { type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface StatCardProps {
  icon?: ReactNode;
  label: string;
  value: string;
  sublabel?: string;
  trend?: 'up' | 'down' | 'neutral';
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info';
  className?: string;
}

const variantStyles: Record<string, string> = {
  default: 'bg-secondary/50 border-border',
  success: 'bg-success/5 border-success/20',
  warning: 'bg-warning/5 border-warning/20',
  danger: 'bg-destructive/5 border-destructive/20',
  info: 'bg-info/5 border-info/20',
};

const trendColors: Record<string, string> = {
  up: 'text-success',
  down: 'text-destructive',
  neutral: 'text-muted-foreground',
};

const trendArrows: Record<string, string> = {
  up: '↑',
  down: '↓',
  neutral: '→',
};

export function StatCard({
  icon,
  label,
  value,
  sublabel,
  trend,
  variant = 'default',
  className,
}: StatCardProps) {
  return (
    <div
      className={cn(
        'p-3 rounded-lg border flex flex-col gap-1',
        variantStyles[variant],
        className,
      )}
    >
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-muted-foreground uppercase tracking-wider font-medium">
          {label}
        </span>
        {icon && <span className="text-muted-foreground/60">{icon}</span>}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="text-lg font-bold font-mono">{value}</span>
        {trend && (
          <span className={cn('text-xs font-mono', trendColors[trend])}>
            {trendArrows[trend]}
          </span>
        )}
      </div>
      {sublabel && (
        <span className="text-[9px] text-muted-foreground">{sublabel}</span>
      )}
    </div>
  );
}
