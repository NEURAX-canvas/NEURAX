import { type ComponentType, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface EmptyChartStateProps {
  icon?: ComponentType<{ className?: string }>;
  title: string;
  description: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyChartState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: EmptyChartStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center h-full text-muted-foreground',
        className,
      )}
    >
      <div className="flex flex-col items-center gap-2 max-w-[200px] text-center">
        {Icon && <Icon className="w-8 h-8 opacity-40" />}
        <span className="text-[11px] font-semibold">{title}</span>
        <span className="text-[10px] leading-relaxed opacity-70">{description}</span>
        {action && <div className="mt-1">{action}</div>}
      </div>
    </div>
  );
}
