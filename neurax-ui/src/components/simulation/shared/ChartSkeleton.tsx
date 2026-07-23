import { cn } from '@/lib/utils';

export interface ChartSkeletonProps {
  variant?: 'bar' | 'line' | 'donut' | 'area' | 'stats-grid' | 'table';
  rows?: number;
  className?: string;
}

export function ChartSkeleton({ variant = 'bar', rows = 4, className }: ChartSkeletonProps) {
  if (variant === 'donut') {
    return (
      <div className={cn('flex items-center justify-center h-full', className)}>
        <div className="relative w-28 h-28">
          <div className="absolute inset-0 rounded-full bg-secondary animate-pulse" />
          <div className="absolute inset-4 rounded-full bg-card" />
        </div>
      </div>
    );
  }

  if (variant === 'stats-grid') {
    return (
      <div className={cn('grid grid-cols-2 gap-3 h-full', className)}>
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="p-3 rounded-lg bg-secondary/50 border border-border animate-pulse">
            <div className="h-3 w-16 bg-secondary rounded mb-2" />
            <div className="h-6 w-20 bg-secondary rounded" />
          </div>
        ))}
      </div>
    );
  }

  if (variant === 'table') {
    return (
      <div className={cn('space-y-2', className)}>
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="flex gap-3">
            <div className="h-4 w-1/4 bg-secondary rounded animate-pulse" />
            <div className="h-4 w-1/6 bg-secondary rounded animate-pulse" />
            <div className="h-4 w-1/6 bg-secondary rounded animate-pulse" />
          </div>
        ))}
      </div>
    );
  }

  // bar / line / area — show a generic chart skeleton
  return (
    <div className={cn('flex items-end gap-1.5 h-full pt-4', className)}>
      {Array.from({ length: 8 }).map((_, i) => (
        <div
          key={i}
          className="flex-1 bg-secondary rounded-t animate-pulse"
          style={{ height: `${30 + Math.random() * 70}%` }}
        />
      ))}
    </div>
  );
}
