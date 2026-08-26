import { type ReactNode, useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

/**
 * Standard margins for Recharts charts — NO negative values.
 * Use `margin` prop on your chart directly.
 */
export const CHART_MARGINS = {
  bar: { top: 8, right: 8, bottom: 24, left: 8 },
  barHorizontal: { top: 8, right: 8, bottom: 8, left: 72 },
  line: { top: 8, right: 8, bottom: 8, left: 8 },
  area: { top: 8, right: 8, bottom: 8, left: 8 },
  composed: { top: 8, right: 16, bottom: 16, left: 8 },
  pie: { top: 8, right: 8, bottom: 8, left: 8 },
} as const;

export type ChartVariant = keyof typeof CHART_MARGINS;

interface ChartContainerProps {
  className?: string;
  children: ReactNode;
  /**
   * Explicit pixel height, for the rare chart that isn't sitting directly
   * inside a `ChartCard` (or needs to be taller than its card). Most charts
   * should omit this and let the container fill the `ChartCard`'s own
   * `flex-1 min-h-0` content slot — one place sets the size (`ChartCard`'s
   * `size` prop) instead of two numbers that had to be kept in sync by hand.
   */
  minH?: number;
}

export function ChartContainer({
  className,
  children,
  minH,
}: ChartContainerProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div
      className={cn('w-full', minH ? undefined : 'h-full', className)}
      style={minH ? { minHeight: minH, height: minH } : undefined}
    >
      {mounted ? (
        children
      ) : (
        <div className="h-full w-full flex items-center justify-center text-muted-foreground text-[10px]">
          Loading chart…
        </div>
      )}
    </div>
  );
}
