import { cn } from '@/lib/utils';

export interface ChartLegendEntry {
  name: string;
  value: number;
  color: string;
  /** Pre-formatted display value (e.g. "6.6B", "43%"). Falls back to the
   *  raw number when omitted — callers with a real unit should always pass
   *  this so the legend doesn't show an unformatted number next to a chart
   *  whose tooltip and axes are formatted. */
  formattedValue?: string;
}

export interface ChartLegendProps {
  entries: ChartLegendEntry[];
  /** Percent-of-total next to each value — on by default since that's what
   *  makes a pie/donut legend actually useful (which slice is 60% vs 6%). */
  showPercent?: boolean;
  className?: string;
}

/**
 * The one legend every pie/donut in Simulation should use. Some charts
 * hand-rolled their own version of this (Peak VRAM Breakdown, Diagnostic
 * Severity); others (Model Size, OpKind Distribution) had none at all —
 * same chart type, inconsistent treatment. This is the shared version.
 *
 * Colors are passed in as theme tokens (`hsl(var(--chart-1))` etc, already
 * the convention across simulationData.ts's SIMULATION_COLORS) so the
 * legend swatches — and the slices they describe — follow whichever theme
 * is active rather than a fixed palette.
 */
export function ChartLegend({ entries, showPercent = true, className }: ChartLegendProps) {
  const total = entries.reduce((sum, e) => sum + e.value, 0) || 1;

  return (
    <div className={cn('flex flex-wrap justify-center gap-x-4 gap-y-1.5 mt-2', className)}>
      {entries.map((entry) => (
        <div key={entry.name} className="flex items-center gap-1.5 text-[11px]">
          <span
            className="w-2.5 h-2.5 rounded-full shrink-0"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name}</span>
          <span className="font-mono font-medium text-foreground">
            {entry.formattedValue ?? entry.value}
            {showPercent && (
              <span className="text-muted-foreground">
                {' '}({((entry.value / total) * 100).toFixed(0)}%)
              </span>
            )}
          </span>
        </div>
      ))}
    </div>
  );
}
