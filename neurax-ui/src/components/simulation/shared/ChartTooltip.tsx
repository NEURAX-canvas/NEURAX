import { type ReactNode } from 'react';

export interface ChartTooltipContentProps {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number; color?: string; dataKey?: string }>;
  label?: string;
  formatter?: (value: number) => string;
  labelFormatter?: (label: string) => string;
  children?: ReactNode;
}

const tooltipContentStyle: React.CSSProperties = {
  backgroundColor: 'hsl(var(--popover))',
  border: '1px solid hsl(var(--border))',
  borderRadius: '8px',
  fontSize: '11px',
  boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
  padding: '8px 10px',
  outline: 'none',
};

export function chartTooltipStyle(): Record<string, string> {
  return {
    backgroundColor: 'hsl(var(--popover))',
    border: '1px solid hsl(var(--border))',
    borderRadius: '8px',
    fontSize: '11px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
    padding: '8px 10px',
    outline: 'none',
  };
}

/**
 * The one dot a line or area chart actually shows — on the point under the
 * cursor, not on every sample. Every `<Line>`/`<Area>` in the app sets
 * `dot={false}` (a mark per point is noise on a dense trace) but left the
 * hover point at recharts' small, unstyled default, so the one moment that
 * should read as "here — exactly this value, right now" looked identical
 * to the rest of the line. A ring in the surface color plus a solid center
 * in the series' own color is what makes it land as emphasis, not just a
 * slightly bigger dot.
 */
export function chartActiveDot(colorVar: string = 'var(--chart-1)') {
  return {
    r: 5,
    strokeWidth: 2,
    stroke: 'hsl(var(--card))',
    fill: colorVar,
  };
}

export function ChartTooltipContent({
  active,
  payload,
  label,
  formatter,
  labelFormatter,
  children,
}: ChartTooltipContentProps) {
  if (!active || !payload || payload.length === 0) return null;

  return (
    <div style={tooltipContentStyle}>
      {label && (
        <div className="text-[10px] font-medium text-muted-foreground mb-1.5 pb-1.5 border-b border-border">
          {labelFormatter ? labelFormatter(label) : label}
        </div>
      )}
      {children ?? (
        <div className="space-y-1">
          {payload.map((entry, idx) => (
            <div key={idx} className="flex items-center gap-2">
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ backgroundColor: entry.color ?? 'var(--chart-1)' }}
              />
              <span className="text-[10px] text-foreground/80">{entry.name ?? entry.dataKey}</span>
              <span className="text-[11px] font-semibold font-mono text-foreground ml-auto">
                {formatter ? formatter(entry.value ?? 0) : entry.value ?? 0}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
