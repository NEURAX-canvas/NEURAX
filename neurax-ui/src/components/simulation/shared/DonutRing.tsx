import { cn } from '@/lib/utils';

export interface DonutRingProps {
  value: number;
  size?: number;
  strokeWidth?: number;
  color?: string;
  centerLabel?: string;
  centerSublabel?: string;
  className?: string;
}

function autoColor(value: number): string {
  if (value >= 80) return '#10b981';
  if (value >= 50) return '#f59e0b';
  return '#ef4444';
}

export function DonutRing({
  value,
  size = 112,
  strokeWidth = 8,
  color,
  centerLabel,
  centerSublabel,
  className,
}: DonutRingProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - Math.min(value, 100) / 100);
  const strokeColor = color ?? autoColor(value);

  return (
    <div className={cn('relative inline-flex items-center justify-center', className)}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="transparent"
          className="text-secondary"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={strokeColor}
          strokeWidth={strokeWidth}
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-700 ease-out"
        />
      </svg>
      {(centerLabel || centerSublabel) && (
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          {centerLabel && (
            <span className="text-xl font-bold font-mono leading-tight">{centerLabel}</span>
          )}
          {centerSublabel && (
            <span className="text-[9px] text-muted-foreground uppercase tracking-wider">
              {centerSublabel}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
