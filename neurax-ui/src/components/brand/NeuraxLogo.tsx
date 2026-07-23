export interface NeuraxLogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
  variant?: 'mark' | 'full';
}

export const NeuraxLogo = ({
  size = 28,
  className = '',
  showText = true,
  variant = 'full',
}: NeuraxLogoProps) => {
  const mark = (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={{ filter: 'drop-shadow(0 0 12px rgba(99,102,241,0.15))' }}
    >
      <defs>
        <linearGradient id="nx-logo-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#22d3ee" />
        </linearGradient>
        <linearGradient id="nx-logo-grad-light" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#818cf8" />
          <stop offset="100%" stopColor="#67e8f9" />
        </linearGradient>
        <filter id="nx-glow">
          <feGaussianBlur stdDeviation="2" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Outer ring — subtle containing boundary */}
      <circle
        cx="50" cy="50" r="46"
        fill="none"
        stroke="url(#nx-logo-grad)"
        strokeWidth="1.2"
        opacity="0.12"
      />

      {/* Core diamond — the primary geometric mark */}
      <path
        d="M50 16 L84 50 L50 84 L16 50 Z"
        fill="url(#nx-logo-grad)"
        opacity="0.92"
      />

      {/* Diamond border accent */}
      <path
        d="M50 16 L84 50 L50 84 L16 50 Z"
        fill="none"
        stroke="url(#nx-logo-grad-light)"
        strokeWidth="0.5"
        opacity="0.4"
      />

      {/* Inner geometric "N" — left stroke */}
      <path
        d="M35 65 L50 42 L65 65"
        fill="none"
        stroke="white"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity="0.9"
      />

      {/* Inner geometric "N" — right stroke (shorter, crossing) */}
      <path
        d="M35 35 L50 58"
        fill="none"
        stroke="white"
        strokeWidth="2.2"
        strokeLinecap="round"
        opacity="0.55"
      />

      {/* Center neural node */}
      <circle
        cx="50" cy="50" r="3.5"
        fill="#22d3ee"
        filter="url(#nx-glow)"
      />

      {/* Center node inner highlight */}
      <circle
        cx="50" cy="50" r="1.5"
        fill="white"
        opacity="0.8"
      />
    </svg>
  );

  if (variant === 'mark' || !showText) return mark;

  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      {mark}
      <span
        className="font-bold tracking-[-0.03em]"
        style={{
          fontSize: `${size * 0.48}px`,
          background: 'linear-gradient(135deg, #f0f0f5 0%, #818cf8 60%, #22d3ee 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
        }}
      >
        NEURAX
      </span>
    </span>
  );
};
