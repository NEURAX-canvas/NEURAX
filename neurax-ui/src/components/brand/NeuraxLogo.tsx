export interface NeuraxLogoProps {
  size?: number;
  className?: string;
  showText?: boolean;
  variant?: 'mark' | 'full';
}

const spiderPath = `
  M 36 18
  C 28 18, 22 24, 22 32
  C 22 40, 28 46, 36 46
  C 40 46, 44 44, 46 41
  L 54 41
  C 56 44, 60 46, 64 46
  C 72 46, 78 40, 78 32
  C 78 24, 72 18, 64 18
  Z
`;

const abdomenPath = `
  M 38 42
  C 30 42, 24 52, 26 64
  C 28 76, 36 84, 46 86
  L 54 86
  C 64 84, 72 76, 74 64
  C 76 52, 70 42, 62 42
  Z
`;

function Leg({ transform }: { transform: string }) {
  return (
    <path
      d="M 0 0 L 8 -4 L 16 2 L 22 -2"
      fill="none"
      stroke="#d65d0e"
      strokeWidth="2.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      transform={transform}
    />
  );
}

function renderSpiderMark(size: number, className: string) {
  const s = size;
  const gs = Math.max(s, 24);

  return (
    <svg
      width={s || gs}
      height={s || gs}
      viewBox="-25 -25 150 150"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={{ filter: 'drop-shadow(0 0 8px rgba(215,153,33,0.2))' }}
    >
      <defs>
        <radialGradient id="spider-body-grad" cx="50%" cy="40%" r="60%">
          <stop offset="0%" stopColor="#d79921" />
          <stop offset="70%" stopColor="#b57614" />
          <stop offset="100%" stopColor="#8f6d11" />
        </radialGradient>
        <radialGradient id="spider-abdomen-grad" cx="50%" cy="30%" r="65%">
          <stop offset="0%" stopColor="#d79921" stopOpacity="0.9" />
          <stop offset="60%" stopColor="#b57614" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#7c5f0e" />
        </radialGradient>
        <filter id="spider-eye-glow">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="spider-glow">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <g transform="translate(50, 50) scale(1.5) translate(-50, -50)">
      {/* Web thread hanging from spinnerets */}
      <path
        d="M 50 86 C 50 90, 48 94, 46 98"
        stroke="#83a598"
        strokeWidth="0.6"
        strokeLinecap="round"
        opacity="0.4"
        fill="none"
      />
      <circle cx="46" cy="98" r="0.8" fill="#83a598" opacity="0.6" />

      {/* Legs - back pair */}
      <Leg transform="translate(32, 32) rotate(-140)" />
      <Leg transform="translate(68, 32) rotate(-40)" />

      {/* Legs - middle pair */}
      <Leg transform="translate(30, 38) rotate(175)" />
      <Leg transform="translate(70, 38) rotate(5)" />

      {/* Legs - front pair */}
      <Leg transform="translate(32, 44) rotate(135)" />
      <Leg transform="translate(68, 44) rotate(45)" />

      {/* Cephalothorax */}
      <path d={spiderPath} fill="url(#spider-body-grad)" stroke="#b57614" strokeWidth="0.8" opacity="0.95" />

      {/* Abdomen */}
      <path d={abdomenPath} fill="url(#spider-abdomen-grad)" stroke="#8f6d11" strokeWidth="0.6" opacity="0.95" />

      {/* Dorsal stripe */}
      <path
        d="M 50 18 C 50 24, 50 34, 50 46"
        stroke="#d79921"
        strokeWidth="1.2"
        strokeLinecap="round"
        opacity="0.3"
        fill="none"
      />

      {/* Eyes */}
      <circle cx="42" cy="28" r="3" fill="#fb4934" filter="url(#spider-eye-glow)" />
      <circle cx="58" cy="28" r="3" fill="#fb4934" filter="url(#spider-eye-glow)" />
      <circle cx="42" cy="28" r="1.2" fill="#ff6a5c" />
      <circle cx="58" cy="28" r="1.2" fill="#ff6a5c" />

      {/* Chelicerae (fangs) */}
      <path d="M 46 32 C 44 36, 42 38, 40 40" stroke="#b57614" strokeWidth="1.2" strokeLinecap="round" fill="none" />
      <path d="M 54 32 C 56 36, 58 38, 60 40" stroke="#b57614" strokeWidth="1.2" strokeLinecap="round" fill="none" />

      {/* Spinnerets */}
      <ellipse cx="50" cy="84" rx="4" ry="2" fill="#8f6d11" opacity="0.8" />
      <circle cx="48" cy="84" r="1" fill="#b57614" />
      <circle cx="52" cy="84" r="1" fill="#b57614" />

      {/* Neural glow node */}
      <circle cx="50" cy="50" r="8" fill="none" stroke="#d79921" strokeWidth="0.3" opacity="0.15" />
      <circle cx="50" cy="50" r="14" fill="none" stroke="#d79921" strokeWidth="0.2" opacity="0.08" />
      </g>
    </svg>
  );
}

export const NeuraxLogo = ({
  size = 28,
  className = '',
  showText = true,
  variant = 'full',
}: NeuraxLogoProps) => {
  const mark = renderSpiderMark(size, className);

  if (variant === 'mark' || !showText) return mark;

  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      {mark}
      <span
        className="font-bold tracking-[-0.03em]"
        style={{
          fontSize: `${size * 0.48}px`,
          background: 'linear-gradient(135deg, #d79921 0%, #d65d0e 60%, #fe8019 100%)',
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
