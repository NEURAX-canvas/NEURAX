import { useState, useEffect } from 'react';

interface NeuraxAtomLogoProps {
  size?: number;
  className?: string;
  animated?: boolean;
  showText?: boolean;
}

export const NeuraxAtomLogo = ({
  size = 28,
  className = '',
  animated = true,
  showText = true,
}: NeuraxAtomLogoProps) => {
  const [phase, setPhase] = useState<'drop' | 'hold' | 'rise'>('drop');

  useEffect(() => {
    if (!animated) return;
    const cycle = async () => {
      // Drop
      setPhase('drop');
      await sleep(600);
      // Hold at bottom
      setPhase('hold');
      await sleep(400);
      // Rise back up
      setPhase('rise');
      await sleep(600);
      // Brief pause at top
      await sleep(500);
      // Restart
      setPhase('drop');
    };
    let running = true;
    const run = async () => {
      while (running) {
        await cycle();
      }
    };
    run();
    return () => { running = false; };
  }, [animated]);

  const nucleusY = phase === 'drop' ? 32 : phase === 'rise' ? 20 : 32;
  const ringY = phase === 'drop' ? 32 : phase === 'rise' ? 20 : 32;
  const opacity = phase === 'hold' ? 0.6 : 1;
  const scale = phase === 'drop' ? 0.85 : phase === 'rise' ? 1 : 0.85;

  const logo = (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 40"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      style={{
        overflow: 'visible',
        filter: 'drop-shadow(0 0 8px rgba(99,102,241,0.3))',
      }}
    >
      <defs>
        <radialGradient id="nucleus-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#818cf8" stopOpacity="1" />
          <stop offset="60%" stopColor="#6366f1" stopOpacity="0.9" />
          <stop offset="100%" stopColor="#4f46e5" stopOpacity="0" />
        </radialGradient>
        <filter id="atom-glow">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <linearGradient id="ring-grad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="50%" stopColor="#06b6d4" />
          <stop offset="100%" stopColor="#6366f1" />
        </linearGradient>
      </defs>

      {/* Outer glow */}
      <circle cx="20" cy="20" r="19" fill="url(#nucleus-glow)" fillOpacity="0.06" />

      {/* Orbital ring — animated vertical drop/rise */}
      <g
        style={{
          transform: `translateY(${(nucleusY - 20) * (size / 40)}px) scaleY(${scale})`,
          transition: 'transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.4s ease',
          opacity,
          transformOrigin: 'center',
        }}
      >
        {/* Ring 1 — main (indigo-cyan gradient) */}
        <ellipse
          cx="20" cy="20"
          rx="17" ry="7"
          stroke="url(#ring-grad)"
          strokeWidth="1.4"
          fill="none"
          opacity="0.9"
          filter="url(#atom-glow)"
        />

        {/* Ring 2 — violet (rotated 60°) */}
        <ellipse
          cx="20" cy="20"
          rx="17" ry="7"
          stroke="#a78bfa"
          strokeWidth="1.2"
          fill="none"
          opacity="0.7"
          transform="rotate(60 20 20)"
          filter="url(#atom-glow)"
        />

        {/* Ring 3 — cyan (rotated 120°) */}
        <ellipse
          cx="20" cy="20"
          rx="17" ry="7"
          stroke="#22d3ee"
          strokeWidth="1.2"
          fill="none"
          opacity="0.65"
          transform="rotate(120 20 20)"
          filter="url(#atom-glow)"
        />

        {/* Ring 4 — subtle trail ring that appears during drop */}
        {phase === 'drop' && (
          <ellipse
            cx="20" cy="20"
            rx="17" ry="7"
            stroke="url(#ring-grad)"
            strokeWidth="0.6"
            fill="none"
            opacity="0.25"
            transform="rotate(30 20 20)"
            style={{
              transform: `rotate(30 20 20) translateY(${12 * (size / 40)}px)`,
              transition: 'transform 0.6s ease',
            }}
          />
        )}
      </g>

      {/* Nucleus outer glow */}
      <circle
        cx="20" cy={nucleusY}
        r="5.5"
        fill="url(#nucleus-glow)"
        opacity="0.35"
        style={{
          transition: 'all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)',
        }}
      />

      {/* Nucleus core */}
      <circle
        cx="20" cy={nucleusY}
        r="3.5"
        fill="#0a0a0f"
        stroke="#818cf8"
        strokeWidth="1.2"
        filter="url(#atom-glow)"
        style={{
          transition: 'all 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)',
        }}
      />

      {/* Waveform pulse in nucleus */}
      <g
        style={{
          transform: `translateY(${(nucleusY - 20) * (size / 40)}px)`,
          transition: 'transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)',
        }}
      >
        <path
          d="M17.5 20 L18.5 18.5 L19.5 21.5 L20.5 19 L21.5 20.5 L22.5 20"
          stroke="#06b6d4"
          strokeWidth="0.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
          opacity="0.8"
        />
      </g>

      {/* Electron dots on ring */}
      <g
        style={{
          transform: `translateY(${(ringY - 20) * (size / 40)}px) scaleY(${scale})`,
          transition: 'transform 0.6s cubic-bezier(0.34, 1.56, 0.64, 1), opacity 0.4s ease',
          opacity,
          transformOrigin: 'center',
        }}
      >
        <circle cx="20" cy="13" r="1.2" fill="#818cf8" filter="url(#atom-glow)" />
        <circle cx="20" cy="27" r="1.2" fill="#22d3ee" filter="url(#atom-glow)" />
        <circle cx="7" cy="20" r="1" fill="#a78bfa" filter="url(#atom-glow)" />
        <circle cx="33" cy="20" r="1" fill="#818cf8" filter="url(#atom-glow)" />
      </g>
    </svg>
  );

  if (!showText) return logo;

  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      {logo}
      <span
        style={{
          fontSize: `${size * 0.5}px`,
          fontWeight: 700,
          letterSpacing: '-0.5px',
          background: 'linear-gradient(135deg, #e8e8ed 0%, #818cf8 50%, #22d3ee 100%)',
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

function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
