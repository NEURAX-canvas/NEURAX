import { NeuraxLogo } from '@/components/brand/NeuraxLogo.tsx';

export function Logo({ className = "w-8 h-8" }: { className?: string }) {
  return <NeuraxLogo size={32} variant="mark" showText={false} className={className} />;
}
