import { Check, X, Minus } from 'lucide-react';

const C = {
  bg: '#1d2021',
  card: '#282828',
  border: '#3c3836',
  text: '#ebdbb2',
  muted: '#a89984',
  faint: '#7c6f64',
  accent: '#d79921',
  green: '#98971a',
  red: '#cc241d',
};

interface ComparisonFeature {
  name: string;
  neurax: 'full' | 'partial' | 'none' | string;
  iree: 'full' | 'partial' | 'none' | string;
  openxla: 'full' | 'partial' | 'none' | string;
  tvm: 'full' | 'partial' | 'none' | string;
}

const FEATURES: ComparisonFeature[] = [
  {
    name: 'Pre-training Cost Analysis',
    neurax: 'full',
    iree: 'none',
    openxla: 'none',
    tvm: 'none'
  },
  {
    name: 'Architecture Families',
    neurax: '11 families',
    iree: 'partial',
    openxla: 'partial',
    tvm: 'partial'
  },
  {
    name: 'Memory Prediction (VRAM)',
    neurax: 'full',
    iree: 'partial',
    openxla: 'partial',
    tvm: 'none'
  },
  {
    name: 'Multi-year Projections',
    neurax: 'full',
    iree: 'none',
    openxla: 'none',
    tvm: 'none'
  },
  {
    name: 'Inference Stability Analysis',
    neurax: 'full',
    iree: 'none',
    openxla: 'none',
    tvm: 'none'
  },
  {
    name: 'AI Design Copilot',
    neurax: 'full',
    iree: 'none',
    openxla: 'none',
    tvm: 'none'
  },
  {
    name: 'Visual Canvas Designer',
    neurax: 'full',
    iree: 'none',
    openxla: 'none',
    tvm: 'none'
  },
  {
    name: 'Hardware Targets',
    neurax: '20 GPUs',
    iree: 'partial',
    openxla: 'full',
    tvm: 'full'
  },
  {
    name: 'MLIR Backend',
    neurax: 'full',
    iree: 'full',
    openxla: 'full',
    tvm: 'partial'
  },
  {
    name: 'Analysis Speed',
    neurax: '<50ms',
    iree: 'N/A',
    openxla: 'N/A',
    tvm: 'N/A'
  }
];

const PRODUCTS = [
  { key: 'neurax', name: 'NEURAX', highlighted: true, subtitle: 'Analytical Compiler' },
  { key: 'iree', name: 'IREE', subtitle: 'Runtime' },
  { key: 'openxla', name: 'OpenXLA', subtitle: 'Compiler' },
  { key: 'tvm', name: 'Apache TVM', subtitle: 'Optimizer' }
];

const StatusIcon = ({ status }: { status: string }) => {
  if (status === 'full') {
    return <Check className="w-5 h-5" style={{ color: C.green }} />;
  }
  if (status === 'partial') {
    return <Minus className="w-5 h-5" style={{ color: C.accent }} />;
  }
  if (status === 'none' || status === 'N/A') {
    return <X className="w-5 h-5" style={{ color: C.faint }} />;
  }
  return <span className="text-[12px] font-medium" style={{ color: C.text }}>{status}</span>;
};

export const ComparisonTable = () => {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th className="text-left p-4 text-[13px] font-semibold uppercase tracking-wider sticky left-0 z-10"
              style={{ backgroundColor: C.bg, color: C.faint, borderBottom: `2px solid ${C.border}` }}>
              Feature
            </th>
            {PRODUCTS.map((product) => (
              <th
                key={product.key}
                className="p-4 text-center text-[13px] font-semibold"
                style={{
                  backgroundColor: product.highlighted ? `${C.accent}10` : C.bg,
                  borderBottom: product.highlighted ? `2px solid ${C.accent}` : `2px solid ${C.border}`,
                  color: product.highlighted ? C.accent : C.muted,
                }}
              >
                <div className="flex flex-col items-center gap-1">
                  <span>{product.name}</span>
                  <span className="text-[10px] font-normal" style={{ color: C.faint }}>
                    {product.subtitle}
                  </span>
                  {product.highlighted && (
                    <span className="text-[9px] px-2 py-0.5 rounded-full font-mono uppercase tracking-wider"
                      style={{ backgroundColor: `${C.accent}20`, border: `1px solid ${C.accent}40`, color: C.accent }}>
                      ⚡ Analytical
                    </span>
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {FEATURES.map((feature, index) => (
            <tr
              key={feature.name}
              className="transition-colors hover:bg-white/5"
              style={{ borderBottom: `1px solid ${C.border}` }}
            >
              <td className="p-4 text-[13px] font-medium sticky left-0 z-10"
                style={{ backgroundColor: index % 2 === 0 ? C.bg : `${C.card}80`, color: C.text }}>
                {feature.name}
              </td>
              <td className="p-4 text-center"
                style={{ backgroundColor: `${C.accent}05` }}>
                <div className="flex items-center justify-center">
                  <StatusIcon status={feature.neurax} />
                </div>
              </td>
              <td className="p-4 text-center">
                <div className="flex items-center justify-center">
                  <StatusIcon status={feature.iree} />
                </div>
              </td>
              <td className="p-4 text-center">
                <div className="flex items-center justify-center">
                  <StatusIcon status={feature.openxla} />
                </div>
              </td>
              <td className="p-4 text-center">
                <div className="flex items-center justify-center">
                  <StatusIcon status={feature.tvm} />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mt-6 flex items-center justify-center gap-8 text-[11px]" style={{ color: C.faint }}>
        <div className="flex items-center gap-2">
          <Check className="w-4 h-4" style={{ color: C.green }} />
          <span>Full Support</span>
        </div>
        <div className="flex items-center gap-2">
          <Minus className="w-4 h-4" style={{ color: C.accent }} />
          <span>Partial Support</span>
        </div>
        <div className="flex items-center gap-2">
          <X className="w-4 h-4" style={{ color: C.faint }} />
          <span>Not Available</span>
        </div>
      </div>
    </div>
  );
};
