import {
  Activity,
  Cpu,
  HardDrive,
  Layers,
  Server,
  TrendingUp,
  Zap,
  CheckCircle,
} from 'lucide-react';

const diagnosticIcons: Record<string, React.ElementType> = {
  shape: Layers,
  vram: HardDrive,
  flops: Zap,
  gradient: TrendingUp,
  router: Activity,
  stability: Activity,
  mup: Cpu,
  bottleneck: Server,
  cluster: Server,
};

const DIAGNOSTIC_FEATURES = [
  { id: 'shape', name: 'Shape Analysis', description: 'Analyze tensor shapes and dimensions', minPlan: 'free' as const },
  { id: 'vram', name: 'VRAM Usage', description: 'Memory consumption per layer', minPlan: 'free' as const },
  { id: 'flops', name: 'FLOPs Counter', description: 'Compute cost per operation', minPlan: 'free' as const },
  { id: 'gradient', name: 'Gradient Flow', description: 'Gradient propagation paths', minPlan: 'free' as const },
  { id: 'router', name: 'Router Analysis', description: 'MoE router utilization', minPlan: 'free' as const },
  { id: 'stability', name: 'Numerical Stability', description: 'FP32 fallback requirements', minPlan: 'free' as const },
  { id: 'mup', name: 'μP Transfer', description: 'Maximal update parameterization', minPlan: 'free' as const },
  { id: 'bottleneck', name: 'Bottleneck Detection', description: 'Identify performance bottlenecks', minPlan: 'free' as const },
  { id: 'cluster', name: 'Cluster Simulation', description: 'Multi-GPU communication patterns', minPlan: 'free' as const },
];

export function DiagnosticsPanel() {

  return (
    <div className="h-full bg-sidebar flex flex-col overflow-hidden">
      <div className="h-10 px-4 flex items-center border-b border-sidebar-border">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Diagnostics
        </span>
      </div>

      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="p-2 space-y-1">
          {DIAGNOSTIC_FEATURES.map((diagnostic) => {
            const Icon = diagnosticIcons[diagnostic.id] || Cpu;
            return (
              <div
                key={diagnostic.id}
                className="flex items-center gap-2 p-2 rounded-md bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-pointer"
              >
                <Icon className="w-3.5 h-3.5 text-primary" />
                <div className="flex-1">
                  <div className="text-xs font-medium">{diagnostic.name}</div>
                  <div className="text-[10px] text-muted-foreground">{diagnostic.description}</div>
                </div>
                <CheckCircle className="w-3.5 h-3.5 text-success" />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
