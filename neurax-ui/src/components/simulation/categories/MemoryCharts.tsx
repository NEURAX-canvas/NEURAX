import { HardDrive, Share2, TrendingUp } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  AreaChart,
  Area,
  LineChart,
  Line
} from 'recharts';
// All chart data is derived unconditionally — no has* checks needed


interface MemoryChartsProps {
  analysis?: AnalysisResult;
}

const COLORS = {
  activations: '#0ea5e9',
  weights: '#f59e0b',
  temp: '#ef4444',
  gradients: '#8b5cf6',
  forward: '#0ea5e9',
  backward: '#ef4444'
};

export function MemoryCharts({ analysis }: MemoryChartsProps) {
  if (!analysis || analysis.peakVramBytes === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground border-2 border-dashed border-border rounded-lg">
        <HardDrive className="w-8 h-8 mb-2 opacity-20" />
        <p className="text-sm">No memory analysis available</p>
        <p className="text-xs">Run analysis to see VRAM breakdown.</p>
      </div>
    );
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // 4.3 Donut Data (always available)
  const donutData = [
    { name: 'Activations', value: analysis.activationMemoryBytes, color: COLORS.activations },
    { name: 'Weights', value: analysis.parameterMemoryBytes, color: COLORS.weights },
    { name: 'Temp Buffers', value: Math.max(0, analysis.peakVramBytes - analysis.activationMemoryBytes - analysis.parameterMemoryBytes), color: COLORS.temp },
  ].filter(d => d.value > 0);

  // ─── 4.1 Heatmap — derive from perLayer maps or gradient_breakdown if no raw heatmap ──
  const rawHeatmap = analysis.memory_heatmap || analysis.live_trace?.memory_heatmap || [];
  const hasRawHeatmap = rawHeatmap.length > 0;
  // Derive heatmap from gradient_memory_breakdown or per-layer data
  const gradientSource = analysis.gradient_memory_breakdown || analysis.live_trace?.gradient_memory_breakdown || [];
  const heatmapData = hasRawHeatmap
    ? rawHeatmap
    : gradientSource.length > 0
      ? gradientSource.map((g) => ({
        layer: g.name,
        timeline: [g.forward > g.backward ? 2 : g.backward > 0 ? 1 : 0].concat(
          Array.from({ length: 19 }, (_, j) => (g.backward > 0 && j < 15 ? 2 : j < 18 ? 1 : 0))
        ),
      }))
      : [];

  // ─── 4.2 Liveness — derive from peak VRAM breakdown ──
  const rawLiveness = analysis.memory_liveness || analysis.live_trace?.memory_liveness || [];
  const hasRawLiveness = rawLiveness.length > 0;
  const livenessData = hasRawLiveness
    ? rawLiveness.map(d => ({ step: d.step, vram: d.value / (1024 ** 2) }))
    : Array.from({ length: 20 }, (_, i) => ({
      step: i,
      vram: (analysis.peakVramBytes / (1024 ** 2)) * (1 - Math.abs(i - 10) / 15),
    }));

  // ─── 4.5 Gradient Memory — derive if not available ──
  const rawGradient = analysis.gradient_memory_breakdown || analysis.live_trace?.gradient_memory_breakdown || [];
  const gradientData = rawGradient.length > 0
    ? rawGradient.map(d => ({
      name: d.name,
      forwardMB: d.forward / (1024 ** 2),
      backwardMB: d.backward / (1024 ** 2),
    }))
    // Derive from parameter/activation split
    : [
      { name: 'Parameters', forwardMB: (analysis.parameterMemoryBytes * 0.6) / (1024 ** 2), backwardMB: (analysis.parameterMemoryBytes * 0.4) / (1024 ** 2) },
      { name: 'Activations', forwardMB: (analysis.activationMemoryBytes * 0.7) / (1024 ** 2), backwardMB: (analysis.activationMemoryBytes * 0.3) / (1024 ** 2) },
    ];
  // Rename keys for chart compatibility
  const gradientChartData = gradientData.map(d => ({
    name: d.name,
    forward: d.forwardMB,
    backward: d.backwardMB,
  }));

  // ─── 4.6 KV Cache — derive from model params if not available ──
  const rawKv = analysis.kv_cache_scaling || analysis.live_trace?.kv_cache_scaling || [];
  const kvData = rawKv.length > 0
    ? rawKv.map(d => ({ seq: d.seq, value: d.value / (1024 ** 2) }))
    : Array.from({ length: 6 }, (_, i) => {
      const seq = Math.pow(2, 10 + i);
      const hiddenDim = analysis.hiddenSize || 4096;
      // KV cache ∼ 2 × seq_len × num_layers × hidden_dim × 2 bytes
      const cacheMb = (2 * seq * (analysis.numLayers || 1) * hiddenDim * 2) / (1024 ** 2);
      return { seq, value: Math.min(cacheMb, 50000) };
    });

  const supportsHeatmap = heatmapData.length > 0;

  return (
    <div className="space-y-6 pb-12">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <HardDrive className="w-5 h-5 text-primary" />
          Memory — VRAM Deep Dive
        </h2>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 4.1 Memory Heatmap */}
        <div className="panel-section p-4 bg-card/30 backdrop-blur-md border border-white/5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">
            4.1 — Memory Heatmap (Timeline)
            {!hasRawHeatmap && <span className="ml-2 text-[8px] text-amber-500/70">estimated</span>}
          </h3>
          {supportsHeatmap ? (
            <div className="overflow-x-auto">
              <div className="min-w-[400px] space-y-1">
                {heatmapData.map((layer: any, idx: number) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div className="w-24 shrink-0 text-[9px] text-muted-foreground truncate" title={layer.layer || layer.name}>
                      {layer.layer || layer.name}
                    </div>
                    <div className="flex-1 flex gap-0.5 h-3">
                      {(layer.timeline || []).map((active: number, stepIdx: number) => (
                        <div
                          key={stepIdx}
                          className={`flex-1 rounded-sm transition-all duration-300 ${active > 0
                            ? stepIdx < 5 ? 'bg-green-500/60' : stepIdx < 12 ? 'bg-yellow-500/60' : 'bg-orange-500/60'
                            : 'bg-white/5'
                            }`}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="h-32 flex items-center justify-center text-[10px] text-muted-foreground">
              <p>No heatmap data — run streaming analysis for live memory trace</p>
            </div>
          )}
          <div className="mt-3 flex justify-between text-[8px] text-muted-foreground uppercase tracking-widest">
            <span>Layer 0</span>
            <span>Step T</span>
            <span>Layer N</span>
          </div>
        </div>

        {/* 4.2 VRAM Liveness */}
        <div className="panel-section p-4 bg-card/30 backdrop-blur-md border border-white/5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">
            4.2 — VRAM Liveness
            {!hasRawLiveness && <span className="ml-2 text-[8px] text-amber-500/70">estimated</span>}
          </h3>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={livenessData}>
                <defs>
                  <linearGradient id="colorVram" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={COLORS.activations} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={COLORS.activations} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis dataKey="step" stroke="#888" fontSize={10} hide={!hasRawLiveness} />
                <YAxis stroke="#888" fontSize={10} tickFormatter={(val) => `${val}M`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #333', fontSize: '10px' }}
                  formatter={(val: number) => [`${val.toFixed(2)} MB`, 'VRAM']}
                />
                <Area
                  type="monotone"
                  dataKey="vram"
                  stroke={COLORS.activations}
                  fillOpacity={1}
                  fill="url(#colorVram)"
                  strokeWidth={2}
                  animationDuration={1500}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4.3 Peak VRAM Breakdown */}
        <div className="panel-section p-4 bg-card/30 backdrop-blur-md border border-white/5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">4.3 — Peak VRAM Breakdown</h3>
          <div className="h-56 relative flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={donutData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                  animationDuration={1000}
                >
                  {donutData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #333', fontSize: '10px' }}
                  formatter={(val: number) => formatBytes(val)}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span className="text-[10px] text-muted-foreground uppercase">Peak</span>
              <span className="text-xl font-bold font-mono">{formatBytes(analysis.peakVramBytes)}</span>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2">
            {donutData.map((d, i) => (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
                  <span className="text-[9px] text-muted-foreground">{d.name}</span>
                </div>
                <span className="text-xs font-mono font-bold">{(d.value / analysis.peakVramBytes * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>

        {/* 4.4 Memory Fragmentation & OOM Risk */}
        <div className="panel-section p-4 bg-card/30 backdrop-blur-md border border-white/5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">4.4 — Memory Fragmentation & OOM Risk</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="text-[9px] text-muted-foreground uppercase">Fragmentation</div>
              <div className="relative h-32 flex items-center justify-center">
                <div className="relative w-28 h-28">
                  <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                    <circle cx="50" cy="50" r="42" fill="none" stroke="#222" strokeWidth="10" />
                    <circle
                      cx="50" cy="50" r="42" fill="none"
                      stroke={(analysis.memoryFragmentationPct ?? 0) > 30 ? '#ef4444' : (analysis.memoryFragmentationPct ?? 0) > 10 ? '#f59e0b' : '#22c55e'}
                      strokeWidth="10"
                      strokeDasharray={`${Math.min(analysis.memoryFragmentationPct ?? 0, 100) * 2.64} 264`}
                      strokeLinecap="round"
                      className="transition-all duration-1000"
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-lg font-bold font-mono">{(analysis.memoryFragmentationPct ?? 0).toFixed(0)}%</span>
                    <span className="text-[9px] text-muted-foreground">fragmented</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <div className="text-[9px] text-muted-foreground uppercase">OOM Risk</div>
              <div className={`rounded-lg border px-3 py-3 text-center ${analysis.oomRisk === 'high' ? 'border-red-500/50 bg-red-500/10' : analysis.oomRisk === 'medium' ? 'border-amber-500/50 bg-amber-500/10' : 'border-green-500/50 bg-green-500/10'}`}>
                <div className="text-xs font-bold uppercase tracking-wider" style={{
                  color: analysis.oomRisk === 'high' ? '#ef4444' : analysis.oomRisk === 'medium' ? '#f59e0b' : '#22c55e'
                }}>
                  {analysis.oomRisk || 'low'}
                </div>
                <div className="text-[10px] text-muted-foreground mt-1">OOM Risk Level</div>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-[9px]">
                  <span className="text-muted-foreground">Max Batch Size Fit</span>
                  <span className="font-mono">{analysis.maxBatchSizeFit || '—'}</span>
                </div>
                <div className="flex justify-between text-[9px]">
                  <span className="text-muted-foreground">Peak VRAM / GPU</span>
                  <span className="font-mono">{(analysis.peakVramBytes / 1e9).toFixed(2)} GB / {analysis.gpuMemoryGb.toFixed(1)} GB</span>
                </div>
                <div className="flex justify-between text-[9px]">
                  <span className="text-muted-foreground">Utilization Rate</span>
                  <span className="font-mono">{analysis.gpuMemoryGb > 0 ? ((analysis.peakVramBytes / 1e9 / analysis.gpuMemoryGb) * 100).toFixed(0) : '—'}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 4.5 Gradient Memory */}
        <div className="panel-section p-4 bg-card/30 backdrop-blur-md border border-white/5">
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-4">
            4.5 — Gradient Memory (Training)
            {rawGradient.length === 0 && <span className="ml-2 text-[8px] text-amber-500/70">derived</span>}
          </h3>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={gradientChartData} margin={{ bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                <XAxis dataKey="name" stroke="#888" fontSize={9} tick={{ fill: '#888' }} interval={0} angle={-30} textAnchor="end" />
                <YAxis stroke="#888" fontSize={10} tickFormatter={(val) => `${val}M`} />
                <Tooltip contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #333', fontSize: '10px' }} />
                <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                <Bar dataKey="forward" name="Forward" stackId="a" fill={COLORS.forward} radius={[0, 0, 0, 0]} />
                <Bar dataKey="backward" name="Backward" stackId="a" fill={COLORS.backward} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4.6 KV Cache Growth */}
        <div className="panel-section p-4 bg-card/30 backdrop-blur-md border border-white/5 lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              4.6 — KV Cache Growth (LLM)
              {rawKv.length === 0 && <span className="ml-2 text-[8px] text-amber-500/70">projected</span>}
            </h3>
            <div className="flex items-center gap-4 text-[10px]">
              <div className="flex items-center gap-1">
                <Share2 className="w-3 h-3 text-primary" />
                <span className="text-muted-foreground">Context: {kvData.length > 0 ? `${kvData[kvData.length - 1].seq}` : '0'} tokens</span>
              </div>
              <div className="flex items-center gap-1">
                <TrendingUp className="w-3 h-3 text-green-500" />
                <span className="text-muted-foreground">Scaling: Linear (O(N))</span>
              </div>
            </div>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={kvData} margin={{ left: 20, right: 20, top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" />
                <XAxis dataKey="seq" stroke="#888" fontSize={10} label={{ value: 'Sequence Length', position: 'bottom', offset: 0, fontSize: 10, fill: '#666' }} />
                <YAxis stroke="#888" fontSize={10} tickFormatter={(val) => `${val}M`} label={{ value: 'Cache Size (MB)', angle: -90, position: 'insideLeft', offset: 0, fontSize: 10, fill: '#666' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid #333', fontSize: '10px' }}
                  formatter={(val: number) => [`${val.toFixed(2)} MB`, 'Cache Size']}
                />
                <Line
                  type="stepAfter"
                  dataKey="value"
                  stroke={COLORS.activations}
                  strokeWidth={3}
                  dot={{ r: 4, fill: COLORS.activations, strokeWidth: 0 }}
                  activeDot={{ r: 6, stroke: '#fff', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
