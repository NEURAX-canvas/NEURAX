import { Layers } from 'lucide-react';
import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, LineChart, Line, Cell,
} from 'recharts';
import {
  hasPerLayerRows,
} from '../simulationData.ts';


interface PerLayerChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const SECTION_CLS = 'panel-section p-4 bg-card/30 border-primary/5 rounded-xl';
const TITLE_CLS = 'text-[10px] font-bold text-muted-foreground uppercase tracking-wider mb-3';
const TT_STYLE = { backgroundColor: '#111', border: '1px solid #333', fontSize: '9px' };

function parseFlops(v: string | number | undefined): number {
  if (v === undefined || v === null) return 0;
  if (typeof v === 'number') return v;
  const m = v.trim().match(/^(-?\d+(\.\d+)?)\s*([A-Za-z]+)?$/);
  if (!m) return 0;
  let num = parseFloat(m[1]);
  const u = m[3]?.toUpperCase() ?? '';
  if (u.startsWith('TFLOP') || u === 'T') num *= 1e12;
  else if (u.startsWith('GFLOP') || u === 'G') num *= 1e9;
  else if (u.startsWith('MFLOP') || u === 'M') num *= 1e6;
  else if (u === 'K') num *= 1e3;
  return num;
}

/** Build a per-layer dataset from perLayer rows */
function buildLayerData(perLayer: PerLayerBreakdownRow[]) {
  return perLayer.map((row) => ({
    key: row.id ?? row.name,
    name: row.name,
    flops: parseFloat((parseFlops(row.flops) / 1e9).toFixed(2)),
    params: row.params ?? 0,
  }));
}

/** Build VRAM data, deriving from params ratio when no per-layer VRAM map */
function buildVramData(analysis: AnalysisResult, perLayer: PerLayerBreakdownRow[]) {
  const vramMap = analysis.perLayerVram ?? {};
  const totalVramBytes = analysis.peakVramBytes || 1;
  const totalParams = analysis.totalParams || 1;

  return perLayer.map((row) => {
    const value = vramMap[row.id] ?? vramMap[row.name];
    if (typeof value === 'number' && value > 0) {
      return { name: row.name, vramMb: value / 1e6 };
    }
    // Derive from params ratio × peak VRAM
    const paramRatio = (row.params ?? 0) / totalParams;
    return { name: row.name, vramMb: parseFloat(((paramRatio * totalVramBytes) / 1e6).toFixed(2)) };
  });
}

/** Build latency data, deriving from FLOPs/throughput when no per-layer latency map */
function buildLatencyData(analysis: AnalysisResult, perLayer: PerLayerBreakdownRow[]) {
  const latencyMap = analysis.perLayerLatency ?? {};
  const totalFlops = analysis.totalFlops || 1;
  const throughputMs = analysis.latencyMs || 100;

  return perLayer.map((row) => {
    const value = latencyMap[row.id] ?? latencyMap[row.name];
    if (typeof value === 'number' && value > 0) {
      return { name: row.name, latency: parseFloat(value.toFixed(2)) };
    }
    // Derive from FLOPs ratio × total latency
    const rowFlops = parseFlops(row.flops);
    const flopRatio = rowFlops / totalFlops;
    return { name: row.name, latency: parseFloat((flopRatio * throughputMs).toFixed(2)) };
  });
}

// ─── 3.1 FLOPs per Layer ─────────────────────────────────────────────────────

function FlopsPerLayer({ data }: { data: ReturnType<typeof buildLayerData> }) {
  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>3.1 — FLOPs per Layer</div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
            <XAxis dataKey="name" stroke="#888" fontSize={8} tickLine={false} axisLine={false} angle={-35} textAnchor="end" interval={0} />
            <YAxis stroke="#555" fontSize={8} tickLine={false} axisLine={false} tickFormatter={v => `${v}G`} />
            <Tooltip contentStyle={TT_STYLE} formatter={(v: number) => [`${v} GFLOPs`, 'Compute']} />
            <Bar dataKey="flops" fill="#3b82f6" radius={[3, 3, 0, 0]} barSize={24} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 3.2 VRAM per Layer ──────────────────────────────────────────────────────

function VramByLayer({ data }: { data: ReturnType<typeof buildVramData> }) {
  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>3.2 — VRAM per Layer</div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, left: -16, bottom: 24 }}>
            <defs>
              <linearGradient id="gV" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.7} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.2} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
            <XAxis dataKey="name" stroke="#888" fontSize={8} tickLine={false} axisLine={false} angle={-35} textAnchor="end" interval={0} />
            <YAxis stroke="#555" fontSize={8} tickLine={false} axisLine={false} tickFormatter={v => `${v}MB`} />
            <Tooltip contentStyle={TT_STYLE} formatter={(v: number, n: string) => [`${v} MB`, n]} />
            <Area type="monotone" dataKey="vramMb" name="VRAM" stroke="#3b82f6" fill="url(#gV)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 3.3 Latency per Layer ──────────────────────────────────────────────────

function LatencyPerLayer({ data }: { data: ReturnType<typeof buildLatencyData> }) {
  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>3.3 — Latency per Layer</div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
            <XAxis dataKey="name" stroke="#888" fontSize={8} tickLine={false} axisLine={false} angle={-35} textAnchor="end" interval={0} />
            <YAxis stroke="#555" fontSize={8} tickLine={false} axisLine={false} tickFormatter={v => `${v}ms`} />
            <Tooltip contentStyle={TT_STYLE} formatter={(v: number) => [`${v} ms`, 'Latency']} />
            <Line type="monotone" dataKey="latency" stroke="#22d3ee" strokeWidth={2} dot={{ r: 3, fill: '#22d3ee', stroke: 'none' }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 3.4 Parameters per Layer ─────────────────────────────────────────────

function ParamsPerLayer({ data }: { data: ReturnType<typeof buildLayerData> }) {
  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>3.4 — Parameters per Layer</div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
            <XAxis dataKey="name" stroke="#888" fontSize={8} tickLine={false} axisLine={false} angle={-35} textAnchor="end" interval={0} />
            <YAxis stroke="#555" fontSize={8} tickLine={false} axisLine={false} tickFormatter={v => `${(v / 1e6).toFixed(1)}M`} />
            <Tooltip contentStyle={TT_STYLE} formatter={(v: number) => [`${(v).toLocaleString()} params`, 'Parameters']} />
            <Bar dataKey="params" fill="#22d3ee" radius={[3, 3, 0, 0]} barSize={24} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 3.5 Compute-Memory Ratio ───────────────────────────────────────

function ComputeMemoryRatio({ merged }: { merged: { name: string; flops: number; vramMb: number; ratio: number }[] }) {
  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>3.5 — Compute-Memory Ratio (FLOPs/MB)</div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={merged} margin={{ top: 4, right: 4, left: -20, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
            <XAxis dataKey="name" stroke="#888" fontSize={8} tickLine={false} axisLine={false} angle={-35} textAnchor="end" interval={0} />
            <YAxis stroke="#555" fontSize={8} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={TT_STYLE} formatter={(v: number, n: string) => [v.toFixed(2), n === 'flops' ? 'GFLOPs' : n === 'vramMb' ? 'MB' : 'Ratio']} />
            <Bar dataKey="flops" name="GFLOPs" fill="#3b82f6" radius={[3, 3, 0, 0]} barSize={12} />
            <Bar dataKey="ratio" name="FLOPs/MB" fill="#22d3ee" radius={[3, 3, 0, 0]} barSize={12} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 3.6 Memory Bandwidth per Layer ──────────────────────────────────

function BandwidthPerLayer({ data: vramData }: { data: ReturnType<typeof buildVramData> }) {
  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>3.6 — Memory per Layer (Bandwidth)</div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={vramData} margin={{ top: 4, right: 4, left: -16, bottom: 24 }}>
            <defs>
              <linearGradient id="gBw" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.7} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0.2} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
            <XAxis dataKey="name" stroke="#888" fontSize={8} tickLine={false} axisLine={false} angle={-35} textAnchor="end" interval={0} />
            <YAxis stroke="#555" fontSize={8} tickLine={false} axisLine={false} tickFormatter={v => `${v}MB`} />
            <Tooltip contentStyle={TT_STYLE} formatter={(v: number) => [`${v} MB`, 'Bandwidth Demand']} />
            <Area type="monotone" dataKey="vramMb" stroke="#8b5cf6" fill="url(#gBw)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── 3.7 Layer Score (Efficiency Index) ─────────────────────────────

function LayerScore({ data }: { data: ReturnType<typeof buildLayerData> }) {
  const scored = data.map(d => {
    const maxP = Math.max(...data.map(x => x.params), 1);
    const maxF = Math.max(...data.map(x => x.flops), 1);
    const efficiency = parseFloat((((d.flops / maxF) / (d.params / maxP + 0.01)) * 100).toFixed(1));
    return { name: d.name, score: Math.min(efficiency, 100) };
  });

  return (
    <div className={SECTION_CLS}>
      <div className={TITLE_CLS}>3.7 — Layer Efficiency Score</div>
      <div className="h-52">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={scored} margin={{ top: 4, right: 4, left: -20, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" vertical={false} />
            <XAxis dataKey="name" stroke="#888" fontSize={8} tickLine={false} axisLine={false} angle={-35} textAnchor="end" interval={0} />
            <YAxis stroke="#555" fontSize={8} tickLine={false} axisLine={false} domain={[0, 100]} tickFormatter={v => `${v}%`} />
            <Tooltip contentStyle={TT_STYLE} formatter={(v: number) => [`${v}%`, 'Efficiency']} />
            <Bar dataKey="score" fill="#f59e0b" radius={[3, 3, 0, 0]} barSize={24}>
              {scored.map((entry, idx) => (
                <Cell key={idx} fill={entry.score >= 50 ? '#22c55e' : entry.score >= 25 ? '#f59e0b' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ─── Main Component ─────────────────────────────────────────────────────────

export function PerLayerCharts({ analysis, perLayer }: PerLayerChartsProps) {
  const hasData = analysis && hasPerLayerRows(perLayer);

  if (!hasData) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground border-2 border-dashed border-border rounded-lg">
        <Layers className="w-8 h-8 mb-2 opacity-20" />
        <p className="text-sm">No per-layer data available</p>
        <p className="text-xs">Run analysis to see individual layer metrics.</p>
      </div>
    );
  }

  const data = buildLayerData(perLayer ?? []);
  const vramData = buildVramData(analysis!, perLayer ?? []);
  const latencyData = buildLatencyData(analysis!, perLayer ?? []);

  // Always have all 7 charts — derive VRAM/latency data when maps are missing
  const merged = data.map(d => {
    const v = vramData.find(v => v.name === d.name);
    return { name: d.name, flops: d.flops, vramMb: v?.vramMb ?? 0, ratio: v?.vramMb ? parseFloat((d.flops / v.vramMb).toFixed(2)) : 0 };
  });

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Layers className="w-4 h-4 text-primary" />
        <h2 className="text-base font-semibold">Per Layer Breakdown</h2>
      </div>

      {/* Row 1: FLOPs · VRAM */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <FlopsPerLayer data={data} />
        <VramByLayer data={vramData} />
      </div>

      {/* Row 2: Latency · Params */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LatencyPerLayer data={latencyData} />
        <ParamsPerLayer data={data} />
      </div>

      {/* Row 3: Compute-Memory Ratio · Bandwidth */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ComputeMemoryRatio merged={merged} />
        <BandwidthPerLayer data={vramData} />
      </div>

      {/* Row 4: Layer Score */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <LayerScore data={data} />
      </div>
    </div>
  );
}
