import { TrendingUp, Info, DollarSign, Clock, Zap, Leaf, Cpu } from 'lucide-react';
import { AnalysisResult, PerLayerBreakdownRow } from '@/types/architecture.ts';
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';

interface TrainingChartsProps {
  analysis?: AnalysisResult;
  perLayer?: PerLayerBreakdownRow[];
}

const CARD_CLS = 'panel-section p-4 bg-card/30 border-primary/5 rounded-xl';

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

export function TrainingCharts({ analysis }: TrainingChartsProps) {
  if (!analysis || analysis.trainingTimeHours === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-muted-foreground border-2 border-dashed border-border rounded-lg">
        <TrendingUp className="w-8 h-8 mb-2 opacity-20" />
        <p className="text-sm">No training analysis available</p>
        <p className="text-xs">Run analysis to see training projections.</p>
      </div>
    );
  }

  // Training cost breakdown data
  const costData = [
    { name: 'Compute', value: analysis.trainingCostUsd * 0.7, color: COLORS[0] },
    { name: 'Storage', value: analysis.trainingCostUsd * 0.15, color: COLORS[1] },
    { name: 'Network', value: analysis.trainingCostUsd * 0.1, color: COLORS[2] },
    { name: 'Other', value: analysis.trainingCostUsd * 0.05, color: COLORS[3] },
  ];

  // Runtime projection (simulated epochs)
  const runtimeData = Array.from({ length: 10 }, (_, i) => ({
    epoch: i + 1,
    cost: (analysis.trainingCostUsd / 10) * (i + 1),
    time: (analysis.trainingTimeHours / 10) * (i + 1),
    energy: (analysis.energyKwh / 10) * (i + 1),
  }));

  // Efficiency metrics
  const efficiencyData = [
    { metric: 'Cost/Token', value: analysis.costPerMillionTokensUsd / 1000000, unit: '$' },
    { metric: 'Energy/Token', value: analysis.energyKwh / (analysis.totalParams * 1000), unit: 'kWh' },
    { metric: 'CO2/Token', value: analysis.co2Kg / (analysis.totalParams * 1000), unit: 'kg' },
    { metric: 'GPU Hours', value: analysis.gpuHours || analysis.trainingTimeHours, unit: 'h' },
  ].filter(d => d.value > 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-primary" />
          Training — Cost & Runtime Analysis
        </h2>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground bg-secondary/50 px-2 py-1 rounded-md">
          <Info className="w-3 h-3" />
          Compiler-emitted training projections
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <div className={CARD_CLS}>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted-foreground">
            <DollarSign className="w-3 h-3" />
            Train Cost
          </div>
          <div className="mt-2 text-lg font-semibold font-mono">
            ${analysis.trainingCostUsd.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </div>
        </div>
        <div className={CARD_CLS}>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted-foreground">
            <Clock className="w-3 h-3" />
            Duration
          </div>
          <div className="mt-2 text-lg font-semibold font-mono">
            {analysis.trainingTimeHours.toFixed(1)} h
          </div>
        </div>
        <div className={CARD_CLS}>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted-foreground">
            <Zap className="w-3 h-3" />
            Energy
          </div>
          <div className="mt-2 text-lg font-semibold font-mono">
            {analysis.energyKwh.toFixed(1)} kWh
          </div>
        </div>
        <div className={CARD_CLS}>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted-foreground">
            <Leaf className="w-3 h-3" />
            CO2
          </div>
          <div className="mt-2 text-lg font-semibold font-mono">
            {analysis.co2Kg.toFixed(1)} kg
          </div>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Cost Breakdown Pie Chart */}
        <div className={CARD_CLS}>
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <DollarSign className="w-4 h-4 text-primary" />
            Cost Breakdown
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={costData}
                cx="50%"
                cy="50%"
                innerRadius={40}
                outerRadius={70}
                paddingAngle={2}
                dataKey="value"
              >
                {costData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value: number) => `$${value.toFixed(0)}`} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Runtime Projection Line Chart */}
        <div className={CARD_CLS}>
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Clock className="w-4 h-4 text-primary" />
            Runtime Projection
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={runtimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
              <XAxis dataKey="epoch" stroke="var(--muted-foreground)" fontSize={10} />
              <YAxis stroke="var(--muted-foreground)" fontSize={10} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="cost" stroke={COLORS[0]} name="Cost ($)" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="time" stroke={COLORS[1]} name="Time (h)" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="energy" stroke={COLORS[2]} name="Energy (kWh)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Efficiency Metrics Bar Chart */}
        <div className={CARD_CLS}>
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-primary" />
            Efficiency Metrics
          </h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={efficiencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
              <XAxis dataKey="metric" stroke="var(--muted-foreground)" fontSize={10} />
              <YAxis stroke="var(--muted-foreground)" fontSize={10} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--card)',
                  border: '1px solid var(--border)',
                  borderRadius: '8px',
                }}
                formatter={(value: number, name: string, props: any) => [`${value.toFixed(4)} ${props.payload.unit}`, name]}
              />
              <Bar dataKey="value" fill={COLORS[4]} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Additional Stats */}
        <div className={CARD_CLS}>
          <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
            <Info className="w-4 h-4 text-primary" />
            Training Details
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="text-muted-foreground">Cost per 1M tokens</span>
              <span className="font-mono font-semibold">
                ${analysis.costPerMillionTokensUsd.toFixed(4)}
              </span>
            </div>
            {(analysis.gpuHours ?? 0) > 0 && (
              <div className="flex justify-between items-center text-xs">
                <span className="text-muted-foreground">GPU Hours</span>
                <span className="font-mono font-semibold">
                  {analysis.gpuHours!.toFixed(1)} h
                </span>
              </div>
            )}
            {analysis.provider && (
              <div className="flex justify-between items-center text-xs">
                <span className="text-muted-foreground">Provider</span>
                <span className="font-mono font-semibold">
                  {analysis.provider}
                </span>
              </div>
            )}
            <div className="flex justify-between items-center text-xs">
              <span className="text-muted-foreground">Model Size</span>
              <span className="font-mono font-semibold">
                {(analysis.totalParams / 1e9).toFixed(2)}B params
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
