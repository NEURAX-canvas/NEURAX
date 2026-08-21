import { TrendingUp, Info, DollarSign, Clock, Zap, Leaf } from 'lucide-react';
import { AnalysisResult } from '@/types/architecture.ts';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell,
} from 'recharts';
import {
  ChartCard,
  ChartContainer,
  StatCard,
  chartTooltipStyle,
  chartActiveDot,
  CHART_MARGINS,
  ChartErrorBoundary,
  EmptyChartState,
} from '../shared';


interface TrainingChartsProps {
  analysis?: AnalysisResult;
}

const COLORS = [
  'hsl(var(--chart-1))',
  'hsl(var(--chart-2))',
  'hsl(var(--chart-3))',
  'hsl(var(--chart-4))',
  'hsl(var(--chart-5))',
  'hsl(var(--chart-6))',
];

export function TrainingCharts({ analysis }: TrainingChartsProps) {
  if (!analysis || analysis.trainingTimeHours === 0) {
    return (
      <EmptyChartState
        icon={TrendingUp}
        title="No training analysis available"
        description="Run analysis to see training projections."
      />
    );
  }

  // Cost breakdown (derived from trainingCostUsd)
  const costData = analysis.trainingCostUsd > 0
    ? [
      { name: 'Compute', value: analysis.trainingCostUsd * 0.7, color: COLORS[0] },
      { name: 'Storage', value: analysis.trainingCostUsd * 0.15, color: COLORS[1] },
      { name: 'Network', value: analysis.trainingCostUsd * 0.1, color: COLORS[2] },
      { name: 'Other', value: analysis.trainingCostUsd * 0.05, color: COLORS[3] },
    ]
    : [];

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
    <ChartErrorBoundary>
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary" />
            Training — Cost & Runtime Analysis
          </h2>
          <div className="flex items-center gap-2 text-[11px] text-muted-foreground bg-secondary/50 px-2 py-1 rounded-md">
            <Info className="w-3 h-3" />
            Compiler-emitted training projections
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
          <StatCard
            icon={<DollarSign className="w-3 h-3" />}
            label="Train Cost"
            value={`$${analysis.trainingCostUsd.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
          />
          <StatCard
            icon={<Clock className="w-3 h-3" />}
            label="Duration"
            value={`${analysis.trainingTimeHours.toFixed(1)} h`}
          />
          <StatCard
            icon={<Zap className="w-3 h-3" />}
            label="Energy"
            value={`${analysis.energyKwh.toFixed(1)} kWh`}
          />
          <StatCard
            icon={<Leaf className="w-3 h-3" />}
            label="CO2"
            value={`${analysis.co2Kg.toFixed(1)} kg`}
          />
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Cost Breakdown Pie Chart */}
          <ChartCard
            title="Cost Breakdown"
            badge={costData.length > 0 ? { text: 'derived', variant: 'derived' } : undefined}
          >
            {costData.length > 0 ? (
              <ChartContainer>
                <ResponsiveContainer width="100%" height="100%">
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
                    <Tooltip contentStyle={chartTooltipStyle()} formatter={(value: number) => `$${value.toFixed(0)}`} />
                    <Legend wrapperStyle={{ fontSize: '11px' }} />
                  </PieChart>
                </ResponsiveContainer>
              </ChartContainer>
            ) : (
              <EmptyChartState
                icon={DollarSign}
                title="No cost breakdown"
                description="Training cost data not available."
              />
            )}
          </ChartCard>

          {/* Runtime Projection Line Chart */}
          <ChartCard title="Runtime Projection" badge={{ text: 'projected', variant: 'info' }}>
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={runtimeData} margin={CHART_MARGINS.line}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="epoch" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={chartTooltipStyle()} />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Line type="monotone" dataKey="cost" stroke={COLORS[0]} name="Cost ($)" strokeWidth={2} dot={false} activeDot={chartActiveDot(COLORS[0])} />
                  <Line type="monotone" dataKey="time" stroke={COLORS[1]} name="Time (h)" strokeWidth={2} dot={false} activeDot={chartActiveDot(COLORS[1])} />
                  <Line type="monotone" dataKey="energy" stroke={COLORS[2]} name="Energy (kWh)" strokeWidth={2} dot={false} activeDot={chartActiveDot(COLORS[2])} />
                </LineChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          {/* Efficiency Metrics Bar Chart */}
          <ChartCard title="Efficiency Metrics">
            <ChartContainer>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={efficiencyData} margin={CHART_MARGINS.barHorizontal}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.3} />
                  <XAxis dataKey="metric" stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={chartTooltipStyle()}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(value: number, _name: string, props: any) => [`${value.toFixed(4)} ${props?.payload?.unit ?? ''}`, 'Value']}
                  />
                  <Bar dataKey="value" fill={COLORS[4]} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartContainer>
          </ChartCard>

          {/* Training Details */}
          <ChartCard title="Training Details">
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
          </ChartCard>
        </div>
      </div>
    </ChartErrorBoundary>
  );
}
