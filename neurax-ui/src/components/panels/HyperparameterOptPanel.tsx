/**
 * HyperparameterOptPanel.tsx
 *
 * Panneau de configuration et d'exécution de l'optimisation
 * d'hyperparamètres pour NEURAX.
 *
 * Fonctionnalités :
 *  - Configuration de l'espace de recherche (bornes par paramètre)
 *  - Sélection de la stratégie (Grid/Random/Bayesian)
 *  - Sélection de l'objectif (Latence/Mémoire/Débit/Coût/Params/Équilibré)
 *  - Recommendation Hardware-Aware
 *  - Exécution de l'optimisation
 *  - Résultats classés avec métriques estimées
 *  - Application de la meilleure configuration
 */

import { useState, useMemo, useCallback } from 'react';
import {
  Sliders,
  Zap,
  Cpu,
  HardDrive,
  Gauge,
  DollarSign,
  Layers,
  Target,
  Search,
  CheckCircle2,
  XCircle,
  ArrowUpDown,
  TrendingUp,
  Info,
  ChevronDown,
  ChevronRight,
  Play,
  RotateCcw,
  Sparkles,
  X,
} from 'lucide-react';
import { ArchitectureFamily } from '@/types/plugins';
import { HardwareConfig, useHardware } from '@/contexts/HardwareContext';
import {
  HyperparameterDef,
  SearchRange,
  getParamsForFamily,
  getConstraintsForFamily,
} from '@/utils/hyperparameterDefs';
import {
  OptimizationStrategy,
  OptimizationObjective,
  OptimizationResult,
  CandidateConfig,
  runOptimization,
} from '@/utils/hyperparameterOptimizer';
import {
  recommendForHardware,
  analyzeHardware,
  getAllGpus,
  getGpuDisplayName,
  HardwareRecommendation,
} from '@/utils/hardwareAwareOptimizer';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { cn } from '@/lib/utils';

// ─── Props ───────────────────────────────────────────────────────────

interface HyperparameterOptPanelProps {
  family: ArchitectureFamily;
  isOpen: boolean;
  onClose: () => void;
}

// ─── Constants ───────────────────────────────────────────────────────

const STRATEGIES: { value: OptimizationStrategy; label: string; description: string }[] = [
  {
    value: 'grid_search',
    label: 'Grille (Grid)',
    description: 'Exploration exhaustive d\'une grille définie — lent mais complet',
  },
  {
    value: 'random_search',
    label: 'Aléatoire (Random)',
    description: 'Échantillonnage aléatoire — rapide, efficace pour hautes dimensions',
  },
  {
    value: 'bayesian',
    label: 'Bayésien (Bayesian)',
    description: 'Recherche adaptative avec raffinement local autour des meilleurs candidats',
  },
];

const OBJECTIVES: { value: OptimizationObjective; label: string; icon: React.ElementType; color: string }[] = [
  { value: 'minimize_latency', label: '⚡ Latence min', icon: Zap, color: 'text-yellow-500' },
  { value: 'minimize_memory', label: '💾 Mémoire min', icon: HardDrive, color: 'text-blue-500' },
  { value: 'maximize_throughput', label: '📈 Débit max', icon: TrendingUp, color: 'text-green-500' },
  { value: 'minimize_cost', label: '💰 Coût min', icon: DollarSign, color: 'text-emerald-500' },
  { value: 'minimize_params', label: '📏 Paramètres min', icon: Layers, color: 'text-purple-500' },
  { value: 'balanced', label: '⚖️ Équilibré', icon: Gauge, color: 'text-orange-500' },
];

// ─── Helpers ─────────────────────────────────────────────────────────

function fmtNum(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

// ─── Main Component ──────────────────────────────────────────────────

export function HyperparameterOptPanel({ family, isOpen, onClose }: HyperparameterOptPanelProps) {
  const { config, updateConfig } = useHardware();

  // UI state
  const [activeTab, setActiveTab] = useState<'optimizer' | 'hardware' | 'results'>('optimizer');
  const [strategy, setStrategy] = useState<OptimizationStrategy>('random_search');
  const [objective, setObjective] = useState<OptimizationObjective>('balanced');
  const [maxCandidates, setMaxCandidates] = useState(50);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showConstraints, setShowConstraints] = useState(false);
  const [targetHardware, setTargetHardware] = useState(config.hardware);

  // Custom search space overrides
  const [searchOverrides, setSearchOverrides] = useState<
    Partial<Record<keyof HardwareConfig, { min: number; max: number }>>
  >({});

  // Optimization state
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<OptimizationResult | null>(null);
  const [expandedCandidate, setExpandedCandidate] = useState<string | null>(null);

  // Hardware recommendations
  const [, setHwRecommendations] = useState<HardwareRecommendation[]>([]);

  const params = useMemo(() => getParamsForFamily(family), [family]);
  const constraints = useMemo(() => getConstraintsForFamily(family), [family]);

  // ─── Run Optimization ──────────────────────────────────────────

  const handleRunOptimization = useCallback(() => {
    setIsRunning(true);
    setResult(null);

    // Use setTimeout to allow UI to update before blocking
    setTimeout(() => {
      const searchSpace = buildSearchSpace(params, searchOverrides);

      const optResult = runOptimization(family, strategy, objective, {
        params,
        searchSpace,
        target: { objective, targetHardware },
        maxCandidates: strategy === 'grid_search' ? maxCandidates : undefined,
        randomCount: strategy !== 'grid_search' ? maxCandidates : undefined,
      });

      setResult(optResult);
      setIsRunning(false);
    }, 50);
  }, [family, strategy, objective, params, searchOverrides, maxCandidates, targetHardware]);

  // ─── Hardware Recommendations ──────────────────────────────────

  const handleHardwareRecommend = useCallback(() => {
    const recommendations = recommendForHardware(family, targetHardware, objective);
    setHwRecommendations(recommendations);
    setActiveTab('hardware');
  }, [family, targetHardware, objective]);

  // ─── Apply Best Config ─────────────────────────────────────────

  const handleApplyConfig = useCallback(
    (candidate: CandidateConfig) => {
      const cfg = candidate.config;
      const updates: Partial<HardwareConfig> = {};

      // Only apply non-zero, non-null values
      for (const [key, value] of Object.entries(cfg)) {
        if (value !== undefined && value !== null && value !== 0 && value !== false) {
          (updates as Record<string, unknown>)[key] = value;
        }
      }

      // Always apply hardware and precision
      if (cfg.hardware) updates.hardware = cfg.hardware;
      if (cfg.precision) updates.precision = cfg.precision;
      if (cfg.batchSize && cfg.batchSize > 0) updates.batchSize = cfg.batchSize;

      updateConfig(updates);
    },
    [updateConfig],
  );

  // ─── Apply Hardware Recommendation ─────────────────────────────

  const handleApplyRecommendation = useCallback(
    (rec: HardwareRecommendation) => {
      updateConfig({
        hardware: rec.gpu,
        precision: rec.recommended.precision as HardwareConfig['precision'],
        batchSize: rec.recommended.batchSize,
        hiddenDim: rec.recommended.hiddenDim,
        numLayers: rec.recommended.numLayers,
        ffnDim: rec.recommended.ffnDim,
      });
    },
    [updateConfig],
  );

  // ─── Search Space Override ─────────────────────────────────────

  const handleOverrideChange = useCallback(
    (key: string, field: 'min' | 'max', value: number) => {
      setSearchOverrides((prev) => ({
        ...prev,
        [key]: { ...((prev as Record<string, { min: number; max: number }>)[key] ?? { min: 0, max: 0 }), [field]: value },
      }));
    },
    [],
  );

  // ─── Render ────────────────────────────────────────────────────

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col bg-card p-0 gap-0">
        <div className="px-3 py-2 border-b border-border bg-panel-header flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Sliders className="w-3.5 h-3.5 text-primary" />
            <span className="text-[10px] font-bold uppercase tracking-widest text-foreground/80">
              Hyperparameter Optimization
            </span>
          </div>
          <button onClick={onClose} className="p-1 rounded-md hover:bg-secondary text-muted-foreground hover:text-foreground transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

      {/* Tabs */}
      <div className="flex border-b border-sidebar-border">
        {[
          { id: 'optimizer' as const, label: 'Optimizer', icon: Target },
          { id: 'hardware' as const, label: 'Hardware', icon: Cpu },
          { id: 'results' as const, label: 'Results', icon: ArrowUpDown, count: result?.candidates.length },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              'flex-1 flex items-center justify-center gap-1.5 py-2 text-[9px] font-semibold uppercase tracking-wider transition-colors',
              activeTab === tab.id
                ? 'text-primary border-b-2 border-primary bg-primary/5'
                : 'text-muted-foreground hover:text-foreground hover:bg-secondary/20',
            )}
          >
            <tab.icon className="w-3 h-3" />
            {tab.label}
            {tab.count !== undefined && tab.count > 0 && (
              <Badge variant="secondary" className="text-[8px] h-3.5 px-1">
                {tab.count}
              </Badge>
            )}
          </button>
        ))}
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'optimizer' && (
          <div className="p-3 space-y-3">
            {/* Strategy */}
            <div className="space-y-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Strategy
              </label>
              <div className="grid grid-cols-3 gap-1">
                {STRATEGIES.map((s) => (
                  <button
                    key={s.value}
                    onClick={() => setStrategy(s.value)}
                    className={cn(
                      'p-2 rounded border text-center transition-all',
                      strategy === s.value
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-border/40 bg-background/30 text-muted-foreground hover:border-border/60',
                    )}
                  >
                    <div className="text-[10px] font-bold">{s.label}</div>
                    <div className="text-[7px] text-muted-foreground/60 mt-0.5 leading-tight">
                      {s.description}
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Objective */}
            <div className="space-y-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Objective
              </label>
              <div className="grid grid-cols-3 gap-1">
                {OBJECTIVES.map((o) => {
                  const Icon = o.icon;
                  return (
                    <button
                      key={o.value}
                      onClick={() => setObjective(o.value)}
                      className={cn(
                        'p-2 rounded border flex flex-col items-center gap-1 transition-all',
                        objective === o.value
                          ? 'border-primary bg-primary/10 text-primary'
                          : 'border-border/40 bg-background/30 text-muted-foreground hover:border-border/60',
                      )}
                    >
                      <Icon className={cn('w-4 h-4', o.color)} />
                      <span className="text-[8px] font-semibold leading-tight text-center">
                        {o.label}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Target Hardware */}
            <div className="space-y-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                <Cpu className="w-3 h-3" />
                Target Hardware
              </label>
              <Select value={targetHardware} onValueChange={setTargetHardware}>
                <SelectTrigger className="h-7 text-[10px] bg-background/50 border-border/40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {getAllGpus().map((gpu) => (
                    <SelectItem key={gpu} value={gpu} className="text-xs">
                      {getGpuDisplayName(gpu)} ({gpu})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Max Candidates */}
            <div className="space-y-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                Max Candidates
              </label>
              <Input
                type="number"
                value={maxCandidates}
                onChange={(e) => setMaxCandidates(Math.max(1, parseInt(e.target.value) || 50))}
                className="h-7 text-[10px] bg-background/50 border-border/40"
                min={1}
                max={1000}
              />
            </div>

            {/* Advanced: Search Space */}
            <div>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full flex items-center justify-between py-1.5 text-[10px] font-semibold text-muted-foreground hover:text-foreground transition-colors"
              >
                <span>Custom Search Space</span>
                {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              </button>

              {showAdvanced && (
                <div className="mt-1 space-y-2 pl-1">
                  {params.slice(0, 8).map((param) => {
                    if (param.type === 'bool' || param.type === 'categorical') return null;
                    const override = searchOverrides[param.key];
                    const defaultRange = param.range;

                    return (
                      <div key={param.key as string} className="space-y-0.5">
                        <div className="flex items-center justify-between text-[9px] text-muted-foreground">
                          <span className="font-medium">{param.label}</span>
                          <span className="text-[8px] font-mono opacity-60">
                            [{defaultRange?.min ?? '—'} – {defaultRange?.max ?? '—'}]
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Input
                            type="number"
                            placeholder={`Min: ${defaultRange?.min ?? ''}`}
                            value={override?.min ?? ''}
                            onChange={(e) =>
                              handleOverrideChange(param.key as string, 'min', parseFloat(e.target.value) || 0)
                            }
                            className="h-6 text-[9px] bg-background/50 border-border/40 w-full"
                          />
                          <span className="text-[8px] text-muted-foreground">→</span>
                          <Input
                            type="number"
                            placeholder={`Max: ${defaultRange?.max ?? ''}`}
                            value={override?.max ?? ''}
                            onChange={(e) =>
                              handleOverrideChange(param.key as string, 'max', parseFloat(e.target.value) || 0)
                            }
                            className="h-6 text-[9px] bg-background/50 border-border/40 w-full"
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Constraints Preview */}
            {constraints.length > 0 && (
              <div>
                <button
                  onClick={() => setShowConstraints(!showConstraints)}
                  className="w-full flex items-center justify-between py-1.5 text-[10px] font-semibold text-muted-foreground hover:text-foreground transition-colors"
                >
                  <span className="flex items-center gap-1">
                    <Info className="w-3 h-3" />
                    Constraints ({constraints.length})
                  </span>
                  {showConstraints ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                </button>

                {showConstraints && (
                  <div className="mt-1 space-y-1">
                    {constraints.map((c, i) => (
                      <div key={i} className="text-[8px] text-muted-foreground/60 px-1 py-0.5 bg-background/20 rounded">
                        {c.description}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Action Buttons */}
            <div className="space-y-2 pt-2">
              <Button
                onClick={handleRunOptimization}
                disabled={isRunning}
                className="w-full h-8 text-[11px] font-semibold"
                size="sm"
              >
                {isRunning ? (
                  <>
                    <RotateCcw className="w-3 h-3 mr-1.5 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Search className="w-3 h-3 mr-1.5" />
                    Run Optimization
                  </>
                )}
              </Button>

              <Button
                onClick={handleHardwareRecommend}
                variant="outline"
                className="w-full h-7 text-[10px] font-medium"
                size="sm"
              >
                <Sparkles className="w-3 h-3 mr-1.5" />
                Hardware Recommendations
              </Button>
            </div>
          </div>
        )}

        {activeTab === 'hardware' && (
          <HardwareRecommendations
            targetHardware={targetHardware}
            onApply={handleApplyRecommendation}
            onBackToOptimizer={() => setActiveTab('optimizer')}
          />
        )}

        {activeTab === 'results' && (
          <ResultsView
            result={result}
            expandedCandidate={expandedCandidate}
            onToggleExpand={setExpandedCandidate}
            onApply={handleApplyConfig}
            onRunAgain={handleRunOptimization}
            isRunning={isRunning}
          />
        )}
      </div>
      </DialogContent>
    </Dialog>
  );
}

// ─── Hardware Recommendations Sub-Component ──────────────────────────

function HardwareRecommendations({
  targetHardware,
  onApply,
  onBackToOptimizer,
}: {
  targetHardware: string;
  onApply: (rec: HardwareRecommendation) => void;
  onBackToOptimizer: () => void;
}) {
  const cap = analyzeHardware(targetHardware);

  if (!cap) {
    return (
      <div className="p-4 text-center">
        <p className="text-[11px] text-muted-foreground">Unknown GPU: {targetHardware}</p>
        <Button onClick={onBackToOptimizer} variant="outline" size="sm" className="mt-2 h-6 text-[10px]">
          Back
        </Button>
      </div>
    );
  }

  return (
    <div className="p-3 space-y-3">
      {/* Hardware Specs */}
      <div className="space-y-1.5">
        <h3 className="text-[10px] font-bold uppercase tracking-wider text-foreground/80 flex items-center gap-1.5">
          <Cpu className="w-3 h-3 text-primary" />
          {getGpuDisplayName(targetHardware)}
        </h3>
        <div className="grid grid-cols-2 gap-1.5">
          {[
            { label: 'Memory', value: `${cap.memoryAnalysis.totalVramGb} GB` },
            { label: 'Bandwidth', value: `${GPU_BW[targetHardware] ?? '?'} GB/s` },
            { label: 'FP16 TFLOPS', value: `${GPU_TF[targetHardware] ?? '?'}` },
            { label: 'Efficiency', value: `${Math.round(cap.efficiencyFactor * 100)}%` },
          ].map((item) => (
            <div key={item.label} className="bg-background/20 rounded p-1.5">
              <div className="text-[7px] text-muted-foreground/60 uppercase tracking-wider">{item.label}</div>
              <div className="text-[11px] font-mono font-bold text-foreground">{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Capacity Analysis */}
      <div className="space-y-1.5">
        <h3 className="text-[10px] font-bold uppercase tracking-wider text-foreground/80">Capacity</h3>
        <div className="space-y-1">
          {[
            { label: 'Max params (fp16)', value: fmtNum(cap.maxParamsFp16) },
            { label: 'Max params (int8)', value: fmtNum(cap.maxParamsInt8) },
            { label: 'Ridge point', value: cap.ridgePoint.toFixed(0) },
            { label: 'Optimal batch', value: String(cap.optimalBatchSize) },
            { label: 'Best for', value: cap.bestFor },
          ].map((item) => (
            <div key={item.label} className="flex items-center justify-between py-0.5">
              <span className="text-[9px] text-muted-foreground">{item.label}</span>
              <span className="text-[9px] font-mono font-medium">{item.value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Recommended Precision */}
      <div className="space-y-1.5">
        <h3 className="text-[10px] font-bold uppercase tracking-wider text-foreground/80">Recommended Precision</h3>
        <Badge
          variant="outline"
          className={cn(
            'text-[10px] font-mono px-2 py-0.5',
            cap.recommendedPrecision === 'bf16' && 'border-blue-500/40 text-blue-400',
            cap.recommendedPrecision === 'fp16' && 'border-green-500/40 text-green-400',
            cap.recommendedPrecision === 'int8' && 'border-orange-500/40 text-orange-400',
            cap.recommendedPrecision === 'fp32' && 'border-muted-foreground/40 text-muted-foreground',
          )}
        >
          {cap.recommendedPrecision.toUpperCase()}
        </Badge>
      </div>

      {/* Recomended Config Button */}
      <div className="pt-2 space-y-2">
        <Button
          onClick={() =>
            onApply({
              gpu: targetHardware,
              capability: cap,
              recommended: {
                precision: cap.recommendedPrecision,
                batchSize: cap.optimalBatchSize,
                hiddenDim: 1024,
                numLayers: 12,
                seqLen: 2048,
                ffnDim: 4096,
              },
              fitAnalysis: {
                estimatedParams: 0,
                estimatedMemoryGb: 0,
                fits: true,
                memoryHeadroomGb: 0,
                utilizationPct: 0,
              },
              bottleneck: '',
              strategy: '',
            })
          }
          className="w-full h-7 text-[10px] font-medium"
          size="sm"
        >
          <Sparkles className="w-3 h-3 mr-1.5" />
          Apply Recommended Config
        </Button>
        <Button onClick={onBackToOptimizer} variant="outline" size="sm" className="w-full h-6 text-[10px]">
          ← Back to Optimizer
        </Button>
      </div>
    </div>
  );
}

// ─── Results View Sub-Component ──────────────────────────────────────

function ResultsView({
  result,
  expandedCandidate,
  onToggleExpand,
  onApply,
  onRunAgain,
  isRunning,
}: {
  result: OptimizationResult | null;
  expandedCandidate: string | null;
  onToggleExpand: (id: string | null) => void;
  onApply: (candidate: CandidateConfig) => void;
  onRunAgain: () => void;
  isRunning: boolean;
}) {
  if (!result || result.candidates.length === 0) {
    return (
      <div className="p-6 text-center">
        <Search className="w-8 h-8 mx-auto text-muted-foreground/30 mb-2" />
        <p className="text-[11px] text-muted-foreground/60">
          Run an optimization to see results here.
        </p>
        <Button
          onClick={onRunAgain}
          variant="outline"
          size="sm"
          disabled={isRunning}
          className="mt-3 h-7 text-[10px]"
        >
          <Play className="w-3 h-3 mr-1" />
          Run Optimization
        </Button>
      </div>
    );
  }

  const topCandidates = result.candidates.slice(0, 20);

  return (
    <div className="p-3 space-y-2">
      {/* Summary */}
      <div className="bg-background/20 rounded p-2 space-y-1">
        <div className="flex items-center justify-between">
          <span className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
            {result.strategy === 'grid_search' ? 'Grid' : result.strategy === 'random_search' ? 'Random' : 'Bayesian'}
          </span>
          <Badge variant="outline" className="text-[8px] px-1.5 h-4">
            {result.evaluatedCount} candidates
          </Badge>
        </div>
        <div className="text-[8px] text-muted-foreground/60">
          Evaluated {result.evaluatedCount} configs in {result.elapsedMs.toFixed(0)}ms
        </div>
      </div>

      {/* Best Score */}
      {result.bestConfig && (
        <div className="bg-primary/5 border border-primary/20 rounded p-2 space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[9px] font-bold uppercase tracking-wider text-primary">
              Best Score: {result.bestConfig.score}/100
            </span>
            <Button
              onClick={() => onApply(result.bestConfig!)}
              size="sm"
              className="h-5 text-[8px] px-2"
            >
              Apply
            </Button>
          </div>
          <div className="flex flex-wrap gap-1">
            {Object.entries(result.bestConfig.config)
              .filter(([_, v]) => v !== undefined && v !== null && v !== 0 && v !== false)
              .slice(0, 6)
              .map(([k, v]) => (
                <Badge key={k} variant="secondary" className="text-[8px] px-1 h-4">
                  {k}={String(v)}
                </Badge>
              ))}
          </div>
        </div>
      )}

      {/* Candidate List */}
      <div className="space-y-1">
        <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">
          Top Configs
        </div>
        {topCandidates.map((candidate) => {
          const isExpanded = expandedCandidate === candidate.id;
          return (
            <div
              key={candidate.id}
              className={cn(
                'rounded border transition-all',
                isExpanded
                  ? 'border-primary/30 bg-primary/5'
                  : 'border-border/30 bg-background/20 hover:border-border/50',
              )}
            >
              <button
                onClick={() => onToggleExpand(isExpanded ? null : candidate.id)}
                className="w-full flex items-center justify-between p-2"
              >
                <div className="flex items-center gap-2">
                  <span className={cn(
                    'w-5 h-5 rounded-full flex items-center justify-center text-[8px] font-bold font-mono',
                    candidate.rank === 1
                      ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                      : candidate.rank === 2
                        ? 'bg-gray-300/20 text-gray-400 border border-gray-300/30'
                        : candidate.rank === 3
                          ? 'bg-orange-500/20 text-orange-400 border border-orange-500/30'
                          : 'bg-background/40 text-muted-foreground border border-border/30',
                  )}>
                    {candidate.rank}
                  </span>
                  <div className="text-left">
                    <div className="text-[9px] font-medium text-foreground">
                      {candidate.estimated.totalParams > 0
                        ? `${fmtNum(candidate.estimated.totalParams)} params`
                        : '?'}
                    </div>
                    <div className="text-[7px] text-muted-foreground/60">
                      {candidate.estimated.peakMemoryGb.toFixed(1)} GB
                      {' · '}
                      {candidate.estimated.latencyMs.toFixed(1)} ms
                      {' · '}
                      {Math.round(candidate.estimated.throughputTokensPerS).toLocaleString()} t/s
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <div className="text-right">
                    <div className="text-[10px] font-bold font-mono">{candidate.score}</div>
                    <div className="text-[7px] text-muted-foreground/60">score</div>
                  </div>
                  <ChevronDown
                    className={cn(
                      'w-3 h-3 text-muted-foreground transition-transform',
                      isExpanded && 'rotate-180',
                    )}
                  />
                </div>
              </button>

              {/* Expanded Details */}
              {isExpanded && (
                <div className="px-2 pb-2 pt-0 space-y-1.5 border-t border-primary/10 mt-0">
                  {/* Config details */}
                  <div className="pt-1.5">
                    <div className="text-[8px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                      Configuration
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(candidate.config)
                        .filter(([_, v]) => v !== undefined && v !== null && v !== 0 && v !== false)
                        .map(([k, v]) => (
                          <Badge key={k} variant="outline" className="text-[7px] px-1 h-3.5 font-mono">
                            {k}: {String(v)}
                          </Badge>
                        ))}
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="grid grid-cols-2 gap-1">
                    {[
                      { label: 'Params', value: fmtNum(candidate.estimated.totalParams) },
                      { label: 'FLOPs', value: fmtNum(candidate.estimated.totalFlops) },
                      { label: 'Memory', value: `${candidate.estimated.peakMemoryGb.toFixed(2)} GB` },
                      { label: 'Latency', value: `${candidate.estimated.latencyMs.toFixed(2)} ms` },
                      { label: 'Throughput', value: `${Math.round(candidate.estimated.throughputTokensPerS).toLocaleString()} t/s` },
                      { label: 'Bottleneck', value: candidate.estimated.bottleneck },
                    ].map((m) => (
                      <div key={m.label} className="bg-background/20 rounded p-1">
                        <div className="text-[7px] text-muted-foreground/60">{m.label}</div>
                        <div className="text-[9px] font-mono font-medium">{m.value}</div>
                      </div>
                    ))}
                  </div>

                  {/* Fits on hardware */}
                  <div className="flex items-center justify-between">
                    <span className="text-[8px] text-muted-foreground">Fits on hardware</span>
                    {candidate.estimated.fitsOnHardware ? (
                      <Badge variant="outline" className="text-[8px] text-green-400 border-green-500/30">
                        <CheckCircle2 className="w-2.5 h-2.5 mr-0.5" /> Yes
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="text-[8px] text-red-400 border-red-500/30">
                        <XCircle className="w-2.5 h-2.5 mr-0.5" /> No
                      </Badge>
                    )}
                  </div>

                  {/* Apply button */}
                  <Button
                    onClick={() => onApply(candidate)}
                    size="sm"
                    className="w-full h-6 text-[9px] font-medium mt-1"
                  >
                    Apply This Configuration
                  </Button>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Build Search Space from Overrides ───────────────────────────────

function buildSearchSpace(
  params: HyperparameterDef[],
  overrides: Partial<Record<keyof HardwareConfig, { min: number; max: number }>>,
): Partial<Record<keyof HardwareConfig, SearchRange>> {
  const space: Partial<Record<keyof HardwareConfig, SearchRange>> = {};

  for (const param of params) {
    const override = overrides[param.key];
    if (override && override.min > 0 && override.max > 0) {
      space[param.key] = {
        min: override.min,
        max: override.max,
        step: param.range?.step,
        logScale: param.range?.logScale,
      };
    }
  }

  return space;
}

// ─── GPU Bandwidth/TFLOPS lookup ─────────────────────────────────────

const GPU_BW: Record<string, number> = {
  'H100-SXM': 3352, 'H100-PCIe': 2000, 'H200': 4800, 'GH200': 4800,
  'A100-SXM': 2039, 'A100-PCIe': 1555, 'L40': 864, 'L40S': 864,
  'V100': 900, 'RTX4090': 1008, 'RTX4080': 717, 'RTX3090': 936,
  'T4': 300, 'A6000': 768,
};

const GPU_TF: Record<string, string> = {
  'H100-SXM': '1979', 'H100-PCIe': '1513', 'H200': '1979', 'GH200': '1979',
  'A100-SXM': '312', 'A100-PCIe': '312', 'L40': '181', 'L40S': '362',
  'V100': '125', 'RTX4090': '165', 'RTX4080': '97', 'RTX3090': '71',
  'T4': '65', 'A6000': '77',
};
