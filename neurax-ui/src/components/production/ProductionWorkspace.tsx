import { useState, useMemo, useEffect, useRef } from 'react';
import { Leaf, Database, Activity, Info, Check, Copy, Save, Sparkles, ChevronDown, ChevronRight, Settings2, GraduationCap, Layers3 } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Badge } from '@/components/ui/badge.tsx';
import { Slider } from '@/components/ui/slider.tsx';
import { Label } from '@/components/ui/label.tsx';
import { Progress } from '@/components/ui/progress.tsx';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group.tsx';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { cn } from '@/lib/utils.ts';
import { AnalysisResult, CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { HardwareConfig } from '@/contexts/HardwareContext.tsx';
import {
  InitializationMethod,
  InitializationConfig,
  INITIALIZATION_METHODS,
  initializeArchitecture,
  buildInitializationRecord,
  getRecommendedInit,
  getRecommendedHyperparams,
  detectHyperparamFamily,
  HyperparameterConfig,
} from '@/utils/weightInitialization.ts';
import { serializeDesign, suggestedFileName, InitializationRecord } from '@/utils/neuraxFile.ts';
import { saveTextFile } from '@/services/desktopRuntime.ts';

interface ProductionWorkspaceProps {
  nodes: CanvasNode[];
  connections: Connection[];
  modelName: string;
  /**
   * The rest of the design, so Save writes a complete `.neurax` document.
   * Named `architectureFamily` rather than `architecture` — that name is
   * already the computed, initialised weights below.
   */
  architectureFamily: ArchitectureFamily;
  groups?: NodeGroup[];
  hardware?: Partial<HardwareConfig>;
  analysis?: AnalysisResult | null;
  /**
   * Called after a successful Save with the record just written. The host
   * document isn't just this tab's concern — the studio's dirty indicator
   * should clear the same way it does after any other save, and the record
   * needs to be remembered outside this component or the *next* save (a
   * plain Ctrl+S from the Architecture tab, which doesn't know this one ran)
   * would overwrite the file without it, silently dropping what this tab
   * just wrote.
   */
  onSaved?: (initialization: InitializationRecord) => void;
}

/** Compact count: 1.2M rather than 1200000. */
function formatCompact(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0';
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(Math.round(value));
}

export function ProductionWorkspace({
  nodes,
  connections,
  modelName,
  architectureFamily,
  groups = [],
  hardware = {},
  analysis = null,
  onSaved,
}: ProductionWorkspaceProps) {
  const { toast } = useToast();
  const [selectedMethod, setSelectedMethod] = useState<InitializationMethod>('xavier_normal');
  const [gain, setGain] = useState([1.0]);
  const [sparsity, setSparsity] = useState([0.9]);
  const [copied, setCopied] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const recommendedMethod = useMemo(() => getRecommendedInit(nodes), [nodes]);
  // Only a real, completed analysis' count is trustworthy enough to steer a
  // recommendation — the zeroed placeholder shown before the first
  // "Run Analysis" has no real count to give, same `generatedAt` guard
  // `ExportPanel`/`SharePanel` use for the same reason.
  const compiledTotalParams = analysis?.generatedAt ? analysis.totalParams : undefined;
  const recommendedHyperparams = useMemo(
    () => getRecommendedHyperparams(nodes, connections, compiledTotalParams),
    [nodes, connections, compiledTotalParams],
  );
  const hyperparamFamily = useMemo(() => detectHyperparamFamily(nodes), [nodes]);

  // Editable hyperparams, initialised from recommendations.
  const [hyperparams, setHyperparamsState] = useState<HyperparameterConfig>(recommendedHyperparams);

  // Whether the user has touched a hyperparameter field directly, as
  // opposed to it being set by the recommender. Tracked so an edit
  // unrelated to the recommendation — moving a block, tweaking a different
  // layer's params — can't silently discard a deliberate manual choice.
  //
  // This used to sync on every `recommendedHyperparams` change unconditionally,
  // from inside a `useMemo` — a side effect (`setHyperparams`) run from a hook
  // React only guarantees to be pure, and with no way to distinguish "the
  // architecture actually changed family" from "an edit happened to touch
  // `nodes`". A user who nudged Learning Rate then added one unrelated block
  // had their slider silently reset on the next render.
  const hasCustomizedHyperparams = useRef(false);
  const lastHyperparamFamily = useRef(hyperparamFamily);

  useEffect(() => {
    // A genuine family change (e.g. the canvas went from a plain CNN to a
    // GAN) always applies the new recommendation and clears any prior
    // customisation — the old manual tweaks were for a different recipe
    // entirely, keeping them would silently apply a GNN learning rate to a
    // GAN. Anything short of that only re-syncs while nothing has been
    // customised yet.
    const familyChanged = lastHyperparamFamily.current !== hyperparamFamily;
    if (familyChanged || !hasCustomizedHyperparams.current) {
      setHyperparamsState(recommendedHyperparams);
      hasCustomizedHyperparams.current = false;
    }
    lastHyperparamFamily.current = hyperparamFamily;
  }, [recommendedHyperparams, hyperparamFamily]);

  /** Any manual edit — the sliders and the optimizer select all go through
   * this, not `setHyperparamsState` directly, so customisation is tracked
   * consistently rather than at each call site. */
  const setHyperparams = (updater: HyperparameterConfig | ((h: HyperparameterConfig) => HyperparameterConfig)) => {
    hasCustomizedHyperparams.current = true;
    setHyperparamsState(updater);
  };

  /** "Reset to Recommended" explicitly un-customises — the next
   * architecture change should go back to auto-syncing. */
  const resetToRecommended = () => {
    hasCustomizedHyperparams.current = false;
    setHyperparamsState(recommendedHyperparams);
  };

  const architecture = useMemo(() => {
    if (nodes.length === 0) return null;
    const config: InitializationConfig = {
      method: selectedMethod,
      gain: gain[0],
      sparsity: selectedMethod === 'sparse' ? sparsity[0] : undefined,
    };
    return initializeArchitecture(nodes, connections, config, modelName);
  }, [nodes, connections, selectedMethod, gain, sparsity, modelName]);

  /**
   * The `.neurax` document Save and Copy both write — one place, so a field
   * added to it can't reach one and not the other the way `initialization`
   * itself once didn't reach either (this whole export used to be ONNX/Python
   * codegen instead). Returns the record alongside the serialised text so a
   * successful save can hand it to `onSaved` without recomputing it.
   */
  const buildDesignDocument = () => {
    if (!architecture) return null;
    const initialization = buildInitializationRecord(architecture, hyperparams);
    const contents = serializeDesign(
      {
        name: modelName,
        architecture: architectureFamily,
        nodes,
        connections,
        groups,
        hardware,
        analysis,
        initialization,
      },
      { generator: 'NEURAX Production' },
    );
    return { contents, initialization };
  };

  /**
   * Save the design as a `.neurax` file, carrying this initialisation as its
   * `initialization` section.
   *
   * Replaces the old ONNX/Python export, which could not run: its constructor
   * was comments for any real block, its forward pass called every layer with
   * the wrong signature, and it never wrote the computed weights into the file
   * at all. This writes what the panel can actually stand behind — the real
   * per-layer shape and variance it computed, and the recipe to reproduce the
   * weights deterministically — through the same serialiser Save already uses
   * on the Architecture tab.
   */
  const handleSaveDesign = async () => {
    const doc = buildDesignDocument();
    if (!doc) {
      toast({ title: "No Architecture", description: "Add layers to the canvas first", variant: "destructive" });
      return;
    }
    try {
      const result = await saveTextFile(doc.contents, suggestedFileName(modelName), 'application/json');
      if (!result.saved) return; // The user dismissed the dialog.

      onSaved?.(doc.initialization);
      toast({
        title: "Design saved",
        description: result.path ?? suggestedFileName(modelName),
      });
    } catch (err) {
      toast({ title: "Save failed", description: String(err), variant: "destructive" });
    }
  };

  const handleCopyJSON = async () => {
    const doc = buildDesignDocument();
    if (!doc) return;
    const contents = doc.contents;
    try {
      await navigator.clipboard.writeText(contents);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      toast({ title: "Copied", description: "Design copied to clipboard as .neurax JSON" });
    } catch (err) {
      toast({ title: "Copy failed", description: String(err), variant: "destructive" });
    }
  };

  const trainableLayers = nodes.filter(n =>
    ['dense', 'conv2d', 'attention', 'transformer', 'layernorm', 'batchnorm'].includes(n.type)
  );

  return (
    <div className="flex-1 flex flex-col overflow-hidden bg-background">
      {/* Top Toolbar */}
      <div className="border-b border-border bg-card flex flex-col sm:flex-row items-start sm:items-center justify-between px-3 sm:px-6 py-2 sm:py-0 sm:h-14 gap-2 sm:gap-0">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-success/10 flex items-center justify-center shrink-0">
            <Leaf className="w-4 h-4 text-success" />
          </div>
          <div className="min-w-0">
            <h2 className="text-xs sm:text-sm font-semibold flex items-center gap-2">
              <span className="truncate">Training Config Optimizer</span>
              <Badge className="bg-success/20 text-success border-0 text-[10px] shrink-0">GREEN AI</Badge>
            </h2>
            <p className="text-[10px] text-muted-foreground hidden sm:block">Optimal weights, biases & hyperparameters for efficient training</p>
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button variant="outline" size="sm" className="text-xs h-7 sm:h-8" onClick={() => void handleCopyJSON()} disabled={!architecture}>
            {copied ? <Check className="w-3.5 h-3.5 sm:mr-1.5" /> : <Copy className="w-3.5 h-3.5 sm:mr-1.5" />}
            <span className="hidden sm:inline">{copied ? 'Copied!' : 'Copy JSON'}</span>
          </Button>
          <Button size="sm" className="bg-success hover:bg-success/90 text-success-foreground text-xs h-7 sm:h-8" onClick={() => void handleSaveDesign()} disabled={!architecture}>
            <Save className="w-3.5 h-3.5 sm:mr-1.5" />
            <span className="hidden sm:inline">Save Design</span>
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-3 sm:p-4 lg:p-6">
        <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">

          {/* Left Column: Weights & Biases */}
          <div className="space-y-6">
            <SectionHeader icon={Settings2} title="Optimal Weights & Biases" />

            {/* What the chosen initialisation does to this model.
                The three cards here used to read "Epochs Saved", "Hours Saved"
                and "Data Efficiency", all derived from a hardcoded 100-epoch,
                24-hour baseline and invented per-method multipliers — so every
                model, on every GPU, was told it saved the same amount. These
                describe the initialisation itself, which is what this panel
                actually computes. */}
            {architecture && (
              <div className="grid grid-cols-3 gap-3">
                <MetricCard
                  icon={Layers3}
                  label="Layers initialised"
                  value={`${architecture.layers.length}`}
                  color="text-info"
                />
                <MetricCard
                  icon={Database}
                  label="Weights"
                  value={formatCompact(
                    architecture.layers.reduce(
                      (total, layer) => total + layer.shape.reduce((a, b) => a * b, 1),
                      0,
                    ),
                  )}
                  color="text-warning"
                />
                <MetricCard
                  icon={Activity}
                  label="Gradient flow"
                  value={`${architecture.metrics.gradientFlowScore}/100`}
                  color="text-success"
                />
              </div>
            )}

            {/* Init Method Selection */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm font-medium">Initialization Method</Label>
                {recommendedMethod && (
                  <Button variant="ghost" size="sm" className="h-6 text-[10px] text-success hover:text-success" onClick={() => setSelectedMethod(recommendedMethod)}>
                    <Sparkles className="w-3 h-3 mr-1" /> Use Recommended
                  </Button>
                )}
              </div>
              <RadioGroup value={selectedMethod} onValueChange={(v) => setSelectedMethod(v as InitializationMethod)} className="grid grid-cols-2 gap-2">
                {INITIALIZATION_METHODS.slice(0, 6).map((method) => (
                  <Tooltip key={method.id}>
                    <TooltipTrigger asChild>
                      <div
                        className={cn(
                          "relative flex items-start space-x-2 rounded-lg border p-3 cursor-pointer transition-all",
                          selectedMethod === method.id ? "border-success bg-success/5" : "border-border hover:border-success/50"
                        )}
                        onClick={() => setSelectedMethod(method.id)}
                      >
                        <RadioGroupItem value={method.id} id={`prod-${method.id}`} className="mt-0.5" />
                        <div className="flex-1 min-w-0">
                          <Label htmlFor={`prod-${method.id}`} className="text-xs font-medium cursor-pointer flex items-center gap-1">
                            {method.name}
                            {method.id === recommendedMethod && (
                              <Badge variant="outline" className="text-[8px] px-1 py-0 h-4 border-success/50 text-success">BEST</Badge>
                            )}
                          </Label>
                          <p className="text-[10px] text-muted-foreground truncate">{method.bestFor}</p>
                        </div>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-[250px]">
                      <p className="text-xs font-medium">{method.name}</p>
                      <p className="text-[10px] text-muted-foreground">{method.description}</p>
                      <code className="text-[9px] bg-secondary px-1 py-0.5 rounded">{method.formula}</code>
                    </TooltipContent>
                  </Tooltip>
                ))}
              </RadioGroup>
            </div>

            {/* Advanced Options */}
            <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
              <CollapsibleTrigger className="flex items-center gap-2 text-xs text-muted-foreground hover:text-foreground transition-colors">
                {showAdvanced ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                Advanced Options
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-4 pt-3">
                <SliderField label="Gain Factor" value={gain} onChange={setGain} min={0.1} max={3.0} step={0.1} format={(v) => v.toFixed(2)} />
                {selectedMethod === 'sparse' && (
                  <SliderField label="Sparsity" value={sparsity} onChange={setSparsity} min={0.5} max={0.99} step={0.01} format={(v) => `${Math.round(v * 100)}%`} />
                )}
              </CollapsibleContent>
            </Collapsible>

            {/* Layer Summary */}
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">Layers to Initialize ({trainableLayers.length})</Label>
              <div className="flex flex-wrap gap-1.5">
                {trainableLayers.slice(0, 8).map((layer) => (
                  <Badge key={layer.id} variant="outline" className="text-[9px] font-mono">{layer.name}</Badge>
                ))}
                {trainableLayers.length > 8 && <Badge variant="outline" className="text-[9px]">+{trainableLayers.length - 8} more</Badge>}
              </div>
            </div>
          </div>

          {/* Right Column: Hyperparameters & Sustainability */}
          <div className="space-y-6">
            <SectionHeader icon={GraduationCap} title="Training Hyperparameters" />

            <div className="space-y-4 p-4 rounded-lg bg-card border border-border">
              <SliderField label="Learning Rate" value={[hyperparams.learningRate]} onChange={([v]) => setHyperparams(h => ({ ...h, learningRate: v }))} min={0.00001} max={0.01} step={0.00001} format={(v) => v.toExponential(1)} />
              <SliderField label="Dropout" value={[hyperparams.dropout]} onChange={([v]) => setHyperparams(h => ({ ...h, dropout: v }))} min={0} max={0.5} step={0.01} format={(v) => v.toFixed(2)} />
              <SliderField label="Weight Decay" value={[hyperparams.weightDecay]} onChange={([v]) => setHyperparams(h => ({ ...h, weightDecay: v }))} min={0} max={0.1} step={0.001} format={(v) => v.toFixed(3)} />
              <SliderField label="Warmup Steps" value={[hyperparams.warmupSteps]} onChange={([v]) => setHyperparams(h => ({ ...h, warmupSteps: Math.round(v) }))} min={0} max={5000} step={50} format={(v) => String(Math.round(v))} />
              <SliderField label="Gradient Clipping" value={[hyperparams.gradientClipping]} onChange={([v]) => setHyperparams(h => ({ ...h, gradientClipping: v }))} min={0.1} max={5.0} step={0.1} format={(v) => v.toFixed(1)} />

              {/* Family-specific fields — only present when the recommender
                  (`getRecommendedHyperparams`) actually detected that family
                  on the canvas. See its own source comments for the paper
                  each default comes from. */}
              {hyperparams.routerAuxLossCoefficient !== undefined && (
                <SliderField
                  label="Router Aux Loss Coefficient"
                  value={[hyperparams.routerAuxLossCoefficient]}
                  onChange={([v]) => setHyperparams(h => ({ ...h, routerAuxLossCoefficient: v }))}
                  min={0} max={0.1} step={0.001}
                  format={(v) => v.toFixed(3)}
                />
              )}
              {hyperparams.emaDecay !== undefined && (
                <SliderField
                  label="EMA Decay"
                  value={[hyperparams.emaDecay]}
                  onChange={([v]) => setHyperparams(h => ({ ...h, emaDecay: v }))}
                  min={0.9} max={0.9999} step={0.0001}
                  format={(v) => v.toFixed(4)}
                />
              )}
              {hyperparams.discriminatorLearningRate !== undefined && (
                <SliderField
                  label="Discriminator Learning Rate"
                  value={[hyperparams.discriminatorLearningRate]}
                  onChange={([v]) => setHyperparams(h => ({ ...h, discriminatorLearningRate: v }))}
                  min={0.00001} max={0.01} step={0.00001}
                  format={(v) => v.toExponential(1)}
                />
              )}

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label className="text-xs text-muted-foreground">Optimizer</Label>
                </div>
                <Select value={hyperparams.optimizer} onValueChange={(v) => setHyperparams(h => ({ ...h, optimizer: v as HyperparameterConfig['optimizer'] }))}>
                  <SelectTrigger className="h-8 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Adam">Adam</SelectItem>
                    <SelectItem value="AdamW">AdamW</SelectItem>
                    <SelectItem value="SGD">SGD</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button variant="ghost" size="sm" className="w-full text-[10px] text-success hover:text-success" onClick={resetToRecommended}>
                <Sparkles className="w-3 h-3 mr-1" /> Reset to Recommended
              </Button>
            </div>

            {/* Sustainability Metrics */}
            {architecture && (
              <div className="space-y-4 p-4 rounded-lg bg-secondary/30 border border-border">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-success" />
                  <span className="text-sm font-medium">Sustainability Impact</span>
                </div>
                {/* Both meters here read a real, computed property of the
                    generated weights. A "Convergence Boost" card used to sit
                    between them, reading a fixed per-method constant with no
                    relationship to this model — removed rather than relabelled. */}
                <div className="space-y-3">
                  <SustainabilityMeter label="Gradient Flow Score" value={architecture.metrics.gradientFlowScore} max={100} color="success" />
                  {architecture.metrics.memoryOptimization > 0 && (
                    <SustainabilityMeter label="Memory Saved (Sparse)" value={architecture.metrics.memoryOptimization} max={100} suffix="%" color="warning" />
                  )}
                </div>
                <div className="text-[10px] text-muted-foreground flex items-start gap-1.5 pt-2 border-t border-border">
                  <Info className="w-3 h-3 mt-0.5 flex-shrink-0" />
                  <span>Pre-computed weights eliminate random initialization overhead and provide better gradient flow from the start.</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// --- Sub-components ---

function SectionHeader({ icon: Icon, title }: { icon: React.ElementType; title: string }) {
  return (
    <div className="flex items-center gap-2 pb-2 border-b border-border">
      <Icon className="w-4 h-4 text-primary" />
      <h3 className="text-sm font-semibold">{title}</h3>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, color = "text-primary" }: { icon: React.ElementType; label: string; value: string; color?: string }) {
  return (
    <div className="p-3 rounded-lg bg-secondary/50 border border-border text-center">
      <Icon className={cn("w-5 h-5 mx-auto mb-1", color)} />
      <div className={cn("text-lg font-bold", color)}>{value}</div>
      <div className="text-[9px] text-muted-foreground uppercase tracking-wide">{label}</div>
    </div>
  );
}

function SliderField({ label, value, onChange, min, max, step, format }: {
  label: string; value: number[]; onChange: (v: number[]) => void;
  min: number; max: number; step: number; format: (v: number) => string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <span className="text-xs font-mono">{format(value[0])}</span>
      </div>
      <Slider value={value} onValueChange={onChange} min={min} max={max} step={step} className="w-full" />
    </div>
  );
}

function SustainabilityMeter({ label, value, max, suffix = '', color = 'success' }: {
  label: string; value: number; max: number; suffix?: string; color?: 'success' | 'info' | 'warning';
}) {
  const percentage = Math.min(100, (value / max) * 100);
  const colorClasses = {
    success: '[&>div]:bg-success',
    info: '[&>div]:bg-info',
    warning: '[&>div]:bg-warning',
  };
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className={cn("font-medium", `text-${color}`)}>{value}{suffix}</span>
      </div>
      <Progress value={percentage} className={cn("h-1.5", colorClasses[color])} />
    </div>
  );
}
