import { useState, useMemo, useCallback } from 'react';
import { useHardware, HardwareConfig } from '@/contexts/HardwareContext.tsx';
import { ArchitectureFamily } from '@/types/plugins.ts';
import {
  FAMILY_HYPERPARAM_DEFS,
  HyperparameterDef,
  FamilyHyperparameterDefs,
} from '@/utils/hyperparameterDefs.ts';
import { cn } from '@/lib/utils.ts';
import { Input } from '@/components/ui/input.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip.tsx';
import {
  Sliders, Maximize, Minimize, Server, Database, Cpu, HardDrive, Activity,
  BrainCircuit, Gauge, FlaskConical, Layers, Zap, CheckCircle2,
  AlertTriangle, Info, BookOpen, ArrowRight, Sparkles, Boxes, Network,
} from 'lucide-react';

const FAMILY_META: Record<ArchitectureFamily, { label: string; color: string; icon: React.ElementType; blocks: string }> = {
  transformer:  { label: 'Transformer / LLM',  color: '#458588', icon: BrainCircuit, blocks: '~66' },
  moe:          { label: 'Mixture of Experts',  color: '#b16286', icon: Network,     blocks: '~67' },
  cnn:          { label: 'CNN / Vision',        color: '#83a598', icon: Layers,      blocks: '~116' },
  diffusion:    { label: 'Diffusion Models',    color: '#d65d0e', icon: Sparkles,    blocks: '~75' },
  ssm:          { label: 'State Space (Mamba)', color: '#8ec07c', icon: Activity,    blocks: '~97' },
  gnn:          { label: 'Graph Neural Nets',   color: '#fb4934', icon: Boxes,       blocks: '~46' },
  gan:          { label: 'GAN / Adversarial',   color: '#fabd2f', icon: FlaskConical,blocks: '~82' },
  rnn:          { label: 'RNN / LSTM / GRU',    color: '#fe8019', icon: Activity,    blocks: '~70' },
  rl:           { label: 'Reinforcement Learn', color: '#cc241d', icon: Gauge,       blocks: '~12' },
  snn:          { label: 'Spiking Neural Nets', color: '#d3869b', icon: Zap,         blocks: '~12' },
  experimental: { label: 'Experimental',        color: '#8f3f71', icon: FlaskConical,blocks: '∞' },
};

function PriorityBadge({ priority }: { priority: HyperparameterDef['priority'] }) {
  const s = {
    critical: { bg: '#ef444420', color: '#ef4444', label: 'CRITICAL' },
    high:     { bg: '#f59e0b20', color: '#f59e0b', label: 'HIGH' },
    medium:   { bg: '#3b82f620', color: '#3b82f6', label: 'MEDIUM' },
    low:      { bg: '#6b728020', color: '#6b7280', label: 'LOW' },
  }[priority];
  return <span className="text-[8px] font-mono uppercase tracking-[0.12em] px-1.5 py-0.5 rounded"
    style={{ backgroundColor: s.bg, color: s.color }}>{s.label}</span>;
}

function GroupIcon({ group }: { group: string }) {
  const icons: Record<string, React.ElementType> = {
    capacity: Boxes, architecture: BrainCircuit, memory: HardDrive, compute: Cpu,
    training: Activity, regularization: Gauge, data: Database, hardware: Server,
  };
  const Icon = icons[group] ?? Info;
  return <Icon className="w-3 h-3 text-muted-foreground/70" />;
}


interface ParamFieldProps {
  def: HyperparameterDef;
  value: unknown;
  onChange: (key: keyof HardwareConfig, value: unknown) => void;
  missingSet: Set<keyof HardwareConfig>;
}

function ParamField({ def, value, onChange, missingSet }: ParamFieldProps) {
  const isMissing = missingSet.has(def.key as keyof HardwareConfig) && (value === undefined || value === 0 || value === '');
  const isAuto = def.isDerived || (def.key as string).endsWith('Dim') && (def.key as string) !== 'ffnDim' && (def.key as string) !== 'hiddenDim';

  return (
    <div className={cn('rounded-[8px] p-3 transition-all border',
      isMissing ? 'border-destructive/50 bg-destructive/5' : 'border-border/40 bg-card/50',
      isAuto ? 'opacity-60' : '')}>
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5">
          <span className="text-[11px] font-semibold text-foreground">{def.label}</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="w-2.5 h-2.5 text-muted-foreground/40 cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="top" className="text-[10px] max-w-[220px]">
                {def.description}
                {isAuto && def.derivedFormula && <div className="mt-1 text-primary/70">Auto: {def.derivedFormula}</div>}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[8px] font-mono text-muted-foreground/50 uppercase">{def.type}</span>
          <PriorityBadge priority={def.priority} />
        </div>
      </div>
      {def.type === 'bool' ? (
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-muted-foreground/60">{isAuto ? 'Auto-computed' : 'Enable'}</span>
          <Switch checked={Boolean(value ?? def.defaultValue)} onCheckedChange={(v) => onChange(def.key as keyof HardwareConfig, v)} disabled={isAuto} className="scale-75" />
        </div>
      ) : def.type === 'categorical' ? (
        <Select value={String(value ?? def.defaultValue)} onValueChange={(v) => onChange(def.key as keyof HardwareConfig, v)} disabled={isAuto}>
          <SelectTrigger className="h-7 text-[10px] px-2 bg-background/50">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(def.options ?? []).map((opt) => (
              <SelectItem key={String(opt.value)} value={String(opt.value)} className="text-[10px]">{opt.label}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <div className="flex items-center gap-2">
          <Input
            type="number"
            value={value !== undefined && value !== 0 ? String(value) : ''}
            placeholder={isAuto ? 'auto' : String(def.defaultValue ?? '')}
            onChange={(e) => {
              const v = e.target.value;
              onChange(def.key as keyof HardwareConfig, v ? (def.type === 'float' ? parseFloat(v) : parseInt(v, 10)) : 0);
            }}
            disabled={isAuto}
            className="h-7 text-[10px] px-2 bg-background/50"
          />
          {def.range && !isAuto && (
            <span className="text-[8px] text-muted-foreground/40 font-mono whitespace-nowrap">[{def.range.min}..{def.range.max}]</span>
          )}
        </div>
      )}
      {def.constraints && def.constraints.length > 0 && !isAuto && (
        <div className="mt-1.5 space-y-0.5">
          {def.constraints.map((c, i) => (
            <div key={i} className="flex items-start gap-1">
              <AlertTriangle className="w-2 h-2 text-amber-400/70 mt-0.5 shrink-0" />
              <span className="text-[8px] text-muted-foreground/60">{c.description}</span>
            </div>
          ))}
        </div>
      )}
      {isMissing && (
        <div className="mt-1 flex items-center gap-1">
          <AlertTriangle className="w-2.5 h-2.5 text-destructive" />
          <span className="text-[8px] font-medium text-destructive">Required for analysis</span>
        </div>
      )}
    </div>
  );
}


interface ParamGroupProps {
  title: string;
  group: string;
  params: HyperparameterDef[];
  values: Partial<HardwareConfig>;
  onChange: (key: keyof HardwareConfig, value: unknown) => void;
  missingSet: Set<keyof HardwareConfig>;
  defaultExpanded?: boolean;
}

function ParamGroup({ title, group, params, values, onChange, missingSet, defaultExpanded = true }: ParamGroupProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const visibleParams = params.filter((p) => {
    if (p.isDerived) return false;
    if (p.parentKey) {
      const pv = values[p.parentKey as keyof HardwareConfig];
      if (pv !== p.parentValue && p.parentValue !== undefined) return false;
    }
    return true;
  });
  if (visibleParams.length === 0) return null;
  return (
    <div className="space-y-2">
      <button type="button" onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-1 py-1 rounded hover:bg-secondary/20 transition-colors">
        <div className="flex items-center gap-1.5">
          <GroupIcon group={group} />
          <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80">{title}</span>
          <span className="text-[8px] font-mono text-muted-foreground/40">({visibleParams.length})</span>
        </div>
        {expanded ? <Minimize className="w-2.5 h-2.5 text-muted-foreground" /> : <Maximize className="w-2.5 h-2.5 text-muted-foreground" />}
      </button>
      {expanded && (
        <div className="grid gap-2 animate-in fade-in slide-in-from-top-1 duration-200"
          style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))' }}>
          {visibleParams.map((p) => (
            <ParamField key={String(p.key)} def={p} value={values[p.key as keyof HardwareConfig]} onChange={onChange} missingSet={missingSet} />
          ))}
        </div>
      )}
    </div>
  );
}


function NeuraxIrPreview({ config, family }: { config: Partial<HardwareConfig>; family: ArchitectureFamily }) {
  const [showPreview, setShowPreview] = useState(false);
  const previewJson = useMemo(() => {
    const mt = family === 'rl' ? 'rnn' : family === 'snn' ? 'ssm' : family === 'experimental' ? 'transformer' : family;
    return {
      schema_version: '1.0.0',
      model: {
        name: 'My ' + (FAMILY_META[family]?.label ?? family) + ' Model',
        type: mt,
        global_params: {
          hidden_size: config.hiddenDim || undefined,
          num_layers: config.numLayers || undefined,
          num_heads: config.numHeads || undefined,
          vocab_size: config.vocabSize || undefined,
          sequence_length: config.seqLen || undefined,
          ffn_dim: config.ffnDim || undefined,
          num_experts: config.numExperts || undefined,
          top_k: config.topK || undefined,
        },
        layers: [
          { id: 'layer_0', layer_type: 'embedding', params: { hidden_size: config.hiddenDim || 768 } },
          { id: 'layer_1', layer_type: 'attention', params: { hidden_size: config.hiddenDim || 768 } },
        ],
      },
      training: {
        batch_size: config.batchSize || 64,
        precision: config.precision || 'fp16',
        learning_rate: config.learningRate || undefined,
        num_epochs: config.numEpochs || undefined,
        sequence_length: config.seqLen || undefined,
      },
      hardware: {
        gpus: [{ name: config.hardware || 'H100', count: config.gpuCount || 1, memory_gb: config.gpuMemoryGb || 80 }],
      },
    };
  }, [config, family]);
  return (
    <div className="space-y-2">
      <button type="button" onClick={() => setShowPreview(!showPreview)}
        className="w-full flex items-center justify-between px-1 py-1 rounded hover:bg-secondary/20 transition-colors">
        <div className="flex items-center gap-1.5">
          <BookOpen className="w-3 h-3 text-muted-foreground/70" />
          <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80">NeuraxIR Preview</span>
        </div>
        {showPreview ? <Minimize className="w-2.5 h-2.5 text-muted-foreground" /> : <Maximize className="w-2.5 h-2.5 text-muted-foreground" />}
      </button>
      {showPreview && (
        <pre className="text-[8px] font-mono leading-[1.4] p-3 rounded-[8px] bg-background/80 border border-border/40 overflow-auto max-h-[300px] text-muted-foreground/80">
          {JSON.stringify(previewJson, null, 2)}
        </pre>
      )}
    </div>
  );
}


export function HyperparameterInitPanel({ onClose }: { onClose?: () => void }) {
  const { config, updateConfig, triggerAttempt } = useHardware();
  const [selectedFamily, setSelectedFamily] = useState<ArchitectureFamily>('transformer');
  const familyDefs = useMemo(() => {
    return FAMILY_HYPERPARAM_DEFS[selectedFamily] ?? { family: selectedFamily, params: [], globalConstraints: [] } as FamilyHyperparameterDefs;
  }, [selectedFamily]);
  const handleParamChange = useCallback((key: keyof HardwareConfig, value: unknown) => {
    updateConfig({ [key]: value } as Partial<HardwareConfig>);
  }, [updateConfig]);

  const groupedParams = useMemo(() => {
    const groups: { title: string; group: string; params: HyperparameterDef[] }[] = [
      { title: 'Capacity', group: 'capacity', params: [] },
      { title: 'Architecture', group: 'architecture', params: [] },
      { title: 'Memory', group: 'memory', params: [] },
      { title: 'Compute', group: 'compute', params: [] },
      { title: 'Training', group: 'training', params: [] },
      { title: 'Regularization', group: 'regularization', params: [] },
      { title: 'Data', group: 'data', params: [] },
      { title: 'Hardware', group: 'hardware', params: [] },
    ];
    for (const param of familyDefs.params) {
      const g = groups.find((g) => g.group === param.group);
      if (g) g.params.push(param);
    }
    if (!groups.find((g) => g.group === 'hardware')?.params.length) {
      groups[7].params = [
        { key: 'hardware', label: 'GPU Model', description: 'Target GPU hardware', type: 'categorical', defaultValue: 'H100', priority: 'critical', group: 'hardware', options: [{ value: 'H100', label: 'H100' }, { value: 'A100', label: 'A100' }, { value: 'RTX4090', label: 'RTX4090' }, { value: 'V100', label: 'V100' }, { value: 'T4', label: 'T4' }, { value: 'L40', label: 'L40' }, { value: 'RTX4080', label: 'RTX4080' }, { value: 'RTX3090', label: 'RTX3090' }] } as HyperparameterDef,
        { key: 'precision', label: 'Precision', description: 'Training precision', type: 'categorical', defaultValue: 'fp16', priority: 'critical', group: 'hardware', options: [{ value: 'fp32', label: 'FP32' }, { value: 'fp16', label: 'FP16' }, { value: 'bf16', label: 'BF16' }, { value: 'int8', label: 'INT8' }] } as HyperparameterDef,
        { key: 'batchSize', label: 'Batch Size', description: 'Samples per batch', type: 'int', defaultValue: 64, range: { min: 1, max: 65536, logScale: true }, priority: 'critical', group: 'hardware' } as HyperparameterDef,
        { key: 'gpuCount', label: 'GPU Count', description: 'Number of GPUs', type: 'int', defaultValue: 1, range: { min: 1, max: 2048, logScale: true }, priority: 'high', group: 'hardware' } as HyperparameterDef,
      ];
    }
    if (!groups.find((g) => g.group === 'training')?.params.length) {
      groups[4].params = [
        { key: 'learningRate', label: 'Learning Rate', description: 'Optimizer LR', type: 'float', defaultValue: 0.0003, range: { min: 1e-7, max: 1, logScale: true }, priority: 'critical', group: 'training' } as HyperparameterDef,
        { key: 'numEpochs', label: 'Epochs', description: 'Training epochs', type: 'int', defaultValue: 100, range: { min: 1, max: 10000, logScale: true }, priority: 'medium', group: 'training' } as HyperparameterDef,
      ];
    }
    if (!groups.find((g) => g.group === 'data')?.params.length) {
      groups[6].params = [
        { key: 'datasetSize', label: 'Dataset Size', description: 'Total training tokens', type: 'int', defaultValue: 10000000000, range: { min: 1000, max: 1e15, logScale: true }, priority: 'medium', group: 'data' } as HyperparameterDef,
        { key: 'vocabSize', label: 'Vocab Size', description: 'Vocabulary size', type: 'int', defaultValue: 32000, range: { min: 1000, max: 512000, logScale: true }, priority: 'medium', group: 'data' } as HyperparameterDef,
      ];
    }
    return groups.filter((g) => g.params.length > 0);
  }, [familyDefs]);

  const missingSet = useMemo(() => {
    const set = new Set<keyof HardwareConfig>();
    for (const p of familyDefs.params.filter((x) => x.priority === 'critical' && !x.isDerived)) {
      const val = config[p.key as keyof HardwareConfig];
      if (val === undefined || val === 0 || val === '' || val === false) set.add(p.key as keyof HardwareConfig);
    }
    if (!config.hardware) set.add('hardware' as keyof HardwareConfig);
    if (!config.batchSize || config.batchSize <= 0) set.add('batchSize' as keyof HardwareConfig);
    return set;
  }, [config, familyDefs]);

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/60 bg-card/50">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-primary" />
          <span className="text-[13px] font-bold tracking-[-0.02em] text-foreground">Hyperparameter Initialization</span>
        </div>
        {onClose && (
          <button onClick={onClose} className="text-[10px] text-muted-foreground hover:text-foreground transition-colors">Close</button>
        )}
      </div>
      <div className="px-4 py-3 border-b border-border/40">
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(FAMILY_META).map(([fam, meta]) => {
            const Icon = meta.icon;
            const isActive = selectedFamily === fam;
            return (
              <button key={fam} onClick={() => setSelectedFamily(fam as ArchitectureFamily)}
                className={cn('flex items-center gap-1.5 px-2.5 py-1.5 rounded-[6px] text-[10px] font-medium transition-all',
                  isActive ? 'text-white shadow-sm' : 'text-muted-foreground hover:text-foreground bg-secondary/30 hover:bg-secondary/60')}
                style={isActive ? { backgroundColor: meta.color } : {}}>
                <Icon className="w-3 h-3" />
                <span className="hidden sm:inline">{meta.label}</span>
                <span className="text-[8px] font-mono opacity-60">{meta.blocks}</span>
              </button>
            );
          })}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4 scrollbar-thin">
        {groupedParams.map((g) => (
          <ParamGroup key={g.group} title={g.title} group={g.group} params={g.params}
            values={config} onChange={handleParamChange} missingSet={missingSet}
            defaultExpanded={g.group !== 'regularization' && g.group !== 'data'} />
        ))}
        {familyDefs.globalConstraints && familyDefs.globalConstraints.length > 0 && (
          <div className="p-3 rounded-[8px] border border-border/40 bg-amber-500/5">
            <div className="flex items-center gap-1.5 mb-1">
              <AlertTriangle className="w-3 h-3 text-amber-400/70" />
              <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80">Constraints</span>
            </div>
            {familyDefs.globalConstraints.map((c, i) => (
              <div key={i} className="flex items-start gap-1.5">
                <span className="text-[8px] font-mono text-amber-400/70 mt-0.5">↳</span>
                <span className="text-[9px] text-muted-foreground/80">{c.description}</span>
              </div>
            ))}
          </div>
        )}
        <NeuraxIrPreview config={config} family={selectedFamily} />
      </div>
      <div className="border-t border-border/60 bg-card/50 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="text-[9px] text-muted-foreground/60 font-mono">
            {missingSet.size > 0 ? `${missingSet.size} required field(s) missing` : 'All required fields set \u2713'}
          </div>
          <button onClick={() => { triggerAttempt(); onClose?.(); }}
            className={cn('flex items-center gap-1.5 px-4 py-2 rounded-[8px] text-[11px] font-semibold transition-all',
              missingSet.size === 0 ? 'text-white hover:scale-[1.02]' : 'text-muted-foreground/50 cursor-not-allowed')}
            style={missingSet.size === 0 ? { backgroundColor: FAMILY_META[selectedFamily]?.color ?? '#458588' } : { backgroundColor: '#3c383630' }}
            disabled={missingSet.size > 0}>
            <CheckCircle2 className="w-3.5 h-3.5" />
            Apply to Environment
            <ArrowRight className="w-3 h-3" />
          </button>
        </div>
      </div>
    </div>
  );
}

