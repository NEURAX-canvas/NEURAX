import { useState, useMemo, useCallback } from 'react';
import { useHardware, HardwareConfig } from '@/contexts/HardwareContext.tsx';
import { ArchitectureFamily } from '@/types/plugins.ts';
import {
  FAMILY_HYPERPARAM_DEFS,
  HyperparameterDef,
  FamilyHyperparameterDefs,
  getParamsForFamily,
} from '@/utils/hyperparameterDefs.ts';
import { cn } from '@/lib/utils.ts';
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Switch } from '@/components/ui/switch.tsx';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select.tsx';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip.tsx';
import {
  Sliders, Maximize, Minimize, Server, Database, Cpu, HardDrive, Activity,
  BrainCircuit, Gauge, FlaskConical, Layers, Zap, CheckCircle2,
  AlertTriangle, Info, BookOpen, ArrowRight, Sparkles, Boxes, Network, Plus, Trash2, Search,
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
          {def.required && (
            <span className="text-[8px] font-mono uppercase tracking-[0.12em] px-1.5 py-0.5 rounded bg-destructive/15 text-destructive">
              Required
            </span>
          )}
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



// ─── User-defined hyperparameters ───────────────────────────────────

type CustomValue = string | number | boolean;

/** Interpret a typed input as the value that will be sent to the compiler. */
function coerceCustomValue(raw: string, type: 'number' | 'text' | 'bool'): CustomValue {
  if (type === 'bool') return raw === 'true';
  if (type === 'number') {
    const n = Number(raw);
    return Number.isFinite(n) ? n : 0;
  }
  return raw;
}

/**
 * Add, edit and remove hyperparameters beyond the built-in catalogue.
 *
 * These are forwarded verbatim into the model's `global_params`, where the
 * backend keeps a flattened catch-all, so a design can carry as many extra
 * parameters as its architecture needs.
 */
function CustomParamsSection({
  customParams,
  onChange,
  reservedKeys,
}: {
  customParams: Record<string, CustomValue>;
  onChange: (next: Record<string, CustomValue>) => void;
  reservedKeys: Set<string>;
}) {
  const [draftKey, setDraftKey] = useState('');
  const [draftValue, setDraftValue] = useState('');
  const [draftType, setDraftType] = useState<'number' | 'text' | 'bool'>('number');

  const entries = Object.entries(customParams);
  const trimmedKey = draftKey.trim();

  // Keys must be usable as JSON object keys and must not shadow a built-in,
  // which would silently override the catalogue entry downstream.
  const keyError = (() => {
    if (trimmedKey === '') return null;
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(trimmedKey)) {
      return 'Letters, digits and underscore only, not starting with a digit';
    }
    if (reservedKeys.has(trimmedKey)) return 'A built-in hyperparameter already uses this name';
    if (Object.prototype.hasOwnProperty.call(customParams, trimmedKey)) {
      return 'Already added';
    }
    return null;
  })();

  const canAdd = trimmedKey !== '' && keyError === null;

  const add = () => {
    if (!canAdd) return;
    onChange({ ...customParams, [trimmedKey]: coerceCustomValue(draftValue, draftType) });
    setDraftKey('');
    setDraftValue('');
  };

  const remove = (key: string) => {
    const next = { ...customParams };
    delete next[key];
    onChange(next);
  };

  const edit = (key: string, raw: string) => {
    const existing = customParams[key];
    const type = typeof existing === 'boolean' ? 'bool' : typeof existing === 'number' ? 'number' : 'text';
    onChange({ ...customParams, [key]: coerceCustomValue(raw, type) });
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5 px-1">
        <Plus className="w-3 h-3 text-muted-foreground/70" />
        <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80">
          Custom Hyperparameters
        </span>
        <span className="text-[8px] font-mono text-muted-foreground/40">({entries.length})</span>
      </div>

      <p className="px-1 text-[9px] leading-relaxed text-muted-foreground/60">
        Anything your architecture needs beyond the catalogue. These are sent to the compiler
        inside <span className="font-mono">global_params</span>; there is no limit on how many you add.
      </p>

      {entries.length > 0 && (
        <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))' }}>
          {entries.map(([key, value]) => (
            <div key={key} className="rounded-[8px] p-3 border border-primary/30 bg-primary/5">
              <div className="flex items-center justify-between mb-1.5 gap-2">
                <span className="text-[11px] font-semibold text-foreground font-mono truncate" title={key}>
                  {key}
                </span>
                <div className="flex items-center gap-1 shrink-0">
                  <span className="text-[8px] font-mono text-muted-foreground/50 uppercase">
                    {typeof value}
                  </span>
                  <button
                    type="button"
                    onClick={() => remove(key)}
                    aria-label={`Remove ${key}`}
                    className="p-0.5 rounded hover:bg-destructive/20 text-muted-foreground hover:text-destructive transition-colors"
                  >
                    <Trash2 className="w-2.5 h-2.5" />
                  </button>
                </div>
              </div>
              {typeof value === 'boolean' ? (
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-muted-foreground/60">Enable</span>
                  <Switch
                    checked={value}
                    onCheckedChange={(v) => onChange({ ...customParams, [key]: v })}
                    className="scale-75"
                  />
                </div>
              ) : (
                <Input
                  type={typeof value === 'number' ? 'number' : 'text'}
                  value={String(value)}
                  onChange={(e) => edit(key, e.target.value)}
                  className="h-7 text-[10px] px-2 bg-background/50"
                />
              )}
            </div>
          ))}
        </div>
      )}

      <div className="rounded-[8px] p-3 border border-dashed border-border/60 bg-card/30 space-y-2">
        <div className="flex gap-2">
          <Input
            value={draftKey}
            onChange={(e) => setDraftKey(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && add()}
            placeholder="parameter_name"
            aria-label="Custom hyperparameter name"
            className="h-7 text-[10px] px-2 bg-background/50 font-mono flex-1"
          />
          <Select value={draftType} onValueChange={(v) => setDraftType(v as typeof draftType)}>
            <SelectTrigger className="h-7 text-[10px] px-2 bg-background/50 w-[92px]" aria-label="Value type">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="number" className="text-[10px]">number</SelectItem>
              <SelectItem value="text" className="text-[10px]">text</SelectItem>
              <SelectItem value="bool" className="text-[10px]">boolean</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex gap-2">
          {draftType === 'bool' ? (
            <Select value={draftValue || 'false'} onValueChange={setDraftValue}>
              <SelectTrigger className="h-7 text-[10px] px-2 bg-background/50 flex-1" aria-label="Value">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="true" className="text-[10px]">true</SelectItem>
                <SelectItem value="false" className="text-[10px]">false</SelectItem>
              </SelectContent>
            </Select>
          ) : (
            <Input
              type={draftType === 'number' ? 'number' : 'text'}
              value={draftValue}
              onChange={(e) => setDraftValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && add()}
              placeholder="value"
              aria-label="Custom hyperparameter value"
              className="h-7 text-[10px] px-2 bg-background/50 flex-1"
            />
          )}
          <button
            type="button"
            onClick={add}
            disabled={!canAdd}
            className={cn(
              'flex items-center gap-1 px-3 rounded-[6px] text-[10px] font-semibold transition-all',
              canAdd
                ? 'bg-primary text-primary-foreground hover:opacity-90'
                : 'bg-secondary/40 text-muted-foreground/50 cursor-not-allowed',
            )}
          >
            <Plus className="w-3 h-3" />
            Add
          </button>
        </div>
        {keyError && (
          <div className="flex items-center gap-1">
            <AlertTriangle className="w-2.5 h-2.5 text-destructive" />
            <span className="text-[8px] font-medium text-destructive">{keyError}</span>
          </div>
        )}
      </div>
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


export function ModelHyperparametersPanel({
  onClose,
  initialFamily = 'transformer',
}: {
  onClose?: () => void;
  initialFamily?: ArchitectureFamily;
}) {
  const { config, updateConfig, triggerAttempt } = useHardware();
  const [selectedFamily, setSelectedFamily] = useState<ArchitectureFamily>(initialFamily);
  const [query, setQuery] = useState('');

  // Follow the architecture selected on the canvas. Seeding state from the prop
  // once left the panel showing a previous family's hyperparameters after the
  // design changed underneath it — the user would be editing settings that do
  // not belong to the model they are building.
  const [syncedFamily, setSyncedFamily] = useState(initialFamily);
  if (initialFamily !== syncedFamily) {
    setSyncedFamily(initialFamily);
    setSelectedFamily(initialFamily);
  }
  const familyDefs = useMemo(() => {
    return FAMILY_HYPERPARAM_DEFS[selectedFamily] ?? { family: selectedFamily, params: [], globalConstraints: [] } as FamilyHyperparameterDefs;
  }, [selectedFamily]);

  // `getParamsForFamily` is the single entry point: it deduplicates keys and
  // marks `required` from MANDATORY_FIELDS, so the badges below and the
  // validation that gates analysis are driven by the same source.
  const familyParams = useMemo(() => getParamsForFamily(selectedFamily), [selectedFamily]);

  // Each family now exposes 20-29 parameters, so a name/description filter is
  // what keeps the list usable.
  const matchesQuery = useCallback(
    (p: HyperparameterDef) => {
      const q = query.trim().toLowerCase();
      if (q === '') return true;
      return (
        p.label.toLowerCase().includes(q) ||
        String(p.key).toLowerCase().includes(q) ||
        p.description.toLowerCase().includes(q)
      );
    },
    [query],
  );

  const visibleParams = useMemo(() => familyParams.filter(matchesQuery), [familyParams, matchesQuery]);
  const requiredParams = useMemo(() => visibleParams.filter((p) => p.required), [visibleParams]);
  const optionalParams = useMemo(() => visibleParams.filter((p) => !p.required), [visibleParams]);

  // Counts for the summary bar are over the whole family, not the filtered view.
  const allRequired = useMemo(() => familyParams.filter((p) => p.required), [familyParams]);

  const customParams = (config.customParams ?? {}) as Record<string, string | number | boolean>;
  const handleCustomChange = useCallback(
    (next: Record<string, string | number | boolean>) => updateConfig({ customParams: next }),
    [updateConfig],
  );
  // Built-in names are off-limits for custom entries, across every family, so a
  // design stays valid after switching architecture.
  const reservedKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const defs of Object.values(FAMILY_HYPERPARAM_DEFS)) {
      for (const p of defs.params) keys.add(p.key as string);
    }
    return keys;
  }, []);
  const handleParamChange = useCallback((key: keyof HardwareConfig, value: unknown) => {
    updateConfig({ [key]: value } as Partial<HardwareConfig>);
  }, [updateConfig]);

  const groupedParams = useMemo(() => {
    const order: { title: string; group: string }[] = [
      { title: 'Capacity', group: 'capacity' },
      { title: 'Architecture', group: 'architecture' },
      { title: 'Memory', group: 'memory' },
      { title: 'Compute', group: 'compute' },
      { title: 'Training', group: 'training' },
      { title: 'Regularization', group: 'regularization' },
      { title: 'Data', group: 'data' },
      { title: 'Hardware', group: 'hardware' },
    ];
    return order
      .map((g) => ({ ...g, params: optionalParams.filter((p) => p.group === g.group) }))
      .filter((g) => g.params.length > 0);
  }, [optionalParams]);

  const missingSet = useMemo(() => {
    const set = new Set<keyof HardwareConfig>();
    for (const p of allRequired) {
      const val = (config as unknown as Record<string, unknown>)[p.key as string];
      // A required parameter counts as unset when it is absent, zero or empty;
      // booleans are legitimately false, so they are never "missing".
      if (val === undefined || val === null || val === 0 || val === '') {
        set.add(p.key as keyof HardwareConfig);
      }
    }
    return set;
  }, [config, allRequired]);

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/60 bg-card/50">
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-primary" />
          <span className="text-[13px] font-bold tracking-[-0.02em] text-foreground">Model Hyperparameters</span>
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

        {/* Where the design stands: what still blocks analysis, and how much of
            the family's surface is on screen. */}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground/50 pointer-events-none" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={`Filter ${familyParams.length} hyperparameters...`}
              aria-label="Filter hyperparameters"
              className="h-7 text-[10px] pl-7 pr-2 bg-background/50"
            />
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <span
              className={cn(
                'text-[9px] font-mono px-2 py-1 rounded',
                missingSet.size > 0
                  ? 'bg-destructive/15 text-destructive'
                  : 'bg-emerald-500/15 text-emerald-500',
              )}
            >
              {missingSet.size > 0
                ? `${missingSet.size} of ${allRequired.length} required missing`
                : `${allRequired.length} required set`}
            </span>
            <span className="text-[9px] font-mono px-2 py-1 rounded bg-secondary/40 text-muted-foreground">
              {Object.keys(customParams).length} custom
            </span>
          </div>
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-visible">
        {query.trim() !== '' && visibleParams.length === 0 && (
          <div className="py-8 text-center text-[10px] text-muted-foreground/60">
            No hyperparameter matches "{query.trim()}".
          </div>
        )}
        {requiredParams.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 px-1">
              <AlertTriangle className="w-3 h-3 text-destructive/80" />
              <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/80">
                Required for {FAMILY_META[selectedFamily]?.label ?? selectedFamily}
              </span>
              <span className="text-[8px] font-mono text-muted-foreground/40">({requiredParams.length})</span>
            </div>
            <p className="px-1 text-[9px] leading-relaxed text-muted-foreground/60">
              This family cannot be compiled until these are set; there is no defensible default for them.
            </p>
            <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))' }}>
              {requiredParams.map((p) => (
                <ParamField
                  key={String(p.key)}
                  def={p}
                  value={(config as unknown as Record<string, unknown>)[p.key as string]}
                  onChange={handleParamChange}
                  missingSet={missingSet}
                />
              ))}
            </div>
          </div>
        )}

        {groupedParams.length > 0 && (
          <div className="flex items-center gap-1.5 px-1 pt-1">
            <Info className="w-3 h-3 text-muted-foreground/50" />
            <span className="text-[10px] font-bold uppercase tracking-[0.15em] text-foreground/60">
              Optional
            </span>
            <span className="text-[8px] font-mono text-muted-foreground/40">
              ({optionalParams.length} — each falls back to a documented default)
            </span>
          </div>
        )}
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
        <CustomParamsSection
          customParams={customParams}
          onChange={handleCustomChange}
          reservedKeys={reservedKeys}
        />

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


/**
 * Dialog wrapper so the panel can be opened from the toolbar.
 *
 * `family` follows the architecture selected on the canvas, so the panel opens
 * on the hyperparameters that actually apply to what is being designed.
 */
export function ModelHyperparametersDialog({
  isOpen,
  onClose,
  family,
}: {
  isOpen: boolean;
  onClose: () => void;
  family?: ArchitectureFamily;
}) {
  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-5xl max-h-[88vh] overflow-hidden flex flex-col bg-card p-0 gap-0">
        <DialogTitle className="sr-only">Model Hyperparameters</DialogTitle>
        <ModelHyperparametersPanel onClose={onClose} initialFamily={family} />
      </DialogContent>
    </Dialog>
  );
}
