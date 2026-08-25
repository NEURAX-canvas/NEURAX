import {
  Activity,
  AlertTriangle,
  Brain,
  CircleDot,
  Database,
  Eye,
  Gauge,
  Info,
  LineChart,
  Link2,
  Sparkles,
  Target,
  TrendingDown,
  Unlink,
  Zap
} from 'lucide-react';
import { cn } from '@/lib/utils.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip.tsx';
import { Progress } from '@/components/ui/progress.tsx';

import { InferenceReport, StabilityLevel, InferenceRiskLevel } from '@/services/neuraxApi.ts';

/** Compact byte formatting, matching the convention used across the app. */
function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unit]}`;
}

function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return '0';
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(Math.round(value));
}

/**
 * The "what does this mean" affordance every widget in this dashboard should
 * have and, before this change, most didn't: a header title paired with an
 * info icon whose tooltip says what is actually being computed and from
 * which inputs — not a restatement of the title.
 */
function CardHeading({ icon: Icon, title, hint }: { icon: typeof Gauge; title: string; hint: string }) {
  return (
    <h3 className="text-sm font-semibold flex items-center gap-2">
      <Icon className="w-4 h-4 text-primary" />
      {title}
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="w-3 h-3 text-muted-foreground/40 cursor-help" />
        </TooltipTrigger>
        <TooltipContent side="top" className="text-xs max-w-[260px]">
          {hint}
        </TooltipContent>
      </Tooltip>
    </h3>
  );
}

interface BehaviorDashboardProps {
  architectureType: ArchitectureFamily;
  report: InferenceReport | null;
  loading: boolean;
  error: string | null;
}

const stabilityColors: Record<StabilityLevel, string> = {
  stable: 'text-success bg-success/10 border-success/30',
  drift: 'text-warning bg-warning/10 border-warning/30',
  unstable: 'text-destructive bg-destructive/10 border-destructive/30',
  chaotic: 'text-destructive bg-destructive/20 border-destructive/50',
};

const riskColors: Record<InferenceRiskLevel, string> = {
  low: 'bg-success text-success-foreground',
  medium: 'bg-warning text-warning-foreground',
  high: 'bg-destructive text-destructive-foreground',
};

function StabilityGauge({ level }: { level: StabilityLevel }) {
  return (
    <div className="inference-card flex flex-col items-center gap-4">
      <CardHeading
        icon={Gauge}
        title="Generation Stability Index"
        hint="How consistent repeated runs at these settings would be — from temperature, quantization level, beam width and the adversarial-prompt stress test, not from an actual generation. Lower temperature, more beams and higher precision push toward Stable; heavy quantization (int4/int8) and adversarial prompts push toward Chaotic."
      />

      <div className={cn(
        "relative w-36 h-36 rounded-full border-4 flex items-center justify-center",
        stabilityColors[level]
      )}>
        {/* Gauge Arc Background */}
        <svg className="absolute inset-0 w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            strokeDasharray="198 66"
            className="opacity-20"
          />
          <circle
            cx="50"
            cy="50"
            r="42"
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            strokeDasharray={level === 'stable' ? '66 198' : level === 'drift' ? '99 165' : level === 'unstable' ? '132 132' : '198 66'}
            className="opacity-80"
          />
        </svg>
        
        {/* Center Label */}
        <div className="text-center z-10">
          <span className="text-lg font-bold capitalize">{level}</span>
        </div>
      </div>
      
      {/* Legend */}
      <div className="flex gap-2 text-[10px]">
        {(['stable', 'drift', 'unstable', 'chaotic'] as StabilityLevel[]).map((s) => (
          <span 
            key={s} 
            className={cn(
              "px-2 py-0.5 rounded-full border capitalize",
              s === level ? stabilityColors[s] : "text-muted-foreground border-border"
            )}
          >
            {s}
          </span>
        ))}
      </div>
    </div>
  );
}

function EntropyChart({ data }: { data: number[] }) {
  const maxEntropy = Math.max(...data);
  
  return (
    <div className="inference-card">
      <div className="mb-4">
        <CardHeading
          icon={LineChart}
          title="Entropy Evolution"
          hint="Models how the model's next-token uncertainty would decay over a 20-step generation, from temperature, top-p and repetition penalty — not measured from an actual run. A steeper decay means the model settles into confident, repetitive choices sooner."
        />
      </div>

      <div className="h-24 flex items-end gap-1">
        {data.map((value, i) => (
          <Tooltip key={i}>
            <TooltipTrigger asChild>
              <div
                className="flex-1 bg-primary/60 hover:bg-primary/80 rounded-t transition-colors cursor-pointer"
                style={{ height: `${(value / maxEntropy) * 100}%` }}
              />
            </TooltipTrigger>
            <TooltipContent className="text-xs">
              Token {i + 1}: {value.toFixed(2)} entropy
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
      
      <div className="flex justify-between mt-2 text-[10px] text-muted-foreground">
        <span>Token Position 1</span>
        <span>Token Position {data.length}</span>
      </div>
    </div>
  );
}

function NoiseScheduleChart({ data }: { data: number[] }) {
  return (
    <div className="inference-card">
      <div className="mb-4">
        <CardHeading
          icon={Sparkles}
          title="Noise Schedule Curve"
          hint="The denoising trajectory a diffusion model follows from pure noise (t=T) to a clean sample (t=0), on a cosine schedule. This is a fixed property of the diffusion process itself — it doesn't change with your sampling settings."
        />
      </div>

      <div className="h-24 flex items-end gap-1">
        {data.map((value, i) => (
          <Tooltip key={i}>
            <TooltipTrigger asChild>
              <div
                className="flex-1 bg-info/60 hover:bg-info/80 rounded-t transition-colors cursor-pointer"
                style={{ height: `${value * 100}%` }}
              />
            </TooltipTrigger>
            <TooltipContent className="text-xs">
              Step {i + 1}: {value.toFixed(2)} noise
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
      
      <div className="flex justify-between mt-2 text-[10px] text-muted-foreground">
        <span>t=T (noise)</span>
        <span>t=0 (clean)</span>
      </div>
    </div>
  );
}

function HallucinationRiskCard({ risk, confidence }: { risk: InferenceRiskLevel; confidence: number }) {
  const descriptions: Record<InferenceRiskLevel, string> = {
    low: 'Model outputs are grounded in context with minimal fabrication risk.',
    medium: 'Some outputs may deviate from provided context. Verify critical information.',
    high: 'High likelihood of generating unsupported claims. Manual review recommended.',
  };
  
  return (
    <div className="inference-card">
      <h3 className="text-sm font-semibold flex items-center gap-2 mb-3">
        <AlertTriangle className="w-4 h-4 text-primary" />
        Hallucination Risk
      </h3>
      
      <div className="flex items-center gap-3 mb-3">
        <span className={cn("px-3 py-1 rounded-full text-xs font-semibold uppercase", riskColors[risk])}>
          {risk}
        </span>
        <span className="text-xs text-muted-foreground">
          {confidence}% confidence
        </span>
      </div>
      
      <Progress value={confidence} className="h-1.5 mb-3" />
      
      <p className="text-xs text-muted-foreground leading-relaxed">
        {descriptions[risk]}
      </p>
    </div>
  );
}

function AttentionFocusStrip({ attention }: { attention: number[] }) {
  const tokens = ['The', 'model', 'will', 'generate', 'a', 'response', 'based', 'on', 'input', 'context', 'and', 'params'];
  
  return (
    <div className="inference-card">
      <CardHeading
        icon={Eye}
        title="Attention Focus Simulation"
        hint="The focus pattern your selected attention type (flash / linear / standard) produces across a 12-token window — a fixed shape derived from that setting, illustrated on placeholder text below, not attention measured over your actual prompt."
      />

      <div className="token-strip mt-3">
        {tokens.map((token, i) => (
          <span
            key={i}
            className="token-item"
            style={{
              backgroundColor: `hsl(var(--primary) / ${attention[i] || 0.3})`,
              color: attention[i] > 0.6 ? 'hsl(var(--primary-foreground))' : 'hsl(var(--foreground))',
            }}
          >
            {token}
          </span>
        ))}
      </div>

      <div className="flex justify-between mt-2 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-primary/30" /> Low attention
        </span>
        <span className="flex items-center gap-1">
          <div className="w-3 h-3 rounded bg-primary" /> High attention
        </span>
      </div>
      <p className="text-[10px] text-muted-foreground/70 mt-2 italic">
        Illustrative text — shows the pattern shape for your selected attention type, not analysis of a real prompt.
      </p>
    </div>
  );
}

function StateStabilityPanel({ stability }: { stability: number }) {
  return (
    <div className="inference-card">
      <div className="mb-3">
        <CardHeading
          icon={Activity}
          title="State Stability (SSM)"
          hint="How much your prompt length compresses the model's fixed-size hidden state, relative to an 8,192-token reference — longer prompts push this down, and the long-context stress test pushes it down further. Only shown for SSM/Mamba models, which carry state forward instead of attending back over the full sequence."
        />
      </div>

      <div className="flex items-center gap-4">
        <div className="relative w-20 h-20">
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r="42"
              fill="none"
              stroke="hsl(var(--border))"
              strokeWidth="10"
            />
            <circle
              cx="50"
              cy="50"
              r="42"
              fill="none"
              stroke="hsl(var(--success))"
              strokeWidth="10"
              strokeDasharray={`${stability * 264} 264`}
            />
          </svg>
          <span className="absolute inset-0 flex items-center justify-center text-lg font-bold">
            {Math.round(stability * 100)}%
          </span>
        </div>
        
        <div className="flex-1 space-y-2 text-xs">
          <p className="text-muted-foreground">Hidden state coherence across sequence length</p>
          <div className={cn(
            "inline-block px-2 py-0.5 rounded-full text-xs",
            stability > 0.8 ? riskColors.low : stability > 0.5 ? riskColors.medium : riskColors.high
          )}>
            {stability > 0.8 ? 'Stable' : stability > 0.5 ? 'Moderate' : 'Unstable'}
          </div>
        </div>
      </div>
    </div>
  );
}

function ContextDegradationMeter({ percentage }: { percentage: number }) {
  const isWarning = percentage < 50;
  
  return (
    <div className="inference-card">
      <div className="mb-3">
        <CardHeading
          icon={TrendingDown}
          title="Context Degradation"
          hint="Percentage of the context window still free after your prompt plus max output tokens. Uses the connected design's real trained context length, or a 32,768-token placeholder if none is connected. Sliding window and KV-cache reuse ease the pressure; the long-context stress test adds to it."
        />
      </div>

      <div className="flex items-center gap-3">
        <div className="flex-1">
          <div className="flex justify-between text-xs mb-1">
            <span className="text-muted-foreground">Effective Context</span>
            <span className={cn(isWarning ? "text-warning" : "text-foreground")}>
              {percentage}%
            </span>
          </div>
          <Progress 
            value={percentage} 
            className={cn("h-2", isWarning && "[&>div]:bg-warning")} 
          />
        </div>
      </div>
      
      {isWarning && (
        <p className="text-[10px] text-warning mt-2 flex items-center gap-1">
          <AlertTriangle className="w-3 h-3" />
          Context window nearing effective limit
        </p>
      )}
    </div>
  );
}

function SamplingVolatilityCard({ diversity, determinism }: { diversity: number; determinism: number }) {
  return (
    <div className="inference-card">
      <div className="mb-3">
        <CardHeading
          icon={Zap}
          title="Sampling Volatility"
          hint="How much output would vary across repeated runs at these settings, from temperature, top-p, repetition penalty and beam width. Diversity and Determinism are computed independently — they don't need to add to 100%."
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="text-center p-3 bg-secondary/50 rounded-lg cursor-help">
              <div className="text-2xl font-bold text-primary">{(diversity * 100).toFixed(0)}%</div>
              <div className="text-[10px] text-muted-foreground">Diversity</div>
            </div>
          </TooltipTrigger>
          <TooltipContent side="top" className="text-xs max-w-[220px]">
            Likelihood of varied word choices across runs — rises with temperature and top-p, falls with repetition penalty and more beams.
          </TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="text-center p-3 bg-secondary/50 rounded-lg cursor-help">
              <div className="text-2xl font-bold text-primary">{(determinism * 100).toFixed(0)}%</div>
              <div className="text-[10px] text-muted-foreground">Determinism</div>
            </div>
          </TooltipTrigger>
          <TooltipContent side="top" className="text-xs max-w-[220px]">
            Likelihood of the same output each run — rises with more beams, falls with higher temperature and top-p.
          </TooltipContent>
        </Tooltip>
      </div>
    </div>
  );
}

function RouterStabilityCard({ stability, distribution }: { stability: number; distribution: number[] }) {
  return (
    <div className="inference-card">
      <div className="mb-3">
        <CardHeading
          icon={CircleDot}
          title="Router Stability (MoE)"
          hint="How consistently the same tokens would route to the same experts, from your router mode: top-k routing is the most consistent, soft routing spreads load more evenly but shifts more between runs. High temperature lowers it slightly. Expert Distribution below uses the connected design's real expert count and top-k, or 8 experts / top-2 if none is connected."
        />
      </div>

      <div className="flex items-center gap-4 mb-4">
        <div className="text-3xl font-bold text-primary">{(stability * 100).toFixed(0)}%</div>
        <span className="text-xs text-muted-foreground">routing consistency</span>
      </div>
      
      <div className="space-y-2">
        <div className="text-xs text-muted-foreground">Expert Distribution</div>
        <div className="flex gap-1 h-12">
          {distribution.map((val, i) => (
            <Tooltip key={i}>
              <TooltipTrigger asChild>
                <div 
                  className="flex-1 bg-primary/60 hover:bg-primary/80 rounded-t transition-colors cursor-pointer"
                  style={{ height: `${val * 400}%` }}
                />
              </TooltipTrigger>
              <TooltipContent className="text-xs">
                Expert {i + 1}: {(val * 100).toFixed(1)}%
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
        <div className="flex justify-between text-[10px] text-muted-foreground">
          <span>Expert 1</span>
          <span>Expert {distribution.length}</span>
        </div>
      </div>
    </div>
  );
}

/**
 * Widget 11 — KV Cache Cost.
 *
 * The one card in this dashboard that is pure arithmetic rather than a
 * heuristic: bytes per token and total follow directly from the design's real
 * layer count, KV head count and head dimension. It existed on the compiler's
 * side since this endpoint shipped and was never rendered — the most
 * trustworthy number in the report was also the only invisible one.
 */
function KvCacheCard({ bytesPerToken, bytesTotal, gqaSavingsFactor }: {
  bytesPerToken: number; bytesTotal: number; gqaSavingsFactor: number;
}) {
  const isGrouped = gqaSavingsFactor > 1.01;
  return (
    <div className="inference-card">
      <div className="mb-3">
        <CardHeading
          icon={Database}
          title="KV Cache Cost"
          hint="Bytes held in the key/value cache for this request — pure arithmetic from the connected design's real layer count, KV head count and head dimension, times prompt + max output tokens. No heuristic here: this only appears once a design is connected."
        />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="text-center p-3 bg-secondary/50 rounded-lg">
          <div className="text-2xl font-bold text-primary">{formatBytes(bytesTotal)}</div>
          <div className="text-[10px] text-muted-foreground">for this request</div>
        </div>
        <div className="text-center p-3 bg-secondary/50 rounded-lg">
          <div className="text-2xl font-bold text-primary">{formatBytes(bytesPerToken)}</div>
          <div className="text-[10px] text-muted-foreground">per token</div>
        </div>
      </div>

      {isGrouped && (
        <p className="text-xs text-muted-foreground leading-relaxed mt-3">
          Grouped-query attention makes this <strong className="text-foreground">{gqaSavingsFactor.toFixed(1)}&times;</strong> smaller
          than full multi-head attention would cost at the same context length.
        </p>
      )}
    </div>
  );
}

/**
 * Says which mode produced this report — grounded in the real design, or
 * sampling behaviour simulated with no model attached.
 *
 * Without this, two reports that differ only because one had a design
 * connected and the other didn't look identical in shape and different only in
 * their numbers, with nothing on screen explaining why.
 */
function ModelGroundingBanner({ profile }: {
  profile?: InferenceReport['model_profile'];
}) {
  // Presence of `profile` is what "a design is connected" means — not
  // whether any one field happens to be truthy. A field can be a real,
  // meaningful `0` (or just absent on this architecture) without that
  // meaning no design was attached.
  if (profile) {
    const parts = [
      typeof profile.total_parameters === 'number' ? `${formatCompactNumber(profile.total_parameters)} params` : null,
      typeof profile.num_layers === 'number' ? `${profile.num_layers} layers` : null,
      typeof profile.hidden_size === 'number' ? `width ${profile.hidden_size}` : null,
    ].filter(Boolean);
    return (
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-success/10 border border-success/20 text-xs text-success">
        <Link2 className="w-3.5 h-3.5 shrink-0" />
        <span>Grounded in your design{parts.length > 0 ? ` — ${parts.join(', ')}` : ''}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-secondary/40 border border-border text-xs text-muted-foreground">
      <Unlink className="w-3.5 h-3.5 shrink-0" />
      <span>Sampling behaviour only — connect a design on the Architecture tab for model-specific figures</span>
    </div>
  );
}

function InferenceRiskOverview({
  risks,
}: {
  // `collapse` is `null`/`undefined` outside MoE — a dense model has no
  // router to collapse (see `InferenceReport['risk_overview']`'s field doc
  // in neuraxApi.ts). Typed to admit that rather than casting it away,
  // since the row is filtered out below precisely because the value can
  // genuinely be absent.
  risks: Record<string, InferenceRiskLevel | null | undefined>;
}) {
  const riskLabels: Record<string, { label: string; tooltip: string }> = {
    coherence: {
      label: 'Coherence Risk',
      tooltip: 'Rises with higher temperature and narrower beam search — a wide, unfocused sampling distribution with little beam correction is more likely to wander off-topic.',
    },
    overconfidence: {
      label: 'Overconfidence Risk',
      tooltip: 'Rises with lower top-p and low repetition penalty — narrow nucleus sampling with nothing discouraging repeats lets the model lock onto one answer without hedging.',
    },
    collapse: {
      label: 'Collapse Risk (MoE)',
      tooltip: 'Risk of most tokens routing to only a few experts. Depends on router mode: top-k is the most collapse-prone, soft routing the least.',
    },
    degeneration: {
      label: 'Degeneration Risk',
      tooltip: 'Rises with low repetition penalty and high temperature; frequency penalty pulls it back down. High risk means loops or repeated phrases are likely.',
    },
  };
  
  return (
    <div className="inference-card">
      <h3 className="text-sm font-semibold flex items-center gap-2 mb-4">
        <Target className="w-4 h-4 text-primary" />
        Inference Risk Overview
      </h3>
      
      <div className="space-y-2">
        {Object.entries(risks)
          .filter((entry): entry is [string, InferenceRiskLevel] => entry[1] != null)
          .map(([key, level]) => (
          <Tooltip key={key}>
            <TooltipTrigger asChild>
              <div className="flex items-center justify-between p-2 rounded-lg bg-secondary/30 hover:bg-secondary/50 transition-colors cursor-help">
                <span className="text-xs">{riskLabels[key]?.label || key}</span>
                <span className={cn(
                  "px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase",
                  riskColors[level]
                )}>
                  {level}
                </span>
              </div>
            </TooltipTrigger>
            <TooltipContent className="text-xs max-w-[200px]">
              {riskLabels[key]?.tooltip}
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </div>
  );
}

export function BehaviorDashboard({ architectureType, report, loading, error }: BehaviorDashboardProps) {
  return (
    <div className="flex-1 h-full overflow-y-auto scrollbar-thin bg-background p-3 sm:p-4 lg:p-6">
      <div className="max-w-7xl mx-auto space-y-4 sm:space-y-6">
        {/* Header */}
        <div className="flex items-center gap-2 sm:gap-3 pb-3 sm:pb-4 border-b border-border">
          <Brain className="w-5 h-5 sm:w-6 sm:h-6 text-primary shrink-0" />
          <div className="min-w-0 flex-1">
            <h2 className="text-sm sm:text-lg font-semibold truncate">Behavior Prediction Dashboard</h2>
            <p className="text-[10px] sm:text-xs text-muted-foreground">
              Inference behavior analysis for {architectureType.charAt(0).toUpperCase() + architectureType.slice(1)} architecture
            </p>
          </div>
          {loading && (
            <span className="text-[10px] text-muted-foreground animate-pulse shrink-0">Simulating…</span>
          )}
        </div>

        {/* Error state */}
        {error && (
          <div className="p-3 rounded-lg border border-destructive/40 bg-destructive/10 text-xs text-destructive">
            {error}
          </div>
        )}

        {/* Skeleton / not yet loaded */}
        {!report && !error && (
          <div className="p-4 rounded-lg border border-border bg-secondary/20">
            <div className="text-xs text-muted-foreground">
              {loading ? 'Running inference simulation…' : 'Adjust controls to simulate inference behavior.'}
            </div>
          </div>
        )}

        {/* ── Real data ── */}
        {report && (
          <>
            <ModelGroundingBanner profile={report.model_profile} />

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
            {/* Widget 1 — Generation Stability Index */}
            <StabilityGauge level={report.stability_index.level} />

            {/* Widget 2 — Entropy Evolution */}
            <EntropyChart data={report.entropy_evolution} />

            {/* Widget 3 — Noise Schedule (diffusion only) */}
            {report.noise_schedule && <NoiseScheduleChart data={report.noise_schedule} />}

            {/* Widget 4 — Hallucination Risk */}
            <HallucinationRiskCard
              risk={report.hallucination_risk.risk}
              confidence={report.hallucination_risk.confidence}
            />

            {/* Widget 5 — Attention Focus */}
            <AttentionFocusStrip attention={report.attention_focus} />

            {/* Widget 6 — State Stability (SSM/Mamba only). A Transformer,
                MoE or diffusion model has no sequential hidden state for
                this to describe. */}
            {typeof report.state_stability === 'number' && (
              <StateStabilityPanel stability={report.state_stability} />
            )}

            {/* Widget 7 — Context Degradation */}
            <ContextDegradationMeter percentage={report.context_degradation} />

            {/* Widget 8 — Sampling Volatility */}
            <SamplingVolatilityCard
              diversity={report.sampling_volatility.diversity}
              determinism={report.sampling_volatility.determinism}
            />

            {/* Widget 9 — Router Stability (MoE only) */}
            {report.router_stability && (
              <RouterStabilityCard
                stability={report.router_stability.stability}
                distribution={report.router_stability.distribution}
              />
            )}

            {/* Widget 10 — Inference Risk Overview */}
            <InferenceRiskOverview risks={report.risk_overview} />

            {/* Widget 11 — KV Cache Cost. The most trustworthy card here: pure
                arithmetic over the design's real dimensions, no coefficient. */}
            {report.kv_cache && (
              <KvCacheCard
                bytesPerToken={report.kv_cache.bytes_per_token}
                bytesTotal={report.kv_cache.bytes_total}
                gqaSavingsFactor={report.kv_cache.gqa_savings_factor}
              />
            )}
          </div>
          </>
        )}
      </div>
    </div>
  );
}
