/**
 * Diagnostics, which are the point of the whole tool.
 *
 * The metrics tell you what a design costs. The diagnostics tell you what to do
 * about it, and they are already good — the compiler emits things like:
 *
 *     Critical / MemoryOverflow / E001
 *     "This model needs 125.0 GB but the target GPU has 85.9 GB — 1.5x over.
 *      It will not start."
 *     → "Optimizer state is the largest single term here — ZeRO stage 1 alone
 *        may be enough."
 *
 *     Hint / MemoryOverflow / H001
 *     "Optimizer state is 43% of memory (53.9 GB) — more than the weights."
 *     → "ZeRO stage 1 shards optimizer state across data-parallel ranks and
 *        leaves the maths unchanged."
 *
 * That second one is the whole value proposition: a specific, quantified,
 * actionable way to make the model fit. It was being rendered as an error.
 *
 * The compiler's `Severity` is `Info | Warning | Critical | Hint`, serialised in
 * PascalCase. The client had a list of `critical | error | warning | info`, so
 * `Hint` matched nothing and fell through to the default — `error` in the
 * issues panel. Every optimisation suggestion NEURAX produced was presented as
 * a failure, which is worse than not showing it: a red row is something to make
 * go away, not something to act on.
 *
 * This module is the one place that understands the compiler's vocabulary.
 */

/** The compiler's severities, normalised. Note: there is no `error`. */
export type Severity = 'critical' | 'warning' | 'hint' | 'info';

/** How a severity should read, rank and look. */
export interface SeverityPresentation {
  severity: Severity;
  /** Word shown on the badge. */
  label: string;
  /**
   * What the reader is being asked to do. A hint is an opportunity, not a
   * defect, and must not look like one.
   */
  tone: 'blocking' | 'caution' | 'opportunity' | 'neutral';
  /** Lower sorts first. */
  rank: number;
  /**
   * Colour by what the row means: red stops you, amber warns you, green is a
   * way to make the model better. Defined once here so the issues panel, the
   * diagnostics list and anything added later cannot drift apart — and so the
   * green is never spent on something that is not an opportunity.
   *
   * Both themes are covered: these render on a light and a dark background.
   */
  colors: {
    /** Card border and background. */
    container: string;
    /** Icon and badge text. */
    accent: string;
  };
}

const SEVERITIES: Record<Severity, SeverityPresentation> = {
  critical: {
    severity: 'critical',
    label: 'Blocking',
    tone: 'blocking',
    rank: 0,
    colors: {
      container: 'border-rose-500/35 bg-rose-500/10',
      accent: 'text-rose-600 dark:text-rose-400',
    },
  },
  warning: {
    severity: 'warning',
    label: 'Warning',
    tone: 'caution',
    rank: 1,
    colors: {
      container: 'border-amber-500/35 bg-amber-500/10',
      accent: 'text-amber-600 dark:text-amber-400',
    },
  },
  hint: {
    severity: 'hint',
    label: 'Opportunity',
    tone: 'opportunity',
    rank: 2,
    colors: {
      container: 'border-emerald-500/35 bg-emerald-500/10',
      accent: 'text-emerald-600 dark:text-emerald-400',
    },
  },
  info: {
    severity: 'info',
    label: 'Note',
    tone: 'neutral',
    rank: 3,
    // Deliberately not one of the three: a note is neither a problem nor a
    // gain, and colouring it would dilute the ones that are.
    colors: {
      container: 'border-border/60 bg-secondary/20',
      accent: 'text-muted-foreground',
    },
  },
};

/**
 * Read the compiler's severity, whatever case it arrives in.
 *
 * `error` is accepted although the compiler never emits it, because older
 * reports and hand-written fixtures use it; it means the same as `critical`.
 * Anything unrecognised becomes a warning rather than an error — an unknown
 * severity is not evidence that something is broken, and guessing "blocking"
 * would make every future severity look like a failure.
 */
export function normalizeSeverity(raw: unknown): Severity {
  const value = typeof raw === 'string' ? raw.trim().toLowerCase() : '';

  switch (value) {
    case 'critical':
    case 'error':
    case 'fatal':
      return 'critical';
    case 'warning':
    case 'warn':
      return 'warning';
    case 'hint':
    case 'suggestion':
    case 'opportunity':
      return 'hint';
    case 'info':
    case 'note':
      return 'info';
    default:
      return 'warning';
  }
}

export function severityPresentation(raw: unknown): SeverityPresentation {
  return SEVERITIES[normalizeSeverity(raw)];
}

/**
 * The compiler's categories, in words a user recognises.
 *
 * `DiagnosticCategory` is PascalCase on the wire — `MemoryOverflow`,
 * `ParallelismSuboptimal`. Lowercasing it gives `memoryoverflow`, which is why
 * the category was never displayed at all. These labels are short enough for a
 * badge and say where in the design to look.
 */
const CATEGORIES: Record<string, string> = {
  memoryoverflow: 'Memory',
  bottleneckdetected: 'Bottleneck',
  parallelismsuboptimal: 'Parallelism',
  architectureinefficiency: 'Architecture',
  costalert: 'Cost',
  customlayerfallback: 'Custom block',
  shapeinference: 'Shapes',
  configuration: 'Configuration',
};

/** A readable label for a diagnostic category. */
export function categoryLabel(raw: unknown): string {
  const value = typeof raw === 'string' ? raw.trim() : '';
  if (!value) return 'General';

  const known = CATEGORIES[value.toLowerCase().replace(/[\s_-]/g, '')];
  if (known) return known;

  // An unrecognised category is still worth showing; split PascalCase into
  // words so a new compiler category reads properly without a release here.
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_-]+/g, ' ')
    .replace(/^./, (c) => c.toUpperCase());
}

/** A diagnostic, once normalised. */
export interface Diagnostic {
  severity: Severity;
  category: string;
  categoryLabel: string;
  code?: string;
  message: string;
  suggestion?: string;
  layerId?: string;
  precisionImpact?: number;
}

/** Normalise one diagnostic as it arrives from the compiler. */
export function normalizeDiagnostic(raw: unknown): Diagnostic {
  const d = (raw ?? {}) as Record<string, unknown>;
  const category = typeof d.category === 'string' ? d.category : '';

  return {
    severity: normalizeSeverity(d.severity),
    category,
    categoryLabel: categoryLabel(category),
    code: typeof d.code === 'string' && d.code ? d.code : undefined,
    message:
      typeof d.message === 'string' && d.message.trim()
        ? d.message.trim()
        : 'The compiler reported a problem but gave no description.',
    suggestion:
      typeof d.suggestion === 'string' && d.suggestion.trim() ? d.suggestion.trim() : undefined,
    // `layer_id` is null rather than absent for design-wide diagnostics.
    layerId: typeof d.layer_id === 'string' && d.layer_id ? d.layer_id : undefined,
    precisionImpact: typeof d.precision_impact === 'number' ? d.precision_impact : undefined,
  };
}

/**
 * Normalise and order a diagnostics array.
 *
 * Ordered by what the reader should deal with first: what stops the model
 * running, then what might, then what would make it better, then what is merely
 * true. Within a severity the compiler's own order is kept — it emits them in
 * pass order, which groups related findings.
 */
export function normalizeDiagnostics(raw: unknown): Diagnostic[] {
  if (!Array.isArray(raw)) return [];

  return raw
    .map(normalizeDiagnostic)
    .map((d, index) => ({ d, index }))
    .sort((a, b) => {
      const byRank = SEVERITIES[a.d.severity].rank - SEVERITIES[b.d.severity].rank;
      return byRank !== 0 ? byRank : a.index - b.index;
    })
    .map(({ d }) => d);
}

/** How many of each severity, for a summary line. */
export interface DiagnosticCounts {
  critical: number;
  warning: number;
  hint: number;
  info: number;
  total: number;
}

export function countDiagnostics(diagnostics: Diagnostic[]): DiagnosticCounts {
  const counts: DiagnosticCounts = { critical: 0, warning: 0, hint: 0, info: 0, total: diagnostics.length };
  for (const d of diagnostics) counts[d.severity] += 1;
  return counts;
}

/**
 * A one-line summary of what the compiler found.
 *
 * Written so the most consequential fact leads. "1 blocking issue" is what
 * someone needs to read first; "and 3 opportunities" is what brings them back.
 */
export function summariseDiagnostics(counts: DiagnosticCounts): string {
  if (counts.total === 0) return 'No problems found.';

  const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  const parts: string[] = [];

  if (counts.critical > 0) parts.push(plural(counts.critical, 'blocking issue', 'blocking issues'));
  if (counts.warning > 0) parts.push(plural(counts.warning, 'warning', 'warnings'));
  if (counts.hint > 0) parts.push(plural(counts.hint, 'opportunity', 'opportunities'));
  if (counts.info > 0 && parts.length === 0) parts.push(plural(counts.info, 'note', 'notes'));

  if (parts.length === 1) return `${parts[0]}.`;
  return `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}.`;
}

/** A recommendation, once normalised. */
export interface Recommendation {
  category: string;
  categoryLabel: string;
  title: string;
  description: string;
  /** e.g. "Save ~12.0 GB VRAM" — the reason to act, so never dropped. */
  impact?: string;
  priority: 'high' | 'medium' | 'low';
  rank: number;
}

const PRIORITY_RANK: Record<Recommendation['priority'], number> = { high: 0, medium: 1, low: 2 };

/** Normalise and order recommendations, highest priority first. */
export function normalizeRecommendations(raw: unknown): Recommendation[] {
  if (!Array.isArray(raw)) return [];

  return raw
    .map((entry) => {
      const r = (entry ?? {}) as Record<string, unknown>;
      const priorityRaw = typeof r.priority === 'string' ? r.priority.toLowerCase() : 'medium';
      const priority = (['high', 'medium', 'low'].includes(priorityRaw)
        ? priorityRaw
        : 'medium') as Recommendation['priority'];
      const category = typeof r.category === 'string' ? r.category : '';

      return {
        category,
        categoryLabel: categoryLabel(category),
        title: typeof r.title === 'string' && r.title.trim() ? r.title.trim() : 'Recommendation',
        description: typeof r.description === 'string' ? r.description.trim() : '',
        impact: typeof r.impact === 'string' && r.impact.trim() ? r.impact.trim() : undefined,
        priority,
        rank: PRIORITY_RANK[priority],
      };
    })
    .sort((a, b) => a.rank - b.rank);
}
