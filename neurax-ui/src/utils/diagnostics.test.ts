/**
 * The compiler's vocabulary, read correctly.
 *
 * Every fixture here was captured from a running `neurax-service`. That is the
 * whole reason this module exists: the compiler serialises `Severity` in
 * PascalCase and emits `Hint` for its optimisation suggestions, while the
 * client had a list of `critical | error | warning | info`. `Hint` matched
 * nothing, fell through to the default, and every suggestion NEURAX produced
 * was rendered as a red error.
 *
 * That inverts the meaning of the most valuable output the tool has. A hint
 * saying "ZeRO stage 1 would make this fit" is not a fault to be silenced; it
 * is the answer.
 */
import { describe, it, expect } from 'vitest';
import {
  normalizeSeverity,
  severityPresentation,
  categoryLabel,
  normalizeDiagnostic,
  normalizeDiagnostics,
  countDiagnostics,
  summariseDiagnostics,
  normalizeRecommendations,
} from './diagnostics';

/** Captured verbatim from `/analyze` on a LLaMA-2 7B that does not fit. */
const REAL_DIAGNOSTICS = [
  {
    severity: 'Critical',
    category: 'MemoryOverflow',
    code: 'E001',
    message:
      'This model needs 125.0 GB but the target GPU has 85.9 GB — 1.5x over. It will not start.',
    layer_id: null,
    suggestion: 'Optimizer state is the largest single term here — ZeRO stage 1 alone may be enough.',
    precision_impact: 0,
  },
  {
    severity: 'Hint',
    category: 'MemoryOverflow',
    code: 'H001',
    message: 'Optimizer state is 43% of memory (53.9 GB) — more than the weights.',
    layer_id: null,
    suggestion:
      'ZeRO stage 1 shards optimizer state across data-parallel ranks and leaves the maths unchanged.',
    precision_impact: 0,
  },
];

/** Captured from the same response. */
const REAL_RECOMMENDATIONS = [
  {
    category: 'MemoryOptimization',
    title: 'Enable Gradient Checkpointing',
    description: 'Reduce activation memory by recomputing during backward pass',
    impact: 'Save ~12.0 GB VRAM',
    priority: 'High',
  },
];

describe('reading the compiler\'s severities', () => {
  it('reads PascalCase, which is how they arrive', () => {
    expect(normalizeSeverity('Critical')).toBe('critical');
    expect(normalizeSeverity('Warning')).toBe('warning');
    expect(normalizeSeverity('Hint')).toBe('hint');
    expect(normalizeSeverity('Info')).toBe('info');
  });

  it('keeps a Hint a hint, not an error', () => {
    // The regression this module was written for.
    const hint = normalizeSeverity('Hint');
    expect(hint).toBe('hint');
    expect(hint).not.toBe('error');
    expect(severityPresentation('Hint').tone).toBe('opportunity');
    expect(severityPresentation('Hint').label).toBe('Opportunity');
  });

  it('accepts `error` from older reports as critical', () => {
    // The compiler has no `Error` variant, but fixtures and old reports use it.
    expect(normalizeSeverity('error')).toBe('critical');
  });

  it('treats an unknown severity as a warning, not as a failure', () => {
    // Guessing "blocking" would make every severity added later look like a
    // fault, and paint the panel red for something nobody understands yet.
    expect(normalizeSeverity('Perplexing')).toBe('warning');
    expect(normalizeSeverity(undefined)).toBe('warning');
    expect(normalizeSeverity(42)).toBe('warning');
  });

  it('is insensitive to case and surrounding space', () => {
    expect(normalizeSeverity('  CRITICAL ')).toBe('critical');
    expect(normalizeSeverity('hint')).toBe('hint');
  });
});

describe('colouring a diagnostic by what it means', () => {
  it('paints a blocking issue red', () => {
    const look = severityPresentation('Critical');
    expect(look.colors.container).toMatch(/rose/);
    expect(look.colors.accent).toMatch(/rose/);
  });

  it('paints a warning amber', () => {
    expect(severityPresentation('Warning').colors.container).toMatch(/amber/);
  });

  it('paints an optimisation opportunity green', () => {
    // Green because acting on it makes the model better — the opposite of a
    // fault, and it must not read as one.
    const look = severityPresentation('Hint');
    expect(look.colors.container).toMatch(/emerald/);
    expect(look.colors.accent).toMatch(/emerald/);
  });

  it('leaves a plain note uncoloured, so the three that matter still stand out', () => {
    const look = severityPresentation('Info');
    for (const colour of ['rose', 'amber', 'emerald']) {
      expect(look.colors.container).not.toMatch(colour);
      expect(look.colors.accent).not.toMatch(colour);
    }
  });

  it('gives every severity a colour for both themes', () => {
    for (const severity of ['Critical', 'Warning', 'Hint', 'Info']) {
      const look = severityPresentation(severity);
      expect(look.colors.container.length).toBeGreaterThan(0);
      expect(look.colors.accent.length).toBeGreaterThan(0);
    }
  });
});

describe('reading the compiler\'s categories', () => {
  it('turns PascalCase into something readable', () => {
    // Lowercasing gave `memoryoverflow`, which is why the category was never
    // displayed at all.
    expect(categoryLabel('MemoryOverflow')).toBe('Memory');
    expect(categoryLabel('ParallelismSuboptimal')).toBe('Parallelism');
    expect(categoryLabel('CustomLayerFallback')).toBe('Custom block');
    expect(categoryLabel('ShapeInference')).toBe('Shapes');
  });

  it('covers every category the compiler defines', () => {
    // From `DiagnosticCategory` in neurax-ir/src/lib.rs.
    const fromRust = [
      'MemoryOverflow',
      'BottleneckDetected',
      'ParallelismSuboptimal',
      'ArchitectureInefficiency',
      'CostAlert',
      'CustomLayerFallback',
      'ShapeInference',
      'Configuration',
    ];
    for (const category of fromRust) {
      const label = categoryLabel(category);
      // Not the fallback, and never left as raw PascalCase run together —
      // `Configuration` is legitimately already its own label.
      expect(label, category).not.toBe('General');
      expect(label, category).not.toMatch(/[a-z][A-Z]/);
      expect(label.length, category).toBeGreaterThan(0);
    }
  });

  it('splits an unknown category into words rather than hiding it', () => {
    // A category added to the compiler should read properly without a release
    // here.
    expect(categoryLabel('QuantisationRisk')).toBe('Quantisation Risk');
  });

  it('falls back to General when there is no category', () => {
    expect(categoryLabel('')).toBe('General');
    expect(categoryLabel(undefined)).toBe('General');
  });
});

describe('normalising a diagnostic', () => {
  it('reads a real one whole', () => {
    const d = normalizeDiagnostic(REAL_DIAGNOSTICS[0]);
    expect(d.severity).toBe('critical');
    expect(d.categoryLabel).toBe('Memory');
    expect(d.code).toBe('E001');
    expect(d.message).toMatch(/125.0 GB/);
    expect(d.suggestion).toMatch(/ZeRO stage 1/);
  });

  it('treats a null layer_id as design-wide rather than as a block named "null"', () => {
    expect(normalizeDiagnostic(REAL_DIAGNOSTICS[0]).layerId).toBeUndefined();
    expect(normalizeDiagnostic({ layer_id: 'n7' }).layerId).toBe('n7');
  });

  it('never produces an empty message', () => {
    for (const broken of [{}, { message: '' }, { message: '   ' }, null, undefined]) {
      expect(normalizeDiagnostic(broken).message.length).toBeGreaterThan(0);
    }
  });

  it('keeps the suggestion, which is the reason the row exists', () => {
    expect(normalizeDiagnostic(REAL_DIAGNOSTICS[1]).suggestion).toMatch(/shards optimizer state/);
  });
});

describe('ordering diagnostics by what to deal with first', () => {
  it('puts blocking issues above opportunities', () => {
    const mixed = [
      { severity: 'Hint', message: 'h' },
      { severity: 'Info', message: 'i' },
      { severity: 'Critical', message: 'c' },
      { severity: 'Warning', message: 'w' },
    ];
    expect(normalizeDiagnostics(mixed).map((d) => d.severity)).toEqual([
      'critical',
      'warning',
      'hint',
      'info',
    ]);
  });

  it('keeps the compiler\'s order within a severity', () => {
    const same = [
      { severity: 'Hint', message: 'first' },
      { severity: 'Hint', message: 'second' },
      { severity: 'Hint', message: 'third' },
    ];
    expect(normalizeDiagnostics(same).map((d) => d.message)).toEqual(['first', 'second', 'third']);
  });

  it('returns nothing for anything that is not an array', () => {
    for (const junk of [null, undefined, {}, 'diagnostics', 7]) {
      expect(normalizeDiagnostics(junk)).toEqual([]);
    }
  });
});

describe('summarising what was found', () => {
  it('counts each severity separately', () => {
    const counts = countDiagnostics(normalizeDiagnostics(REAL_DIAGNOSTICS));
    expect(counts).toEqual({ critical: 1, warning: 0, hint: 1, info: 0, total: 2 });
  });

  it('leads with what blocks, and still mentions what would help', () => {
    const counts = countDiagnostics(normalizeDiagnostics(REAL_DIAGNOSTICS));
    expect(summariseDiagnostics(counts)).toBe('1 blocking issue and 1 opportunity.');
  });

  it('does not call an opportunity a problem', () => {
    const hintsOnly = countDiagnostics(
      normalizeDiagnostics([
        { severity: 'Hint', message: 'a' },
        { severity: 'Hint', message: 'b' },
      ]),
    );
    const summary = summariseDiagnostics(hintsOnly);
    expect(summary).toBe('2 opportunities.');
    expect(summary).not.toMatch(/error|issue|problem|fail/i);
  });

  it('says so plainly when there is nothing to report', () => {
    expect(summariseDiagnostics(countDiagnostics([]))).toBe('No problems found.');
  });

  it('gets the grammar right for one and for many', () => {
    const one = countDiagnostics(normalizeDiagnostics([{ severity: 'Critical', message: 'x' }]));
    expect(summariseDiagnostics(one)).toBe('1 blocking issue.');

    const three = countDiagnostics(
      normalizeDiagnostics([
        { severity: 'Critical', message: 'a' },
        { severity: 'Warning', message: 'b' },
        { severity: 'Hint', message: 'c' },
      ]),
    );
    expect(summariseDiagnostics(three)).toBe('1 blocking issue, 1 warning and 1 opportunity.');
  });
});

describe('recommendations', () => {
  it('reads a real one, keeping its quantified impact', () => {
    const [rec] = normalizeRecommendations(REAL_RECOMMENDATIONS);
    expect(rec.title).toBe('Enable Gradient Checkpointing');
    expect(rec.priority).toBe('high');
    // The number is the reason to act; dropping it leaves advice with no weight.
    expect(rec.impact).toBe('Save ~12.0 GB VRAM');
  });

  it('orders the highest priority first', () => {
    const mixed = [
      { title: 'c', priority: 'Low' },
      { title: 'a', priority: 'High' },
      { title: 'b', priority: 'Medium' },
    ];
    expect(normalizeRecommendations(mixed).map((r) => r.title)).toEqual(['a', 'b', 'c']);
  });

  it('defaults an unknown priority to medium rather than dropping the row', () => {
    const [rec] = normalizeRecommendations([{ title: 'x', priority: 'Urgent' }]);
    expect(rec.priority).toBe('medium');
    expect(rec.title).toBe('x');
  });

  it('survives a malformed entry', () => {
    const recs = normalizeRecommendations([null, {}, { title: 'ok' }]);
    expect(recs).toHaveLength(3);
    for (const rec of recs) expect(rec.title.length).toBeGreaterThan(0);
  });
});
