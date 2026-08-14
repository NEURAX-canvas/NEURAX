/**
 * Turning a failed analysis into something the user can act on.
 *
 * The compiler already says useful things when it refuses a design:
 *
 *     Schema validation error: At least one GPU must be specified
 *     Schema validation error: Model must have at least one layer
 *     JSON parsing error: missing field `schema_version` at line 1 column 2
 *
 * None of it reached anyone. The page tried `JSON.parse` on the body, expecting
 * a diagnostics array; these are plain text, so the parse threw and the user
 * was shown "The backend rejected the current topology (400)." — which names no
 * cause, suggests no fix, and uses a word ("backend") that on the desktop means
 * a thread inside the same process.
 *
 * This module is the translation layer. Three principles:
 *
 *  - **Never discard what the compiler said.** Even when nothing here
 *    recognises a message, it is shown verbatim. An unfamiliar real error beats
 *    a familiar useless one.
 *  - **Say what to do, where the cause is knowable.** "At least one GPU must be
 *    specified" is accurate but describes the schema; the user needs to know
 *    the Target dialog exists.
 *  - **Distinguish the user's mistake from ours.** A design that cannot be
 *    analysed is the user's to fix. An IR that fails to deserialise is a defect
 *    in NEURAX, and saying so — rather than implying the design is at fault —
 *    is the difference between a bug report and an hour lost.
 */

import { NeuraxApiError } from '@/services/neuraxApi.ts';
import { isDesktop } from '@/services/desktopRuntime.ts';
import { normalizeDiagnostics, Diagnostic } from '@/utils/diagnostics.ts';

/** Re-exported so callers need only one import. */
export type CompilerDiagnostic = Diagnostic;

/** A failed analysis, explained. */
export interface AnalysisFailure {
  /** Short, for a toast title. */
  title: string;
  /** One or two sentences naming the cause. */
  detail: string;
  /** What to do next, when that is knowable. */
  hint?: string;
  /**
   * True when the cause is a defect in NEURAX rather than in the design, so the
   * UI can say so instead of sending the user hunting through their blocks.
   */
  internal: boolean;
  /** Structured diagnostics, when the compiler returned them. */
  diagnostics: CompilerDiagnostic[];
  /** The compiler's own words, always preserved. */
  raw?: string;
}

/** What the engine is called, so the wording fits where it is running. */
function engineName(): string {
  // On the desktop the compiler is in this process; calling it "the server"
  // would send someone looking for a service that does not exist.
  return isDesktop() ? 'the NEURAX engine' : 'the NEURAX server';
}

/**
 * Known compiler messages, and what the user should do about them.
 *
 * Matched on a distinctive fragment rather than the whole string, so a reworded
 * message still matches and a new one simply falls through to being shown
 * verbatim.
 */
const KNOWN: Array<{
  match: RegExp;
  title: string;
  detail: string;
  hint: string;
  internal?: boolean;
}> = [
  {
    match: /at least one gpu must be specified/i,
    title: 'No target hardware',
    detail: 'The analysis needs a chip to compute against, and none is selected.',
    hint: 'Open Target in the toolbar and choose a GPU.',
  },
  {
    match: /model must have at least one layer/i,
    title: 'Nothing to analyse',
    detail: 'The design has no blocks.',
    hint: 'Drag a block from the palette, or load a template.',
  },
  {
    match: /missing field `?schema_version`?/i,
    title: 'NEURAX sent an incomplete design',
    detail:
      'The compiler could not read the design NEURAX produced — a required field was missing.',
    hint: 'This is a defect in NEURAX rather than in your architecture. Export the NEURAX IR and attach it to a bug report.',
    internal: true,
  },
  {
    match: /json parsing error|json deserialize error|missing field/i,
    title: 'NEURAX sent a design the compiler could not read',
    detail: 'The description NEURAX produced did not match what the compiler expects.',
    hint: 'This is a defect in NEURAX rather than in your architecture. Export the NEURAX IR and attach it to a bug report.',
    internal: true,
  },
  {
    match: /cycle|circular/i,
    title: 'The design contains a loop',
    detail: 'A path through the blocks returns to where it started, so there is no order to analyse them in.',
    hint: 'Find the connection that closes the loop and remove it. A repeated layer body belongs on a layer stack, not a cycle.',
  },
  {
    match: /shape|dimension mismatch|cannot resolve/i,
    title: 'Shapes do not line up',
    detail: 'Two connected blocks disagree about the size of the tensor between them.',
    hint: 'Check the widths on the blocks named below — a block left at its default width is the usual cause.',
  },
  {
    match: /no input|missing input layer/i,
    title: 'The design has no input',
    detail: 'Nothing marks where data enters the model.',
    hint: 'Add an Input block and connect it to the first layer.',
  },
  {
    match: /no output|missing output layer/i,
    title: 'The design has no output',
    detail: 'Nothing marks where the model produces its result.',
    hint: 'Add an Output block and connect the last layer to it.',
  },
  {
    match: /unsupported|unknown layer|unrecognised block/i,
    title: 'A block the compiler does not know',
    detail: 'The design contains a block this build of the compiler cannot analyse.',
    hint: 'Check the block named below. If it came from an import, the source may use a layer NEURAX does not model yet.',
  },
];

/**
 * Read a diagnostics array out of a parsed JSON body, if there is one.
 *
 * Delegates to the shared normaliser so severities and categories are read the
 * same way here as in the panels — a second copy is how `Hint` came to mean one
 * thing in the issues list and another in the diagnostics list.
 */
function readDiagnostics(value: unknown): Diagnostic[] {
  const container = value as Record<string, unknown> | null;
  const report = (container?.report ?? container) as Record<string, unknown> | undefined;
  return normalizeDiagnostics(report?.diagnostics);
}

/**
 * Strip the compiler's internal prefix from a message.
 *
 * "Schema validation error: Model must have at least one layer" is two things:
 * a category the user does not need, and a sentence they do.
 */
function withoutPrefix(message: string): string {
  return message
    .replace(/^(schema validation error|json parsing error|json deserialize error|validation error|compilation error)\s*:\s*/i, '')
    .trim();
}

/**
 * Explain why an analysis failed.
 *
 * Takes whatever was thrown — an API error, a network failure, anything — and
 * always returns something worth showing. There is no path here that produces
 * "something went wrong".
 */
export function explainAnalysisFailure(error: unknown): AnalysisFailure {
  // ── Not an HTTP response at all: the request never completed ──
  if (!(error instanceof NeuraxApiError)) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      title: 'Could not reach the compiler',
      detail: isDesktop()
        ? 'The analysis engine inside NEURAX did not respond.'
        : 'The request to the NEURAX server did not complete.',
      hint: isDesktop()
        ? 'Restarting NEURAX usually clears this. If it persists, the engine failed to start — run NEURAX from a terminal to see why.'
        : 'Check your connection, then try again.',
      internal: isDesktop(),
      diagnostics: [],
      raw: message,
    };
  }

  const body = typeof error.body === 'string' ? error.body.trim() : '';

  // ── Structured diagnostics, when the compiler sent them ──
  let diagnostics: CompilerDiagnostic[] = [];
  if (body.startsWith('{') || body.startsWith('[')) {
    try {
      diagnostics = readDiagnostics(JSON.parse(body));
    } catch {
      // Not JSON after all; fall through to the text handling below.
    }
  }

  if (diagnostics.length > 0) {
    // Only `critical` blocks. A hint is an optimisation NEURAX is offering, and
    // counting it as an error is what made every suggestion look like a fault.
    const blocking = diagnostics.filter((d) => d.severity === 'critical');
    const first = blocking[0] ?? diagnostics[0];
    return {
      title:
        blocking.length > 0
          ? `Compilation failed — ${blocking.length} error${blocking.length > 1 ? 's' : ''}`
          : `Analysis reported ${diagnostics.length} issue${diagnostics.length > 1 ? 's' : ''}`,
      detail: first.message,
      hint: first.suggestion ?? (first.layerId ? `Reported on block ${first.layerId}.` : undefined),
      internal: false,
      diagnostics,
      raw: body,
    };
  }

  // ── Status codes that mean something specific ──
  switch (error.status) {
    case 401:
      return {
        title: 'Sign in to run an analysis',
        detail: 'This analysis needs an account.',
        internal: false,
        diagnostics: [],
        raw: body || undefined,
      };
    case 403: {
      const verify = /verify/i.test(body) && /email/i.test(body);
      return {
        title: verify ? 'Verify your email first' : 'Your plan does not cover this',
        detail: verify
          ? 'Check your inbox for the verification link, then run the analysis again.'
          : withoutPrefix(body) || 'This analysis is not included in your current plan.',
        internal: false,
        diagnostics: [],
        raw: body || undefined,
      };
    }
    case 429:
      return {
        title: 'Too many analyses',
        detail: 'You have run more analyses than the rate limit allows.',
        hint: 'Wait a moment and try again.',
        internal: false,
        diagnostics: [],
        raw: body || undefined,
      };
  }

  // ── A plain-text message from the compiler ──
  if (body) {
    const cleaned = withoutPrefix(body);

    for (const known of KNOWN) {
      if (known.match.test(body)) {
        return {
          title: known.title,
          detail: known.detail,
          hint: known.hint,
          internal: known.internal ?? false,
          diagnostics: [],
          raw: cleaned,
        };
      }
    }

    // Nothing recognised it. Show exactly what the compiler said — an
    // unfamiliar real message is worth more than a familiar empty one.
    return {
      title: error.status >= 500 ? 'The compiler failed' : 'The compiler rejected this design',
      detail: cleaned,
      hint:
        error.status >= 500
          ? `This is a fault in ${engineName()}, not in your design. Export the NEURAX IR and attach it to a bug report.`
          : undefined,
      internal: error.status >= 500,
      diagnostics: [],
      raw: cleaned,
    };
  }

  // ── A status with no body at all ──
  return {
    title: error.status >= 500 ? 'The compiler failed' : 'The analysis was refused',
    detail:
      error.status >= 500
        ? `${engineName()} returned ${error.status} without saying why.`
        : `${engineName()} refused the request with status ${error.status}.`,
    hint:
      error.status >= 500
        ? 'This is a fault in NEURAX, not in your design. Export the NEURAX IR and attach it to a bug report.'
        : undefined,
    internal: error.status >= 500,
    diagnostics: [],
  };
}

/**
 * The failure as rows for the issues panel.
 *
 * Structured diagnostics become one row each, keeping their block reference and
 * suggestion. A failure without diagnostics still produces a row, so the panel
 * is never empty while a toast claims something went wrong.
 */
export function failureAsWarnings(
  failure: AnalysisFailure,
): Array<{ id: string; type: 'error' | 'warning'; message: string; code?: string }> {
  if (failure.diagnostics.length > 0) {
    return failure.diagnostics.map((d, i) => ({
      id: `diag-${i}`,
      // Only a blocking diagnostic is an error. Warnings, notes and — above all
      // — optimisation hints are not failures, and marking them as such trains
      // the reader to dismiss the rows worth reading.
      type: d.severity === 'critical' ? ('error' as const) : ('warning' as const),
      message: d.layerId ? `${d.message} (block ${d.layerId})` : d.message,
      code: d.code,
    }));
  }

  return [
    {
      id: 'analysis-failed',
      type: 'error',
      message: failure.hint ? `${failure.detail} ${failure.hint}` : failure.detail,
    },
  ];
}
