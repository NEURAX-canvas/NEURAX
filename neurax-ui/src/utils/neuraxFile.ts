/**
 * The `.neurax` document — a design as a file on disk.
 *
 * NEURAX kept projects in an application-owned store: real persistence, but not
 * a document. A design that only exists inside the application cannot be put in
 * a repository beside the training code it describes, cannot be attached to a
 * review, cannot be diffed against last week's, and cannot be handed to someone
 * who then opens it. Those are not extras — they are most of what "working on a
 * model every day" consists of.
 *
 * So the format is chosen for the things a file has to survive:
 *
 *  - **Diffable.** Pretty-printed JSON with one key per line and a stable key
 *    order, so `git diff` shows "num_layers 32 → 48" rather than one enormous
 *    changed line. This is why the file is not minified and why keys are
 *    written in a fixed order rather than whatever order the object happens to
 *    have.
 *  - **Self-describing.** `format` and `version` are the first two keys, so a
 *    reader — this application, a script, a future version — can tell what it
 *    is holding before parsing the rest.
 *  - **Complete.** Everything needed to reconstruct the design exactly: blocks,
 *    connections, groups, the architecture family, and the full hardware and
 *    training configuration. Reopening a file must not silently lose a setting
 *    that was on screen when it was saved.
 *  - **Honest about what is derived.** The last analysis travels with the file
 *    because it is useful to see yesterday's numbers without recomputing — but
 *    it is stored under `analysis` and clearly is an output, recomputed on
 *    demand. It is never read back as input.
 */

import { AnalysisResult, CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';
import { HardwareConfig } from '@/contexts/HardwareContext.tsx';

/** The `format` marker every `.neurax` file carries. */
export const NEURAX_FORMAT = 'neurax-design';

/**
 * Current format version.
 *
 * Bump when a change would make an older reader misinterpret a file, not when
 * a field is merely added — readers ignore fields they do not know, so additive
 * changes cost nothing.
 */
export const NEURAX_FORMAT_VERSION = 1;

/** Conventional extension. `.neurax.json` also opens, and editors highlight it. */
export const NEURAX_EXTENSION = 'neurax';

/** A design, as it exists in a file. */
export interface NeuraxDocument {
  format: typeof NEURAX_FORMAT;
  version: number;
  /** What wrote this file, for when a file outlives the version that made it. */
  generator: string;
  savedAt: string;

  name: string;
  architecture: ArchitectureFamily;

  design: {
    nodes: CanvasNode[];
    connections: Connection[];
    groups: NodeGroup[];
  };

  /** Hardware, training and model hyperparameters — the analysis inputs. */
  hardware: Partial<HardwareConfig>;

  /** Output, not input: the last analysis, kept for reference. */
  analysis?: AnalysisResult | null;
}

/** What the application hands over to be written. */
export interface DesignSnapshot {
  name: string;
  architecture: ArchitectureFamily;
  nodes: CanvasNode[];
  connections: Connection[];
  groups: NodeGroup[];
  hardware: Partial<HardwareConfig>;
  analysis?: AnalysisResult | null;
}

/**
 * Order object keys so the same design always serialises byte-for-byte the
 * same.
 *
 * Without this, a design that has not changed can still produce a different
 * file — JSON.stringify follows insertion order, and a node whose parameters
 * were edited and edited back keeps the later order. A file that changes when
 * nothing changed makes every diff suspect, so ordering is enforced rather than
 * hoped for.
 *
 * Arrays keep their order: the sequence of blocks is meaningful.
 */
function withSortedKeys(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(withSortedKeys);
  if (value && typeof value === 'object') {
    const source = value as Record<string, unknown>;
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(source).sort()) {
      sorted[key] = withSortedKeys(source[key]);
    }
    return sorted;
  }
  return value;
}

/**
 * Serialise a design to the text of a `.neurax` file.
 *
 * `savedAt` is injectable so tests can assert on exact bytes; in the
 * application it is always now.
 */
export function serializeDesign(
  snapshot: DesignSnapshot,
  options: { generator?: string; savedAt?: string } = {},
): string {
  const document: NeuraxDocument = {
    format: NEURAX_FORMAT,
    version: NEURAX_FORMAT_VERSION,
    generator: options.generator ?? 'NEURAX',
    savedAt: options.savedAt ?? new Date().toISOString(),
    name: snapshot.name,
    architecture: snapshot.architecture,
    design: {
      nodes: snapshot.nodes,
      connections: snapshot.connections,
      groups: snapshot.groups,
    },
    hardware: snapshot.hardware,
    analysis: snapshot.analysis ?? null,
  };

  // The header keys stay in written order — they are the first thing a human
  // or a script reads — while everything below them is sorted for stable
  // diffs.
  const ordered = {
    format: document.format,
    version: document.version,
    generator: document.generator,
    savedAt: document.savedAt,
    name: document.name,
    architecture: document.architecture,
    design: withSortedKeys(document.design),
    hardware: withSortedKeys(document.hardware),
    analysis: withSortedKeys(document.analysis),
  };

  return `${JSON.stringify(ordered, null, 2)}\n`;
}

/** Outcome of reading a file: a document, or a reason it could not be read. */
export type ParsedDocument =
  | { ok: true; document: NeuraxDocument; warnings: string[] }
  | { ok: false; error: string };

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value);
}

/**
 * Read the text of a `.neurax` file.
 *
 * Validates enough to guarantee the caller gets a usable design — a file
 * claiming the right format but carrying no blocks array would otherwise crash
 * the canvas rather than report a bad file.
 */
export function parseNeuraxFile(text: string): ParsedDocument {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? `Not valid JSON: ${err.message}` : 'Not valid JSON',
    };
  }

  if (!isRecord(parsed)) {
    return { ok: false, error: 'A .neurax file must contain a JSON object.' };
  }

  if (parsed.format !== NEURAX_FORMAT) {
    // Being specific here matters: the most likely mistake is opening an
    // exported NEURAX IR or a HuggingFace config through the wrong menu item.
    return {
      ok: false,
      error:
        `Not a NEURAX design (its "format" is ${JSON.stringify(parsed.format) ?? 'missing'}). ` +
        'Exported IR and HuggingFace configs go through Import, not Open.',
    };
  }

  const version = typeof parsed.version === 'number' ? parsed.version : 0;
  const warnings: string[] = [];

  if (version > NEURAX_FORMAT_VERSION) {
    // Forward compatibility is a promise this format has not made, so say what
    // is happening rather than opening it and quietly dropping half of it.
    return {
      ok: false,
      error:
        `This file was written by a newer version of NEURAX (format version ${version}, ` +
        `this build reads ${NEURAX_FORMAT_VERSION}). Update NEURAX to open it.`,
    };
  }

  const design = isRecord(parsed.design) ? parsed.design : null;
  if (!design || !Array.isArray(design.nodes)) {
    return { ok: false, error: 'This design has no blocks — the file is incomplete or truncated.' };
  }

  const connections = Array.isArray(design.connections) ? design.connections : [];
  if (!Array.isArray(design.connections)) {
    warnings.push('No connections in the file; opened as unconnected blocks.');
  }

  const groups = Array.isArray(design.groups) ? design.groups : [];

  // Drop connections whose endpoints are not in the file. A dangling edge
  // renders as a line to nowhere and makes the compiler's topology wrong; a
  // file that has been hand-edited is exactly where this happens.
  const nodeIds = new Set(
    (design.nodes as unknown[])
      .filter(isRecord)
      .map((n) => n.id)
      .filter((id): id is string => typeof id === 'string'),
  );
  const liveConnections = (connections as unknown[])
    .filter(isRecord)
    .filter((c) => typeof c.from === 'string' && typeof c.to === 'string')
    .filter((c) => nodeIds.has(c.from as string) && nodeIds.has(c.to as string));

  const dropped = connections.length - liveConnections.length;
  if (dropped > 0) {
    warnings.push(
      `${dropped} ${dropped === 1 ? 'connection refers' : 'connections refer'} to blocks that are not in the file; dropped.`,
    );
  }

  const document: NeuraxDocument = {
    format: NEURAX_FORMAT,
    version,
    generator: typeof parsed.generator === 'string' ? parsed.generator : 'unknown',
    savedAt: typeof parsed.savedAt === 'string' ? parsed.savedAt : '',
    name: typeof parsed.name === 'string' && parsed.name ? parsed.name : 'Untitled',
    architecture: (typeof parsed.architecture === 'string'
      ? parsed.architecture
      : 'transformer') as ArchitectureFamily,
    design: {
      nodes: design.nodes as CanvasNode[],
      connections: liveConnections as unknown as Connection[],
      groups: groups as NodeGroup[],
    },
    hardware: isRecord(parsed.hardware) ? (parsed.hardware as Partial<HardwareConfig>) : {},
    analysis: isRecord(parsed.analysis) ? (parsed.analysis as unknown as AnalysisResult) : null,
  };

  return { ok: true, document, warnings };
}

/**
 * A filename for a design, safe on every platform NEURAX ships to.
 *
 * Windows rejects `<>:"/\|?*`, and a name that is only punctuation leaves
 * nothing to type in a file dialog, so an unusable name falls back to a
 * generic one rather than producing a file the user cannot find.
 */
export function suggestedFileName(name: string): string {
  const cleaned = name
    .trim()
    // Characters Windows refuses outright, and control characters.
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, '')
    // Whitespace becomes a hyphen, so the name survives a shell without quotes.
    .replace(/\s+/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^[.-]+|[.-]+$/g, '')
    .slice(0, 80);

  return `${cleaned || 'untitled-design'}.${NEURAX_EXTENSION}`;
}
