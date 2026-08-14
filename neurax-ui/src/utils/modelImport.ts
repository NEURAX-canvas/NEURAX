/**
 * Choosing which importer a pasted file belongs to.
 *
 * There are two things a user drops into the import dialog: a NEURAX design
 * they exported earlier, and a `config.json` from HuggingFace. They are
 * unambiguously different documents, so asking which one it is would be asking
 * the user to do work the file already answers.
 */

import { ImportResult } from '@/utils/architectureImporter.ts';
import { parseArchitectureJSON } from '@/utils/architectureImporter.ts';
import {
  parseHuggingFaceConfig,
  looksLikeHuggingFaceConfig,
  describeHuggingFaceConfig,
} from '@/utils/huggingfaceImporter.ts';

export type ImportSource = 'huggingface' | 'neurax' | 'unknown';

export interface DetectedImport extends ImportResult {
  /** Which importer produced this. */
  source: ImportSource;
  /** What the importer read and inferred; empty for NEURAX designs. */
  notes: string[];
  /** Fields absent from the file, which fell back to a default. */
  assumptions: string[];
  /** e.g. `LlamaForCausalLM`, for the dialog header. */
  detail: string | null;
}

/** Which importer this text belongs to, without parsing it twice downstream. */
export function detectImportSource(jsonString: string): ImportSource {
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonString);
  } catch {
    return 'unknown';
  }

  if (looksLikeHuggingFaceConfig(parsed)) return 'huggingface';

  const model = (parsed as Record<string, unknown> | null)?.model;
  if (model && typeof model === 'object' && Array.isArray((model as Record<string, unknown>).layers)) {
    return 'neurax';
  }

  return 'unknown';
}

/**
 * Parse a model file of either supported kind.
 *
 * `fileName` only supplies a fallback display name, for a config that carries
 * no `_name_or_path` — which is most configs downloaded rather than pushed.
 */
export function parseModelJSON(jsonString: string, fileName?: string): DetectedImport {
  const fallbackName = fileName
    ? fileName.replace(/\.json$/i, '').replace(/[-_]/g, ' ')
    : 'Imported Model';

  const source = detectImportSource(jsonString);

  if (source === 'huggingface') {
    const result = parseHuggingFaceConfig(jsonString, fallbackName);
    return { ...result, source, detail: describeHuggingFaceConfig(jsonString) };
  }

  if (source === 'neurax') {
    const result = parseArchitectureJSON(jsonString);
    return { ...result, source, notes: [], assumptions: [], detail: null };
  }

  // Neither shape matched. Say what was looked for rather than "invalid JSON",
  // which is usually not the problem — the JSON parsed fine, it just is not a
  // model.
  let error =
    'Unrecognised file. Expected either a HuggingFace config.json (with model_type ' +
    'or architectures) or a NEURAX design (with model.layers).';
  try {
    JSON.parse(jsonString);
  } catch (err) {
    error = err instanceof Error ? `Not valid JSON: ${err.message}` : 'Not valid JSON';
  }

  return {
    nodes: [],
    connections: [],
    modelName: fallbackName,
    error,
    source,
    notes: [],
    assumptions: [],
    detail: null,
  };
}
