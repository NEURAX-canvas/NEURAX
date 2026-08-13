/**
 * Every workspace must analyse the model the user is actually designing.
 *
 * Each section had its own way of losing that link: Time Machine compiled the
 * IR with nothing but a model name, Inference started from hardcoded sampling
 * defaults, and Production derived its headline numbers from a fixed 100-epoch
 * baseline. A panel that reports on a different model than the canvas is worse
 * than one that reports nothing.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildDefaultInferenceParams } from './inference/InferenceControls';

const here = dirname(fileURLToPath(import.meta.url));
const read = (p: string) => readFileSync(join(here, p), 'utf8');

describe('inference follows the design', () => {
  it('takes its context length from the design, not a constant', () => {
    const withDesign = buildDefaultInferenceParams('transformer' as any, { seqLen: 4096 } as any);
    expect(withDesign.prompt_length).toBe(4096);
    // Falls back only when the design says nothing.
    expect(buildDefaultInferenceParams('transformer' as any).prompt_length).toBe(2048);
  });

  it('takes its storage width from the design', () => {
    const params = buildDefaultInferenceParams('transformer' as any, { precision: 'int8' } as any);
    expect(params.quantization_level).toBe('int8');
  });

  it('detects grouped-query attention from the design', () => {
    const gqa = buildDefaultInferenceParams('transformer' as any, {
      numHeads: 32, kvHeads: 8,
    } as any);
    expect(gqa.attention_type).toBe('gqa');

    const mha = buildDefaultInferenceParams('transformer' as any, {
      numHeads: 32, kvHeads: 32,
    } as any);
    expect(mha.attention_type).toBe('standard');
  });

  it('leaves room for output within the context', () => {
    const params = buildDefaultInferenceParams('transformer' as any, { seqLen: 4096 } as any);
    expect(params.max_output_tokens).toBeLessThanOrEqual(params.prompt_length);
    expect(params.max_output_tokens).toBeGreaterThan(0);
  });
});

describe('time machine compiles the configured model', () => {
  it('passes the hardware configuration into the IR it projects on', () => {
    const source = read('timemachine/TimeMachineWorkspace.tsx');
    // Compiling with only a model name left precision, batch size and target
    // GPU out of a multi-year cost and carbon projection.
    expect(source).toContain('...hwConfig');
    expect(source).toMatch(/compileToNeuraxIR\(/);
  });

  it('does not declare a prop it never reads', () => {
    const source = read('timemachine/TimeMachineWorkspace.tsx');
    expect(source).not.toContain('analysis?: unknown');
  });
});

describe('production reports what it computes', () => {
  it('no longer advertises savings from a fixed baseline', () => {
    const source = read('production/ProductionWorkspace.tsx');
    // Match the rendered labels, not any mention: the file explains in a
    // comment why these were removed.
    for (const claim of ['Epochs Saved', 'Hours Saved', 'Data Efficiency']) {
      expect(source, `${claim} came from a hardcoded 100-epoch baseline`)
        .not.toContain(`label="${claim}"`);
    }
    // And the fabricated fields themselves are no longer read.
    for (const field of ['estimatedEpochsSaved', 'computeHoursSaved', 'datasetEfficiency']) {
      expect(source, `${field} is derived from invented constants`).not.toContain(field);
    }
  });

  it('reports figures derived from the architecture itself', () => {
    const source = read('production/ProductionWorkspace.tsx');
    expect(source).toContain('Layers initialised');
    expect(source).toContain('Gradient flow');
    // Weight count comes from the real layer shapes.
    expect(source).toMatch(/layer\.shape\.reduce/);
  });
});
