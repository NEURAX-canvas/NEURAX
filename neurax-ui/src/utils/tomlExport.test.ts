/**
 * The TOML export exists to be a faithful copy of the JSON one, not merely
 * a plausible-looking one — so the check here is a real round trip, not a
 * string match: compile a real design, export it both ways, parse the TOML
 * back, and assert the two describe the same object.
 */
import { describe, it, expect } from 'vitest';
import { parse } from 'smol-toml';
import { toToml } from './tomlExport';
import { compileToNeuraxIR } from './neuraxCompiler';
import { CanvasNode, Connection } from '@/types/architecture.ts';

const nodes: CanvasNode[] = [
  { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: { sequence_length: 2048 } },
  {
    id: 'n2',
    type: 'token_embedding',
    name: 'Token Embedding',
    x: 250,
    y: 140,
    params: { vocab_size: 32000, hidden_size: 4096 },
  },
  {
    id: 'n3',
    type: 'gqa_attention',
    name: 'GQA',
    x: 450,
    y: 140,
    params: { hidden_size: 4096, num_heads: 32, num_kv_heads: 8, causal: true },
  },
];
const connections: Connection[] = [
  { id: 'c1', from: 'n1', to: 'n2' },
  { id: 'c2', from: 'n2', to: 'n3' },
];

describe('toToml', () => {
  it('round-trips a real compiled design losslessly', () => {
    const ir = compileToNeuraxIR(nodes, connections, {
      modelName: 'test-model',
      family: 'transformer',
    });

    const toml = toToml(ir);
    const parsedBack = parse(toml);

    // The same object JSON would describe, since both are derived from one
    // JSON.stringify/parse pass — this is the property that makes the two
    // exports interchangeable rather than two different descriptions of the
    // design that could quietly drift apart.
    const jsonRoundTrip = JSON.parse(JSON.stringify(ir));
    expect(parsedBack).toEqual(jsonRoundTrip);
  });

  it('renders an array of layer objects as TOML array-of-tables, not an inline array', () => {
    const ir = compileToNeuraxIR(nodes, connections, {
      modelName: 'test-model',
      family: 'transformer',
    });
    const toml = toToml(ir);
    expect(toml).toMatch(/\[\[model\.layers\]\]/);
    // An inline array-of-objects would instead show up as `layers = [{`.
    expect(toml).not.toMatch(/layers\s*=\s*\[\s*\{/);
  });

  it('drops null fields rather than erroring on them', () => {
    expect(() => toToml({ a: 1, b: null, c: { d: null, e: 2 } })).not.toThrow();
    const toml = toToml({ a: 1, b: null, c: { d: null, e: 2 } });
    expect(parse(toml)).toEqual({ a: 1, c: { e: 2 } });
  });

  it('is real TOML a standard parser accepts, not merely TOML-shaped text', () => {
    const ir = compileToNeuraxIR(nodes, connections, {
      modelName: 'test-model',
      family: 'transformer',
    });
    expect(() => parse(toToml(ir))).not.toThrow();
  });
});
