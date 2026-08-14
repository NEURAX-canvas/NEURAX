/**
 * The initialisation math was always correct. What it ran on was not.
 *
 * `calculateFans` and `getWeightShape` used to switch on four literal type
 * strings — `dense`, `conv2d`, `attention`, `transformer` — that no reference
 * template or import has emitted since the block catalogue moved to specific
 * names (`gqa_attention`, `ffn_gated`, `rmsnorm`, `token_embedding`). Every
 * real block fell through to a flat 512×512 default, so a LLaMA-2 7B layer at
 * width 4096 was initialised with the variance meant for width 512 — the
 * Xavier formula computed exactly right, for a model that does not exist.
 *
 * These tests build real, LLaMA-shaped nodes — the same block types and
 * parameter names the HuggingFace importer and the reference templates
 * actually emit — and assert the resolved dimensions match the model, not the
 * old fallback. A regression here means every layer silently drops back to
 * 512×512, on every real design, with no error to notice it by.
 */
import { describe, it, expect } from 'vitest';
import {
  initializeArchitecture,
  buildInitializationRecord,
  getRecommendedInit,
  getRecommendedHyperparams,
  isTrainableBlock,
  InitializationConfig,
  HyperparameterConfig,
} from './weightInitialization';
import { CanvasNode, Connection } from '@/types/architecture.ts';

/** A LLaMA-2-7B-shaped decoder block — real block types, real param names. */
const LLAMA_NODES: CanvasNode[] = [
  { id: 'n1', type: 'input', name: 'Input', x: 0, y: 0, params: { sequence_length: 4096 } },
  {
    id: 'n2',
    type: 'token_embedding',
    name: 'Token Embedding',
    x: 200,
    y: 0,
    params: { vocab_size: 32000, hidden_size: 4096 },
  },
  {
    id: 'n3',
    type: 'gqa_attention',
    name: 'GQA',
    x: 400,
    y: 0,
    params: { hidden_size: 4096, num_heads: 32, num_kv_heads: 8, head_dim: 128, causal: true },
  },
  {
    id: 'n4',
    type: 'ffn_gated',
    name: 'SwiGLU FFN',
    x: 600,
    y: 0,
    params: { hidden_size: 4096, intermediate_size: 11008, activation: 'silu' },
  },
  {
    id: 'n5',
    type: 'rmsnorm',
    name: 'RMSNorm',
    x: 800,
    y: 0,
    params: { hidden_size: 4096, eps: 1e-5 },
  },
  {
    id: 'n6',
    type: 'lm_head',
    name: 'LM Head',
    x: 1000,
    y: 0,
    params: { vocab_size: 32000, hidden_size: 4096 },
  },
];

const LLAMA_CONNECTIONS: Connection[] = [
  { id: 'c1', from: 'n1', to: 'n2' },
  { id: 'c2', from: 'n2', to: 'n3' },
  { id: 'c3', from: 'n3', to: 'n4' },
  { id: 'c4', from: 'n4', to: 'n5' },
  { id: 'c5', from: 'n5', to: 'n6' },
];

function layerFor(id: string, config: InitializationConfig) {
  const architecture = initializeArchitecture(LLAMA_NODES, LLAMA_CONNECTIONS, config, 'LLaMA-test');
  const layer = architecture.layers.find((l) => l.layerId === id);
  if (!layer) throw new Error(`no layer for ${id} — is it in TRAINABLE_BLOCK_KINDS?`);
  return layer;
}

const XAVIER: InitializationConfig = { method: 'xavier_normal', gain: 1.0 };

describe('resolving real dimensions from real blocks', () => {
  it('resolves attention width from the block\'s own hidden_size, not a 512 default', () => {
    const layer = layerFor('n3', XAVIER);
    // Attention projects to a combined Q/K/V width: d_model in, 3*d_model out.
    expect(layer.fanIn).toBe(4096);
    expect(layer.fanOut).toBe(4096 * 3);
    expect(layer.shape).toEqual([4096, 4096 * 3]);
  });

  it('resolves a gated FFN\'s width to its real intermediate_size, not d_model', () => {
    const layer = layerFor('n4', XAVIER);
    expect(layer.fanIn).toBe(4096);
    expect(layer.fanOut).toBe(11008); // the real d_ff, aliased from intermediate_size
  });

  it('resolves a norm layer to one value per real channel', () => {
    const layer = layerFor('n5', XAVIER);
    expect(layer.shape).toEqual([4096]);
    expect(layer.fanIn).toBe(4096);
    expect(layer.fanOut).toBe(4096);
  });

  it('resolves the LM head to the model\'s real vocabulary, not a guess', () => {
    const layer = layerFor('n6', XAVIER);
    expect(layer.fanIn).toBe(4096);
    expect(layer.fanOut).toBe(32000);
  });

  it('resolves token embedding to the real vocabulary and width', () => {
    const layer = layerFor('n2', XAVIER);
    expect(layer.fanIn).toBe(4096);
    expect(layer.fanOut).toBe(32000);
  });

  it('does not fall back to 512 for any layer in a real design', () => {
    const architecture = initializeArchitecture(LLAMA_NODES, LLAMA_CONNECTIONS, XAVIER, 'LLaMA-test');
    for (const layer of architecture.layers) {
      expect(layer.fanIn, `${layer.layerName} fanIn`).not.toBe(512);
    }
  });

  it('resolves a convolution from its own channel and kernel parameters', () => {
    const convNodes: CanvasNode[] = [
      { id: 'c1', type: 'input', name: 'Input', x: 0, y: 0, params: {} },
      {
        id: 'c2',
        type: 'conv2d',
        name: 'Conv',
        x: 100,
        y: 0,
        params: { in_channels: 3, out_channels: 64, kernel_size: 7 },
      },
    ];
    const architecture = initializeArchitecture(convNodes, [], XAVIER, 'CNN-test');
    const layer = architecture.layers.find((l) => l.layerId === 'c2')!;
    expect(layer.fanIn).toBe(3 * 7 * 7);
    expect(layer.fanOut).toBe(64 * 7 * 7);
    expect(layer.shape).toEqual([64, 3, 7, 7]);
  });

  it('still falls back gracefully for a block with no stated width, rather than crashing', () => {
    const bare: CanvasNode[] = [
      { id: 'b1', type: 'dense', name: 'Bare', x: 0, y: 0, params: {} },
    ];
    const architecture = initializeArchitecture(bare, [], XAVIER, 'bare-test');
    expect(architecture.layers[0].fanIn).toBe(512);
    expect(architecture.layers[0].fanOut).toBe(512);
  });
});

describe('initialising a real model completes quickly', () => {
  // A regression guard, not a benchmark: the specific bug this pins was a
  // norm layer at real width falling into a matrix path meant for a 2-D
  // shape, triggering an O(n³) Gram-Schmidt over a 4096×4096 matrix under
  // `orthogonal` — tens of seconds to minutes, on the method every RNN or
  // very-deep-network recommendation points users toward. A generous bound
  // catches "this now takes forever" without being sensitive to machine
  // speed.
  it('initialises a full LLaMA-shaped design under every method within 2 seconds', () => {
    const methods: InitializationConfig['method'][] = [
      'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal',
      'orthogonal', 'lsuv', 'sparse', 'delta_orthogonal',
    ];
    for (const method of methods) {
      const start = performance.now();
      initializeArchitecture(LLAMA_NODES, LLAMA_CONNECTIONS, { method, gain: 1.0, sparsity: 0.9 }, 'perf-test');
      const elapsed = performance.now() - start;
      expect(elapsed, `${method} took ${elapsed.toFixed(0)}ms`).toBeLessThan(2000);
    }
  });
});

describe('the initialisation formulas, on the real resolved shape', () => {
  it('computes Xavier normal variance from the published formula: 2·gain²/(fanIn+fanOut)', () => {
    const layer = layerFor('n5', { method: 'xavier_normal', gain: 1.0 }); // norm: 4096/4096
    const expected = (2 * 1.0 * 1.0) / (4096 + 4096);
    expect(layer.variance).toBeCloseTo(expected, 10);
  });

  it('computes He normal variance from the published formula: 2·gain²/fanIn', () => {
    const layer = layerFor('n5', { method: 'he_normal', gain: Math.sqrt(2) });
    const expected = (2 * 2) / 4096;
    expect(layer.variance).toBeCloseTo(expected, 10);
  });

  it('uniform and normal variants of the same family target the same variance', () => {
    // Xavier's uniform bound is a = gain·√(6/(fanIn+fanOut)), chosen so that
    // Var(U(-a,a)) = a²/3 lands on exactly the normal variant's 2·gain²/(fanIn+fanOut)
    // target — they are two ways of hitting the same variance, not two
    // different variances. Same relationship for He. A stray extra ÷3 on the
    // uniform branches once made them report a third of the correct value.
    const xavierUniform = layerFor('n5', { method: 'xavier_uniform', gain: 1.0 });
    const xavierNormal = layerFor('n5', { method: 'xavier_normal', gain: 1.0 });
    expect(xavierUniform.variance).toBeCloseTo(xavierNormal.variance, 10);

    const heUniform = layerFor('n5', { method: 'he_uniform', gain: Math.sqrt(2) });
    const heNormal = layerFor('n5', { method: 'he_normal', gain: Math.sqrt(2) });
    expect(heUniform.variance).toBeCloseTo(heNormal.variance, 10);
  });

  it('delta-orthogonal reports unit variance, not the layer width', () => {
    // Delta-orthogonal places an orthogonal matrix at the kernel's centre tap
    // and zeroes the rest — the orthogonal block is what propagates signal,
    // so the effective variance stays at the orthogonal case. Reporting the
    // layer's width instead (a real model's is in the thousands) used to
    // drive `calculateSustainabilityMetrics`'s gradient-flow score to zero
    // for any design containing one.
    const layer = layerFor('n3', { method: 'delta_orthogonal' }); // width 4096, 12288
    expect(layer.variance).toBe(1.0);
  });

  it('reports the real resolved shape for every method, on a 2-D layer', () => {
    const methods: InitializationConfig['method'][] = [
      'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal',
      'orthogonal', 'lsuv', 'sparse', 'delta_orthogonal',
    ];
    for (const method of methods) {
      // n3 (attention, shape [4096, 12288]) rather than n5 (a 1-D norm layer,
      // covered separately below).
      const layer = layerFor('n3', { method, gain: 1.0, sparsity: 0.9 });
      expect(layer.shape, method).toEqual([4096, 12288]);
      expect(Number.isFinite(layer.variance), method).toBe(true);
    }
  });

  it('reports a 1-D shape for a norm layer, for every method', () => {
    // The regression this pins: a norm layer's shape is [width], one
    // dimension. The matrix-generation path this used to go through read the
    // missing second dimension as `fanOut` — the model's real width once
    // dimensions were resolved correctly — and would have materialised (and,
    // for `orthogonal`, Gram-Schmidt'd) a width×width matrix. No matrix is
    // built for any method now; see "completes quickly" above for the
    // performance half of this regression.
    const methods: InitializationConfig['method'][] = [
      'xavier_uniform', 'xavier_normal', 'he_uniform', 'he_normal',
      'orthogonal', 'lsuv', 'sparse', 'delta_orthogonal',
    ];
    for (const method of methods) {
      const layer = layerFor('n5', { method, gain: 1.0, sparsity: 0.9 });
      expect(layer.shape, method).toEqual([4096]);
      expect(Number.isFinite(layer.variance), method).toBe(true);
    }
  });

  it('sparse initialisation reduces variance by the configured fraction', () => {
    // No sampled array to inspect any more — the sparsity fraction shows up
    // in the closed-form variance instead: `(1 - sparsity)` of the dense
    // variance, since a sparse fraction of the weights are exactly zero.
    const dense = layerFor('n5', { method: 'xavier_normal', sparsity: 0 });
    const sparse90 = layerFor('n5', { method: 'sparse', sparsity: 0.9 });
    expect(sparse90.variance).toBeCloseTo(dense.variance * 0.1, 10);
  });
});

describe('what the panel can honestly report', () => {
  it('keeps only metrics grounded in the real generated weights', () => {
    const architecture = initializeArchitecture(LLAMA_NODES, LLAMA_CONNECTIONS, XAVIER, 'LLaMA-test');
    // Exhaustive key check: a fabricated field added back in would show up
    // here even if nothing reads it yet.
    expect(Object.keys(architecture.metrics).sort()).toEqual(
      ['gradientFlowScore', 'memoryOptimization'].sort(),
    );
  });

  it('scores gradient flow near 100 when the variance lands close to 1', () => {
    // Xavier/He initialisation targets unit variance across layers by
    // construction; the score should reward getting there, not just exist.
    const architecture = initializeArchitecture(
      [{ id: 'x', type: 'rmsnorm', name: 'x', x: 0, y: 0, params: { hidden_size: 2 } }],
      [],
      { method: 'lsuv' }, // lsuv targets variance 1.0 directly
      'unit-variance-test',
    );
    expect(architecture.metrics.gradientFlowScore).toBeGreaterThanOrEqual(95);
  });

  it('reports memory saved only for sparse initialisation, and echoes the real sparsity', () => {
    const dense = initializeArchitecture(LLAMA_NODES, LLAMA_CONNECTIONS, XAVIER, 'x');
    expect(dense.metrics.memoryOptimization).toBe(0);

    const sparse = initializeArchitecture(
      LLAMA_NODES, LLAMA_CONNECTIONS, { method: 'sparse', sparsity: 0.75 }, 'x',
    );
    expect(sparse.metrics.memoryOptimization).toBe(75);
  });

  it('never divides by zero into NaN when there are no trainable layers', () => {
    const architecture = initializeArchitecture(
      [{ id: 'i', type: 'input', name: 'Input', x: 0, y: 0, params: {} }],
      [],
      XAVIER,
      'empty-test',
    );
    expect(architecture.layers).toEqual([]);
    expect(Number.isNaN(architecture.metrics.gradientFlowScore)).toBe(false);
    expect(architecture.metrics.gradientFlowScore).toBe(0);
  });
});

describe('the .neurax export replacing the broken ONNX/Python one', () => {
  const hyperparams: HyperparameterConfig = {
    learningRate: 0.0003,
    dropout: 0.1,
    weightDecay: 0.01,
    warmupSteps: 500,
    optimizer: 'AdamW',
    gradientClipping: 1.0,
  };

  it('carries the recipe — method and config — not the generated random weights', () => {
    const architecture = initializeArchitecture(LLAMA_NODES, LLAMA_CONNECTIONS, XAVIER, 'LLaMA-test');
    const record = buildInitializationRecord(architecture, hyperparams);

    expect(record.method).toBe('xavier_normal');
    expect(record.gain).toBe(1.0);
    expect(record.hyperparameters).toEqual(hyperparams);
    // The record type has no field for the weight arrays themselves — this
    // assertion is really about the type, but checking the runtime value
    // keeps it honest against a future field added without updating the type.
    expect((record as unknown as { weights?: unknown }).weights).toBeUndefined();
  });

  it('carries every real per-layer fact the panel computed', () => {
    const architecture = initializeArchitecture(LLAMA_NODES, LLAMA_CONNECTIONS, XAVIER, 'LLaMA-test');
    const record = buildInitializationRecord(architecture, hyperparams);

    expect(record.layers).toHaveLength(architecture.layers.length);
    const attn = record.layers.find((l) => l.layerId === 'n3')!;
    expect(attn.shape).toEqual([4096, 12288]);
    expect(attn.fanIn).toBe(4096);
    expect(attn.fanOut).toBe(12288);
    expect(typeof attn.variance).toBe('number');
  });
});

describe('recommendations still work over a real design', () => {
  it('recommends xavier for attention-bearing architectures', () => {
    expect(getRecommendedInit(LLAMA_NODES)).toBe('xavier_normal');
  });

  it('recommends AdamW when the design has attention', () => {
    expect(getRecommendedHyperparams(LLAMA_NODES, LLAMA_CONNECTIONS).optimizer).toBe('AdamW');
  });

  it('recognises every block type this test relies on as trainable', () => {
    for (const type of ['token_embedding', 'gqa_attention', 'ffn_gated', 'rmsnorm', 'lm_head']) {
      expect(isTrainableBlock(type), type).toBe(true);
    }
    expect(isTrainableBlock('input')).toBe(false);
  });
});
