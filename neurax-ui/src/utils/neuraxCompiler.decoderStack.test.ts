import { describe, it, expect } from 'vitest';
import { compileToNeuraxIR } from './neuraxCompiler.ts';
import type { CanvasNode, Connection } from '@/types/architecture.ts';

/**
 * Real bug found auditing the backend today, then found again — separately
 * implemented — in the frontend's own `decoderStackParamsAndFlops`: the
 * same head_dim-truncation and gated-activation-set mistakes
 * `neurax-formulas::attention`/`neurax-opspec` just fixed on the Rust side
 * existed independently here, for the exact reason OpSpec-IR exists on the
 * backend — one formula defined in two places drifts.
 */

function node(id: string, type: CanvasNode['type'], params: Record<string, any> = {}): CanvasNode {
  return { id, type, name: id, x: 0, y: 0, params };
}

function link(from: string, to: string): Connection {
  return { id: `${from}->${to}`, from, to };
}

function layerStackGraph(numLayers: number): { nodes: CanvasNode[]; connections: Connection[] } {
  const nodes: CanvasNode[] = [
    node('in', 'input', {}),
    node('stack', 'layer_stack', { num_layers: numLayers }),
    node('out', 'output', {}),
  ];
  const connections: Connection[] = [link('in', 'stack'), link('stack', 'out')];
  return { nodes, connections };
}

function stackLayer(ir: any) {
  return ir.model.layers.find((l: any) => l.id === 'stack');
}

describe('decoderStackParamsAndFlops (layer_stack params/FLOPs)', () => {
  it('GLM-4.5-shaped head_dim (HuggingFace-verified) widens the Q/output projection past hidden_size', () => {
    // zai-org/GLM-4.5 config.json: hidden_size=5120, num_attention_heads=96,
    // num_key_value_heads=8, head_dim=128. 96*128=12288 != 5120 — a real
    // widened projection, not a rounding artifact.
    const { nodes, connections } = layerStackGraph(1);
    const withRealHeadDim = compileToNeuraxIR(nodes, connections, {
      batchSize: 1,
      seqLen: 4096,
      hiddenDim: 5120,
      numHeads: 96,
      kvHeads: 8,
      headDim: 128,
      ffnDim: 12288,
      activation: 'silu',
      numLayers: 1,
    });
    const withDerivedHeadDim = compileToNeuraxIR(nodes, connections, {
      batchSize: 1,
      seqLen: 4096,
      hiddenDim: 5120,
      numHeads: 96,
      kvHeads: 8,
      ffnDim: 12288,
      activation: 'silu',
      numLayers: 1,
    });

    const real = stackLayer(withRealHeadDim).params.param_count;
    const derived = stackLayer(withDerivedHeadDim).params.param_count;
    expect(real).toBeGreaterThan(0);
    expect(real).not.toBe(derived);
  });

  it('a plain (non-gated) SiLU MLP is no longer double-counted as SwiGLU', () => {
    // Plain SiLU is NOT gated (neurax-formulas::activation: "silu" =>
    // gated: false — only swiglu/geglu/glu/reglu are). A layer_stack whose
    // activation is stated as plain "silu" must use the 2-matrix MLP
    // formula, not the 3-matrix SwiGLU one.
    const { nodes, connections } = layerStackGraph(1);
    const ir = compileToNeuraxIR(nodes, connections, {
      batchSize: 1,
      seqLen: 128,
      hiddenDim: 512,
      numHeads: 8,
      ffnDim: 2048,
      activation: 'silu',
      numLayers: 1,
    });
    const irGated = compileToNeuraxIR(nodes, connections, {
      batchSize: 1,
      seqLen: 128,
      hiddenDim: 512,
      numHeads: 8,
      ffnDim: 2048,
      activation: 'swiglu',
      numLayers: 1,
    });

    const plainParams = stackLayer(ir).params.param_count;
    const gatedParams = stackLayer(irGated).params.param_count;
    expect(plainParams).toBeGreaterThan(0);
    // Gated adds a third (gate) matrix on top of up+down — must exceed the
    // plain, two-matrix count for the same hidden/ffn dims.
    expect(gatedParams).toBeGreaterThan(plainParams);
  });
});
