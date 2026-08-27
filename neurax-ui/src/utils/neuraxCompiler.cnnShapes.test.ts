import { describe, it, expect } from 'vitest';
import { compileToNeuraxIR } from './neuraxCompiler.ts';
import type { CanvasNode, Connection } from '@/types/architecture.ts';

/**
 * Real shape propagation for CNN. Every non-Input/Embedding/DenseProjection
 * block used to fall through to a `default` case that hardcoded
 * `[batch, seq, hidden]` regardless of family — harmless for a transformer,
 * but for a CNN it meant every layer's shape was a sequence-model
 * placeholder with 0 in every dimension nothing had set. A live build hit
 * exactly this: `conv1` had a correct `in_channels: 3` in its own params,
 * and an `input_shape` of `[64, 0, 0]` anyway, because nothing propagated
 * an actual image shape through the graph.
 */
function node(id: string, type: CanvasNode['type'], params: Record<string, any> = {}): CanvasNode {
  return { id, type, name: id, x: 0, y: 0, params };
}

function link(from: string, to: string): Connection {
  return { id: `${from}->${to}`, from, to };
}

describe('CNN shape propagation', () => {
  it('carries a real image shape through conv/pool/flatten/dense, not [batch, seq, hidden]', () => {
    const nodes: CanvasNode[] = [
      node('in', 'input', {}),
      node('conv1', 'conv2d', { in_channels: 3, out_channels: 16, kernel_size: 3, stride: 1, padding: 1 }),
      node('bn1', 'batchnorm', { num_features: 16 }),
      node('pool1', 'max_pool', { pool_size: 2, stride: 2 }),
      node('conv2', 'conv2d', { in_channels: 16, out_channels: 32, kernel_size: 3, stride: 1, padding: 1 }),
      node('gpool', 'global_pool', {}),
      node('flat', 'flatten', {}),
      node('fc', 'linear_projection', { out_features: 128 }),
      node('head', 'classification_head', { num_classes: 10 }),
      node('out', 'output', {}),
    ];
    const connections: Connection[] = [
      link('in', 'conv1'), link('conv1', 'bn1'), link('bn1', 'pool1'),
      link('pool1', 'conv2'), link('conv2', 'gpool'), link('gpool', 'flat'),
      link('flat', 'fc'), link('fc', 'head'), link('head', 'out'),
    ];

    const ir = compileToNeuraxIR(nodes, connections, {
      family: 'cnn',
      batchSize: 4,
      inChannels: 3,
      imgHeight: 32,
      imgWidth: 32,
    });

    const byId = new Map(ir.model.layers.map((l: any) => [l.id, l]));

    // conv1: 3->16 channels, 3x3/stride1/pad1 keeps spatial size at 32.
    expect(byId.get('conv1').input_shape).toEqual([4, 3, 32, 32]);
    expect(byId.get('conv1').output_shape).toEqual([4, 16, 32, 32]);

    // bn1: channel-preserving passthrough.
    expect(byId.get('bn1').output_shape).toEqual([4, 16, 32, 32]);

    // pool1: 2x2/stride2 halves spatial size, channels unchanged.
    expect(byId.get('pool1').output_shape).toEqual([4, 16, 16, 16]);

    // conv2 picks up pool1's actual output — 16 in, 32 out, still 16x16.
    expect(byId.get('conv2').input_shape).toEqual([4, 16, 16, 16]);
    expect(byId.get('conv2').output_shape).toEqual([4, 32, 16, 16]);

    // global_pool collapses spatial dims to 1x1, channels unchanged.
    expect(byId.get('gpool').output_shape).toEqual([4, 32, 1, 1]);

    // flatten: 32*1*1 = 32 features.
    expect(byId.get('flat').output_shape).toEqual([4, 32]);

    // linear_projection: real out_features, not hidden_size or seq_len.
    expect(byId.get('fc').output_shape).toEqual([4, 128]);

    // classification_head: real num_classes.
    expect(byId.get('head').output_shape).toEqual([4, 10]);

    // Nothing anywhere in the chain is 0 — the original bug's exact symptom.
    for (const layer of ir.model.layers as any[]) {
      expect(layer.output_shape.every((d: number) => d > 0), `${layer.id} has a 0 dimension: ${layer.output_shape}`).toBe(true);
    }
  });

  it('sums channels across branches at a Concat merge, keeping spatial size', () => {
    const nodes: CanvasNode[] = [
      node('in', 'input', {}),
      node('convA', 'conv2d', { in_channels: 3, out_channels: 8, kernel_size: 3, stride: 1, padding: 1 }),
      node('convB', 'conv2d', { in_channels: 3, out_channels: 24, kernel_size: 3, stride: 1, padding: 1 }),
      node('merge', 'concat', {}),
      node('out', 'output', {}),
    ];
    const connections: Connection[] = [
      link('in', 'convA'), link('in', 'convB'),
      link('convA', 'merge'), link('convB', 'merge'),
      link('merge', 'out'),
    ];

    const ir = compileToNeuraxIR(nodes, connections, {
      family: 'cnn', batchSize: 2, inChannels: 3, imgHeight: 16, imgWidth: 16,
    });
    const byId = new Map(ir.model.layers.map((l: any) => [l.id, l]));
    expect(byId.get('merge').output_shape).toEqual([2, 32, 16, 16]);
  });
});
