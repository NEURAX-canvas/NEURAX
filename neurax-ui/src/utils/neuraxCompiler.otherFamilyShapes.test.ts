import { describe, it, expect } from 'vitest';
import { compileToNeuraxIR } from './neuraxCompiler.ts';
import type { CanvasNode, Connection } from '@/types/architecture.ts';

/**
 * Shape propagation for the families beyond CNN that shared the same root
 * cause: gan/diffusion use the identical [B, C, H, W] image convention CNN
 * does; gnn uses its own [batch, num_nodes, node_features] convention; rnn
 * had a distinct bug (its hwConfig preset sets `hiddenSize`, but the
 * compiler's shape logic only ever read `hiddenDim`, so `hidden` computed
 * to 0 for every RNN build regardless of shape propagation).
 */
function node(id: string, type: CanvasNode['type'], params: Record<string, any> = {}): CanvasNode {
  return { id, type, name: id, x: 0, y: 0, params };
}

function link(from: string, to: string): Connection {
  return { id: `${from}->${to}`, from, to };
}

describe('GNN shape propagation', () => {
  it('changes feature dim at conv layers, keeps node count until an explicit pool', () => {
    const nodes: CanvasNode[] = [
      node('in', 'input', {}),
      node('gc1', 'gcn_conv', { out_channels: 32 }),
      node('gat', 'gat_conv', { out_channels: 16, num_heads: 4, concat: true }),
      node('pool', 'topk_pooling', { ratio: 0.5 }),
      node('readout', 'global_mean_pool', {}),
      node('head', 'classification_head', { num_classes: 7 }),
      node('out', 'output', {}),
    ];
    const connections: Connection[] = [
      link('in', 'gc1'), link('gc1', 'gat'), link('gat', 'pool'),
      link('pool', 'readout'), link('readout', 'head'), link('head', 'out'),
    ];

    const ir = compileToNeuraxIR(nodes, connections, {
      family: 'gnn', batchSize: 1, numNodes: 1000, nodeFeatDim: 64,
    });
    const byId = new Map(ir.model.layers.map((l: any) => [l.id, l]));

    // gc1: feature dim becomes out_channels (32), node count unchanged.
    expect(byId.get('gc1').input_shape).toEqual([1, 1000, 64]);
    expect(byId.get('gc1').output_shape).toEqual([1, 1000, 32]);

    // gat: 4 heads x 16 channels, concatenated -> 64 features. Node count still unchanged.
    expect(byId.get('gat').output_shape).toEqual([1, 1000, 64]);

    // topk_pooling halves the node count, features unchanged.
    expect(byId.get('pool').output_shape).toEqual([1, 500, 64]);

    // global_mean_pool: graph-level readout collapses nodes into one vector.
    expect(byId.get('readout').output_shape).toEqual([1, 64]);

    expect(byId.get('head').output_shape).toEqual([1, 7]);
  });
});

describe('GAN shape propagation (same [B,C,H,W] engine as CNN)', () => {
  it('a discriminator-style conv chain shrinks spatial size, never hits 0', () => {
    const nodes: CanvasNode[] = [
      node('in', 'input', {}),
      node('c1', 'conv2d', { in_channels: 3, out_channels: 64, kernel_size: 4, stride: 2, padding: 1 }),
      node('c2', 'conv2d', { in_channels: 64, out_channels: 128, kernel_size: 4, stride: 2, padding: 1 }),
      node('out', 'output', {}),
    ];
    const connections: Connection[] = [link('in', 'c1'), link('c1', 'c2'), link('c2', 'out')];

    const ir = compileToNeuraxIR(nodes, connections, {
      family: 'gan', batchSize: 8, inChannels: 3, imgHeight: 64, imgWidth: 64,
    });
    const byId = new Map(ir.model.layers.map((l: any) => [l.id, l]));

    expect(byId.get('c1').input_shape).toEqual([8, 3, 64, 64]);
    expect(byId.get('c1').output_shape).toEqual([8, 64, 32, 32]);
    expect(byId.get('c2').output_shape).toEqual([8, 128, 16, 16]);
  });
});

describe('Diffusion shape propagation (UNet: downsample, upsample, skip concat)', () => {
  it('a down/up pair with a skip connection lands back at the original resolution', () => {
    const nodes: CanvasNode[] = [
      node('in', 'input', {}),
      // A stem block right after input — the skip branches off a real,
      // tracked block, the common UNet pattern (skipping the canvas's
      // literal `input` node itself, which never appears in the compiled
      // graph at all, is a rarer edge case not covered here).
      node('stem', 'conv2d', { in_channels: 4, out_channels: 4, kernel_size: 3, stride: 1, padding: 1 }),
      node('down', 'conv2d', { in_channels: 4, out_channels: 32, kernel_size: 3, stride: 2, padding: 1 }),
      node('up', 'transposed_conv', { in_channels: 32, out_channels: 4, kernel_size: 4, stride: 2, padding: 1 }),
      node('skip', 'concat', {}),
      node('out', 'output', {}),
    ];
    const connections: Connection[] = [
      link('in', 'stem'), link('stem', 'down'), link('down', 'up'),
      link('up', 'skip'), link('stem', 'skip'),
      link('skip', 'out'),
    ];

    const ir = compileToNeuraxIR(nodes, connections, {
      family: 'diffusion', batchSize: 2, inChannels: 4, imgHeight: 64, imgWidth: 64,
    });
    const byId = new Map(ir.model.layers.map((l: any) => [l.id, l]));

    // Downsample halves resolution, upsample (stride-2 transpose conv) brings it back.
    expect(byId.get('down').output_shape).toEqual([2, 32, 32, 32]);
    expect(byId.get('up').output_shape).toEqual([2, 4, 64, 64]);
    // Skip connects the stem branch (4ch @ 64x64) with the upsampled
    // branch (also 4ch @ 64x64, matching spatial size) — channels add up.
    expect(byId.get('skip').output_shape).toEqual([2, 8, 64, 64]);
  });
});

describe('RNN hidden size', () => {
  it('an RNN build\'s configured hiddenSize reaches shape inference instead of 0', () => {
    const nodes: CanvasNode[] = [
      node('in', 'input', {}),
      node('emb', 'embedding', {}),
      node('lstm', 'lstm', {}),
      node('out', 'output', {}),
    ];
    const connections: Connection[] = [link('in', 'emb'), link('emb', 'lstm'), link('lstm', 'out')];

    const ir = compileToNeuraxIR(nodes, connections, {
      family: 'rnn', batchSize: 4, seqLen: 50, hiddenSize: 256,
    });
    const byId = new Map(ir.model.layers.map((l: any) => [l.id, l]));

    // Falls through to the default [batch, seq, hidden] case — but `hidden`
    // must now be the real configured 256, not 0.
    expect(byId.get('lstm').output_shape).toEqual([4, 50, 256]);
  });

  it('a node\'s own configured hidden_size reaches the backend as rnn_hidden_size, not the generic hidden_size fallback', () => {
    // Exact shape of a real template node (modelTemplates.ts's "BiGRU Layer
    // 2 (256)"): the node's own hidden_size means its recurrent state size
    // to whoever wrote the template, not an input dimension. Before this
    // fix, rnn_hidden_size was never set on the compiled layer at all, so
    // the backend fell back to a hardcoded 512 regardless of this value —
    // any bilstm/bigru/lstm_cell/gru_cell node's configured size was
    // silently discarded.
    const nodes: CanvasNode[] = [
      node('in', 'input', {}),
      node('emb', 'embedding', { embedding_dim: 300 }),
      node('rnn', 'bigru', { hidden_size: 256, num_layers: 1, bidirectional: true }),
      node('out', 'output', {}),
    ];
    const connections: Connection[] = [link('in', 'emb'), link('emb', 'rnn'), link('rnn', 'out')];

    const ir = compileToNeuraxIR(nodes, connections, { family: 'rnn', batchSize: 4, seqLen: 50 });
    const byId = new Map(ir.model.layers.map((l: any) => [l.id, l]));
    const rnnLayer = byId.get('rnn');

    expect(rnnLayer.layer_type).toBe('gru_block');
    expect(rnnLayer.params.rnn_hidden_size).toBe(256);
    expect(rnnLayer.params.bidirectional_rnn).toBe(true);
  });
});
