/**
 * The point of auto-layout is that nothing overlaps afterwards — so the
 * checks here are about the actual geometry the result implies, not about
 * dagre having run without throwing.
 */
import { describe, it, expect } from 'vitest';
import { computeAutoLayout } from './autoLayout';
import type { CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';

const NODE_WIDTH = 200;
const NODE_HEIGHT = 120;

function overlaps(
  a: { x: number; y: number },
  b: { x: number; y: number },
  w = NODE_WIDTH,
  h = NODE_HEIGHT,
): boolean {
  return a.x < b.x + w && b.x < a.x + w && a.y < b.y + h && b.y < a.y + h;
}

describe('computeAutoLayout', () => {
  it('separates blocks that were dropped on top of each other', () => {
    const nodes: CanvasNode[] = [
      { id: 'n1', type: 'input', name: 'Input', x: 50, y: 50, params: {} },
      { id: 'n2', type: 'token_embedding', name: 'Embedding', x: 55, y: 52, params: {} },
      { id: 'n3', type: 'gqa_attention', name: 'GQA', x: 60, y: 48, params: {} },
      { id: 'n4', type: 'output', name: 'Output', x: 58, y: 55, params: {} },
    ];
    const connections: Connection[] = [
      { id: 'c1', from: 'n1', to: 'n2' },
      { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' },
    ];

    // All four start on top of each other.
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        expect(overlaps(nodes[i], nodes[j])).toBe(true);
      }
    }

    const { nodePositions } = computeAutoLayout(nodes, connections, []);
    const positions = nodes.map((n) => nodePositions.get(n.id)!);
    for (let i = 0; i < positions.length; i++) {
      for (let j = i + 1; j < positions.length; j++) {
        expect(overlaps(positions[i], positions[j]), `n${i + 1} vs n${j + 1}`).toBe(false);
      }
    }
  });

  it('orders connected blocks left to right, following the edges', () => {
    const nodes: CanvasNode[] = [
      { id: 'a', type: 'input', name: 'A', x: 0, y: 0, params: {} },
      { id: 'b', type: 'token_embedding', name: 'B', x: 0, y: 0, params: {} },
      { id: 'c', type: 'output', name: 'C', x: 0, y: 0, params: {} },
    ];
    const connections: Connection[] = [
      { id: 'c1', from: 'a', to: 'b' },
      { id: 'c2', from: 'b', to: 'c' },
    ];
    const { nodePositions } = computeAutoLayout(nodes, connections, []);
    const xA = nodePositions.get('a')!.x;
    const xB = nodePositions.get('b')!.x;
    const xC = nodePositions.get('c')!.x;
    expect(xA).toBeLessThan(xB);
    expect(xB).toBeLessThan(xC);
  });

  it('treats a collapsed group as one block and moves its members with it', () => {
    const nodes: CanvasNode[] = [
      { id: 'n1', type: 'input', name: 'Input', x: 0, y: 0, params: {} },
      { id: 'g-a', type: 'gqa_attention', name: 'GQA', x: 10, y: 10, params: {} },
      { id: 'g-b', type: 'ffn_standard', name: 'FFN', x: 10, y: 10, params: {} },
      { id: 'n4', type: 'output', name: 'Output', x: 0, y: 0, params: {} },
    ];
    const groups: NodeGroup[] = [
      {
        id: 'grp1', name: 'Block', nodeIds: ['g-a', 'g-b'], connectionIds: ['inner'],
        repeatCount: 1, x: 400, y: 400, collapsed: true,
      },
    ];
    const connections: Connection[] = [
      { id: 'c1', from: 'n1', to: 'g-a' },
      { id: 'inner', from: 'g-a', to: 'g-b' },
      { id: 'c2', from: 'g-b', to: 'n4' },
    ];

    const { nodePositions, groupPositions } = computeAutoLayout(nodes, connections, groups);

    // Only n1 and n4 are independently positioned — g-a/g-b move with the group.
    expect(nodePositions.has('n1')).toBe(true);
    expect(nodePositions.has('n4')).toBe(true);
    expect(nodePositions.has('g-a')).toBe(false);
    expect(nodePositions.has('g-b')).toBe(false);

    expect(groupPositions.has('grp1')).toBe(true);
    const xN1 = nodePositions.get('n1')!.x;
    const xGrp = groupPositions.get('grp1')!.x;
    const xN4 = nodePositions.get('n4')!.x;
    expect(xN1).toBeLessThan(xGrp);
    expect(xGrp).toBeLessThan(xN4);
  });

  it('does not error on a dangling connection to a node that no longer exists', () => {
    const nodes: CanvasNode[] = [
      { id: 'n1', type: 'input', name: 'Input', x: 0, y: 0, params: {} },
    ];
    const connections: Connection[] = [{ id: 'c1', from: 'n1', to: 'ghost' }];
    expect(() => computeAutoLayout(nodes, connections, [])).not.toThrow();
    const { nodePositions } = computeAutoLayout(nodes, connections, []);
    expect(nodePositions.has('n1')).toBe(true);
  });

  it('lays out disconnected blocks without overlapping, even with no edges at all', () => {
    const nodes: CanvasNode[] = [
      { id: 'n1', type: 'input', name: 'A', x: 0, y: 0, params: {} },
      { id: 'n2', type: 'input', name: 'B', x: 5, y: 5, params: {} },
      { id: 'n3', type: 'input', name: 'C', x: 10, y: 10, params: {} },
    ];
    const { nodePositions } = computeAutoLayout(nodes, [], []);
    const positions = nodes.map((n) => nodePositions.get(n.id)!);
    for (let i = 0; i < positions.length; i++) {
      for (let j = i + 1; j < positions.length; j++) {
        expect(overlaps(positions[i], positions[j])).toBe(false);
      }
    }
  });
});
