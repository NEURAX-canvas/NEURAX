/**
 * Untangle overlapping blocks automatically, instead of dragging each one
 * apart by hand.
 *
 * Built on `dagre` rather than a hand-rolled layout: placing boxes so a
 * directed graph's edges read left-to-right without crossing is a solved,
 * well-tested problem, and re-deriving it here would mean re-discovering
 * dagre's own edge cases (cycles, disconnected components, multi-edges)
 * one bug report at a time.
 *
 * Operates on exactly the entities the canvas itself treats as
 * independently positionable — the same rule `ArchitectureCanvas` already
 * uses to decide what to render as its own box:
 *
 *   - a node that belongs to no group, and
 *   - each group, as one block (its member nodes move with it as a unit;
 *     rearranging what's inside an expanded group is a separate, smaller
 *     concern this doesn't attempt)
 *
 * — and the same *resolved* edges the canvas draws: a connection whose
 * endpoint is a grouped node is redirected to that node's group, and an
 * edge with both endpoints inside the same group (never drawn — it is
 * already implied by the group being one box) is dropped. Laying out a
 * different graph than the one actually drawn would rearrange boxes to
 * satisfy edges nobody sees.
 */
import dagre from 'dagre';
import type { CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';

// Must match ArchitectureCanvas's own NODE_WIDTH/GROUP_WIDTH and the
// heights `getNodeOrGroupPos` assumes — this positions the same boxes at
// the same sizes the canvas will actually draw, not an estimate of them.
const NODE_WIDTH = 200;
const NODE_HEIGHT = 120;
const GROUP_WIDTH = 240;
const GROUP_HEIGHT = 200;

// Gaps, not sizes: how much daylight to leave between boxes in the same
// column (rank) and between columns. Roomier than dagre's own defaults —
// this runs on request, to fix crowding, so it should not recreate a
// milder version of the crowding it was asked to fix.
const RANK_SEP = 120;
const NODE_SEP = 80;

export interface LayoutResult {
  /** New top-left position for every node not inside a group. */
  nodePositions: Map<string, { x: number; y: number }>;
  /** New top-left position for every group (its members move as a unit). */
  groupPositions: Map<string, { x: number; y: number }>;
}

/**
 * Compute an uncrowded position for every top-level block.
 *
 * Pure — reads `nodes`/`connections`/`groups`, returns where things should
 * go, and touches nothing itself. The caller applies the result through
 * whatever update path it already uses (`onUpdateNode`/`onUpdateGroup`),
 * the same as a manual drag would.
 */
export function computeAutoLayout(
  nodes: CanvasNode[],
  connections: Connection[],
  groups: NodeGroup[],
): LayoutResult {
  const graph = new dagre.graphlib.Graph();
  graph.setGraph({ rankdir: 'LR', ranksep: RANK_SEP, nodesep: NODE_SEP, marginx: 40, marginy: 40 });
  graph.setDefaultEdgeLabel(() => ({}));

  const groupedNodeIds = new Set(groups.flatMap((g) => g.nodeIds));
  const topLevelNodes = nodes.filter((n) => !groupedNodeIds.has(n.id));

  for (const node of topLevelNodes) {
    graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const group of groups) {
    graph.setNode(group.id, { width: GROUP_WIDTH, height: GROUP_HEIGHT });
  }

  // Resolve each connection's endpoints the same way the canvas does before
  // drawing it, so the graph laid out is the graph actually shown.
  const groupOf = (id: string): string | undefined =>
    groups.find((g) => g.id === id)?.id ?? groups.find((g) => g.nodeIds.includes(id))?.id;

  const seenEdges = new Set<string>();
  for (const conn of connections) {
    const fromId = groupOf(conn.from) ?? conn.from;
    const toId = groupOf(conn.to) ?? conn.to;
    if (fromId === toId) continue; // Internal to one group — never drawn.
    if (!graph.hasNode(fromId) || !graph.hasNode(toId)) continue; // Dangling reference.
    const key = `${fromId}::${toId}`;
    if (seenEdges.has(key)) continue;
    seenEdges.add(key);
    graph.setEdge(fromId, toId);
  }

  dagre.layout(graph);

  const nodePositions = new Map<string, { x: number; y: number }>();
  const groupPositions = new Map<string, { x: number; y: number }>();

  for (const node of topLevelNodes) {
    const pos = graph.node(node.id);
    if (!pos) continue;
    // dagre positions by center; the canvas positions by top-left.
    nodePositions.set(node.id, { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 });
  }
  for (const group of groups) {
    const pos = graph.node(group.id);
    if (!pos) continue;
    groupPositions.set(group.id, { x: pos.x - GROUP_WIDTH / 2, y: pos.y - GROUP_HEIGHT / 2 });
  }

  return { nodePositions, groupPositions };
}
