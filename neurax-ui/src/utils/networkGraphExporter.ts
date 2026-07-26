/**
 * Network Graph Export — Interactive HTML Visualization
 *
 * Generates a self-contained HTML file with an interactive network graph
 * built with vis-network (loaded from CDN). The graph visualizes the
 * neural network architecture as nodes and connections.
 */

import { CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';
import { ArchitectureFamily } from '@/types/plugins.ts';

// ─── Color palette by family ────────────────────────────────────────

const FAMILY_COLORS: Record<string, { node: string; border: string; highlight: string }> = {
  transformer: { node: '#3b82f6', border: '#1d4ed8', highlight: '#60a5fa' },
  moe:         { node: '#8b5cf6', border: '#6d28d9', highlight: '#a78bfa' },
  cnn:         { node: '#10b981', border: '#047857', highlight: '#34d399' },
  ssm:         { node: '#f59e0b', border: '#b45309', highlight: '#fbbf24' },
  diffusion:   { node: '#ec4899', border: '#be185d', highlight: '#f472b6' },
  gnn:         { node: '#14b8a6', border: '#0d9488', highlight: '#2dd4bf' },
  gan:         { node: '#f97316', border: '#c2410c', highlight: '#fb923c' },
  rl:          { node: '#6366f1', border: '#4338ca', highlight: '#818cf8' },
  snn:         { node: '#84cc16', border: '#4d7c0f', highlight: '#a3e635' },
  rnn:         { node: '#06b6d4', border: '#0891b2', highlight: '#22d3ee' },
  experimental:{ node: '#a855f7', border: '#7e22ce', highlight: '#c084fc' },
};

const DEFAULT_COLOR = { node: '#6b7280', border: '#4b5563', highlight: '#9ca3af' };

function getFamilyColor(family: ArchitectureFamily) {
  return FAMILY_COLORS[family] ?? DEFAULT_COLOR;
}

// ─── Layer type icons (Unicode) ─────────────────────────────────────

const LAYER_ICONS: Record<string, string> = {
  input: '⬇',
  output: '⬆',
  dense: '●',
  linear: '●',
  conv2d: '▣',
  conv1d: '▣',
  conv3d: '▣',
  relu: '📈',
  gelu: '📈',
  attention: '🔗',
  cross_attention: '🔗',
  self_attention: '🔗',
  mha_attention: '🔗',
  gqa_attention: '🔗',
  residual: '➕',
  residual_add: '➕',
  transformer: '📐',
  embedding: '📋',
  token_embedding: '📋',
  layernorm: '⚖',
  rmsnorm: '⚖',
  batchnorm: '⚖',
  dropout: '⊙',
  flatten: '▤',
  concat: '⊞',
  lstm: '📎',
  gru: '📎',
  max_pool: '▼',
  avg_pool: '▼',
  global_pool: '▼',
  moe_block: '🧩',
  moe_layer: '🧩',
  mamba_block: '⚡',
  s4_block: '⚡',
};

function getLayerIcon(type: string): string {
  const t = type.toLowerCase();
  for (const [key, icon] of Object.entries(LAYER_ICONS)) {
    if (t.includes(key)) return icon;
  }
  return '◇';
}

// ─── HTML template generator ────────────────────────────────────────

export function generateNetworkGraphHTML(
  nodes: CanvasNode[],
  connections: Connection[],
  groups: NodeGroup[],
  options: {
    modelName?: string;
    family?: ArchitectureFamily;
  } = {}
): string {
  const { modelName = 'NeuraxModel', family = 'transformer' } = options;
  const colors = getFamilyColor(family);

  // Build node list with group membership
  const nodeGroupMap = new Map<string, string>();
  for (const g of groups) {
    for (const nid of g.nodeIds) {
      nodeGroupMap.set(nid, g.name);
    }
  }

  // Prepare nodes JSON for vis-network
  const visNodes = nodes.map((n) => {
    const icon = getLayerIcon(n.type);
    const inGroup = nodeGroupMap.get(n.id);
    return {
      id: n.id,
      label: `${icon} ${n.name || n.type}`,
      title: `<div style="font-size:12px;line-height:1.6">
        <b>${n.name || n.id}</b><br/>
        Type: ${n.type}<br/>
        ${inGroup ? `Group: ${inGroup}<br/>` : ''}
        ${n.params ? Object.entries(n.params).slice(0, 4).map(([k, v]) => `${k}: ${v}`).join('<br/>') : ''}
      </div>`,
      shape: 'box',
      color: {
        background: colors.node,
        border: colors.border,
        highlight: { background: colors.highlight, border: colors.border },
      },
      font: { color: '#fff', size: 12, face: 'monospace' },
      size: Math.max(20, 40 - nodes.length * 0.5),
      margin: { top: 8, bottom: 8, left: 12, right: 12 },
      borderWidth: 2,
      shadow: { enabled: true, size: 4 },
      ...(inGroup ? { group: inGroup } : {}),
    };
  });

  // Prepare edges JSON
  const visEdges = connections.map(c => ({
    from: c.from,
    to: c.to,
    id: c.id,
    arrows: { to: { enabled: true, scaleFactor: 0.8 } },
    color: { color: '#64748b', highlight: colors.highlight, hover: colors.highlight },
    smooth: { type: 'curvedCW', roundness: 0.15 },
    width: 1.5,
  }));

  // Group sub-networks (visual clusters)
  const groupDefinitions = groups.map(g => ({
    id: g.name,
    label: g.name,
    color: {
      background: 'rgba(59,130,246,0.05)',
      border: 'rgba(59,130,246,0.2)',
    },
    borderWidth: 1,
    font: { size: 14, face: 'monospace', color: '#94a3b8' },
    shadow: { enabled: true, size: 2 },
  }));

  // Stats
  const layerTypes = [...new Set(nodes.map(n => n.type))];

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>${modelName} — Architecture Graph</title>
<script src="https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js"></script>
<link rel="stylesheet" href="https://unpkg.com/vis-network@9.1.9/dist/dist/vis-network.min.css" />
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; overflow: hidden; height: 100vh; }
#header {
  position: fixed; top: 0; left: 0; right: 0; z-index: 100;
  background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
  border-bottom: 1px solid #334155;
  padding: 12px 24px;
  display: flex; align-items: center; justify-content: space-between;
}
#header h1 { font-size: 16px; font-weight: 600; color: #f1f5f9; }
#header .subtitle { font-size: 11px; color: #64748b; margin-top: 2px; }
#header .stats { display: flex; gap: 16px; font-size: 11px; }
#header .stats span { color: #94a3b8; }
#header .stats strong { color: #e2e8f0; }
#header .badge {
  background: ${colors.node}22; color: ${colors.highlight};
  border: 1px solid ${colors.border}44;
  padding: 2px 10px; border-radius: 4px; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
}
#mynetwork {
  position: fixed; top: 60px; left: 0; right: 0; bottom: 0;
}
#legend {
  position: fixed; bottom: 20px; left: 20px; z-index: 100;
  background: #1e293bdd; border: 1px solid #334155;
  border-radius: 8px; padding: 12px 16px; font-size: 10px;
  max-height: 300px; overflow-y: auto;
  backdrop-filter: blur(8px);
}
#legend h3 { font-size: 11px; margin-bottom: 6px; color: #94a3b8; }
#legend .item { display: flex; align-items: center; gap: 6px; margin: 3px 0; }
#legend .color-dot { width: 8px; height: 8px; border-radius: 2px; flex-shrink: 0; }
#tooltip {
  position: fixed; z-index: 200;
  background: #1e293be8; border: 1px solid #334155;
  border-radius: 8px; padding: 8px 12px; font-size: 11px; line-height: 1.5;
  pointer-events: none; backdrop-filter: blur(8px);
  max-width: 280px;
  display: none;
}
#controls {
  position: fixed; bottom: 20px; right: 20px; z-index: 100;
  display: flex; gap: 6px;
}
#controls button {
  background: #1e293bdd; border: 1px solid #334155;
  color: #e2e8f0; padding: 6px 12px; border-radius: 6px;
  font-size: 11px; cursor: pointer; backdrop-filter: blur(8px);
  transition: all 0.15s;
}
#controls button:hover { background: #334155; border-color: ${colors.border}; }
</style>
</head>
<body>
<div id="header">
  <div>
    <h1>${modelName}</h1>
    <div class="subtitle">NEURAX — Neural Architecture Graph</div>
  </div>
  <div class="stats">
    <span>Blocks <strong>${nodes.length}</strong></span>
    <span>Connections <strong>${connections.length}</strong></span>
    <span>Groups <strong>${groups.length}</strong></span>
    <span class="badge">${family}</span>
  </div>
</div>
<div id="mynetwork"></div>

<div id="legend">
  <h3>Layer Types</h3>
  ${layerTypes.slice(0, 15).map(t => `
    <div class="item">
      <span>${getLayerIcon(t)}</span>
      <span>${t}</span>
    </div>
  `).join('')}
  ${layerTypes.length > 15 ? `<div class="item" style="color:#64748b;">… +${layerTypes.length - 15} more</div>` : ''}
</div>

<div id="controls">
  <button onclick="fitView()">Fit</button>
  <button onclick="togglePhysics()">Toggle Physics</button>
  <button onclick="exportPNG()">Export PNG</button>
</div>

<div id="tooltip"></div>

<script>
const nodes = new vis.DataSet(${JSON.stringify(visNodes)});
const edges = new vis.DataSet(${JSON.stringify(visEdges)});
const groups = ${JSON.stringify(groupDefinitions)};

const container = document.getElementById('mynetwork');
const data = { nodes, edges };
let physicsEnabled = true;

const options = {
  autoResize: true,
  groups,
  nodes: {
    shape: 'box',
    font: { color: '#fff', size: 12, face: 'monospace' },
    borderWidth: 2,
    shadow: { enabled: true, size: 4 },
  },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.8 } },
    color: { color: '#64748b', highlight: '${colors.highlight}' },
    smooth: { type: 'curvedCW', roundness: 0.15 },
    width: 1.5,
  },
  physics: {
    enabled: true,
    solver: 'forceAtlas2Based',
    forceAtlas2Based: {
      gravitationalConstant: -32,
      centralGravity: 0.005,
      springLength: 160,
      springConstant: 0.04,
      damping: 0.5,
    },
    stabilization: { iterations: 200 },
  },
  interaction: {
    hover: true,
    tooltipDelay: 200,
    navigationButtons: true,
    keyboard: true,
  },
  layout: {
    improvedLayout: true,
  },
};

const network = new vis.Network(container, data, options);

network.on('showPopup', function(params) {
  document.getElementById('tooltip').style.display = 'block';
});
network.on('hidePopup', function() {
  document.getElementById('tooltip').style.display = 'none';
});

// Fit view after stabilization
network.once('stabilizationIterationsDone', function() {
  network.fit({ animation: true });
});

window.fitView = function() {
  network.fit({ animation: { duration: 300, easingFunction: 'easeInOutQuad' } });
};

window.togglePhysics = function() {
  physicsEnabled = !physicsEnabled;
  network.setOptions({ physics: { enabled: physicsEnabled } });
  if (physicsEnabled) network.fit({ animation: true });
};

window.exportPNG = function() {
  const canvas = container.querySelector('canvas');
  if (!canvas) return;
  const link = document.createElement('a');
  link.download = '${modelName.toLowerCase().replace(/[^a-z0-9]/g, '_')}_graph.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
};

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.key === 'f' || e.key === 'F') fitView();
  if (e.key === 'p' || e.key === 'P') togglePhysics();
});
</script>
</body>
</html>`;

  return html;
}
