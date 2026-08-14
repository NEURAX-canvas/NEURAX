// A dragged node almost never settles at a whole-pixel position: its x/y
// come from `(mouseClient - canvasOffset) / zoom`, and canvasOffset itself
// accumulates from raw pointer deltas during panning. Left as float CSS
// `left`/`top`, the browser rasterises the card's text and borders off the
// pixel grid — the canvas reads as permanently soft, most visible on the
// desktop build's WebKitGTK webview, which anti-aliases more aggressively
// than Chromium. This guards the fix: node and group cards are always
// painted at a whole device pixel, however fractional their stored
// coordinates are, while the coordinates themselves stay float (drag math,
// connection anchors and alignment guides all still need the precision).
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render } from '@testing-library/react';
import { CanvasNode } from './CanvasNode';
import { GroupNode } from './GroupNode';
import { snapToDevicePixel } from './ArchitectureCanvas';
import type { CanvasNode as CanvasNodeType, NodeGroup } from '@/types/architecture.ts';

const FRACTIONAL_NODE: CanvasNodeType = {
  id: 'n1',
  type: 'dense',
  name: 'Dense',
  x: 142.3729,
  y: 87.618,
  params: {},
};

const FRACTIONAL_GROUP: NodeGroup = {
  id: 'g1',
  name: 'Block',
  nodeIds: ['n1'],
  connectionIds: [],
  repeatCount: 1,
  collapsed: false,
  x: 55.912,
  y: 12.077,
};

describe('canvas cards are painted on a whole pixel', () => {
  it('CanvasNode rounds a fractional position to whole CSS pixels', () => {
    // Captured before rendering: the expected style comes from this
    // snapshot, not from re-reading the fixture after render, so a future
    // bug that mutated `node` in place couldn't drag the expectation along
    // with it and pass anyway.
    const originalX = FRACTIONAL_NODE.x;
    const originalY = FRACTIONAL_NODE.y;
    // Sanity: the fixture actually is fractional, so this test would fail
    // without the rounding — it isn't trivially true.
    expect(Number.isInteger(originalX)).toBe(false);
    expect(Number.isInteger(originalY)).toBe(false);

    const { container } = render(
      <CanvasNode
        node={FRACTIONAL_NODE}
        isSelected={false}
        onSelect={() => {}}
        onDragStart={() => {}}
      />,
    );
    const card = container.firstElementChild as HTMLElement;
    expect(card.style.left).toBe(`${Math.round(originalX)}px`);
    expect(card.style.top).toBe(`${Math.round(originalY)}px`);
    expect(FRACTIONAL_NODE.x).toBe(originalX);
    expect(FRACTIONAL_NODE.y).toBe(originalY);
  });

  it('GroupNode rounds a fractional position to whole CSS pixels', () => {
    const originalX = FRACTIONAL_GROUP.x;
    const originalY = FRACTIONAL_GROUP.y;
    expect(Number.isInteger(originalX)).toBe(false);
    expect(Number.isInteger(originalY)).toBe(false);

    const { container } = render(
      <GroupNode
        group={FRACTIONAL_GROUP}
        nodes={[FRACTIONAL_NODE]}
        isSelected={false}
        onSelect={() => {}}
        onDragStart={() => {}}
        onUpdateGroup={() => {}}
        onUngroupGroup={() => {}}
        onDeleteGroup={() => {}}
      />,
    );
    const card = container.firstElementChild as HTMLElement;
    expect(card.style.left).toBe(`${Math.round(originalX)}px`);
    expect(card.style.top).toBe(`${Math.round(originalY)}px`);
    expect(FRACTIONAL_GROUP.x).toBe(originalX);
    expect(FRACTIONAL_GROUP.y).toBe(originalY);
  });
});

describe('snapToDevicePixel', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('rounds to a whole CSS pixel at devicePixelRatio 1', () => {
    vi.stubGlobal('window', { ...window, devicePixelRatio: 1 });
    expect(snapToDevicePixel(142.3729)).toBe(142);
    expect(snapToDevicePixel(87.618)).toBe(88);
  });

  it('rounds to the nearest physical device pixel on a HiDPI display', () => {
    // At 2x, a CSS pixel is two device pixels — a value can legitimately
    // land on a CSS half-pixel and still be crisp, because the physical
    // pixel grid is twice as fine.
    vi.stubGlobal('window', { ...window, devicePixelRatio: 2 });
    expect(snapToDevicePixel(142.3729)).toBe(142.5);
  });

  it('falls back to whole CSS pixels when devicePixelRatio is unavailable', () => {
    vi.stubGlobal('window', { ...window, devicePixelRatio: 0 });
    expect(snapToDevicePixel(10.6)).toBe(11);
  });
});

describe('composed screen position (offset + zoom·node.x)', () => {
  // Pins the documented scope of the fix in ArchitectureCanvas: snapping
  // `offset` and each node's world position independently guarantees a
  // crisp screen pixel at a whole-number zoom (concretely 100%, the default
  // and what "Reset View" restores) — not at every zoom level. This isn't a
  // gap introduced by the fix: an unsnapped canvas was exactly as fractional
  // at 125% zoom as a snapped one is. See the comment on snapToDevicePixel.
  //
  // `snapToDevicePixel` reads `window.devicePixelRatio`, so it's pinned to 1
  // here rather than left to whatever the test environment happens to
  // default to — these assertions are about the zoom arithmetic, not about
  // jsdom's DPR.
  beforeEach(() => {
    vi.stubGlobal('window', { ...window, devicePixelRatio: 1 });
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('lands on a whole screen pixel at 100% zoom', () => {
    const zoom = 1;
    const snappedOffset = snapToDevicePixel(37.812);
    const snappedNodeX = Math.round(214.5501);
    expect(Number.isInteger(snappedOffset + zoom * snappedNodeX)).toBe(true);
  });

  it('is not guaranteed a whole screen pixel at a fractional zoom', () => {
    const zoom = 1.25;
    const snappedOffset = snapToDevicePixel(37.812);
    const snappedNodeX = Math.round(214.5501);
    const screenX = snappedOffset + zoom * snappedNodeX;
    expect(Number.isInteger(screenX)).toBe(false);
  });
});
