// The documentation (F1 → Keyboard shortcuts) promises "while the cursor is
// in a text field these shortcuts stand down". That was only true for the
// app-level shortcuts (Ctrl+S/O/Z/Y, handled in Index.tsx) and this file's
// own Space-bar pan handler — ArchitectureCanvas's own Delete/Ctrl+A/Ctrl+D/
// Ctrl+G/Escape handler had no such guard. A node selected so its Inspector
// panel input is focused, then Backspace to fix a typo in its name, deleted
// the whole node instead of a character.
import { describe, it, expect, vi, beforeAll } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { ArchitectureCanvas } from './ArchitectureCanvas';
import type { CanvasNode, Connection } from '@/types/architecture.ts';

// jsdom has no ResizeObserver; the canvas's zoom slider (Radix UI) needs one
// just to mount, unrelated to anything this file tests.
beforeAll(() => {
  (globalThis as any).ResizeObserver ??= class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
});

const NODES: CanvasNode[] = [
  { id: 'n1', type: 'dense', name: 'Dense', x: 100, y: 100, params: {} },
];
const CONNECTIONS: Connection[] = [];

function renderCanvas(onDeleteNode: (id: string) => void) {
  return render(
    <div>
      <input aria-label="node-name" defaultValue="Dense" />
      <ArchitectureCanvas
        nodes={NODES}
        connections={CONNECTIONS}
        groups={[]}
        selectedNodeId="n1"
        onSelectNode={() => {}}
        onUpdateNode={() => {}}
        onAddNode={() => {}}
        onDeleteNode={onDeleteNode}
        onDuplicateNode={() => {}}
        onAddConnection={() => {}}
      />
    </div>,
  );
}

describe('canvas shortcuts stand down while typing', () => {
  it('Backspace in a text field does not delete the selected node', () => {
    const onDeleteNode = vi.fn();
    const { getByLabelText } = renderCanvas(onDeleteNode);

    const input = getByLabelText('node-name');
    input.focus();
    fireEvent.keyDown(input, { key: 'Backspace', bubbles: true });

    expect(onDeleteNode).not.toHaveBeenCalled();
  });

  it('Backspace outside a text field still deletes the selected node', () => {
    // The fix must not silently disable the shortcut everywhere — only while
    // typing. Dispatched on `window` itself, matching where a keystroke
    // lands when focus is on the canvas (a plain div, not a form field).
    const onDeleteNode = vi.fn();
    renderCanvas(onDeleteNode);

    fireEvent.keyDown(window, { key: 'Backspace', bubbles: true });

    expect(onDeleteNode).toHaveBeenCalledWith('n1');
  });

  it('Ctrl+A in a text field is left alone, so the browser can select the field\'s own text', () => {
    // Regression for the same missing guard on the other shortcuts in this
    // handler. jsdom doesn't simulate the browser's own "select all text"
    // behaviour, so the reliable signal is whether the canvas handler called
    // preventDefault — it unconditionally did, before this fix, regardless
    // of focus.
    const { getByLabelText } = renderCanvas(() => {});
    const input = getByLabelText('node-name') as HTMLInputElement;
    input.focus();

    const event = new KeyboardEvent('keydown', { key: 'a', ctrlKey: true, bubbles: true, cancelable: true });
    input.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(false);
  });
});
