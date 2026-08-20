/**
 * Deleting a connection looked broken without being broken in any single
 * event handler: the delete (×) button rendered correctly and its click
 * handler called `onDeleteConnection` with the right ID, but it was drawn
 * inline inside its own connection's <g> — a sibling of every *other*
 * connection's 20px-wide invisible hit-area, one of which could paint on
 * top of it in SVG's document order and swallow the click a few pixels off
 * centre. The button was there, visible, red — and unreliable, which reads
 * exactly like "it doesn't work". Fixed by rendering the selected
 * connection's delete button in its own pass, after every connection, so
 * nothing painted later can sit on top of it.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { ArchitectureCanvas } from './ArchitectureCanvas.tsx';
import type { CanvasNode, Connection } from '@/types/architecture.ts';

// jsdom doesn't implement ResizeObserver; the zoom slider (Radix UI) uses it.
(globalThis as any).ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

const baseProps = {
  groups: [],
  selectedNodeId: null,
  onSelectNode: () => {},
  onUpdateNode: () => {},
  onAddNode: () => {},
  onDeleteNode: () => {},
  onDuplicateNode: () => {},
  onAddConnection: () => {},
};

function selectConnection(container: HTMLElement, index = 0) {
  const paths = container.querySelectorAll('path[stroke="transparent"]');
  fireEvent.click(paths[index]);
}

function findDeleteButton(container: HTMLElement) {
  const deleteText = Array.from(container.querySelectorAll('text')).find(
    (t) => t.textContent === '×',
  );
  return deleteText?.closest('g') ?? null;
}

describe('deleting a connection', () => {
  const nodes: CanvasNode[] = [
    { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {} },
    { id: 'n2', type: 'output', name: 'Output', x: 400, y: 140, params: {} },
  ];
  const connections: Connection[] = [{ id: 'c1', from: 'n1', to: 'n2' }];

  it('clicking the connection selects it and shows a delete button', () => {
    const { container } = render(
      <ArchitectureCanvas {...baseProps} nodes={nodes} connections={connections} />,
    );
    expect(findDeleteButton(container)).toBeNull();
    selectConnection(container);
    expect(findDeleteButton(container)).not.toBeNull();
  });

  it('clicking the delete button removes exactly that connection', () => {
    const onDeleteConnection = vi.fn();
    const { container } = render(
      <ArchitectureCanvas
        {...baseProps}
        nodes={nodes}
        connections={connections}
        onDeleteConnection={onDeleteConnection}
      />,
    );
    selectConnection(container);
    fireEvent.click(findDeleteButton(container)!);
    expect(onDeleteConnection).toHaveBeenCalledTimes(1);
    expect(onDeleteConnection).toHaveBeenCalledWith('c1');
  });

  it('pressing Delete with a connection selected removes it, same as the button', () => {
    const onDeleteConnection = vi.fn();
    const { container } = render(
      <ArchitectureCanvas
        {...baseProps}
        nodes={nodes}
        connections={connections}
        onDeleteConnection={onDeleteConnection}
      />,
    );
    selectConnection(container);
    fireEvent.keyDown(window, { key: 'Delete' });
    expect(onDeleteConnection).toHaveBeenCalledWith('c1');
  });

  it('pressing Backspace with a connection selected removes it too', () => {
    const onDeleteConnection = vi.fn();
    const { container } = render(
      <ArchitectureCanvas
        {...baseProps}
        nodes={nodes}
        connections={connections}
        onDeleteConnection={onDeleteConnection}
      />,
    );
    selectConnection(container);
    fireEvent.keyDown(window, { key: 'Backspace' });
    expect(onDeleteConnection).toHaveBeenCalledWith('c1');
  });

  it("the delete button is not nested inside any connection's own group", () => {
    // The regression this guards: the button used to be a sibling of its
    // own connection's wide invisible hit-area, inside that connection's
    // <g> — meaning a *different* connection's <g>, rendered later in the
    // same connections pass, could paint on top of it and swallow the
    // click. Rendered in its own pass after every connection instead, its
    // parent is the connections layer itself, not any one connection's <g>.
    const manyConnections: Connection[] = [
      { id: 'c1', from: 'n1', to: 'n2' },
      { id: 'c2', from: 'n2', to: 'n1' },
      { id: 'c3', from: 'n1', to: 'n2' },
    ];
    const { container } = render(
      <ArchitectureCanvas {...baseProps} nodes={nodes} connections={manyConnections} />,
    );
    selectConnection(container, 0);
    const deleteButton = findDeleteButton(container)!;
    // A connection's own <g> always contains its invisible hit-area path;
    // the delete button's ancestor chain should contain none of them.
    const ancestorGroups = [];
    let el: Element | null = deleteButton;
    while (el) {
      ancestorGroups.push(el);
      el = el.parentElement;
    }
    for (const ancestor of ancestorGroups) {
      const ownHitArea = ancestor.querySelector(':scope > path[stroke="transparent"]');
      expect(ownHitArea, "an ancestor of the delete button owns a connection's hit-area").toBeNull();
    }
  });

  it('selecting a different connection moves the delete button to it, not both at once', () => {
    const twoConnections: Connection[] = [
      { id: 'c1', from: 'n1', to: 'n2' },
      { id: 'c2', from: 'n2', to: 'n1' },
    ];
    const onDeleteConnection = vi.fn();
    const { container } = render(
      <ArchitectureCanvas
        {...baseProps}
        nodes={nodes}
        connections={twoConnections}
        onDeleteConnection={onDeleteConnection}
      />,
    );
    selectConnection(container, 0);
    selectConnection(container, 1);
    // Only one delete button exists at a time.
    const deleteButtons = Array.from(container.querySelectorAll('text')).filter(
      (t) => t.textContent === '×',
    );
    expect(deleteButtons).toHaveLength(1);

    fireEvent.click(findDeleteButton(container)!);
    expect(onDeleteConnection).toHaveBeenCalledWith('c2');
  });
});
