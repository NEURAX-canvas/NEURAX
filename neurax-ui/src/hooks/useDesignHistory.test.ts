/**
 * Undo has to be trustworthy, or it is worse than not having it.
 *
 * The failure everyone has met: pressing Ctrl+Z, watching the wrong thing come
 * back, pressing it again, and losing more. That comes from three specific
 * bugs, and each has a test here — an observer that records its own writes and
 * so fights itself; a drag recorded as three hundred entries; and a redo branch
 * that survives a new edit and reappears later out of nowhere.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useDesignHistory, DesignSnapshot } from './useDesignHistory';
import { CanvasNode } from '@/types/architecture.ts';

function block(id: string, x = 0): CanvasNode {
  return { id, type: 'dense', name: id, x, y: 0, params: {} };
}

function design(nodes: CanvasNode[]): DesignSnapshot {
  return { nodes, connections: [], groups: [] };
}

/**
 * Drives the hook the way the page does: state lives outside, the hook observes
 * it, and `apply` writes back.
 */
function harness(initial: DesignSnapshot) {
  let state = initial;

  const view = renderHook(
    ({ snapshot }: { snapshot: DesignSnapshot }) =>
      useDesignHistory(snapshot, (next) => {
        state = next;
      }),
    { initialProps: { snapshot: initial } },
  );

  return {
    get state() {
      return state;
    },
    get history() {
      return view.result.current;
    },
    /** An edit made by the user. */
    edit(next: DesignSnapshot) {
      state = next;
      view.rerender({ snapshot: next });
    },
    /** Let the coalescing window expire, so the pending change becomes an entry. */
    settle() {
      act(() => {
        vi.advanceTimersByTime(500);
      });
    },
    /** Push whatever `apply` last wrote back into the observed props. */
    sync() {
      view.rerender({ snapshot: state });
    },
  };
}

describe('design history', () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it('starts with nothing to undo', () => {
    const h = harness(design([block('a')]));
    expect(h.history.canUndo).toBe(false);
    expect(h.history.canRedo).toBe(false);
  });

  it('restores the previous design', () => {
    const first = design([block('a')]);
    const h = harness(first);

    h.edit(design([block('a'), block('b')]));
    h.settle();
    expect(h.history.canUndo).toBe(true);

    act(() => h.history.undo());
    expect(h.state).toBe(first);
    expect(h.state.nodes.map((n) => n.id)).toEqual(['a']);
  });

  it('redoes what it undid', () => {
    const first = design([block('a')]);
    const second = design([block('a'), block('b')]);
    const h = harness(first);

    h.edit(second);
    h.settle();

    act(() => h.history.undo());
    h.sync();
    expect(h.state.nodes).toHaveLength(1);
    expect(h.history.canRedo).toBe(true);

    act(() => h.history.redo());
    expect(h.state).toBe(second);
    expect(h.state.nodes).toHaveLength(2);
  });

  it('does not record its own writes', () => {
    // The bug this guards: undo applies a snapshot, the observer sees state
    // change and records it, so the next undo returns to where you just were
    // and the two arrows never make progress.
    const first = design([block('a')]);
    const h = harness(first);

    h.edit(design([block('a'), block('b')]));
    h.settle();

    act(() => h.history.undo());
    h.sync();
    h.settle();

    expect(h.history.canUndo, 'the undo itself became an entry').toBe(false);
    expect(h.history.depth).toBe(0);
  });

  it('records a drag as one step, not one per frame', () => {
    const h = harness(design([block('a', 0)]));

    // A drag: many updates inside the coalescing window.
    for (let x = 1; x <= 60; x++) {
      h.edit(design([block('a', x)]));
    }
    h.settle();

    expect(h.history.depth).toBe(1);

    act(() => h.history.undo());
    expect(h.state.nodes[0].x, 'undo should reach the start of the drag').toBe(0);
  });

  it('keeps deliberate edits separate', () => {
    const h = harness(design([block('a')]));

    h.edit(design([block('a'), block('b')]));
    h.settle();
    h.edit(design([block('a'), block('b'), block('c')]));
    h.settle();

    expect(h.history.depth).toBe(2);

    act(() => h.history.undo());
    h.sync();
    expect(h.state.nodes.map((n) => n.id)).toEqual(['a', 'b']);

    act(() => h.history.undo());
    h.sync();
    expect(h.state.nodes.map((n) => n.id)).toEqual(['a']);
  });

  it('undoes a change that has not settled yet', () => {
    // Someone who drops a block and immediately presses Ctrl+Z means that drop.
    const first = design([block('a')]);
    const h = harness(first);

    h.edit(design([block('a'), block('b')]));
    // Deliberately no settle().

    act(() => h.history.undo());
    expect(h.state).toBe(first);
  });

  it('abandons the redo branch once a new edit is made', () => {
    const h = harness(design([block('a')]));

    h.edit(design([block('a'), block('b')]));
    h.settle();

    act(() => h.history.undo());
    h.sync();
    expect(h.history.canRedo).toBe(true);

    h.edit(design([block('a'), block('z')]));
    h.settle();

    expect(h.history.canRedo, 'the abandoned branch must not come back').toBe(false);
  });

  it('does nothing when there is nothing to undo or redo', () => {
    const only = design([block('a')]);
    const h = harness(only);

    act(() => h.history.undo());
    act(() => h.history.redo());
    expect(h.state).toBe(only);
  });

  it('ignores a re-render that changed nothing', () => {
    const same = design([block('a')]);
    const h = harness(same);

    h.edit(same);
    h.settle();
    expect(h.history.canUndo).toBe(false);
  });

  it('forgets history when a new document is opened', () => {
    const h = harness(design([block('a')]));

    h.edit(design([block('a'), block('b')]));
    h.settle();
    expect(h.history.canUndo).toBe(true);

    const opened = design([block('x'), block('y')]);
    act(() => {
      h.history.reset(opened);
    });
    h.edit(opened);

    expect(h.history.canUndo, 'undo must not reach into the previous document').toBe(false);
    expect(h.history.canRedo).toBe(false);
  });

  it('records edits again after a reset', () => {
    const h = harness(design([block('a')]));

    const opened = design([block('x')]);
    act(() => {
      h.history.reset(opened);
    });
    h.edit(opened);

    h.edit(design([block('x'), block('y')]));
    h.settle();

    expect(h.history.canUndo).toBe(true);
    act(() => h.history.undo());
    expect(h.state.nodes.map((n) => n.id)).toEqual(['x']);
  });

  it('survives a long editing session without unbounded growth', () => {
    const h = harness(design([block('a')]));

    for (let i = 0; i < 260; i++) {
      h.edit(design([block('a', i)]));
      h.settle();
    }

    // Capped, and still usable — the oldest steps are the ones dropped.
    expect(h.history.depth).toBeLessThanOrEqual(200);
    expect(h.history.canUndo).toBe(true);
    act(() => h.history.undo());
    expect(h.state.nodes[0].x).toBe(258);
  });
});
