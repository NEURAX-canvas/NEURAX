/**
 * Undo and redo for the design canvas.
 *
 * A drag-and-drop editor without Ctrl+Z is not a tool anyone can work in for a
 * day. Every mis-drop, every parameter typed into the wrong field, every
 * Delete pressed with the wrong block selected was permanent — and on a canvas
 * the wrong thing is selected surprisingly often, because selection follows the
 * pointer.
 *
 * The design is three pieces of state — blocks, connections, groups — mutated
 * from about twenty call sites across the page. Rather than route all of them
 * through a reducer, which would mean rewriting every handler and risking a
 * regression in each, this observes the three values and records them. The
 * trade-off is deliberate: an observer cannot know *what* changed, only that
 * something did, so it cannot label an entry "moved Attention" — but it also
 * cannot miss a change, including ones made by code paths written later.
 *
 * Two details make it behave the way people expect:
 *
 *  - **Coalescing.** Dragging a block fires an update per animation frame.
 *    Recording each would make Ctrl+Z rewind one pixel at a time, so changes
 *    settle for a moment before becoming an entry. A drag is one undo.
 *  - **Not recording its own writes.** Applying a snapshot changes the very
 *    state being observed. Without a guard, undo would push the undone state
 *    onto the stack and the two arrows would fight each other.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { CanvasNode, Connection, NodeGroup } from '@/types/architecture.ts';

/** The whole design at one moment. */
export interface DesignSnapshot {
  nodes: CanvasNode[];
  connections: Connection[];
  groups: NodeGroup[];
}

/**
 * How long changes must settle before becoming one undo entry.
 *
 * Long enough to swallow a drag, short enough that two deliberate edits a
 * second apart stay separate.
 */
const COALESCE_MS = 400;

/**
 * How many steps back the history goes.
 *
 * Snapshots hold references to the same immutable arrays the page already
 * holds, so an entry costs three pointers plus whatever genuinely changed —
 * not a copy of the design. Two hundred is far more than anyone reaches for
 * and still negligible.
 */
const MAX_ENTRIES = 200;

export interface DesignHistory {
  canUndo: boolean;
  canRedo: boolean;
  undo: () => void;
  redo: () => void;
  /**
   * Forget everything and start again from the given design — for opening a
   * file or starting a blank page, where undoing back into the previous
   * document would be nonsense.
   *
   * Takes the new design rather than reading it, because the caller has just
   * called the setters and React has not re-rendered yet: reading state here
   * would see the outgoing design and record the replacement as an edit.
   */
  reset: (snapshot: DesignSnapshot) => void;
  /** Number of steps available backwards, for a tooltip. */
  depth: number;
}

function sameDesign(a: DesignSnapshot, b: DesignSnapshot): boolean {
  // Reference comparison is exactly right here: every handler on the page
  // replaces the array rather than mutating it, so a new reference means a real
  // change and an unchanged reference means none.
  return a.nodes === b.nodes && a.connections === b.connections && a.groups === b.groups;
}

export function useDesignHistory(
  current: DesignSnapshot,
  apply: (snapshot: DesignSnapshot) => void,
): DesignHistory {
  const past = useRef<DesignSnapshot[]>([]);
  const future = useRef<DesignSnapshot[]>([]);
  const present = useRef<DesignSnapshot>(current);

  /** Set while a snapshot is being applied, so the write is not re-recorded. */
  const applying = useRef(false);

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Mirrors the refs into render, so the toolbar buttons can be disabled.
  const [{ canUndo, canRedo, depth }, setStatus] = useState({
    canUndo: false,
    canRedo: false,
    depth: 0,
  });

  const publish = useCallback(() => {
    setStatus({
      canUndo: past.current.length > 0,
      canRedo: future.current.length > 0,
      depth: past.current.length,
    });
  }, []);

  /** Turn the pending change into an entry now, rather than when it settles. */
  const commitPending = useCallback(
    (latest: DesignSnapshot) => {
      if (timer.current) {
        clearTimeout(timer.current);
        timer.current = null;
      }
      if (sameDesign(present.current, latest)) return;

      past.current.push(present.current);
      if (past.current.length > MAX_ENTRIES) past.current.shift();
      present.current = latest;

      // A new edit after undoing abandons the redo branch, which is what every
      // editor does and what users expect.
      future.current = [];
      publish();
    },
    [publish],
  );

  // Observe the design and record it once it settles.
  useEffect(() => {
    if (applying.current) {
      // This change is our own doing; adopt it silently.
      applying.current = false;
      present.current = current;
      return;
    }

    if (sameDesign(present.current, current)) return;

    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      timer.current = null;
      commitPending(current);
    }, COALESCE_MS);

    return () => {
      if (timer.current) clearTimeout(timer.current);
    };
  }, [current, commitPending]);

  const undo = useCallback(() => {
    // A change still settling is part of what the user means to undo.
    commitPending(current);

    const previous = past.current.pop();
    if (!previous) return;

    future.current.push(present.current);
    present.current = previous;
    applying.current = true;
    apply(previous);
    publish();
  }, [apply, commitPending, current, publish]);

  const redo = useCallback(() => {
    const next = future.current.pop();
    if (!next) return;

    past.current.push(present.current);
    present.current = next;
    applying.current = true;
    apply(next);
    publish();
  }, [apply, publish]);

  const reset = useCallback(
    (snapshot: DesignSnapshot) => {
      if (timer.current) {
        clearTimeout(timer.current);
        timer.current = null;
      }
      past.current = [];
      future.current = [];
      present.current = snapshot;
      applying.current = false;
      publish();
    },
    [publish],
  );

  return { canUndo, canRedo, undo, redo, reset, depth };
}
