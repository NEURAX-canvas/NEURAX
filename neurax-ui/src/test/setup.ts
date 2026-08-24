import '@testing-library/jest-dom';

// jsdom has no ResizeObserver — Radix's Slider (and other size-aware
// primitives) call it unconditionally on mount, so any test rendering one
// throws `ResizeObserver is not defined` before jest-dom even gets a
// chance to run an assertion. A no-op stub is all any test here needs:
// nothing asserts on a resize callback actually firing.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}