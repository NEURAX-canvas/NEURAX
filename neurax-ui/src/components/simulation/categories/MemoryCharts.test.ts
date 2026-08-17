// The multi-precision weight-memory comparison exists to answer "will this
// fit at a lower precision" without re-running a full analysis for each one
// — r/LocalLLaMA's most common question, and NEURAX's strongest hook. It has
// to use the exact bytes-per-parameter the real compiler uses
// (neurax-formulas::dtype_bytes), not a separate estimate that could quietly
// drift from what a real analysis at that precision would report.
import { describe, it, expect } from 'vitest';
import { computePrecisionMemory } from './MemoryCharts';

describe('computePrecisionMemory', () => {
  it('returns nothing for a design with no computed parameters', () => {
    expect(computePrecisionMemory(0)).toEqual([]);
    expect(computePrecisionMemory(-5)).toEqual([]);
  });

  it('matches neurax-formulas::dtype_bytes exactly, for a real model size', () => {
    // Mistral-7B: 7,243,463,935 real parameters (verified against the
    // published size through a live compiler — see huggingfaceImporter
    // .integration.test.ts).
    const totalParams = 7_243_463_935;
    const rows = computePrecisionMemory(totalParams);

    const byId = Object.fromEntries(rows.map((r) => [r.id, r.bytes]));
    expect(byId.fp32).toBe(totalParams * 4);
    expect(byId.fp16).toBe(totalParams * 2);
    expect(byId.bf16).toBe(totalParams * 2);
    expect(byId.int8).toBe(totalParams * 1);
    expect(byId.int4).toBe(totalParams * 1);
  });

  it('orders precisions from largest to smallest footprint', () => {
    const rows = computePrecisionMemory(1_000_000_000);
    const bytesInOrder = rows.map((r) => r.bytes);
    expect(bytesInOrder).toEqual([...bytesInOrder].sort((a, b) => b - a));
  });

  it('halving from FP32 to FP16 is exact, not approximate', () => {
    const totalParams = 123_456_789;
    const rows = computePrecisionMemory(totalParams);
    const fp32 = rows.find((r) => r.id === 'fp32')!.bytes;
    const fp16 = rows.find((r) => r.id === 'fp16')!.bytes;
    expect(fp16).toBe(fp32 / 2);
  });
});
