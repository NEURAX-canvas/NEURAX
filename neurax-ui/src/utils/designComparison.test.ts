/**
 * A comparison is only useful if its verdicts are right.
 *
 * The traps are all about direction and about honesty over missing data.
 * "Latency went down" is an improvement; "VRAM went down" is an improvement;
 * "throughput went down" is not, and a table that colours all three the same
 * way is worse than no colouring. The other trap is comparing two analyses
 * taken under different conditions and reading the result as if the
 * architecture caused it.
 */
import { describe, it, expect } from 'vitest';
import {
  compareDesigns,
  formatMetric,
  formatDelta,
  DesignVariant,
  MetricComparison,
} from './designComparison';
import { AnalysisResult } from '@/types/architecture.ts';

function analysis(overrides: Partial<AnalysisResult>): AnalysisResult {
  return {
    totalParams: 7e9,
    numLayers: 32,
    peakVramBytes: 40 * 1024 ** 3,
    parameterMemoryBytes: 14 * 1024 ** 3,
    activationMemoryBytes: 8 * 1024 ** 3,
    maxBatchSizeFit: 8,
    totalFlops: 1e18,
    flopsPerToken: 1.4e10,
    arithmeticIntensity: 120,
    latencyMs: 50,
    throughputTokensPerS: 2000,
    gpuUtilization: 0.7,
    trainingCostUsd: 100000,
    trainingTimeHours: 240,
    costPerMillionTokensUsd: 0.5,
    energyKwh: 5000,
    co2Kg: 1200,
    ...overrides,
  } as AnalysisResult;
}

function variant(name: string, over: Partial<AnalysisResult>, hardware = {}): DesignVariant {
  return {
    id: name,
    name,
    capturedAt: '2026-08-14T10:00:00.000Z',
    architecture: 'transformer',
    blockCount: 14,
    connectionCount: 13,
    analysis: analysis(over),
    hardware: { hardware: 'H100', gpuCount: 8, precision: 'bf16', batchSize: 8, seqLen: 4096, ...hardware },
  };
}

function metric(metrics: MetricComparison[], key: string): MetricComparison {
  const found = metrics.find((m) => m.key === key);
  if (!found) throw new Error(`no metric ${key}`);
  return found;
}

describe('comparing two designs', () => {
  describe('knows which direction is an improvement', () => {
    it('treats less VRAM as better', () => {
      const report = compareDesigns(
        variant('A', {}),
        variant('B', { peakVramBytes: 30 * 1024 ** 3 }),
      );
      expect(metric(report.metrics, 'peakVramBytes').verdict).toBe('better');
    });

    it('treats more VRAM as worse', () => {
      const report = compareDesigns(
        variant('A', {}),
        variant('B', { peakVramBytes: 60 * 1024 ** 3 }),
      );
      expect(metric(report.metrics, 'peakVramBytes').verdict).toBe('worse');
    });

    it('treats more throughput as better, unlike every other rising number', () => {
      const report = compareDesigns(
        variant('A', {}),
        variant('B', { throughputTokensPerS: 3000 }),
      );
      expect(metric(report.metrics, 'throughputTokensPerS').verdict).toBe('better');
    });

    it('treats a bigger batch that fits as better', () => {
      const report = compareDesigns(variant('A', {}), variant('B', { maxBatchSizeFit: 16 }));
      expect(metric(report.metrics, 'maxBatchSizeFit').verdict).toBe('better');
    });

    it('passes no judgement on parameter count', () => {
      // More parameters is neither good nor bad; it is the question, not the
      // answer.
      const report = compareDesigns(variant('A', {}), variant('B', { totalParams: 70e9 }));
      expect(metric(report.metrics, 'totalParams').verdict).toBe('neutral');
      expect(metric(report.metrics, 'totalParams').ratio).toBeCloseTo(9, 5);
    });
  });

  describe('computes the change', () => {
    it('reports the delta and the ratio', () => {
      const report = compareDesigns(
        variant('A', { trainingCostUsd: 100000 }),
        variant('B', { trainingCostUsd: 75000 }),
      );
      const cost = metric(report.metrics, 'trainingCostUsd');
      expect(cost.delta).toBe(-25000);
      expect(cost.ratio).toBeCloseTo(-0.25, 5);
      expect(formatDelta(cost)).toBe('-25.0 %');
    });

    it('calls a change below half a percent no change at all', () => {
      const report = compareDesigns(
        variant('A', { trainingCostUsd: 100000 }),
        variant('B', { trainingCostUsd: 100200 }),
      );
      const cost = metric(report.metrics, 'trainingCostUsd');
      expect(cost.verdict).toBe('neutral');
      expect(formatDelta(cost)).toBe('=');
    });

    it('does not divide by a zero baseline', () => {
      const report = compareDesigns(
        variant('A', { trainingCostUsd: 0 }),
        variant('B', { trainingCostUsd: 500 }),
      );
      const cost = metric(report.metrics, 'trainingCostUsd');
      expect(cost.ratio).toBeNull();
      expect(cost.delta).toBe(500);
      expect(Number.isFinite(cost.delta!)).toBe(true);
    });

    it('reports a missing side as missing rather than as zero', () => {
      // `latencyMs` is nullable in the analysis; a design that was never
      // analysed on hardware has no latency, and showing 0 ms would read as
      // infinitely fast.
      const report = compareDesigns(variant('A', { latencyMs: null }), variant('B', {}));
      const latency = metric(report.metrics, 'latencyMs');
      expect(latency.baseline).toBeNull();
      expect(latency.delta).toBeNull();
      expect(latency.verdict).toBe('neutral');
      expect(formatDelta(latency)).toBe('new');
    });

    it('ignores a non-finite value from the compiler', () => {
      const report = compareDesigns(
        variant('A', { arithmeticIntensity: Number.POSITIVE_INFINITY }),
        variant('B', {}),
      );
      expect(metric(report.metrics, 'arithmeticIntensity').baseline).toBeNull();
    });
  });

  describe('says when two designs are not comparable', () => {
    it('flags a change of GPU', () => {
      const report = compareDesigns(
        variant('A', {}, { hardware: 'A100' }),
        variant('B', {}, { hardware: 'H100' }),
      );
      expect(report.incomparable.join(' ')).toMatch(/GPU: A100 → H100/);
    });

    it('flags a change of precision, which moves every memory figure', () => {
      const report = compareDesigns(
        variant('A', {}, { precision: 'fp32' }),
        variant('B', {}, { precision: 'bf16' }),
      );
      expect(report.incomparable.join(' ')).toMatch(/Precision/);
    });

    it('flags a change of batch size and sequence length', () => {
      const report = compareDesigns(
        variant('A', {}, { batchSize: 8, seqLen: 4096 }),
        variant('B', {}, { batchSize: 32, seqLen: 8192 }),
      );
      expect(report.incomparable).toHaveLength(2);
    });

    it('stays quiet when the terms match', () => {
      const report = compareDesigns(variant('A', {}), variant('B', { totalParams: 8e9 }));
      expect(report.incomparable).toEqual([]);
    });

    it('flags a comparison across architecture families', () => {
      const dense = variant('A', {});
      const routed = { ...variant('B', {}), architecture: 'moe' as const };
      expect(compareDesigns(dense, routed).incomparable.join(' ')).toMatch(/family/i);
    });
  });

  describe('formats figures the way an engineer reads them', () => {
    it('renders bytes in binary units', () => {
      expect(formatMetric(40 * 1024 ** 3, 'bytes')).toBe('40.00 GB');
      expect(formatMetric(1536, 'bytes')).toBe('1.5 KB');
    });

    it('renders parameter counts in billions', () => {
      expect(formatMetric(6.74e9, 'count')).toBe('6.74 B');
      expect(formatMetric(124e6, 'count')).toBe('124.00 M');
    });

    it('renders FLOPs in the usual scale', () => {
      expect(formatMetric(1e18, 'flops')).toBe('1.00 EFLOP');
      expect(formatMetric(1.4e10, 'flops')).toBe('14.00 GFLOP');
    });

    it('renders long training runs in days and short latencies in ms', () => {
      expect(formatMetric(240, 'hours')).toBe('10.0 d');
      expect(formatMetric(6, 'hours')).toBe('6.0 h');
      expect(formatMetric(50, 'ms')).toBe('50.00 ms');
      expect(formatMetric(2500, 'ms')).toBe('2.50 s');
    });

    it('renders utilisation whether the compiler gave a fraction or a percent', () => {
      expect(formatMetric(0.7, 'percent')).toBe('70.0 %');
      expect(formatMetric(70, 'percent')).toBe('70.0 %');
    });

    it('renders an absent value as a dash, never as zero', () => {
      expect(formatMetric(null, 'bytes')).toBe('—');
      expect(formatMetric(null, 'usd')).toBe('—');
    });
  });
});
