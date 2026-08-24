/**
 * The recommended-hyperparameters sync used to run from inside a `useMemo`
 * — a side effect (`setHyperparams`) called from a hook React only
 * guarantees to be pure — and reset on every `nodes`/`connections` change
 * with no way to tell "the architecture actually changed family" apart
 * from "an edit happened to touch `nodes`". A user who nudged a slider
 * then made one unrelated canvas edit had it silently reset on the next
 * render, with no warning.
 *
 * These tests hold the real behaviour to what a user actually needs: the
 * recommendation applies when the family actually changes, and manual
 * customisation survives an edit that doesn't.
 */
import { describe, it, expect } from 'vitest';
import type { ReactElement } from 'react';
import { render as rtlRender, screen, fireEvent } from '@testing-library/react';
import { ProductionWorkspace } from './ProductionWorkspace';
import type { AnalysisResult, CanvasNode } from '@/types/architecture.ts';
import { TooltipProvider } from '@/components/ui/tooltip.tsx';

/** A minimal, complete `AnalysisResult` — same shape `projectExport.test.ts`
 * builds, since a real compiled report has ~50 required fields and only
 * `generatedAt`/`totalParams` matter to what's under test here. */
function fakeAnalysis(totalParams: number): AnalysisResult {
  return {
    totalParams, activeParams: totalParams, numLayers: 2, modelType: 'transformer',
    graphDepth: 2, totalOperations: 2, criticalPathLength: 2, tensorResolutionRatio: 1,
    unresolvedDimCount: 0, totalTensorCount: 2, largestTensorBytes: 0, opsDistribution: {},
    totalFlops: 0, forwardFlops: 0, backwardFlops: 0, flopsPerToken: 0,
    flopsIncrementalDecode: 0, arithmeticIntensity: 0, bottleneck: 'compute',
    rooflinePosition: 0, estimatedFlops: '', forwardFlopsHuman: '', backwardFlopsHuman: '',
    peakVramBytes: 0, parameterMemoryBytes: 0, activationMemoryBytes: 0, gradientMemoryBytes: 0,
    optimizerStateBytes: 0, maxBatchSizeFit: 4, memoryFragmentation: 0, memoryUsage: '',
    gpuName: 'A100', gpuCount: 1, gpuMemoryGb: 80, gpuTflops: 300, gpuBandwidthGbs: 2000,
    dataParallelEfficiency: 1, communicationOverhead: 0, optimalGpuCount: 1, pipelineStages: 1,
    tensorParallelDegree: 1, latencyMs: 1, throughputTokensPerS: 100, gpuUtilization: 0.9,
    trainingCostUsd: 0.5, trainingTimeHours: 0.1, energyKwh: 0.1, co2Kg: 0.05,
    costPerMillionTokensUsd: 0.01, confidenceScore: 0.9, depth: 2,
    generatedAt: new Date().toISOString(),
  };
}

function render(ui: ReactElement) {
  const { rerender: rtlRerender, ...rest } = rtlRender(<TooltipProvider>{ui}</TooltipProvider>);
  return {
    ...rest,
    rerender: (nextUi: ReactElement) => rtlRerender(<TooltipProvider>{nextUi}</TooltipProvider>),
  };
}

const gnnNodes: CanvasNode[] = [
  {
    id: 'n1',
    type: 'graph_conv',
    name: 'GCN Layer',
    x: 0,
    y: 0,
    params: { in_features: 128, out_features: 64 },
  },
];

const ganNodes: CanvasNode[] = [
  { id: 'n1', type: 'dcgan_generator_block', name: 'Generator', x: 0, y: 0, params: {} },
];

// A node with no family-specific block type, added to a GNN design to
// simulate "an edit that doesn't change what family the design is".
const extraDenseNode: CanvasNode = {
  id: 'n2',
  type: 'dense',
  name: 'Extra Dense',
  x: 100,
  y: 0,
  params: { d_model: 64 },
};

// No family-specific block type — the generic, size-scaled learning rate
// path is the one that reads `analysis.totalParams`.
const plainDenseNodes: CanvasNode[] = [
  { id: 'n1', type: 'dense', name: 'Dense', x: 0, y: 0, params: { d_model: 128 } },
];

const baseProps = {
  connections: [],
  modelName: 'test-model',
  architectureFamily: 'gnn' as const,
};

describe('ProductionWorkspace hyperparameter sync', () => {
  it('shows the family-specific recommended learning rate for a GNN design', () => {
    render(<ProductionWorkspace {...baseProps} nodes={gnnNodes} />);
    // GNN: lr=0.01 (Kipf & Welling 2017) — SliderField formats as toExponential(1).
    expect(screen.getByText('1.0e-2')).toBeTruthy();
  });

  it('applies the new recommendation when the architecture actually changes family', () => {
    const { rerender } = render(<ProductionWorkspace {...baseProps} nodes={gnnNodes} />);
    expect(screen.getByText('1.0e-2')).toBeTruthy(); // GNN lr

    rerender(<ProductionWorkspace {...baseProps} nodes={ganNodes} />);
    expect(screen.getAllByText('2.0e-4')).toHaveLength(2); // GAN lr: generator + discriminator both default to it
  });

  /** GNN's recommendation is Adam (not AdamW) — switching to AdamW is
   * unambiguously a manual choice, never something the recommender itself
   * would produce for this family, so seeing it stick is real evidence a
   * customisation survived, not a coincidence of two recommendations
   * agreeing. Driven through the real Optimizer `<Select>`, not a direct
   * state assignment: Radix's Slider needs pointer geometry jsdom doesn't
   * provide to drive reliably via `fireEvent` alone, but Select's options
   * are plain clickable elements once open. */
  function switchOptimizerToAdamW() {
    fireEvent.click(screen.getByRole('combobox'));
    fireEvent.click(screen.getByText('AdamW'));
  }

  it('does not silently discard a manual edit when an unrelated node is added to the same family', () => {
    const { rerender } = render(<ProductionWorkspace {...baseProps} nodes={gnnNodes} />);

    expect(screen.getByRole('combobox').textContent).toBe('Adam'); // GNN's own recommendation
    switchOptimizerToAdamW();
    expect(screen.getByRole('combobox').textContent).toBe('AdamW');

    // Same family (still has a graph_conv node) — an unrelated node was
    // added, nothing about what recipe applies has changed.
    rerender(<ProductionWorkspace {...baseProps} nodes={[...gnnNodes, extraDenseNode]} />);

    expect(
      screen.getByRole('combobox').textContent,
      'an edit unrelated to family should not have reset the manual customisation',
    ).toBe('AdamW');
  });

  it('"Reset to Recommended" re-enables auto-sync on the next family change', () => {
    const { rerender } = render(<ProductionWorkspace {...baseProps} nodes={gnnNodes} />);

    switchOptimizerToAdamW();
    expect(screen.getByRole('combobox').textContent).toBe('AdamW');

    fireEvent.click(screen.getByText('Reset to Recommended'));
    expect(screen.getByRole('combobox').textContent).toBe('Adam'); // back to GNN's recommendation

    // A real family change now re-syncs again, same as it would have before
    // any manual edit ever happened.
    rerender(<ProductionWorkspace {...baseProps} nodes={ganNodes} />);
    expect(screen.getAllByText('2.0e-4')).toHaveLength(2); // GAN lr: generator + discriminator both default to it
  });

  it('recalculates the size-scaled learning rate from a real compilation, not the local estimate', () => {
    const genericProps = { ...baseProps, architectureFamily: 'transformer' as const };

    // Before any compilation: falls back to `estimateNodeParams`'s local
    // guess, which for one small dense block stays under the 1M-parameter
    // threshold — the smallest learning-rate bucket, 0.001 -> "1.0e-3".
    const { rerender } = render(<ProductionWorkspace {...genericProps} nodes={plainDenseNodes} />);
    expect(screen.getByText('1.0e-3')).toBeTruthy();

    // A real "Run Analysis" reports 200M parameters — far past what the
    // local per-node guess would ever produce for one small block. The
    // recommendation should read from the compiled report, not repeat its
    // own local estimate: 0.0001 -> "1.0e-4".
    rerender(
      <ProductionWorkspace {...genericProps} nodes={plainDenseNodes} analysis={fakeAnalysis(200_000_000)} />,
    );
    expect(screen.getByText('1.0e-4')).toBeTruthy();
  });
});
