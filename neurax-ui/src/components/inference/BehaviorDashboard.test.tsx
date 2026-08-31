/**
 * "State Stability (SSM)" and "Collapse Risk (MoE)" used to render for every
 * architecture — a Transformer showed a precise-looking "State Stability
 * (SSM): 95%" for a concept (sequential hidden-state coherence) it doesn't
 * have, computed from a flat per-family constant rather than anything real
 * about the model. Confirmed live across Transformer/SSM/MoE designs before
 * this fix (see conversation), fixed in `neurax-ir`'s inference pass to
 * return `None` outside the family each concept actually applies to.
 *
 * These tests hold the frontend's half of that fix: the cards must actually
 * disappear when the backend omits the field, not just type-check as
 * optional.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TooltipProvider } from '@/components/ui/tooltip.tsx';
import { BehaviorDashboard } from './BehaviorDashboard';
import type { InferenceReport } from '@/services/neuraxApi.ts';

function baseReport(overrides: Partial<InferenceReport> = {}): InferenceReport {
  return {
    stability_index: { score: 0.5, level: 'drift' },
    entropy_evolution: Array(20).fill(1.5),
    hallucination_risk: { risk: 'low', confidence: 80, capacity_component: null, sampling_component: 0.2 },
    attention_focus: Array(12).fill(0.5),
    context_degradation: 42,
    sampling_volatility: { diversity: 30, determinism: 60 },
    risk_overview: { coherence: 'low', overconfidence: 'low', degeneration: 'low' },
    ...overrides,
  };
}

function renderDashboard(report: InferenceReport) {
  return render(
    <TooltipProvider>
      <BehaviorDashboard architectureType="transformer" report={report} loading={false} error={null} />
    </TooltipProvider>,
  );
}

describe('BehaviorDashboard family-relevant widgets', () => {
  it('hides "State Stability (SSM)" and "Collapse Risk (MoE)" when the backend omits them', () => {
    renderDashboard(baseReport({ state_stability: undefined }));
    expect(screen.queryByText('State Stability (SSM)')).toBeNull();
    expect(screen.queryByText('Collapse Risk (MoE)')).toBeNull();
    // The universal risks are still there — only the MoE-specific row is gone.
    expect(screen.getByText('Coherence Risk')).toBeTruthy();
    expect(screen.getByText('Degeneration Risk')).toBeTruthy();
  });

  it('shows "State Stability (SSM)" when the backend provides it (a real SSM design)', () => {
    renderDashboard(baseReport({ state_stability: 0.9 }));
    expect(screen.getByText('State Stability (SSM)')).toBeTruthy();
    expect(screen.getByText('90%')).toBeTruthy();
  });

  it('shows "Collapse Risk (MoE)" when the backend provides it (a real MoE design)', () => {
    renderDashboard(baseReport({ risk_overview: { coherence: 'low', overconfidence: 'low', collapse: 'low', degeneration: 'low' } }));
    expect(screen.getByText('Collapse Risk (MoE)')).toBeTruthy();
  });

  it('handles null the same as undefined (Option<T> with no skip_serializing_if serialises to JSON null)', () => {
    renderDashboard(baseReport({
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      state_stability: null as any,
      risk_overview: { coherence: 'low', overconfidence: 'low', collapse: null as unknown as undefined, degeneration: 'low' },
    }));
    expect(screen.queryByText('State Stability (SSM)')).toBeNull();
    expect(screen.queryByText('Collapse Risk (MoE)')).toBeNull();
  });
});
