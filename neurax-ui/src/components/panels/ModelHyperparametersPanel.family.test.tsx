import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ModelHyperparametersPanel } from './ModelHyperparametersPanel';
import { HardwareProvider } from '@/contexts/HardwareContext';
import { getParamsForFamily } from '@/utils/hyperparameterDefs';

const renderPanel = (family: any) =>
  render(
    <HardwareProvider>
      <ModelHyperparametersPanel initialFamily={family} />
    </HardwareProvider>,
  );

describe('hyperparameters are scoped to the selected family', () => {
  it('transformer shows its own params and not other families\' ', () => {
    renderPanel('transformer');
    // Own
    expect(screen.queryByText('Number of Heads')).toBeTruthy();
    // Foreign — must NOT be on screen
    expect(screen.queryByText('State Dimension')).toBeNull();     // ssm
    expect(screen.queryByText('Number of Experts')).toBeNull();   // moe
    expect(screen.queryByText('Image Height')).toBeNull();        // cnn
  });

  it('ssm shows state-space params and not transformer-only ones', () => {
    renderPanel('ssm');
    const keys = getParamsForFamily('ssm' as any).map(p => String(p.key));
    expect(keys).toContain('dState');
    expect(keys).not.toContain('numExperts');
    expect(screen.queryByText('Number of Experts')).toBeNull();
  });

  it('switching family swaps the visible hyperparameters', () => {
    renderPanel('transformer');
    expect(screen.queryByText('Number of Heads')).toBeTruthy();
    expect(screen.queryByText('Number of Experts')).toBeNull();

    // Click the MoE family tab
    fireEvent.click(screen.getByText('Mixture of Experts'));

    expect(screen.queryByText('Number of Experts')).toBeTruthy();
  });

  it('family-specific key sets genuinely differ', () => {
    const t = new Set(getParamsForFamily('transformer' as any).map(p => String(p.key)));
    const s = new Set(getParamsForFamily('ssm' as any).map(p => String(p.key)));
    const c = new Set(getParamsForFamily('cnn' as any).map(p => String(p.key)));
    expect(t.has('numHeads')).toBe(true);
    expect(s.has('numHeads')).toBe(false);
    expect(s.has('dState')).toBe(true);
    expect(c.has('imgHeight')).toBe(true);
    expect(t.has('imgHeight')).toBe(false);
  });

  it('follows the family selected on the canvas, not just the one it opened with', () => {
    const { rerender } = render(
      <HardwareProvider><ModelHyperparametersPanel initialFamily={'transformer' as any} /></HardwareProvider>,
    );
    expect(screen.queryByText('Number of Heads')).toBeTruthy();
    expect(screen.queryByText('Number of Experts')).toBeNull();

    // The canvas architecture changed while this panel is mounted; editing
    // another family's settings would silently configure the wrong model.
    rerender(
      <HardwareProvider><ModelHyperparametersPanel initialFamily={'moe' as any} /></HardwareProvider>,
    );
    expect(screen.queryByText('Number of Experts')).toBeTruthy();
  });

  it('never shows a parameter that belongs to another family', () => {
    // Markers are keys tied to one architecture. Anything appearing outside its
    // family is a leak — except the pairs listed as shared, where the parameter
    // genuinely applies to both (MoE models are transformers with expert FFNs;
    // GANs and diffusion models are both image models; a spiking neuron has an
    // internal state dimension just as a state-space block does).
    const OWNER: Record<string, string> = {
      numHeads: 'transformer', kvHeads: 'transformer', ropeTheta: 'transformer',
      numExperts: 'moe', topK: 'moe', expertCapacity: 'moe',
      imgHeight: 'cnn', poolType: 'cnn',
      dState: 'ssm', dtRank: 'ssm', expandFactor: 'ssm',
      numNodes: 'gnn', nodeFeatDim: 'gnn', aggrType: 'gnn',
      isBidirectional: 'rnn',
      timesteps: 'snn', spikeRate: 'snn',
      actionDim: 'rl', stateDim: 'rl',
      numDenoisingSteps: 'diffusion', guidanceScale: 'diffusion',
    };
    const SHARED = new Set([
      'moe:numHeads',        // a MoE block sits inside a transformer stack
      'gan:imgHeight',       // image model
      'diffusion:imgHeight', // image model
      'snn:dState',          // membrane state, its own meaning under the same key
    ]);

    const families = ['transformer','moe','cnn','ssm','gnn','rnn','gan','snn','rl','diffusion','experimental'];
    const leaks: string[] = [];
    for (const family of families) {
      for (const param of getParamsForFamily(family as any)) {
        const key = String(param.key);
        const owner = OWNER[key];
        if (owner && owner !== family && !SHARED.has(`${family}:${key}`)) {
          leaks.push(`${family} shows ${key}, which belongs to ${owner}`);
        }
      }
    }
    expect(leaks).toEqual([]);
  });

  it('universal training params are present in every family by design', () => {
    for (const f of ['transformer','moe','cnn','ssm','gnn','rnn','gan','snn','rl','diffusion','experimental']) {
      const keys = new Set(getParamsForFamily(f as any).map(p => String(p.key)));
      expect(keys.has('learningRate'), `${f}`).toBe(true);
      expect(keys.has('optimizer'), `${f}`).toBe(true);
    }
  });
});
