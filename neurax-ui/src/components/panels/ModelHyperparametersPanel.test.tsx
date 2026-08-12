import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ModelHyperparametersPanel } from './ModelHyperparametersPanel';
import { HardwareProvider } from '@/contexts/HardwareContext';

const renderPanel = () =>
  render(
    <HardwareProvider>
      <ModelHyperparametersPanel initialFamily="transformer" />
    </HardwareProvider>,
  );

describe('ModelHyperparametersPanel', () => {
  it('separates required from optional hyperparameters', () => {
    renderPanel();
    // "Required for analysis" also appears on individual unset fields, so match
    // the section heading specifically.
    expect(screen.getByText(/^Required for Transformer/i)).toBeTruthy();
    expect(screen.getByText(/^Optional$/i)).toBeTruthy();
  });

  it('filters the list by name', () => {
    renderPanel();
    const filter = screen.getByLabelText('Filter hyperparameters');
    fireEvent.change(filter, { target: { value: 'warmup' } });
    expect(screen.getByText('Warmup Steps')).toBeTruthy();
    expect(screen.queryByText('Vocabulary Size')).toBeNull();
  });

  it('reports when nothing matches the filter', () => {
    renderPanel();
    fireEvent.change(screen.getByLabelText('Filter hyperparameters'), {
      target: { value: 'zzzz-not-a-parameter' },
    });
    expect(screen.getByText(/No hyperparameter matches/i)).toBeTruthy();
  });

  it('adds a custom hyperparameter and shows it', () => {
    renderPanel();
    fireEvent.change(screen.getByLabelText('Custom hyperparameter name'), {
      target: { value: 'my_knob' },
    });
    fireEvent.change(screen.getByLabelText('Custom hyperparameter value'), {
      target: { value: '42' },
    });
    fireEvent.click(screen.getByText('Add'));
    expect(screen.getByText('my_knob')).toBeTruthy();
    expect(screen.getByLabelText('Remove my_knob')).toBeTruthy();
  });

  it('removes a custom hyperparameter', () => {
    renderPanel();
    fireEvent.change(screen.getByLabelText('Custom hyperparameter name'), {
      target: { value: 'temp_knob' },
    });
    fireEvent.click(screen.getByText('Add'));
    expect(screen.getByText('temp_knob')).toBeTruthy();
    fireEvent.click(screen.getByLabelText('Remove temp_knob'));
    expect(screen.queryByText('temp_knob')).toBeNull();
  });

  it('rejects a custom name that collides with a built-in', () => {
    renderPanel();
    fireEvent.change(screen.getByLabelText('Custom hyperparameter name'), {
      target: { value: 'learningRate' },
    });
    expect(screen.getByText(/already uses this name/i)).toBeTruthy();
  });

  it('rejects a syntactically invalid custom name', () => {
    renderPanel();
    fireEvent.change(screen.getByLabelText('Custom hyperparameter name'), {
      target: { value: '9 bad name!' },
    });
    expect(screen.getByText(/Letters, digits and underscore/i)).toBeTruthy();
  });

  it('accepts many custom hyperparameters', () => {
    renderPanel();
    const nameInput = screen.getByLabelText('Custom hyperparameter name');
    const valueInput = screen.getByLabelText('Custom hyperparameter value');
    for (let i = 0; i < 12; i++) {
      fireEvent.change(nameInput, { target: { value: `knob_${i}` } });
      fireEvent.change(valueInput, { target: { value: String(i) } });
      fireEvent.click(screen.getByText('Add'));
    }
    for (let i = 0; i < 12; i++) {
      expect(screen.getByText(`knob_${i}`)).toBeTruthy();
    }
  });
});
