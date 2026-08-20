/**
 * The document name used to be display-only — visible nowhere in the header
 * and, where it appeared at all, only inside a Save button's hover tooltip.
 * A design imported from HuggingFace, or one started from a template, had
 * no way to be called anything else without a full Save As. This is the
 * rename control that replaces that: click the name, type, commit.
 */
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DocumentNameField } from './TopNav';

describe('DocumentNameField', () => {
  it('shows the current name as a clickable label', () => {
    render(<DocumentNameField name="My LLaMA" />);
    expect(screen.getByRole('button', { name: /rename this design/i })).toHaveTextContent(
      'My LLaMA',
    );
  });

  it('commits a new name on Enter', () => {
    const onRename = vi.fn();
    render(<DocumentNameField name="My LLaMA" onRename={onRename} />);

    fireEvent.click(screen.getByRole('button', { name: /rename this design/i }));
    const input = screen.getByLabelText('Design name');
    fireEvent.change(input, { target: { value: 'Custom Mixtral' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(onRename).toHaveBeenCalledWith('Custom Mixtral');
    // Back to the label — showing the parent's `name` prop, not yet updated
    // here, exactly as it will be in the real app until the rename lands in
    // `documentName` state and flows back down as a new prop.
    expect(screen.queryByLabelText('Design name')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /rename this design/i })).toBeInTheDocument();
  });

  it('commits on blur too, not only Enter', () => {
    const onRename = vi.fn();
    render(<DocumentNameField name="My LLaMA" onRename={onRename} />);

    fireEvent.click(screen.getByRole('button', { name: /rename this design/i }));
    const input = screen.getByLabelText('Design name');
    fireEvent.change(input, { target: { value: 'Renamed On Blur' } });
    fireEvent.blur(input);

    expect(onRename).toHaveBeenCalledWith('Renamed On Blur');
  });

  it('reverts without committing on Escape', () => {
    const onRename = vi.fn();
    render(<DocumentNameField name="My LLaMA" onRename={onRename} />);

    fireEvent.click(screen.getByRole('button', { name: /rename this design/i }));
    const input = screen.getByLabelText('Design name');
    fireEvent.change(input, { target: { value: 'Should not stick' } });
    fireEvent.keyDown(input, { key: 'Escape' });

    expect(onRename).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /rename this design/i })).toHaveTextContent(
      'My LLaMA',
    );
  });

  it('rejects a blank name rather than committing one', () => {
    const onRename = vi.fn();
    render(<DocumentNameField name="My LLaMA" onRename={onRename} />);

    fireEvent.click(screen.getByRole('button', { name: /rename this design/i }));
    const input = screen.getByLabelText('Design name');
    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(onRename).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /rename this design/i })).toHaveTextContent(
      'My LLaMA',
    );
  });

  it('does not call onRename when the name did not actually change', () => {
    const onRename = vi.fn();
    render(<DocumentNameField name="My LLaMA" onRename={onRename} />);

    fireEvent.click(screen.getByRole('button', { name: /rename this design/i }));
    fireEvent.keyDown(screen.getByLabelText('Design name'), { key: 'Enter' });

    expect(onRename).not.toHaveBeenCalled();
  });
});
