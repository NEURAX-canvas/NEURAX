/**
 * The simulation target decides what every other number means, so the five
 * offered chips must exist in the compiler's hardware database and their
 * specifications must come from it rather than from a copy in the UI.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { SimulationTargetPanel } from './SimulationTargetPanel';
import { HardwareProvider } from '@/contexts/HardwareContext';

const CATALOGUE = [
  { name: 'H100-SXM', manufacturer: 'NVIDIA', memory_gb: 80, memory_bandwidth_gbs: 3352, tflops_fp64: 34, tflops_fp32: 67, tflops_fp16: 989, tflops_bf16: 989 },
  { name: 'A100-SXM', manufacturer: 'NVIDIA', memory_gb: 80, memory_bandwidth_gbs: 2039, tflops_fp64: 19.5, tflops_fp32: 19.5, tflops_fp16: 312, tflops_bf16: 312 },
  { name: 'L40S', manufacturer: 'NVIDIA', memory_gb: 48, memory_bandwidth_gbs: 864, tflops_fp64: 0, tflops_fp32: 91, tflops_fp16: 362, tflops_bf16: 362 },
  { name: 'RTX4090', manufacturer: 'NVIDIA', memory_gb: 24, memory_bandwidth_gbs: 1008, tflops_fp64: 0, tflops_fp32: 82, tflops_fp16: 165, tflops_bf16: 165 },
  { name: 'T4', manufacturer: 'NVIDIA', memory_gb: 16, memory_bandwidth_gbs: 300, tflops_fp64: 0, tflops_fp32: 8.1, tflops_fp16: 65, tflops_bf16: 0 },
  { name: 'V100', manufacturer: 'NVIDIA', memory_gb: 32, memory_bandwidth_gbs: 900, tflops_fp64: 7.8, tflops_fp32: 15.7, tflops_fp16: 125, tflops_bf16: 0 },
];

vi.mock('@/services/neuraxApi.ts', () => ({
  listHardware: () => Promise.resolve(CATALOGUE),
}));

const open = () =>
  render(
    <HardwareProvider>
      <SimulationTargetPanel isOpen onClose={() => {}} />
    </HardwareProvider>,
  );

describe('simulation target', () => {
  beforeEach(() => vi.clearAllMocks());

  it('offers the five most widely deployed chips', async () => {
    open();
    for (const name of ['H100-SXM', 'A100-SXM', 'L40S', 'RTX4090', 'T4']) {
      expect(await screen.findByText(name), `${name} should be offered`).toBeTruthy();
    }
  });

  it('shows each chip s specifications from the database, not from the UI', async () => {
    open();
    // H100: 80 GB, 989 TFLOP/s BF16, 3,352 GB/s — the database's own values.
    // 80 GB is shared with the A100, so scope to the H100's own card.
    const h100Card = (await screen.findByText('H100-SXM')).closest('button')!;
    expect(within(h100Card).getByText('80 GB')).toBeTruthy();
    expect(within(h100Card).getByText('989 TFLOP/s BF16')).toBeTruthy();
    // The thousands separator is a narrow no-break space, which the DOM
    // normalises; match on the digits instead of the exact glyph.
    expect(
      within(h100Card).getByText((text) => /^3\s?352 GB\/s$/.test(text)),
    ).toBeTruthy();
  });

  it('falls back to fp16 for parts that publish no bf16 figure', async () => {
    open();
    // The T4 reports 0 for bf16; showing "0 TFLOP/s" would be wrong.
    expect(await screen.findByText('65 TFLOP/s FP16')).toBeTruthy();
  });

  it('selecting a chip records it as the target', async () => {
    open();
    fireEvent.click(await screen.findByText('A100-SXM'));
    await waitFor(() =>
      expect(screen.getByText(/Analysing for A100-SXM/)).toBeTruthy(),
    );
  });

  it('keeps the rest of the catalogue reachable', async () => {
    open();
    const toggle = await screen.findByText(/Show the other 1 chips/);
    fireEvent.click(toggle);
    expect(await screen.findByText('V100')).toBeTruthy();
  });

  it('lets the device count be set', async () => {
    open();
    const input = (await screen.findByLabelText('Device count')) as HTMLInputElement;
    fireEvent.change(input, { target: { value: '8' } });
    await waitFor(() => expect(screen.getByText(/× 8/)).toBeTruthy());
  });
});
