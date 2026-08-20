/**
 * Reference templates must reproduce their published parameter counts.
 *
 * A template that loads at the wrong width is worse than no template: it looks
 * like a known model and reports numbers for a different one. Three separate
 * defects did exactly that, and none of them was visible without loading a
 * template and comparing the result to the paper:
 *
 *  - templates state `hidden_size` while block schemas use `d_model`, so the
 *    width was ignored and every model loaded at the 768 default;
 *  - a gated feed-forward block collapses to `mlp` for the compiler, which then
 *    counted three weight matrices as two — 1.5B short on a 7B model;
 *  - templates do not repeat the width on every block, so blocks that omitted
 *    it silently shrank to the schema default.
 */
import { describe, it, expect } from 'vitest';
import { MODEL_TEMPLATES } from './modelTemplates';
import { getBlockDefaults, normalizeBlockParams } from '@/utils/blockDefaults';

/** Published parameter counts, and the tolerance a fair accounting allows. */
const PUBLISHED: Record<string, { params: number; tolerance: number }> = {
  'tpl-bert-base': { params: 110e6, tolerance: 0.05 },
  'tpl-gpt2-xl': { params: 1.56e9, tolerance: 0.05 },
  'tpl-llama2-7b': { params: 6.74e9, tolerance: 0.05 },
};

/** Width a block ends up with after hydration, mirroring the page's logic. */
function widthOf(nodes: any[]): number | undefined {
  const stated = nodes
    .map((n) => {
      const p = n.params ?? {};
      const v = p.d_model ?? p.hidden_size ?? p.embedding_dim;
      return typeof v === 'number' && v > 0 ? v : null;
    })
    .filter((v): v is number => v !== null);
  if (!stated.length) return undefined;
  const counts = new Map<number, number>();
  for (const v of stated) counts.set(v, (counts.get(v) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0];
}

describe('reference templates', () => {
  it('state a width the alias layer can resolve', () => {
    for (const id of Object.keys(PUBLISHED)) {
      const tpl = MODEL_TEMPLATES.find((t) => t.id === id);
      expect(tpl, `${id} should exist`).toBeTruthy();
      expect(widthOf(tpl!.nodes as any[]), `${id} states no width`).toBeGreaterThan(0);
    }
  });

  it('resolve HuggingFace-style names onto the block schema', () => {
    // `hidden_size: 4096` must become the block's own `d_model`.
    const normalized = normalizeBlockParams('token_embedding', {
      vocab_size: 32000,
      hidden_size: 4096,
    });
    expect(normalized.d_model).toBe(4096);

    const attention = normalizeBlockParams('gqa_attention', {
      hidden_size: 4096,
      num_heads: 32,
      num_kv_heads: 8,
    });
    expect(attention.d_model).toBe(4096);
    expect(attention.n_heads).toBe(32);
    expect(attention.n_kv_heads).toBe(8);
  });

  it('never overwrite a value already given under the canonical name', () => {
    const p = normalizeBlockParams('token_embedding', { d_model: 1024, hidden_size: 4096 });
    expect(p.d_model, 'the canonical name wins').toBe(1024);
  });

  it('leave a block its own parameter when the alias is a real key', () => {
    // An LSTM genuinely takes `hidden_size`; it is not a synonym there.
    const schema = getBlockDefaults('lstm_cell');
    expect('hidden_size' in schema).toBe(true);
    const p = normalizeBlockParams('lstm_cell', { hidden_size: 512 });
    expect(p.hidden_size).toBe(512);
  });

  it('keep user-defined parameters rather than dropping them', () => {
    const p = normalizeBlockParams('token_embedding', { my_own_knob: 7 });
    expect(p.my_own_knob).toBe(7);
  });
});

/**
 * The SSM (Mamba/Mamba-2) templates used to decompose one block into
 * ssm_in_proj/causal_conv1d/s6_block/ssm_out_proj — visually descriptive,
 * numerically wrong: the compiler's layer-type resolution gave two of those
 * nodes the *whole* block's formula each (double-counted) and the actual
 * selective-scan step none of its own. Rebuilt as one `mamba_block` node per
 * layer, carrying `neurax_formulas::ssm::mamba_params`'s real formula —
 * reimplemented independently here (not imported from either the graph or
 * the Rust formula) so a bug shared between them still has a chance of
 * being caught.
 */
describe('SSM reference templates', () => {
  const SSM_TEMPLATE_IDS = [
    'tpl-mamba-130m', 'tpl-mamba-370m', 'tpl-mamba-790m', 'tpl-mamba-1.4b', 'tpl-mamba-2.8b',
    'tpl-mamba2-130m', 'tpl-mamba2-2.7b',
  ];

  function mambaParamCount(nodes: any[]): number {
    const embedding = nodes.find((n) => n.type === 'token_embedding');
    const width = embedding.params.hidden_size;
    const vocab = embedding.params.vocab_size;

    const block = nodes.find((n) => n.type === 'mamba_block');
    const dInner = width * block.params.expansion_factor;
    const perBlock =
      width * (dInner * 2) + // in_proj
      dInner * 4 + // conv1d
      (dInner * block.params.state_dim * 3 + dInner) + // A, B, C, D
      dInner * width; // out_proj

    const stack = nodes.find((n) => n.type === 'layer_stack');
    const layers = stack.params.num_layers;

    const head = nodes.find((n) => n.type === 'lm_head');
    const untiedHead = head.params.tie_weights !== true ? vocab * width : 0;

    return vocab * width + layers * (perBlock + width) + width + untiedHead;
  }

  it('carry exactly one parameter-bearing SSM node per layer, not the old double-counted three', () => {
    for (const id of SSM_TEMPLATE_IDS) {
      const tpl = MODEL_TEMPLATES.find((t) => t.id === id);
      expect(tpl, `${id} should exist`).toBeTruthy();
      const mambaNodes = (tpl!.nodes as any[]).filter((n) => n.type === 'mamba_block');
      const staleNodes = (tpl!.nodes as any[]).filter((n) =>
        ['s6_block', 'ssd_block', 'ssm_in_proj', 'ssm_out_proj', 'state_space'].includes(n.type),
      );
      expect(mambaNodes, `${id} should have exactly one mamba_block`).toHaveLength(1);
      expect(staleNodes, `${id} should have no leftover decomposed SSM nodes`).toHaveLength(0);
    }
  });

  it('Mamba-2.8B reproduces its published parameter count (±10%, matching neurax-core/tests/published_model_accuracy.rs)', () => {
    const tpl = MODEL_TEMPLATES.find((t) => t.id === 'tpl-mamba-2.8b')!;
    const counted = mambaParamCount(tpl.nodes as any[]);
    const published = 2.8e9;
    const error = Math.abs(counted - published) / published;
    expect(
      error,
      `graph implies ${(counted / 1e9).toFixed(3)} B, published ${(published / 1e9).toFixed(1)} B`,
    ).toBeLessThan(0.1);
  });

  it('Mamba-130M states its real 24-layer depth, not the previous (wrong) 12', () => {
    const tpl = MODEL_TEMPLATES.find((t) => t.id === 'tpl-mamba-130m')!;
    const stack = (tpl.nodes as any[]).find((n) => n.type === 'layer_stack');
    expect(stack.params.num_layers).toBe(24);
  });
});
