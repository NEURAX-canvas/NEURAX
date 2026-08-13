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
