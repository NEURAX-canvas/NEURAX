/**
 * The landing page's numbers must be checkable against the repository.
 *
 * A wrong figure here is a claim, not a bug: the previous copy advertised 680+
 * blocks against a catalogue of 208, with a per-family breakdown in the FAQ that
 * matched nothing.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CATALOGUE, BLOCK_COUNT, FAMILY_COUNT, HERO_STATS, IR_PASS_COUNT } from './projectFacts';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const catalogue = JSON.parse(
  readFileSync(join(repoRoot, 'neurax-agent', 'catalogue.json'), 'utf8'),
);

describe('project facts', () => {
  it('lists exactly the families the catalogue defines', () => {
    expect(Object.keys(CATALOGUE).sort()).toEqual(Object.keys(catalogue).sort());
    expect(FAMILY_COUNT).toBe(Object.keys(catalogue).length);
  });

  it('counts each family s blocks as the catalogue does', () => {
    for (const [family, counts] of Object.entries(CATALOGUE)) {
      expect(counts.blocks, `${family} blocks`).toBe(catalogue[family].blocks?.length ?? 0);
      expect(counts.macroBlocks, `${family} macro-blocks`).toBe(
        catalogue[family].macroBlocks?.length ?? 0,
      );
    }
  });

  it('totals the blocks it advertises', () => {
    const expected = Object.values(catalogue).reduce(
      (total: number, family: any) =>
        total + (family.blocks?.length ?? 0) + (family.macroBlocks?.length ?? 0),
      0,
    );
    expect(BLOCK_COUNT).toBe(expected);
  });

  it('counts the IR passes the pipeline actually runs', () => {
    const source = readFileSync(
      join(repoRoot, 'neurax-core', 'src', 'lib.rs'),
      'utf8',
    );
    // Each pass appears as a `<Name>Pass` type in the analysis pipeline.
    const passes = new Set(source.match(/\b(\w+)Pass\b/g) ?? []);
    expect(passes.size).toBeGreaterThanOrEqual(IR_PASS_COUNT);
  });

  it('gives every hero stat a value, a label and an explanation', () => {
    expect(HERO_STATS).toHaveLength(4);
    for (const stat of HERO_STATS) {
      expect(stat.value).toBeTruthy();
      expect(stat.label).toBeTruthy();
      expect(stat.detail.length).toBeGreaterThan(20);
    }
  });
});
