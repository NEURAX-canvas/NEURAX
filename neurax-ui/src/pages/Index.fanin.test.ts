/**
 * `isFanInCapable`'s hardcoded list must match the file it claims to be
 * synchronized with.
 *
 * Regression coverage for a real, already-live drift: this list and two
 * separate Python-side lists (`constants.py`'s `MERGE_BLOCK_TYPES` and
 * `catalogue_store.py`'s own set) had each diverged from the others —
 * `unet_block` was missing from one, `output_combination` from another.
 * `block_constraints.json`'s `merge_capable_types` is the file the Python
 * side's own consistency test (`test_catalogue_consistency.py`) now checks
 * both of those against; this test checks the frontend's copy the same way,
 * against the same file, rather than trusting a comment.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');

const constraints = JSON.parse(
  readFileSync(join(repoRoot, 'neurax-agent', 'block_constraints.json'), 'utf8'),
);

const indexSource = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'Index.tsx'), 'utf8');

function extractIsFanInCapableList(source: string): string[] {
  const match = source.match(/const isFanInCapable = \[([\s\S]*?)\]\.includes/);
  if (!match) {
    throw new Error("Could not find `const isFanInCapable = [...]` in Index.tsx — did it move or get renamed?");
  }
  return Array.from(match[1].matchAll(/'([^']+)'/g)).map((m) => m[1]);
}

describe('fan-in capable block list', () => {
  it('matches block_constraints.json exactly', () => {
    const fromIndex = extractIsFanInCapableList(indexSource).sort();
    const fromConstraints = [...constraints.merge_capable_types].sort();
    expect(fromIndex).toEqual(fromConstraints);
  });
});
