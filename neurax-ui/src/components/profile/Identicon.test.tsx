/**
 * An identicon is an identity, so the only property that really matters is that
 * the same seed always draws the same thing — and that different seeds do not.
 */
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { Identicon, identiconCells, identiconColor } from './Identicon';
import { AVATAR_OPTIONS, resolveAvatar } from './NotionistsAvatarPicker';

describe('identicon', () => {
  it('offers more than thirty avatars', () => {
    expect(AVATAR_OPTIONS.length).toBeGreaterThan(30);
    expect(new Set(AVATAR_OPTIONS.map((a) => a.id)).size).toBe(AVATAR_OPTIONS.length);
    expect(new Set(AVATAR_OPTIONS.map((a) => a.name)).size).toBe(AVATAR_OPTIONS.length);
  });

  it('is deterministic for a seed', () => {
    expect(identiconCells('alice')).toEqual(identiconCells('alice'));
    expect(identiconColor('alice')).toBe(identiconColor('alice'));
  });

  it('differs between seeds, including neighbouring ones', () => {
    expect(identiconCells('user1')).not.toEqual(identiconCells('user2'));
    // Collisions matter more as the set grows: two avatars that draw the same
    // pattern are the same avatar as far as a user is concerned.
    const patterns = new Set(
      AVATAR_OPTIONS.map((a) => identiconCells(a.seed).map(Number).join('')),
    );
    expect(patterns.size, 'every offered avatar should look different').toBe(
      AVATAR_OPTIONS.length,
    );
    // Colour is the other half of the identity; near-identical hues on similar
    // patterns would read as duplicates too.
    expect(new Set(AVATAR_OPTIONS.map((a) => identiconColor(a.seed))).size)
      .toBeGreaterThanOrEqual(AVATAR_OPTIONS.length - 2);
  });

  it('is left-right symmetric, which is what makes it read as a face', () => {
    const cells = identiconCells('symmetry-check');
    for (let row = 0; row < 5; row++) {
      for (let col = 0; col < 5; col++) {
        expect(cells[row * 5 + col]).toBe(cells[row * 5 + (4 - col)]);
      }
    }
  });

  it('never draws a near-empty grid', () => {
    for (const seed of ['a', 'b', 'c', '', 'x', '1', 'zzz']) {
      expect(identiconCells(seed).filter(Boolean).length).toBeGreaterThanOrEqual(5);
    }
  });

  it('renders inline SVG with no network request', () => {
    const { container } = render(<Identicon seed="render-test" size={40} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeTruthy();
    expect(container.querySelector('img')).toBeNull();
    expect(svg!.querySelectorAll('rect').length).toBeGreaterThan(1);
  });

  it('carries an accessible name', () => {
    const { container } = render(<Identicon seed="a11y" />);
    expect(container.querySelector('svg')!.getAttribute('aria-label')).toContain('a11y');
  });
});

describe('avatar migration', () => {
  it('keeps a profile saved by the old emoji picker on a stable avatar', () => {
    // The previous picker stored an emoji; it must map somewhere consistent
    // rather than silently resetting everyone to the first option.
    expect(resolveAvatar('🚀').id).toBe(resolveAvatar('🚀').id);
    expect(AVATAR_OPTIONS.map((a) => a.id)).toContain(resolveAvatar('🚀').id);
  });

  it('resolves a known id to itself, and nothing to the first option', () => {
    expect(resolveAvatar('ax-05').id).toBe('ax-05');
    expect(resolveAvatar(null).id).toBe(AVATAR_OPTIONS[0].id);
  });
});
