/**
 * `resolveAvatar` used to only match an avatar's `id` ('ax-07') — but a
 * profile's `avatarSeed` field stores the *seed* ('neurax-quanta'), a
 * different string for the same option. A caller resolving a profile's
 * current avatar back to a picker selection (Account's picker, opening
 * pre-selected to whatever the profile actually has) got the hash-fallback
 * instead of the real match, landing on an arbitrary option.
 */
import { describe, it, expect } from 'vitest';
import { AVATAR_OPTIONS, resolveAvatar } from './NotionistsAvatarPicker';

describe('resolveAvatar', () => {
  it('resolves by id, the picker\'s own identifier', () => {
    expect(resolveAvatar('ax-07').name).toBe('Quanta');
  });

  it('resolves by seed, what a profile\'s avatarSeed field actually stores', () => {
    expect(resolveAvatar('neurax-quanta').name).toBe('Quanta');
    expect(resolveAvatar('neurax-quanta').id).toBe('ax-07');
  });

  it('every option round-trips: id -> option -> same option via its seed', () => {
    for (const option of AVATAR_OPTIONS) {
      expect(resolveAvatar(option.seed)).toEqual(option);
    }
  });

  it('falls back to a stable hashed option for an unrecognised value, not the first one blindly', () => {
    const resolved = resolveAvatar('some-legacy-emoji-value');
    expect(AVATAR_OPTIONS).toContainEqual(resolved);
    // Same input, same output — "stable" means deterministic, not random.
    expect(resolveAvatar('some-legacy-emoji-value')).toEqual(resolved);
  });

  it('defaults to the first option when nothing is stored', () => {
    expect(resolveAvatar(null)).toEqual(AVATAR_OPTIONS[0]);
    expect(resolveAvatar(undefined)).toEqual(AVATAR_OPTIONS[0]);
  });
});
