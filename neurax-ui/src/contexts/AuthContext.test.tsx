// NEURAX's account used to be a Supabase project — a real one is required
// for any of this to work, and the one checked into this repository's own
// .env didn't resolve at all (verified live: no DNS record for its project
// ref), so account creation failed for every user by default. This is now a
// local profile instead: nothing here can be unreachable, because nothing
// here is a network call.
import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { AuthProvider, useAuth } from './AuthContext';

const STORAGE_KEY = 'neurax_demo_user';

describe('AuthContext — local account', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('creates a profile automatically, with no sign-up step', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).not.toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).not.toBeNull();
  });

  it('is authenticated on the very first render, not one tick later', () => {
    // ProtectedRoute reads isAuthenticated during the first render — before
    // any effect runs. The bootstrap used to live in a useEffect, so a
    // returning visitor opening /app directly (a bookmark, a refresh) was
    // bounced to `/` by <Navigate> before their already-stored profile had
    // a chance to load, even though it was sitting in localStorage the
    // whole time. No `act(async ...)` here on purpose: this asserts what
    // the very first synchronous render sees.
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        id: 'demo-existing', email: 'returning@example.com', username: 'Returning',
        avatarSeed: 'avatar-3', plan: 'elite', createdAt: new Date().toISOString(),
      }),
    );

    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user?.username).toBe('Returning');
  });

  it('reuses the stored profile across a remount, rather than minting a new one', async () => {
    const first = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});
    const firstId = first.result.current.user?.id;

    const second = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});

    expect(second.result.current.user?.id).toBe(firstId);
  });

  it('signIn replaces the profile with the given name', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});

    act(() => {
      result.current.signIn('explorer@example.com', 'Ada');
    });

    expect(result.current.user?.username).toBe('Ada');
    expect(result.current.user?.email).toBe('explorer@example.com');
  });

  it('signOut clears the profile, both in memory and in storage', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});

    act(() => {
      result.current.signOut();
    });

    expect(result.current.isAuthenticated).toBe(false);
    expect(result.current.user).toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it('updateProfile patches the existing profile without resetting its id or avatar', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});
    const before = result.current.user!;

    act(() => {
      result.current.updateProfile({ username: 'Renamed' });
    });

    expect(result.current.user?.username).toBe('Renamed');
    expect(result.current.user?.id).toBe(before.id);
    expect(result.current.user?.avatarSeed).toBe(before.avatarSeed);
  });

  it('persists an updateProfile patch to storage, so a reload keeps it', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});

    act(() => {
      result.current.updateProfile({ username: 'Persisted' });
    });

    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!);
    expect(stored.username).toBe('Persisted');
  });

  // Changing avatar used to have no effect on anything reading `avatarUrl` —
  // `updateProfile`'s own type didn't accept an avatar field, so the
  // "Save" button on the Account page wrote somewhere the header never
  // looked. This is the actual property that was broken: changing the
  // avatar changes what `avatarUrl` — the value the header renders — is.
  it('updateProfile can change the avatar, and avatarUrl reflects it immediately', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});
    const before = result.current.avatarUrl;

    act(() => {
      result.current.updateProfile({ avatarSeed: 'neurax-vertex' });
    });

    expect(result.current.user?.avatarSeed).toBe('neurax-vertex');
    expect(result.current.avatarUrl).not.toBe(before);
  });

  it('persists a changed avatar to storage, so a reload keeps it too', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});

    act(() => {
      result.current.updateProfile({ avatarSeed: 'neurax-photon' });
    });

    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!);
    expect(stored.avatarSeed).toBe('neurax-photon');
  });

  // The avatar picked at sign-up used to be discarded — `signIn` always drew
  // a random seed of its own regardless of what was passed through from the
  // picker.
  it('signIn honours the avatar it is given, instead of always randomising one', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});

    act(() => {
      result.current.signIn('new@example.com', 'New', 'neurax-cortex');
    });

    expect(result.current.user?.avatarSeed).toBe('neurax-cortex');
  });

  it('signIn still picks a random avatar when none is given, same as before', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper: AuthProvider });
    await act(async () => {});

    act(() => {
      result.current.signIn('new@example.com', 'New');
    });

    expect(result.current.user?.avatarSeed).toBeTruthy();
  });
});
