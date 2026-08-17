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

    // The bootstrap runs in an effect; let it flush.
    await act(async () => {});

    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).not.toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).not.toBeNull();
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
});
