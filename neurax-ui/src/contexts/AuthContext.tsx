/**
 * NEURAX's account: a local profile, not a cloud identity.
 *
 * There is no NEURAX-operated identity server. An "account" is a name, an
 * avatar, and an id, generated on first launch and kept in this browser's
 * (or, in the desktop build, this machine's) local storage — the same
 * information a Supabase-backed account would have held, without a network
 * dependency, a service to keep available, or a place outside this machine
 * where it's ever written. This is not a fallback for when a cloud project
 * isn't configured; it's the whole system.
 *
 * A user is never asked to sign up before they can use the compiler — that
 * would be ceremony protecting nothing on a single-user local tool — so the
 * profile is created automatically on first launch and the app opens ready
 * to work. It's an ordinary profile: name and avatar are editable in
 * Account, exactly as they would be on any account system.
 */
import React, { createContext, useContext, useMemo, useState, useCallback } from 'react';
import {
  getStoredDemoUser,
  clearDemoUser,
  createDemoUser,
  saveDemoUser,
  demoAvatarUrl,
  type DemoUser,
} from '@/lib/demoAuth.ts';

interface AuthContextType {
  isAuthenticated: boolean;
  user: DemoUser | null;
  signIn: (email: string, username?: string, avatarSeed?: string) => void;
  signOut: () => void;
  /** Patch the current profile's editable fields (Account settings). */
  updateProfile: (patch: Partial<Pick<DemoUser, 'username' | 'email' | 'avatarSeed'>>) => void;
  avatarUrl: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  // Read (or create) the profile synchronously, in the initializer, not in
  // an effect: `ProtectedRoute` reads `isAuthenticated` on the very first
  // render, before any effect has run. Deferring the bootstrap to an effect
  // meant that first render always saw `isAuthenticated === false` — for a
  // fresh visitor that's invisible (they land on `/` anyway), but a
  // returning one whose profile is already in storage, opening `/app`
  // directly (a bookmark, a refresh), was bounced straight back to the
  // landing page by `<Navigate to="/" />` before the profile even loaded.
  const [user, setUser] = useState<DemoUser | null>(
    () => getStoredDemoUser() ?? createDemoUser('local@neurax', 'Explorer'),
  );

  const signIn = useCallback((email: string, username?: string, avatarSeed?: string) => {
    setUser(createDemoUser(email, username || email.split('@')[0] || 'Explorer', avatarSeed));
  }, []);

  const signOut = useCallback(() => {
    clearDemoUser();
    setUser(null);
  }, []);

  const updateProfile = useCallback((patch: Partial<Pick<DemoUser, 'username' | 'email' | 'avatarSeed'>>) => {
    setUser((prev) => {
      if (!prev) return prev;
      const next = { ...prev, ...patch };
      saveDemoUser(next);
      return next;
    });
  }, []);

  const avatarUrl = useMemo(() => (user ? demoAvatarUrl(user.avatarSeed) : ''), [user]);

  const value = useMemo<AuthContextType>(() => ({
    isAuthenticated: user !== null,
    user,
    signIn,
    signOut,
    updateProfile,
    avatarUrl,
  }), [user, signIn, signOut, updateProfile, avatarUrl]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
