import React, { createContext, useContext, useEffect, useMemo, useState, useCallback } from 'react';
import type { Session } from '@supabase/supabase-js';
import { supabase } from '@/lib/supabaseClient.ts';
import {
  getStoredDemoUser,
  clearDemoUser,
  createDemoUser,
  demoAvatarUrl,
  type DemoUser,
} from '@/lib/demoAuth.ts';

const SUPABASE_DISABLED = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

interface AuthContextType {
  session: Session | null;
  accessToken: string | null;
  isAuthenticated: boolean;
  // Demo auth helpers
  demoUser: DemoUser | null;
  demoSignIn: (email: string, username?: string) => void;
  demoSignOut: () => void;
  demoAvatarUrl: string;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

function buildDemoSession(user: DemoUser): Session {
  return {
    access_token: 'dev-token',
    token_type: 'bearer',
    expires_in: 60 * 60 * 24 * 365,
    expires_at: Math.floor(Date.now() / 1000) + 60 * 60 * 24 * 365,
    refresh_token: 'dev-refresh',
    user: {
      id: user.id,
      aud: 'authenticated',
      role: 'authenticated',
      email: user.email,
      app_metadata: {},
      user_metadata: {
        username: user.username,
        avatar_url: demoAvatarUrl(user.avatarSeed),
      },
      created_at: user.createdAt,
    },
  } as unknown as Session;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [demoUser, setDemoUser] = useState<DemoUser | null>(null);

  // On mount: check for existing demo user or Supabase session
  useEffect(() => {
    if (SUPABASE_DISABLED) {
      // No authentication backend means there is nobody to authenticate
      // against — this is the desktop application, or a local frontend with no
      // Supabase project. Making the user invent an account before they can
      // open the studio would be ceremony protecting nothing, so a local
      // profile is created on first launch and the app opens ready to work.
      // It is an ordinary profile: the name and avatar are editable in Account
      // exactly as they are on the web.
      const user = getStoredDemoUser() ?? createDemoUser('local@neurax', 'Explorer');
      setDemoUser(user);
      setSession(buildDemoSession(user));
      return;
    }

    let mounted = true;
    supabase.auth.getSession().then(({ data }) => {
      if (!mounted) return;
      setSession(data.session ?? null);
    });

    const { data: sub } = supabase.auth.onAuthStateChange((_event, next) => {
      if (!mounted) return;
      setSession(next);
    });

    return () => {
      mounted = false;
      sub.subscription.unsubscribe();
    };
  }, []);

  const demoSignIn = useCallback((email: string, username?: string) => {
    const user = createDemoUser(email, username || email.split('@')[0] || 'Explorer');
    setDemoUser(user);
    setSession(buildDemoSession(user));
  }, []);

  const demoSignOut = useCallback(() => {
    clearDemoUser();
    setDemoUser(null);
    setSession(null);
  }, []);

  const demoAvatar = useMemo(() => {
    if (!demoUser) return '';
    return demoAvatarUrl(demoUser.avatarSeed);
  }, [demoUser]);

  const value = useMemo<AuthContextType>(() => {
    const accessToken = session?.access_token ?? null;
    return {
      session,
      accessToken,
      isAuthenticated: !!accessToken,
      demoUser,
      demoSignIn,
      demoSignOut,
      demoAvatarUrl: demoAvatar,
    };
  }, [session, demoUser, demoSignIn, demoSignOut, demoAvatar]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
