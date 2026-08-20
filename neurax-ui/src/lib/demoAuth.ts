/**
 * The local profile NEURAX's account is built on — see AuthContext.tsx for
 * why there's no cloud identity behind it. "Demo" in the naming here is a
 * holdover from when this was a fallback for an unconfigured cloud project;
 * it's the one account system now, not a stand-in for a missing one.
 */

import { identiconDataUri } from '@/components/profile/Identicon.tsx';

const DEMO_USER_KEY = 'neurax_demo_user';

export interface DemoUser {
  id: string;
  email: string;
  username: string;
  avatarSeed: string;
  plan: 'free' | 'essential' | 'architect' | 'elite';
  createdAt: string;
}

const AVATAR_SEEDS = [
  'avatar-1', 'avatar-2', 'avatar-3', 'avatar-4', 'avatar-5',
  'avatar-6', 'avatar-7', 'avatar-8', 'avatar-9', 'avatar-10',
  'avatar-11', 'avatar-12', 'avatar-13', 'avatar-14', 'avatar-15',
];

export function randomAvatarSeed(): string {
  return AVATAR_SEEDS[Math.floor(Math.random() * AVATAR_SEEDS.length)];
}

export function demoAvatarUrl(seed: string): string {
  return identiconDataUri(seed);
}

export function getStoredDemoUser(): DemoUser | null {
  try {
    const raw = localStorage.getItem(DEMO_USER_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as DemoUser;
  } catch {
    return null;
  }
}

export function saveDemoUser(user: DemoUser): void {
  localStorage.setItem(DEMO_USER_KEY, JSON.stringify(user));
}

export function clearDemoUser(): void {
  localStorage.removeItem(DEMO_USER_KEY);
}

export function createDemoUser(email: string, username: string, avatarSeed?: string): DemoUser {
  const user: DemoUser = {
    id: `demo-${Date.now()}`,
    email,
    username: username || email.split('@')[0] || 'Explorer',
    avatarSeed: avatarSeed || randomAvatarSeed(),
    plan: 'elite',
    createdAt: new Date().toISOString(),
  };
  saveDemoUser(user);
  return user;
}

export function isDemoLoggedIn(): boolean {
  return getStoredDemoUser() !== null;
}
