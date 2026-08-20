/**
 * Changing avatar on the Account page used to show a "saved" toast and
 * change nothing anywhere else — the Save button wrote to a localStorage
 * key nothing displaying the avatar ever read, because `updateProfile`'s
 * own type didn't accept an avatar field at all. This renders the real
 * page, clicks a real avatar option, clicks the real Save button, and
 * checks the property that was actually broken: does `useAuth()` — the
 * same hook the header reads — see the new avatar afterward.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Account from './Account';
import { AuthProvider, useAuth } from '@/contexts/AuthContext.tsx';
import { ApiKeyProvider } from '@/contexts/ApiKeyContext.tsx';
import { Toaster } from '@/components/ui/toaster.tsx';

/** Renders next to Account, reading the same context AuthControl's header does. */
function AvatarProbe() {
  const { user, avatarUrl } = useAuth();
  return (
    <div data-testid="probe" data-avatar-seed={user?.avatarSeed} data-avatar-url={avatarUrl} />
  );
}

function renderAccount() {
  return render(
    <MemoryRouter>
      <AuthProvider>
        <ApiKeyProvider>
          <AvatarProbe />
          <Account />
          <Toaster />
        </ApiKeyProvider>
      </AuthProvider>
    </MemoryRouter>,
  );
}

describe('changing avatar on the Account page', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('reaches the profile everything else reads, not just this page', () => {
    renderAccount();
    const probeBefore = screen.getByTestId('probe');
    const seedBefore = probeBefore.dataset.avatarSeed;
    const urlBefore = probeBefore.dataset.avatarUrl;

    fireEvent.click(screen.getByRole('radio', { name: 'Select Photon' }));
    fireEvent.click(screen.getByRole('button', { name: /save avatar/i }));

    const probeAfter = screen.getByTestId('probe');
    expect(probeAfter.dataset.avatarSeed).toBe('neurax-photon');
    expect(probeAfter.dataset.avatarSeed).not.toBe(seedBefore);
    expect(probeAfter.dataset.avatarUrl).not.toBe(urlBefore);
  });

  it('confirms the save with a toast, same as before', () => {
    renderAccount();
    fireEvent.click(screen.getByRole('radio', { name: 'Select Cortex' }));
    fireEvent.click(screen.getByRole('button', { name: /save avatar/i }));
    expect(screen.getByText(/avatar saved/i)).toBeInTheDocument();
  });

  it("opens with the current profile's avatar selected, not a stale separate value", () => {
    localStorage.setItem(
      'neurax_demo_user',
      JSON.stringify({
        id: 'demo-1', email: 'a@b.com', username: 'A',
        avatarSeed: 'neurax-quanta', plan: 'elite', createdAt: new Date().toISOString(),
      }),
    );
    renderAccount();
    const selected = screen.getByRole('radio', { name: 'Select Quanta' });
    expect(selected).toHaveAttribute('aria-checked', 'true');
  });
});
