/**
 * The web and desktop applications must stay the same application.
 *
 * They share this codebase, which is what makes them identical today — but
 * nothing stops someone from writing `if (isDesktop())` around a panel and
 * quietly forking the interface. These tests fail when that happens.
 *
 * The rule they encode: host detection may decide *how* something is done
 * (which file dialog, which entry route), never *what exists*. Layout,
 * panels, wording and controls are the same on both.
 */

import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const SRC = join(process.cwd(), 'src');

/**
 * Files allowed to ask which host they are on, and why.
 *
 * Adding to this list is a deliberate act. If a new entry is a component that
 * renders differently per host, it does not belong here — the interfaces are
 * supposed to be indistinguishable.
 */
const ALLOWED_HOST_CHECKS: Record<string, string> = {
  'services/desktopRuntime.ts': 'defines the host detection itself',
  'services/desktopRuntime.test.ts': 'tests it',
  'services/hostParity.test.ts': 'this file',
  'services/neuraxApi.ts': 'resolves the API base address',
  'components/panels/ExportPanel.tsx': 'native save dialog instead of a browser download',
  'App.tsx': 'skips the landing page once a desktop profile exists',
  'components/desktop/TitleBar.tsx':
    'window chrome, not application interface — drawn only where the platform draws none',
  'services/compilerErrors.ts':
    'wording only — the same failures are reported on both hosts, but "check your ' +
    'connection" is wrong advice when the compiler is a thread in this process',
  'services/compilerErrors.test.ts': 'tests that wording',
};

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) return sourceFiles(full);
    return /\.(ts|tsx)$/.test(entry) ? [full] : [];
  });
}

describe('web and desktop stay one interface', () => {
  const files = sourceFiles(SRC);

  it('finds the frontend sources', () => {
    expect(files.length).toBeGreaterThan(50);
  });

  it('only checks the host where it is sanctioned to', () => {
    const offenders = files
      .filter((file) => {
        const body = readFileSync(file, 'utf8');
        return /\bisDesktop\s*\(/.test(body) || /__NEURAX_DESKTOP__/.test(body);
      })
      .map((file) => relative(SRC, file).replace(/\\/g, '/'))
      .filter((file) => !(file in ALLOWED_HOST_CHECKS));

    expect(
      offenders,
      `These files branch on the host. The desktop and web interfaces must be ` +
        `the same; if this branch is genuinely about *how* something is done ` +
        `rather than *what exists*, add it to ALLOWED_HOST_CHECKS with a reason.`,
    ).toEqual([]);
  });

  it('has no desktop-only component files', () => {
    const desktopOnly = files
      .map((file) => relative(SRC, file).replace(/\\/g, '/'))
      .filter(
        (file) =>
          /(^|\/)(desktop|tauri)[A-Z-]/.test(file) &&
          !file.startsWith('services/') &&
          // The window frame is allowed to be desktop-only; it is not part of
          // the application's interface.
          file !== 'components/desktop/TitleBar.tsx',
      );

    expect(
      desktopOnly,
      'A component that exists only on desktop is a second interface.',
    ).toEqual([]);
  });

  /**
   * `@tauri-apps/api` must never be imported at module scope: a static import
   * would put it in the web bundle's main chunk, where it can only fail.
   */
  it('never imports the Tauri API statically', () => {
    const staticImports = files
      .filter((file) => /^\s*import\s[^\n]*['"]@tauri-apps\//m.test(readFileSync(file, 'utf8')))
      .map((file) => relative(SRC, file).replace(/\\/g, '/'));

    expect(staticImports, 'Use `await import(...)` inside a desktop-only branch.').toEqual([]);
  });
});
