/**
 * Where the UI is running, and what that lets it do.
 *
 * NEURAX ships one frontend to two hosts. In the browser it talks to a
 * deployed `neurax-service` whose address is baked in at build time. In the
 * desktop app the very same bundle is loaded from disk by Tauri, which runs the
 * service in its own process on a loopback port chosen by the OS at launch —
 * so there is no address to bake in, and it is handed over at runtime instead
 * through a frozen global set before any page code executes.
 *
 * Everything host-specific is behind this module. No component should test for
 * Tauri itself, and nothing here imports `@tauri-apps/api` at module scope: the
 * web bundle must not carry it, so the native paths load it dynamically and
 * only when they are actually taken.
 */

/** Injected by `neurax-desktop` before the page loads. Absent in a browser. */
interface DesktopBridge {
  /** Base URL of the in-process API, e.g. `http://127.0.0.1:41234`. */
  readonly apiBase: string;
  /** `linux`, `macos` or `windows`. */
  readonly platform?: string;
  /**
   * Whether the application must draw its own window controls, because the
   * platform's window manager draws none.
   */
  readonly ownTitleBar?: boolean;
}

declare global {
  interface Window {
    __NEURAX_DESKTOP__?: DesktopBridge;
  }
}

function bridge(): DesktopBridge | undefined {
  if (typeof window === 'undefined') return undefined;
  const injected = window.__NEURAX_DESKTOP__;
  // Guard the shape rather than mere presence: a malformed injection should
  // fall back to the web behaviour, not produce `undefined` in every URL.
  if (!injected || typeof injected.apiBase !== 'string' || !injected.apiBase) {
    return undefined;
  }
  return injected;
}

/** True when running inside the desktop application. */
export function isDesktop(): boolean {
  return bridge() !== undefined;
}

/**
 * The API base URL, with the desktop's runtime value taking precedence over
 * the build-time one.
 *
 * `fallback` is the browser's value, passed in rather than read here so this
 * module stays free of build-time environment access and is testable.
 */
export function resolveApiBase(fallback: string): string {
  const injected = bridge()?.apiBase;
  return stripTrailingSlash(injected ?? fallback);
}

function stripTrailingSlash(url: string): string {
  return url.replace(/\/+$/, '');
}

/** Outcome of a save. `path` is only ever known on desktop. */
export interface SaveResult {
  saved: boolean;
  path?: string;
}

/**
 * Write text to a file the user names.
 *
 * On the desktop this opens the system save dialog and returns the real path,
 * so the UI can tell the user where the file went. In a browser the best
 * available behaviour is handing a blob to the download manager, which chooses
 * the directory itself and reports nothing back.
 */
export async function saveTextFile(
  contents: string,
  filename: string,
  mimeType = 'application/json',
): Promise<SaveResult> {
  if (isDesktop()) {
    const { invoke } = await import('@tauri-apps/api/core');
    const path = await invoke<string | null>('save_text_file', {
      defaultName: filename,
      contents,
    });
    return path ? { saved: true, path } : { saved: false };
  }

  const url = URL.createObjectURL(new Blob([contents], { type: mimeType }));
  try {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
  } finally {
    URL.revokeObjectURL(url);
  }
  return { saved: true };
}

/** A file the user chose to open. */
export interface OpenedFile {
  name: string;
  contents: string;
}

/**
 * Read a file the user picks.
 *
 * Returns `null` when the user dismisses the picker — a cancellation, not a
 * failure, so callers should stay silent rather than reporting an error.
 */
export async function openTextFile(extensions: string[]): Promise<OpenedFile | null> {
  if (isDesktop()) {
    const { invoke } = await import('@tauri-apps/api/core');
    const picked = await invoke<[string, string] | null>('open_text_file', { extensions });
    return picked ? { name: picked[0], contents: picked[1] } : null;
  }

  return new Promise((resolve) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = extensions.map((ext) => `.${ext}`).join(',');
    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) {
        resolve(null);
        return;
      }
      file
        .text()
        .then((contents) => resolve({ name: file.name, contents }))
        .catch(() => resolve(null));
    };
    // A dismissed browser file input fires no event at all, so the promise
    // simply never settles — matching the desktop's "nothing happened".
    input.click();
  });
}

/** What the window frame looks like on this host. */
export interface DesktopChrome {
  /** True when the application draws its own title bar and window buttons. */
  ownTitleBar: boolean;
  /** `linux`, `macos`, `windows`, or `web` in a browser. */
  platform: string;
}

/**
 * Whether this host expects the application to supply window controls.
 *
 * In a browser the answer is always no — the browser is the frame. On the
 * desktop it is the host's call, decided in Rust where the platform is known;
 * see `OWN_TITLE_BAR` in `neurax-desktop/src/main.rs`.
 */
export function desktopChrome(): DesktopChrome {
  const injected = bridge();
  return {
    ownTitleBar: injected?.ownTitleBar === true,
    platform: injected?.platform ?? 'web',
  };
}
