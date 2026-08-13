/**
 * Window controls for the desktop application.
 *
 * This is window chrome, not application interface: it is the frame the
 * operating system would otherwise draw, and it appears only where the
 * platform does not draw one. The studio inside it is byte-for-byte the same
 * as the web application's — nothing here changes what the app contains.
 *
 * The buttons sit at the top right, in minimise / maximise / close order,
 * because that is where every window on Windows and on the common Linux
 * desktops puts them. macOS puts its own on the left and draws them itself, so
 * this component never renders there.
 */

import { useCallback, useEffect, useState } from 'react';
import { Minus, Square, Copy, X } from 'lucide-react';
import { desktopChrome } from '@/services/desktopRuntime.ts';

/** Lazily loaded so the web bundle never carries the Tauri window API. */
async function currentWindow() {
  const { getCurrentWindow } = await import('@tauri-apps/api/window');
  return getCurrentWindow();
}

export function TitleBar() {
  const [maximized, setMaximized] = useState(false);

  useEffect(() => {
    let cancelled = false;
    currentWindow()
      .then((w) => w.isMaximized())
      .then((value) => !cancelled && setMaximized(value))
      .catch(() => {
        /* Not in a desktop window; the component will not be rendered anyway. */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const minimize = useCallback(async () => {
    (await currentWindow()).minimize();
  }, []);

  const toggleMaximize = useCallback(async () => {
    const w = await currentWindow();
    await w.toggleMaximize();
    setMaximized(await w.isMaximized());
  }, []);

  const close = useCallback(async () => {
    // `close()` rather than `destroy()`: it raises the close request the Rust
    // side listens for, which is what writes unsaved projects to disk.
    (await currentWindow()).close();
  }, []);

  return (
    <div
      // Dragging anywhere on the bar moves the window, as a real title bar does.
      data-tauri-drag-region
      className="flex h-8 shrink-0 select-none items-center justify-between border-b border-border bg-card px-3"
    >
      <span
        data-tauri-drag-region
        className="pointer-events-none text-xs font-medium tracking-wide text-muted-foreground"
      >
        NEURAX
      </span>

      <div className="flex items-center">
        <TitleBarButton label="Minimize" onClick={minimize}>
          <Minus className="h-3.5 w-3.5" />
        </TitleBarButton>

        <TitleBarButton
          label={maximized ? 'Restore' : 'Maximize'}
          onClick={toggleMaximize}
        >
          {maximized ? <Copy className="h-3 w-3" /> : <Square className="h-3 w-3" />}
        </TitleBarButton>

        <TitleBarButton label="Close" onClick={close} destructive>
          <X className="h-3.5 w-3.5" />
        </TitleBarButton>
      </div>
    </div>
  );
}

function TitleBarButton({
  label,
  onClick,
  children,
  destructive = false,
}: {
  label: string;
  onClick: () => void;
  children: React.ReactNode;
  destructive?: boolean;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className={`flex h-8 w-11 items-center justify-center text-muted-foreground transition-colors ${
        destructive
          ? 'hover:bg-destructive hover:text-destructive-foreground'
          : 'hover:bg-secondary hover:text-foreground'
      }`}
    >
      {children}
    </button>
  );
}

/** Renders the title bar only where the platform does not provide one. */
export function DesktopChrome() {
  return desktopChrome().ownTitleBar ? <TitleBar /> : null;
}
