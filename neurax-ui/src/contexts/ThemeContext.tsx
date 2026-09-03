import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

// Exactly 4 — two identities, each in light and dark: the brand's own
// gold-yellow, and Signal & Static's instrument-panel graphite/phosphor-
// green. Previously as many as 16 — 10 were incomplete third-party ports
// (missing several semantic color slots, silently falling back to
// `.dark`'s values), 2 more (Molten Core, Web & Amber) were redundant with
// each other in tone, and a fifth, light-only accent (Light — Red) was
// tried and then dropped to keep the set at exactly 4. A visitor with any
// of those in localStorage falls back to 'dark' below.
export type Theme = 'light' | 'dark' | 'signal' | 'signal-light';

const THEMES: Theme[] = ['light', 'signal-light', 'dark', 'signal'];

// Which themes are dark-mode (get the `dark` class NEURAX's Tailwind config
// and shadcn components key off) vs light-mode.
const DARK_THEMES: Set<Theme> = new Set(['dark', 'signal']);

// The extra class (beyond `dark`/no-`dark`) each theme needs on <html> to
// pick up its own CSS block — 'light' and 'dark' need none, they're the
// unqualified `:root`/`.dark` blocks. Signal & Static's light and dark
// variants share one class name and differ only by whether `dark` is also
// present.
const THEME_CLASSES: Partial<Record<Theme, string>> = {
  signal: 'theme-signal',
  'signal-light': 'theme-signal',
};
const ALL_THEME_CLASSES = Array.from(new Set(Object.values(THEME_CLASSES)));

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('neurax-theme') as Theme;
      if (stored && THEMES.includes(stored)) return stored;
      return 'dark';
    }
    return 'dark';
  });

  useEffect(() => {
    const root = document.documentElement;

    for (const cls of ALL_THEME_CLASSES) {
      root.classList.remove(cls);
    }
    root.classList.toggle('dark', DARK_THEMES.has(theme));
    const named = THEME_CLASSES[theme];
    if (named) {
      root.classList.add(named);
    }

    localStorage.setItem('neurax-theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => {
      const idx = THEMES.indexOf(prev);
      const next = THEMES[(idx + 1) % THEMES.length];
      return next;
    });
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
