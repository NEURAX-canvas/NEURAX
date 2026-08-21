import { createContext, useContext, useEffect, useState, ReactNode } from 'react';

// 'gruvbox' isn't a separate option here anymore — Light and Dark *are* the
// Gruvbox pair now (see index.css), not an alternative to it. A visitor with
// 'gruvbox' still in localStorage from before falls back to 'light' below,
// which today looks the same as what they had picked.
export type Theme = 'light' | 'dark' | 'nord' | 'onedark' | 'kanagawa' | 'catppuccin' | 'tokyonight' | 'everforest' | 'dracula' | 'nightfox' | 'rose-pine' | 'solarized-dark' | 'molten' | 'signal' | 'amber' | 'slate';

const THEMES: Theme[] = ['light', 'dark', 'nord', 'onedark', 'kanagawa', 'catppuccin', 'tokyonight', 'everforest', 'dracula', 'nightfox', 'rose-pine', 'solarized-dark', 'molten', 'signal', 'amber', 'slate'];
const VIM_THEMES: Theme[] = ['nord', 'onedark', 'kanagawa', 'catppuccin', 'tokyonight', 'everforest', 'dracula', 'nightfox', 'rose-pine', 'solarized-dark', 'molten', 'signal', 'amber', 'slate'];

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
      return 'light';
    }
    return 'light';
  });

  useEffect(() => {
    const root = document.documentElement;

    for (const t of VIM_THEMES) {
      root.classList.remove(`theme-${t}`);
    }

    if (theme === 'light') {
      root.classList.remove('dark');
    } else if (theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.add('dark');
      root.classList.add(`theme-${theme}`);
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
