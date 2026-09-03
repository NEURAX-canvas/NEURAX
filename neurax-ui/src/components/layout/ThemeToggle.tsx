import { Palette } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Theme, useTheme } from '@/contexts/ThemeContext.tsx';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip.tsx';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu.tsx';

const LABELS: Record<Theme, string> = {
  light: 'Light — Gold',
  dark: 'Dark — Gold',
  'signal-light': 'Signal & Static — Light',
  signal: 'Signal & Static — Dark',
};

// Exactly 4 — two identities (the brand's gold-yellow, and Signal &
// Static's instrument-panel graphite/phosphor-green) each in light and
// dark. Every semantic color slot is filled explicitly for each (see
// index.css); the 13 themes this list used to also carry (10 incomplete
// third-party ports, Molten Core and Web & Amber redundant with each other
// in tone, and a fifth accent, Light — Red, tried and dropped) are gone
// from index.css entirely, not just hidden here.
const ORDERED_THEMES: Theme[] = ['light', 'signal-light', 'dark', 'signal'];

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();

  const label = (t: Theme) => LABELS[t];

  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-foreground">
              <Palette className="w-4 h-4" />
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent>
          <p>Theme: {label(theme)}</p>
        </TooltipContent>
      </Tooltip>

      <DropdownMenuContent align="end" className="max-h-80 overflow-y-auto">
        {ORDERED_THEMES.map((t) => (
          <DropdownMenuItem key={t} onClick={() => setTheme(t)} className={t === theme ? 'font-semibold' : undefined}>
            {label(t)}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
