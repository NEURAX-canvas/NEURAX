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
  light: 'Light — White & Blue',
  dark: 'Dark — Pink',
  molten: 'Molten Core',
  signal: 'Signal & Static',
  amber: 'Web & Amber',
  slate: 'Slate & Ember',
  nord: 'Nord',
  onedark: 'One Dark',
  kanagawa: 'Kanagawa',
  catppuccin: 'Catppuccin',
  tokyonight: 'Tokyo Night',
  everforest: 'Everforest',
  dracula: 'Dracula',
  nightfox: 'Nightfox',
  'rose-pine': 'Rosé Pine',
  'solarized-dark': 'Solarized Dark',
};

// Exactly the 6 NEURAX palettes — every semantic color slot filled
// explicitly for each (see index.css). The other themes index.css defines
// (nord, onedark, kanagawa, catppuccin, tokyonight, everforest, dracula,
// nightfox, rose-pine, solarized-dark) are left out of this list on
// purpose: several only define background/card/primary/border/sidebar/
// canvas and fall back to `.dark`'s destructive/warning/success/info/
// chart-* otherwise — a real inconsistency once actually selected, and
// not one this change is fixing.
const ORDERED_THEMES: Theme[] = ['light', 'dark', 'molten', 'signal', 'amber', 'slate'];

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
