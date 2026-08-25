import { ArchitectureFamily, ARCHITECTURE_FAMILIES } from '@/types/plugins.ts';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
} from '@/components/ui/select.tsx';
import { cn } from '@/lib/utils.ts';

interface ArchitectureSelectorProps {
  value: ArchitectureFamily;
  onChange: (value: ArchitectureFamily) => void;
  className?: string;
}

export function ArchitectureSelector({ value, onChange, className }: ArchitectureSelectorProps) {
  const currentFamily = ARCHITECTURE_FAMILIES.find(f => f.id === value);

  return (
    <Select value={value} onValueChange={(v) => onChange(v as ArchitectureFamily)}>
      <SelectTrigger
        title={currentFamily ? `${currentFamily.name} — ${currentFamily.description}` : undefined}
        className={cn(
          "w-[220px] h-9 shrink-0",
          "bg-secondary/50 border-border/50 hover:bg-secondary transition-colors",
          "focus:ring-1 focus:ring-primary/50",
          className
        )}
      >
        {/*
          The trigger draws the family name itself rather than delegating to
          `SelectValue`. `SelectValue` mirrors the whole selected row — icon,
          name, description and badge — which is right in a dropdown and far
          too much for a trigger that already carries an icon: inside the
          toolbar it overflowed and left a fragment of the name on screen,
          "Transformer / LLM" showing as "LLM". The full name and its
          description are on the trigger's tooltip.
        */}
        <div className="flex items-center gap-2 min-w-0">
          <div
            className="w-5 h-5 rounded flex items-center justify-center shrink-0 text-[13px] leading-none"
            style={{ backgroundColor: `${currentFamily?.color}20` }}
          >
            {currentFamily?.emoji ?? '🧠'}
          </div>
          <span className="truncate text-sm font-medium">
            {currentFamily?.name ?? 'Select architecture'}
          </span>
        </div>
      </SelectTrigger>
      <SelectContent className="bg-popover border-border z-50">
        {ARCHITECTURE_FAMILIES.map((family) => (
          <SelectItem
            key={family.id}
            value={family.id}
            className="cursor-pointer focus:bg-secondary"
          >
            <div className="flex items-center gap-3 py-0.5">
              <div
                className="w-6 h-6 rounded flex items-center justify-center text-[15px] leading-none"
                style={{ backgroundColor: `${family.color}20` }}
              >
                {family.emoji}
              </div>
              <div className="flex flex-col">
                <span className="text-sm font-medium">{family.name}</span>
                <span className="text-[10px] text-muted-foreground">{family.description}</span>
              </div>
            </div>
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
