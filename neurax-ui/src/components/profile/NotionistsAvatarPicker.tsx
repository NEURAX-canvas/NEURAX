/**
 * Avatar picker.
 *
 * The set used to be twelve emoji on coloured discs. It is now generated
 * identicons, in the style GitHub gives accounts without a picture: a
 * deterministic symmetric grid drawn from a seed. They read as an identity
 * rather than a decoration, they cannot be confused with the emoji the rest of
 * the product uses for status, and nothing is fetched to render one.
 */
import { useState } from 'react';
import { Check } from 'lucide-react';
import { cn } from '@/lib/utils.ts';
import { Identicon } from '@/components/profile/Identicon.tsx';

export interface AvatarOption {
  id: string;
  name: string;
  /** Seed the identicon is drawn from; stable for the lifetime of the option. */
  seed: string;
}

/**
 * The offered set.
 *
 * Seeds are fixed words rather than random values so the same avatar appears
 * for the same person on every device, and so this list can be extended without
 * changing anyone's existing picture.
 */
export const AVATAR_OPTIONS: AvatarOption[] = [
  { id: 'ax-01', name: 'Vector', seed: 'neurax-vector' },
  { id: 'ax-02', name: 'Tensor', seed: 'neurax-tensor' },
  { id: 'ax-03', name: 'Kernel', seed: 'neurax-kernel' },
  { id: 'ax-04', name: 'Gradient', seed: 'neurax-gradient' },
  { id: 'ax-05', name: 'Lattice', seed: 'neurax-lattice' },
  { id: 'ax-06', name: 'Cipher', seed: 'neurax-cipher' },
  { id: 'ax-07', name: 'Quanta', seed: 'neurax-quanta' },
  { id: 'ax-08', name: 'Circuit', seed: 'neurax-circuit' },
  { id: 'ax-09', name: 'Entropy', seed: 'neurax-entropy' },
  { id: 'ax-10', name: 'Manifold', seed: 'neurax-manifold' },
  { id: 'ax-11', name: 'Spectra', seed: 'neurax-spectra' },
  { id: 'ax-12', name: 'Nucleus', seed: 'neurax-nucleus' },
  { id: 'ax-13', name: 'Photon', seed: 'neurax-photon' },
  { id: 'ax-14', name: 'Fractal', seed: 'neurax-fractal' },
  { id: 'ax-15', name: 'Vertex', seed: 'neurax-vertex' },
  { id: 'ax-16', name: 'Synapse', seed: 'neurax-synapse' },
  { id: 'ax-17', name: 'Cortex', seed: 'neurax-cortex' },
  { id: 'ax-18', name: 'Neutrino', seed: 'neurax-neutrino' },
  { id: 'ax-19', name: 'Plasma', seed: 'neurax-plasma' },
  { id: 'ax-20', name: 'Helix', seed: 'neurax-helix' },
  { id: 'ax-21', name: 'Prism', seed: 'neurax-prism' },
  { id: 'ax-22', name: 'Quark', seed: 'neurax-quark' },
  { id: 'ax-23', name: 'Nebula', seed: 'neurax-nebula' },
  { id: 'ax-24', name: 'Axiom', seed: 'neurax-axiom' },
  { id: 'ax-25', name: 'Codex', seed: 'neurax-codex' },
  { id: 'ax-26', name: 'Matrix', seed: 'neurax-matrix' },
  { id: 'ax-27', name: 'Vortex', seed: 'neurax-vortex' },
  { id: 'ax-28', name: 'Zenith', seed: 'neurax-zenith' },
  { id: 'ax-29', name: 'Pulsar', seed: 'neurax-pulsar' },
  { id: 'ax-30', name: 'Cascade', seed: 'neurax-cascade' },
  { id: 'ax-31', name: 'Nexus', seed: 'neurax-nexus' },
  { id: 'ax-32', name: 'Radian', seed: 'neurax-radian' },
  { id: 'ax-33', name: 'Simplex', seed: 'neurax-simplex' },
  { id: 'ax-34', name: 'Boson', seed: 'neurax-boson' },
  { id: 'ax-35', name: 'Cypher', seed: 'neurax-cypher' },
  { id: 'ax-36', name: 'Halcyon', seed: 'neurax-halcyon' },
];

/**
 * Kept under its previous name so existing imports keep working.
 * @deprecated Prefer {@link AVATAR_OPTIONS}.
 */
export const NOTIONISTS_AVATARS = AVATAR_OPTIONS;

/** Resolve a stored identifier, tolerating values written by the emoji set. */
export function resolveAvatar(stored: string | null | undefined): AvatarOption {
  if (!stored) return AVATAR_OPTIONS[0];
  const byId = AVATAR_OPTIONS.find((option) => option.id === stored);
  if (byId) return byId;
  // Anything else — including an emoji saved by the previous picker — is
  // hashed to a stable option, so an existing profile keeps a consistent
  // avatar instead of silently resetting to the first one.
  let sum = 0;
  for (let i = 0; i < stored.length; i++) sum = (sum + stored.charCodeAt(i)) % 9973;
  return AVATAR_OPTIONS[sum % AVATAR_OPTIONS.length];
}

interface AvatarPickerProps {
  selectedId?: string;
  onSelect: (avatarId: string) => void;
}

export const NotionistsAvatarPicker = ({ selectedId, onSelect }: AvatarPickerProps) => {
  const selected = resolveAvatar(selectedId);

  return (
    <div className="space-y-3">
      <div className="mb-4">
        <h3 className="text-[15px] font-semibold text-foreground">Choose your avatar</h3>
        <p className="text-[12px] mt-0.5 text-muted-foreground">
          {AVATAR_OPTIONS.length} generated patterns — each drawn from its own seed.
        </p>
      </div>

      <div
        className="grid grid-cols-4 sm:grid-cols-6 gap-2.5 max-h-[320px] overflow-y-auto pr-1 scrollbar-thin"
        role="radiogroup"
        aria-label="Avatar"
      >
        {AVATAR_OPTIONS.map((avatar) => {
          const isSelected = selected.id === avatar.id;
          return (
            <button
              key={avatar.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              aria-label={avatar.name}
              onClick={() => onSelect(avatar.id)}
              className={cn(
                'relative flex flex-col items-center gap-1.5 p-2 rounded-[8px] border transition-all duration-150 hover:scale-105',
                isSelected
                  ? 'border-primary bg-primary/10'
                  : 'border-border/60 bg-card hover:border-border',
              )}
            >
              <div className="text-foreground">
                <Identicon seed={avatar.seed} size={38} />
              </div>
              <span
                className={cn(
                  'text-[10px] font-medium text-center leading-tight',
                  isSelected ? 'text-primary' : 'text-muted-foreground',
                )}
              >
                {avatar.name}
              </span>

              {isSelected && (
                <div className="absolute top-1 right-1 w-5 h-5 rounded-full flex items-center justify-center bg-primary">
                  <Check className="w-3 h-3 text-primary-foreground" />
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

/** Track the selected avatar. */
export const useNotionistAvatar = (initialId?: string) => {
  const [selectedId, setSelectedId] = useState<string>(
    () => resolveAvatar(initialId).id,
  );
  return {
    selectedId,
    selectedAvatar: resolveAvatar(selectedId),
    setSelectedId,
  };
};

interface AvatarDisplayProps {
  avatarId: string;
  size?: 'sm' | 'md' | 'lg';
  showName?: boolean;
}

const SIZES = { sm: 32, md: 40, lg: 56 } as const;

export const NotionistAvatarDisplay = ({
  avatarId,
  size = 'md',
  showName = false,
}: AvatarDisplayProps) => {
  const avatar = resolveAvatar(avatarId);
  return (
    <div className="flex items-center gap-2">
      <div className="text-foreground">
        <Identicon seed={avatar.seed} size={SIZES[size]} />
      </div>
      {showName && (
        <span className="font-medium text-[12px] text-foreground">{avatar.name}</span>
      )}
    </div>
  );
};
