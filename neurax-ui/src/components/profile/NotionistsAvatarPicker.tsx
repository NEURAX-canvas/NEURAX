import { useState } from 'react';
import { Check } from 'lucide-react';

const C = {
  bg: '#1d2021',
  card: '#282828',
  border: '#3c3836',
  text: '#ebdbb2',
  muted: '#a89984',
  faint: '#7c6f64',
  accent: '#d79921',
  green: '#98971a',
};

// Famille d'avatars Notionists
export const NOTIONISTS_AVATARS = [
  { id: 'n1', name: 'Notion Alpha', emoji: '🚀', color: '#d79921' },
  { id: 'n2', name: 'Notion Beta', emoji: '⚡', color: '#83a598' },
  { id: 'n3', name: 'Notion Gamma', emoji: '🎯', color: '#98971a' },
  { id: 'n4', name: 'Notion Delta', emoji: '🔥', color: '#d65d0e' },
  { id: 'n5', name: 'Notion Epsilon', emoji: '💎', color: '#b16286' },
  { id: 'n6', name: 'Notion Zeta', emoji: '🌟', color: '#fabd2f' },
  { id: 'n7', name: 'Notion Eta', emoji: '🎨', color: '#fb4934' },
  { id: 'n8', name: 'Notion Theta', emoji: '🧠', color: '#8ec07c' },
  { id: 'n9', name: 'Notion Iota', emoji: '🔮', color: '#d3869b' },
  { id: 'n10', name: 'Notion Kappa', emoji: '⚙️', color: '#fe8019' },
  { id: 'n11', name: 'Notion Lambda', emoji: '🎭', color: '#458588' },
  { id: 'n12', name: 'Notion Mu', emoji: '🌈', color: '#689d6a' },
];

interface NotionistsAvatarPickerProps {
  selectedId?: string;
  onSelect: (avatarId: string) => void;
}

export const NotionistsAvatarPicker = ({ selectedId, onSelect }: NotionistsAvatarPickerProps) => {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-[15px] font-semibold" style={{ color: C.text }}>
            Choisissez votre Notionist
          </h3>
          <p className="text-[12px] mt-0.5" style={{ color: C.muted }}>
            Sélectionnez votre avatar de profil
          </p>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3">
        {NOTIONISTS_AVATARS.map((avatar) => {
          const isSelected = selectedId === avatar.id;
          return (
            <button
              key={avatar.id}
              onClick={() => onSelect(avatar.id)}
              className="relative flex flex-col items-center gap-2 p-3 rounded-[8px] transition-all duration-150 hover:scale-105"
              style={{
                backgroundColor: isSelected ? `${avatar.color}15` : C.card,
                border: isSelected ? `2px solid ${avatar.color}` : `1px solid ${C.border}`,
              }}
            >
              {/* Avatar Circle */}
              <div
                className="w-12 h-12 rounded-full flex items-center justify-center text-[24px] transition-transform"
                style={{
                  backgroundColor: `${avatar.color}20`,
                  border: `2px solid ${avatar.color}40`,
                }}
              >
                {avatar.emoji}
              </div>

              {/* Name */}
              <span
                className="text-[10px] font-medium text-center leading-tight"
                style={{ color: isSelected ? avatar.color : C.muted }}
              >
                {avatar.name}
              </span>

              {/* Check Badge */}
              {isSelected && (
                <div
                  className="absolute top-1 right-1 w-5 h-5 rounded-full flex items-center justify-center"
                  style={{ backgroundColor: avatar.color }}
                >
                  <Check className="w-3 h-3 text-white" />
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};

// Hook pour gérer l'avatar sélectionné
export const useNotionistAvatar = (initialId?: string) => {
  const [selectedId, setSelectedId] = useState<string>(initialId || NOTIONISTS_AVATARS[0].id);
  
  const selectedAvatar = NOTIONISTS_AVATARS.find(a => a.id === selectedId) || NOTIONISTS_AVATARS[0];
  
  return {
    selectedId,
    selectedAvatar,
    setSelectedId,
  };
};

// Composant pour afficher un avatar (dans header, profil, etc.)
interface NotionistAvatarDisplayProps {
  avatarId: string;
  size?: 'sm' | 'md' | 'lg';
  showName?: boolean;
}

export const NotionistAvatarDisplay = ({ 
  avatarId, 
  size = 'md', 
  showName = false 
}: NotionistAvatarDisplayProps) => {
  const avatar = NOTIONISTS_AVATARS.find(a => a.id === avatarId) || NOTIONISTS_AVATARS[0];
  
  const sizeMap = {
    sm: { circle: 'w-8 h-8', emoji: 'text-[16px]', text: 'text-[11px]' },
    md: { circle: 'w-10 h-10', emoji: 'text-[20px]', text: 'text-[12px]' },
    lg: { circle: 'w-14 h-14', emoji: 'text-[28px]', text: 'text-[14px]' },
  };
  
  const s = sizeMap[size];
  
  return (
    <div className="flex items-center gap-2">
      <div
        className={`${s.circle} rounded-full flex items-center justify-center ${s.emoji}`}
        style={{
          backgroundColor: `${avatar.color}20`,
          border: `2px solid ${avatar.color}40`,
        }}
      >
        {avatar.emoji}
      </div>
      
      {showName && (
        <span className={`font-medium ${s.text}`} style={{ color: C.text }}>
          {avatar.name}
        </span>
      )}
    </div>
  );
};
