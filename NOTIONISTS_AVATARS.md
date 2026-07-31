# 🎭 Système d'Avatars Notionists

**Version :** 1.0  
**Date :** 31 Juillet 2026  
**Fichier :** `/neurax-ui/src/components/profile/NotionistsAvatarPicker.tsx`

---

## 🎨 Vue d'Ensemble

Les **Notionists** sont une famille de 12 avatars par défaut pour les profils utilisateurs NEURAX. Chaque Notionist a :
- Un **nom** unique (Alpha, Beta, Gamma, etc.)
- Un **emoji** distinctif
- Une **couleur** Gruvbox dédiée

---

## 👥 La Famille Notionists (12)

| ID | Nom | Emoji | Couleur | Hex | Signification |
|----|-----|-------|---------|-----|---------------|
| n1 | **Notion Alpha** | 🚀 | Doré | #d79921 | Innovation, Pionnier |
| n2 | **Notion Beta** | ⚡ | Cyan | #83a598 | Rapidité, Énergie |
| n3 | **Notion Gamma** | 🎯 | Vert | #98971a | Précision, Focus |
| n4 | **Notion Delta** | 🔥 | Orange | #d65d0e | Passion, Intensité |
| n5 | **Notion Epsilon** | 💎 | Violet | #b16286 | Qualité, Excellence |
| n6 | **Notion Zeta** | 🌟 | Jaune | #fabd2f | Brillance, Créativité |
| n7 | **Notion Eta** | 🎨 | Rouge | #fb4934 | Art, Expression |
| n8 | **Notion Theta** | 🧠 | Vert clair | #8ec07c | Intelligence, Logique |
| n9 | **Notion Iota** | 🔮 | Rose | #d3869b | Intuition, Vision |
| n10 | **Notion Kappa** | ⚙️ | Orange vif | #fe8019 | Technique, Ingénierie |
| n11 | **Notion Lambda** | 🎭 | Bleu | #458588 | Adaptabilité, Polyvalence |
| n12 | **Notion Mu** | 🌈 | Vert aqua | #689d6a | Harmonie, Équilibre |

---

## 🛠️ Utilisation

### 1. Sélecteur d'Avatar (Création de Profil)

```tsx
import { NotionistsAvatarPicker, useNotionistAvatar } from '@/components/profile/NotionistsAvatarPicker';

function ProfileCreation() {
  const { selectedId, setSelectedId } = useNotionistAvatar();

  return (
    <div>
      <h2>Créer votre profil</h2>
      <NotionistsAvatarPicker
        selectedId={selectedId}
        onSelect={setSelectedId}
      />
      <button onClick={() => saveProfile({ avatarId: selectedId })}>
        Créer
      </button>
    </div>
  );
}
```

### 2. Affichage d'Avatar (Header, Profil)

```tsx
import { NotionistAvatarDisplay } from '@/components/profile/NotionistsAvatarPicker';

// Petit avatar dans header
<NotionistAvatarDisplay avatarId="n1" size="sm" />

// Avatar moyen avec nom
<NotionistAvatarDisplay avatarId="n5" size="md" showName />

// Grand avatar pour page profil
<NotionistAvatarDisplay avatarId="n8" size="lg" showName />
```

### 3. Hook pour Gestion d'État

```tsx
import { useNotionistAvatar } from '@/components/profile/NotionistsAvatarPicker';

function MyComponent() {
  const { selectedId, selectedAvatar, setSelectedId } = useNotionistAvatar('n3');
  
  console.log(selectedAvatar);
  // { id: 'n3', name: 'Notion Gamma', emoji: '🎯', color: '#98971a' }

  return (
    <div>
      <p>Avatar actuel : {selectedAvatar.emoji} {selectedAvatar.name}</p>
      <button onClick={() => setSelectedId('n7')}>
        Changer pour Notion Eta
      </button>
    </div>
  );
}
```

---

## 🎨 Design Système

### Tailles Disponibles

```typescript
type Size = 'sm' | 'md' | 'lg';

const sizeMap = {
  sm: { circle: '32px', emoji: '16px', text: '11px' },  // Header, liste
  md: { circle: '40px', emoji: '20px', text: '12px' },  // Défaut, menu
  lg: { circle: '56px', emoji: '28px', text: '14px' },  // Profil, modal
};
```

### États Visuels

#### Non sélectionné
```css
background: #282828 (card)
border: 1px solid #3c3836 (border)
opacity: 1.0
```

#### Sélectionné
```css
background: ${color}15 (15% alpha)
border: 2px solid ${color}
checkmark: Visible
scale: 1.0
```

#### Hover
```css
transform: scale(1.05)
transition: 150ms ease
```

### Cercle Avatar
```css
background: ${color}20 (20% alpha)
border: 2px solid ${color}40 (40% alpha)
emoji: center, no antialiasing
```

---

## 📦 Stockage

### Format Recommandé (Base de Données)

```typescript
interface UserProfile {
  id: string;
  username: string;
  email: string;
  avatarId: string;  // 'n1' à 'n12'
  createdAt: Date;
  updatedAt: Date;
}
```

### Exemple Supabase

```sql
CREATE TABLE profiles (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id),
  username TEXT UNIQUE NOT NULL,
  avatar_id TEXT NOT NULL CHECK (avatar_id ~ '^n([1-9]|1[0-2])$'),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);
```

### Migration Utilisateurs Existants

Si vous avez déjà des utilisateurs sans avatar :

```typescript
// Assigner un avatar aléatoire
const randomAvatar = NOTIONISTS_AVATARS[
  Math.floor(Math.random() * NOTIONISTS_AVATARS.length)
].id;

// Ou basé sur l'ID utilisateur (déterministe)
const avatarIndex = parseInt(userId.slice(0, 8), 16) % 12;
const avatarId = NOTIONISTS_AVATARS[avatarIndex].id;
```

---

## 🔧 Personnalisation

### Ajouter de Nouveaux Avatars

```typescript
// Dans NotionistsAvatarPicker.tsx
export const NOTIONISTS_AVATARS = [
  // ... 12 existants
  { id: 'n13', name: 'Notion Nu', emoji: '🦋', color: '#abc123' },
  { id: 'n14', name: 'Notion Xi', emoji: '🌸', color: '#def456' },
];
```

### Changer les Couleurs

```typescript
// Utiliser uniquement des couleurs Gruvbox
const gruvboxPalette = [
  '#d79921', '#83a598', '#98971a', '#d65d0e',
  '#b16286', '#fabd2f', '#fb4934', '#8ec07c',
  '#d3869b', '#fe8019', '#458588', '#689d6a'
];
```

---

## 🎯 Cas d'Usage

### 1. Création de Compte
```tsx
<SignupForm>
  <Input name="username" />
  <Input name="email" />
  <NotionistsAvatarPicker onSelect={setAvatar} />
  <Button>Créer mon compte</Button>
</SignupForm>
```

### 2. Paramètres Profil
```tsx
<ProfileSettings>
  <Section title="Avatar">
    <NotionistAvatarDisplay avatarId={user.avatarId} size="lg" showName />
    <Button onClick={() => setEditMode(true)}>Changer</Button>
  </Section>
  
  {editMode && (
    <Modal>
      <NotionistsAvatarPicker
        selectedId={user.avatarId}
        onSelect={updateAvatar}
      />
    </Modal>
  )}
</ProfileSettings>
```

### 3. Liste d'Utilisateurs
```tsx
<UserList>
  {users.map(user => (
    <UserCard key={user.id}>
      <NotionistAvatarDisplay avatarId={user.avatarId} size="sm" />
      <span>{user.username}</span>
    </UserCard>
  ))}
</UserList>
```

### 4. Header Navigation
```tsx
<Header>
  <Logo />
  <Nav />
  <UserMenu>
    <NotionistAvatarDisplay
      avatarId={currentUser.avatarId}
      size="sm"
    />
    <Dropdown>
      <Item>Profil</Item>
      <Item>Paramètres</Item>
      <Item>Déconnexion</Item>
    </Dropdown>
  </UserMenu>
</Header>
```

---

## 📊 Statistiques Recommandées

### Analytics à Tracker
```typescript
// Popularité des avatars
interface AvatarStats {
  avatarId: string;
  count: number;
  percentage: number;
}

// Exemple query
SELECT 
  avatar_id,
  COUNT(*) as count,
  ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM profiles), 2) as percentage
FROM profiles
GROUP BY avatar_id
ORDER BY count DESC;
```

### Dashboard Admin
- 📊 Avatar le plus populaire
- 📈 Distribution par couleur
- 🎯 Taux d'abandon sans avatar sélectionné
- 🔄 Fréquence changement d'avatar

---

## 🎨 Principes de Design

### Cohérence Visuelle
✅ **DO:**
- Utiliser uniquement les 12 avatars Notionists
- Respecter les couleurs Gruvbox
- Garder les emojis natifs du système
- Afficher l'avatar de manière cohérente partout

❌ **DON'T:**
- Ajouter des avatars custom/upload image
- Changer les couleurs hors palette Gruvbox
- Utiliser des SVG à la place d'emojis
- Varier les styles selon les pages

### Accessibilité
- ✅ Les emojis ont un bon contrast avec le background
- ✅ Les noms d'avatars sont lisibles (12px minimum)
- ✅ Keyboard navigation sur le sélecteur
- ✅ ARIA labels sur les boutons avatar

---

## 🐛 Dépannage

### Avatar ne s'affiche pas
```typescript
// Vérifier que l'ID existe
const avatar = NOTIONISTS_AVATARS.find(a => a.id === avatarId);
if (!avatar) {
  // Fallback sur Notion Alpha
  avatarId = 'n1';
}
```

### Emoji ne s'affiche pas
```css
/* S'assurer que les fonts systèmes sont utilisées */
font-family: 'Apple Color Emoji', 'Segoe UI Emoji', 'Noto Color Emoji', sans-serif;
```

### Couleur incorrecte
```typescript
// Toujours utiliser la couleur du avatar, pas hardcodée
style={{ color: selectedAvatar.color }}
```

---

## 🚀 Prochaines Étapes

### Phase 2 : Avatars Animés
- Ajouter des micro-animations au hover
- Glow effect sur sélection
- Bounce animation lors du changement

### Phase 3 : Achievements
- Débloquer des avatars spéciaux (Notion Omega, Notion Sigma)
- Badges sur avatars (contributeur, early adopter)
- Avatars saisonniers (Noël, Halloween)

### Phase 4 : Personnalisation
- Changer la couleur de n'importe quel avatar
- Ajouter des accessoires (lunettes, casquette)
- Mode avatar "holographique" premium

---

## 📚 Ressources

### Code Source
- `/neurax-ui/src/components/profile/NotionistsAvatarPicker.tsx`

### Documentation Liée
- `AUDIT_FINAL.md` - Audit complet du projet
- `RENDU_FINAL.md` - Vue d'ensemble landing page
- `README.md` - Documentation principale

### Design System
- Gruvbox Theme : https://github.com/morhetz/gruvbox
- Emoji Specs : https://unicode.org/emoji/charts/full-emoji-list.html

---

**Créé par :** Kiro AI Agent  
**Date :** 31 Juillet 2026  
**Version :** 1.0

🎭 **Welcome to the Notionists Family!**
