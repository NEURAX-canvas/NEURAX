import { useMemo, useState, useCallback, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  Check, Key, LogOut, User,
  Settings, Bot, Save, X, Eye, EyeOff,
  Palette, AtSign, Mail, Shield,
} from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import { Label } from '@/components/ui/label.tsx';
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from '@/components/ui/select.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { useApiKey, PROVIDER_DEFAULTS, type ApiProvider, type ApiKeyConfig } from '@/contexts/ApiKeyContext.tsx';

import { supabase } from '@/lib/supabaseClient.ts';
import { useToast } from '@/hooks/use-toast.ts';


const SUPABASE_DISABLED = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

// ─── Notionists Avatar Family ────────────────────────────────────
import { NOTIONISTS_AVATARS } from '@/components/profile/NotionistsAvatarPicker.tsx';

// Extraire uniquement les emojis des Notionists
const AVATAR_EMOJIS = NOTIONISTS_AVATARS.map(avatar => avatar.emoji);

const EMOJI_KEY = 'neurax_account_emoji';

const PROVIDERS: { value: ApiProvider; label: string; icon: string }[] = [
  { value: 'openai', label: 'OpenAI', icon: '🤖' },
  { value: 'anthropic', label: 'Anthropic', icon: '🧠' },
  { value: 'google', label: 'Google AI', icon: '🔬' },
  { value: 'mistral', label: 'Mistral', icon: '🌬️' },
  { value: 'custom', label: 'Custom (OpenAI-compatible)', icon: '⚙️' },
];

export default function Account() {
  const { session, isAuthenticated, demoUser, demoSignOut } = useAuth();
  const { config: apiKeyConfig, isConfigured: hasApiKey, setApiKey, clearApiKey } = useApiKey();

  const navigate = useNavigate();
  const { toast } = useToast();

  // ── Navigation tabs ──
  const [activeTab, setActiveTab] = useState('profile');

  // ── Loading states ──
  const [busy, setBusy] = useState(false);

  // ── Émoji ──
  const [selectedEmoji, setSelectedEmoji] = useState<string>(() => {
    return localStorage.getItem(EMOJI_KEY) || NOTIONISTS_AVATARS[0].emoji;
  });

  // ── Profile edit ──
  const [editName, setEditName] = useState('');
  const [editEmail, setEditEmail] = useState('');
  const [isEditing, setIsEditing] = useState(false);

  // ── API Key ──
  const [apiProvider, setApiProvider] = useState<ApiProvider>(
    (apiKeyConfig?.provider as ApiProvider) || 'openai',
  );
  const [apiKeyValue, setApiKeyValue] = useState(apiKeyConfig?.key || '');
  const [apiCustomEndpoint, setApiCustomEndpoint] = useState(apiKeyConfig?.customEndpoint || '');
  const [apiModel, setApiModel] = useState(apiKeyConfig?.model || '');
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiKeyEditMode, setApiKeyEditMode] = useState(!hasApiKey);

  // ── Password ──
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPasswordForm, setShowPasswordForm] = useState(false);

  const username = useMemo(() => {
    if (SUPABASE_DISABLED && demoUser) return demoUser.username;
    const m = (session?.user?.user_metadata ?? {}) as Record<string, unknown>;
    return (typeof m.username === 'string' ? m.username : null) ?? session?.user?.email?.split('@')[0] ?? 'User';
  }, [session, demoUser]);

  const email = useMemo(() => {
    if (SUPABASE_DISABLED && demoUser) return demoUser.email;
    return session?.user?.email ?? null;
  }, [session, demoUser]);

  // Init edit fields
  useEffect(() => {
    if (!isEditing) {
      setEditName(username);
      setEditEmail(email || '');
    }
  }, [username, email, isEditing]);

  // ── Emoji ──
  const onSaveEmoji = useCallback(() => {
    localStorage.setItem(EMOJI_KEY, selectedEmoji);
    toast({ title: 'Emoji saved', description: 'Your account emoji has been updated.' });
  }, [selectedEmoji, toast]);

  // ── Profile ──
  const onSaveProfile = async () => {
    setBusy(true);
    try {
      if (SUPABASE_DISABLED) {
        toast({ title: 'Profile updated (demo)' });
        setIsEditing(false);
        return;
      }
      const { error: authErr } = await supabase.auth.updateUser({
        data: { username: editName.trim() },
      });
      if (authErr) throw authErr;

      const userId = session?.user?.id;
      if (userId) {
        const { error: profErr } = await supabase
          .from('user_profiles')
          .update({ username: editName.trim() })
          .eq('id', userId);
        if (profErr) throw profErr;
      }

      toast({ title: 'Profile saved' });
      setIsEditing(false);
    } catch (e: any) {
      toast({ title: 'Update failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  // ── API Key ──
  const onSaveApiKey = () => {
    if (!apiKeyValue.trim()) {
      toast({ title: 'API key required', description: 'Please enter your API key.', variant: 'destructive' });
      return;
    }

    const newConfig: ApiKeyConfig = {
      key: apiKeyValue.trim(),
      provider: apiProvider,
      label: PROVIDERS.find(p => p.value === apiProvider)?.label ?? apiProvider,
      ...(apiProvider === 'custom' && apiCustomEndpoint.trim() ? { customEndpoint: apiCustomEndpoint.trim() } : {}),
      ...(apiModel.trim() ? { model: apiModel.trim() } : {}),
    };

    setApiKey(newConfig);
    setApiKeyEditMode(false);
    toast({ title: 'API key saved', description: `Connected to ${newConfig.label}.` });
  };

  const onClearApiKey = () => {
    clearApiKey();
    setApiKeyValue('');
    setApiKeyEditMode(true);
    toast({ title: 'API key removed' });
  };

  const onProviderChange = (value: string) => {
    const p = value as ApiProvider;
    setApiProvider(p);
    const defaults = PROVIDER_DEFAULTS[p];
    setApiModel(defaults.defaultModel);
    if (p !== 'custom') setApiCustomEndpoint('');
  };

  // ── Sign out ──
  const onSignOut = async () => {
    setBusy(true);
    try {
      if (SUPABASE_DISABLED) {
        demoSignOut();
        toast({ title: 'Signed out' });
        navigate('/');
        return;
      }
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      toast({ title: 'Signed out' });
      navigate('/');
    } catch (e: any) {
      toast({ title: 'Sign out failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  // ── Not authenticated ──
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <div className="mx-auto max-w-2xl px-4 sm:px-6 py-10">
          <div className="flex items-center justify-between">
            <h1 className="text-xl font-semibold">Account</h1>
            <Button asChild variant="outline" size="sm">
              <Link to="/">Back</Link>
            </Button>
          </div>
          <div className="mt-6 rounded-xl border bg-card/50 p-6 text-sm text-muted-foreground">
            You're not signed in.
            <Button asChild variant="link" className="px-0">
              <Link to="/">Sign in</Link>
            </Button>
          </div>
        </div>
      </div>
    );
  }



  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-4xl px-4 sm:px-6 py-8">

        {/* ── Header ── */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <span className="text-4xl">{selectedEmoji}</span>
            <div>
              <h1 className="text-2xl font-bold">{username}</h1>
              <p className="text-sm text-muted-foreground">{email}</p>
            </div>
          </div>
          <Button asChild variant="outline" size="sm">
            <Link to="/app">Back to Studio</Link>
          </Button>
        </div>

        {/* ── Tabs ── */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="w-full justify-start bg-transparent border-b border-border rounded-none h-auto p-0 mb-6 space-x-6">
            <TabsTrigger value="profile" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none pb-3 px-1 text-sm">
              <User className="w-4 h-4 mr-2" /> Profile
            </TabsTrigger>
            <TabsTrigger value="api" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none pb-3 px-1 text-sm">
              <Key className="w-4 h-4 mr-2" /> API & Agent
            </TabsTrigger>

            <TabsTrigger value="security" className="data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none pb-3 px-1 text-sm">
              <Shield className="w-4 h-4 mr-2" /> Security
            </TabsTrigger>
          </TabsList>

          {/* ════════════════════════════════════════ PROFILE ════════════════════════════════════════ */}
          <TabsContent value="profile" className="space-y-6">
            {/* Avatar Emoji */}
            <div className="rounded-xl border bg-card p-6">
              <div className="flex items-center gap-3 mb-4">
                <Palette className="w-5 h-5 text-primary" />
                <h2 className="text-sm font-semibold">Notionist Avatar</h2>
              </div>
              <p className="text-xs text-muted-foreground mb-4">
                Choose your Notionist avatar from the family of 12 unique characters.
              </p>
              <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-12 gap-2">
                {NOTIONISTS_AVATARS.map((notionist) => {
                  const isSelected = selectedEmoji === notionist.emoji;
                  return (
                    <button
                      key={notionist.id}
                      type="button"
                      className={`h-12 w-12 rounded-lg border text-xl flex items-center justify-center transition-all relative group ${
                        isSelected
                          ? 'ring-2 scale-110'
                          : 'hover:scale-105'
                      }`}
                      style={{
                        borderColor: isSelected ? notionist.color : 'hsl(var(--border))',
                        backgroundColor: isSelected ? `${notionist.color}15` : 'transparent',
                        ...(isSelected && { 
                          '--tw-ring-color': `${notionist.color}40`
                        } as React.CSSProperties)
                      }}
                      onClick={() => setSelectedEmoji(notionist.emoji)}
                      aria-label={`Select ${notionist.name}`}
                      title={notionist.name}
                    >
                      {notionist.emoji}
                      {/* Tooltip on hover */}
                      <span className="absolute -top-8 left-1/2 -translate-x-1/2 bg-popover text-popover-foreground px-2 py-1 rounded text-[10px] font-medium whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10 shadow-md border">
                        {notionist.name.replace('Notion ', '')}
                      </span>
                    </button>
                  );
                })}
              </div>
              <div className="mt-4 flex items-center justify-between">
                <span className="text-xs text-muted-foreground">
                  {NOTIONISTS_AVATARS.find(n => n.emoji === selectedEmoji)?.name || 'Select an avatar'}
                </span>
                <Button size="sm" onClick={onSaveEmoji} disabled={busy}>
                  <Save className="w-3.5 h-3.5 mr-1.5" /> Save Avatar
                </Button>
              </div>
            </div>

            {/* Profile Info */}
            <div className="rounded-xl border bg-card p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <User className="w-5 h-5 text-primary" />
                  <h2 className="text-sm font-semibold">Personal Info</h2>
                </div>
                {!isEditing && (
                  <Button variant="outline" size="sm" onClick={() => setIsEditing(true)}>
                    <Settings className="w-3.5 h-3.5 mr-1.5" /> Edit
                  </Button>
                )}
              </div>

              {isEditing ? (
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="edit-name" className="text-xs text-muted-foreground">Username</Label>
                    <Input id="edit-name" value={editName} onChange={(e) => setEditName(e.target.value)} className="mt-1" />
                  </div>
                  {!SUPABASE_DISABLED && (
                    <div>
                      <Label htmlFor="edit-email" className="text-xs text-muted-foreground">Email</Label>
                      <Input id="edit-email" type="email" value={editEmail} onChange={(e) => setEditEmail(e.target.value)} className="mt-1" />
                    </div>
                  )}
                  <div className="flex justify-end gap-2 pt-2">
                    <Button variant="ghost" size="sm" onClick={() => setIsEditing(false)}>
                      <X className="w-3.5 h-3.5 mr-1" /> Cancel
                    </Button>
                    <Button size="sm" onClick={onSaveProfile} disabled={busy}>
                      <Save className="w-3.5 h-3.5 mr-1" /> Save
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm">
                    <AtSign className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Username:</span>
                    <span className="font-medium">{username}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Mail className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Email:</span>
                    <span className="font-medium">{email || '—'}</span>
                  </div>
                </div>
              )}
            </div>
          </TabsContent>

          {/* ════════════════════════════════════════ API & AGENT ════════════════════════════════════════ */}
          <TabsContent value="api" className="space-y-6">
            <div className="rounded-xl border bg-card p-6">
              <div className="flex items-center gap-3 mb-4">
                <Bot className="w-5 h-5 text-primary" />
                <h2 className="text-sm font-semibold">Neurax Agent — API Key</h2>
              </div>
              <p className="text-xs text-muted-foreground mb-4">
                Connect your AI provider to use Neurax Agent — the intelligent assistant that helps you design and analyze architectures.
                Your API key is stored locally and never sent to our servers.
              </p>

              {/* Provider */}
              <div className="space-y-4">
                <div>
                  <Label className="text-xs text-muted-foreground">AI Provider</Label>
                  <Select value={apiProvider} onValueChange={onProviderChange}>
                    <SelectTrigger className="mt-1">
                      <SelectValue placeholder="Select provider" />
                    </SelectTrigger>
                    <SelectContent>
                      {PROVIDERS.map((p) => (
                        <SelectItem key={p.value} value={p.value}>
                          <span className="flex items-center gap-2">
                            <span>{p.icon}</span>
                            <span>{p.label}</span>
                          </span>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* API Key */}
                {apiKeyEditMode ? (
                  <>
                    <div>
                      <Label className="text-xs text-muted-foreground">API Key</Label>
                      <div className="relative mt-1">
                        <Input
                          type={showApiKey ? 'text' : 'password'}
                          placeholder={
                            apiProvider === 'openai' ? 'sk-...' :
                            apiProvider === 'anthropic' ? 'sk-ant-...' :
                            apiProvider === 'google' ? 'AIza...' :
                            apiProvider === 'mistral' ? 'MISTRAL_...' :
                            'Enter your API key'
                          }
                          value={apiKeyValue}
                          onChange={(e) => setApiKeyValue(e.target.value)}
                          className="pr-10 font-mono text-sm"
                        />
                        <button
                          type="button"
                          className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                          onClick={() => setShowApiKey(!showApiKey)}
                          aria-label={showApiKey ? 'Hide API key' : 'Show API key'}
                        >
                          {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                    </div>

                    <div>
                      <Label className="text-xs text-muted-foreground">Model <span className="text-muted-foreground/50">(optional)</span></Label>
                      <Input
                        placeholder={PROVIDER_DEFAULTS[apiProvider].defaultModel}
                        value={apiModel}
                        onChange={(e) => setApiModel(e.target.value)}
                        className="mt-1"
                      />
                    </div>

                    {apiProvider === 'custom' && (
                      <div>
                        <Label className="text-xs text-muted-foreground">API Endpoint</Label>
                        <Input
                          placeholder="https://your-endpoint.com/v1"
                          value={apiCustomEndpoint}
                          onChange={(e) => setApiCustomEndpoint(e.target.value)}
                          className="mt-1"
                        />
                      </div>
                    )}

                    <div className="flex justify-end gap-2 pt-2">
                      {hasApiKey && (
                        <Button variant="ghost" size="sm" onClick={() => { setApiKeyEditMode(false); setApiKeyValue(apiKeyConfig?.key || ''); }}>
                          <X className="w-3.5 h-3.5 mr-1" /> Cancel
                        </Button>
                      )}
                      <Button size="sm" onClick={onSaveApiKey} disabled={busy || !apiKeyValue.trim()}>
                        <Save className="w-3.5 h-3.5 mr-1" /> Save API Key
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/40 border">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium">{apiKeyConfig?.label || 'Not configured'}</div>
                        <div className="text-xs text-muted-foreground truncate font-mono">
                          {apiKeyConfig?.key
                            ? `${apiKeyConfig.key.slice(0, 8)}${'•'.repeat(Math.min(16, apiKeyConfig.key.length - 8))}`
                            : 'No API key set'}
                        </div>
                      </div>
                      {hasApiKey && (
                        <span className="flex items-center gap-1 text-[10px] text-emerald-500 font-mono uppercase">
                          <Check className="w-3 h-3" /> Active
                        </span>
                      )}
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="outline" size="sm" onClick={() => setApiKeyEditMode(true)}>
                        <Key className="w-3.5 h-3.5 mr-1" /> Update Key
                      </Button>
                      {hasApiKey && (
                        <Button variant="outline" size="sm" className="text-destructive hover:text-destructive" onClick={onClearApiKey}>
                          <X className="w-3.5 h-3.5 mr-1" /> Remove
                        </Button>
                      )}
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-6 p-3 rounded-lg bg-primary/5 border border-primary/10">
                <p className="text-[10px] text-muted-foreground leading-relaxed">
                  <strong className="text-primary">Not sure?</strong> Your API key stays in your browser — we never see it.
                  It's used to power the Neurax Agent chat assistant directly from your browser to your chosen provider.
                </p>
              </div>
            </div>
          </TabsContent>



          {/* ════════════════════════════════════════ SECURITY ════════════════════════════════════════ */}
          <TabsContent value="security" className="space-y-6">
            {/* Password */}
            {!SUPABASE_DISABLED && (
              <div className="rounded-xl border bg-card p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Shield className="w-5 h-5 text-primary" />
                  <h2 className="text-sm font-semibold">Password</h2>
                </div>

                {showPasswordForm ? (
                  <div className="space-y-4">
                    <div>
                      <Label className="text-xs text-muted-foreground">Current Password</Label>
                      <Input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} className="mt-1" />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">New Password</Label>
                      <Input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} className="mt-1" />
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">Confirm New Password</Label>
                      <Input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="mt-1" />
                    </div>
                    <div className="flex justify-end gap-2">
                      <Button variant="ghost" size="sm" onClick={() => setShowPasswordForm(false)}>
                        Cancel
                      </Button>
                      <Button size="sm" onClick={async () => {
                        if (newPassword !== confirmPassword) {
                          toast({ title: 'Passwords do not match', variant: 'destructive' });
                          return;
                        }
                        setBusy(true);
                        try {
                          const { error } = await supabase.auth.updateUser({ password: newPassword });
                          if (error) throw error;
                          toast({ title: 'Password updated' });
                          setShowPasswordForm(false);
                          setCurrentPassword('');
                          setNewPassword('');
                          setConfirmPassword('');
                        } catch (e: any) {
                          toast({ title: 'Update failed', description: String(e?.message ?? e), variant: 'destructive' });
                        } finally {
                          setBusy(false);
                        }
                      }} disabled={busy || !newPassword || !currentPassword}>
                        Update Password
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div>
                    <p className="text-sm text-muted-foreground mb-3">
                      Update your account password.
                    </p>
                    <Button variant="outline" size="sm" onClick={() => setShowPasswordForm(true)}>
                      Change Password
                    </Button>
                  </div>
                )}
              </div>
            )}

            {/* Sign Out */}
            <div className="rounded-xl border bg-card p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <LogOut className="w-5 h-5 text-destructive" />
                  <div>
                    <h2 className="text-sm font-semibold">Sign Out</h2>
                    <p className="text-xs text-muted-foreground">Sign out of your account on this device.</p>
                  </div>
                </div>
                <Button variant="outline" size="sm" className="text-destructive hover:text-destructive" onClick={onSignOut} disabled={busy}>
                  <LogOut className="w-3.5 h-3.5 mr-1.5" /> Sign Out
                </Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>

    </div>
  );
}


