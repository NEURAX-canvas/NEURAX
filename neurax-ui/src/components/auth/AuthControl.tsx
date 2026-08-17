import { useMemo, useState, type ComponentProps } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, Mail, Github, Zap, LogOut, Key, Check } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { useApiKey, PROVIDER_DEFAULTS, type ApiProvider, type ApiKeyConfig } from '@/contexts/ApiKeyContext.tsx';
import {
  OpenAIIcon, AnthropicIcon, GeminiIcon, MistralIcon,
  FireworksIcon, DeepSeekIcon, GlmIcon, CustomProviderIcon,
} from '@/components/icons/ProviderIcons.tsx';
import { supabase } from '@/lib/supabaseClient.ts';
import { identiconDataUri } from '@/components/profile/Identicon.tsx';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog.tsx';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { NotionistsAvatarPicker, AVATAR_OPTIONS, resolveAvatar } from '@/components/profile/NotionistsAvatarPicker.tsx';
import { Identicon } from '@/components/profile/Identicon.tsx';


const SUPABASE_DISABLED = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

type AuthTab = 'password' | 'signup' | 'magic' | 'oauth';

interface AuthControlProps {
  initialTab?: AuthTab;
  triggerLabel?: string;
  triggerVariant?: ComponentProps<typeof Button>['variant'];
  triggerSize?: ComponentProps<typeof Button>['size'];
  triggerClassName?: string;
}

const PROVIDERS: { value: ApiProvider; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { value: 'openai', label: 'OpenAI', icon: OpenAIIcon },
  { value: 'anthropic', label: 'Anthropic', icon: AnthropicIcon },
  { value: 'google', label: 'Google AI (Gemini)', icon: GeminiIcon },
  { value: 'mistral', label: 'Mistral', icon: MistralIcon },
  { value: 'fireworks', label: 'Fireworks AI', icon: FireworksIcon },
  { value: 'deepseek', label: 'DeepSeek', icon: DeepSeekIcon },
  { value: 'glm', label: 'GLM (Zhipu)', icon: GlmIcon },
  { value: 'custom', label: 'Custom (OpenAI-compatible)', icon: CustomProviderIcon },
];

type SetupStep = 'auth' | 'apikey';

export function AuthControl({
  initialTab,
  triggerLabel,
  triggerVariant,
  triggerSize,
  triggerClassName,
}: AuthControlProps) {
  const { session, isAuthenticated, demoUser, demoSignIn, demoSignOut, demoAvatarUrl } = useAuth();
  const { config: apiKeyConfig, isConfigured: hasApiKey, setApiKey, markSetupComplete } = useApiKey();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [open, setOpen] = useState(false);
  const [setupStep, setSetupStep] = useState<SetupStep>('auth');
  const [activeTab, setActiveTab] = useState<AuthTab>('password');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [username, setUsername] = useState('');
  const [busy, setBusy] = useState(false);
  const [selectedAvatarId, setSelectedAvatarId] = useState<string>(AVATAR_OPTIONS[0].id);

  // API Key state
  const [apiProvider, setApiProvider] = useState<ApiProvider>('openai');
  const [apiKeyValue, setApiKeyValue] = useState('');
  const [apiCustomEndpoint, setApiCustomEndpoint] = useState('');
  const [apiModel, setApiModel] = useState('');

  const redirectTo = useMemo(() => {
    return `${window.location.origin}/app`;
  }, []);

  // The stored value is an avatar id, not an emoji. Rendering it as text put
  // the literal "ax-01" inside a 32px circle, where it wrapped onto two lines
  // and collided with the badge beside it.
  const storedAvatar = useMemo(() => localStorage.getItem('neurax_account_emoji'), []);
  const avatarEmoji = null;
  const avatarSrc = useMemo(() => {
    if (avatarEmoji) return null;
    if (SUPABASE_DISABLED && demoUser) {
      return demoAvatarUrl;
    }
    const m = (session?.user?.user_metadata ?? {}) as Record<string, unknown>;
    const metaUrl = typeof m.avatar_url === 'string' ? m.avatar_url : null;
    const fallback = identiconDataUri(session?.user?.email ?? 'user');
    return metaUrl ?? fallback;
  }, [session, demoUser, demoAvatarUrl, avatarEmoji]);

  const displayName = useMemo(() => {
    if (SUPABASE_DISABLED && demoUser) return demoUser.username;
    const m = session?.user?.user_metadata as Record<string, unknown> | undefined;
    return (typeof m?.username === 'string' ? m.username : session?.user?.email) ?? 'User';
  }, [session, demoUser]);

  // ── Reset dialog state ──
  const resetDialog = () => {
    setSetupStep('auth');
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setUsername('');
    setApiKeyValue('');
    setApiCustomEndpoint('');
    setApiModel('');
    setApiProvider('openai');
    setSelectedAvatarId(AVATAR_OPTIONS[0].id);
    setBusy(false);
  };

  const closeDialog = () => {
    setOpen(false);
    resetDialog();
  };

  // ── API Key Save ──
  const onSaveApiKey = () => {
    if (!apiKeyValue.trim()) {
      toast({ title: 'API key required', description: 'Please enter your API key to use Neurax Agent.', variant: 'destructive' });
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
    markSetupComplete();

    toast({
      title: 'API key saved',
      description: `You're ready to use Neurax with ${newConfig.label}.`,
    });

    closeDialog();
    navigate('/app');
  };

  const onSkipApiKey = () => {
    markSetupComplete();
    closeDialog();
    navigate('/app');
  };

  // ── Demo / Dev mode login ──────────────────────────────────
  const onDemoSignIn = () => {
    if (!email.trim()) {
      toast({ title: 'Email required', description: 'Please enter an email to continue.', variant: 'destructive' });
      return;
    }

    // Persist the chosen avatar by id; the identicon is drawn from its seed.
    localStorage.setItem('neurax_account_emoji', resolveAvatar(selectedAvatarId).id);

    demoSignIn(email.trim(), username.trim() || undefined);

    // Always land in the studio. An API key is only needed for the AI agent —
    // the compiler itself runs without one — so diverting to account settings
    // blocked a visitor on a requirement unrelated to what they came to do.
    closeDialog();
    if (hasApiKey) {
      toast({
        title: 'Welcome back!',
        description: `Signed in as ${username.trim() || email.trim().split('@')[0]}`,
      });
    } else {
      toast({
        title: 'Ready to analyse',
        description: 'Add an API key in Account settings when you want the AI agent to design for you.',
      });
    }
    navigate('/app');
  };

  // ── Supabase auth (prod mode) ──────────────────────────────
  const onSignUp = async () => {
    setBusy(true);
    try {
      if (password !== confirmPassword) {
        toast({ title: 'Passwords do not match', description: 'Please re-type your password.', variant: 'destructive' });
        return;
      }
      const { error } = await supabase.auth.signUp({
        email, password,
        options: {
          emailRedirectTo: redirectTo,
          data: {
            username: username || email.split('@')[0],
            avatar_url: identiconDataUri(email),
          },
        },
      });
      if (error) throw error;
      toast({ title: 'Account created', description: 'If email confirmation is enabled, check your inbox.' });
      closeDialog();
    } catch (e: any) {
      toast({ title: 'Sign up failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onPasswordSignIn = async () => {
    setBusy(true);
    try {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw error;
      toast({ title: 'Signed in' });
      closeDialog();
      navigate('/app');
    } catch (e: any) {
      toast({ title: 'Sign in failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onMagicLink = async () => {
    setBusy(true);
    try {
      const { error } = await supabase.auth.signInWithOtp({
        email,
        options: { emailRedirectTo: redirectTo },
      });
      if (error) throw error;
      toast({ title: 'Magic link sent', description: 'Check your email to finish signing in.' });
      closeDialog();
    } catch (e: any) {
      toast({ title: 'Magic link failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onOAuth = async (provider: 'google' | 'github') => {
    setBusy(true);
    try {
      const { error } = await supabase.auth.signInWithOAuth({ provider, options: { redirectTo } });
      if (error) throw error;
    } catch (e: any) {
      toast({ title: 'OAuth sign in failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  // ── Update provider and model ──
  const onProviderChange = (value: string) => {
    const p = value as ApiProvider;
    setApiProvider(p);
    const defaults = PROVIDER_DEFAULTS[p];
    setApiModel(defaults.defaultModel);
    if (p !== 'custom') {
      setApiCustomEndpoint('');
    }
  };

  // ── Authenticated state ────────────────────────────────────
  if (isAuthenticated) {
    return (
      <div className="flex items-center gap-2">
        {/* There was a plan badge here reading "OSS".
            NEURAX has one plan and it is open source, so the badge told the
            user nothing they could act on and took the place beside their own
            name — which is information. The plan itself is still on the
            account page for anyone who wants it. */}

        {/* API Key indicator */}
        {hasApiKey && (
          <div className="hidden sm:flex items-center gap-1 h-8 px-2 rounded-md border border-emerald-500/20 bg-emerald-500/5 text-[10px] font-mono text-emerald-400">
            <Key className="w-3 h-3" />
            <span className="truncate max-w-[80px]">{apiKeyConfig?.label ?? 'API'}</span>
          </div>
        )}

        {/* User avatar + name */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="h-8 w-8 rounded-full overflow-hidden border border-white/20 hover:border-white/40 transition-colors flex-shrink-0"
            onClick={() => navigate('/account')}
            aria-label="Open account"
            disabled={busy}
          >
            {avatarSrc ? (
              <img src={avatarSrc} alt="avatar" className="h-full w-full object-cover" />
            ) : (
              <Identicon seed={resolveAvatar(storedAvatar).seed} size={32} />
            )}
          </button>
          <span className="hidden sm:inline text-xs text-white/60 font-medium max-w-[100px] truncate">
            {displayName}
          </span>
          {/* Demo sign out */}
          {SUPABASE_DISABLED && (
            <button
              type="button"
              onClick={demoSignOut}
              className="h-7 w-7 rounded-md border border-white/10 hover:border-red-400/40 hover:bg-red-500/10 flex items-center justify-center transition-colors"
              aria-label="Sign out"
              title="Sign out"
            >
              <LogOut className="w-3.5 h-3.5 text-white/40 hover:text-red-400" />
            </button>
          )}
        </div>
      </div>
    );
  }

  // ── Dialog content: Auth step ──
  const renderAuthStep = () => (
    <>
      <DialogHeader>
        <DialogTitle className="text-white text-xl">
          {SUPABASE_DISABLED ? 'Welcome to NEURAX' : (activeTab === 'signup' ? 'Create Account' : 'Sign In')}
        </DialogTitle>
        <DialogDescription className="text-white/40">
          {SUPABASE_DISABLED
            ? 'Enter your name to start exploring the platform.'
            : 'Sign in to run analysis and access plan features.'}
        </DialogDescription>
      </DialogHeader>

      {SUPABASE_DISABLED ? (
        <div className="space-y-4 mt-2">
          <div>
            <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">Your Name</label>
            <Input placeholder="e.g. AI Explorer" value={username} onChange={(e) => setUsername(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
          </div>
          <div>
            <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">Email (optional)</label>
            <Input placeholder="you@example.com" type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
          </div>
          <div>
            <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-3 block">Choose Your Avatar</label>
            <div className="rounded-[8px] p-3 bg-white/[0.03] border border-white/[0.06]">
              <NotionistsAvatarPicker
                selectedId={selectedAvatarId}
                onSelect={setSelectedAvatarId}
              />
            </div>
          </div>
          <div className="pt-2 space-y-2">
            <Button className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold h-11" onClick={onDemoSignIn}>
              <Zap className="w-4 h-4 mr-2" />
              Enter Platform
            </Button>
            <p className="text-[10px] text-center text-white/20">Demo mode — no account required</p>
          </div>
        </div>
      ) : (
        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-white/5">
            <TabsTrigger value="password" className="text-xs">Password</TabsTrigger>
            <TabsTrigger value="signup" className="text-xs">Sign up</TabsTrigger>
            <TabsTrigger value="magic" className="text-xs">Magic link</TabsTrigger>
            <TabsTrigger value="oauth" className="text-xs">OAuth</TabsTrigger>
          </TabsList>
          <TabsContent value="password" className="space-y-3 mt-3">
            <Input placeholder="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
            <Input placeholder="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
            <Button className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold" onClick={onPasswordSignIn} disabled={busy || !email || !password}>
              <LogIn className="w-4 h-4 mr-2" /> Sign in
            </Button>
          </TabsContent>
          <TabsContent value="signup" className="space-y-3 mt-3">
            <Input placeholder="Username (optional)" value={username} onChange={(e) => setUsername(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
            <Input placeholder="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
            <Input placeholder="Password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
            <Input placeholder="Confirm password" type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
            <Button className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold" onClick={onSignUp} disabled={busy || !email || !password || !confirmPassword}>
              <LogIn className="w-4 h-4 mr-2" /> Create account
            </Button>
          </TabsContent>
          <TabsContent value="magic" className="space-y-3 mt-3">
            <Input placeholder="Email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="bg-white/5 border-white/10 text-white placeholder:text-white/20" />
            <Button className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold" onClick={onMagicLink} disabled={busy || !email}>
              <Mail className="w-4 h-4 mr-2" /> Send magic link
            </Button>
          </TabsContent>
          <TabsContent value="oauth" className="space-y-2 mt-3">
            <Button className="w-full bg-white/5 border-white/10 text-white hover:bg-white/10" variant="outline" onClick={() => onOAuth('google')} disabled={busy}>
              <Mail className="w-4 h-4 mr-2" /> Continue with Google
            </Button>
            <Button className="w-full bg-white/5 border-white/10 text-white hover:bg-white/10" variant="outline" onClick={() => onOAuth('github')} disabled={busy}>
              <Github className="w-4 h-4 mr-2" /> Continue with GitHub
            </Button>
          </TabsContent>
        </Tabs>
      )}
    </>
  );

  // ── Dialog content: API key setup step ──
  const renderApiKeyStep = () => (
    <>
      <DialogHeader>
        <DialogTitle className="text-white text-xl flex items-center gap-2">
          <Key className="w-5 h-5 text-emerald-400" />
          Configure AI Agent
        </DialogTitle>
        <DialogDescription className="text-white/40">
          Connect your AI provider to use Neurax Agent — the intelligent assistant that helps you design and analyze architectures.
        </DialogDescription>
      </DialogHeader>

      <div className="space-y-4 mt-2">
        {/* Provider Selector */}
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
            AI Provider
          </label>
          <Select value={apiProvider} onValueChange={onProviderChange}>
            <SelectTrigger className="bg-white/5 border-white/10 text-white">
              <SelectValue placeholder="Select provider" />
            </SelectTrigger>
            <SelectContent className="bg-[#1a1a2e] border-white/10">
              {PROVIDERS.map((p) => (
                <SelectItem key={p.value} value={p.value} className="text-white focus:bg-white/10 focus:text-white">
                  <span className="flex items-center gap-2">
                    <p.icon className="w-4 h-4 shrink-0" />
                    <span>{p.label}</span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* API Key */}
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
            API Key
          </label>
          <Input
            type="password"
            placeholder={
              apiProvider === 'openai' ? 'sk-...' :
              apiProvider === 'anthropic' ? 'sk-ant-...' :
              apiProvider === 'google' ? 'AIza...' :
              apiProvider === 'mistral' ? 'MISTRAL_...' :
              apiProvider === 'fireworks' ? 'fw_...' :
              apiProvider === 'deepseek' ? 'sk-...' :
              apiProvider === 'glm' ? 'GLM key' :
              'Enter your API key'
            }
            value={apiKeyValue}
            onChange={(e) => setApiKeyValue(e.target.value)}
            className="bg-white/5 border-white/10 text-white placeholder:text-white/20 font-mono text-[12px]"
          />
        </div>

        {/* Model (optional) */}
        <div>
          <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
            Model <span className="text-white/20">(optional)</span>
          </label>
          <Input
            placeholder={PROVIDER_DEFAULTS[apiProvider].defaultModel}
            value={apiModel}
            onChange={(e) => setApiModel(e.target.value)}
            className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
          />
        </div>

        {/* Custom endpoint (only for custom provider) */}
        {apiProvider === 'custom' && (
          <div>
            <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
              API Endpoint
            </label>
            <Input
              placeholder="https://your-endpoint.com/v1"
              value={apiCustomEndpoint}
              onChange={(e) => setApiCustomEndpoint(e.target.value)}
              className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
            />
          </div>
        )}

        {/* Actions */}
        <div className="pt-4 space-y-2">
          <Button
            className="w-full bg-emerald-500 text-white hover:bg-emerald-600 font-semibold h-11"
            onClick={onSaveApiKey}
          >
            <Check className="w-4 h-4 mr-2" />
            Save & Start Using Neurax
          </Button>
          <Button
            variant="ghost"
            className="w-full text-white/30 hover:text-white/50 hover:bg-white/5"
            onClick={onSkipApiKey}
          >
            Skip for now — I'll configure later
          </Button>
        </div>

        <p className="text-[10px] text-center text-white/20 leading-relaxed">
          Your API key is stored locally and never sent to our servers.
          <br />
          Neurax Agent uses your own AI provider for intelligent assistance.
        </p>
      </div>
    </>
  );

  // ── Not authenticated ──────────────────────────────────────
  return (
    <>
      <Button
        variant={triggerVariant ?? 'default'}
        size={triggerSize ?? 'sm'}
        className={triggerClassName ?? 'bg-white/10 text-white hover:bg-white/20 border border-white/10'}
        onClick={() => {
          setActiveTab(initialTab ?? 'password');
          setOpen(true);
        }}
      >
        <LogIn className="w-4 h-4 sm:mr-1.5" />
        <span className="hidden sm:inline">{triggerLabel ?? 'Sign in'}</span>
      </Button>

      <Dialog open={open} onOpenChange={(isOpen) => {
        if (!isOpen) {
          closeDialog();
        }
        setOpen(isOpen);
      }}>
        <DialogContent className={`sm:max-w-md bg-[#0c0c1a] border border-white/10 shadow-2xl ${setupStep === 'apikey' ? 'sm:max-w-lg' : SUPABASE_DISABLED ? 'sm:max-w-lg' : ''}`}>
          {setupStep === 'auth' ? renderAuthStep() : renderApiKeyStep()}
        </DialogContent>
      </Dialog>
    </>
  );
}
