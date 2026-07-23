import { useMemo, useState, type ComponentProps } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, Mail, Github, Gift, Zap, Sparkles, Crown, LogOut, type LucideIcon } from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext.tsx';
import { usePlan } from '@/contexts/PlanContext.tsx';
import { PLAN_CONFIGS, type PlanTier } from '@/types/plans.ts';
import { supabase } from '@/lib/supabaseClient.ts';
import { Button } from '@/components/ui/button.tsx';
import { Input } from '@/components/ui/input.tsx';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover.tsx';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog.tsx';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.tsx';
import { useToast } from '@/hooks/use-toast.ts';
import { createBillingPortalSession, createCheckoutSession } from '@/services/neuraxApi.ts';

const SUPABASE_DISABLED = import.meta.env.VITE_SUPABASE_DISABLED === 'true';

type AuthTab = 'password' | 'signup' | 'magic' | 'oauth';

interface AuthControlProps {
  initialTab?: AuthTab;
  triggerLabel?: string;
  triggerVariant?: ComponentProps<typeof Button>['variant'];
  triggerSize?: ComponentProps<typeof Button>['size'];
  triggerClassName?: string;
}

export function AuthControl({
  initialTab,
  triggerLabel,
  triggerVariant,
  triggerSize,
  triggerClassName,
}: AuthControlProps) {
  const { session, isAuthenticated, demoUser, demoSignIn, demoSignOut, demoAvatarUrl } = useAuth();
  const { currentPlan, planConfig } = usePlan();
  const { toast } = useToast();
  const navigate = useNavigate();

  const [open, setOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<AuthTab>('password');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [username, setUsername] = useState('');
  const [busy, setBusy] = useState(false);
  const [planPopoverOpen, setPlanPopoverOpen] = useState(false);

  const redirectTo = useMemo(() => {
    return `${window.location.origin}/app`;
  }, []);

  const avatarSrc = useMemo(() => {
    // In demo mode, use the custom avatar
    if (SUPABASE_DISABLED && demoUser) {
      return demoAvatarUrl;
    }
    const m = (session?.user?.user_metadata ?? {}) as Record<string, unknown>;
    const metaUrl = typeof m.avatar_url === 'string' ? m.avatar_url : null;
    const fallback = `https://api.dicebear.com/7.x/identicon/svg?seed=${encodeURIComponent(session?.user?.email ?? 'user')}`;
    return metaUrl ?? fallback;
  }, [session, demoUser, demoAvatarUrl]);

  const displayName = useMemo(() => {
    if (SUPABASE_DISABLED && demoUser) return demoUser.username;
    const m = session?.user?.user_metadata as Record<string, unknown> | undefined;
    return (typeof m?.username === 'string' ? m.username : session?.user?.email) ?? 'User';
  }, [session, demoUser]);

  const PlanIcon = useMemo<LucideIcon>(() => {
    const icons: Record<PlanTier, LucideIcon> = {
      free: Gift,
      essential: Zap,
      architect: Sparkles,
      elite: Crown,
    };
    return icons[currentPlan];
  }, [currentPlan]);

  const startCheckout = async (plan: Exclude<PlanTier, 'free'>) => {
    setBusy(true);
    try {
      const origin = window.location.origin;
      const { url } = await createCheckoutSession({
        plan,
        interval: 'year',
        success_url: `${origin}/account?checkout=success`,
        cancel_url: `${origin}/account?checkout=cancel`,
      });
      window.location.assign(url);
    } catch (e: any) {
      toast({ title: 'Checkout failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onManageBilling = async () => {
    setBusy(true);
    try {
      const { url } = await createBillingPortalSession();
      window.location.assign(url);
    } catch (e: any) {
      toast({ title: 'Billing portal failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  // ── Demo / Dev mode login ──────────────────────────────────
  const onDemoSignIn = () => {
    if (!email.trim()) {
      toast({ title: 'Email required', description: 'Please enter an email to continue.', variant: 'destructive' });
      return;
    }
    demoSignIn(email.trim(), username.trim() || undefined);
    toast({
      title: 'Welcome to NEURAX!',
      description: `Signed in as ${username.trim() || email.trim().split('@')[0]}`,
    });
    setOpen(false);
    setEmail('');
    setPassword('');
    setUsername('');
  };

  // ── Supabase auth (prod mode) ──────────────────────────────
  const onSignUp = async () => {
    setBusy(true);
    try {
      if (password !== confirmPassword) {
        toast({
          title: 'Passwords do not match',
          description: 'Please re-type your password.',
          variant: 'destructive',
        });
        return;
      }
      const { error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          emailRedirectTo: redirectTo,
          data: {
            username: username || email.split('@')[0],
            avatar_url: `https://api.dicebear.com/7.x/bottts/svg?seed=${encodeURIComponent(email)}`,
          },
        },
      });
      if (error) throw error;
      toast({
        title: 'Account created',
        description: 'If email confirmation is enabled, check your inbox to confirm your account.',
      });
      setOpen(false);
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
      setOpen(false);
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
      setOpen(false);
    } catch (e: any) {
      toast({ title: 'Magic link failed', description: String(e?.message ?? e), variant: 'destructive' });
    } finally {
      setBusy(false);
    }
  };

  const onOAuth = async (provider: 'google' | 'github') => {
    setBusy(true);
    try {
      const { error } = await supabase.auth.signInWithOAuth({
        provider,
        options: { redirectTo },
      });
      if (error) throw error;
    } catch (e: any) {
      toast({ title: 'OAuth sign in failed', description: String(e?.message ?? e), variant: 'destructive' });
      setBusy(false);
    }
  };

  // ── Authenticated state ────────────────────────────────────
  if (isAuthenticated) {
    return (
      <div className="flex items-center gap-2">
        {/* Plan badge */}
        <Popover open={planPopoverOpen} onOpenChange={setPlanPopoverOpen}>
          <PopoverTrigger asChild>
            <button
              type="button"
              className={
                `h-8 px-2 rounded-md border border-white/20 hover:border-white/40 transition-colors text-[10px] font-mono uppercase tracking-wider flex items-center gap-1.5 ` +
                planConfig.badge
              }
              disabled={busy}
              aria-label="Open plans"
            >
              <span className="text-current"><PlanIcon className="w-3.5 h-3.5" /></span>
              {planConfig.displayName}
            </button>
          </PopoverTrigger>
          <PopoverContent className="w-72 p-2" align="end" alignOffset={44} sideOffset={6}>
            <div className="px-2 py-1.5">
              <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">Current plan</div>
              <div className="text-sm font-semibold" style={{ color: planConfig.color }}>{planConfig.name}</div>
            </div>

            <div className="mt-2 space-y-1">
              {(Object.keys(PLAN_CONFIGS) as PlanTier[])
                .filter((tier) => tier !== currentPlan)
                .map((tier) => {
                  const cfg = PLAN_CONFIGS[tier];
                  const isPaid = tier !== 'free';
                  const Icon = tier === 'free' ? Gift : tier === 'essential' ? Zap : tier === 'architect' ? Sparkles : Crown;

                  return (
                    <button
                      key={tier}
                      type="button"
                      disabled={busy}
                      className={
                        `w-full flex items-center justify-between rounded-md border px-3 py-2 text-left transition-colors border-white/20 hover:bg-white/5 ` +
                        cfg.badge
                      }
                      onClick={() => {
                        if (!isPaid) {
                          toast({ title: 'Free plan', description: 'Free plan is active by default.' });
                          setPlanPopoverOpen(false);
                          return;
                        }
                        setPlanPopoverOpen(false);
                        void startCheckout(tier);
                      }}
                      aria-label={`Upgrade to ${cfg.name}`}
                    >
                      <div className="min-w-0 flex items-center gap-2">
                        <Icon className="w-4 h-4" />
                        <div className="min-w-0">
                          <div className="text-xs font-semibold truncate" style={{ color: cfg.color }}>{cfg.displayName}</div>
                          <div className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground truncate">
                            {isPaid ? 'Upgrade' : 'Free'}
                          </div>
                        </div>
                      </div>
                    </button>
                  );
                })}
            </div>

            {currentPlan !== 'free' && (
              <div className="mt-2 pt-2 border-t border-white/20">
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-white/20"
                  onClick={() => {
                    setPlanPopoverOpen(false);
                    void onManageBilling();
                  }}
                  disabled={busy}
                >
                  Manage billing
                </Button>
              </div>
            )}
          </PopoverContent>
        </Popover>

        {/* User avatar + name */}
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="h-8 w-8 rounded-full overflow-hidden border border-white/20 hover:border-white/40 transition-colors flex-shrink-0"
            onClick={() => navigate('/account')}
            aria-label="Open account"
            disabled={busy}
          >
            <img src={avatarSrc} alt="avatar" className="h-full w-full object-cover" />
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

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-md bg-[#0c0c1a] border border-white/10 shadow-2xl">
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
            /* ── Demo Login (dev mode) ── */
            <div className="space-y-4 mt-2">
              <div>
                <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
                  Your Name
                </label>
                <Input
                  placeholder="e.g. AI Explorer"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
              </div>
              <div>
                <label className="text-[10px] font-mono uppercase tracking-wider text-white/30 mb-1.5 block">
                  Email (optional)
                </label>
                <Input
                  placeholder="you@example.com"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
              </div>

              <div className="pt-2 space-y-2">
                <Button
                  className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold h-11"
                  onClick={onDemoSignIn}
                >
                  <Zap className="w-4 h-4 mr-2" />
                  Enter Platform
                </Button>
                <p className="text-[10px] text-center text-white/20">
                  Demo mode — no account required
                </p>
              </div>
            </div>
          ) : (
            /* ── Supabase Auth (prod mode) ── */
            <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)} className="w-full">
              <TabsList className="grid w-full grid-cols-4 bg-white/5">
                <TabsTrigger value="password" className="text-xs">Password</TabsTrigger>
                <TabsTrigger value="signup" className="text-xs">Sign up</TabsTrigger>
                <TabsTrigger value="magic" className="text-xs">Magic link</TabsTrigger>
                <TabsTrigger value="oauth" className="text-xs">OAuth</TabsTrigger>
              </TabsList>

              <TabsContent value="password" className="space-y-3 mt-3">
                <Input
                  placeholder="Email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
                <Input
                  placeholder="Password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
                <Button className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold" onClick={onPasswordSignIn} disabled={busy || !email || !password}>
                  <LogIn className="w-4 h-4 mr-2" />
                  Sign in
                </Button>
              </TabsContent>

              <TabsContent value="signup" className="space-y-3 mt-3">
                <Input
                  placeholder="Username (optional)"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
                <Input
                  placeholder="Email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
                <Input
                  placeholder="Password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
                <Input
                  placeholder="Confirm password"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
                <Button
                  className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold"
                  onClick={onSignUp}
                  disabled={busy || !email || !password || !confirmPassword}
                >
                  <LogIn className="w-4 h-4 mr-2" />
                  Create account
                </Button>
              </TabsContent>

              <TabsContent value="magic" className="space-y-3 mt-3">
                <Input
                  placeholder="Email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="bg-white/5 border-white/10 text-white placeholder:text-white/20"
                />
                <Button className="w-full bg-white text-[#0c0c1a] hover:bg-white/90 font-semibold" onClick={onMagicLink} disabled={busy || !email}>
                  <Mail className="w-4 h-4 mr-2" />
                  Send magic link
                </Button>
              </TabsContent>

              <TabsContent value="oauth" className="space-y-2 mt-3">
                <Button className="w-full bg-white/5 border-white/10 text-white hover:bg-white/10" variant="outline" onClick={() => onOAuth('google')} disabled={busy}>
                  <Mail className="w-4 h-4 mr-2" />
                  Continue with Google
                </Button>
                <Button className="w-full bg-white/5 border-white/10 text-white hover:bg-white/10" variant="outline" onClick={() => onOAuth('github')} disabled={busy}>
                  <Github className="w-4 h-4 mr-2" />
                  Continue with GitHub
                </Button>
              </TabsContent>
            </Tabs>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
