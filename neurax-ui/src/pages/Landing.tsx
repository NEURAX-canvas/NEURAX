import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Github, Sparkles, Cpu, Brain, BarChart3, Shield, Zap, Layers, Boxes, Network, Check, Book } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { AuthControl } from '@/components/auth/AuthControl.tsx';
import { NeuraxLogo } from '@/components/brand/NeuraxLogo.tsx';
import { useApiKey } from '@/contexts/ApiKeyContext.tsx';
import { NeuralParticles } from '@/components/landing/NeuralParticles.tsx';
import SpiderLogo from '@/components/landing/SpiderLogo.tsx';
import { ScreenshotCarousel } from '@/components/landing/ScreenshotCarousel.tsx';
import { ComparisonTable } from '@/components/landing/ComparisonTable.tsx';
import { UseCaseGrid, SocialProofBanner } from '@/components/landing/UseCaseGrid.tsx';
import { FAQAccordion } from '@/components/landing/FAQAccordion.tsx';
import { ScrollProgressBar } from '@/components/landing/ScrollProgressBar.tsx';

const ANIM_STYLES = (
  <style>{`
    @keyframes nxFloat {
      0%, 100% { transform: translateY(0px); }
      50% { transform: translateY(-8px); }
    }
    @keyframes nxPulse {
      0%, 100% { opacity: 0.06; transform: scale(1); }
      50% { opacity: 0.12; transform: scale(1.05); }
    }
    @keyframes nxShimmer {
      0% { background-position: -200% center; }
      100% { background-position: 200% center; }
    }
    @keyframes nxFadeUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @keyframes nxGlow {
      0%, 100% { box-shadow: 0 0 20px rgba(215,153,33,0.15); }
      50% { box-shadow: 0 0 40px rgba(215,153,33,0.3), 0 0 60px rgba(214,93,14,0.1); }
    }
    .nx-float { animation: nxFloat 4s ease-in-out infinite; }
    .nx-float-delayed { animation: nxFloat 5s ease-in-out 1s infinite; }
    .nx-pulse-glow { animation: nxPulse 5s ease-in-out infinite; }
    .nx-shimmer {
      background: linear-gradient(90deg, #d79921 0%, #83a598 25%, #d79921 50%, #83a598 75%, #d79921 100%);
      background-size: 200% auto;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: nxShimmer 4s linear infinite;
    }
    .nx-fade-up { animation: nxFadeUp 0.8s ease-out forwards; opacity: 0; }
    .nx-fade-up-1 { animation: nxFadeUp 0.8s ease-out 0.1s forwards; opacity: 0; }
    .nx-fade-up-2 { animation: nxFadeUp 0.8s ease-out 0.2s forwards; opacity: 0; }
    .nx-fade-up-3 { animation: nxFadeUp 0.8s ease-out 0.3s forwards; opacity: 0; }
    .nx-fade-up-4 { animation: nxFadeUp 0.8s ease-out 0.4s forwards; opacity: 0; }
    .nx-btn-glow:hover { animation: nxGlow 2s ease-in-out infinite; }
  `}</style>
);

const FONT_LINK = (
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
);

const C = {
  bg: '#1d2021',
  card: '#282828',
  border: '#3c3836',
  text: '#ebdbb2',
  muted: '#a89984',
  faint: '#7c6f64',
  accent: '#d79921',
  orange: '#d65d0e',
  green: '#98971a',
  cyan: '#83a598',
  purple: '#b16286',
};

/* ═══════════════════════════════════════════════════════
   NAVBAR
   ═══════════════════════════════════════════════════════ */
const Navbar = () => (
  <header
    className="fixed top-0 left-0 right-0 z-50 border-b"
    style={{
      backgroundColor: 'rgba(29,32,33,0.85)',
      backdropFilter: 'blur(16px)',
      borderColor: C.border,
    }}
  >
    <div className="mx-auto max-w-[1300px] px-6 h-[56px] flex items-center justify-between">
      <Link to="/" className="flex items-center gap-2.5 group shrink-0">
        <NeuraxLogo size={20} showText={false} variant="mark" />
        <span className="text-[15px] font-bold tracking-[-0.02em]" style={{ color: C.text }}>NEURAX</span>
        <span className="hidden sm:inline text-[9px] font-mono uppercase tracking-[0.2em]" style={{ color: C.faint }}>
          Analytic Compiler
        </span>
      </Link>

      <nav className="hidden md:flex items-center gap-8">
        {[
          { label: 'Features', href: '#features' },
          { label: 'Architectures', href: '#architectures' },
          { label: 'Pipeline', href: '#pipeline' },
        ].map((item) => (
          <a
            key={item.label} href={item.href}
            className="text-[13px] font-medium transition-colors duration-150"
            style={{ color: C.muted }}
            onMouseEnter={(e) => e.currentTarget.style.color = C.text}
            onMouseLeave={(e) => e.currentTarget.style.color = C.muted}
          >
            {item.label}
          </a>
        ))}
        <a
          href="https://github.com/rustnew/NEURAX"
          target="_blank" rel="noopener noreferrer"
          className="flex items-center gap-1.5 text-[13px] font-medium transition-colors duration-150"
          style={{ color: C.faint }}
          onMouseEnter={(e) => e.currentTarget.style.color = C.accent}
          onMouseLeave={(e) => e.currentTarget.style.color = C.faint}
        >
          <Github size={14} />
          GitHub
        </a>
      </nav>

      <div className="flex items-center gap-3">
        <AuthControl triggerLabel="Sign in" triggerSize="sm" triggerVariant="ghost" />
        <Button asChild size="sm" className="h-[34px] px-5 text-[13px] font-semibold rounded-[8px] border-0 transition-all duration-150 hover:scale-[1.02]" style={{ background: `linear-gradient(135deg, ${C.accent}, ${C.orange})`, color: '#1d2021' }}>
          <Link to="/app">
            Launch Studio
            <ArrowRight className="w-3.5 h-3.5 ml-1.5" />
          </Link>
        </Button>
      </div>
    </div>
  </header>
);

/* ═══════════════════════════════════════════════════════
   HERO — Live, Dynamic, Neural
   ═══════════════════════════════════════════════════════ */
const AnimatedCounter = ({ target, suffix = '' }: { target: number; suffix?: string }) => {
  const [val, setVal] = useState(0);
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        let start = 0;
        const duration = 1500;
        const step = 16;
        const increment = target / (duration / step);
        const timer = setInterval(() => {
          start += increment;
          if (start >= target) { clearInterval(timer); setVal(target); }
          else setVal(Math.floor(start));
        }, step);
        observer.disconnect();
      }
    }, { threshold: 0.3 });
    observer.observe(el);
    return () => observer.disconnect();
  }, [target]);
  return <span ref={ref}>{val}{suffix}</span>;
};

const Hero = () => (
  <section className="relative min-h-[90vh] flex flex-col items-center justify-center overflow-hidden" style={{ backgroundColor: C.bg }}>
    <SpiderLogo />
    <NeuralParticles />

    <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/3 w-[800px] h-[800px] rounded-full pointer-events-none nx-pulse-glow"
      style={{ background: `radial-gradient(circle, ${C.accent} 0%, transparent 60%)` }}
    />
    <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] rounded-full pointer-events-none nx-float-delayed"
      style={{ background: `radial-gradient(circle, ${C.cyan} 0%, transparent 60%)`, opacity: 0.06 }}
    />
    <div className="absolute top-1/4 left-[10%] w-[300px] h-[300px] rounded-full pointer-events-none nx-float"
      style={{ background: `radial-gradient(circle, ${C.orange} 0%, transparent 60%)`, opacity: 0.04 }}
    />

    <div className="relative z-10 mx-auto max-w-[1000px] px-6 pt-32 pb-20 text-center">
      <div className="nx-fade-up inline-flex items-center gap-2 mb-8 px-4 py-1.5 rounded-full" style={{ backgroundColor: `${C.accent}15`, border: `1px solid ${C.accent}30` }}>
        <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: C.accent }} />
        <span className="text-[11px] font-mono tracking-[0.1em] uppercase" style={{ color: C.accent }}>
          AI Architecture Compiler · Research-Grade · Open Source
        </span>
      </div>

      <h1 className="nx-fade-up-1 text-[48px] sm:text-[60px] lg:text-[76px] font-bold leading-[1.05] tracking-[-0.03em] mb-6 text-balance" style={{ color: C.text }}>
        Design AI that{' '}
        <span className="nx-shimmer">actually ships.</span>
      </h1>

      <p className="nx-fade-up-2 max-w-[680px] mx-auto text-[17px] leading-[1.6] mb-10 text-balance" style={{ color: C.muted }}>
        NEURAX is a free, open-source architecture compiler that <span className="font-semibold" style={{ color: C.text }}>covers 11 architecture families and 680+ configurable block types</span> —
        from Transformers to SSMs, Diffusion to SNNs. Research-grade, deterministic, and interoperable with every major framework.
      </p>

      <div className="nx-fade-up-3 flex flex-wrap items-center justify-center gap-4 mb-16">
        <Button asChild size="lg"
          className="nx-btn-glow h-[48px] px-8 text-[15px] font-semibold rounded-[10px] border-0 transition-all duration-150 hover:scale-[1.02] hover:shadow-lg"
          style={{ background: `linear-gradient(135deg, ${C.accent}, ${C.orange})`, color: '#1d2021' }}
        >
          <Link to="/app">
            Start Building Free
            <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
          </Link>
        </Button>
        <Button asChild variant="outline" size="lg"
          className="h-[48px] px-8 text-[15px] font-medium rounded-[10px] transition-all duration-150 hover:scale-[1.02] hover:bg-white/5"
          style={{ backgroundColor: C.card, borderColor: C.border, color: C.text }}
        >
          <a href="#pipeline">See How It Works</a>
        </Button>
      </div>

      <div className="nx-fade-up-4 grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-[760px] mx-auto">
        {[
          { display: true, target: 11, suffix: '', label: 'Architecture Families' },
          { display: true, target: 680, suffix: '+', label: 'Configurable Block Types' },
          { display: true, target: 55, suffix: '+', label: 'Metrics per Analysis' },
          { display: '<50ms', label: 'End-to-End Analysis' },
        ].map((s) => (
          <div key={s.label}
            className="group rounded-[10px] p-4 text-center transition-all duration-200 hover:scale-[1.03] hover:-translate-y-0.5"
            style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}
          >
            <div className="text-[28px] font-bold tracking-[-0.5px] leading-none mb-1" style={{ color: C.accent }}>
              {typeof s.display === 'string' ? s.display : <><AnimatedCounter target={s.target} suffix={s.suffix} /></>}
            </div>
            <div className="text-[11px]" style={{ color: C.muted }}>{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   PROBLEM SECTION
   ═══════════════════════════════════════════════════════ */
const PROBLEMS = [
  { icon: Cpu, title: 'Architecture roulette', desc: 'Teams commit months to architectures that fail at step one — because nobody runs the numbers first.', stat: '$2M+', statLabel: 'average cost of a failed training run' },
  { icon: Brain, title: 'Blind design decisions', desc: 'Transformer vs Mamba? 7B vs 13B? Without instant analysis, every architecture choice is a gamble.', stat: '73%', statLabel: 'of models need redesign after first training' },
  { icon: BarChart3, title: 'Production fire drills', desc: 'Latency spikes, VRAM limits, hardware mismatches — discovered only after deployment.', stat: '5×', statLabel: 'cost overrun when issues surface in production' },
];

const ProblemSection = () => (
  <section style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[1100px] px-6 py-24">
      <div className="text-center mb-14">
        <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4" style={{ color: C.text }}>
          Most AI models never{' '}
          <span style={{ color: C.muted }}>make it to production.</span>
        </h2>
        <p className="text-[16px] max-w-[600px] mx-auto leading-[1.6]" style={{ color: C.muted }}>
          Not because the ideas are bad — but because teams have no way to predict cost, memory, or
          feasibility before committing GPU hours. NEURAX closes that gap.
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {PROBLEMS.map((item) => {
          const Icon = item.icon;
          return (
            <div key={item.title}
              className="rounded-[10px] p-6 transition-all duration-200 hover:scale-[1.02]"
              style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}
            >
              <div className="w-10 h-10 rounded-[8px] flex items-center justify-center mb-4" style={{ backgroundColor: `${C.accent}15` }}>
                <Icon className="w-5 h-5" style={{ color: C.accent }} />
              </div>
              <h3 className="text-[16px] font-semibold mb-2" style={{ color: C.text }}>
                {item.title}
              </h3>
              <p className="text-[13px] leading-[1.6] mb-5" style={{ color: C.muted }}>
                {item.desc}
              </p>
              <div className="pt-4" style={{ borderTop: `1px solid ${C.border}` }}>
                <div className="text-[26px] font-bold tracking-[-0.02em]" style={{ color: C.accent }}>
                  {item.stat}
                </div>
                <div className="text-[11px] mt-1" style={{ color: C.faint }}>{item.statLabel}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   FEATURES
   ═══════════════════════════════════════════════════════ */
const FEATURES = [
  { icon: Layers, title: 'Instant Architecture Intelligence', desc: 'Design any neural topology and get instant FLOPs, memory, and latency predictions — no GPU required.' },
  { icon: Shield, title: 'Production-Ready Analysis', desc: 'Deterministic cost engine built in Rust. Know exactly how your model will behave in production before writing a single training script.' },
  { icon: Zap, title: 'Inference Quality Prediction', desc: 'Predict inference stability, hallucination risk, and sampling behavior before serving your first request.' },
  { icon: Boxes, title: 'Time Machine', desc: 'Roll back architecture states, compare hardware profiles, and version your designs with complete analytical history.' },
  { icon: Network, title: '10-Pass Compiler Pipeline', desc: 'A deterministic 10-pass IR pipeline transforms your architecture into a complete analytical report in under 50ms.' },
  { icon: Sparkles, title: 'Neurax Agent', desc: 'AI-powered co-pilot that helps you design, analyze, and optimize architectures. Connect your preferred model — your key, your privacy.' },
];

const FeaturesSection = () => (
  <section id="features" style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[1100px] px-6 py-24">
      <div className="text-center mb-14">
        <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4" style={{ color: C.text }}>
          Research-grade analysis.{' '}
          <span style={{ color: C.muted }}>Deterministic by design.</span>
        </h2>
        <p className="text-[16px] max-w-[580px] mx-auto leading-[1.6]" style={{ color: C.muted }}>
          Every architecture family ever published — from foundational to experimental — can be modelled, analyzed, and exported
          through NEURAX. <span className="font-semibold" style={{ color: C.text }}>Interoperable by default, reproducible every time.</span>
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {FEATURES.map((f) => {
          const Icon = f.icon;
          return (
            <div
              key={f.title}
              className="group rounded-[10px] p-5 transition-all duration-200 hover:-translate-y-0.5"
              style={{
                backgroundColor: C.card,
                border: `1px solid ${C.border}`,
              }}
              onMouseEnter={(e) => e.currentTarget.style.borderColor = C.accent + '60'}
              onMouseLeave={(e) => e.currentTarget.style.borderColor = C.border}
            >
              <div className="w-8 h-0.5 rounded-full mb-4 bg-gradient-to-r from-[#d79921] to-[#83a598] opacity-70" />
              <div className="w-9 h-9 rounded-[7px] flex items-center justify-center mb-3" style={{ backgroundColor: `${C.accent}15` }}>
                <Icon className="w-[18px] h-[18px]" style={{ color: C.accent }} />
              </div>
              <h3 className="text-[15px] font-semibold mb-1.5" style={{ color: C.text }}>{f.title}</h3>
              <p className="text-[13px] leading-[1.6]" style={{ color: C.muted }}>{f.desc}</p>
            </div>
          );
        })}
      </div>
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   ARCHITECTURES
   ═══════════════════════════════════════════════════════ */
const FAMILIES = [
  { title: 'Transformer / LLM', color: '#458588', ops: ['MHA / GQA / MLA', 'Flash Attention', 'SwiGLU FFN', 'RoPE / ALiBi', 'RMSNorm', 'LoRA / DoRA'], models: ['GPT-4', 'LLaMA 3', 'BERT', 'Mistral', 'DeepSeek'], blocks: '~66' },
  { title: 'CNN / Vision', color: '#83a598', ops: ['Conv2D/3D', 'Depthwise / Deformable', 'BatchNorm / GroupNorm', 'ResNet / DenseNet', 'ViT Patch Embed', 'U-Net / FPN'], models: ['ResNet-152', 'YOLOv8', 'EfficientNet', 'U-Net'], blocks: '~116' },
  { title: 'State Space (Mamba)', color: '#8ec07c', ops: ['S4 / S6 Block', 'Selective Scan', 'LRU / H3 / Hyena', 'VSS / Mamba-2', 'SSM Discretize', 'Gated MLP'], models: ['Mamba-2', 'S4', 'H3', 'RWKV'], blocks: '~97' },
  { title: 'MoE', color: '#b16286', ops: ['Top-K Router', 'Expert-Choice', 'Noisy / PEER', 'FFN / Gated Expert', 'Load Balancing', 'Fine-Grained MoE'], models: ['Mixtral 8×7B', 'Switch Transformer', 'DeepSeek-V2'], blocks: '~67' },
  { title: 'Diffusion Models', color: '#d65d0e', ops: ['U-Net Denoiser', 'DiT / MMDiT', 'Cross Attention', 'CFG / IP-Adapter', 'DDPM / DDIM', 'Flow Matching'], models: ['SD 3', 'DALL-E 3', 'Imagen', 'Flux'], blocks: '~75' },
  { title: 'GAN', color: '#fabd2f', ops: ['Generator / Critic', 'StyleGAN Synthesis', 'BigGAN ResBlock', 'PatchGAN', 'Adversarial Loss', 'CycleGAN'], models: ['StyleGAN 3', 'BigGAN', 'CycleGAN', 'ESRGAN'], blocks: '~82' },
  { title: 'GNN', color: '#fb4934', ops: ['GCNConv / GATConv', 'SAGEConv / GIN', 'Message Passing', 'TopK / SAG Pooling', 'RGCN / HGT', 'EdgeConv / DNA'], models: ['GCN', 'GraphSAGE', 'GAT'], blocks: '~46' },
  { title: 'RNN / LSTM / GRU', color: '#fe8019', ops: ['LSTM / GRU Cell', 'BiLSTM / BiGRU', 'ConvLSTM / ST-LSTM', 'SRU / QRNN', 'Peephole / IndyLSTM', 'Seq2Seq Attention'], models: ['LSTM', 'GRU', 'ConvLSTM'], blocks: '~70' },
  { title: 'Reinforcement Learning', color: '#cc241d', ops: ['Policy Head', 'Value Head', 'Actor-Critic', 'Dueling DQN', 'Advantage Stream', 'PPO Stream'], models: ['PPO', 'DQN', 'A2C'], blocks: '~12' },
  { title: 'Spiking Neural Networks', color: '#d3869b', ops: ['LIF Neuron', 'Spike Encoder', 'Rate Encoding', 'STDP Synapse', 'Synaptic Layer', 'Neuromorphic'], models: ['SNN Classifier', 'Neuromorphic'], blocks: '~12' },
  { title: 'Experimental', color: '#8f3f71', ops: ['Hybrid SSM+Attention', 'Custom Blocks', 'Research Primitives', 'Architecture Proposals', 'Experimental Macros', 'Custom Formulas'], models: ['Jamba', 'Lumina-Next', 'Custom Architectures'], blocks: '∞' },
];

const ArchitecturesSection = () => (
  <section id="architectures" style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[1200px] px-6 py-24">
      <div className="text-center mb-14">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6" style={{ backgroundColor: `${C.accent}15`, border: `1px solid ${C.accent}30` }}>
          <span className="text-[11px] font-mono tracking-[0.1em] uppercase" style={{ color: C.accent }}>
            Research-Grade · 100% Reproducible
          </span>
        </div>
        <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4" style={{ color: C.text }}>
          11 architecture families.{' '}
          <span style={{ color: C.muted }}>680+ configurable blocks.</span>
        </h2>
        <p className="text-[16px] max-w-[600px] mx-auto leading-[1.6]" style={{ color: C.muted }}>
          Every major neural architecture family — from Transformers to Spiking Neural Networks — is fully representable
          on the NEURAX canvas. Design any model, in any framework, with full analytical fidelity.
        </p>
        <div className="flex items-center justify-center gap-6 mt-8 text-[12px] font-mono" style={{ color: C.faint }}>
          <span>Torch · ONNX · Rust/Burn · Triton · MLIR · Megatron-LM · JSON</span>
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {FAMILIES.map((f) => (
          <div key={f.title}
            className="rounded-[10px] p-4 transition-all duration-200 hover:-translate-y-0.5"
            style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-[14px] font-semibold" style={{ color: C.text }}>{f.title}</h3>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded" style={{ backgroundColor: `${f.color}20`, color: f.color }}>
                {f.blocks}
              </span>
            </div>
            <div className="flex flex-wrap gap-1 mb-3">
              {f.ops.slice(0, 4).map((op) => (
                <span key={op}
                  className="text-[9px] px-1.5 py-0.5 rounded-[3px] font-mono"
                  style={{ color: f.color, backgroundColor: `${f.color}15` }}
                >
                  {op}
                </span>
              ))}
              {f.ops.length > 4 && (
                <span className="text-[9px] font-mono" style={{ color: C.faint }}>+{f.ops.length - 4}</span>
              )}
            </div>
            <div style={{ height: '1px', backgroundColor: C.border, margin: '8px 0' }} />
            <div className="flex flex-wrap gap-x-3 gap-y-0.5">
              {f.models.map((m) => (
                <span key={m} className="text-[11px]" style={{ color: C.muted }}>{m}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   PIPELINE
   ═══════════════════════════════════════════════════════ */
const STEPS = [
  { num: '01', title: 'Define your architecture', desc: 'Use the visual canvas, write a topology.json, or import from HuggingFace. Design any model in seconds.', icon: '○' },
  { num: '02', title: 'Run the 10-pass analysis', desc: 'NEURAX compiles your design through a deterministic pipeline — 10 IR passes, 55+ metrics, under 50ms.', icon: '◉' },
  { num: '03', title: 'Ship with a full report', desc: 'Get FLOPs, VRAM, latency, per-layer breakdown, OOM risk, hardware fit, and cost projections — before spending a cent.', icon: '◎' },
];

const PipelineSection = () => (
  <section id="pipeline" style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[900px] px-6 py-24">
      <div className="text-center mb-14">
        <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4" style={{ color: C.text }}>
          Design. Analyze. Export.
        </h2>
        <p className="text-[16px] max-w-[560px] mx-auto leading-[1.6]" style={{ color: C.muted }}>
          10 compiler passes. 55+ metrics. Export to PyTorch, ONNX, Rust/Burn, Triton, MLIR, or Megatron-LM.
          <span className="font-semibold" style={{ color: C.text }}> Zero GPU time. Fully deterministic.</span>
        </p>
      </div>
      <div className="relative">
        <div className="absolute left-[23px] top-0 bottom-0 w-px" style={{ backgroundColor: `${C.accent}30` }} />
        <div className="space-y-6">
          {STEPS.map((s) => (
            <div key={s.num} className="flex items-start gap-6 relative">
              <div className="relative z-10 flex-shrink-0 w-[46px] h-[46px] rounded-full flex items-center justify-center text-[14px] font-bold font-mono"
                style={{
                  background: s.num === '03'
                    ? `linear-gradient(135deg, ${C.accent}, ${C.cyan})`
                    : `${C.accent}15`,
                  border: s.num === '03'
                    ? `1px solid ${C.accent}50`
                    : `1px solid ${C.accent}30`,
                  color: s.num === '03' ? C.bg : C.accent,
                }}
              >
                {s.num}
              </div>
              <div className="flex-1 min-w-0 pt-2.5">
                <h3 className="text-[16px] font-semibold mb-1.5" style={{ color: C.text }}>{s.title}</h3>
                <p className="text-[14px] leading-[1.6]" style={{ color: C.muted }}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   RUST + OPEN SOURCE
   ═══════════════════════════════════════════════════════ */
const AgentSetupPreview = () => {
  const { config: apiKeyConfig, isConfigured } = useApiKey();
  return (
    <div className="flex items-center gap-3 p-3 rounded-[8px]" style={{ backgroundColor: isConfigured ? `${C.green}10` : `${C.accent}10`, border: `1px solid ${isConfigured ? `${C.green}30` : `${C.accent}30`}` }}>
      <div className="w-8 h-8 rounded-[6px] flex items-center justify-center" style={{ backgroundColor: isConfigured ? `${C.green}15` : `${C.accent}15` }}>
        {isConfigured ? <Check className="w-4 h-4" style={{ color: C.green }} /> : <Sparkles className="w-4 h-4" style={{ color: C.accent }} />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-[13px] font-semibold" style={{ color: C.text }}>
          {isConfigured ? `${apiKeyConfig?.label || 'Connected'} Agent` : 'Neurax Agent'}
        </div>
        <div className="text-[11px]" style={{ color: C.faint }}>
          {isConfigured ? `Using ${apiKeyConfig?.model || 'default model'}` : 'Connect your AI provider'}
        </div>
      </div>
    </div>
  );
};

const RustSection = () => {
  const RUST_POINTS = [
    { label: 'Memory Safety', desc: 'No GC pauses, no undefined behavior. The engine is built for correctness from the ground up.' },
    { label: 'Blazing Performance', desc: 'Full analysis on 8B-param architectures in under 50ms. Benchmarked and battle-tested.' },
    { label: 'Deterministic by Design', desc: 'Same topology always produces the identical report. No randomness, no surprises.' },
    { label: 'Concurrent by Default', desc: 'Per-layer analysis parallelized with rayon. DAG resolved via petgraph.' },
  ];

  const AGENT_FEATURES = [
    { label: 'Intelligent Design Assistance', desc: 'Describe your use case, and Neurax Agent suggests optimal architectures with full cost breakdowns.' },
    { label: 'Your Model, Your Privacy', desc: 'Connect OpenAI, Anthropic, Google, or Mistral. Your API key stays in your browser — never on our servers.' },
    { label: 'Plain-English Explanations', desc: 'Complex metrics, bottlenecks, and trade-offs explained in language anyone can understand.' },
    { label: 'Automated Optimization', desc: 'Agent analyzes your design and recommends changes to reduce cost, improve speed, and fit your hardware.' },
  ];

  return (
    <section style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
      <div className="mx-auto max-w-[1100px] px-6 py-24">
        <div className="text-center mb-16">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full" style={{ backgroundColor: `${C.green}15`, border: `1px solid ${C.green}30` }}>
            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: C.green }} />
            <span className="text-[11px] font-mono tracking-[0.1em] uppercase" style={{ color: C.green }}>
              Enterprise-Grade · 100% Open Source
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center mb-24">
          <div>
            <div className="text-[11px] font-mono uppercase tracking-[0.1em] mb-4" style={{ color: C.accent }}>Deterministic Engine · 100% Rust</div>
            <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4 leading-[1.2]" style={{ color: C.text }}>
              10-pass IR compiler.{' '}
              <span style={{ color: C.muted }}>680+ block types.</span>
            </h2>
            <p className="text-[15px] leading-[1.6] mb-8" style={{ color: C.muted }}>
              Every analysis pass — parser, cost engine, dialect router, hardware mapper — runs in pure Rust.
              From 11 architecture families to production code: NEURAX is the only open-source compiler that
              covers the entire neural architecture design space with deterministic fidelity.
            </p>
            <div className="space-y-4">
              {RUST_POINTS.map((pt) => (
                <div key={pt.label} className="flex gap-3">
                  <div className="w-1.5 h-1.5 rounded-full mt-2 flex-shrink-0" style={{ backgroundColor: C.accent }} />
                  <div>
                    <div className="text-[13px] font-semibold" style={{ color: C.text }}>{pt.label}</div>
                    <div className="text-[13px] mt-0.5 leading-relaxed" style={{ color: C.muted }}>{pt.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="rounded-[10px] overflow-hidden" style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}>
              <div className="flex items-center gap-1.5 px-4 h-[34px]" style={{ backgroundColor: '#32302f', borderBottom: `1px solid ${C.border}` }}>
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: '#cc241d' }} />
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: C.accent }} />
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: C.green }} />
                <span className="ml-3 text-[10px] font-mono" style={{ color: C.faint }}>neurax-engine/src/analysis.rs</span>
              </div>
              <pre className="overflow-x-auto p-5 text-[11px] leading-[1.65] font-mono" style={{ color: C.muted }}>
                {`pub fn analyze(config: &ModelConfig) -> Result<AnalysisReport> {
  let ctx = NeuraxContext::new(config, gpu_db);
  let arch   = ArchitecturePass::build(&ctx)?;
  let graph  = GraphPass::build(&ctx, &arch)?;
  let tensor = TensorPass::build(&ctx, &graph)?;
  let ops    = OperatorPass::build(&ctx, &tensor)?;
  let compute = ComputePass::build(&ctx, &ops)?;
  let memory = MemoryPass::build(&ctx, &compute)?;
  let (parallel, hardware) = rayon::join(
    || ParallelismPass::build(&ctx, &memory),
    || HardwarePass::build(&ctx, &memory),
  );
  let cost  = CostPass::build(&ctx, &hardware)?;
  let report = ReportPass::build(&ctx, &all_metrics);
  Ok(report)
}`.split('\n').map((line, i) => (
                  <div key={i} className="flex gap-3">
                    <span className="select-none w-6 text-right flex-shrink-0 text-[10px]" style={{ color: C.faint }}>{i + 1}</span>
                    <span>{line || '\u00a0'}</span>
                  </div>
                ))}
              </pre>
            </div>
          </div>
        </div>

        <div id="agent" className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div>
            <div className="text-[11px] font-mono uppercase tracking-[0.1em] mb-4" style={{ color: C.cyan }}>Featured — Neurax Agent</div>
            <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4 leading-[1.2]" style={{ color: C.text }}>
              An AI co-pilot for{' '}
              <span className="bg-gradient-to-r from-[#83a598] to-[#d79921] bg-clip-text text-transparent">
                architecture design.
              </span>
            </h2>
            <p className="text-[15px] leading-[1.6] mb-8" style={{ color: C.muted }}>
              Bring your own API key to unlock Neurax Agent. Get intelligent design suggestions,
              plain-English metric explanations, and automated optimization recommendations — powered by your preferred AI model.
            </p>
            <div className="space-y-4">
              {AGENT_FEATURES.map((pt) => (
                <div key={pt.label} className="flex gap-3">
                  <div className="w-1.5 h-1.5 rounded-full mt-2 flex-shrink-0" style={{ backgroundColor: C.cyan }} />
                  <div>
                    <div className="text-[13px] font-semibold" style={{ color: C.text }}>{pt.label}</div>
                    <div className="text-[13px] mt-0.5 leading-relaxed" style={{ color: C.muted }}>{pt.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div>
            <div className="rounded-[10px] p-6" style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}>
              <h3 className="text-[15px] font-semibold mb-4" style={{ color: C.text }}>
                Bring your own API key
              </h3>
              <AgentSetupPreview />
              <div className="mt-4 space-y-2">
                {['OpenAI', 'Anthropic Claude', 'Google Gemini', 'Mistral AI'].map((provider) => (
                  <div key={provider} className="flex items-center gap-3 p-2 rounded-[6px]" style={{ backgroundColor: C.bg }}>
                    <Check className="w-3.5 h-3.5" style={{ color: C.green }} />
                    <span className="text-[12px]" style={{ color: C.muted }}>{provider}</span>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-[11px] leading-relaxed" style={{ color: C.faint }}>
                Your API key is stored locally in your browser. Never sent to our servers.
                <br />
                Fully private, fully secure.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

/* ═══════════════════════════════════════════════════════
   CLOSING CTA
   ═══════════════════════════════════════════════════════ */
const ClosingCTA = () => (
  <section style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[700px] px-6 py-32 text-center">
      <NeuraxLogo size={48} showText={false} variant="mark" className="mx-auto mb-6 opacity-30" />
      <h2 className="text-[42px] sm:text-[48px] font-bold tracking-[-0.03em] mb-4 leading-[1.1]" style={{ color: C.text }}>
        The future of AI is{' '}
        <br />
        <span className="bg-gradient-to-r from-[#d79921] to-[#83a598] bg-clip-text text-transparent">
          designed, not guessed.
        </span>
      </h2>
      <p className="text-[17px] mb-10 max-w-[560px] mx-auto" style={{ color: C.muted }}>
        NEURAX is the free, open-source architecture compiler for the AI research community.
        11 families. 680 blocks. Deterministic analysis. Research-grade. Free forever.
      </p>
      <Button asChild size="lg"
        className="h-[52px] px-10 text-[16px] font-semibold rounded-[10px] border-0 transition-all duration-150 hover:scale-[1.02]"
        style={{ background: `linear-gradient(135deg, ${C.accent}, ${C.orange})`, color: '#1d2021' }}
      >
        <Link to="/app">
          Start Building Free
          <ArrowRight className="w-4 h-4 ml-2" />
        </Link>
      </Button>
      <div className="mt-10 flex items-center justify-center gap-8 text-[10px] font-mono uppercase tracking-[0.1em]" style={{ color: C.faint }}>
        <span>11 Architecture Families</span>
        <span>680+ Block Types</span>
        <span>7 Export Formats</span>
        <span>100% Open Source</span>
      </div>
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════════════════ */
const Footer = () => (
  <footer style={{ backgroundColor: '#1a1c1a', borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[1200px] px-6 py-16">
      <div className="grid grid-cols-1 md:grid-cols-6 gap-10 mb-10">
        <div className="md:col-span-2">
          <div className="flex items-center gap-3 mb-4">
            <NeuraxLogo size={20} showText={false} variant="mark" />
            <span className="text-[16px] font-bold tracking-[-0.02em]" style={{ color: C.text }}>NEURAX</span>
          </div>
          <p className="text-[13px] leading-relaxed max-w-sm" style={{ color: C.muted }}>
            An open-source architecture compiler that covers the entire neural design space: 11 families, 680+ block types, 7 export formats, all deterministic.
          </p>
          <div className="mt-6 flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wider"
              style={{ backgroundColor: '#98971a20', color: '#98971a', border: '1px solid #98971a30' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#98971a' }} />
              MIT License
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-semibold uppercase tracking-wider"
              style={{ backgroundColor: '#83a59820', color: '#83a598', border: '1px solid #83a59830' }}>
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: '#83a598' }} />
              Free forever
            </span>
          </div>
          <div className="mt-6 flex items-center gap-2">
            <span className="text-[10px] font-mono uppercase tracking-[0.05em]" style={{ color: C.faint }}>Created by</span>
            <a href="https://github.com/rustnew" target="_blank" rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-[12px] font-medium transition-colors hover:brightness-125"
              style={{ color: C.muted }}
            >
              <Github size={12} />
              <span style={{ color: C.text }}>Martial-Christian</span>
            </a>
          </div>
        </div>
        <div>
          <h4 className="text-[10px] font-mono uppercase tracking-[0.12em] mb-4" style={{ color: C.faint }}>Product</h4>
          <div className="space-y-3">
            {[
            { label: 'Features', href: '#features' },
            { label: 'Architectures', href: '#architectures' },
            { label: 'Pipeline', href: '#pipeline' },
            { label: 'Studio', href: '/app' },
          ].map(({ label, href }) => (
            <Link key={label} to={href}
              className="block text-[13px] transition-colors hover:brightness-125"
              style={{ color: C.muted }}
            >
              {label}
            </Link>
          ))}
          </div>
        </div>
        <div>
          <h4 className="text-[10px] font-mono uppercase tracking-[0.12em] mb-4" style={{ color: C.faint }}>Architecture Families</h4>
          <div className="space-y-2">
            {FAMILIES.map((f) => (
              <div key={f.title} className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: f.color }} />
                <span className="text-[12px]" style={{ color: C.muted }}>{f.title}</span>
                <span className="text-[9px] font-mono ml-auto" style={{ color: C.faint }}>{f.blocks}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <h4 className="text-[10px] font-mono uppercase tracking-[0.12em] mb-4" style={{ color: C.faint }}>Connect</h4>
          <div className="space-y-3">
            <a href="https://github.com/rustnew/NEURAX" target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-2 text-[13px] transition-colors hover:brightness-125"
              style={{ color: C.muted }}
            >
              <Github size={13} />
              <span style={{ color: C.text }}>GitHub</span>
              <span className="text-[9px] font-mono px-1.5 py-0.5 rounded" style={{ backgroundColor: C.border, color: C.faint }}>rustnew/NEURAX</span>
            </a>
            <div className="flex items-center gap-2 text-[13px]" style={{ color: C.muted }}>
              <Book size={13} />
              <span>Documentation</span>
              <span className="text-[9px] font-mono" style={{ color: C.faint }}>(WIP)</span>
            </div>
            <div className="mt-4 pt-4" style={{ borderTop: `1px solid ${C.border}` }}>
              <div className="text-[10px] font-mono uppercase tracking-[0.1em] mb-2" style={{ color: C.faint }}>Community</div>
              <div className="flex gap-2">
                <a href="https://github.com/rustnew/NEURAX/discussions" target="_blank" rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-medium transition-all hover:brightness-125"
                  style={{ backgroundColor: C.border, color: C.text }}
                >
                  <Github size={11} /> Discussions
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4" style={{ borderTop: `1px solid ${C.border}` }}>
        <div className="text-[11px] font-mono" style={{ color: C.faint }}>
          © {new Date().getFullYear()} NEURAX — AI Architecture Intelligence Platform
        </div>
        <div className="flex items-center gap-4 text-[11px] font-mono" style={{ color: C.faint }}>
          <span>Built with Rust + TypeScript</span>
          <span className="w-1 h-1 rounded-full" style={{ backgroundColor: C.faint }} />
          <span>MIT — Free forever</span>
        </div>
      </div>
    </div>
  </footer>
);

/* ═══════════════════════════════════════════════════════
   SCREENSHOT SHOWCASE SECTION
   ═══════════════════════════════════════════════════════ */
const ScreenshotShowcaseSection = () => (
  <section id="showcase" style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[1100px] px-6 py-24">
      <div className="text-center mb-14">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6"
          style={{ backgroundColor: `${C.accent}15`, border: `1px solid ${C.accent}30` }}>
          <span className="text-[11px] font-mono tracking-[0.1em] uppercase" style={{ color: C.accent }}>
            Visual Tour · 6 Screenshots
          </span>
        </div>
        <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4" style={{ color: C.text }}>
          See NEURAX in action.{' '}
          <span style={{ color: C.muted }}>Every feature, live.</span>
        </h2>
        <p className="text-[16px] max-w-[580px] mx-auto leading-[1.6]" style={{ color: C.muted }}>
          From visual canvas to inference intelligence — explore the full NEURAX experience
          before writing a single line of code.
        </p>
      </div>
      <ScreenshotCarousel />
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   COMPARISON SECTION
   ═══════════════════════════════════════════════════════ */
const ComparisonSection = () => (
  <section id="comparison" style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[1100px] px-6 py-24">
      <div className="text-center mb-14">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6"
          style={{ backgroundColor: `${C.cyan}15`, border: `1px solid ${C.cyan}30` }}>
          <span className="text-[11px] font-mono tracking-[0.1em] uppercase" style={{ color: C.cyan }}>
            Compiler Comparison
          </span>
        </div>
        <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4" style={{ color: C.text }}>
          NEURAX vs ML Compilers{' '}
          <span style={{ color: C.muted }}>— Built for design, not just runtime.</span>
        </h2>
        <p className="text-[16px] max-w-[640px] mx-auto leading-[1.6]" style={{ color: C.muted }}>
          IREE, OpenXLA, and TVM are excellent runtime compilers for optimizing execution.
          NEURAX is an <span className="font-semibold" style={{ color: C.text }}>analytical compiler</span> that predicts
          cost, memory, and feasibility <span className="font-semibold" style={{ color: C.text }}>before training starts</span>.
        </p>
      </div>
      <ComparisonTable />
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   SOCIAL PROOF SECTION
   ═══════════════════════════════════════════════════════ */
const SocialProofSection = () => (
  <section id="social-proof" style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[1100px] px-6 py-24">
      <div className="text-center mb-14">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6"
          style={{ backgroundColor: `${C.green}15`, border: `1px solid ${C.green}30` }}>
          <span className="text-[11px] font-mono tracking-[0.1em] uppercase" style={{ color: C.green }}>
            Real-World Impact
          </span>
        </div>
        <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4" style={{ color: C.text }}>
          Built for researchers,{' '}
          <span style={{ color: C.muted }}>loved by engineers.</span>
        </h2>
        <p className="text-[16px] max-w-[580px] mx-auto leading-[1.6]" style={{ color: C.muted }}>
          From academic labs to enterprise ML teams — NEURAX accelerates the architecture design loop
          and saves real GPU budget.
        </p>
      </div>
      <div className="space-y-10">
        <SocialProofBanner />
        <UseCaseGrid />
      </div>
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   FAQ SECTION
   ═══════════════════════════════════════════════════════ */
const FAQSection = () => (
  <section id="faq" style={{ backgroundColor: C.bg, borderTop: `1px solid ${C.border}` }}>
    <div className="mx-auto max-w-[800px] px-6 py-24">
      <div className="text-center mb-14">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-6"
          style={{ backgroundColor: `${C.accent}15`, border: `1px solid ${C.accent}30` }}>
          <span className="text-[11px] font-mono tracking-[0.1em] uppercase" style={{ color: C.accent }}>
            FAQ · 6 Questions
          </span>
        </div>
        <h2 className="text-[34px] sm:text-[38px] font-bold tracking-[-0.02em] mb-4" style={{ color: C.text }}>
          Questions?{' '}
          <span style={{ color: C.muted }}>We've got answers.</span>
        </h2>
        <p className="text-[16px] max-w-[500px] mx-auto leading-[1.6]" style={{ color: C.muted }}>
          Everything you need to know about NEURAX — from pricing to predictions,
          integrations to architecture support.
        </p>
      </div>
      <FAQAccordion />
    </div>
  </section>
);

/* ═══════════════════════════════════════════════════════
   LANDING PAGE
   ═══════════════════════════════════════════════════════ */
export default function Landing() {
  return (
    <div className="min-h-screen selection:bg-[#d79921]/20 antialiased" style={{ backgroundColor: C.bg, color: C.text }}>
      {ANIM_STYLES}
      {FONT_LINK}
      <ScrollProgressBar />
      <Navbar />
      <main>
        <Hero />
        <ProblemSection />
        <FeaturesSection />
        <ScreenshotShowcaseSection />
        <ArchitecturesSection />
        <ComparisonSection />
        <PipelineSection />
        <SocialProofSection />
        <RustSection />
        <ClosingCTA />
        <FAQSection />
      </main>
      <Footer />
    </div>
  );
}
