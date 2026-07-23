import { Link } from 'react-router-dom';
import { ArrowRight, Github } from 'lucide-react';
import { Button } from '@/components/ui/button.tsx';
import { AuthControl } from '@/components/auth/AuthControl.tsx';
import { NeuraxAtomLogo } from '@/components/landing/NeuraxAtomLogo.tsx';

/* ─── Design System (Cursor.com style adapted for NEURAX) ──────
 *  Canvas:  #f7f7f4 (warm cream)
 *  Ink:     #26251e (warm near-black)
 *  Accent:  #3b82f6 (NEURAX blue — instead of cursor's orange)
 *  Cards:   #f2f1ed (cream-card)
 *  Hairlines: oklab warm borders
 *  Typography: Inter display, Georgia body, JetBrains Mono code
 *  Ethos:   "Warm minimalism meets analytical compiler"
 * ─────────────────────────────────────────────────────────────── */

/* ─── Inline style for Google Fonts ──────────────────────────── */
const FONT_LINK = (
  <link
    href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap"
    rel="stylesheet"
  />
);

/* ─── Navbar (fixed, cream) ──────────────────────────────────── */
const Navbar = () => (
  <header className="fixed top-0 left-0 right-0 z-50 bg-[#f7f7f4] h-[52px]">
    <div className="mx-auto max-w-[1300px] px-6 h-full flex items-center justify-between">
      {/* Logo */}
      <Link to="/" className="flex items-center gap-2.5 group shrink-0">
        <NeuraxAtomLogo size={24} />
        <span className="text-[14px] font-medium tracking-tight text-[#26251e] leading-none">
          NEURAX
        </span>
        <span className="hidden sm:inline text-[9px] font-mono text-[#a09c92] tracking-[0.2em] uppercase ml-1">
          Analytic Compiler
        </span>
      </Link>

      {/* Nav links */}
      <nav className="hidden md:flex items-center gap-7">
        {['Features', 'Architectures', 'Pipeline', 'Pricing'].map((label) => (
          <a
            key={label}
            href={label === 'Pricing' ? '#pricing' : `#${label.toLowerCase()}`}
            className="text-[12px] text-[#a09c92] hover:text-[#26251e] transition-colors duration-150 font-medium"
          >
            {label}
          </a>
        ))}
        <span className="text-[#e6e5e0]">·</span>
        <a
          href="https://github.com/Martial-Christian/Universal_Neurax"
          target="_blank"
          rel="noopener noreferrer"
          className="text-[12px] text-[#a09c92] hover:text-[#26251e] transition-colors duration-150 flex items-center gap-1"
        >
          <Github size={12} />
          GitHub
        </a>
      </nav>

      {/* Actions */}
      <div className="flex items-center gap-2.5">
        <AuthControl triggerLabel="Sign in" triggerSize="sm" triggerVariant="ghost" />
        <Button
          asChild
          size="sm"
          className="bg-[#f54e00] text-white hover:bg-[#d04200] rounded-[8px] px-5 text-[13px] font-medium h-[34px] transition-all duration-150"
        >
          <Link to="/app">
            Launch Studio
            <ArrowRight className="w-3.5 h-3.5 ml-1" />
          </Link>
        </Button>
      </div>
    </div>
  </header>
);

/* ─── Hero ────────────────────────────────────────────────────── */
const Hero = () => (
  <section className="relative pt-[52px] min-h-screen flex flex-col items-center justify-center overflow-hidden bg-[#f7f7f4]">
    {/* Subtle background — geometric dots */}
    <div className="absolute inset-0 pointer-events-none">
      <svg className="w-full h-full opacity-[0.035]" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="hero-dots" width="40" height="40" patternUnits="userSpaceOnUse">
            <circle cx="20" cy="20" r="1" fill="#26251e" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#hero-dots)" />
      </svg>
    </div>

    <div className="relative z-10 mx-auto max-w-[1000px] px-6 pt-28 pb-16 text-center">
      {/* Eyebrow */}
      <div className="inline-flex items-center gap-2 mb-8 px-4 py-1.5 rounded-full bg-[#e6e5e0]">
        <div className="w-1.5 h-1.5 rounded-full bg-[#3b82f6]" />
        <span className="text-[11px] font-mono text-[#807d72] tracking-[0.1em] uppercase">
          Analytical Compiler v0.1
        </span>
      </div>

      {/* Headline */}
      <h1 className="font-['Inter',system-ui,sans-serif] text-[48px] sm:text-[60px] lg:text-[72px] font-[400] leading-[1.1] tracking-[-1.44px] sm:tracking-[-1.8px] lg:tracking-[-2.16px] text-[#26251e] mb-6 text-balance">
        Know your model's cost
        <br />
        <span className="text-[#3b82f6]">before you train.</span>
      </h1>

      {/* Subhead */}
      <p className="max-w-[650px] mx-auto text-[17px] sm:text-[19px] font-['Georgia','Palatino','serif'] text-[#5a5852] leading-[1.5] mb-10 text-balance">
        NEURAX is the analytical compiler that predicts cost, memory, speed, and feasibility
        of any neural architecture — before a single GPU hour is spent.
      </p>

      {/* CTAs */}
      <div className="flex flex-wrap items-center justify-center gap-4 mb-20">
        <Button
          asChild
          size="lg"
          className="bg-white text-[#26251e] hover:opacity-90 rounded-[8px] px-8 py-[18px] text-[15px] font-medium shadow-[0_0_0_1px_#e6e5e0] transition-all duration-150"
        >
          <Link to="/app">
            Start Analyzing
            <ArrowRight className="w-4 h-4 ml-2" />
          </Link>
        </Button>
        <Button
          asChild
          variant="outline"
          size="lg"
          className="border-[#e6e5e0] text-[#26251e] hover:bg-[#f2f1ed] rounded-[8px] px-8 py-[18px] text-[15px] font-medium transition-all duration-150"
        >
          <a href="#pipeline">See How It Works</a>
        </Button>
      </div>

      {/* Stats (cursor-style metrics) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-[720px] mx-auto">
        {[
          { value: '±3–8%', label: 'Prediction Accuracy', sub: 'vs real hardware' },
          { value: '55+', label: 'Metrics', sub: 'per analysis' },
          { value: '80+', label: 'Architectures', sub: 'supported' },
          { value: '<50ms', label: 'Analysis Time', sub: 'for 8B params' },
        ].map((s) => (
          <div
            key={s.label}
            className="rounded-[8px] bg-[#f2f1ed] border border-[#e6e5e0] px-5 py-4 text-left hover:bg-[#efeee8] transition-colors duration-150"
          >
            <div className="text-[26px] font-['Inter',system-ui,sans-serif] font-[400] tracking-[-0.325px] text-[#26251e] leading-none">
              {s.value}
            </div>
            <div className="text-[11px] font-medium text-[#807d72] mt-1.5 leading-tight">{s.label}</div>
            <div className="text-[11px] text-[#a09c92] mt-0.5">{s.sub}</div>
          </div>
        ))}
      </div>
    </div>

    {/* IDE mockup (simplified canvas showcase) */}
    <div className="relative z-10 w-full max-w-[1000px] mx-auto px-6 pb-20">
      <CanvasShowcaseSimplified />
    </div>
  </section>
);

/* ─── Simplified Canvas Showcase (cursor-style IDE mockup) ────── */
const CanvasShowcaseSimplified = () => (
  <div className="relative group">
    {/* Subtle shadow — cursor.com style layered */}
    <div className="absolute -inset-[1px] rounded-[10px] bg-white opacity-60" />
    <div
      className="relative rounded-[10px] overflow-hidden border border-[#e6e5e0] bg-[#f2f1ed]"
      style={{
        boxShadow: '0 14px 32px rgba(0,0,0,0.04), 0 28px 70px rgba(0,0,0,0.06), 0 0 0 1px rgba(0,0,0,0.04)',
      }}
    >
      {/* Title Bar */}
      <div className="flex items-center gap-2 px-4 h-[36px] bg-white border-b border-[#e6e5e0]">
        <div className="flex gap-1.5">
          <div className="w-3 h-3 rounded-full bg-[#e6e5e0]" />
          <div className="w-3 h-3 rounded-full bg-[#e6e5e0]" />
          <div className="w-3 h-3 rounded-full bg-[#e6e5e0]" />
        </div>
        <span className="ml-4 text-[11px] font-['JetBrains_Mono',monospace] text-[#a09c92]">
          neurax_studio — system_graph_v2
        </span>
        <div className="ml-auto text-[10px] font-mono text-[#3b82f6] border border-[#3b82f6]/20 bg-[#3b82f6]/5 px-2 py-0.5 rounded-[4px]">
          LIVE
        </div>
      </div>

      {/* Canvas area */}
      <div className="relative h-[400px] bg-[radial-gradient(#26251e_0.5px,transparent_0.5px)] [background-size:20px_20px] opacity-[0.04]" />

      {/* Stats bar */}
      <div className="grid grid-cols-5 divide-x divide-[#e6e5e0] border-t border-[#e6e5e0] bg-white">
        {[
          { label: 'TFLOP/S', val: '312' },
          { label: 'VRAM', val: '24GB' },
          { label: 'LATENCY', val: '42ms' },
          { label: 'NODES', val: '24' },
          { label: 'CONF.', val: '99%' },
        ].map((s, i) => (
          <div key={i} className="px-4 py-2.5 text-center">
            <div className="text-[9px] font-mono text-[#a09c92] mb-0.5 tracking-[0.05em]">{s.label}</div>
            <div className="text-[11px] font-semibold text-[#26251e] font-['Inter',system-ui,sans-serif]">{s.val}</div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

/* ─── Problem Section ──────────────────────────────────────────── */
const PROBLEM_ITEMS = [
  {
    title: 'Months of GPU time wasted',
    desc: 'Teams spend weeks training architectures that OOM at step 10,000 or converge poorly. No way to predict before committing resources.',
    stat: '$2M+',
    statLabel: 'avg. cost of a failed training run',
  },
  {
    title: 'Blind architecture decisions',
    desc: 'Choosing between GQA, MQA, Flash Attention, or MoE? Without cost estimates, you\'re guessing. Memory, latency, and FLOPs remain unknowns.',
    stat: '73%',
    statLabel: 'of models need redesign after first training',
  },
  {
    title: 'Production surprises',
    desc: 'Inference latency spikes. VRAM overflow under load. Cost per token exceeds budget. These are discovered in production, not in design.',
    stat: '5×',
    statLabel: 'cost overrun when issues surface late',
  },
];

const ProblemSection = () => (
  <section className="py-28 bg-[#f7f7f4]">
    <div className="mx-auto max-w-[1100px] px-6">
      {/* Section header */}
      <div className="text-center mb-16">
        <h2 className="font-['Inter',system-ui,sans-serif] text-[36px] font-[400] tracking-[-0.72px] text-[#26251e] mb-4 leading-[1.2]">
          Building AI models is{' '}
          <span className="text-[#807d72]">expensive guesswork.</span>
        </h2>
        <p className="text-[17px] font-['Georgia','Palatino','serif'] text-[#5a5852] max-w-[650px] mx-auto leading-[1.5]">
          Every year, organizations waste millions on neural network architectures that fail.
          Not because the ideas are bad — but because there's no way to predict before committing GPU hours.
        </p>
      </div>

      {/* Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {PROBLEM_ITEMS.map((item) => (
          <div
            key={item.title}
            className="rounded-[8px] bg-[#f2f1ed] border border-[#e6e5e0] p-7 hover:bg-[#efeee8] transition-colors duration-150"
          >
            <h3 className="text-[18px] font-['Inter',system-ui,sans-serif] font-[500] text-[#26251e] mb-3 leading-tight">
              {item.title}
            </h3>
            <p className="text-[14px] text-[#5a5852] leading-[1.55] mb-6 font-['Georgia','Palatino','serif']">
              {item.desc}
            </p>
            <div className="pt-4 border-t border-[#e6e5e0]">
              <div className="text-[28px] font-['Inter',system-ui,sans-serif] font-[400] tracking-[-0.35px] text-[#3b82f6]">{item.stat}</div>
              <div className="text-[11px] text-[#807d72] mt-1">{item.statLabel}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ─── Features Section ──────────────────────────────────────────── */
const FEATURES = [
  {
    title: 'Architecture Simulation',
    desc: 'Real-time neural system modeling. Define any topology and get instant FLOPs, memory, and latency predictions — no GPU required.',
  },
  {
    title: 'Production Readiness',
    desc: 'Compile-ready architecture analysis on Rust. From design to deployment with deterministic cost estimates and OOM risk assessment.',
  },
  {
    title: 'Inference Intelligence',
    desc: 'Predict inference stability, hallucination risk, and sampling volatility before serving a single request. 20-point entropy evolution analysis.',
  },
  {
    title: 'Time Machine',
    desc: 'Rollback architecture states. Compare runs across hardware. Version your intelligence with full analytical history.',
  },
  {
    title: '10-Pass Pipeline',
    desc: 'Run 10 deterministic IR passes from architecture to report. Parallel execution for independent passes. Results in milliseconds.',
  },
  {
    title: 'Multi-Architecture',
    desc: 'Support for Transformers, MoE, Mamba, CNN, Diffusion, GNN, GAN, RNN, and more. 80+ architecture families with specialized formulas.',
  },
];

const FeaturesSection = () => (
  <section id="features" className="py-28 bg-[#f7f7f4] border-t border-[#e6e5e0]">
    <div className="mx-auto max-w-[1100px] px-6">
      <div className="text-center mb-16">
        <h2 className="font-['Inter',system-ui,sans-serif] text-[36px] font-[400] tracking-[-0.72px] text-[#26251e] mb-4">
          Engineered for every stage
        </h2>
        <p className="text-[17px] font-['Georgia','Palatino','serif'] text-[#5a5852] max-w-[600px] mx-auto">
          From architecture design to production deployment — one unified analytical platform.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {FEATURES.map((f) => (
          <div
            key={f.title}
            className="rounded-[8px] bg-[#f2f1ed] border border-[#e6e5e0] p-6 hover:bg-[#efeee8] transition-colors duration-150"
          >
            <h3 className="text-[17px] font-['Inter',system-ui,sans-serif] font-[500] text-[#26251e] mb-2">{f.title}</h3>
            <p className="text-[14px] text-[#5a5852] leading-[1.55] font-['Georgia','Palatino','serif']">{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ─── Architectures Section ────────────────────────────────────── */
const FAMILIES = [
  {
    title: 'Transformer / LLM',
    ops: ['Multi-Head Attention', 'Feed Forward', 'LayerNorm', 'Positional Encoding', 'GQA', 'MQA'],
    models: ['GPT-4', 'LLaMA 3', 'BERT', 'ViT', 'Whisper', 'Mistral'],
  },
  {
    title: 'CNN / Vision',
    ops: ['Conv2D', 'MaxPool', 'BatchNorm', 'Depthwise Conv', 'Skip Connection', 'Inception'],
    models: ['YOLOv8', 'ResNet-152', 'EfficientNet', 'U-Net', 'MobileNet', 'DINOv2'],
  },
  {
    title: 'State Space (Mamba)',
    ops: ['SSM Block', 'Linear Recurrence', 'Selective Scan', 'Gated MLP', 'Mamba Conv1d'],
    models: ['Mamba-2', 'S4', 'H3', 'Hyena', 'RWKV'],
  },
  {
    title: 'Mixture of Experts',
    ops: ['Expert Router', 'Top-K Gate', 'Expert FFN', 'Load Balancer', 'GQA + MoE'],
    models: ['Mixtral 8×7B', 'Switch Transformer', 'GShard', 'DeepSeek-V3', 'DBRX'],
  },
  {
    title: 'Diffusion',
    ops: ['U-Net', 'Cross-Attention', 'Noise Scheduler', 'VAE Decoder', 'DiT Block'],
    models: ['Stable Diffusion 3', 'DALL-E 3', 'Imagen', 'DiT', 'SDXL'],
  },
  {
    title: 'Graph Neural Nets',
    ops: ['Message Passing', 'Graph Attention', 'Readout', 'Edge Conv', 'GIN'],
    models: ['GCN', 'GAT', 'GraphSAGE', 'GIN', 'SchNet', 'PNA'],
  },
];

const ArchitecturesSection = () => (
  <section id="architectures" className="py-28 bg-[#f7f7f4] border-t border-[#e6e5e0]">
    <div className="mx-auto max-w-[1100px] px-6">
      <div className="text-center mb-16">
        <h2 className="font-['Inter',system-ui,sans-serif] text-[36px] font-[400] tracking-[-0.72px] text-[#26251e] mb-4">
          Every architecture family, one canvas
        </h2>
        <p className="text-[17px] font-['Georgia','Palatino','serif'] text-[#5a5852] max-w-[600px] mx-auto">
          Design any model type with specialized blocks, real model presets, and family-specific analytical formulas.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {FAMILIES.map((f) => (
          <div
            key={f.title}
            className="rounded-[8px] bg-[#f2f1ed] border border-[#e6e5e0] p-6 hover:bg-[#efeee8] transition-colors duration-150"
          >
            {/* Title */}
            <h3 className="text-[15px] font-['Inter',system-ui,sans-serif] font-[500] text-[#3b82f6] mb-3">
              {f.title}
            </h3>

            {/* Op badges */}
            <div className="flex flex-wrap gap-1.5 mb-4">
              {f.ops.map((op) => (
                <span
                  key={op}
                  className="text-[10px] px-2 py-0.5 rounded-[4px] font-mono text-[#807d72] bg-white border border-[#e6e5e0]"
                >
                  {op}
                </span>
              ))}
            </div>

            {/* Divider */}
            <div className="h-px bg-[#e6e5e0] my-3" />

            {/* Model names */}
            <div className="flex flex-wrap gap-x-4 gap-y-1">
              {f.models.map((m) => (
                <span key={m} className="text-[12px] text-[#5a5852] font-medium">{m}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ─── Pipeline Section ─────────────────────────────────────────── */
const STEPS = [
  {
    num: '01',
    title: 'Define your topology',
    desc: 'Write a topology.json with blocks, shapes, and hardware config. Or import from HuggingFace. Or use the visual canvas.',
  },
  {
    num: '02',
    title: 'Run the 10-pass analysis',
    desc: 'NEURAX compiles your architecture through 10 IR passes: Architecture → Graph → Tensor → Operator → Compute → Memory → Parallelism → Hardware → Cost → Report.',
  },
  {
    num: '03',
    title: 'Get the full report',
    desc: '55+ metrics: FLOPs, VRAM, latency, per-layer breakdown, confidence score, OOM risk, cost projections, and optimization recommendations.',
  },
];

const PipelineSection = () => (
  <section id="pipeline" className="py-28 bg-[#f7f7f4] border-t border-[#e6e5e0]">
    <div className="mx-auto max-w-[900px] px-6">
      <div className="text-center mb-16">
        <h2 className="font-['Inter',system-ui,sans-serif] text-[36px] font-[400] tracking-[-0.72px] text-[#26251e] mb-4">
          From JSON to full report
        </h2>
        <p className="text-[17px] font-['Georgia','Palatino','serif'] text-[#5a5852] max-w-[550px] mx-auto">
          Three steps. No GPU required. Deterministic results every time.
        </p>
      </div>

      <div className="space-y-1">
        {STEPS.map((s) => (
          <div key={s.num} className="flex items-start gap-6 py-6 border-b border-[#e6e5e0] last:border-0">
            {/* Step number */}
            <div className="flex-shrink-0 w-10 h-10 rounded-[6px] bg-[#e6e5e0] flex items-center justify-center text-[13px] font-['JetBrains_Mono',monospace] font-medium text-[#807d72]">
              {s.num}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <h3 className="text-[17px] font-['Inter',system-ui,sans-serif] font-[500] text-[#26251e] mb-1">{s.title}</h3>
              <p className="text-[14px] text-[#5a5852] leading-[1.5] font-['Georgia','Palatino','serif']">{s.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

/* ─── Rust Section ─────────────────────────────────────────────── */
const RUST_POINTS = [
  { label: 'Memory Safety', desc: 'No runtime GC pauses. Zero overhead abstractions. Exactly what a compiler deserves.' },
  { label: 'Blazing Performance', desc: 'FLOPs analysis on an 8B parameter model in < 50ms. Benchmarked with criterion.' },
  { label: 'Deterministic Builds', desc: 'Same topology.json always produces the same report. No probabilistic surprises.' },
  { label: 'Concurrency by Design', desc: 'Per-layer analysis parallelized with rayon. DAG traversal via petgraph.' },
];

const CODE_SNIPPET = `pub fn run_analysis(config: &ModelConfig) -> AnalysisResult {
  let ctx = NeuraxContext::new(config, gpu_db);

  let arch  = ArchitecturePass::build(&ctx)?;
  let graph = GraphPass::build(&ctx, &arch)?;
  let tensor = TensorPass::build(&ctx, &graph)?;
  let ops   = OperatorPass::build(&ctx, &tensor)?;
  let compute = ComputePass::build(&ctx, &ops)?;
  let memory  = MemoryPass::build(&ctx, &compute)?;

  let (par, hw) = rayon::join(
    || ParallelismPass::build(&ctx, &memory),
    || HardwarePass::build(&ctx, &memory),
  );

  let cost   = CostPass::build(&ctx, &hw)?;
  let report = ReportPass::build_report(&ctx, &all_metrics);
  Ok(report)
}`;

const RustSection = () => (
  <section className="py-28 bg-[#f3f2ee] border-t border-[#e6e5e0]">
    <div className="mx-auto max-w-[1100px] px-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
        {/* Copy */}
        <div>
          <div className="text-[11px] font-mono text-[#807d72] uppercase tracking-[0.1em] mb-4">Powered by Rust</div>
          <h2 className="font-['Inter',system-ui,sans-serif] text-[36px] font-[400] tracking-[-0.72px] text-[#26251e] mb-4 leading-[1.2]">
            Built on <span className="text-[#3b82f6]">Rust</span>.
          </h2>
          <p className="text-[15px] text-[#5a5852] leading-[1.55] mb-8 font-['Georgia','Palatino','serif']">
            Neural engineering without compromise. The entire analytical compiler — parser, cost engine, dialect router — runs in pure Rust with zero runtime surprises.
          </p>

          <div className="space-y-4">
            {RUST_POINTS.map((pt) => (
              <div key={pt.label} className="flex gap-3">
                <div className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 bg-[#3b82f6]" />
                <div>
                  <div className="text-[13px] font-medium text-[#26251e]">{pt.label}</div>
                  <div className="text-[13px] text-[#807d72] mt-0.5 leading-relaxed">{pt.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Code window */}
        <div className="relative group">
          <div className="rounded-[8px] border border-[#e6e5e0] bg-white overflow-hidden">
            {/* Titlebar */}
            <div className="flex items-center gap-1.5 px-4 h-[34px] border-b border-[#e6e5e0] bg-[#f7f7f4]">
              <span className="w-2.5 h-2.5 rounded-full bg-[#e6e5e0]" />
              <span className="w-2.5 h-2.5 rounded-full bg-[#e6e5e0]" />
              <span className="w-2.5 h-2.5 rounded-full bg-[#e6e5e0]" />
              <span className="ml-3 text-[10px] text-[#a09c92] font-mono">neurax-core/src/engine.rs</span>
            </div>
            {/* Code */}
            <pre className="overflow-x-auto p-5 text-[11px] leading-[1.65] font-['JetBrains_Mono',monospace] text-[#5a5852]">
              {CODE_SNIPPET.split('\n').map((line, i) => (
                <div key={i} className="flex gap-3">
                  <span className="select-none text-[#c0bdb4] w-6 text-right flex-shrink-0 text-[10px]">{i + 1}</span>
                  <span className={line.trim().startsWith('//') ? 'text-[#c0bdb4]' : ''}>
                    {line || '\u00a0'}
                  </span>
                </div>
              ))}
            </pre>
          </div>
        </div>
      </div>
    </div>
  </section>
);

/* ─── Closing CTA Section ──────────────────────────────────────── */
const ClosingCTA = () => (
  <section className="py-32 bg-[#f7f7f4] border-t border-[#e6e5e0] text-center">
    <div className="mx-auto max-w-[700px] px-6">
      <NeuraxAtomLogo size={36} className="mx-auto mb-6 opacity-60" />
      <h2 className="font-['Inter',system-ui,sans-serif] text-[42px] font-[400] tracking-[-0.84px] text-[#26251e] mb-4 leading-[1.15]">
        Stop guessing.
        <br />
        <span className="text-[#3b82f6]">Start compiling.</span>
      </h2>
      <p className="text-[17px] font-['Georgia','Palatino','serif'] text-[#5a5852] mb-10 max-w-[500px] mx-auto">
        The future of neural architecture design is analytical, deterministic, and instant. NEURAX makes it possible today.
      </p>
      <Button
        asChild
        size="lg"
        className="bg-white text-[#26251e] hover:opacity-90 rounded-[8px] px-10 py-[20px] text-[15px] font-medium shadow-[0_0_0_1px_#e6e5e0] transition-all duration-150"
      >
        <Link to="/app">
          Launch Studio
          <ArrowRight className="w-4 h-4 ml-2" />
        </Link>
      </Button>
      <div className="mt-10 flex items-center justify-center gap-8 text-[10px] font-mono text-[#c0bdb4] uppercase tracking-[0.1em]">
        <span>MIT Licensed</span>
        <span>Rust Powered</span>
        <span>±3–8% Accuracy</span>
      </div>
    </div>
  </section>
);

/* ─── Footer ───────────────────────────────────────────────────── */
const Footer = () => (
  <footer className="border-t border-[#e6e5e0] bg-[#f7f7f4]">
    <div className="mx-auto max-w-[1200px] px-6 py-12">
      <div className="grid grid-cols-1 md:grid-cols-5 gap-8 mb-8">
        {/* Brand */}
        <div className="md:col-span-2">
          <div className="flex items-center gap-2.5 mb-4">
            <NeuraxAtomLogo size={20} />
            <span className="text-[13px] font-medium text-[#26251e]">NEURAX</span>
          </div>
          <p className="text-[13px] text-[#807d72] leading-relaxed max-w-xs font-['Georgia','Palatino','serif']">
            Analytical compiler for neural architectures. Predict cost, memory, speed, and feasibility before training.
          </p>
          <div className="mt-5 flex items-center gap-2">
            <span className="text-[10px] font-mono text-[#c0bdb4] uppercase tracking-[0.05em]">Created by</span>
            <a
              href="https://github.com/Martial-Christian"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-[12px] font-medium text-[#26251e] hover:text-[#3b82f6] transition-colors"
            >
              <Github size={12} />
              Martial-Christian
            </a>
          </div>
        </div>

        {/* Product */}
        <div>
          <h4 className="text-[10px] font-mono text-[#a09c92] uppercase tracking-[0.1em] mb-3">Product</h4>
          <div className="space-y-2">
            <a href="#features" className="block text-[13px] text-[#807d72] hover:text-[#26251e] transition-colors">Features</a>
            <a href="#architectures" className="block text-[13px] text-[#807d72] hover:text-[#26251e] transition-colors">Architectures</a>
            <a href="#pipeline" className="block text-[13px] text-[#807d72] hover:text-[#26251e] transition-colors">Pipeline</a>
            <Link to="/app" className="block text-[13px] text-[#807d72] hover:text-[#26251e] transition-colors">Studio</Link>
          </div>
        </div>

        {/* Technical */}
        <div>
          <h4 className="text-[10px] font-mono text-[#a09c92] uppercase tracking-[0.1em] mb-3">Technical</h4>
          <div className="space-y-2">
            <span className="block text-[13px] text-[#807d72]">10 IR Passes</span>
            <span className="block text-[13px] text-[#807d72]">55+ Metrics</span>
            <span className="block text-[13px] text-[#807d72]">Rust Core</span>
            <span className="block text-[13px] text-[#807d72]">MLIR Backend</span>
          </div>
        </div>

        {/* Connect */}
        <div>
          <h4 className="text-[10px] font-mono text-[#a09c92] uppercase tracking-[0.1em] mb-3">Connect</h4>
          <div className="space-y-2">
            <a href="https://github.com/Martial-Christian/Universal_Neurax" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1.5 text-[13px] text-[#807d72] hover:text-[#26251e] transition-colors">
              <Github size={12} />
              GitHub
            </a>
            <span className="block text-[13px] text-[#807d72]">Documentation</span>
            <span className="block text-[13px] text-[#807d72]">API Reference</span>
          </div>
        </div>
      </div>

      {/* Bottom */}
      <div className="pt-6 border-t border-[#e6e5e0] flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-4 text-[11px] font-mono text-[#c0bdb4]">
          <span>© {new Date().getFullYear()} NEURAX — Deep Analytics Platform</span>
        </div>
        <div className="flex items-center gap-1.5 text-[11px] text-[#c0bdb4] font-mono">
          Created by <span className="text-[#807d72]">Martial-Christian</span>
        </div>
      </div>
    </div>
  </footer>
);

/* ═══════════════════════════════════════════════════════════════
   PAGE
   ═══════════════════════════════════════════════════════════════ */
export default function Landing() {
  return (
    <div className="min-h-screen bg-[#f7f7f4] text-[#26251e] selection:bg-[#3b82f6]/20">
      {FONT_LINK}
      <Navbar />

      <main>
        <Hero />
        <ProblemSection />
        <FeaturesSection />
        <ArchitecturesSection />
        <PipelineSection />
        <RustSection />
        <ClosingCTA />
      </main>

      <Footer />
    </div>
  );
}
