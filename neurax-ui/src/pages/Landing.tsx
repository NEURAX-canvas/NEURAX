/**
 * Landing page.
 *
 * Written against the theme tokens rather than a fixed palette, so it holds up
 * in both light and dark mode. Every figure comes from `projectFacts`, which is
 * checked against the repository — a landing page is the one surface where a
 * wrong number is a claim rather than a bug.
 *
 * The argument it makes, in order: you are about to spend money on a guess;
 * here is the guess resolved; here is how; here is what you get back.
 */
import { Link } from 'react-router-dom';
import { ArrowRight, Github, Check } from 'lucide-react';
import { AuthControl } from '@/components/auth/AuthControl.tsx';
import { NeuraxLogo } from '@/components/brand/NeuraxLogo.tsx';
import { ScrollProgressBar } from '@/components/landing/ScrollProgressBar.tsx';
import {
  HERO_STATS,
  BLOCK_COUNT,
  FAMILY_COUNT,
  METRIC_COUNT,
  IR_PASS_COUNT,
  PRESET_COUNT,
  EXPORT_FORMATS,
} from '@/data/projectFacts.ts';

const GITHUB_URL = 'https://github.com/rustnew/NEURAX';

/** Section shell: one rhythm for the whole page. */
function Section({
  children,
  className = '',
  id,
}: {
  children: React.ReactNode;
  className?: string;
  id?: string;
}) {
  return (
    <section id={id} className={`border-t border-border/60 ${className}`}>
      <div className="mx-auto max-w-5xl px-6 py-20 sm:py-28">{children}</div>
    </section>
  );
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[11px] font-mono uppercase tracking-[0.2em] text-muted-foreground mb-4">
      {children}
    </p>
  );
}

// ─── The three questions NEURAX exists to answer ────────────────────
//
// Phrased as the question rather than the feature: a designer does not want
// "memory analysis", they want to know whether the thing will run.
const QUESTIONS = [
  {
    question: 'Will it fit?',
    answer:
      'Peak VRAM against the GPU you name, split across parameters, gradients, activations and optimizer state — per layer, per precision.',
    detail: 'Before you queue the job.',
  },
  {
    question: 'What will it cost?',
    answer:
      'Training hours, GPU spend, energy and carbon, derived from your step budget and the real specifications of the hardware you chose.',
    detail: 'Before you commit the budget.',
  },
  {
    question: 'Where is the bottleneck?',
    answer:
      'Roofline position, compute-bound or memory-bound, and which layers are responsible — with the arithmetic intensity that put them there.',
    detail: 'Before the first training step.',
  },
];

// ─── The pipeline, named honestly ───────────────────────────────────
const PIPELINE = [
  { name: 'Architecture', gives: 'Layer count, parameters, model family' },
  { name: 'Graph', gives: 'Topology validation, DAG structure' },
  { name: 'Tensor', gives: 'Shapes, liveness, tensor sizes' },
  { name: 'Operator', gives: 'Atomic operations, FLOPs per op' },
  { name: 'Compute', gives: 'Forward and backward FLOPs' },
  { name: 'Memory', gives: 'Peak VRAM, fragmentation, OOM risk' },
  { name: 'Parallelism', gives: 'Data, tensor and pipeline strategy' },
  { name: 'Hardware', gives: 'Latency, throughput, roofline' },
  { name: 'Cost', gives: 'GPU hours, spend, energy, carbon' },
  { name: 'Report', gives: 'Diagnostics and recommendations' },
];

const AUDIENCES = [
  {
    who: 'Researchers',
    what: 'Compare an idea against a baseline before writing the training script — including architectures nobody has published, described block by block with their own cost equations.',
  },
  {
    who: 'ML engineers',
    what: 'Size a model to the hardware you actually have. Find the parallelism strategy that fits, and the precision that makes it fit.',
  },
  {
    who: 'Teams with a budget',
    what: 'Put a number on a training run before approving it, with the energy and carbon that come with it.',
  },
];

export default function Landing() {
  return (
    <div className="min-h-screen bg-background text-foreground antialiased">
      <ScrollProgressBar />

      {/* ── Navigation ── */}
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/80 backdrop-blur-md">
        <nav className="mx-auto max-w-5xl px-6 h-14 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2.5" aria-label="NEURAX home">
            <NeuraxLogo variant="mark" size={24} />
            <span className="text-[15px] font-semibold tracking-[-0.02em]">NEURAX</span>
          </Link>

          <div className="flex items-center gap-1 sm:gap-4">
            <a
              href="#how"
              className="hidden sm:block text-[13px] text-muted-foreground hover:text-foreground transition-colors"
            >
              How it works
            </a>
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              className="hidden sm:flex items-center gap-1.5 text-[13px] text-muted-foreground hover:text-foreground transition-colors"
            >
              <Github className="w-3.5 h-3.5" />
              Source
            </a>
            <AuthControl triggerLabel="Sign in" triggerSize="sm" triggerVariant="ghost" />
            <Link
              to="/app"
              className="inline-flex items-center gap-1.5 h-8 px-3.5 rounded-md bg-primary text-primary-foreground text-[13px] font-medium hover:opacity-90 transition-opacity"
            >
              Open studio
            </Link>
          </div>
        </nav>
      </header>

      {/* ── Hero ── */}
      <div className="mx-auto max-w-5xl px-6 pt-24 pb-20 sm:pt-32 sm:pb-28">
        <div className="max-w-3xl">
          <Eyebrow>Analytical compiler for neural architectures</Eyebrow>

          <h1 className="text-[clamp(2.25rem,6vw,3.75rem)] font-semibold leading-[1.05] tracking-[-0.03em] text-balance">
            Know what a model costs
            <br />
            <span className="text-muted-foreground">before you train it.</span>
          </h1>

          <p className="mt-6 text-[17px] sm:text-[19px] leading-[1.6] text-muted-foreground max-w-2xl text-balance">
            NEURAX reads an architecture and returns its memory, latency, cost and carbon —
            deterministically, in milliseconds, without touching a GPU. The questions you would
            otherwise answer by running the job and finding out.
          </p>

          <div className="mt-9 flex flex-wrap items-center gap-3">
            <AuthControl
              triggerLabel="Analyse an architecture"
              triggerVariant="default"
              triggerClassName="h-11 px-6 rounded-lg text-[15px] font-medium"
            />
            <a
              href="#how"
              className="inline-flex items-center h-11 px-6 rounded-lg border border-border text-[15px] font-medium hover:bg-secondary/50 transition-colors"
            >
              How it works
            </a>
          </div>

          <p className="mt-5 text-[13px] text-muted-foreground">
            Free and open source · MIT · No sign-up, nothing to configure
          </p>
        </div>

        {/* Figures, stated plainly. */}
        <dl className="mt-16 grid grid-cols-2 sm:grid-cols-4 gap-px bg-border/60 rounded-lg overflow-hidden border border-border/60">
          {HERO_STATS.map((stat) => (
            <div key={stat.label} className="bg-background p-5">
              <dt className="text-[12px] text-muted-foreground mb-1.5">{stat.label}</dt>
              <dd className="text-[26px] font-semibold tracking-[-0.02em] font-mono">
                {stat.value}
              </dd>
            </div>
          ))}
        </dl>
      </div>

      {/* ── The problem, as questions ── */}
      <Section>
        <Eyebrow>The questions</Eyebrow>
        <h2 className="text-[clamp(1.75rem,4vw,2.5rem)] font-semibold leading-[1.15] tracking-[-0.02em] max-w-2xl text-balance">
          Every architecture decision is a bet on numbers nobody has computed.
        </h2>
        <p className="mt-5 text-[16px] leading-relaxed text-muted-foreground max-w-2xl">
          Training frameworks execute models. Runtime compilers lower them. Neither answers what you
          need at design time, when the decision is still cheap to change.
        </p>

        <div className="mt-14 grid gap-px bg-border/60 rounded-lg overflow-hidden border border-border/60 sm:grid-cols-3">
          {QUESTIONS.map((item) => (
            <div key={item.question} className="bg-background p-6">
              <h3 className="text-[17px] font-semibold tracking-[-0.01em]">{item.question}</h3>
              <p className="mt-3 text-[14px] leading-relaxed text-muted-foreground">{item.answer}</p>
              <p className="mt-4 text-[12px] font-mono text-primary">{item.detail}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* ── How ── */}
      <Section id="how">
        <Eyebrow>How it works</Eyebrow>
        <h2 className="text-[clamp(1.75rem,4vw,2.5rem)] font-semibold leading-[1.15] tracking-[-0.02em] max-w-2xl text-balance">
          A compiler, not an estimator.
        </h2>
        <p className="mt-5 text-[16px] leading-relaxed text-muted-foreground max-w-2xl">
          Your design is lowered through {IR_PASS_COUNT} intermediate representations, each computing
          what the next one needs. The same input always produces the same output — there is no
          sampling, no model of a model, nothing to re-run to be sure.
        </p>

        <ol className="mt-14 grid gap-px bg-border/60 rounded-lg overflow-hidden border border-border/60 sm:grid-cols-2">
          {PIPELINE.map((pass, index) => (
            <li key={pass.name} className="bg-background p-4 flex gap-4 items-baseline">
              <span className="text-[11px] font-mono text-muted-foreground tabular-nums w-5 shrink-0">
                {String(index + 1).padStart(2, '0')}
              </span>
              <div>
                <p className="text-[14px] font-medium">{pass.name}</p>
                <p className="text-[13px] text-muted-foreground mt-0.5">{pass.gives}</p>
              </div>
            </li>
          ))}
        </ol>
      </Section>

      {/* ── What you get ── */}
      <Section>
        <Eyebrow>What comes back</Eyebrow>
        <div className="grid gap-12 lg:grid-cols-2">
          <div>
            <h2 className="text-[clamp(1.75rem,4vw,2.5rem)] font-semibold leading-[1.15] tracking-[-0.02em] text-balance">
              {METRIC_COUNT} metrics, and the reasoning behind them.
            </h2>
            <p className="mt-5 text-[16px] leading-relaxed text-muted-foreground">
              Not a score. The parameter count per layer, the bytes each tensor holds, the FLOPs each
              operator costs, the point on the roofline where your model sits — and the diagnostics
              that say what to change.
            </p>

            <ul className="mt-8 space-y-3">
              {[
                `${FAMILY_COUNT} architecture families, from transformers to spiking networks`,
                `${BLOCK_COUNT} catalogue blocks, plus your own with their cost equations`,
                `${PRESET_COUNT} reference architectures to start from or compare against`,
                'Deterministic: the same design always returns the same numbers',
              ].map((line) => (
                <li key={line} className="flex gap-3 text-[15px] leading-relaxed">
                  <Check className="w-4 h-4 text-primary mt-1 shrink-0" />
                  <span className="text-muted-foreground">{line}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg border border-border/60 overflow-hidden self-start">
            <div className="px-4 py-2.5 border-b border-border/60 bg-secondary/30">
              <span className="text-[11px] font-mono uppercase tracking-[0.15em] text-muted-foreground">
                Export
              </span>
            </div>
            <ul className="divide-y divide-border/60">
              {EXPORT_FORMATS.map((format) => (
                <li
                  key={format}
                  className="px-4 py-3 text-[14px] flex items-center justify-between"
                >
                  <span>{format}</span>
                  <ArrowRight className="w-3.5 h-3.5 text-muted-foreground" />
                </li>
              ))}
            </ul>
          </div>
        </div>
      </Section>

      {/* ── Who ── */}
      <Section>
        <Eyebrow>Who it is for</Eyebrow>
        <div className="grid gap-10 sm:grid-cols-3 mt-2">
          {AUDIENCES.map((audience) => (
            <div key={audience.who}>
              <h3 className="text-[15px] font-semibold mb-3">{audience.who}</h3>
              <p className="text-[14px] leading-relaxed text-muted-foreground">{audience.what}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Close ── */}
      <Section>
        <div className="max-w-2xl">
          <h2 className="text-[clamp(1.75rem,4vw,2.5rem)] font-semibold leading-[1.15] tracking-[-0.02em] text-balance">
            Resolve the guess.
          </h2>
          <p className="mt-5 text-[16px] leading-relaxed text-muted-foreground">
            Place a few blocks and read what the design costs. Nothing to install in the browser,
            and no account to create first.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-3">
            <AuthControl
              triggerLabel="Open the studio"
              triggerVariant="default"
              triggerClassName="h-11 px-6 rounded-lg text-[15px] font-medium"
            />
            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 h-11 px-6 rounded-lg border border-border text-[15px] font-medium hover:bg-secondary/50 transition-colors"
            >
              <Github className="w-4 h-4" />
              Read the source
            </a>
          </div>
        </div>
      </Section>

      {/* ── Footer ── */}
      <footer className="border-t border-border/60">
        <div className="mx-auto max-w-5xl px-6 py-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <NeuraxLogo variant="mark" size={20} />
            <span className="text-[13px] text-muted-foreground">
              NEURAX — analytical compiler for neural architectures
            </span>
          </div>
          <div className="flex items-center gap-5 text-[13px] text-muted-foreground">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="hover:text-foreground transition-colors">
              GitHub
            </a>
            <a href={`${GITHUB_URL}/blob/main/LICENSE`} target="_blank" rel="noreferrer" className="hover:text-foreground transition-colors">
              MIT
            </a>
          </div>
        </div>
      </footer>
    </div>
  );
}
