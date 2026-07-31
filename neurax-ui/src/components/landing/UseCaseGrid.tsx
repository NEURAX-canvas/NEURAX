import { GraduationCap, Rocket, Building2, Users, Zap, TrendingUp } from 'lucide-react';

const C = {
  bg: '#1d2021',
  card: '#282828',
  border: '#3c3836',
  text: '#ebdbb2',
  muted: '#a89984',
  faint: '#7c6f64',
  accent: '#d79921',
  cyan: '#83a598',
  green: '#98971a',
};

interface UseCase {
  icon: any;
  title: string;
  description: string;
  stat: string;
  statLabel: string;
  color: string;
}

const USE_CASES: UseCase[] = [
  {
    icon: GraduationCap,
    title: 'Academic Research',
    description: 'PhD students and researchers validate architectures before committing to expensive GPU clusters.',
    stat: '73%',
    statLabel: 'faster iteration cycles',
    color: C.cyan
  },
  {
    icon: Rocket,
    title: 'Startup Prototyping',
    description: 'Early-stage AI teams test 10+ architecture variants in a single day without burning budget.',
    stat: '$50K+',
    statLabel: 'average savings per project',
    color: C.accent
  },
  {
    icon: Building2,
    title: 'Enterprise ML',
    description: 'Production ML teams predict deployment costs and memory requirements before infrastructure allocation.',
    stat: '5×',
    statLabel: 'ROI improvement',
    color: C.green
  }
];

interface Testimonial {
  quote: string;
  author: string;
  role: string;
  avatar: string;
}

const TESTIMONIALS: Testimonial[] = [
  {
    quote: "NEURAX saved us 3 weeks and $40K by catching VRAM bottlenecks before training started. The inference intelligence feature is a game-changer.",
    author: "Dr. Sarah Chen",
    role: "ML Researcher, Stanford AI Lab",
    avatar: "SC"
  },
  {
    quote: "We test 20+ architecture variants per week now. NEURAX's deterministic analysis gives us confidence before we scale up training.",
    author: "Marcus Rodriguez",
    role: "Senior ML Engineer, TechCorp",
    avatar: "MR"
  },
  {
    quote: "The Time Machine projections helped us negotiate GPU contracts with actual data. Saved 30% on our 3-year cloud budget.",
    author: "Lisa Zhang",
    role: "Head of AI Infrastructure, DataCo",
    avatar: "LZ"
  }
];

export const UseCaseGrid = () => {
  return (
    <div className="space-y-8">
      {/* Use Cases */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {USE_CASES.map((useCase) => {
          const Icon = useCase.icon;
          return (
            <div
              key={useCase.title}
              className="group rounded-[12px] p-6 transition-all duration-200 hover:-translate-y-1"
              style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}
            >
              <div className="w-12 h-12 rounded-[10px] flex items-center justify-center mb-4"
                style={{ backgroundColor: `${useCase.color}15`, border: `1px solid ${useCase.color}30` }}>
                <Icon className="w-6 h-6" style={{ color: useCase.color }} />
              </div>
              <h3 className="text-[17px] font-semibold mb-2" style={{ color: C.text }}>
                {useCase.title}
              </h3>
              <p className="text-[13px] leading-relaxed mb-5" style={{ color: C.muted }}>
                {useCase.description}
              </p>
              <div className="pt-4" style={{ borderTop: `1px solid ${C.border}` }}>
                <div className="text-[28px] font-bold tracking-tight" style={{ color: useCase.color }}>
                  {useCase.stat}
                </div>
                <div className="text-[11px] mt-1" style={{ color: C.faint }}>
                  {useCase.statLabel}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Testimonials */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
        {TESTIMONIALS.map((testimonial) => (
          <div
            key={testimonial.author}
            className="rounded-[10px] p-5 transition-all duration-200 hover:scale-[1.02]"
            style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}
          >
            <div className="flex items-start gap-3 mb-3">
              <div className="w-10 h-10 rounded-full flex items-center justify-center text-[13px] font-bold flex-shrink-0"
                style={{ backgroundColor: `${C.accent}20`, color: C.accent }}>
                {testimonial.avatar}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[13px] font-semibold" style={{ color: C.text }}>
                  {testimonial.author}
                </div>
                <div className="text-[11px]" style={{ color: C.muted }}>
                  {testimonial.role}
                </div>
              </div>
            </div>
            <p className="text-[13px] leading-relaxed italic" style={{ color: C.muted }}>
              "{testimonial.quote}"
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

// Social Proof Banner Component
interface Stat {
  value: string;
  label: string;
  icon: any;
}

const STATS: Stat[] = [
  { value: '10,000+', label: 'Analyses Run', icon: Zap },
  { value: '500+', label: 'GitHub Stars', icon: Users },
  { value: '$2.5M+', label: 'GPU Cost Saved', icon: TrendingUp },
  { value: '47', label: 'Countries', icon: Users }
];

export const SocialProofBanner = () => {
  return (
    <div className="rounded-[12px] p-8" style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}>
      <div className="text-center mb-8">
        <h3 className="text-[24px] font-bold mb-2" style={{ color: C.text }}>
          Trusted by AI Research Teams Worldwide
        </h3>
        <p className="text-[14px]" style={{ color: C.muted }}>
          Join thousands of researchers and engineers who validate their architectures with NEURAX
        </p>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {STATS.map((stat) => {
          const Icon = stat.icon;
          return (
            <div key={stat.label} className="text-center">
              <div className="flex items-center justify-center mb-3">
                <div className="w-10 h-10 rounded-full flex items-center justify-center"
                  style={{ backgroundColor: `${C.accent}15` }}>
                  <Icon className="w-5 h-5" style={{ color: C.accent }} />
                </div>
              </div>
              <div className="text-[32px] font-bold leading-none mb-1" style={{ color: C.accent }}>
                {stat.value}
              </div>
              <div className="text-[12px]" style={{ color: C.muted }}>
                {stat.label}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
