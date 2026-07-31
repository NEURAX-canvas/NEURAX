import { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Maximize2 } from 'lucide-react';

interface Screenshot {
  id: string;
  title: string;
  description: string;
  image: string;
  category: string;
}

const SCREENSHOTS: Screenshot[] = [
  {
    id: '01',
    title: 'Visual Canvas with 680+ Blocks',
    description: 'Drag-and-drop neural network design with real-time parameter editing and visual connections.',
    image: '/screenshots/01-architecture.png',
    category: 'Architecture'
  },
  {
    id: '02',
    title: '88 Pre-built Templates',
    description: 'From GPT-4 to Stable Diffusion — production-ready architectures across 11 families.',
    image: '/screenshots/02-architecture-templates.png',
    category: 'Templates'
  },
  {
    id: '03',
    title: 'Real-time Metrics Dashboard',
    description: '55+ metrics computed in under 50ms — FLOPs, VRAM, latency, cost, and energy consumption.',
    image: '/screenshots/03-simulation.png',
    category: 'Analysis'
  },
  {
    id: '04',
    title: 'Export to 7 Formats',
    description: 'PyTorch, ONNX, Triton, MLIR, Rust/Burn, JSON, and interactive network graphs.',
    image: '/screenshots/04-production.png',
    category: 'Export'
  },
  {
    id: '05',
    title: 'Multi-year Cost Projections',
    description: 'Time Machine projects training costs, carbon emissions, and regulatory compliance over multiple years.',
    image: '/screenshots/05-timemachine.png',
    category: 'Planning'
  },
  {
    id: '06',
    title: 'Inference Intelligence',
    description: 'Predict stability, hallucination risk, and sampling behavior before serving your first request.',
    image: '/screenshots/06-inference.png',
    category: 'Intelligence'
  }
];

const C = {
  bg: '#1d2021',
  card: '#282828',
  border: '#3c3836',
  text: '#ebdbb2',
  muted: '#a89984',
  faint: '#7c6f64',
  accent: '#d79921',
  orange: '#d65d0e',
};

export const ScreenshotCarousel = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAutoplay, setIsAutoplay] = useState(true);

  useEffect(() => {
    if (!isAutoplay) return;
    
    const timer = setInterval(() => {
      setCurrentIndex((prev) => (prev + 1) % SCREENSHOTS.length);
    }, 5000);

    return () => clearInterval(timer);
  }, [isAutoplay]);

  const goToPrevious = () => {
    setIsAutoplay(false);
    setCurrentIndex((prev) => (prev - 1 + SCREENSHOTS.length) % SCREENSHOTS.length);
  };

  const goToNext = () => {
    setIsAutoplay(false);
    setCurrentIndex((prev) => (prev + 1) % SCREENSHOTS.length);
  };

  const goToSlide = (index: number) => {
    setIsAutoplay(false);
    setCurrentIndex(index);
  };

  const currentScreenshot = SCREENSHOTS[currentIndex];

  return (
    <div className="relative">
      {/* Main Display */}
      <div className="relative rounded-[12px] overflow-hidden" style={{ backgroundColor: C.card, border: `1px solid ${C.border}` }}>
        {/* Image Container */}
        <div className="relative aspect-[16/9] bg-gradient-to-br from-[#1d2021] to-[#32302f]">
          <img
            src={currentScreenshot.image}
            alt={currentScreenshot.title}
            className="w-full h-full object-contain transition-opacity duration-500"
            style={{ opacity: 1 }}
          />
          
          {/* Navigation Arrows */}
          <button
            onClick={goToPrevious}
            className="absolute left-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full flex items-center justify-center transition-all hover:scale-110"
            style={{ backgroundColor: 'rgba(40,40,40,0.9)', border: `1px solid ${C.border}` }}
          >
            <ChevronLeft className="w-5 h-5" style={{ color: C.text }} />
          </button>
          
          <button
            onClick={goToNext}
            className="absolute right-4 top-1/2 -translate-y-1/2 w-10 h-10 rounded-full flex items-center justify-center transition-all hover:scale-110"
            style={{ backgroundColor: 'rgba(40,40,40,0.9)', border: `1px solid ${C.border}` }}
          >
            <ChevronRight className="w-5 h-5" style={{ color: C.text }} />
          </button>

          {/* Category Badge */}
          <div className="absolute top-4 left-4 px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-wider"
            style={{ backgroundColor: `${C.accent}20`, border: `1px solid ${C.accent}40`, color: C.accent }}
          >
            {currentScreenshot.category}
          </div>

          {/* Fullscreen Button */}
          <button
            className="absolute top-4 right-4 w-9 h-9 rounded-lg flex items-center justify-center transition-all hover:scale-110"
            style={{ backgroundColor: 'rgba(40,40,40,0.9)', border: `1px solid ${C.border}` }}
          >
            <Maximize2 className="w-4 h-4" style={{ color: C.muted }} />
          </button>
        </div>

        {/* Info Panel */}
        <div className="p-6" style={{ borderTop: `1px solid ${C.border}` }}>
          <div className="flex items-start justify-between gap-4 mb-3">
            <div className="flex-1">
              <h3 className="text-[18px] font-semibold mb-1.5" style={{ color: C.text }}>
                {currentScreenshot.title}
              </h3>
              <p className="text-[14px] leading-relaxed" style={{ color: C.muted }}>
                {currentScreenshot.description}
              </p>
            </div>
            <div className="flex-shrink-0">
              <span className="text-[11px] font-mono px-2 py-1 rounded" style={{ backgroundColor: C.border, color: C.faint }}>
                {currentScreenshot.id} / {SCREENSHOTS.length.toString().padStart(2, '0')}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Thumbnails */}
      <div className="mt-4 grid grid-cols-6 gap-2">
        {SCREENSHOTS.map((screenshot, index) => (
          <button
            key={screenshot.id}
            onClick={() => goToSlide(index)}
            className="relative aspect-video rounded-lg overflow-hidden transition-all hover:scale-105"
            style={{
              border: currentIndex === index ? `2px solid ${C.accent}` : `1px solid ${C.border}`,
              opacity: currentIndex === index ? 1 : 0.5,
            }}
          >
            <img
              src={screenshot.image}
              alt={screenshot.title}
              className="w-full h-full object-cover"
            />
            {currentIndex === index && (
              <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent flex items-end p-2">
                <span className="text-[9px] font-mono text-white">{screenshot.id}</span>
              </div>
            )}
          </button>
        ))}
      </div>

      {/* Progress Indicators */}
      <div className="flex items-center justify-center gap-1.5 mt-4">
        {SCREENSHOTS.map((_, index) => (
          <button
            key={index}
            onClick={() => goToSlide(index)}
            className="h-1 rounded-full transition-all"
            style={{
              width: currentIndex === index ? '32px' : '8px',
              backgroundColor: currentIndex === index ? C.accent : C.border,
            }}
          />
        ))}
      </div>
    </div>
  );
};
