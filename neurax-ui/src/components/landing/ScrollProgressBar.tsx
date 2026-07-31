import { useState, useEffect } from 'react';
import { ArrowUp } from 'lucide-react';

const C = {
  accent: '#d79921',
  border: '#3c3836',
  card: '#282828',
};

export const ScrollProgressBar = () => {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [showBackToTop, setShowBackToTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight - windowHeight;
      const scrollTop = window.scrollY;
      const progress = (scrollTop / documentHeight) * 100;
      
      setScrollProgress(progress);
      setShowBackToTop(scrollTop > windowHeight * 0.5);
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial call
    
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <>
      {/* Progress Bar */}
      <div
        className="fixed top-0 left-0 right-0 h-[2px] z-50 pointer-events-none"
        style={{ backgroundColor: `${C.accent}20` }}
      >
        <div
          className="h-full transition-all duration-150"
          style={{
            width: `${scrollProgress}%`,
            backgroundColor: C.accent,
            boxShadow: `0 0 10px ${C.accent}`,
          }}
        />
      </div>

      {/* Back to Top Button */}
      {showBackToTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-8 right-8 w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110 z-40"
          style={{
            backgroundColor: C.card,
            border: `1px solid ${C.border}`,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          }}
        >
          <ArrowUp className="w-5 h-5" style={{ color: C.accent }} />
        </button>
      )}
    </>
  );
};
