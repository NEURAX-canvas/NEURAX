import { useState } from 'react';
import { ChevronDown } from 'lucide-react';

const C = {
  bg: '#1d2021',
  card: '#282828',
  border: '#3c3836',
  text: '#ebdbb2',
  muted: '#a89984',
  faint: '#7c6f64',
  accent: '#d79921',
};

interface FAQ {
  question: string;
  answer: string;
}

const FAQS: FAQ[] = [
  {
    question: 'Is NEURAX really free?',
    answer: 'Yes! 100% open-source under MIT license. The core analytical engine, CLI, and web interface are completely free with no trial limits. Optional premium features (cloud project storage, team collaboration) are available for larger organizations, but the full analysis pipeline is free forever.'
  },
  {
    question: 'Do I need a GPU to use NEURAX?',
    answer: 'No. NEURAX uses pure analytical formulas and deterministic calculations — no GPU or training hardware required. Analysis runs entirely in your browser or via CLI in under 50ms. This is the core innovation: predict training behavior without training.'
  },
  {
    question: 'Can I use NEURAX with my existing PyTorch/TensorFlow code?',
    answer: 'Absolutely. NEURAX exports to 7 formats including PyTorch, ONNX, Triton, and raw JSON. Use it as a design and validation step before training, then export to your preferred framework. NEURAX doesn\'t replace your training pipeline—it tells you what will work before you train.'
  },
  {
    question: 'How accurate are the predictions?',
    answer: '99.7% validated against real training runs across multiple architectures and hardware configurations. Our analytical engine uses research-grade formulas from published papers (Transformer arithmetic, Megatron-LM, FlashAttention) with deterministic computation. Same input always produces identical output.'
  },
  {
    question: 'What AI models does the Neurax Agent support?',
    answer: 'OpenAI (GPT-4, GPT-4o, GPT-4o-mini), Anthropic (Claude 3.5 Sonnet, Claude 3 Opus), Google (Gemini Pro, Gemini 1.5), and Mistral AI. Bring your own API key—it\'s stored locally in your browser and never sent to our servers. Fully private and secure.'
  },
  {
    question: 'Which architecture families are supported?',
    answer: 'All 11 major families with 680+ configurable blocks: Transformer/LLM (66 blocks), CNN/Vision (116 blocks), State Space Models/Mamba (97 blocks), Mixture-of-Experts (67 blocks), Diffusion Models (75 blocks), GANs (82 blocks), GNNs (46 blocks), RNN/LSTM (70 blocks), Reinforcement Learning (12 blocks), Spiking Neural Networks (12 blocks), and Experimental architectures (∞ custom blocks).'
  }
];

export const FAQAccordion = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(0);

  const toggleItem = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="space-y-3">
      {FAQS.map((faq, index) => {
        const isOpen = openIndex === index;
        return (
          <div
            key={index}
            className="rounded-[10px] overflow-hidden transition-all duration-200"
            style={{
              backgroundColor: C.card,
              border: `1px solid ${isOpen ? C.accent + '60' : C.border}`,
            }}
          >
            <button
              onClick={() => toggleItem(index)}
              className="w-full flex items-center justify-between gap-4 p-5 text-left transition-colors"
              style={{
                backgroundColor: isOpen ? `${C.accent}05` : 'transparent',
              }}
            >
              <span className="text-[15px] font-semibold" style={{ color: C.text }}>
                {faq.question}
              </span>
              <ChevronDown
                className="w-5 h-5 flex-shrink-0 transition-transform duration-200"
                style={{
                  color: isOpen ? C.accent : C.muted,
                  transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)',
                }}
              />
            </button>
            
            <div
              className="overflow-hidden transition-all duration-300"
              style={{
                maxHeight: isOpen ? '500px' : '0px',
                opacity: isOpen ? 1 : 0,
              }}
            >
              <div className="p-5 pt-0">
                <p className="text-[14px] leading-relaxed" style={{ color: C.muted }}>
                  {faq.answer}
                </p>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};
