import { VariantPreset } from '@/types/catalog.ts';

export const MODEL_TEMPLATES: VariantPreset[] = [
  // =========================================================================
  // TRANSFORMER (8)
  // =========================================================================
  {
    id: 'tpl-bert-base',
    name: 'BERT-base',
    family: 'transformer',
    description: 'Bidirectional Encoder Representations from Transformers — 110M params, 12 layers, 768 hidden, 12 heads',
    tags: ['encoder-only', 'bidirectional', 'nlp', 'classification'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: { sequence_length: 512 } },
      { id: 'n2', type: 'token_embedding', name: 'Token + Segment Embedding', x: 250, y: 140, params: { vocab_size: 30522, hidden_size: 768, max_position_embeddings: 512, type_vocab_size: 2 } },
      { id: 'n3', type: 'pos_absolute', name: 'Positional Encoding', x: 450, y: 140, params: { max_length: 512, hidden_size: 768 } },
      { id: 'n4', type: 'dropout', name: 'Embedding Dropout', x: 650, y: 140, params: { rate: 0.1 } },
      { id: 'n5', type: 'layer_stack', name: '12× Transformer Encoder', x: 850, y: 140, params: { num_layers: 12 } },
      { id: 'n6', type: 'mha_attention', name: 'Multi-Head Self-Attention', x: 850, y: 80, params: { hidden_size: 768, num_heads: 12, num_kv_heads: 12, dropout: 0.1 } },
      { id: 'n7', type: 'layernorm', name: 'Attention LayerNorm', x: 1050, y: 80, params: { eps: 1e-12, hidden_size: 768 } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'ffn_standard', name: 'FFN (2× Linear+GELU)', x: 1050, y: 200, params: { intermediate_size: 3072, activation: 'gelu' } },
      { id: 'n10', type: 'layernorm', name: 'FFN LayerNorm', x: 1250, y: 200, params: { eps: 1e-12, hidden_size: 768 } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 140, params: {} },
      { id: 'n12', type: 'layernorm', name: 'Final LayerNorm', x: 1650, y: 140, params: { eps: 1e-12, hidden_size: 768 } },
      { id: 'n13', type: 'classification_head', name: 'Classification Head', x: 1850, y: 140, params: { num_labels: 2 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n13', to: 'n14' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n11' }, { id: 'c13', from: 'n11', to: 'n5' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-bert-large',
    name: 'BERT-large',
    family: 'transformer',
    description: 'Bidirectional Encoder — 340M params, 24 layers, 1024 hidden, 16 heads',
    tags: ['encoder-only', 'bidirectional', 'nlp', 'classification'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: { sequence_length: 512 } },
      { id: 'n2', type: 'token_embedding', name: 'Token + Segment Embedding', x: 250, y: 140, params: { vocab_size: 30522, hidden_size: 1024, max_position_embeddings: 512, type_vocab_size: 2 } },
      { id: 'n3', type: 'pos_absolute', name: 'Positional Encoding', x: 450, y: 140, params: { max_length: 512, hidden_size: 1024 } },
      { id: 'n4', type: 'dropout', name: 'Embedding Dropout', x: 650, y: 140, params: { rate: 0.1 } },
      { id: 'n5', type: 'layer_stack', name: '24× Transformer Encoder', x: 850, y: 140, params: { num_layers: 24 } },
      { id: 'n6', type: 'mha_attention', name: 'Multi-Head Self-Attention', x: 850, y: 80, params: { hidden_size: 1024, num_heads: 16, num_kv_heads: 16, dropout: 0.1 } },
      { id: 'n7', type: 'layernorm', name: 'Attention LayerNorm', x: 1050, y: 80, params: { eps: 1e-12, hidden_size: 1024 } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'ffn_standard', name: 'FFN (2× Linear+GELU)', x: 1050, y: 200, params: { intermediate_size: 4096, activation: 'gelu' } },
      { id: 'n10', type: 'layernorm', name: 'FFN LayerNorm', x: 1250, y: 200, params: { eps: 1e-12, hidden_size: 1024 } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 140, params: {} },
      { id: 'n12', type: 'layernorm', name: 'Final LayerNorm', x: 1650, y: 140, params: { eps: 1e-12, hidden_size: 1024 } },
      { id: 'n13', type: 'classification_head', name: 'Classification Head', x: 1850, y: 140, params: { num_labels: 2 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n13', to: 'n14' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n11' }, { id: 'c13', from: 'n11', to: 'n5' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-gpt2-xl',
    name: 'GPT-2 XL',
    family: 'transformer',
    description: 'GPT-2 XL — 1.56B params, 48 layers, 1600 hidden, 25 heads',
    tags: ['decoder-only', 'autoregressive', 'nlp', 'generation'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: { sequence_length: 1024 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50257, hidden_size: 1600 } },
      { id: 'n3', type: 'pos_absolute', name: 'Positional Encoding', x: 450, y: 140, params: { max_length: 1024, hidden_size: 1600 } },
      { id: 'n4', type: 'dropout', name: 'Embedding Dropout', x: 650, y: 140, params: { rate: 0.1 } },
      { id: 'n5', type: 'layer_stack', name: '48× Transformer Decoder', x: 850, y: 140, params: { num_layers: 48 } },
      { id: 'n6', type: 'mha_attention', name: 'Masked Self-Attention', x: 850, y: 80, params: { hidden_size: 1600, num_heads: 25, num_kv_heads: 25, dropout: 0.1, causal: true } },
      { id: 'n7', type: 'layernorm', name: 'Attention LayerNorm', x: 1050, y: 80, params: { eps: 1e-5, hidden_size: 1600 } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'ffn_standard', name: 'FFN (2× Linear+GELU)', x: 1050, y: 200, params: { intermediate_size: 6400, activation: 'gelu' } },
      { id: 'n10', type: 'layernorm', name: 'FFN LayerNorm', x: 1250, y: 200, params: { eps: 1e-5, hidden_size: 1600 } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 140, params: {} },
      { id: 'n12', type: 'layernorm', name: 'Final LayerNorm', x: 1650, y: 140, params: { eps: 1e-5, hidden_size: 1600 } },
      { id: 'n13', type: 'lm_head', name: 'LM Head (tied)', x: 1850, y: 140, params: { vocab_size: 50257, hidden_size: 1600, weight_tied: true } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n13', to: 'n14' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n11' }, { id: 'c13', from: 'n11', to: 'n5' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-llama2-7b',
    name: 'LLaMA 2 7B',
    family: 'transformer',
    description: 'LLaMA 2 — 6.7B params, 32 layers, 4096 hidden, 32 heads, GQA, RoPE, SwiGLU',
    tags: ['decoder-only', 'autoregressive', 'llm', 'open-source'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 4096 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 32000, hidden_size: 4096 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: { hidden_size: 4096, theta: 10000.0, max_length: 4096 } },
      { id: 'n4', type: 'layer_stack', name: '32× Decoder Block', x: 650, y: 140, params: { num_layers: 32 } },
      { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n6', type: 'gqa_attention', name: 'GQA Attention', x: 850, y: 80, params: { hidden_size: 4096, num_heads: 32, num_kv_heads: 32, head_dim: 128, causal: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'rmsnorm', name: 'Pre-FFN RMSNorm', x: 650, y: 200, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n9', type: 'ffn_gated', name: 'SwiGLU FFN', x: 850, y: 200, params: { hidden_size: 4096, intermediate_size: 11008, activation: 'silu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'rmsnorm', name: 'Final RMSNorm', x: 1250, y: 140, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n12', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 32000, hidden_size: 4096 } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-llama2-13b',
    name: 'LLaMA 2 13B',
    family: 'transformer',
    description: 'LLaMA 2 — 13B params, 40 layers, 5120 hidden, 40 heads, GQA, RoPE, SwiGLU',
    tags: ['decoder-only', 'autoregressive', 'llm', 'open-source'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 4096 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 32000, hidden_size: 5120 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: { hidden_size: 5120, theta: 10000.0, max_length: 4096 } },
      { id: 'n4', type: 'layer_stack', name: '40× Decoder Block', x: 650, y: 140, params: { num_layers: 40 } },
      { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: { hidden_size: 5120, eps: 1e-5 } },
      { id: 'n6', type: 'gqa_attention', name: 'GQA Attention', x: 850, y: 80, params: { hidden_size: 5120, num_heads: 40, num_kv_heads: 40, head_dim: 128, causal: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'rmsnorm', name: 'Pre-FFN RMSNorm', x: 650, y: 200, params: { hidden_size: 5120, eps: 1e-5 } },
      { id: 'n9', type: 'ffn_gated', name: 'SwiGLU FFN', x: 850, y: 200, params: { hidden_size: 5120, intermediate_size: 13824, activation: 'silu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'rmsnorm', name: 'Final RMSNorm', x: 1250, y: 140, params: { hidden_size: 5120, eps: 1e-5 } },
      { id: 'n12', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 32000, hidden_size: 5120 } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-llama3-8b',
    name: 'LLaMA 3 8B',
    family: 'transformer',
    description: 'Meta LLaMA 3 — 8.03B params, 32 layers, 4096 hidden, 32 heads, GQA (8 KV heads), RoPE, SwiGLU, grouped query attention',
    tags: ['decoder-only', 'autoregressive', 'llm', 'gqa'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 8192 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 128256, hidden_size: 4096 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: { hidden_size: 4096, theta: 500000.0, max_length: 8192 } },
      { id: 'n4', type: 'layer_stack', name: '32× Decoder Block', x: 650, y: 140, params: { num_layers: 32 } },
      { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n6', type: 'gqa_attention', name: 'GQA (8 KV heads)', x: 850, y: 80, params: { hidden_size: 4096, num_heads: 32, num_kv_heads: 8, head_dim: 128, causal: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'rmsnorm', name: 'Pre-FFN RMSNorm', x: 650, y: 200, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n9', type: 'ffn_gated', name: 'SwiGLU FFN', x: 850, y: 200, params: { hidden_size: 4096, intermediate_size: 14336, activation: 'silu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'rmsnorm', name: 'Final RMSNorm', x: 1250, y: 140, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n12', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 128256, hidden_size: 4096 } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mistral-7b',
    name: 'Mistral 7B',
    family: 'transformer',
    description: 'Mistral 7B — 7.3B params, 32 layers, 4096 hidden, 32 heads, sliding window attention, GQA (8 KV heads), RoPE, SwiGLU',
    tags: ['decoder-only', 'autoregressive', 'llm', 'gqa', 'sliding-window'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 8192 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 32000, hidden_size: 4096 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: { hidden_size: 4096, theta: 10000.0, max_length: 8192 } },
      { id: 'n4', type: 'layer_stack', name: '32× Decoder Block', x: 650, y: 140, params: { num_layers: 32 } },
      { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n6', type: 'sliding_window_attention', name: 'Sliding Window GQA', x: 850, y: 80, params: { hidden_size: 4096, num_heads: 32, num_kv_heads: 8, head_dim: 128, window_size: 4096, causal: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'rmsnorm', name: 'Pre-FFN RMSNorm', x: 650, y: 200, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n9', type: 'ffn_gated', name: 'SwiGLU FFN', x: 850, y: 200, params: { hidden_size: 4096, intermediate_size: 14336, activation: 'silu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'rmsnorm', name: 'Final RMSNorm', x: 1250, y: 140, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n12', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 32000, hidden_size: 4096 } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-falcon-7b',
    name: 'Falcon 7B',
    family: 'transformer',
    description: 'TII Falcon 7B — 7B params, 32 layers, 4544 hidden, 71 heads, multi-query attention, RoPE, GELU',
    tags: ['decoder-only', 'autoregressive', 'llm', 'mqa'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 65024, hidden_size: 4544 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: { hidden_size: 4544, theta: 10000.0, max_length: 2048 } },
      { id: 'n4', type: 'layer_stack', name: '32× Decoder Block', x: 650, y: 140, params: { num_layers: 32 } },
      { id: 'n5', type: 'layernorm', name: 'Pre-Attention LayerNorm', x: 650, y: 80, params: { hidden_size: 4544, eps: 1e-5 } },
      { id: 'n6', type: 'mqa_attention', name: 'Multi-Query Attention', x: 850, y: 80, params: { hidden_size: 4544, num_heads: 71, num_kv_heads: 1, head_dim: 64, causal: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'layernorm', name: 'Pre-FFN LayerNorm', x: 650, y: 200, params: { hidden_size: 4544, eps: 1e-5 } },
      { id: 'n9', type: 'ffn_standard', name: 'FFN (2× Linear+GELU)', x: 850, y: 200, params: { hidden_size: 4544, intermediate_size: 18176, activation: 'gelu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'layernorm', name: 'Final LayerNorm', x: 1250, y: 140, params: { hidden_size: 4544, eps: 1e-5 } },
      { id: 'n12', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 65024, hidden_size: 4544 } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },
  // =========================================================================
  // CNN (8)
  // =========================================================================
  {
    id: 'tpl-resnet50',
    name: 'ResNet-50',
    family: 'cnn',
    description: 'ResNet-50 — 25.6M params, 4 stages (3+4+6+3 bottleneck blocks), 7×7 conv stem, global avg pool',
    tags: ['vision', 'classification', 'residual', 'imagenet'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: '7×7 Conv (stride 2)', x: 250, y: 140, params: { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3 } },
      { id: 'n3', type: 'batchnorm', name: 'BatchNorm', x: 450, y: 140, params: { num_features: 64, eps: 1e-5 } },
      { id: 'n4', type: 'relu', name: 'ReLU', x: 650, y: 140, params: {} },
      { id: 'n5', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 850, y: 140, params: { kernel_size: 3, stride: 2, padding: 1 } },
      { id: 'n6', type: 'bottleneck_block', name: 'Stage 1: 3× Bottleneck (256→64→256)', x: 1050, y: 140, params: { planes: 256, blocks: 3, stride: 1, expansion: 4 } },
      { id: 'n7', type: 'bottleneck_block', name: 'Stage 2: 4× Bottleneck (512→128→512)', x: 1250, y: 140, params: { planes: 512, blocks: 4, stride: 2, expansion: 4 } },
      { id: 'n8', type: 'bottleneck_block', name: 'Stage 3: 6× Bottleneck (1024→256→1024)', x: 1450, y: 140, params: { planes: 1024, blocks: 6, stride: 2, expansion: 4 } },
      { id: 'n9', type: 'bottleneck_block', name: 'Stage 4: 3× Bottleneck (2048→512→2048)', x: 1650, y: 140, params: { planes: 2048, blocks: 3, stride: 2, expansion: 4 } },
      { id: 'n10', type: 'global_pool', name: 'Global Avg Pool', x: 1850, y: 140, params: {} },
      { id: 'n11', type: 'classification_head', name: 'FC-1000', x: 2050, y: 140, params: { num_labels: 1000 } },
      { id: 'n12', type: 'output', name: 'Output', x: 2250, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-resnet152',
    name: 'ResNet-152',
    family: 'cnn',
    description: 'ResNet-152 — 60.2M params, 4 stages (3+8+36+3 bottleneck blocks), 7×7 conv stem',
    tags: ['vision', 'classification', 'residual', 'deep'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: '7×7 Conv (stride 2)', x: 250, y: 140, params: { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3 } },
      { id: 'n3', type: 'batchnorm', name: 'BatchNorm', x: 450, y: 140, params: { num_features: 64, eps: 1e-5 } },
      { id: 'n4', type: 'relu', name: 'ReLU', x: 650, y: 140, params: {} },
      { id: 'n5', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 850, y: 140, params: { kernel_size: 3, stride: 2, padding: 1 } },
      { id: 'n6', type: 'bottleneck_block', name: 'Stage 1: 3× Bottleneck (256)', x: 1050, y: 140, params: { planes: 256, blocks: 3, stride: 1, expansion: 4 } },
      { id: 'n7', type: 'bottleneck_block', name: 'Stage 2: 8× Bottleneck (512)', x: 1250, y: 140, params: { planes: 512, blocks: 8, stride: 2, expansion: 4 } },
      { id: 'n8', type: 'bottleneck_block', name: 'Stage 3: 36× Bottleneck (1024)', x: 1450, y: 140, params: { planes: 1024, blocks: 36, stride: 2, expansion: 4 } },
      { id: 'n9', type: 'bottleneck_block', name: 'Stage 4: 3× Bottleneck (2048)', x: 1650, y: 140, params: { planes: 2048, blocks: 3, stride: 2, expansion: 4 } },
      { id: 'n10', type: 'global_pool', name: 'Global Avg Pool', x: 1850, y: 140, params: {} },
      { id: 'n11', type: 'classification_head', name: 'FC-1000', x: 2050, y: 140, params: { num_labels: 1000 } },
      { id: 'n12', type: 'output', name: 'Output', x: 2250, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-vgg16',
    name: 'VGG-16',
    family: 'cnn',
    description: 'VGG-16 — 138M params, 13 conv layers + 3 FC, 3×3 conv blocks, 2×2 max pool',
    tags: ['vision', 'classification', 'classic', 'conv'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: 'Conv Block 1: 2× [64, 3×3]', x: 250, y: 140, params: { in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1, blocks: 2 } },
      { id: 'n3', type: 'relu', name: 'ReLU', x: 450, y: 140, params: {} },
      { id: 'n4', type: 'max_pool', name: '2×2 Max Pool', x: 650, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n5', type: 'conv2d', name: 'Conv Block 2: 2× [128, 3×3]', x: 850, y: 140, params: { in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1, blocks: 2 } },
      { id: 'n6', type: 'relu', name: 'ReLU', x: 1050, y: 140, params: {} },
      { id: 'n7', type: 'max_pool', name: '2×2 Max Pool', x: 1250, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n8', type: 'conv2d', name: 'Conv Block 3: 3× [256, 3×3]', x: 1450, y: 140, params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1, blocks: 3 } },
      { id: 'n9', type: 'relu', name: 'ReLU', x: 1650, y: 140, params: {} },
      { id: 'n10', type: 'max_pool', name: '2×2 Max Pool', x: 1850, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n11', type: 'conv2d', name: 'Conv Block 4: 3× [512, 3×3]', x: 2050, y: 140, params: { in_channels: 256, out_channels: 512, kernel_size: 3, padding: 1, blocks: 3 } },
      { id: 'n12', type: 'relu', name: 'ReLU', x: 2250, y: 140, params: {} },
      { id: 'n13', type: 'max_pool', name: '2×2 Max Pool', x: 2450, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n14', type: 'conv2d', name: 'Conv Block 5: 3× [512, 3×3]', x: 2650, y: 140, params: { in_channels: 512, out_channels: 512, kernel_size: 3, padding: 1, blocks: 3 } },
      { id: 'n15', type: 'relu', name: 'ReLU', x: 2850, y: 140, params: {} },
      { id: 'n16', type: 'max_pool', name: '2×2 Max Pool', x: 3050, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n17', type: 'flatten', name: 'Flatten', x: 3250, y: 140, params: {} },
      { id: 'n18', type: 'dense', name: 'FC 4096', x: 3450, y: 140, params: { in_features: 25088, out_features: 4096 } },
      { id: 'n19', type: 'relu', name: 'ReLU', x: 3650, y: 140, params: {} },
      { id: 'n20', type: 'dropout', name: 'Dropout', x: 3850, y: 140, params: { rate: 0.5 } },
      { id: 'n21', type: 'dense', name: 'FC 4096', x: 4050, y: 140, params: { in_features: 4096, out_features: 4096 } },
      { id: 'n22', type: 'relu', name: 'ReLU', x: 4250, y: 140, params: {} },
      { id: 'n23', type: 'dropout', name: 'Dropout', x: 4450, y: 140, params: { rate: 0.5 } },
      { id: 'n24', type: 'classification_head', name: 'FC-1000', x: 4650, y: 140, params: { num_labels: 1000 } },
      { id: 'n25', type: 'output', name: 'Output', x: 4850, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' }, { id: 'c12', from: 'n12', to: 'n13' },
      { id: 'c13', from: 'n13', to: 'n14' }, { id: 'c14', from: 'n14', to: 'n15' },
      { id: 'c15', from: 'n15', to: 'n16' }, { id: 'c16', from: 'n16', to: 'n17' },
      { id: 'c17', from: 'n17', to: 'n18' }, { id: 'c18', from: 'n18', to: 'n19' },
      { id: 'c19', from: 'n19', to: 'n20' }, { id: 'c20', from: 'n20', to: 'n21' },
      { id: 'c21', from: 'n21', to: 'n22' }, { id: 'c22', from: 'n22', to: 'n23' },
      { id: 'c23', from: 'n23', to: 'n24' }, { id: 'c24', from: 'n24', to: 'n25' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-efficientnet-b0',
    name: 'EfficientNet-B0',
    family: 'cnn',
    description: 'EfficientNet-B0 — 5.3M params, MBConv blocks with SE, compound scaling baseline',
    tags: ['vision', 'classification', 'efficient', 'mbconv'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'stem_block', name: 'Stem: 3×3 Conv (32)', x: 250, y: 140, params: { out_channels: 32, kernel_size: 3, stride: 2 } },
      { id: 'n3', type: 'mbconv_block', name: 'MBConv1 (16), k3×3', x: 450, y: 140, params: { in_channels: 32, out_channels: 16, kernel_size: 3, expand_ratio: 1, se_ratio: 0.25, stride: 1 } },
      { id: 'n4', type: 'mbconv_block', name: 'MBConv6 (24), k3×3 ×2', x: 650, y: 140, params: { in_channels: 16, out_channels: 24, kernel_size: 3, expand_ratio: 6, se_ratio: 0.25, stride: 2, num_blocks: 2 } },
      { id: 'n5', type: 'mbconv_block', name: 'MBConv6 (40), k5×5 ×2', x: 850, y: 140, params: { in_channels: 24, out_channels: 40, kernel_size: 5, expand_ratio: 6, se_ratio: 0.25, stride: 2, num_blocks: 2 } },
      { id: 'n6', type: 'mbconv_block', name: 'MBConv6 (80), k3×3 ×3', x: 1050, y: 140, params: { in_channels: 40, out_channels: 80, kernel_size: 3, expand_ratio: 6, se_ratio: 0.25, stride: 2, num_blocks: 3 } },
      { id: 'n7', type: 'mbconv_block', name: 'MBConv6 (112), k5×5 ×3', x: 1250, y: 140, params: { in_channels: 80, out_channels: 112, kernel_size: 5, expand_ratio: 6, se_ratio: 0.25, stride: 1, num_blocks: 3 } },
      { id: 'n8', type: 'mbconv_block', name: 'MBConv6 (192), k5×5 ×4', x: 1450, y: 140, params: { in_channels: 112, out_channels: 192, kernel_size: 5, expand_ratio: 6, se_ratio: 0.25, stride: 2, num_blocks: 4 } },
      { id: 'n9', type: 'mbconv_block', name: 'MBConv6 (320), k3×3 ×1', x: 1650, y: 140, params: { in_channels: 192, out_channels: 320, kernel_size: 3, expand_ratio: 6, se_ratio: 0.25, stride: 1, num_blocks: 1 } },
      { id: 'n10', type: 'conv2d', name: '1×1 Conv (1280)', x: 1850, y: 140, params: { in_channels: 320, out_channels: 1280, kernel_size: 1 } },
      { id: 'n11', type: 'global_pool', name: 'Global Avg Pool', x: 2050, y: 140, params: {} },
      { id: 'n12', type: 'classification_head', name: 'FC-1000', x: 2250, y: 140, params: { num_labels: 1000 } },
      { id: 'n13', type: 'output', name: 'Output', x: 2450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' }, { id: 'c12', from: 'n12', to: 'n13' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-efficientnet-b7',
    name: 'EfficientNet-B7',
    family: 'cnn',
    description: 'EfficientNet-B7 — 66M params, compound-scaled (width 2.0, depth 3.1, res 600), MBConv with SE',
    tags: ['vision', 'classification', 'efficient', 'large'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 600, width: 600 } },
      { id: 'n2', type: 'stem_block', name: 'Stem: 3×3 Conv (64)', x: 250, y: 140, params: { out_channels: 64, kernel_size: 3, stride: 2 } },
      { id: 'n3', type: 'mbconv_block', name: 'MBConv1 (32) ×2', x: 450, y: 140, params: { in_channels: 64, out_channels: 32, kernel_size: 3, expand_ratio: 1, se_ratio: 0.25, num_blocks: 2 } },
      { id: 'n4', type: 'mbconv_block', name: 'MBConv6 (48) ×5', x: 650, y: 140, params: { in_channels: 32, out_channels: 48, kernel_size: 3, expand_ratio: 6, se_ratio: 0.25, stride: 2, num_blocks: 5 } },
      { id: 'n5', type: 'mbconv_block', name: 'MBConv6 (80) ×5', x: 850, y: 140, params: { in_channels: 48, out_channels: 80, kernel_size: 5, expand_ratio: 6, se_ratio: 0.25, stride: 2, num_blocks: 5 } },
      { id: 'n6', type: 'mbconv_block', name: 'MBConv6 (160) ×7', x: 1050, y: 140, params: { in_channels: 80, out_channels: 160, kernel_size: 3, expand_ratio: 6, se_ratio: 0.25, stride: 2, num_blocks: 7 } },
      { id: 'n7', type: 'mbconv_block', name: 'MBConv6 (224) ×12', x: 1250, y: 140, params: { in_channels: 160, out_channels: 224, kernel_size: 5, expand_ratio: 6, se_ratio: 0.25, stride: 1, num_blocks: 12 } },
      { id: 'n8', type: 'mbconv_block', name: 'MBConv6 (384) ×9', x: 1450, y: 140, params: { in_channels: 224, out_channels: 384, kernel_size: 5, expand_ratio: 6, se_ratio: 0.25, stride: 2, num_blocks: 9 } },
      { id: 'n9', type: 'mbconv_block', name: 'MBConv6 (640) ×1', x: 1650, y: 140, params: { in_channels: 384, out_channels: 640, kernel_size: 3, expand_ratio: 6, se_ratio: 0.25, num_blocks: 1 } },
      { id: 'n10', type: 'conv2d', name: '1×1 Conv (2560)', x: 1850, y: 140, params: { in_channels: 640, out_channels: 2560, kernel_size: 1 } },
      { id: 'n11', type: 'global_pool', name: 'Global Avg Pool', x: 2050, y: 140, params: {} },
      { id: 'n12', type: 'classification_head', name: 'FC-1000', x: 2250, y: 140, params: { num_labels: 1000 } },
      { id: 'n13', type: 'output', name: 'Output', x: 2450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' }, { id: 'c12', from: 'n12', to: 'n13' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-convnext-tiny',
    name: 'ConvNeXt-Tiny',
    family: 'cnn',
    description: 'ConvNeXt-Tiny — 28.6M params, 4 stages, 7×7 depthwise conv, LayerNorm, GELU, 96→768 dims',
    tags: ['vision', 'classification', 'modern', 'convnext'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: 'Stem: 4×4 Conv (stride 4)', x: 250, y: 140, params: { in_channels: 3, out_channels: 96, kernel_size: 4, stride: 4 } },
      { id: 'n3', type: 'layernorm', name: 'LayerNorm', x: 450, y: 140, params: { hidden_size: 96, eps: 1e-6 } },
      { id: 'n4', type: 'convnext_block', name: 'Stage 1: 3× ConvNeXt Block (96)', x: 650, y: 140, params: { dim: 96, num_blocks: 3, kernel_size: 7 } },
      { id: 'n5', type: 'convnext_block', name: 'Stage 2: 3× ConvNeXt Block (192)', x: 850, y: 140, params: { dim: 192, num_blocks: 3, kernel_size: 7, downsample: true } },
      { id: 'n6', type: 'convnext_block', name: 'Stage 3: 9× ConvNeXt Block (384)', x: 1050, y: 140, params: { dim: 384, num_blocks: 9, kernel_size: 7, downsample: true } },
      { id: 'n7', type: 'convnext_block', name: 'Stage 4: 3× ConvNeXt Block (768)', x: 1250, y: 140, params: { dim: 768, num_blocks: 3, kernel_size: 7, downsample: true } },
      { id: 'n8', type: 'layernorm', name: 'Final LayerNorm', x: 1450, y: 140, params: { hidden_size: 768, eps: 1e-6 } },
      { id: 'n9', type: 'global_pool', name: 'Global Avg Pool', x: 1650, y: 140, params: {} },
      { id: 'n10', type: 'classification_head', name: 'FC-1000', x: 1850, y: 140, params: { num_labels: 1000 } },
      { id: 'n11', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mobilenetv2',
    name: 'MobileNetV2',
    family: 'cnn',
    description: 'MobileNetV2 — 3.5M params, inverted residuals with linear bottlenecks, 3×3 depthwise conv',
    tags: ['vision', 'classification', 'efficient', 'mobile'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: '3×3 Conv (32, stride 2)', x: 250, y: 140, params: { out_channels: 32, kernel_size: 3, stride: 2 } },
      { id: 'n3', type: 'inverted_bottleneck', name: 'Bottleneck (16) ×1', x: 450, y: 140, params: { in_channels: 32, out_channels: 16, expand_ratio: 1, num_blocks: 1 } },
      { id: 'n4', type: 'inverted_bottleneck', name: 'Bottleneck (24) ×2', x: 650, y: 140, params: { in_channels: 16, out_channels: 24, expand_ratio: 6, stride: 2, num_blocks: 2 } },
      { id: 'n5', type: 'inverted_bottleneck', name: 'Bottleneck (32) ×3', x: 850, y: 140, params: { in_channels: 24, out_channels: 32, expand_ratio: 6, stride: 2, num_blocks: 3 } },
      { id: 'n6', type: 'inverted_bottleneck', name: 'Bottleneck (64) ×4', x: 1050, y: 140, params: { in_channels: 32, out_channels: 64, expand_ratio: 6, stride: 2, num_blocks: 4 } },
      { id: 'n7', type: 'inverted_bottleneck', name: 'Bottleneck (96) ×3', x: 1250, y: 140, params: { in_channels: 64, out_channels: 96, expand_ratio: 6, stride: 1, num_blocks: 3 } },
      { id: 'n8', type: 'inverted_bottleneck', name: 'Bottleneck (160) ×3', x: 1450, y: 140, params: { in_channels: 96, out_channels: 160, expand_ratio: 6, stride: 2, num_blocks: 3 } },
      { id: 'n9', type: 'inverted_bottleneck', name: 'Bottleneck (320) ×1', x: 1650, y: 140, params: { in_channels: 160, out_channels: 320, expand_ratio: 6, num_blocks: 1 } },
      { id: 'n10', type: 'conv2d', name: '1×1 Conv (1280)', x: 1850, y: 140, params: { in_channels: 320, out_channels: 1280, kernel_size: 1 } },
      { id: 'n11', type: 'global_pool', name: 'Global Avg Pool', x: 2050, y: 140, params: {} },
      { id: 'n12', type: 'classification_head', name: 'FC-1000', x: 2250, y: 140, params: { num_labels: 1000 } },
      { id: 'n13', type: 'output', name: 'Output', x: 2450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' }, { id: 'c12', from: 'n12', to: 'n13' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-densenet121',
    name: 'DenseNet-121',
    family: 'cnn',
    description: 'DenseNet-121 — 7.9M params, 4 dense blocks with transition layers, growth rate 32, bottleneck + compression',
    tags: ['vision', 'classification', 'dense', 'concat'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: '7×7 Conv (stride 2)', x: 250, y: 140, params: { out_channels: 64, kernel_size: 7, stride: 2, padding: 3 } },
      { id: 'n3', type: 'batchnorm', name: 'BatchNorm + ReLU', x: 450, y: 140, params: { num_features: 64 } },
      { id: 'n4', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 650, y: 140, params: { kernel_size: 3, stride: 2, padding: 1 } },
      { id: 'n5', type: 'dense_block', name: 'Dense Block 1: 6× BN-ReLU-3×3 (32)', x: 850, y: 140, params: { num_layers: 6, growth_rate: 32, in_channels: 64, bottleneck: true } },
      { id: 'n6', type: 'transition_layer', name: 'Transition Layer 1: 1×1 Conv + 2×2 Pool', x: 1050, y: 140, params: { in_channels: 256, out_channels: 128, reduction: 0.5 } },
      { id: 'n7', type: 'dense_block', name: 'Dense Block 2: 12× BN-ReLU-3×3 (32)', x: 1250, y: 140, params: { num_layers: 12, growth_rate: 32, in_channels: 128, bottleneck: true } },
      { id: 'n8', type: 'transition_layer', name: 'Transition Layer 2: 1×1 Conv + 2×2 Pool', x: 1450, y: 140, params: { in_channels: 512, out_channels: 256, reduction: 0.5 } },
      { id: 'n9', type: 'dense_block', name: 'Dense Block 3: 24× BN-ReLU-3×3 (32)', x: 1650, y: 140, params: { num_layers: 24, growth_rate: 32, in_channels: 256, bottleneck: true } },
      { id: 'n10', type: 'transition_layer', name: 'Transition Layer 3: 1×1 Conv + 2×2 Pool', x: 1850, y: 140, params: { in_channels: 1024, out_channels: 512, reduction: 0.5 } },
      { id: 'n11', type: 'dense_block', name: 'Dense Block 4: 16× BN-ReLU-3×3 (32)', x: 2050, y: 140, params: { num_layers: 16, growth_rate: 32, in_channels: 512, bottleneck: true } },
      { id: 'n12', type: 'global_pool', name: 'Global Avg Pool', x: 2250, y: 140, params: {} },
      { id: 'n13', type: 'classification_head', name: 'FC-1000', x: 2450, y: 140, params: { num_labels: 1000 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' }, { id: 'c12', from: 'n12', to: 'n13' },
      { id: 'c13', from: 'n13', to: 'n14' },
    ],
    defaultParams: {},
  },
  // =========================================================================
  // SSM (8)
  // =========================================================================
  // Every Mamba/Mamba-2 template below is one real block — `mamba_block`,
  // carrying `neurax_formulas::ssm::mamba_params`'s real formula — rather
  // than the visual in_proj/conv1d/S6/out_proj decomposition these templates
  // used to show. That decomposition read well and computed wrong: two of
  // its nodes (`ssm_in_proj`, `ssm_out_proj`) each independently resolved to
  // the compiler's generic `state_space` layer type, which carries the
  // *whole* block's formula — so the pair double-counted one block's worth
  // of parameters — while the step that is conceptually the block's core,
  // the selective scan itself (`s6_block`/`ssd_block`), matched no layer
  // type the compiler recognises and fell back to zero. Every size below
  // also had its shape re-verified against the real `config.json` on the
  // Hub rather than trusted from the original description text: 130M/370M/
  // 790M were stated at half their real layer count (12 vs. 24, 24 vs. 48,
  // 24 vs. 48).
  {
    id: 'tpl-mamba-130m',
    name: 'Mamba-130M',
    family: 'ssm',
    description: 'Mamba — 130M params, 24 layers, 768 hidden, selective SSM, state dim 16, expansion ×2 (state-spaces/mamba-130m-hf)',
    tags: ['ssm', 'selective-scan', 'linear-time', 'efficient'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50280, hidden_size: 768 } },
      { id: 'n3', type: 'layer_stack', name: '24× Mamba Block', x: 450, y: 140, params: { num_layers: 24 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block RMSNorm', x: 450, y: 80, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n5', type: 'mamba_block', name: 'Mamba (selective SSM, state 16)', x: 650, y: 80, params: { hidden_size: 768, state_dim: 16, expansion_factor: 2, conv_kernel_size: 4 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final RMSNorm', x: 1050, y: 140, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50280, hidden_size: 768, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba-370m',
    name: 'Mamba-370M',
    family: 'ssm',
    description: 'Mamba — 370M params, 48 layers, 1024 hidden, selective SSM, state dim 16, expansion ×2 (state-spaces/mamba-370m-hf)',
    tags: ['ssm', 'selective-scan', 'linear-time'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50280, hidden_size: 1024 } },
      { id: 'n3', type: 'layer_stack', name: '48× Mamba Block', x: 450, y: 140, params: { num_layers: 48 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block RMSNorm', x: 450, y: 80, params: { hidden_size: 1024, eps: 1e-5 } },
      { id: 'n5', type: 'mamba_block', name: 'Mamba (selective SSM, state 16)', x: 650, y: 80, params: { hidden_size: 1024, state_dim: 16, expansion_factor: 2, conv_kernel_size: 4 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final RMSNorm', x: 1050, y: 140, params: { hidden_size: 1024, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50280, hidden_size: 1024, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba-790m',
    name: 'Mamba-790M',
    family: 'ssm',
    description: 'Mamba — 790M params, 48 layers, 1536 hidden, selective SSM, state dim 16, expansion ×2 (state-spaces/mamba-790m-hf)',
    tags: ['ssm', 'selective-scan', 'linear-time'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50280, hidden_size: 1536 } },
      { id: 'n3', type: 'layer_stack', name: '48× Mamba Block', x: 450, y: 140, params: { num_layers: 48 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block RMSNorm', x: 450, y: 80, params: { hidden_size: 1536, eps: 1e-5 } },
      { id: 'n5', type: 'mamba_block', name: 'Mamba (selective SSM, state 16)', x: 650, y: 80, params: { hidden_size: 1536, state_dim: 16, expansion_factor: 2, conv_kernel_size: 4 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final RMSNorm', x: 1050, y: 140, params: { hidden_size: 1536, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50280, hidden_size: 1536, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba-1.4b',
    name: 'Mamba-1.4B',
    family: 'ssm',
    description: 'Mamba — 1.4B params, 48 layers, 2048 hidden, selective SSM, state dim 16, expansion ×2 (state-spaces/mamba-1.4b-hf)',
    tags: ['ssm', 'selective-scan', 'linear-time', 'deep'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50280, hidden_size: 2048 } },
      { id: 'n3', type: 'layer_stack', name: '48× Mamba Block', x: 450, y: 140, params: { num_layers: 48 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block RMSNorm', x: 450, y: 80, params: { hidden_size: 2048, eps: 1e-5 } },
      { id: 'n5', type: 'mamba_block', name: 'Mamba (selective SSM, state 16)', x: 650, y: 80, params: { hidden_size: 2048, state_dim: 16, expansion_factor: 2, conv_kernel_size: 4 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final RMSNorm', x: 1050, y: 140, params: { hidden_size: 2048, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50280, hidden_size: 2048, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba-2.8b',
    name: 'Mamba-2.8B',
    family: 'ssm',
    // Verified against the published 2.8B figure to -4.89% —
    // neurax-core/tests/published_model_accuracy.rs, tolerance ±10%.
    description: 'Mamba — 2.8B params, 64 layers, 2560 hidden, selective SSM, state dim 16, expansion ×2 (state-spaces/mamba-2.8b-hf)',
    tags: ['ssm', 'selective-scan', 'linear-time', 'deep'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50280, hidden_size: 2560 } },
      { id: 'n3', type: 'layer_stack', name: '64× Mamba Block', x: 450, y: 140, params: { num_layers: 64 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block RMSNorm', x: 450, y: 80, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n5', type: 'mamba_block', name: 'Mamba (selective SSM, state 16)', x: 650, y: 80, params: { hidden_size: 2560, state_dim: 16, expansion_factor: 2, conv_kernel_size: 4 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final RMSNorm', x: 1050, y: 140, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50280, hidden_size: 2560, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba2-130m',
    name: 'Mamba 2 130M',
    family: 'ssm',
    // Mamba-2's head-structured SSD algorithm reshapes the same total state
    // rather than adding a separate parameter tensor per head, so the
    // formula below (shared with Mamba-1) is a stated approximation, not a
    // verified-exact figure the way the Mamba-1 sizes above are.
    description: 'Mamba 2 — 130M params, 24 layers, 768 hidden, SSD (State Space Dual), state dim 128 — parameter count approximated with the Mamba-1 formula',
    tags: ['ssm', 'mamba2', 'ssd', 'selective-scan'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50288, hidden_size: 768 } },
      { id: 'n3', type: 'layer_stack', name: '24× Mamba-2 Block', x: 450, y: 140, params: { num_layers: 24 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block RMSNorm', x: 450, y: 80, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n5', type: 'mamba_block', name: 'Mamba-2 (SSD, state 128, 24 heads × 64)', x: 650, y: 80, params: { hidden_size: 768, state_dim: 128, expansion_factor: 2, conv_kernel_size: 4 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final RMSNorm', x: 1050, y: 140, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50288, hidden_size: 768, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba2-2.7b',
    name: 'Mamba 2 2.7B',
    family: 'ssm',
    description: 'Mamba 2 — 2.7B params, 64 layers, 2560 hidden, SSD, state dim 128, 80 heads × 64 — parameter count approximated with the Mamba-1 formula',
    tags: ['ssm', 'mamba2', 'ssd', 'selective-scan', 'large'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50288, hidden_size: 2560 } },
      { id: 'n3', type: 'layer_stack', name: '64× Mamba-2 Block', x: 450, y: 140, params: { num_layers: 64 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block RMSNorm', x: 450, y: 80, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n5', type: 'mamba_block', name: 'Mamba-2 (SSD, state 128, 80 heads × 64)', x: 650, y: 80, params: { hidden_size: 2560, state_dim: 128, expansion_factor: 2, conv_kernel_size: 4 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final RMSNorm', x: 1050, y: 140, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50288, hidden_size: 2560, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  // =========================================================================
  // DIFFUSION (8)
  // =========================================================================
  {
    id: 'tpl-ddpm',
    name: 'DDPM (Denoising Diffusion Probabilistic Model)',
    family: 'diffusion',
    description: 'DDPM — 1000 timesteps, linear noise schedule, UNet backbone, cosine scheduling alternative, 35.7M',
    tags: ['diffusion', 'ddpm', 'unet', 'markov-chain'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input (noise)', x: 50, y: 140, params: { channels: 3, height: 32, width: 32 } },
      { id: 'n2', type: 'linear', name: 'Time Embedding (256)', x: 250, y: 80, params: { hidden_size: 256 } },
      { id: 'n3', type: 'unet_encoder', name: 'UNet Encoder (2× down)', x: 450, y: 140, params: { in_channels: 3, channels: [128, 256, 256], num_res_blocks: 2 } },
      { id: 'n4', type: 'unet_mid', name: 'UNet Middle (attn)', x: 650, y: 140, params: { channels: 256, with_attention: true } },
      { id: 'n5', type: 'unet_decoder', name: 'UNet Decoder (2× up)', x: 850, y: 140, params: { out_channels: 3, channels: [256, 256, 128], num_res_blocks: 2 } },
      { id: 'n6', type: 'output', name: 'Output (pred noise)', x: 1050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n3' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n2', to: 'n5' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-ddim',
    name: 'DDIM (Denoising Diffusion Implicit Model)',
    family: 'diffusion',
    description: 'DDIM — 50-1000 timesteps, deterministic reverse, non-Markovian, accelerates sampling 10-50×, same UNet as DDPM',
    tags: ['diffusion', 'ddim', 'deterministic', 'fast-sampling'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input (noise)', x: 50, y: 140, params: { channels: 3, height: 32, width: 32 } },
      { id: 'n2', type: 'linear', name: 'Time Embedding (256)', x: 250, y: 80, params: { hidden_size: 256 } },
      { id: 'n3', type: 'unet_encoder', name: 'UNet Encoder', x: 450, y: 140, params: { in_channels: 3, channels: [128, 256, 256], num_res_blocks: 2 } },
      { id: 'n4', type: 'ddim_step', name: 'DDIM Step (η=0)', x: 650, y: 140, params: { num_steps: 50, eta: 0.0 } },
      { id: 'n5', type: 'unet_decoder', name: 'UNet Decoder', x: 850, y: 140, params: { out_channels: 3, channels: [256, 256, 128], num_res_blocks: 2 } },
      { id: 'n6', type: 'output', name: 'Output', x: 1050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n3' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-stable-diffusion-v1',
    name: 'Stable Diffusion v1',
    family: 'diffusion',
    description: 'Stable Diffusion v1 — 860M UNet, 8× downsample VAE, CLIP text encoder, 512×512 latent diffusion, 1.45B total',
    tags: ['diffusion', 'stable-diffusion', 'latent', 'text-to-image'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 200, params: { max_length: 77 } },
      { id: 'n2', type: 'text_encoder', name: 'CLIP Text Encoder', x: 250, y: 200, params: { vocab_size: 49408, hidden_size: 768, num_layers: 12 } },
      { id: 'n3', type: 'vae_encoder', name: 'VAE Encoder (8× down)', x: 50, y: 140, params: { in_channels: 3, latent_channels: 4, compression: 8 } },
      { id: 'n4', type: 'unet_latent', name: 'UNet (860M, cross-attn)', x: 250, y: 140, params: { in_channels: 4, model_channels: 320, num_res_blocks: 2, channel_mult: [1, 2, 4, 4], num_heads: 8, cross_attn_dim: 768 } },
      { id: 'n5', type: 'noise_scheduler', name: 'Noise Scheduler (1000 steps)', x: 450, y: 80, params: { num_train_timesteps: 1000, beta_start: 0.00085, beta_end: 0.012, beta_schedule: 'scaled_linear' } },
      { id: 'n6', type: 'vae_decoder', name: 'VAE Decoder (8× up)', x: 450, y: 140, params: { latent_channels: 4, out_channels: 3, compression: 8 } },
      { id: 'n7', type: 'output', name: 'Generated Image 512×512', x: 650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n4' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n6' },
      { id: 'c5', from: 'n5', to: 'n4' }, { id: 'c6', from: 'n6', to: 'n7' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-stable-diffusion-xl',
    name: 'Stable Diffusion XL',
    family: 'diffusion',
    description: 'Stable Diffusion XL — 2.6B UNet, dual text encoders (CLIP + OpenCLIP), 1024×1024, refinement pipeline',
    tags: ['diffusion', 'stable-diffusion', 'sdxl', 'high-res'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 200, params: { max_length: 77 } },
      { id: 'n2', type: 'text_encoder', name: 'CLIP Text Encoder', x: 250, y: 200, params: { hidden_size: 768 } },
      { id: 'n3', type: 'text_encoder', name: 'OpenCLIP (1024 dim)', x: 450, y: 200, params: { hidden_size: 1024 } },
      { id: 'n4', type: 'input', name: 'Size Conditions', x: 50, y: 80, params: { target_size: 1024, crop_top: 0, crop_left: 0 } },
      { id: 'n5', type: 'concat', name: 'Concat Text + Pooled', x: 650, y: 200, params: {} },
      { id: 'n6', type: 'vae_encoder', name: 'VAE Encoder', x: 250, y: 140, params: { latent_channels: 4 } },
      { id: 'n7', type: 'unet_latent', name: 'UNet XL (2.6B)', x: 450, y: 140, params: { in_channels: 4, model_channels: 320, num_res_blocks: 2, channel_mult: [1, 2, 4, 4], num_heads_dim: 64, cross_attn_dim: 2048 } },
      { id: 'n8', type: 'refiner', name: 'Refiner UNet', x: 650, y: 80, params: { model_channels: 384 } },
      { id: 'n9', type: 'vae_decoder', name: 'VAE Decoder', x: 650, y: 140, params: { latent_channels: 4, out_channels: 3 } },
      { id: 'n10', type: 'output', name: 'Generated Image 1024×1024', x: 850, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n1', to: 'n3' },
      { id: 'c3', from: 'n2', to: 'n5' }, { id: 'c4', from: 'n3', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n7' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n4', to: 'n7' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-imagen',
    name: 'Imagen (Google)',
    family: 'diffusion',
    description: 'Imagen — 3 cascaded diffusion models (64→256→1024), frozen T5-XXL text encoder, 11B total params, dynamic thresholding',
    tags: ['diffusion', 'imagen', 'cascaded', 'text-to-image', 't5'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 140, params: { max_length: 128 } },
      { id: 'n2', type: 'text_encoder', name: 'T5-XXL Text Encoder', x: 250, y: 140, params: { vocab_size: 32128, hidden_size: 4096, num_layers: 24 } },
      { id: 'n3', type: 'unet_eff', name: 'Base UNet (64×64)', x: 450, y: 140, params: { in_channels: 3, channels: [128, 256, 512, 1024], num_res_blocks: 2 } },
      { id: 'n4', type: 'unet_eff', name: 'Super-Res UNet (64→256)', x: 650, y: 140, params: { in_channels: 6, channels: [128, 256, 256, 512], num_res_blocks: 2, conditioning: 'noise' } },
      { id: 'n5', type: 'unet_eff', name: 'Super-Res UNet (256→1024)', x: 850, y: 140, params: { in_channels: 6, channels: [64, 128, 256, 512], num_res_blocks: 2, conditioning: 'noise' } },
      { id: 'n6', type: 'output', name: 'Generated Image 1024×1024', x: 1050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-dalle-3',
    name: 'DALL-E 3',
    family: 'diffusion',
    description: 'DALL-E 3 — caption improvement, T5 text encoder, 1024×1024 output, improved prompt adherence, score distillation',
    tags: ['diffusion', 'dalle', 'text-to-image', 'caption-improvement'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 140, params: { max_length: 400 } },
      { id: 'n2', type: 'caption_refiner', name: 'Caption Improvement', x: 250, y: 140, params: { improvement_type: 'descriptive', verbose: true } },
      { id: 'n3', type: 'text_encoder', name: 'T5 Text Encoder', x: 450, y: 140, params: { hidden_size: 4096, num_layers: 24 } },
      { id: 'n4', type: 'unet_latent', name: 'Diffusion UNet', x: 650, y: 140, params: { model_channels: 320, channel_mult: [1, 2, 4, 4], num_heads: 16, cross_attn_dim: 4096 } },
      { id: 'n5', type: 'vae_decoder', name: 'Decoder', x: 850, y: 140, params: { latent_channels: 4, out_channels: 3 } },
      { id: 'n6', type: 'output', name: 'Generated Image', x: 1050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-midjourney',
    name: 'Midjourney (v6)',
    family: 'diffusion',
    description: 'Midjourney v6 — proprietary diffusion model, 3.5B estimated, 1024×1024, NF4 quantized, contrastive preference',
    tags: ['diffusion', 'midjourney', 'proprietary', 'aesthetic'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 140, params: { max_length: 256 } },
      { id: 'n2', type: 'text_encoder', name: 'CLIP + T5 Encoder', x: 250, y: 140, params: { clip_hidden: 768, t5_hidden: 2048 } },
      { id: 'n3', type: 'concat', name: 'Fuse Embeddings', x: 450, y: 140, params: { total_dim: 2816 } },
      { id: 'n4', type: 'unet_latent', name: 'UNet (3.5B)', x: 650, y: 140, params: { model_channels: 320, channel_mult: [1, 2, 4, 4, 4], num_res_blocks: 3 } },
      { id: 'n5', type: 'aesthetic_head', name: 'Aesthetic Scoring', x: 850, y: 140, params: { scoring_type: 'preference' } },
      { id: 'n6', type: 'vae_decoder', name: 'Decoder', x: 1050, y: 140, params: {} },
      { id: 'n7', type: 'output', name: 'Generated Image', x: 1250, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-flux',
    name: 'Flux',
    family: 'diffusion',
    description: 'Flux (Black Forest Labs) — 12B rectified flow transformer, 3.5B guidance, dual CLIP+T5, mmdit, 1024×1024, FP8',
    tags: ['diffusion', 'flux', 'rectified-flow', 'transformer', 'mmdit'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 140, params: { max_length: 512 } },
      { id: 'n2', type: 'text_encoder', name: 'T5-XXL + CLIP', x: 250, y: 200, params: { t5_hidden: 4096, clip_hidden: 768 } },
      { id: 'n3', type: 'input', name: 'Latent Noise', x: 50, y: 80, params: { latent_shape: [1, 64, 64, 16] } },
      { id: 'n4', type: 'pos_rope', name: 'RoPE (dual axes)', x: 250, y: 80, params: { hidden_size: 3072, theta: 10000.0, axes: ['height', 'width'] } },
      { id: 'n5', type: 'mmdit_block', name: 'MMDiT Block ×12 (3B)', x: 450, y: 140, params: { hidden_size: 3072, num_heads: 24, num_layers: 12, double_stream: true } },
      { id: 'n6', type: 'mmdit_block', name: 'MMDiT Block ×12 (3B) double', x: 650, y: 140, params: { hidden_size: 3072, num_heads: 24, num_layers: 12, double_stream: true } },
      { id: 'n7', type: 'linear', name: 'Proj Out → VAE Latents', x: 850, y: 140, params: { in_features: 3072, out_features: 16 } },
      { id: 'n8', type: 'reshape', name: 'Reshape 64×64×16', x: 1050, y: 140, params: { shape: [64, 64, 16] } },
      { id: 'n9', type: 'vae_decoder', name: 'VAE Decoder', x: 1250, y: 140, params: { latent_channels: 16, out_channels: 3 } },
      { id: 'n10', type: 'output', name: 'Generated Image', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n5' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' },
    ],
    defaultParams: {},
  },
  // ====== GNN (8) ======
  {
    id: 'tpl-gcn',
    name: 'GCN (Graph Convolutional Network)',
    family: 'gnn',
    description: 'GCN (Kipf & Welling 2017) — 2 layers, 16 hidden, ReLU, symmetric normalization, 10-class semi-supervised node classification, ~60K params on Cora',
    tags: ['gnn', 'gcn', 'node-classification', 'semi-supervised', 'kipf'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: { num_nodes: 2708, num_features: 1433 } },
      { id: 'n2', type: 'graph_conv', name: 'GCN Layer 1 (1433→16)', x: 250, y: 140, params: { in_features: 1433, out_features: 16, activation: 'relu', dropout: 0.5, normalization: 'symmetric' } },
      { id: 'n3', type: 'dropout', name: 'Dropout (0.5)', x: 450, y: 80, params: { rate: 0.5 } },
      { id: 'n4', type: 'graph_conv', name: 'GCN Layer 2 (16→10)', x: 450, y: 140, params: { in_features: 16, out_features: 10, activation: 'softmax', normalization: 'symmetric' } },
      { id: 'n5', type: 'output', name: 'Node Classes (10)', x: 650, y: 140, params: { num_classes: 10 } },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-gat',
    name: 'GAT (Graph Attention Network)',
    family: 'gnn',
    description: 'GAT (Velickovic 2018) — 2 layers, 8-head attention (64 hidden), ELU, 10-class node classification, ~80K params on Cora, 0.6 dropout',
    tags: ['gnn', 'gat', 'attention', 'node-classification', 'velickovic'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: { num_nodes: 2708, num_features: 1433 } },
      { id: 'n2', type: 'gat_attention', name: 'GAT Layer 1 (8 heads)', x: 250, y: 140, params: { in_features: 1433, out_features: 64, num_heads: 8, concat: true, activation: 'elu', dropout: 0.6 } },
      { id: 'n3', type: 'dropout', name: 'Dropout (0.6)', x: 450, y: 140, params: { rate: 0.6 } },
      { id: 'n4', type: 'gat_attention', name: 'GAT Layer 2 (1 head)', x: 650, y: 140, params: { in_features: 64, out_features: 10, num_heads: 1, concat: false, activation: 'softmax', dropout: 0.6 } },
      { id: 'n5', type: 'output', name: 'Node Classes (10)', x: 850, y: 140, params: { num_classes: 10 } },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
    ],
    defaultParams: {},
  },

  // ====== SSM (1 more) ======
    {
      id: 'tpl-vim',
      name: 'ViM (Vision Mamba)',
      family: 'ssm',
      description: 'Vision Mamba (Zhu 2024) — bidirectional SSM for image classification, 26M params, patch 16×16, 196 seq len, d_model 256',
      tags: ['ssm', 'vision', 'bidirectional', 'classification', 'zhu'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Image Input', x: 50, y: 140, params: {
            image_size: 224,
            channels: 3
          } },
        { id: 'n2', type: 'conv2d', name: 'Patch Embedding (16×16)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 256,
            kernel_size: 16,
            stride: 16
          } },
        { id: 'n3', type: 'linear', name: 'Linear Projection', x: 450, y: 140, params: {
            in_features: 256,
            out_features: 256
          } },
        { id: 'n4', type: 'layernorm', name: 'LayerNorm', x: 650, y: 140, params: {
            eps: 1e-06,
            hidden_size: 256
          } },
        { id: 'n5', type: 'layer_stack', name: '12× Vim Block', x: 850, y: 140, params: {
            num_layers: 12
          } },
        { id: 'n6', type: 'mamba_ssm', name: 'Forward SSM', x: 850, y: 80, params: {
            d_model: 256,
            d_state: 16,
            expand_factor: 2,
            dt_rank: 'auto',
            bias: false
          } },
        { id: 'n7', type: 'layernorm', name: 'Forward Norm', x: 1050, y: 80, params: {
            eps: 1e-06,
            hidden_size: 256
          } },
        { id: 'n8', type: 'mamba_ssm', name: 'Backward SSM', x: 850, y: 200, params: {
            d_model: 256,
            d_state: 16,
            expand_factor: 2,
            dt_rank: 'auto',
            bidirectional: true
          } },
        { id: 'n9', type: 'layernorm', name: 'Backward Norm', x: 1050, y: 200, params: {
            eps: 1e-06,
            hidden_size: 256
          } },
        { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1250, y: 140, params: {

          } },
        { id: 'n11', type: 'layernorm', name: 'Final LayerNorm', x: 1450, y: 140, params: {
            eps: 1e-06,
            hidden_size: 256
          } },
        { id: 'n12', type: 'linear', name: 'Classification Head', x: 1650, y: 140, params: {
            in_features: 256,
            out_features: 1000
          } },
        { id: 'n13', type: 'output', name: 'Output', x: 1850, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n11' },
        { id: 'c6', from: 'n11', to: 'n12' },
        { id: 'c7', from: 'n12', to: 'n13' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n7', to: 'n10' },
        { id: 'c10', from: 'n8', to: 'n9' },
        { id: 'c11', from: 'n9', to: 'n10' }
      ],
      defaultParams: {},
    },

  // ====== GNN (5 more) ======
    {
      id: 'tpl-graphsage',
      name: 'GraphSAGE',
      family: 'gnn',
      description: 'GraphSAGE (Hamilton 2017) — 2 layers, 256 hidden, mean aggregator, supervised, ~200K params on Reddit',
      tags: ['gnn', 'graphsage', 'inductive', 'hamilton'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: {
            num_features: 602
          } },
        { id: 'n2', type: 'linear', name: 'FC (602→256)', x: 250, y: 140, params: {
            in_features: 602,
            out_features: 256
          } },
        { id: 'n3', type: 'message_passing', name: 'SAGE Conv 1 (mean)', x: 450, y: 140, params: {
            aggregator: 'mean',
            in_features: 256,
            out_features: 256
          } },
        { id: 'n4', type: 'dropout', name: 'Dropout (0.3)', x: 650, y: 140, params: {
            rate: 0.3
          } },
        { id: 'n5', type: 'message_passing', name: 'SAGE Conv 2 (mean)', x: 850, y: 140, params: {
            aggregator: 'mean',
            in_features: 256,
            out_features: 256
          } },
        { id: 'n6', type: 'dropout', name: 'Dropout (0.3)', x: 1050, y: 140, params: {
            rate: 0.3
          } },
        { id: 'n7', type: 'linear', name: 'FC (256→41)', x: 1250, y: 140, params: {
            in_features: 256,
            out_features: 41
          } },
        { id: 'n8', type: 'output', name: 'Node Classes (41)', x: 1450, y: 140, params: {
            num_classes: 41
          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-gaan',
      name: 'GAAN (Graph Attention Attention Network)',
      family: 'gnn',
      description: 'GAAN (Zhang 2018) — multi-head attention with adaptive weights, 128 hidden, 4 heads, 2 layers, ~50K params',
      tags: ['gnn', 'gaan', 'attention', 'adaptive'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: {
            num_features: 1433
          } },
        { id: 'n2', type: 'gat_attention', name: 'GAAN Layer 1 (4 heads, adaptive)', x: 250, y: 140, params: {
            in_features: 1433,
            out_features: 128,
            num_heads: 4,
            adaptive: true,
            activation: 'elu',
            dropout: 0.6
          } },
        { id: 'n3', type: 'dropout', name: 'Dropout (0.6)', x: 450, y: 140, params: {
            rate: 0.6
          } },
        { id: 'n4', type: 'gat_attention', name: 'GAAN Layer 2 (1 head)', x: 650, y: 140, params: {
            in_features: 128,
            out_features: 10,
            num_heads: 1,
            adaptive: false,
            activation: 'softmax',
            dropout: 0.6
          } },
        { id: 'n5', type: 'output', name: 'Node Classes (10)', x: 850, y: 140, params: {
            num_classes: 10
          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-mpnn',
      name: 'MPNN (Message Passing Neural Network)',
      family: 'gnn',
      description: 'MPNN (Gilmer 2017) — 3-step message passing, 128 hidden, virtual node, ~100K params on QM9',
      tags: ['gnn', 'mpnn', 'message-passing', 'gilmer', 'qm9'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: {
            num_features: 11
          } },
        { id: 'n2', type: 'linear', name: 'FC (11→128)', x: 250, y: 140, params: {
            in_features: 11,
            out_features: 128
          } },
        { id: 'n3', type: 'message_passing', name: 'MPNN Step 1 (message)', x: 450, y: 140, params: {
            message_dim: 128,
            hidden_dim: 128,
            aggregator: 'sum',
            virtual_node: true
          } },
        { id: 'n4', type: 'message_passing', name: 'MPNN Step 2 (message)', x: 650, y: 140, params: {
            message_dim: 128,
            hidden_dim: 128,
            aggregator: 'sum',
            virtual_node: true
          } },
        { id: 'n5', type: 'message_passing', name: 'MPNN Step 3 (message)', x: 850, y: 140, params: {
            message_dim: 128,
            hidden_dim: 128,
            aggregator: 'sum',
            virtual_node: true
          } },
        { id: 'n6', type: 'graph_readout', name: 'Set2Set Readout', x: 1050, y: 140, params: {
            readout: 'set2set',
            steps: 3
          } },
        { id: 'n7', type: 'linear', name: 'FC (256→1)', x: 1250, y: 140, params: {
            in_features: 256,
            out_features: 1
          } },
        { id: 'n8', type: 'output', name: 'Graph Property', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-rgcn',
      name: 'R-GCN (Relational Graph Conv Network)',
      family: 'gnn',
      description: 'R-GCN (Schlichtkrull 2018) — 2 layers, 64 hidden, 10 relation types, basis decomposition (5 bases), ~40K params',
      tags: ['gnn', 'rgcn', 'relational', 'knowledge-graph', 'schlichtkrull'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: {
            num_features: 100
          } },
        { id: 'n2', type: 'graph_conv', name: 'R-GCN Layer 1 (10 relations, 5 bases)', x: 250, y: 140, params: {
            in_features: 100,
            out_features: 64,
            num_relations: 10,
            num_bases: 5,
            activation: 'relu',
            dropout: 0.2
          } },
        { id: 'n3', type: 'dropout', name: 'Dropout (0.2)', x: 450, y: 140, params: {
            rate: 0.2
          } },
        { id: 'n4', type: 'graph_conv', name: 'R-GCN Layer 2 (10 relations, 5 bases)', x: 650, y: 140, params: {
            in_features: 64,
            out_features: 50,
            num_relations: 10,
            num_bases: 5,
            activation: 'softmax'
          } },
        { id: 'n5', type: 'output', name: 'Entity Classes (50)', x: 850, y: 140, params: {
            num_classes: 50
          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-seal',
      name: 'SEAL (Subgraph Embedding + Attn Layers)',
      family: 'gnn',
      description: 'SEAL (Zhang 2021) — 3 layers GCN with DGCNN, labeling trick, sort pooling, link prediction, ~200K params',
      tags: ['gnn', 'seal', 'link-prediction', 'subgraph', 'zhang'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: {
            num_features: 100
          } },
        { id: 'n2', type: 'graph_conv', name: 'DGCNN Conv 1 (sort pool)', x: 250, y: 140, params: {
            in_features: 100,
            out_features: 64,
            k: 30,
            activation: 'relu'
          } },
        { id: 'n3', type: 'graph_conv', name: 'DGCNN Conv 2 (sort pool)', x: 450, y: 140, params: {
            in_features: 64,
            out_features: 64,
            k: 30,
            activation: 'relu'
          } },
        { id: 'n4', type: 'graph_conv', name: 'DGCNN Conv 3 (sort pool)', x: 650, y: 140, params: {
            in_features: 64,
            out_features: 32,
            k: 30,
            activation: 'relu'
          } },
        { id: 'n5', type: 'graph_pool', name: 'Sort Pooling (k=30)', x: 850, y: 140, params: {
            k: 30
          } },
        { id: 'n6', type: 'linear', name: 'FC (960→128)', x: 1050, y: 140, params: {
            in_features: 960,
            out_features: 128,
            activation: 'relu'
          } },
        { id: 'n7', type: 'linear', name: 'FC (128→2)', x: 1250, y: 140, params: {
            in_features: 128,
            out_features: 2
          } },
        { id: 'n8', type: 'output', name: 'Link Prediction', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

  // ====== MOE (8) ======
    {
      id: 'tpl-mixtral-8x7b',
      name: 'Mixtral 8×7B',
      family: 'moe',
      description: 'Mixtral 8×7B (Mistral AI 2023) — 46.7B params, 12.9B active, 32 layers, d_model 4096, 8 experts (top-2), SwiGLU, RoPE 1e6',
      tags: ['moe', 'mixtral', 'mistral', 'top-2', 'swiglu'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 32768
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 32000,
            hidden_size: 4096
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 4096,
            theta: 1000000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '32× MoE Layer', x: 650, y: 140, params: {
            num_layers: 32
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 4096,
            eps: 1e-05
          } },
        { id: 'n6', type: 'gqa_attention', name: 'GQA (32 heads, 8 KV)', x: 850, y: 80, params: {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 4096,
            eps: 1e-05
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-2 Router', x: 850, y: 200, params: {
            num_experts: 8,
            top_k: 2,
            hidden_size: 4096
          } },
        { id: 'n10', type: 'moe_layer', name: '8× SwiGLU Experts', x: 1050, y: 200, params: {
            num_experts: 8,
            top_k: 2,
            intermediate_size: 14336,
            hidden_size: 4096,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 4096,
            eps: 1e-05
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 32000,
            hidden_size: 4096
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-mixtral-8x22b',
      name: 'Mixtral 8×22B',
      family: 'moe',
      description: 'Mixtral 8×22B (Mistral AI 2024) — 141B params, 39B active, 56 layers, d_model 6144, 8 experts (top-2), SwiGLU, RoPE 1e6',
      tags: ['moe', 'mixtral', 'mistral', 'top-2', 'swiglu'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 65536
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 32000,
            hidden_size: 6144
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 6144,
            theta: 1000000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '56× MoE Layer', x: 650, y: 140, params: {
            num_layers: 56
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 6144,
            eps: 1e-05
          } },
        { id: 'n6', type: 'gqa_attention', name: 'GQA (48 heads, 8 KV)', x: 850, y: 80, params: {
            hidden_size: 6144,
            num_heads: 48,
            num_kv_heads: 8,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 6144,
            eps: 1e-05
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-2 Router', x: 850, y: 200, params: {
            num_experts: 8,
            top_k: 2,
            hidden_size: 6144
          } },
        { id: 'n10', type: 'moe_layer', name: '8× SwiGLU Experts', x: 1050, y: 200, params: {
            num_experts: 8,
            top_k: 2,
            intermediate_size: 21504,
            hidden_size: 6144,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 6144,
            eps: 1e-05
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 32000,
            hidden_size: 6144
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-deepseek-moe-16b',
      name: 'DeepSeek MoE 16B',
      family: 'moe',
      description: 'DeepSeek MoE 16B (DeepSeek 2024) — 16.3B params, 1.3B active, 28 layers, d_model 2048, 64 experts (top-6), fine-grained MoE',
      tags: ['moe', 'deepseek', 'top-6', 'fine-grained'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 4096
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 102400,
            hidden_size: 2048
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 2048,
            theta: 10000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '28× MoE Layer', x: 650, y: 140, params: {
            num_layers: 28
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n6', type: 'mha_attention', name: 'MHA (16 heads)', x: 850, y: 80, params: {
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-6 Router', x: 850, y: 200, params: {
            num_experts: 64,
            top_k: 6,
            hidden_size: 2048
          } },
        { id: 'n10', type: 'moe_layer', name: '64× FFN Experts', x: 1050, y: 200, params: {
            num_experts: 64,
            top_k: 6,
            intermediate_size: 2816,
            hidden_size: 2048,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 102400,
            hidden_size: 2048
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-deepseek-v2',
      name: 'DeepSeek-V2 (MoE)',
      family: 'moe',
      description: 'DeepSeek-V2 (DeepSeek 2024) — 236B params, 21B active, 60 layers, d_model 5120, MLA attention, 160 experts (top-6), RoPE 1e6',
      tags: ['moe', 'deepseek', 'v2', 'mla', 'top-6'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 4096
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 102400,
            hidden_size: 5120
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 5120,
            theta: 1000000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '60× MoE Layer', x: 650, y: 140, params: {
            num_layers: 60
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-MLA RMSNorm', x: 650, y: 80, params: {
            hidden_size: 5120,
            eps: 1e-06
          } },
        { id: 'n6', type: 'mha_attention', name: 'MLA (40 heads, 8 KV)', x: 850, y: 80, params: {
            hidden_size: 5120,
            num_heads: 40,
            num_kv_heads: 8,
            head_dim: 128,
            causal: true,
            mla: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 5120,
            eps: 1e-06
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-6 Router', x: 850, y: 200, params: {
            num_experts: 160,
            top_k: 6,
            hidden_size: 5120
          } },
        { id: 'n10', type: 'moe_layer', name: '160× FFN Experts', x: 1050, y: 200, params: {
            num_experts: 160,
            top_k: 6,
            intermediate_size: 1536,
            hidden_size: 5120,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 5120,
            eps: 1e-06
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 102400,
            hidden_size: 5120
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-deepseek-v3',
      name: 'DeepSeek-V3',
      family: 'moe',
      description: 'DeepSeek-V3 (DeepSeek 2024) — 671B params, 37B active, 61 layers, d_model 7168, MLA, 256 experts (top-8), Multi-Token Prediction (2 extra), RoPE 1e6',
      tags: ['moe', 'deepseek', 'v3', 'mla', 'top-8', 'multi-token'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 8192
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 151936,
            hidden_size: 7168
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 7168,
            theta: 1000000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '61× MoE Block', x: 650, y: 140, params: {
            num_layers: 61
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-MLA RMSNorm', x: 650, y: 80, params: {
            hidden_size: 7168,
            eps: 1e-06
          } },
        { id: 'n6', type: 'mha_attention', name: 'MLA (64 heads, 8 KV)', x: 850, y: 80, params: {
            hidden_size: 7168,
            num_heads: 64,
            num_kv_heads: 8,
            head_dim: 128,
            causal: true,
            mla: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 7168,
            eps: 1e-06
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-8 Router', x: 850, y: 200, params: {
            num_experts: 256,
            top_k: 8,
            hidden_size: 7168
          } },
        { id: 'n10', type: 'moe_layer', name: '256× SwiGLU Experts', x: 1050, y: 200, params: {
            num_experts: 256,
            top_k: 8,
            intermediate_size: 2048,
            hidden_size: 7168,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 7168,
            eps: 1e-06
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head (shared)', x: 1850, y: 140, params: {
            vocab_size: 151936,
            hidden_size: 7168
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-qwen-moe-a2.7b',
      name: 'Qwen1.5-A2.7B (MoE)',
      family: 'moe',
      description: 'Qwen1.5-A2.7B (Alibaba 2024) — 2.7B params, 1.3B active, 24 layers, d_model 2048, 60 experts (top-2), RoPE 1e6',
      tags: ['moe', 'qwen', 'alibaba', 'top-2'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 32768
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 151936,
            hidden_size: 2048
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 2048,
            theta: 1000000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '24× MoE Block', x: 650, y: 140, params: {
            num_layers: 24
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n6', type: 'gqa_attention', name: 'GQA (16 heads, 2 KV)', x: 850, y: 80, params: {
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-2 Router', x: 850, y: 200, params: {
            num_experts: 60,
            top_k: 2,
            hidden_size: 2048
          } },
        { id: 'n10', type: 'moe_layer', name: '60× Activated Experts', x: 1050, y: 200, params: {
            num_experts: 60,
            top_k: 2,
            intermediate_size: 2816,
            hidden_size: 2048,
            activation: 'silu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 151936,
            hidden_size: 2048
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-qwen2-moe',
      name: 'Qwen2-Ah-14B (MoE)',
      family: 'moe',
      description: 'Qwen2-Ah-14B (Alibaba 2025) — 14B params, ~5.9B active, 24 layers (14 MoE), d_model 2048, 60 experts (top-2), SwiGLU, RoPE 1e6',
      tags: ['moe', 'qwen2', 'ah', 'top-2', 'swiglu'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 32768
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 151936,
            hidden_size: 2048
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 2048,
            theta: 1000000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '24× MoE Block', x: 650, y: 140, params: {
            num_layers: 24
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n6', type: 'gqa_attention', name: 'GQA (16 heads, 2 KV)', x: 850, y: 80, params: {
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 2,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-2 Router', x: 850, y: 200, params: {
            num_experts: 60,
            top_k: 2,
            hidden_size: 2048
          } },
        { id: 'n10', type: 'moe_layer', name: '60× SwiGLU Experts', x: 1050, y: 200, params: {
            num_experts: 60,
            top_k: 2,
            intermediate_size: 2816,
            hidden_size: 2048,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 151936,
            hidden_size: 2048
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-dbrx-base',
      name: 'DBRX 132B',
      family: 'moe',
      description: 'DBRX 132B (MosaicML 2024) — 132B params, 36B active, 36 layers, d_model 4096, 32 heads, 16 experts (top-4), SwiGLU, RoPE 5e5',
      tags: ['moe', 'dbrx', 'mosaicml', 'top-4', 'swiglu'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 8192
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 100352,
            hidden_size: 4096
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 4096,
            theta: 500000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '36× MoE Block', x: 650, y: 140, params: {
            num_layers: 36
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 4096,
            eps: 1e-05
          } },
        { id: 'n6', type: 'gqa_attention', name: 'GQA (32 heads, 8 KV)', x: 850, y: 80, params: {
            hidden_size: 4096,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 4096,
            eps: 1e-05
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-4 Router', x: 850, y: 200, params: {
            num_experts: 16,
            top_k: 4,
            hidden_size: 4096
          } },
        { id: 'n10', type: 'moe_layer', name: '16× SwiGLU Experts', x: 1050, y: 200, params: {
            num_experts: 16,
            top_k: 4,
            intermediate_size: 16384,
            hidden_size: 4096,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 4096,
            eps: 1e-05
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 100352,
            hidden_size: 4096
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

  // ====== RNN (8) ======
    {
      id: 'tpl-lstm-base',
      name: 'BiLSTM Base',
      family: 'rnn',
      description: '2-layer Bidirectional LSTM, 512 hidden, 100K vocab, 300d embeddings, dropout 0.3, ~68M params',
      tags: ['rnn', 'lstm', 'bilstm', 'classification', 'nlp'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 512
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (300d)', x: 250, y: 140, params: {
            vocab_size: 100000,
            embedding_dim: 300
          } },
        { id: 'n3', type: 'dropout', name: 'Embedding Dropout (0.3)', x: 450, y: 140, params: {
            rate: 0.3
          } },
        { id: 'n4', type: 'bilstm', name: 'BiLSTM Layer 1 (512)', x: 650, y: 140, params: {
            hidden_size: 512,
            num_layers: 1,
            bidirectional: true,
            dropout: 0.3
          } },
        { id: 'n5', type: 'bilstm', name: 'BiLSTM Layer 2 (512)', x: 850, y: 140, params: {
            hidden_size: 512,
            num_layers: 1,
            bidirectional: true,
            dropout: 0.3
          } },
        { id: 'n6', type: 'concat', name: 'Concat Final States', x: 1050, y: 140, params: {
            num_directions: 2
          } },
        { id: 'n7', type: 'linear', name: 'FC (1024→256)', x: 1250, y: 140, params: {
            in_features: 1024,
            out_features: 256,
            activation: 'tanh'
          } },
        { id: 'n8', type: 'linear', name: 'FC (256→2)', x: 1450, y: 140, params: {
            in_features: 256,
            out_features: 2
          } },
        { id: 'n9', type: 'output', name: 'Output', x: 1650, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-gru-base',
      name: 'BiGRU Base',
      family: 'rnn',
      description: '2-layer Bidirectional GRU, 512 hidden, 100K vocab, 300d embeddings, dropout 0.3, ~52M params',
      tags: ['rnn', 'gru', 'bigru', 'classification', 'nlp'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 512
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (300d)', x: 250, y: 140, params: {
            vocab_size: 100000,
            embedding_dim: 300
          } },
        { id: 'n3', type: 'dropout', name: 'Embedding Dropout (0.3)', x: 450, y: 140, params: {
            rate: 0.3
          } },
        { id: 'n4', type: 'bigru', name: 'BiGRU Layer 1 (512)', x: 650, y: 140, params: {
            hidden_size: 512,
            num_layers: 1,
            bidirectional: true,
            dropout: 0.3
          } },
        { id: 'n5', type: 'bigru', name: 'BiGRU Layer 2 (256)', x: 850, y: 140, params: {
            hidden_size: 256,
            num_layers: 1,
            bidirectional: true,
            dropout: 0.3
          } },
        { id: 'n6', type: 'concat', name: 'Concat Final States', x: 1050, y: 140, params: {
            num_directions: 2
          } },
        { id: 'n7', type: 'linear', name: 'FC (512→128)', x: 1250, y: 140, params: {
            in_features: 512,
            out_features: 128,
            activation: 'tanh'
          } },
        { id: 'n8', type: 'linear', name: 'FC (128→2)', x: 1450, y: 140, params: {
            in_features: 128,
            out_features: 2
          } },
        { id: 'n9', type: 'output', name: 'Output', x: 1650, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-lstm-seq2seq',
      name: 'LSTM Seq2Seq + Attention',
      family: 'rnn',
      description: 'Encoder-decoder LSTM with Bahdanau attention, 4 layers, 512 hidden, 100K vocab, ~85M params',
      tags: ['rnn', 'lstm', 'seq2seq', 'attention', 'translation'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Source Input', x: 50, y: 140, params: {
            sequence_length: 128
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (512d)', x: 250, y: 140, params: {
            vocab_size: 100000,
            embedding_dim: 512
          } },
        { id: 'n3', type: 'lstm_cell', name: 'Encoder LSTM (4 layers, 512)', x: 450, y: 140, params: {
            hidden_size: 512,
            num_layers: 4,
            bidirectional: false,
            dropout: 0.3
          } },
        { id: 'n4', type: 'concat', name: 'Encoder States', x: 650, y: 140, params: {

          } },
        { id: 'n5', type: 'attention', name: 'Bahdanau Attention', x: 850, y: 140, params: {
            hidden_size: 512,
            attention_type: 'bahdanau'
          } },
        { id: 'n6', type: 'lstm_cell', name: 'Decoder LSTM (4 layers, 512)', x: 1050, y: 140, params: {
            hidden_size: 512,
            num_layers: 4,
            bidirectional: false,
            dropout: 0.3
          } },
        { id: 'n7', type: 'linear', name: 'FC (512→100K)', x: 1250, y: 140, params: {
            in_features: 512,
            out_features: 100000
          } },
        { id: 'n8', type: 'output', name: 'Target Output', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-gru-seq2seq',
      name: 'GRU Seq2Seq',
      family: 'rnn',
      description: 'GRU encoder-decoder, 512 hidden, 300d embeddings, 2 layers each, ~60M params',
      tags: ['rnn', 'gru', 'seq2seq', 'translation'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Source Input', x: 50, y: 140, params: {
            sequence_length: 128
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (300d)', x: 250, y: 140, params: {
            vocab_size: 100000,
            embedding_dim: 300
          } },
        { id: 'n3', type: 'gru_cell', name: 'Encoder GRU (2 layers, 512)', x: 450, y: 140, params: {
            hidden_size: 512,
            num_layers: 2,
            bidirectional: false,
            dropout: 0.3
          } },
        { id: 'n4', type: 'linear', name: 'Encoder FC (512→512)', x: 650, y: 140, params: {
            in_features: 512,
            out_features: 512
          } },
        { id: 'n5', type: 'gru_cell', name: 'Decoder GRU (2 layers, 512)', x: 850, y: 140, params: {
            hidden_size: 512,
            num_layers: 2,
            bidirectional: false,
            dropout: 0.3
          } },
        { id: 'n6', type: 'linear', name: 'FC (512→100K)', x: 1050, y: 140, params: {
            in_features: 512,
            out_features: 100000
          } },
        { id: 'n7', type: 'output', name: 'Target Output', x: 1250, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-indrnn',
      name: 'IndRNN',
      family: 'rnn',
      description: 'Independently Recurrent NN (Li 2018) — 6 layers, 128 hidden, ReLU, gradient clipping, ~500K params',
      tags: ['rnn', 'indrnn', 'independent', 'recurrent', 'li'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 128
          } },
        { id: 'n2', type: 'linear', name: 'FC (128→128)', x: 250, y: 140, params: {
            in_features: 128,
            out_features: 128
          } },
        { id: 'n3', type: 'temporal_conv', name: 'IndRNN Cell 1 (128)', x: 450, y: 140, params: {
            hidden_size: 128,
            activation: 'relu',
            recurrent_bias: 0.5
          } },
        { id: 'n4', type: 'temporal_conv', name: 'IndRNN Cell 2 (128)', x: 650, y: 140, params: {
            hidden_size: 128,
            activation: 'relu',
            recurrent_bias: 0.5
          } },
        { id: 'n5', type: 'temporal_conv', name: 'IndRNN Cell 3 (128)', x: 850, y: 140, params: {
            hidden_size: 128,
            activation: 'relu',
            recurrent_bias: 0.5
          } },
        { id: 'n6', type: 'temporal_conv', name: 'IndRNN Cell 4 (128)', x: 1050, y: 140, params: {
            hidden_size: 128,
            activation: 'relu',
            recurrent_bias: 0.5
          } },
        { id: 'n7', type: 'temporal_conv', name: 'IndRNN Cell 5 (128)', x: 1250, y: 140, params: {
            hidden_size: 128,
            activation: 'relu',
            recurrent_bias: 0.5
          } },
        { id: 'n8', type: 'temporal_conv', name: 'IndRNN Cell 6 (128)', x: 1450, y: 140, params: {
            hidden_size: 128,
            activation: 'relu',
            recurrent_bias: 0.5
          } },
        { id: 'n9', type: 'linear', name: 'FC (128→10)', x: 1650, y: 140, params: {
            in_features: 128,
            out_features: 10
          } },
        { id: 'n10', type: 'output', name: 'Output', x: 1850, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' },
        { id: 'c9', from: 'n9', to: 'n10' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-phaser-lstm',
      name: 'Phased LSTM',
      family: 'rnn',
      description: 'Phased LSTM (Neil 2016) — 3 layers, 256 hidden, multiple timescales (τ=4,8,16), rhythmic gates, ~3.3M params',
      tags: ['rnn', 'phaser-lstm', 'rhythmic', 'timescale', 'neil'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 1000
          } },
        { id: 'n2', type: 'linear', name: 'FC (100→256)', x: 250, y: 140, params: {
            in_features: 100,
            out_features: 256
          } },
        { id: 'n3', type: 'lstm_cell', name: 'Phased LSTM 1 (256, τ=4)', x: 450, y: 140, params: {
            hidden_size: 256,
            time_gate: true,
            period: 4,
            num_layers: 1,
            dropout: 0.2
          } },
        { id: 'n4', type: 'lstm_cell', name: 'Phased LSTM 2 (256, τ=8)', x: 650, y: 140, params: {
            hidden_size: 256,
            time_gate: true,
            period: 8,
            num_layers: 1,
            dropout: 0.2
          } },
        { id: 'n5', type: 'lstm_cell', name: 'Phased LSTM 3 (256, τ=16)', x: 850, y: 140, params: {
            hidden_size: 256,
            time_gate: true,
            period: 16,
            num_layers: 1,
            dropout: 0.2
          } },
        { id: 'n6', type: 'linear', name: 'FC (256→10)', x: 1050, y: 140, params: {
            in_features: 256,
            out_features: 10
          } },
        { id: 'n7', type: 'output', name: 'Output', x: 1250, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-sru',
      name: 'SRU (Simple Recurrent Unit)',
      family: 'rnn',
      description: 'SRU (Lei 2018) — 4 layers, 512 hidden, highway connections, parallelizable, ~8M params',
      tags: ['rnn', 'sru', 'simple-recurrent', 'highway', 'lei'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 512
          } },
        { id: 'n2', type: 'linear', name: 'FC (128→512)', x: 250, y: 140, params: {
            in_features: 128,
            out_features: 512
          } },
        { id: 'n3', type: 'temporal_conv', name: 'SRU Layer 1 (512, highway)', x: 450, y: 140, params: {
            hidden_size: 512,
            highway: true,
            dropout: 0.2
          } },
        { id: 'n4', type: 'temporal_conv', name: 'SRU Layer 2 (512, highway)', x: 650, y: 140, params: {
            hidden_size: 512,
            highway: true,
            dropout: 0.2
          } },
        { id: 'n5', type: 'temporal_conv', name: 'SRU Layer 3 (512, highway)', x: 850, y: 140, params: {
            hidden_size: 512,
            highway: true,
            dropout: 0.2
          } },
        { id: 'n6', type: 'temporal_conv', name: 'SRU Layer 4 (512, highway)', x: 1050, y: 140, params: {
            hidden_size: 512,
            highway: true,
            dropout: 0.2
          } },
        { id: 'n7', type: 'linear', name: 'FC (512→10)', x: 1250, y: 140, params: {
            in_features: 512,
            out_features: 10
          } },
        { id: 'n8', type: 'output', name: 'Output', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-janet',
      name: 'JANET (Just Another Net)',
      family: 'rnn',
      description: 'JANET (van der Westhuizen 2018) — 3 layers, 512 hidden, forget bias 1.0, gated recurrent with simplified gate, ~6M params',
      tags: ['rnn', 'janet', 'gated', 'forget-bias', 'simplified'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 512
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (256d)', x: 250, y: 140, params: {
            vocab_size: 50000,
            embedding_dim: 256
          } },
        { id: 'n3', type: 'lstm_cell', name: 'JANET Layer 1 (512, forget=1.0)', x: 450, y: 140, params: {
            hidden_size: 512,
            forget_bias: 1.0,
            num_layers: 1,
            dropout: 0.3,
            gate_type: 'janet'
          } },
        { id: 'n4', type: 'lstm_cell', name: 'JANET Layer 2 (256, forget=1.0)', x: 650, y: 140, params: {
            hidden_size: 256,
            forget_bias: 1.0,
            num_layers: 1,
            dropout: 0.3,
            gate_type: 'janet'
          } },
        { id: 'n5', type: 'lstm_cell', name: 'JANET Layer 3 (128, forget=1.0)', x: 850, y: 140, params: {
            hidden_size: 128,
            forget_bias: 1.0,
            num_layers: 1,
            dropout: 0.3,
            gate_type: 'janet'
          } },
        { id: 'n6', type: 'linear', name: 'FC (128→10)', x: 1050, y: 140, params: {
            in_features: 128,
            out_features: 10
          } },
        { id: 'n7', type: 'output', name: 'Output', x: 1250, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' }
      ],
      defaultParams: {},
    },

  // ====== GAN (8) ======
    {
      id: 'tpl-dcgan',
      name: 'DCGAN',
      family: 'gan',
      description: 'DCGAN (Radford 2016) — 4-layer CNN generator/discriminator, 100-dim latent z, 64 base filters, 32×32 images, 3.2M params',
      tags: ['gan', 'dcgan', 'conv', 'unsupervised', 'radford'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=100)', x: 50, y: 140, params: {
            latent_dim: 100
          } },
        { id: 'n2', type: 'linear', name: 'FC (100→4×4×512)', x: 250, y: 140, params: {
            in_features: 100,
            out_features: 8192
          } },
        { id: 'n3', type: 'reshape', name: 'Reshape (512×4×4)', x: 450, y: 140, params: {
            shape: [512, 4, 4]
          } },
        { id: 'n4', type: 'conv2d', name: 'Deconv 1 (512→256, 5×5, stride 2)', x: 650, y: 140, params: {
            in_channels: 512,
            out_channels: 256,
            kernel_size: 5,
            stride: 2,
            padding: 2
          } },
        { id: 'n5', type: 'batchnorm', name: 'BatchNorm + ReLU', x: 850, y: 140, params: {

          } },
        { id: 'n6', type: 'conv2d', name: 'Deconv 2 (256→128, 5×5, stride 2)', x: 1050, y: 140, params: {
            in_channels: 256,
            out_channels: 128,
            kernel_size: 5,
            stride: 2,
            padding: 2
          } },
        { id: 'n7', type: 'batchnorm', name: 'BatchNorm + ReLU', x: 1250, y: 140, params: {

          } },
        { id: 'n8', type: 'conv2d', name: 'Deconv 3 (128→64, 5×5, stride 2)', x: 1450, y: 140, params: {
            in_channels: 128,
            out_channels: 64,
            kernel_size: 5,
            stride: 2,
            padding: 2
          } },
        { id: 'n9', type: 'batchnorm', name: 'BatchNorm + ReLU', x: 1650, y: 140, params: {

          } },
        { id: 'n10', type: 'conv2d', name: 'Deconv 4 (64→3, 5×5, stride 2)', x: 1850, y: 140, params: {
            in_channels: 64,
            out_channels: 3,
            kernel_size: 5,
            stride: 2,
            padding: 2,
            activation: 'tanh'
          } },
        { id: 'n11', type: 'output', name: 'Generated Image (32×32)', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' },
        { id: 'c9', from: 'n9', to: 'n10' },
        { id: 'c10', from: 'n10', to: 'n11' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-sngan',
      name: 'SNGAN (Spectral Norm GAN)',
      family: 'gan',
      description: 'Spectral Norm GAN (Miyato 2018) — ResNet generator, SN on all conv layers, 128-dim latent, 128×128, ~22M params',
      tags: ['gan', 'sngan', 'spectral-norm', 'resnet', 'miyato'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=128)', x: 50, y: 140, params: {
            latent_dim: 128
          } },
        { id: 'n2', type: 'linear', name: 'FC (128→4×4×512)', x: 250, y: 140, params: {
            in_features: 128,
            out_features: 8192
          } },
        { id: 'n3', type: 'reshape', name: 'Reshape (512×4×4)', x: 450, y: 140, params: {
            shape: [512, 4, 4]
          } },
        { id: 'n4', type: 'conv2d', name: 'ResBlock Up 1 (512→256)', x: 650, y: 140, params: {
            in_channels: 512,
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            spectral_norm: true,
            upsample: 2
          } },
        { id: 'n5', type: 'conv2d', name: 'ResBlock Up 2 (256→128)', x: 850, y: 140, params: {
            in_channels: 256,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            spectral_norm: true,
            upsample: 2
          } },
        { id: 'n6', type: 'conv2d', name: 'ResBlock Up 3 (128→64)', x: 1050, y: 140, params: {
            in_channels: 128,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            spectral_norm: true,
            upsample: 2
          } },
        { id: 'n7', type: 'batchnorm', name: 'BatchNorm + ReLU', x: 1250, y: 140, params: {

          } },
        { id: 'n8', type: 'conv2d', name: 'Conv Out (64→3, 3×3)', x: 1450, y: 140, params: {
            in_channels: 64,
            out_channels: 3,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            activation: 'tanh'
          } },
        { id: 'n9', type: 'output', name: 'Generated Image (128×128)', x: 1650, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-sagan',
      name: 'SAGAN (Self-Attention GAN)',
      family: 'gan',
      description: 'Self-Attention GAN (Zhang 2019) — ResNet, attention at 32×32, 2 critic updates per generator step, spectral norm, 64×64',
      tags: ['gan', 'sagan', 'self-attention', 'spectral-norm', 'zhang'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=128)', x: 50, y: 140, params: {
            latent_dim: 128
          } },
        { id: 'n2', type: 'linear', name: 'FC (128→4×4×512)', x: 250, y: 140, params: {
            in_features: 128,
            out_features: 8192
          } },
        { id: 'n3', type: 'reshape', name: 'Reshape (512×4×4)', x: 450, y: 140, params: {
            shape: [512, 4, 4]
          } },
        { id: 'n4', type: 'conv2d', name: 'ResBlock Up 1 (512→256)', x: 650, y: 140, params: {
            in_channels: 512,
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            spectral_norm: true,
            upsample: 2
          } },
        { id: 'n5', type: 'self_attention', name: 'Self-Attention (32×32)', x: 850, y: 140, params: {
            in_channels: 256,
            attention_dim: 128,
            num_heads: 4
          } },
        { id: 'n6', type: 'conv2d', name: 'ResBlock Up 2 (256→128)', x: 1050, y: 140, params: {
            in_channels: 256,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            spectral_norm: true,
            upsample: 2
          } },
        { id: 'n7', type: 'batchnorm', name: 'BatchNorm + ReLU', x: 1250, y: 140, params: {

          } },
        { id: 'n8', type: 'conv2d', name: 'Conv Out (128→3)', x: 1450, y: 140, params: {
            in_channels: 128,
            out_channels: 3,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            activation: 'tanh'
          } },
        { id: 'n9', type: 'output', name: 'Generated Image (64×64)', x: 1650, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-stylegan2',
      name: 'StyleGAN2',
      family: 'gan',
      description: 'StyleGAN2 (Karras 2020) — 30M params, 8-layer mapping network (512), 14 synthesis blocks (4×4→1024×1024), AdaIN, noise injection',
      tags: ['gan', 'stylegan2', 'karras', 'adain', 'noise', 'progressive'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=512)', x: 50, y: 140, params: {
            latent_dim: 512
          } },
        { id: 'n2', type: 'linear', name: 'Mapping Network (8×512)', x: 250, y: 140, params: {
            in_features: 512,
            out_features: 512,
            num_layers: 8
          } },
        { id: 'n3', type: 'linear', name: 'FC (512→4×4×512)', x: 450, y: 140, params: {
            in_features: 512,
            out_features: 8192
          } },
        { id: 'n4', type: 'reshape', name: 'Reshape (512×4×4)', x: 650, y: 140, params: {
            shape: [512, 4, 4]
          } },
        { id: 'n5', type: 'synthesis_block', name: 'Synthesis Block (4×4)', x: 850, y: 140, params: {
            resolution: 4,
            in_channels: 512,
            out_channels: 512,
            demodulate: true
          } },
        { id: 'n6', type: 'synthesis_block', name: 'Synthesis Block (8×8)', x: 1050, y: 140, params: {
            resolution: 8,
            in_channels: 512,
            out_channels: 256,
            demodulate: true
          } },
        { id: 'n7', type: 'synthesis_block', name: 'Synthesis Block (16×16)', x: 1250, y: 140, params: {
            resolution: 16,
            in_channels: 256,
            out_channels: 128,
            demodulate: true
          } },
        { id: 'n8', type: 'synthesis_block', name: 'Synthesis Block (32×32)', x: 1450, y: 140, params: {
            resolution: 32,
            in_channels: 128,
            out_channels: 64,
            demodulate: true
          } },
        { id: 'n9', type: 'synthesis_block', name: 'Synthesis Block (64×64)', x: 1650, y: 140, params: {
            resolution: 64,
            in_channels: 64,
            out_channels: 32,
            demodulate: true
          } },
        { id: 'n10', type: 'to_rgb', name: 'ToRGB (32→3)', x: 1850, y: 140, params: {
            in_channels: 32,
            out_channels: 3
          } },
        { id: 'n11', type: 'output', name: 'Generated Image', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' },
        { id: 'c9', from: 'n9', to: 'n10' },
        { id: 'c10', from: 'n10', to: 'n11' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-stylegan3',
      name: 'StyleGAN3',
      family: 'gan',
      description: 'StyleGAN3 (Karras 2021) — 30M params, alias-free, Fourier features, 14 synthesis blocks, equivariant, 1024×1024',
      tags: ['gan', 'stylegan3', 'karras', 'alias-free', 'equivariant', 'fourier'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=512)', x: 50, y: 140, params: {
            latent_dim: 512
          } },
        { id: 'n2', type: 'linear', name: 'Mapping Network (8×512)', x: 250, y: 140, params: {
            in_features: 512,
            out_features: 512,
            num_layers: 8
          } },
        { id: 'n3', type: 'linear', name: 'Fourier Features', x: 450, y: 140, params: {
            in_features: 512,
            out_features: 512,
            fourier: true
          } },
        { id: 'n4', type: 'linear', name: 'FC (512→4×4×512)', x: 650, y: 140, params: {
            in_features: 512,
            out_features: 8192
          } },
        { id: 'n5', type: 'reshape', name: 'Reshape (512×4×4)', x: 850, y: 140, params: {
            shape: [512, 4, 4]
          } },
        { id: 'n6', type: 'synthesis_block', name: 'Synthesis Block (4×4)', x: 1050, y: 140, params: {
            resolution: 4,
            in_channels: 512,
            out_channels: 512,
            filter: 'alias_free'
          } },
        { id: 'n7', type: 'synthesis_block', name: 'Synthesis Block (8×8)', x: 1250, y: 140, params: {
            resolution: 8,
            in_channels: 512,
            out_channels: 256,
            filter: 'alias_free'
          } },
        { id: 'n8', type: 'synthesis_block', name: 'Synthesis Block (16×16)', x: 1450, y: 140, params: {
            resolution: 16,
            in_channels: 256,
            out_channels: 128,
            filter: 'alias_free'
          } },
        { id: 'n9', type: 'synthesis_block', name: 'Synthesis Block (32×32)', x: 1650, y: 140, params: {
            resolution: 32,
            in_channels: 128,
            out_channels: 64,
            filter: 'alias_free'
          } },
        { id: 'n10', type: 'to_rgb', name: 'ToRGB (64→3)', x: 1850, y: 140, params: {
            in_channels: 64,
            out_channels: 3
          } },
        { id: 'n11', type: 'output', name: 'Generated Image', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' },
        { id: 'c9', from: 'n9', to: 'n10' },
        { id: 'c10', from: 'n10', to: 'n11' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-progan',
      name: 'ProGAN (Progressive Growing GAN)',
      family: 'gan',
      description: 'Progressive Growing GAN (Karras 2018) — 4×4→1024×1024 progressive, 8 stages, fade-in, minibatch std, ~46M params',
      tags: ['gan', 'progan', 'progressive', 'karras', 'minibatch-std'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=512)', x: 50, y: 140, params: {
            latent_dim: 512
          } },
        { id: 'n2', type: 'linear', name: 'FC (512→4×4×512)', x: 250, y: 140, params: {
            in_features: 512,
            out_features: 8192
          } },
        { id: 'n3', type: 'reshape', name: 'Reshape (512×4×4)', x: 450, y: 140, params: {
            shape: [512, 4, 4]
          } },
        { id: 'n4', type: 'conv2d', name: 'Block 1 (4×4, 512)', x: 650, y: 140, params: {
            in_channels: 512,
            out_channels: 512,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n5', type: 'conv2d', name: 'Block 2 (8×8, 512→256)', x: 850, y: 140, params: {
            in_channels: 512,
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            upsample: 2
          } },
        { id: 'n6', type: 'conv2d', name: 'Block 3 (16×16, 256→128)', x: 1050, y: 140, params: {
            in_channels: 256,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            upsample: 2
          } },
        { id: 'n7', type: 'conv2d', name: 'Block 4 (32×32, 128→64)', x: 1250, y: 140, params: {
            in_channels: 128,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            upsample: 2
          } },
        { id: 'n8', type: 'conv2d', name: 'Block 5 (64×64, 64→32)', x: 1450, y: 140, params: {
            in_channels: 64,
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            upsample: 2
          } },
        { id: 'n9', type: 'conv2d', name: 'ToRGB (32→3)', x: 1650, y: 140, params: {
            in_channels: 32,
            out_channels: 3,
            kernel_size: 1,
            stride: 1
          } },
        { id: 'n10', type: 'output', name: 'Generated Image (64×64)', x: 1850, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' },
        { id: 'c9', from: 'n9', to: 'n10' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-cyclegan',
      name: 'CycleGAN',
      family: 'gan',
      description: 'CycleGAN (Zhu 2017) — 9 ResBlocks, 256×256 input, identity loss, cycle consistency, two GANs (G+F), ~11M params per generator',
      tags: ['gan', 'cyclegan', 'zhu', 'unpaired', 'translation'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Source Image (256×256)', x: 50, y: 140, params: {
            image_size: 256,
            channels: 3
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv IN (3→64, 7×7, stride 1)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 64,
            kernel_size: 7,
            stride: 1,
            padding: 3,
            activation: 'relu'
          } },
        { id: 'n3', type: 'conv2d', name: 'Down 1 (64→128, 3×3, stride 2)', x: 450, y: 140, params: {
            in_channels: 64,
            out_channels: 128,
            kernel_size: 3,
            stride: 2,
            padding: 1
          } },
        { id: 'n4', type: 'conv2d', name: 'Down 2 (128→256, 3×3, stride 2)', x: 650, y: 140, params: {
            in_channels: 128,
            out_channels: 256,
            kernel_size: 3,
            stride: 2,
            padding: 1
          } },
        { id: 'n5', type: 'residual_add', name: '9× ResBlock (256)', x: 850, y: 140, params: {
            num_blocks: 9,
            hidden_size: 256
          } },
        { id: 'n6', type: 'conv2d', name: 'Up 1 (256→128, 3×3, stride 2)', x: 1050, y: 140, params: {
            in_channels: 256,
            out_channels: 128,
            kernel_size: 3,
            stride: 2,
            padding: 1,
            output_padding: 1
          } },
        { id: 'n7', type: 'conv2d', name: 'Up 2 (128→64, 3×3, stride 2)', x: 1250, y: 140, params: {
            in_channels: 128,
            out_channels: 64,
            kernel_size: 3,
            stride: 2,
            padding: 1,
            output_padding: 1
          } },
        { id: 'n8', type: 'conv2d', name: 'Conv Out (64→3, 7×7, tanh)', x: 1450, y: 140, params: {
            in_channels: 64,
            out_channels: 3,
            kernel_size: 7,
            stride: 1,
            padding: 3,
            activation: 'tanh'
          } },
        { id: 'n9', type: 'output', name: 'Generated Image', x: 1650, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-biggan',
      name: 'BigGAN-deep',
      family: 'gan',
      description: 'BigGAN-deep (Brock 2019) — 80M params, class-conditional (shared embeddings), spectral norm, 128×128, 2 discriminator updates per gen step',
      tags: ['gan', 'biggan', 'class-conditional', 'brock', 'spectral-norm'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=120)', x: 50, y: 140, params: {
            latent_dim: 120
          } },
        { id: 'n2', type: 'embedding', name: 'Class Embedding (1000→128)', x: 250, y: 140, params: {
            num_classes: 1000,
            embedding_dim: 128
          } },
        { id: 'n3', type: 'concat', name: 'Concat (z + class embed)', x: 450, y: 140, params: {
            dim: 1
          } },
        { id: 'n4', type: 'linear', name: 'FC (248→4×4×512)', x: 650, y: 140, params: {
            in_features: 248,
            out_features: 8192
          } },
        { id: 'n5', type: 'reshape', name: 'Reshape (512×4×4)', x: 850, y: 140, params: {
            shape: [512, 4, 4]
          } },
        { id: 'n6', type: 'conv2d', name: 'ResBlock Up 1 (512→256)', x: 1050, y: 140, params: {
            in_channels: 512,
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            spectral_norm: true,
            class_conditional: true,
            upsample: 2
          } },
        { id: 'n7', type: 'conv2d', name: 'ResBlock Up 2 (256→128)', x: 1250, y: 140, params: {
            in_channels: 256,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            spectral_norm: true,
            class_conditional: true,
            upsample: 2
          } },
        { id: 'n8', type: 'conv2d', name: 'ResBlock Up 3 (128→64)', x: 1450, y: 140, params: {
            in_channels: 128,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            spectral_norm: true,
            class_conditional: true,
            upsample: 2
          } },
        { id: 'n9', type: 'batchnorm', name: 'BatchNorm + ReLU', x: 1650, y: 140, params: {

          } },
        { id: 'n10', type: 'conv2d', name: 'Conv Out (64→3, 3×3, tanh)', x: 1850, y: 140, params: {
            in_channels: 64,
            out_channels: 3,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            activation: 'tanh'
          } },
        { id: 'n11', type: 'output', name: 'Generated Image (128×128)', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n1', to: 'n3' },
        { id: 'c3', from: 'n2', to: 'n3' },
        { id: 'c4', from: 'n3', to: 'n4' },
        { id: 'c5', from: 'n4', to: 'n5' },
        { id: 'c6', from: 'n5', to: 'n6' },
        { id: 'c7', from: 'n6', to: 'n7' },
        { id: 'c8', from: 'n7', to: 'n8' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' }
      ],
      defaultParams: {},
    },

  {
    id: 'tpl-gin',
    name: 'GIN (Graph Isomorphism Network)',
    family: 'gnn',
    description: 'GIN (Xu et al 2019) — 5 layers, 64 hidden, eps=0 (learnable), MLP 2-layer aggregator, sum pooling, graph classification, ~60K params',
    tags: ['gnn', 'gin', 'graph-classification', 'expressive', 'wl-test'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: { num_features: 64 } },
      { id: 'n2', type: 'message_passing', name: 'GIN Conv 1 (64->64)', x: 250, y: 140, params: { in_features: 64, out_features: 64, aggregator: 'sum', mlp_layers: 2, mlp_hidden: 128, eps: 0.0, trainable_eps: true } },
      { id: 'n3', type: 'message_passing', name: 'GIN Conv 2 (64->64)', x: 450, y: 140, params: { in_features: 64, out_features: 64, aggregator: 'sum', mlp_layers: 2, mlp_hidden: 128 } },
      { id: 'n4', type: 'message_passing', name: 'GIN Conv 3 (64->64)', x: 650, y: 140, params: { in_features: 64, out_features: 64, aggregator: 'sum', mlp_layers: 2, mlp_hidden: 128 } },
      { id: 'n5', type: 'message_passing', name: 'GIN Conv 4 (64->64)', x: 850, y: 140, params: { in_features: 64, out_features: 64, aggregator: 'sum', mlp_layers: 2, mlp_hidden: 128 } },
      { id: 'n6', type: 'message_passing', name: 'GIN Conv 5 (64->64)', x: 1050, y: 140, params: { in_features: 64, out_features: 64, aggregator: 'sum', mlp_layers: 2, mlp_hidden: 128 } },
      { id: 'n7', type: 'graph_readout', name: 'Concat Readout', x: 1250, y: 140, params: { readout: 'sum', pool_all_layers: true } },
      { id: 'n8', type: 'linear', name: 'FC (320->2)', x: 1450, y: 140, params: { in_features: 320, out_features: 2 } },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' },
    ],
    defaultParams: {},
  },

  // ====== TRANSFORMER — batch 2 (2026-09 research pass) ======
{
    id: 'tpl-gpt3-175b',
    name: 'GPT-3 175B',
    family: 'transformer',
    description: 'OpenAI GPT-3 (Brown et al. 2020, Table 2.1) — 175B params, 96 layers, 12288 hidden, 96 heads, learned absolute positions, standard GELU FFN — no RoPE, no GQA, unlike every LLaMA-family entry in this catalogue',
    tags: ['decoder-only', 'autoregressive', 'llm', 'absolute-position'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50257, hidden_size: 12288 } },
      { id: 'n3', type: 'pos_absolute', name: 'Learned Positional Embedding', x: 450, y: 140, params: { max_length: 2048, hidden_size: 12288 } },
      { id: 'n4', type: 'layer_stack', name: '96× Decoder Block', x: 650, y: 140, params: { num_layers: 96 } },
      { id: 'n5', type: 'layernorm', name: 'Pre-Attention LayerNorm', x: 650, y: 80, params: { hidden_size: 12288, eps: 1e-5 } },
      { id: 'n6', type: 'mha_attention', name: 'Multi-Head Self-Attention', x: 850, y: 80, params: { hidden_size: 12288, num_heads: 96, num_kv_heads: 96, head_dim: 128, causal: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'layernorm', name: 'Pre-FFN LayerNorm', x: 650, y: 200, params: { hidden_size: 12288, eps: 1e-5 } },
      { id: 'n9', type: 'ffn_standard', name: 'FFN (2× Linear+GELU)', x: 850, y: 200, params: { hidden_size: 12288, intermediate_size: 49152, activation: 'gelu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'layernorm', name: 'Final LayerNorm', x: 1250, y: 140, params: { hidden_size: 12288, eps: 1e-5 } },
      { id: 'n12', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 50257, hidden_size: 12288 } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-t5-large',
    name: 'T5-Large',
    family: 'transformer',
    description: 'Google T5-Large — 770M params, 24 encoder + 24 decoder layers, 1024 hidden, 16 heads (head_dim 64), encoder-decoder — the only encoder-decoder entry in this catalogue, every other Transformer template here is decoder-only or encoder-only. Relative-position attention bias (T5-style) is not modelled — its own parameter table is negligible next to 770M, unlike a real positional-embedding matrix. Standard ReLU FFN (T5 v1.0, not the gated-GELU v1.1 variant).',
    tags: ['encoder-decoder', 'seq2seq', 'translation', 'summarization'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 512 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 32128, hidden_size: 1024 } },
      { id: 'n3', type: 'layer_stack', name: '24× Encoder Block', x: 450, y: 140, params: { num_layers: 24 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Attention Norm', x: 450, y: 60, params: { hidden_size: 1024, eps: 1e-6 } },
      { id: 'n5', type: 'attention', name: 'Self-Attention (bidirectional)', x: 600, y: 60, params: { hidden_size: 1024, num_heads: 16, head_dim: 64, causal: false } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 750, y: 60, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Pre-FFN Norm', x: 450, y: 220, params: { hidden_size: 1024, eps: 1e-6 } },
      { id: 'n8', type: 'ffn_standard', name: 'FFN (2× Linear+ReLU)', x: 600, y: 220, params: { hidden_size: 1024, intermediate_size: 4096, activation: 'relu' } },
      { id: 'n9', type: 'residual_add', name: 'Residual Add', x: 750, y: 220, params: {} },
      { id: 'n10', type: 'rmsnorm', name: 'Encoder Final Norm', x: 950, y: 140, params: { hidden_size: 1024, eps: 1e-6 } },
      { id: 'n11', type: 'layer_stack', name: '24× Decoder Block', x: 1150, y: 140, params: { num_layers: 24 } },
      { id: 'n12', type: 'rmsnorm', name: 'Pre-Self-Attention Norm', x: 1150, y: 20, params: { hidden_size: 1024, eps: 1e-6 } },
      { id: 'n13', type: 'attention', name: 'Causal Self-Attention', x: 1300, y: 20, params: { hidden_size: 1024, num_heads: 16, head_dim: 64, causal: true, encoder_decoder_role: 'decoder' } },
      { id: 'n14', type: 'residual_add', name: 'Residual Add', x: 1450, y: 20, params: {} },
      { id: 'n15', type: 'rmsnorm', name: 'Pre-Cross-Attention Norm', x: 1150, y: 140, params: { hidden_size: 1024, eps: 1e-6 } },
      { id: 'n16', type: 'cross_attention', name: 'Cross-Attention', x: 1300, y: 140, params: { hidden_size: 1024, num_heads: 16, encoder_decoder_role: 'decoder' } },
      { id: 'n17', type: 'residual_add', name: 'Residual Add', x: 1450, y: 140, params: {} },
      { id: 'n18', type: 'rmsnorm', name: 'Pre-FFN Norm', x: 1150, y: 260, params: { hidden_size: 1024, eps: 1e-6 } },
      { id: 'n19', type: 'ffn_standard', name: 'FFN (2× Linear+ReLU)', x: 1300, y: 260, params: { hidden_size: 1024, intermediate_size: 4096, activation: 'relu', encoder_decoder_role: 'decoder' } },
      { id: 'n20', type: 'residual_add', name: 'Residual Add', x: 1450, y: 260, params: {} },
      { id: 'n21', type: 'rmsnorm', name: 'Decoder Final Norm', x: 1650, y: 140, params: { hidden_size: 1024, eps: 1e-6 } },
      { id: 'n22', type: 'lm_head', name: 'LM Head (tied)', x: 1850, y: 140, params: { vocab_size: 32128, hidden_size: 1024, tie_weights: true } },
      { id: 'n23', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n10' }, { id: 'c4', from: 'n10', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n21' }, { id: 'c6', from: 'n21', to: 'n22' },
      { id: 'c7', from: 'n22', to: 'n23' },
      { id: 'c8', from: 'n4', to: 'n5' }, { id: 'c9', from: 'n5', to: 'n6' }, { id: 'c10', from: 'n6', to: 'n3' },
      { id: 'c11', from: 'n7', to: 'n8' }, { id: 'c12', from: 'n8', to: 'n9' }, { id: 'c13', from: 'n9', to: 'n3' },
      { id: 'c14', from: 'n12', to: 'n13' }, { id: 'c15', from: 'n13', to: 'n14' }, { id: 'c16', from: 'n14', to: 'n11' },
      { id: 'c17', from: 'n15', to: 'n16' }, { id: 'c18', from: 'n16', to: 'n17' }, { id: 'c19', from: 'n17', to: 'n11' },
      { id: 'c20', from: 'n18', to: 'n19' }, { id: 'c21', from: 'n19', to: 'n20' }, { id: 'c22', from: 'n20', to: 'n11' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-gemma-7b',
    name: 'Gemma 7B',
    family: 'transformer',
    description: 'Google Gemma 7B — 8.5B params, 28 layers, 3072 hidden, 16 heads, head_dim 256 (widened — 16×256=4096≠3072, a real non-square projection, not a rounding artifact), RoPE, RMSNorm, GeGLU FFN, 256K vocab (tied embedding)',
    tags: ['decoder-only', 'autoregressive', 'llm', 'geglu'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 8192 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 256000, hidden_size: 3072 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: { hidden_size: 3072, theta: 10000.0, max_length: 8192 } },
      { id: 'n4', type: 'layer_stack', name: '28× Decoder Block', x: 650, y: 140, params: { num_layers: 28 } },
      { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: { hidden_size: 3072, eps: 1e-6 } },
      { id: 'n6', type: 'mha_attention', name: 'Self-Attention (head_dim 256)', x: 850, y: 80, params: { hidden_size: 3072, num_heads: 16, num_kv_heads: 16, head_dim: 256, causal: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'rmsnorm', name: 'Pre-FFN RMSNorm', x: 650, y: 200, params: { hidden_size: 3072, eps: 1e-6 } },
      { id: 'n9', type: 'ffn_gated', name: 'GeGLU FFN', x: 850, y: 200, params: { hidden_size: 3072, intermediate_size: 24576, activation: 'gelu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'rmsnorm', name: 'Final RMSNorm', x: 1250, y: 140, params: { hidden_size: 3072, eps: 1e-6 } },
      { id: 'n12', type: 'lm_head', name: 'LM Head (tied)', x: 1450, y: 140, params: { vocab_size: 256000, hidden_size: 3072, tie_weights: true } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-qwen2-7b',
    name: 'Qwen2 7B',
    family: 'transformer',
    description: 'Alibaba Qwen2 7B — 7.6B params, 28 layers, 3584 hidden, 28 heads, 4 KV heads (GQA), QKV projections carry bias (unlike every LLaMA-family entry in this catalogue), RoPE theta 1e6, SwiGLU, 152K vocab',
    tags: ['decoder-only', 'autoregressive', 'llm', 'gqa', 'qkv-bias'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 32768 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 152064, hidden_size: 3584 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: { hidden_size: 3584, theta: 1000000.0, max_length: 32768 } },
      { id: 'n4', type: 'layer_stack', name: '28× Decoder Block', x: 650, y: 140, params: { num_layers: 28 } },
      { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: { hidden_size: 3584, eps: 1e-6 } },
      { id: 'n6', type: 'gqa_attention', name: 'GQA (28 heads, 4 KV, QKV bias)', x: 850, y: 80, params: { hidden_size: 3584, num_heads: 28, num_kv_heads: 4, head_dim: 128, causal: true, bias: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'rmsnorm', name: 'Pre-FFN RMSNorm', x: 650, y: 200, params: { hidden_size: 3584, eps: 1e-6 } },
      { id: 'n9', type: 'ffn_gated', name: 'SwiGLU FFN', x: 850, y: 200, params: { hidden_size: 3584, intermediate_size: 18944, activation: 'silu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'rmsnorm', name: 'Final RMSNorm', x: 1250, y: 140, params: { hidden_size: 3584, eps: 1e-6 } },
      { id: 'n12', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 152064, hidden_size: 3584 } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-gptj-6b',
    name: 'GPT-J 6B',
    family: 'transformer',
    description: 'EleutherAI GPT-J-6B — 6B params, 28 layers, 4096 hidden, 16 heads, parallel attention+FFN block — both branches read the same pre-norm input and sum into one residual add, not the sequential norm→attn→add→norm→ffn→add every other decoder in this catalogue uses. Partial RoPE (rotary_dim 64 of 256), standard GELU FFN.',
    tags: ['decoder-only', 'autoregressive', 'llm', 'parallel-block'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50400, hidden_size: 4096 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE (partial, rotary_dim 64)', x: 450, y: 140, params: { hidden_size: 4096, theta: 10000.0, max_length: 2048 } },
      { id: 'n4', type: 'layer_stack', name: '28× Parallel Block', x: 650, y: 140, params: { num_layers: 28 } },
      { id: 'n5', type: 'layernorm', name: 'Shared Pre-Block LayerNorm', x: 650, y: 140, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n6', type: 'mha_attention', name: 'Self-Attention (parallel branch)', x: 850, y: 80, params: { hidden_size: 4096, num_heads: 16, num_kv_heads: 16, head_dim: 256, causal: true } },
      { id: 'n7', type: 'ffn_standard', name: 'FFN (parallel branch, GELU)', x: 850, y: 200, params: { hidden_size: 4096, intermediate_size: 16384, activation: 'gelu' } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add (attn + ffn)', x: 1050, y: 140, params: {} },
      { id: 'n9', type: 'layernorm', name: 'Final LayerNorm', x: 1250, y: 140, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n10', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 50400, hidden_size: 4096 } },
      { id: 'n11', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n9' },
      { id: 'c5', from: 'n9', to: 'n10' }, { id: 'c6', from: 'n10', to: 'n11' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n5', to: 'n7' },
      { id: 'c9', from: 'n6', to: 'n8' }, { id: 'c10', from: 'n7', to: 'n8' },
      { id: 'c11', from: 'n8', to: 'n4' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-olmo-7b',
    name: 'OLMo 7B',
    family: 'transformer',
    description: 'AI2 OLMo-7B — 6.9B params, 32 layers, 4096 hidden, 32 heads, non-parametric LayerNorm (no learnable scale or bias at all — approximated here with a standard LayerNorm carrying a tiny weight/bias, negligible next to 6.9B params, not a real formula for the zero-param case), RoPE, SwiGLU. Fully open weights, training data and code, unlike every other entry in this catalogue.',
    tags: ['decoder-only', 'autoregressive', 'llm', 'open-source', 'non-parametric-norm'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50304, hidden_size: 4096 } },
      { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: { hidden_size: 4096, theta: 10000.0, max_length: 2048 } },
      { id: 'n4', type: 'layer_stack', name: '32× Decoder Block', x: 650, y: 140, params: { num_layers: 32 } },
      { id: 'n5', type: 'layernorm', name: 'Pre-Attention Norm (non-parametric)', x: 650, y: 80, params: { hidden_size: 4096, eps: 1e-5, elementwise_affine: false } },
      { id: 'n6', type: 'mha_attention', name: 'Self-Attention', x: 850, y: 80, params: { hidden_size: 4096, num_heads: 32, num_kv_heads: 32, head_dim: 128, causal: true } },
      { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {} },
      { id: 'n8', type: 'layernorm', name: 'Pre-FFN Norm (non-parametric)', x: 650, y: 200, params: { hidden_size: 4096, eps: 1e-5, elementwise_affine: false } },
      { id: 'n9', type: 'ffn_gated', name: 'SwiGLU FFN', x: 850, y: 200, params: { hidden_size: 4096, intermediate_size: 11008, activation: 'silu' } },
      { id: 'n10', type: 'residual_add', name: 'Residual Add', x: 1050, y: 200, params: {} },
      { id: 'n11', type: 'layernorm', name: 'Final Norm (non-parametric)', x: 1250, y: 140, params: { hidden_size: 4096, eps: 1e-5, elementwise_affine: false } },
      { id: 'n12', type: 'lm_head', name: 'LM Head', x: 1450, y: 140, params: { vocab_size: 50304, hidden_size: 4096 } },
      { id: 'n13', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n11' },
      { id: 'c5', from: 'n11', to: 'n12' }, { id: 'c6', from: 'n12', to: 'n13' },
      { id: 'c7', from: 'n5', to: 'n6' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n8', to: 'n9' }, { id: 'c10', from: 'n9', to: 'n10' },
      { id: 'c11', from: 'n7', to: 'n4' }, { id: 'c12', from: 'n10', to: 'n4' },
    ],
    defaultParams: {},
  },

  // ====== CNN — batch 2 (2026-09 research pass) ======
{
    id: 'tpl-alexnet',
    name: 'AlexNet',
    family: 'cnn',
    description: 'AlexNet — ~61M params, 5 conv layers + 3 FC, ReLU, dropout, the 2012 ILSVRC winner (Krizhevsky, Sutskever & Hinton)',
    tags: ['vision', 'classification', 'classic', 'historic'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: 'Conv1 (96, 11×11, stride 4)', x: 250, y: 140, params: { in_channels: 3, out_channels: 96, kernel_size: 11, stride: 4, padding: 2 } },
      { id: 'n3', type: 'relu', name: 'ReLU', x: 450, y: 140, params: {} },
      { id: 'n4', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 650, y: 140, params: { kernel_size: 3, stride: 2 } },
      { id: 'n5', type: 'conv2d', name: 'Conv2 (256, 5×5)', x: 850, y: 140, params: { in_channels: 96, out_channels: 256, kernel_size: 5, padding: 2 } },
      { id: 'n6', type: 'relu', name: 'ReLU', x: 1050, y: 140, params: {} },
      { id: 'n7', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 1250, y: 140, params: { kernel_size: 3, stride: 2 } },
      { id: 'n8', type: 'conv2d', name: 'Conv3 (384, 3×3)', x: 1450, y: 140, params: { in_channels: 256, out_channels: 384, kernel_size: 3, padding: 1 } },
      { id: 'n9', type: 'relu', name: 'ReLU', x: 1650, y: 140, params: {} },
      { id: 'n10', type: 'conv2d', name: 'Conv4 (384, 3×3)', x: 1850, y: 140, params: { in_channels: 384, out_channels: 384, kernel_size: 3, padding: 1 } },
      { id: 'n11', type: 'relu', name: 'ReLU', x: 2050, y: 140, params: {} },
      { id: 'n12', type: 'conv2d', name: 'Conv5 (256, 3×3)', x: 2250, y: 140, params: { in_channels: 384, out_channels: 256, kernel_size: 3, padding: 1 } },
      { id: 'n13', type: 'relu', name: 'ReLU', x: 2450, y: 140, params: {} },
      { id: 'n14', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 2650, y: 140, params: { kernel_size: 3, stride: 2 } },
      { id: 'n15', type: 'flatten', name: 'Flatten', x: 2850, y: 140, params: {} },
      { id: 'n16', type: 'dense', name: 'FC 4096', x: 3050, y: 140, params: { in_features: 9216, out_features: 4096 } },
      { id: 'n17', type: 'relu', name: 'ReLU', x: 3250, y: 140, params: {} },
      { id: 'n18', type: 'dropout', name: 'Dropout', x: 3450, y: 140, params: { rate: 0.5 } },
      { id: 'n19', type: 'dense', name: 'FC 4096', x: 3650, y: 140, params: { in_features: 4096, out_features: 4096 } },
      { id: 'n20', type: 'relu', name: 'ReLU', x: 3850, y: 140, params: {} },
      { id: 'n21', type: 'dropout', name: 'Dropout', x: 4050, y: 140, params: { rate: 0.5 } },
      { id: 'n22', type: 'classification_head', name: 'FC-1000', x: 4250, y: 140, params: { num_labels: 1000 } },
      { id: 'n23', type: 'output', name: 'Output', x: 4450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' }, { id: 'c12', from: 'n12', to: 'n13' },
      { id: 'c13', from: 'n13', to: 'n14' }, { id: 'c14', from: 'n14', to: 'n15' },
      { id: 'c15', from: 'n15', to: 'n16' }, { id: 'c16', from: 'n16', to: 'n17' },
      { id: 'c17', from: 'n17', to: 'n18' }, { id: 'c18', from: 'n18', to: 'n19' },
      { id: 'c19', from: 'n19', to: 'n20' }, { id: 'c20', from: 'n20', to: 'n21' },
      { id: 'c21', from: 'n21', to: 'n22' }, { id: 'c22', from: 'n22', to: 'n23' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-vgg19',
    name: 'VGG-19',
    family: 'cnn',
    description: 'VGG-19 — 143.7M params, 16 conv layers + 3 FC (blocks 3-5 deepened to 4 convs each vs VGG-16\'s 3), 3×3 conv blocks, 2×2 max pool',
    tags: ['vision', 'classification', 'classic', 'conv'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: 'Conv Block 1: 2× [64, 3×3]', x: 250, y: 140, params: { in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1, blocks: 2 } },
      { id: 'n3', type: 'relu', name: 'ReLU', x: 450, y: 140, params: {} },
      { id: 'n4', type: 'max_pool', name: '2×2 Max Pool', x: 650, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n5', type: 'conv2d', name: 'Conv Block 2: 2× [128, 3×3]', x: 850, y: 140, params: { in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1, blocks: 2 } },
      { id: 'n6', type: 'relu', name: 'ReLU', x: 1050, y: 140, params: {} },
      { id: 'n7', type: 'max_pool', name: '2×2 Max Pool', x: 1250, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n8', type: 'conv2d', name: 'Conv Block 3: 4× [256, 3×3]', x: 1450, y: 140, params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1, blocks: 4 } },
      { id: 'n9', type: 'relu', name: 'ReLU', x: 1650, y: 140, params: {} },
      { id: 'n10', type: 'max_pool', name: '2×2 Max Pool', x: 1850, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n11', type: 'conv2d', name: 'Conv Block 4: 4× [512, 3×3]', x: 2050, y: 140, params: { in_channels: 256, out_channels: 512, kernel_size: 3, padding: 1, blocks: 4 } },
      { id: 'n12', type: 'relu', name: 'ReLU', x: 2250, y: 140, params: {} },
      { id: 'n13', type: 'max_pool', name: '2×2 Max Pool', x: 2450, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n14', type: 'conv2d', name: 'Conv Block 5: 4× [512, 3×3]', x: 2650, y: 140, params: { in_channels: 512, out_channels: 512, kernel_size: 3, padding: 1, blocks: 4 } },
      { id: 'n15', type: 'relu', name: 'ReLU', x: 2850, y: 140, params: {} },
      { id: 'n16', type: 'max_pool', name: '2×2 Max Pool', x: 3050, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n17', type: 'flatten', name: 'Flatten', x: 3250, y: 140, params: {} },
      { id: 'n18', type: 'dense', name: 'FC 4096', x: 3450, y: 140, params: { in_features: 25088, out_features: 4096 } },
      { id: 'n19', type: 'relu', name: 'ReLU', x: 3650, y: 140, params: {} },
      { id: 'n20', type: 'dropout', name: 'Dropout', x: 3850, y: 140, params: { rate: 0.5 } },
      { id: 'n21', type: 'dense', name: 'FC 4096', x: 4050, y: 140, params: { in_features: 4096, out_features: 4096 } },
      { id: 'n22', type: 'relu', name: 'ReLU', x: 4250, y: 140, params: {} },
      { id: 'n23', type: 'dropout', name: 'Dropout', x: 4450, y: 140, params: { rate: 0.5 } },
      { id: 'n24', type: 'classification_head', name: 'FC-1000', x: 4650, y: 140, params: { num_labels: 1000 } },
      { id: 'n25', type: 'output', name: 'Output', x: 4850, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' }, { id: 'c12', from: 'n12', to: 'n13' },
      { id: 'c13', from: 'n13', to: 'n14' }, { id: 'c14', from: 'n14', to: 'n15' },
      { id: 'c15', from: 'n15', to: 'n16' }, { id: 'c16', from: 'n16', to: 'n17' },
      { id: 'c17', from: 'n17', to: 'n18' }, { id: 'c18', from: 'n18', to: 'n19' },
      { id: 'c19', from: 'n19', to: 'n20' }, { id: 'c20', from: 'n20', to: 'n21' },
      { id: 'c21', from: 'n21', to: 'n22' }, { id: 'c22', from: 'n22', to: 'n23' },
      { id: 'c23', from: 'n23', to: 'n24' }, { id: 'c24', from: 'n24', to: 'n25' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-wideresnet50-2',
    name: 'Wide ResNet-50-2',
    family: 'cnn',
    description: 'Wide ResNet-50-2 — 68.9M params, same 4-stage bottleneck depth as ResNet-50 (3+4+6+3) but with the bottleneck\'s inner width doubled (base_width 128 vs 64)',
    tags: ['vision', 'classification', 'residual', 'wide'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: '7×7 Conv (stride 2)', x: 250, y: 140, params: { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3 } },
      { id: 'n3', type: 'batchnorm', name: 'BatchNorm', x: 450, y: 140, params: { num_features: 64, eps: 1e-5 } },
      { id: 'n4', type: 'relu', name: 'ReLU', x: 650, y: 140, params: {} },
      { id: 'n5', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 850, y: 140, params: { kernel_size: 3, stride: 2, padding: 1 } },
      { id: 'n6', type: 'bottleneck_block', name: 'Stage 1: 3× Bottleneck (256, base_width 128)', x: 1050, y: 140, params: { planes: 256, blocks: 3, stride: 1, expansion: 4, base_width: 128 } },
      { id: 'n7', type: 'bottleneck_block', name: 'Stage 2: 4× Bottleneck (512, base_width 128)', x: 1250, y: 140, params: { planes: 512, blocks: 4, stride: 2, expansion: 4, base_width: 128 } },
      { id: 'n8', type: 'bottleneck_block', name: 'Stage 3: 6× Bottleneck (1024, base_width 128)', x: 1450, y: 140, params: { planes: 1024, blocks: 6, stride: 2, expansion: 4, base_width: 128 } },
      { id: 'n9', type: 'bottleneck_block', name: 'Stage 4: 3× Bottleneck (2048, base_width 128)', x: 1650, y: 140, params: { planes: 2048, blocks: 3, stride: 2, expansion: 4, base_width: 128 } },
      { id: 'n10', type: 'global_pool', name: 'Global Avg Pool', x: 1850, y: 140, params: {} },
      { id: 'n11', type: 'classification_head', name: 'FC-1000', x: 2050, y: 140, params: { num_labels: 1000 } },
      { id: 'n12', type: 'output', name: 'Output', x: 2250, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-resnext50-32x4d',
    name: 'ResNeXt-50 (32×4d)',
    family: 'cnn',
    description: 'ResNeXt-50 (32×4d) — 25.0M params, same 4-stage bottleneck depth as ResNet-50 (3+4+6+3) with grouped convolutions (cardinality 32, base width 4 per group)',
    tags: ['vision', 'classification', 'residual', 'grouped-conv'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: '7×7 Conv (stride 2)', x: 250, y: 140, params: { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3 } },
      { id: 'n3', type: 'batchnorm', name: 'BatchNorm', x: 450, y: 140, params: { num_features: 64, eps: 1e-5 } },
      { id: 'n4', type: 'relu', name: 'ReLU', x: 650, y: 140, params: {} },
      { id: 'n5', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 850, y: 140, params: { kernel_size: 3, stride: 2, padding: 1 } },
      { id: 'n6', type: 'bottleneck_block', name: 'Stage 1: 3× Bottleneck (256, 32×4d)', x: 1050, y: 140, params: { planes: 256, blocks: 3, stride: 1, expansion: 4, cardinality: 32, base_width: 4 } },
      { id: 'n7', type: 'bottleneck_block', name: 'Stage 2: 4× Bottleneck (512, 32×4d)', x: 1250, y: 140, params: { planes: 512, blocks: 4, stride: 2, expansion: 4, cardinality: 32, base_width: 4 } },
      { id: 'n8', type: 'bottleneck_block', name: 'Stage 3: 6× Bottleneck (1024, 32×4d)', x: 1450, y: 140, params: { planes: 1024, blocks: 6, stride: 2, expansion: 4, cardinality: 32, base_width: 4 } },
      { id: 'n9', type: 'bottleneck_block', name: 'Stage 4: 3× Bottleneck (2048, 32×4d)', x: 1650, y: 140, params: { planes: 2048, blocks: 3, stride: 2, expansion: 4, cardinality: 32, base_width: 4 } },
      { id: 'n10', type: 'global_pool', name: 'Global Avg Pool', x: 1850, y: 140, params: {} },
      { id: 'n11', type: 'classification_head', name: 'FC-1000', x: 2050, y: 140, params: { num_labels: 1000 } },
      { id: 'n12', type: 'output', name: 'Output', x: 2250, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-se-resnet50',
    name: 'SE-ResNet-50',
    family: 'cnn',
    description: 'SE-ResNet-50 — 28.1M params, ResNet-50\'s 4-stage bottleneck (3+4+6+3) with a Squeeze-and-Excitation channel-attention gate after each block (reduction ratio 16), Hu et al. 2018, ILSVRC 2017 winner',
    tags: ['vision', 'classification', 'residual', 'attention', 'squeeze-excite'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'conv2d', name: '7×7 Conv (stride 2)', x: 250, y: 140, params: { in_channels: 3, out_channels: 64, kernel_size: 7, stride: 2, padding: 3 } },
      { id: 'n3', type: 'batchnorm', name: 'BatchNorm', x: 450, y: 140, params: { num_features: 64, eps: 1e-5 } },
      { id: 'n4', type: 'relu', name: 'ReLU', x: 650, y: 140, params: {} },
      { id: 'n5', type: 'max_pool', name: '3×3 Max Pool (stride 2)', x: 850, y: 140, params: { kernel_size: 3, stride: 2, padding: 1 } },
      { id: 'n6', type: 'bottleneck_block', name: 'Stage 1: 3× SE-Bottleneck (256)', x: 1050, y: 140, params: { planes: 256, blocks: 3, stride: 1, expansion: 4, se_ratio: 0.0625 } },
      { id: 'n7', type: 'bottleneck_block', name: 'Stage 2: 4× SE-Bottleneck (512)', x: 1250, y: 140, params: { planes: 512, blocks: 4, stride: 2, expansion: 4, se_ratio: 0.0625 } },
      { id: 'n8', type: 'bottleneck_block', name: 'Stage 3: 6× SE-Bottleneck (1024)', x: 1450, y: 140, params: { planes: 1024, blocks: 6, stride: 2, expansion: 4, se_ratio: 0.0625 } },
      { id: 'n9', type: 'bottleneck_block', name: 'Stage 4: 3× SE-Bottleneck (2048)', x: 1650, y: 140, params: { planes: 2048, blocks: 3, stride: 2, expansion: 4, se_ratio: 0.0625 } },
      { id: 'n10', type: 'global_pool', name: 'Global Avg Pool', x: 1850, y: 140, params: {} },
      { id: 'n11', type: 'classification_head', name: 'FC-1000', x: 2050, y: 140, params: { num_labels: 1000 } },
      { id: 'n12', type: 'output', name: 'Output', x: 2250, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
      { id: 'c9', from: 'n9', to: 'n10' }, { id: 'c10', from: 'n10', to: 'n11' },
      { id: 'c11', from: 'n11', to: 'n12' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-regnety-4gf',
    name: 'RegNetY-4.0GF',
    family: 'cnn',
    description: 'RegNetY-4.0GF — ~20.6M params, 4 stages of grouped-convolution + Squeeze-Excitation blocks (depths 1-3-6-6, widths 48-104-208-440, group width 40), Radosavovic et al. 2020',
    tags: ['vision', 'classification', 'grouped-conv', 'squeeze-excite', 'design-space'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Image', x: 50, y: 140, params: { channels: 3, height: 224, width: 224 } },
      { id: 'n2', type: 'stem_block', name: 'Stem: 3×3 Conv (32, stride 2)', x: 250, y: 140, params: { out_channels: 32, kernel_size: 3, stride: 2 } },
      { id: 'n3', type: 'bottleneck_block', name: 'Stage 1: 1× Block (48)', x: 450, y: 140, params: { planes: 48, blocks: 1, stride: 2, expansion: 1, cardinality: 1, base_width: 40, se_ratio: 0.25 } },
      { id: 'n4', type: 'bottleneck_block', name: 'Stage 2: 3× Block (104)', x: 650, y: 140, params: { planes: 104, blocks: 3, stride: 2, expansion: 1, cardinality: 2, base_width: 40, se_ratio: 0.25 } },
      { id: 'n5', type: 'bottleneck_block', name: 'Stage 3: 6× Block (208)', x: 850, y: 140, params: { planes: 208, blocks: 6, stride: 2, expansion: 1, cardinality: 5, base_width: 40, se_ratio: 0.25 } },
      { id: 'n6', type: 'bottleneck_block', name: 'Stage 4: 6× Block (440)', x: 1050, y: 140, params: { planes: 440, blocks: 6, stride: 2, expansion: 1, cardinality: 11, base_width: 40, se_ratio: 0.25 } },
      { id: 'n7', type: 'global_pool', name: 'Global Avg Pool', x: 1250, y: 140, params: {} },
      { id: 'n8', type: 'classification_head', name: 'FC-1000', x: 1450, y: 140, params: { num_labels: 1000 } },
      { id: 'n9', type: 'output', name: 'Output', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
    ],
    defaultParams: {},
  },

  // ====== SSM — batch 2 (2026-09 research pass) ======
{
    id: 'tpl-s4-wikitext103',
    name: 'S4 (WikiText-103 LM)',
    family: 'ssm',
    // Gu, Goel & Ré 2021, "Efficiently Modeling Long Sequences with
    // Structured State Spaces" — the WikiText-103 language-modeling
    // configuration: 16 S4 blocks, feature dimension 1024, closing to
    // within 0.8 perplexity of a Transformer, the first SSM to do so.
    // State dimension N=64 is S4's own standard default throughout the
    // paper. Vocabulary is WikiText-103's own word-level vocab (~267,735
    // tokens), not a subword tokenizer.
    description: 'S4 — WikiText-103 LM config, 16 layers, 1024 hidden, structured state space, state dim 64 (Gu, Goel & Ré 2021)',
    tags: ['ssm', 's4', 'structured-state-space', 'linear-time'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 267735, hidden_size: 1024 } },
      { id: 'n3', type: 'layer_stack', name: '16× S4 Block', x: 450, y: 140, params: { num_layers: 16 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block Norm', x: 450, y: 80, params: { hidden_size: 1024, eps: 1e-5 } },
      { id: 'n5', type: 's4_block', name: 'S4 (structured state space, state 64)', x: 650, y: 80, params: { hidden_size: 1024, state_dim: 64 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final Norm', x: 1050, y: 140, params: { hidden_size: 1024, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 267735, hidden_size: 1024, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-h3-125m',
    name: 'H3-125M',
    family: 'ssm',
    // Fu, Dao, Saab, Thomas, Rudra & Ré 2023 (ICLR), "Hungry Hungry Hippos:
    // Towards Language Modeling with State Space Models" — the 125M
    // configuration, deliberately built to the same width/depth as GPT-2
    // small (768 hidden, 12 layers) for a direct, apples-to-apples
    // comparison; the paper reports it matching or beating GPT-2 small on
    // OpenWebText/Pile/WikiText103 perplexity. State dimension 64 follows
    // S4's convention, which H3's SSM layer is built on.
    description: 'H3 — 125M-equivalent config (matches GPT-2 small: 768 hidden, 12 layers), state dim 64 (Fu, Dao et al. 2023)',
    tags: ['ssm', 'h3', 'hungry-hungry-hippos', 'structured-state-space'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 1024 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50257, hidden_size: 768 } },
      { id: 'n3', type: 'layer_stack', name: '12× H3 Block', x: 450, y: 140, params: { num_layers: 12 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block Norm', x: 450, y: 80, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n5', type: 'h3_block', name: 'H3 (SSM sandwiched by gates, state 64)', x: 650, y: 80, params: { hidden_size: 768, state_dim: 64 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final Norm', x: 1050, y: 140, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50257, hidden_size: 768, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-rwkv4-7b',
    name: 'RWKV-4 7B',
    family: 'ssm',
    // BlinkDL's RWKV-4 — the 7B release size (RWKV-4-Pile-7B / the base
    // for RWKV-4-Raven-7B), 32 layers, 4096 embedding width. FFN width
    // uses RWKV's own real convention of ~3.5× the embedding dimension
    // (4096 × 3.5 = 14336), not a guessed 4×. Vocabulary is RWKV's own
    // 50277-token tokenizer (GPT-NeoX-derived), not GPT-2's 50257.
    description: 'RWKV-4 — 7B config, 32 layers, 4096 embedding, linear-attention RNN, FFN width 14336 (BlinkDL, RWKV-4-Pile-7B)',
    tags: ['ssm', 'rwkv', 'linear-attention', 'rnn-transformer-hybrid', 'large'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50277, hidden_size: 4096 } },
      { id: 'n3', type: 'layer_stack', name: '32× RWKV Block', x: 450, y: 140, params: { num_layers: 32 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block Norm', x: 450, y: 80, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n5', type: 'rwkv_block', name: 'RWKV (time-mix + channel-mix, ffn 14336)', x: 650, y: 80, params: { hidden_size: 4096, intermediate_size: 14336 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final Norm', x: 1050, y: 140, params: { hidden_size: 4096, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50277, hidden_size: 4096, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-retnet-2.7b',
    name: 'RetNet 2.7B',
    family: 'ssm',
    // Sun, Dong, Huang et al. 2023 (Microsoft), "Retentive Network: A
    // Successor to Transformer for Large Language Models" — the 2.7B
    // configuration from the paper's own scaling table: 32 layers, 2560
    // hidden, FFN size 5120, 10 retention heads (head dim 256 for Q/K,
    // 512 for V — the opspec formula's own `retention_params` doesn't
    // take a head count, matching how `rwkv_params` also ignores it).
    description: 'RetNet — 2.7B config, 32 layers, 2560 hidden, 10 retention heads, FFN 5120 (Sun, Dong, Huang et al. 2023, Microsoft)',
    tags: ['ssm', 'retnet', 'retention', 'parallel-recurrent-chunkwise'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50257, hidden_size: 2560 } },
      { id: 'n3', type: 'layer_stack', name: '32× Retention Block', x: 450, y: 140, params: { num_layers: 32 } },
      { id: 'n4', type: 'rmsnorm', name: 'Pre-Block Norm', x: 450, y: 80, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n5', type: 'retention_block', name: 'Retention (10 heads, ffn 5120)', x: 650, y: 80, params: { hidden_size: 2560, intermediate_size: 5120 } },
      { id: 'n6', type: 'residual_add', name: 'Residual Add', x: 850, y: 80, params: {} },
      { id: 'n7', type: 'rmsnorm', name: 'Final Norm', x: 1050, y: 140, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n8', type: 'lm_head', name: 'LM Head (tied)', x: 1250, y: 140, params: { vocab_size: 50257, hidden_size: 2560, tie_weights: true } },
      { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n7' }, { id: 'c4', from: 'n7', to: 'n8' },
      { id: 'c5', from: 'n8', to: 'n9' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n3' },
    ],
    defaultParams: {},
  },

  // ====== DIFFUSION — batch 2 (2026-09 research pass) ======
{
    id: 'tpl-stable-diffusion-2-1',
    name: 'Stable Diffusion 2.1',
    family: 'diffusion',
    description: 'Stable Diffusion 2.1 — 865M UNet, OpenCLIP ViT-H/14 text encoder (354M, 1024-dim), 8× downsample VAE, 768×768, v-prediction',
    tags: ['diffusion', 'stable-diffusion', 'latent', 'text-to-image'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 200, params: { max_length: 77 } },
      { id: 'n2', type: 'text_encoder', name: 'OpenCLIP ViT-H/14', x: 250, y: 200, params: { vocab_size: 49408, hidden_size: 1024, num_layers: 24 } },
      { id: 'n3', type: 'vae_encoder', name: 'VAE Encoder (8× down)', x: 50, y: 140, params: { in_channels: 3, latent_channels: 4, compression: 8 } },
      { id: 'n4', type: 'unet_latent', name: 'UNet (865M, cross-attn)', x: 250, y: 140, params: { in_channels: 4, model_channels: 320, num_res_blocks: 2, channel_mult: [1, 2, 4, 4], num_heads: 64, cross_attn_dim: 1024 } },
      { id: 'n5', type: 'noise_scheduler', name: 'Noise Scheduler (1000 steps, v-pred)', x: 450, y: 80, params: { num_train_timesteps: 1000, beta_start: 0.00085, beta_end: 0.012, beta_schedule: 'scaled_linear', prediction_type: 'v_prediction' } },
      { id: 'n6', type: 'vae_decoder', name: 'VAE Decoder (8× up)', x: 450, y: 140, params: { latent_channels: 4, out_channels: 3, compression: 8 } },
      { id: 'n7', type: 'output', name: 'Generated Image 768×768', x: 650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n4' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n6' },
      { id: 'c5', from: 'n5', to: 'n4' }, { id: 'c6', from: 'n6', to: 'n7' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-stable-diffusion-3-medium',
    name: 'Stable Diffusion 3 Medium',
    family: 'diffusion',
    description: 'Stable Diffusion 3 Medium — 2B params, MMDiT, 24 blocks (hidden 1536, the paper\'s own depth×64 scaling), triple text encoder (CLIP ViT-L + OpenCLIP ViT-G + T5-XXL 4.7B), 16-channel VAE, rectified flow',
    tags: ['diffusion', 'stable-diffusion', 'mmdit', 'rectified-flow', 'text-to-image'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 220, params: { max_length: 77 } },
      { id: 'n2', type: 'text_encoder', name: 'CLIP ViT-L', x: 250, y: 260, params: { hidden_size: 768 } },
      { id: 'n3', type: 'text_encoder', name: 'OpenCLIP ViT-G', x: 250, y: 200, params: { hidden_size: 1280 } },
      { id: 'n4', type: 'text_encoder', name: 'T5-XXL (4.7B, frozen)', x: 250, y: 140, params: { vocab_size: 32128, hidden_size: 4096, num_layers: 24 } },
      { id: 'n5', type: 'concat', name: 'Concat Text Embeddings', x: 450, y: 200, params: {} },
      { id: 'n6', type: 'input', name: 'Latent Noise', x: 50, y: 80, params: { latent_shape: [1, 128, 128, 16] } },
      { id: 'n7', type: 'mmdit_block', name: 'MMDiT Block ×12 (1B)', x: 650, y: 140, params: { hidden_size: 1536, num_heads: 24, num_layers: 12, double_stream: true } },
      { id: 'n8', type: 'mmdit_block', name: 'MMDiT Block ×12 (1B)', x: 850, y: 140, params: { hidden_size: 1536, num_heads: 24, num_layers: 12, double_stream: true } },
      { id: 'n9', type: 'vae_decoder', name: 'VAE Decoder (16-channel)', x: 1050, y: 140, params: { latent_channels: 16, out_channels: 3 } },
      { id: 'n10', type: 'output', name: 'Generated Image 1024×1024', x: 1250, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n1', to: 'n3' }, { id: 'c3', from: 'n1', to: 'n4' },
      { id: 'c4', from: 'n2', to: 'n5' }, { id: 'c5', from: 'n3', to: 'n5' }, { id: 'c6', from: 'n4', to: 'n5' },
      { id: 'c7', from: 'n5', to: 'n7' }, { id: 'c8', from: 'n6', to: 'n7' },
      { id: 'c9', from: 'n7', to: 'n8' }, { id: 'c10', from: 'n8', to: 'n9' }, { id: 'c11', from: 'n9', to: 'n10' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-pixart-alpha',
    name: 'PixArt-α',
    family: 'diffusion',
    description: 'PixArt-α — 0.6B diffusion transformer (DiT-XL structure: 28 blocks, hidden 1152), frozen T5 text encoder with cross-attention conditioning, trained for well under $30K',
    tags: ['diffusion', 'dit', 'transformer', 'text-to-image', 'efficient-training'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 140, params: { max_length: 120 } },
      { id: 'n2', type: 'text_encoder', name: 'T5 Text Encoder (frozen)', x: 250, y: 140, params: { vocab_size: 32128, hidden_size: 4096, num_layers: 24 } },
      { id: 'n3', type: 'input', name: 'Latent Noise', x: 50, y: 60, params: { latent_shape: [1, 32, 32, 4] } },
      { id: 'n4', type: 'mmdit_block', name: 'DiT Block ×14 (0.3B)', x: 450, y: 100, params: { hidden_size: 1152, num_heads: 16, num_layers: 14, double_stream: false } },
      { id: 'n5', type: 'mmdit_block', name: 'DiT Block ×14 (0.3B)', x: 650, y: 100, params: { hidden_size: 1152, num_heads: 16, num_layers: 14, double_stream: false } },
      { id: 'n6', type: 'vae_decoder', name: 'VAE Decoder', x: 850, y: 100, params: { latent_channels: 4, out_channels: 3 } },
      { id: 'n7', type: 'output', name: 'Generated Image', x: 1050, y: 100, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n4' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-dit-xl-2',
    name: 'DiT-XL/2 (Diffusion Transformer)',
    family: 'diffusion',
    description: 'DiT-XL/2 — 675M params, class-conditional (ImageNet, not text-to-image), patch size 2, 28 transformer blocks, hidden 1152, 16 heads (Peebles & Xie, 2022 — the paper Flux/PixArt/SD3\'s DiT-style backbones descend from)',
    tags: ['diffusion', 'dit', 'transformer', 'class-conditional', 'imagenet'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Class Label + Latent Noise', x: 50, y: 140, params: { num_classes: 1000, latent_shape: [1, 32, 32, 4] } },
      { id: 'n2', type: 'linear', name: 'Class Embedding', x: 250, y: 140, params: { hidden_size: 1152 } },
      { id: 'n3', type: 'mmdit_block', name: 'DiT Block ×14 (patch 2)', x: 450, y: 140, params: { hidden_size: 1152, num_heads: 16, num_layers: 14, double_stream: false } },
      { id: 'n4', type: 'mmdit_block', name: 'DiT Block ×14 (patch 2)', x: 650, y: 140, params: { hidden_size: 1152, num_heads: 16, num_layers: 14, double_stream: false } },
      { id: 'n5', type: 'vae_decoder', name: 'VAE Decoder', x: 850, y: 140, params: { latent_channels: 4, out_channels: 3 } },
      { id: 'n6', type: 'output', name: 'Generated Image 256×256', x: 1050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-stable-cascade',
    name: 'Stable Cascade',
    family: 'diffusion',
    description: 'Stable Cascade (Würstchen v3) — three cascaded stages instead of one UNet: Stage C prior (3.6B), Stage B decoder (1.5B), Stage A latent compressor (20M), 42× compression vs. 8× for standard Stable Diffusion',
    tags: ['diffusion', 'cascaded', 'stable-cascade', 'wurstchen', 'multi-stage'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Text Prompt', x: 50, y: 140, params: { max_length: 77 } },
      { id: 'n2', type: 'text_encoder', name: 'CLIP Text Encoder', x: 250, y: 140, params: { hidden_size: 1280 } },
      { id: 'n3', type: 'unet_latent', name: 'Stage C Prior (3.6B, 24×24 latent)', x: 450, y: 140, params: { in_channels: 16, model_channels: 2048, num_res_blocks: 2, channel_mult: [1, 1], cross_attn_dim: 1280 } },
      { id: 'n4', type: 'unet_latent', name: 'Stage B Decoder (1.5B)', x: 650, y: 140, params: { in_channels: 4, model_channels: 320, num_res_blocks: 2, channel_mult: [1, 2, 4, 4], cross_attn_dim: 1280 } },
      { id: 'n5', type: 'vae_decoder', name: 'Stage A (VQGAN, 20M, 42× compression)', x: 850, y: 140, params: { latent_channels: 4, out_channels: 3, compression: 42 } },
      { id: 'n6', type: 'output', name: 'Generated Image 1024×1024', x: 1050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n2', to: 'n4' },
      { id: 'c5', from: 'n4', to: 'n5' }, { id: 'c6', from: 'n5', to: 'n6' },
    ],
    defaultParams: {},
  },

  // ====== GNN — batch 2 (2026-09 research pass) ======
{
    id: 'tpl-chebnet',
    name: 'ChebNet (Chebyshev Spectral GCN)',
    family: 'gnn',
    description: 'ChebNet (Defferrard, Bresson & Vandergheynst 2016) — K=2 Chebyshev polynomial spectral filters, 2 layers, 16 hidden, 10-class semi-supervised node classification on Cora',
    tags: ['gnn', 'chebnet', 'spectral', 'node-classification', 'defferrard'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: { num_nodes: 2708, num_features: 1433 } },
      { id: 'n2', type: 'graph_conv', name: 'ChebConv Layer 1 (K=2, 1433→16)', x: 250, y: 140, params: { in_features: 1433, out_features: 16, k: 2, activation: 'relu', dropout: 0.5 } },
      { id: 'n3', type: 'dropout', name: 'Dropout (0.5)', x: 450, y: 140, params: { rate: 0.5 } },
      { id: 'n4', type: 'graph_conv', name: 'ChebConv Layer 2 (K=2, 16→10)', x: 650, y: 140, params: { in_features: 16, out_features: 10, k: 2, activation: 'softmax' } },
      { id: 'n5', type: 'output', name: 'Node Classes (10)', x: 850, y: 140, params: { num_classes: 10 } },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
    ],
    defaultParams: {},
  },

  {
    id: 'tpl-tagcn',
    name: 'TAGCN (Topology Adaptive GCN)',
    family: 'gnn',
    description: 'TAGCN (Du et al. 2017) — K=2 topology-adaptive polynomial graph filters (a fixed-size K-hop filter bank per layer, distinct from ChebNet\'s recursive Chebyshev basis), 2 layers, 16 hidden, Cora node classification',
    tags: ['gnn', 'tagcn', 'polynomial-filter', 'node-classification', 'du'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: { num_nodes: 2708, num_features: 1433 } },
      { id: 'n2', type: 'graph_conv', name: 'TAGConv Layer 1 (K=2, 1433→16)', x: 250, y: 140, params: { in_features: 1433, out_features: 16, k: 2, activation: 'relu', dropout: 0.5 } },
      { id: 'n3', type: 'dropout', name: 'Dropout (0.5)', x: 450, y: 140, params: { rate: 0.5 } },
      { id: 'n4', type: 'graph_conv', name: 'TAGConv Layer 2 (K=2, 16→10)', x: 650, y: 140, params: { in_features: 16, out_features: 10, k: 2, activation: 'softmax' } },
      { id: 'n5', type: 'output', name: 'Node Classes (10)', x: 850, y: 140, params: { num_classes: 10 } },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
    ],
    defaultParams: {},
  },

  {
    id: 'tpl-appnp',
    name: 'APPNP (Predict then Propagate)',
    family: 'gnn',
    description: 'APPNP (Klicpera, Bojchevski & Gunnemann 2019) — a plain 2-layer MLP predictor (no graph structure inside it) followed by K=10 steps of personalized-PageRank propagation, teleport alpha=0.1, Cora node classification',
    tags: ['gnn', 'appnp', 'personalized-pagerank', 'propagation', 'klicpera'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: { num_nodes: 2708, num_features: 1433 } },
      { id: 'n2', type: 'linear', name: 'MLP Layer 1 (1433→64)', x: 250, y: 140, params: { in_features: 1433, out_features: 64, activation: 'relu' } },
      { id: 'n3', type: 'dropout', name: 'Dropout (0.5)', x: 450, y: 140, params: { rate: 0.5 } },
      { id: 'n4', type: 'linear', name: 'MLP Layer 2 (64→10)', x: 650, y: 140, params: { in_features: 64, out_features: 10 } },
      { id: 'n5', type: 'message_passing', name: 'Personalized PageRank Propagation (K=10, α=0.1)', x: 850, y: 140, params: { aggregator: 'ppr', k_hops: 10, alpha: 0.1, in_features: 10, out_features: 10 } },
      { id: 'n6', type: 'output', name: 'Node Classes (10)', x: 1050, y: 140, params: { num_classes: 10 } },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' },
    ],
    defaultParams: {},
  },

  {
    id: 'tpl-jknet',
    name: 'JK-Net (Jumping Knowledge Network, GCN base)',
    family: 'gnn',
    description: 'Jumping Knowledge Network (Xu et al. 2018), GCN aggregator variant — 6 GCN layers of 16 hidden units each, jumping-knowledge concatenation of every layer\'s output before the classifier (instead of only the last layer, the fix this architecture exists for), Cora node classification',
    tags: ['gnn', 'jknet', 'jumping-knowledge', 'multi-layer', 'xu'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Graph Input', x: 50, y: 140, params: { num_nodes: 2708, num_features: 1433 } },
      { id: 'n2', type: 'graph_conv', name: 'GCN Layer 1 (1433→16)', x: 250, y: 140, params: { in_features: 1433, out_features: 16, activation: 'relu' } },
      { id: 'n3', type: 'graph_conv', name: 'GCN Layer 2 (16→16)', x: 450, y: 140, params: { in_features: 16, out_features: 16, activation: 'relu' } },
      { id: 'n4', type: 'graph_conv', name: 'GCN Layer 3 (16→16)', x: 650, y: 140, params: { in_features: 16, out_features: 16, activation: 'relu' } },
      { id: 'n5', type: 'graph_conv', name: 'GCN Layer 4 (16→16)', x: 850, y: 140, params: { in_features: 16, out_features: 16, activation: 'relu' } },
      { id: 'n6', type: 'graph_conv', name: 'GCN Layer 5 (16→16)', x: 1050, y: 140, params: { in_features: 16, out_features: 16, activation: 'relu' } },
      { id: 'n7', type: 'graph_conv', name: 'GCN Layer 6 (16→16)', x: 1250, y: 140, params: { in_features: 16, out_features: 16, activation: 'relu' } },
      { id: 'n8', type: 'linear', name: 'Jumping-Knowledge Concat (6×16→10)', x: 1450, y: 140, params: { in_features: 96, out_features: 10, aggregation: 'concat' } },
      { id: 'n9', type: 'output', name: 'Node Classes (10)', x: 1650, y: 140, params: { num_classes: 10 } },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n2', to: 'n8' }, { id: 'c8', from: 'n3', to: 'n8' },
      { id: 'c9', from: 'n4', to: 'n8' }, { id: 'c10', from: 'n5', to: 'n8' },
      { id: 'c11', from: 'n6', to: 'n8' }, { id: 'c12', from: 'n7', to: 'n8' },
      { id: 'c13', from: 'n8', to: 'n9' },
    ],
    defaultParams: {},
  },

  {
    id: 'tpl-egnn',
    name: 'EGNN (E(n)-Equivariant GNN)',
    family: 'gnn',
    description: 'EGNN (Satorras, Hoogeboom & Welling 2021) — 7 equivariant message-passing layers, 128 hidden, molecular property regression on QM9 (E(3)-equivariant coordinate updates alongside the usual feature messages, the property this architecture exists for)',
    tags: ['gnn', 'egnn', 'equivariant', 'molecular', 'qm9', 'satorras'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Molecular Graph Input (QM9)', x: 50, y: 140, params: { num_features: 15 } },
      { id: 'n2', type: 'linear', name: 'FC (15→128)', x: 250, y: 140, params: { in_features: 15, out_features: 128 } },
      { id: 'n3', type: 'message_passing', name: 'Equivariant Message Layer 1 (128, coord update)', x: 450, y: 140, params: { aggregator: 'sum', in_features: 128, out_features: 128, equivariant: true } },
      { id: 'n4', type: 'message_passing', name: 'Equivariant Message Layer 2-7 (128, coord update)', x: 650, y: 140, params: { aggregator: 'sum', in_features: 128, out_features: 128, equivariant: true, num_layers: 6 } },
      { id: 'n5', type: 'message_passing', name: 'Sum Readout', x: 850, y: 140, params: { aggregator: 'sum', in_features: 128, out_features: 128 } },
      { id: 'n6', type: 'linear', name: 'FC (128→1)', x: 1050, y: 140, params: { in_features: 128, out_features: 1 } },
      { id: 'n7', type: 'output', name: 'Molecular Property', x: 1250, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
    ],
    defaultParams: {},
  },

  {
    id: 'tpl-pna',
    name: 'PNA (Principal Neighbourhood Aggregation)',
    family: 'gnn',
    description: 'PNA (Corso et al. 2020) — 4 layers, 80 hidden, four aggregators combined per layer (mean, max, min, standard-deviation, the multi-aggregator design this architecture exists for) each scaled by degree, ZINC molecular graph regression',
    tags: ['gnn', 'pna', 'multi-aggregator', 'molecular', 'zinc', 'corso'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Molecular Graph Input (ZINC)', x: 50, y: 140, params: { num_features: 28 } },
      { id: 'n2', type: 'linear', name: 'FC (28→80)', x: 250, y: 140, params: { in_features: 28, out_features: 80 } },
      { id: 'n3', type: 'message_passing', name: 'PNA Layer 1 (mean/max/min/std, 80)', x: 450, y: 140, params: { aggregator: 'mean_max_min_std', in_features: 80, out_features: 80, degree_scalers: true } },
      { id: 'n4', type: 'message_passing', name: 'PNA Layer 2 (mean/max/min/std, 80)', x: 650, y: 140, params: { aggregator: 'mean_max_min_std', in_features: 80, out_features: 80, degree_scalers: true } },
      { id: 'n5', type: 'message_passing', name: 'PNA Layer 3 (mean/max/min/std, 80)', x: 850, y: 140, params: { aggregator: 'mean_max_min_std', in_features: 80, out_features: 80, degree_scalers: true } },
      { id: 'n6', type: 'message_passing', name: 'PNA Layer 4 (mean/max/min/std, 80)', x: 1050, y: 140, params: { aggregator: 'mean_max_min_std', in_features: 80, out_features: 80, degree_scalers: true } },
      { id: 'n7', type: 'message_passing', name: 'Sum Readout', x: 1250, y: 140, params: { aggregator: 'sum', in_features: 80, out_features: 80 } },
      { id: 'n8', type: 'linear', name: 'FC (80→1)', x: 1450, y: 140, params: { in_features: 80, out_features: 1 } },
      { id: 'n9', type: 'output', name: 'Molecular Property', x: 1650, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n4' }, { id: 'c4', from: 'n4', to: 'n5' },
      { id: 'c5', from: 'n5', to: 'n6' }, { id: 'c6', from: 'n6', to: 'n7' },
      { id: 'c7', from: 'n7', to: 'n8' }, { id: 'c8', from: 'n8', to: 'n9' },
    ],
    defaultParams: {},
  },

  // ====== MOE — batch 2 (2026-09 research pass) ======
// 4 new MoE templates, researched and verified (web search, September 2026) — NOT
// invented. Sources noted per entry. Ready to insert into MODEL_TEMPLATES in
// neurax-ui/src/data/modelTemplates.ts, following the exact pattern of the
// existing Mixtral 8x7B / DeepSeek-V3 entries.

    {
      id: 'tpl-grok-1',
      name: 'Grok-1',
      family: 'moe',
      description: 'Grok-1 (xAI 2024, open-sourced) — 314B params, ~78.5B active (25%), 64 layers, d_model 6144, 8 experts (top-2), GeGLU, intermediate 32768, RoPE. Source: xai-org/grok-1 GitHub release, hpcai-tech/grok-1 HF config.',
      tags: ['moe', 'grok', 'xai', 'top-2', 'geglu'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 4096
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 131072,
            hidden_size: 6144
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 6144,
            theta: 10000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '64× MoE Layer', x: 650, y: 140, params: {
            num_layers: 64
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 6144,
            eps: 1e-05
          } },
        { id: 'n6', type: 'gqa_attention', name: 'GQA (48 heads, 8 KV)', x: 850, y: 80, params: {
            hidden_size: 6144,
            num_heads: 48,
            num_kv_heads: 8,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 6144,
            eps: 1e-05
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-2 Router', x: 850, y: 200, params: {
            num_experts: 8,
            top_k: 2,
            hidden_size: 6144
          } },
        { id: 'n10', type: 'moe_layer', name: '8× GeGLU Experts', x: 1050, y: 200, params: {
            num_experts: 8,
            top_k: 2,
            intermediate_size: 32768,
            hidden_size: 6144,
            activation: 'geglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 6144,
            eps: 1e-05
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 131072,
            hidden_size: 6144
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-olmoe-1b-7b',
      name: 'OLMoE-1B-7B',
      family: 'moe',
      description: 'OLMoE-1B-7B (AllenAI 2024) — 6.9B params, ~1.3B active, 16 layers, d_model 2048, 64 experts (top-8), fully open weights+data+code. Source: allenai/OLMoE-1B-7B-0924 HF config, Muennighoff et al. 2024 paper.',
      tags: ['moe', 'olmoe', 'allenai', 'top-8', 'open'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 4096
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 50304,
            hidden_size: 2048
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 2048,
            theta: 10000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '16× MoE Layer', x: 650, y: 140, params: {
            num_layers: 16
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 2048,
            eps: 1e-05
          } },
        { id: 'n6', type: 'mha_attention', name: 'Multi-Head Attention (16 heads)', x: 850, y: 80, params: {
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 2048,
            eps: 1e-05
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-8 Router', x: 850, y: 200, params: {
            num_experts: 64,
            top_k: 8,
            hidden_size: 2048
          } },
        { id: 'n10', type: 'moe_layer', name: '64× SwiGLU Experts', x: 1050, y: 200, params: {
            num_experts: 64,
            top_k: 8,
            intermediate_size: 1024,
            hidden_size: 2048,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 2048,
            eps: 1e-05
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 50304,
            hidden_size: 2048
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-qwen3-30b-a3b',
      name: 'Qwen3-30B-A3B',
      family: 'moe',
      description: 'Qwen3-30B-A3B (Alibaba 2025) — 30.5B params, ~3.3B active, 48 layers, d_model 2048, 128 experts (top-8), expert width 768, GQA 4 KV heads. Source: Qwen/Qwen3-30B-A3B HF config.json.',
      tags: ['moe', 'qwen', 'alibaba', 'top-8', 'gqa'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 40960
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 151936,
            hidden_size: 2048
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 2048,
            theta: 1000000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '48× MoE Layer', x: 650, y: 140, params: {
            num_layers: 48
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n6', type: 'gqa_attention', name: 'GQA (32 heads, 4 KV)', x: 850, y: 80, params: {
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 64,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-8 Router', x: 850, y: 200, params: {
            num_experts: 128,
            top_k: 8,
            hidden_size: 2048
          } },
        { id: 'n10', type: 'moe_layer', name: '128× SwiGLU Experts', x: 1050, y: 200, params: {
            num_experts: 128,
            top_k: 8,
            intermediate_size: 768,
            hidden_size: 2048,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n12', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n13', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n14', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 151936,
            hidden_size: 2048
          } },
        { id: 'n15', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n13' },
        { id: 'c5', from: 'n13', to: 'n14' },
        { id: 'c6', from: 'n14', to: 'n15' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n11' },
        { id: 'c12', from: 'n11', to: 'n12' },
        { id: 'c13', from: 'n7', to: 'n4' },
        { id: 'c14', from: 'n12', to: 'n4' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-deepseek-v2-lite',
      name: 'DeepSeek-V2-Lite',
      family: 'moe',
      description: 'DeepSeek-V2-Lite (DeepSeek 2024) — 15.7B params, ~2.4B active, 27 layers, d_model 2048, 64 routed experts (top-6) + 2 shared experts, expert width 1408. Source: deepseek-ai/DeepSeek-V2-Lite HF config. Unlike the other MoE templates here, models this shared-expert branch explicitly (the existing DeepSeek MoE 16B/V2/V3 templates in this catalogue omit it) rather than folding it into the routed count.',
      tags: ['moe', 'deepseek', 'v2-lite', 'top-6', 'shared-expert'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 4096
          } },
        { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: {
            vocab_size: 102400,
            hidden_size: 2048
          } },
        { id: 'n3', type: 'pos_rope', name: 'RoPE', x: 450, y: 140, params: {
            hidden_size: 2048,
            theta: 10000.0
          } },
        { id: 'n4', type: 'layer_stack', name: '27× MoE Layer', x: 650, y: 140, params: {
            num_layers: 27
          } },
        { id: 'n5', type: 'rmsnorm', name: 'Pre-Attention RMSNorm', x: 650, y: 80, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n6', type: 'mha_attention', name: 'MLA-style Attention (16 heads)', x: 850, y: 80, params: {
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 128,
            causal: true
          } },
        { id: 'n7', type: 'residual_add', name: 'Residual Add', x: 1050, y: 80, params: {

          } },
        { id: 'n8', type: 'rmsnorm', name: 'Pre-MoE RMSNorm', x: 650, y: 200, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n9', type: 'noisy_topk_router', name: 'Top-6 Router', x: 850, y: 200, params: {
            num_experts: 64,
            top_k: 6,
            hidden_size: 2048
          } },
        { id: 'n10', type: 'moe_layer', name: '64× SwiGLU Routed Experts', x: 1050, y: 200, params: {
            num_experts: 64,
            top_k: 6,
            intermediate_size: 1408,
            hidden_size: 2048,
            activation: 'swiglu'
          } },
        { id: 'n11', type: 'shared_expert', name: '2× Shared Experts', x: 1050, y: 280, params: {
            num_experts: 2,
            intermediate_size: 1408,
            hidden_size: 2048
          } },
        { id: 'n12', type: 'expert_combine', name: 'Expert Combine', x: 1250, y: 200, params: {

          } },
        { id: 'n13', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {

          } },
        { id: 'n14', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: {
            hidden_size: 2048,
            eps: 1e-06
          } },
        { id: 'n15', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: {
            vocab_size: 102400,
            hidden_size: 2048
          } },
        { id: 'n16', type: 'output', name: 'Output', x: 2050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n14' },
        { id: 'c5', from: 'n14', to: 'n15' },
        { id: 'c6', from: 'n15', to: 'n16' },
        { id: 'c7', from: 'n5', to: 'n6' },
        { id: 'c8', from: 'n6', to: 'n7' },
        { id: 'c9', from: 'n8', to: 'n9' },
        { id: 'c10', from: 'n9', to: 'n10' },
        { id: 'c11', from: 'n10', to: 'n12' },
        { id: 'c12', from: 'n8', to: 'n11' },
        { id: 'c13', from: 'n11', to: 'n12' },
        { id: 'c14', from: 'n12', to: 'n13' },
        { id: 'c15', from: 'n7', to: 'n4' },
        { id: 'c16', from: 'n13', to: 'n4' }
      ],
      defaultParams: {},
    },

  // ====== RNN — batch 2 (2026-09 research pass) ======
{
      id: 'tpl-awd-lstm',
      name: 'AWD-LSTM',
      family: 'rnn',
      description: 'Merity, Keskar & Socher 2017 (arXiv:1708.02182) — 3-layer LSTM language model, 400d tied embeddings, hidden 1150/1150/400, DropConnect + variational dropout, ~24M params on Penn Treebank',
      tags: ['rnn', 'lstm', 'language-model', 'regularization', 'ptb'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            sequence_length: 70
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (400d, tied)', x: 250, y: 140, params: {
            vocab_size: 10000,
            embedding_dim: 400
          } },
        { id: 'n3', type: 'dropout', name: 'Embedding Dropout (0.1)', x: 450, y: 140, params: {
            rate: 0.1
          } },
        { id: 'n4', type: 'lstm_cell', name: 'LSTM Layer 1 (400→1150)', x: 650, y: 140, params: {
            hidden_size: 1150,
            num_layers: 1,
            bidirectional: false,
            dropout: 0.25
          } },
        { id: 'n5', type: 'lstm_cell', name: 'LSTM Layer 2 (1150→1150)', x: 850, y: 140, params: {
            hidden_size: 1150,
            num_layers: 1,
            bidirectional: false,
            dropout: 0.25
          } },
        { id: 'n6', type: 'lstm_cell', name: 'LSTM Layer 3 (1150→400)', x: 1050, y: 140, params: {
            hidden_size: 400,
            num_layers: 1,
            bidirectional: false,
            dropout: 0.25
          } },
        { id: 'n7', type: 'linear', name: 'Output Projection (tied, 400→10000)', x: 1250, y: 140, params: {
            in_features: 400,
            out_features: 10000
          } },
        { id: 'n8', type: 'output', name: 'Output', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-elmo',
      name: 'ELMo',
      family: 'rnn',
      description: 'Peters et al. 2018 (arXiv:1802.05365) — character-CNN word representation (2048 filters, 2 highway layers, 512d projection) feeding a 2-layer bidirectional LSTM, 4096 hidden units per direction per layer, 512d projections, residual connection between layers',
      tags: ['rnn', 'lstm', 'bilstm', 'contextual-embeddings', 'char-cnn'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Character Input', x: 50, y: 140, params: {
            sequence_length: 512
          } },
        { id: 'n2', type: 'conv1d', name: 'Char-CNN (2048 filters)', x: 250, y: 140, params: {
            out_channels: 2048,
            kernel_size: 7
          } },
        { id: 'n3', type: 'linear', name: 'Highway ×2 + Projection (→512)', x: 450, y: 140, params: {
            in_features: 2048,
            out_features: 512
          } },
        { id: 'n4', type: 'bilstm', name: 'biLSTM Layer 1 (4096/dir → 512 proj)', x: 650, y: 140, params: {
            hidden_size: 4096,
            num_layers: 1,
            bidirectional: true,
            proj_size: 512
          } },
        { id: 'n5', type: 'bilstm', name: 'biLSTM Layer 2 (4096/dir → 512 proj, residual)', x: 850, y: 140, params: {
            hidden_size: 4096,
            num_layers: 1,
            bidirectional: true,
            proj_size: 512
          } },
        { id: 'n6', type: 'concat', name: 'Layer-Weighted Sum', x: 1050, y: 140, params: {
            num_directions: 2
          } },
        { id: 'n7', type: 'output', name: 'Contextual Embeddings (1024d)', x: 1250, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-seq2seq-sutskever',
      name: 'Seq2Seq (Sutskever et al.)',
      family: 'rnn',
      description: 'Sutskever, Vinyals & Le 2014 (NeurIPS, arXiv:1409.3215) — 4-layer encoder LSTM + 4-layer decoder LSTM, 1000 cells per layer, 1000d word embeddings, 160K source / 80K target vocabulary, WMT-14 En-Fr',
      tags: ['rnn', 'lstm', 'seq2seq', 'encoder-decoder', 'translation'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Source Input', x: 50, y: 140, params: {
            sequence_length: 128
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (1000d)', x: 250, y: 140, params: {
            vocab_size: 160000,
            embedding_dim: 1000
          } },
        { id: 'n3', type: 'lstm_cell', name: 'Encoder LSTM (4 layers, 1000)', x: 450, y: 140, params: {
            hidden_size: 1000,
            num_layers: 4,
            bidirectional: false,
            dropout: 0.2
          } },
        { id: 'n4', type: 'lstm_cell', name: 'Decoder LSTM (4 layers, 1000)', x: 650, y: 140, params: {
            hidden_size: 1000,
            num_layers: 4,
            bidirectional: false,
            dropout: 0.2
          } },
        { id: 'n5', type: 'linear', name: 'FC (1000→80K)', x: 850, y: 140, params: {
            in_features: 1000,
            out_features: 80000
          } },
        { id: 'n6', type: 'output', name: 'Target Output', x: 1050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-bahdanau-nmt',
      name: 'Bahdanau NMT (RNNsearch)',
      family: 'rnn',
      description: 'Bahdanau, Cho & Bengio 2014 (arXiv:1409.0473) — the original attention paper: single-layer bidirectional GRU encoder (1000 hidden units/direction), single-layer GRU decoder (1000 hidden), 620d word embeddings, additive alignment MLP',
      tags: ['rnn', 'gru', 'seq2seq', 'attention', 'translation', 'bidirectional'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Source Input', x: 50, y: 140, params: {
            sequence_length: 128
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (620d)', x: 250, y: 140, params: {
            vocab_size: 30000,
            embedding_dim: 620
          } },
        { id: 'n3', type: 'bigru', name: 'Bidirectional GRU Encoder (1000/dir)', x: 450, y: 140, params: {
            hidden_size: 1000,
            num_layers: 1,
            bidirectional: true
          } },
        { id: 'n4', type: 'attention', name: 'Additive (Bahdanau) Attention', x: 650, y: 140, params: {
            hidden_size: 1000,
            attention_type: 'bahdanau'
          } },
        { id: 'n5', type: 'gru_cell', name: 'GRU Decoder (1000)', x: 850, y: 140, params: {
            hidden_size: 1000,
            num_layers: 1,
            bidirectional: false
          } },
        { id: 'n6', type: 'linear', name: 'FC (1000→30K)', x: 1050, y: 140, params: {
            in_features: 1000,
            out_features: 30000
          } },
        { id: 'n7', type: 'output', name: 'Target Output', x: 1250, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' }
      ],
      defaultParams: {},
    },

  // ====== GAN — batch 2 (2026-09 research pass) ======
{
      id: 'tpl-stylegan1',
      name: 'StyleGAN',
      family: 'gan',
      description: 'StyleGAN (Karras et al. 2019, CVPR) — 8-layer mapping network (512→512), 18-layer synthesis network (2 per resolution, 4×4→1024×1024) with AdaIN, 26.2M generator params on FFHQ',
      tags: ['gan', 'stylegan', 'adain', 'style-based', 'karras'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=512)', x: 50, y: 140, params: {
            latent_dim: 512
          } },
        { id: 'n2', type: 'linear', name: 'Mapping Network (8×512)', x: 250, y: 140, params: {
            in_features: 512,
            out_features: 512,
            num_layers: 8
          } },
        { id: 'n3', type: 'style_mod', name: 'AdaIN Style Modulation', x: 450, y: 140, params: {
            style_dim: 512,
            channels: 512
          } },
        { id: 'n4', type: 'synthesis_block', name: 'Synthesis Block (4×4, const input)', x: 650, y: 140, params: {
            channels: 512,
            resolution: 4
          } },
        { id: 'n5', type: 'synthesis_block', name: 'Synthesis Block (8×8)', x: 850, y: 140, params: {
            channels: 512,
            resolution: 8
          } },
        { id: 'n6', type: 'synthesis_block', name: 'Synthesis Block (16×16)', x: 1050, y: 140, params: {
            channels: 512,
            resolution: 16
          } },
        { id: 'n7', type: 'synthesis_block', name: 'Synthesis Block (32×32)', x: 1250, y: 140, params: {
            channels: 512,
            resolution: 32
          } },
        { id: 'n8', type: 'synthesis_block', name: 'Synthesis Block (64×64→1024×1024, 5 more)', x: 1450, y: 140, params: {
            channels: 256,
            resolution: 1024,
            remaining_blocks: 5
          } },
        { id: 'n9', type: 'to_rgb', name: 'ToRGB (16→3)', x: 1650, y: 140, params: {
            in_channels: 16,
            out_channels: 3
          } },
        { id: 'n10', type: 'output', name: 'Generated Image (1024×1024)', x: 1850, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' },
        { id: 'c9', from: 'n9', to: 'n10' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-pix2pix',
      name: 'Pix2Pix',
      family: 'gan',
      description: 'Pix2Pix (Isola et al. 2017) — U-Net generator (8 encoder + 8 mirrored decoder blocks, skip connections, channels 64→512 bottleneck) + 70×70 PatchGAN discriminator, 256×256',
      tags: ['gan', 'pix2pix', 'unet', 'image-to-image', 'patchgan', 'isola'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Source Image (256×256×3)', x: 50, y: 140, params: {

          } },
        { id: 'n2', type: 'conv2d', name: 'Encoder 1-4 (64→128→256→512, stride 2)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 512,
            kernel_size: 4,
            stride: 2,
            num_blocks: 4
          } },
        { id: 'n3', type: 'conv2d', name: 'Encoder 5-8 (512, stride 2, to 1×1 bottleneck)', x: 450, y: 140, params: {
            in_channels: 512,
            out_channels: 512,
            kernel_size: 4,
            stride: 2,
            num_blocks: 4
          } },
        { id: 'n4', type: 'conv2d', name: 'Decoder 1-4 (512, transposed, dropout)', x: 650, y: 140, params: {
            in_channels: 512,
            out_channels: 512,
            kernel_size: 4,
            stride: 2,
            transposed: true,
            num_blocks: 4
          } },
        { id: 'n5', type: 'concat', name: 'Skip Concat (encoder 4)', x: 850, y: 140, params: {

          } },
        { id: 'n6', type: 'conv2d', name: 'Decoder 5-8 (512→256→128→64, transposed)', x: 1050, y: 140, params: {
            in_channels: 1024,
            out_channels: 64,
            kernel_size: 4,
            stride: 2,
            transposed: true,
            num_blocks: 4
          } },
        { id: 'n7', type: 'concat', name: 'Skip Concat (encoder 1)', x: 1250, y: 140, params: {

          } },
        { id: 'n8', type: 'conv2d', name: 'Output Conv (128→3, tanh)', x: 1450, y: 140, params: {
            in_channels: 128,
            out_channels: 3,
            kernel_size: 4,
            stride: 1,
            activation: 'tanh'
          } },
        { id: 'n9', type: 'output', name: 'Generated Image (256×256)', x: 1650, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n2', to: 'n5' },
        { id: 'c6', from: 'n5', to: 'n6' },
        { id: 'c7', from: 'n6', to: 'n7' },
        { id: 'c8', from: 'n1', to: 'n7' },
        { id: 'c9', from: 'n7', to: 'n8' },
        { id: 'c10', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-pix2pixhd',
      name: 'pix2pixHD',
      family: 'gan',
      description: 'pix2pixHD (Wang et al. 2018) — coarse-to-fine generator: global generator G1 (conv front-end + 9 residual blocks + transposed-conv back-end, 1024 channels) + local enhancer G2 for 2048×1024 output, multi-scale PatchGAN discriminators',
      tags: ['gan', 'pix2pixhd', 'coarse-to-fine', 'multi-scale', 'wang'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Semantic Label Map (1024×512)', x: 50, y: 140, params: {

          } },
        { id: 'n2', type: 'conv2d', name: 'G1 Front-End (3→1024, downsample ×4)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 1024,
            kernel_size: 7,
            stride: 2,
            num_blocks: 4
          } },
        { id: 'n3', type: 'conv2d', name: 'G1 Residual Blocks ×9 (1024)', x: 450, y: 140, params: {
            in_channels: 1024,
            out_channels: 1024,
            kernel_size: 3,
            num_blocks: 9
          } },
        { id: 'n4', type: 'conv2d', name: 'G1 Back-End (1024→64, upsample ×4, transposed)', x: 650, y: 140, params: {
            in_channels: 1024,
            out_channels: 64,
            kernel_size: 3,
            stride: 2,
            transposed: true,
            num_blocks: 4
          } },
        { id: 'n5', type: 'conv2d', name: 'G2 Front-End (3→64, downsample ×1)', x: 850, y: 140, params: {
            in_channels: 3,
            out_channels: 64,
            kernel_size: 7,
            stride: 2
          } },
        { id: 'n6', type: 'residual_add', name: 'G1+G2 Feature Sum', x: 1050, y: 140, params: {

          } },
        { id: 'n7', type: 'conv2d', name: 'G2 Residual Blocks ×3 (64)', x: 1250, y: 140, params: {
            in_channels: 64,
            out_channels: 64,
            kernel_size: 3,
            num_blocks: 3
          } },
        { id: 'n8', type: 'conv2d', name: 'G2 Back-End (64→3, upsample ×1, transposed, tanh)', x: 1450, y: 140, params: {
            in_channels: 64,
            out_channels: 3,
            kernel_size: 7,
            stride: 2,
            transposed: true,
            activation: 'tanh'
          } },
        { id: 'n9', type: 'output', name: 'Generated Image (2048×1024)', x: 1650, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n1', to: 'n5' },
        { id: 'c5', from: 'n4', to: 'n6' },
        { id: 'c6', from: 'n5', to: 'n6' },
        { id: 'c7', from: 'n6', to: 'n7' },
        { id: 'c8', from: 'n7', to: 'n8' },
        { id: 'c9', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-gaugan',
      name: 'GauGAN (SPADE)',
      family: 'gan',
      description: 'GauGAN / SPADE (Park et al. 2019, CVPR) — semantic image synthesis, no encoder downsampling: a small learned constant/latent feeds a stack of SPADE ResBlocks with spatially-adaptive normalization from the segmentation map, nearest-neighbor upsampling to 256×256',
      tags: ['gan', 'gaugan', 'spade', 'semantic-synthesis', 'nvidia', 'park'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Segmentation Map + Latent (z=256)', x: 50, y: 140, params: {
            latent_dim: 256
          } },
        { id: 'n2', type: 'linear', name: 'FC (256→4×4×1024)', x: 250, y: 140, params: {
            in_features: 256,
            out_features: 16384
          } },
        { id: 'n3', type: 'reshape', name: 'Reshape (1024×4×4)', x: 450, y: 140, params: {
            shape: [1024, 4, 4]
          } },
        { id: 'n4', type: 'conv2d', name: 'SPADE ResBlk 1 (1024, upsample ×2)', x: 650, y: 140, params: {
            in_channels: 1024,
            out_channels: 1024,
            kernel_size: 3,
            spade: true,
            upsample: 2
          } },
        { id: 'n5', type: 'conv2d', name: 'SPADE ResBlk 2-4 (1024→512→256, upsample ×2 each)', x: 850, y: 140, params: {
            in_channels: 1024,
            out_channels: 256,
            kernel_size: 3,
            spade: true,
            upsample: 2,
            num_blocks: 3
          } },
        { id: 'n6', type: 'conv2d', name: 'SPADE ResBlk 5-6 (256→64, upsample ×2 each)', x: 1050, y: 140, params: {
            in_channels: 256,
            out_channels: 64,
            kernel_size: 3,
            spade: true,
            upsample: 2,
            num_blocks: 2
          } },
        { id: 'n7', type: 'conv2d', name: 'Output Conv (64→3, 3×3, tanh)', x: 1250, y: 140, params: {
            in_channels: 64,
            out_channels: 3,
            kernel_size: 3,
            activation: 'tanh'
          } },
        { id: 'n8', type: 'output', name: 'Generated Image (256×256)', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-stargan-v2',
      name: 'StarGAN v2',
      family: 'gan',
      description: 'StarGAN v2 (Choi et al. 2020, CVPR) — 4 components: generator with AdaIN (style code injected), mapping network (16-dim latent → 64-dim domain-specific style code), style encoder, and a multi-branch discriminator with one output per domain',
      tags: ['gan', 'stargan-v2', 'adain', 'multi-domain', 'choi'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Latent Input (z=16)', x: 50, y: 140, params: {
            latent_dim: 16
          } },
        { id: 'n2', type: 'linear', name: 'Mapping Network (shared 4×512 + per-domain branch)', x: 250, y: 140, params: {
            in_features: 16,
            out_features: 512,
            num_layers: 4
          } },
        { id: 'n3', type: 'dense', name: 'Domain-Specific Style Code (64-dim)', x: 450, y: 140, params: {
            in_features: 512,
            out_features: 64
          } },
        { id: 'n4', type: 'input', name: 'Source Image (256×256×3)', x: 50, y: 320, params: {

          } },
        { id: 'n5', type: 'conv2d', name: 'Encoder (64→512, downsample ×6)', x: 250, y: 320, params: {
            in_channels: 3,
            out_channels: 512,
            kernel_size: 3,
            stride: 2,
            num_blocks: 6
          } },
        { id: 'n6', type: 'style_mod', name: 'AdaIN Style Injection', x: 650, y: 230, params: {
            style_dim: 64,
            channels: 512
          } },
        { id: 'n7', type: 'conv2d', name: 'Decoder (512→64, upsample ×6, AdaIN ResBlocks)', x: 850, y: 230, params: {
            in_channels: 512,
            out_channels: 64,
            kernel_size: 3,
            stride: 2,
            transposed: true,
            num_blocks: 6
          } },
        { id: 'n8', type: 'conv2d', name: 'Output Conv (64→3, 1×1, tanh)', x: 1050, y: 230, params: {
            in_channels: 64,
            out_channels: 3,
            kernel_size: 1,
            activation: 'tanh'
          } },
        { id: 'n9', type: 'output', name: 'Generated Image (256×256)', x: 1250, y: 230, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n4', to: 'n5' },
        { id: 'c4', from: 'n3', to: 'n6' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-esrgan',
      name: 'ESRGAN',
      family: 'gan',
      description: 'ESRGAN (Wang et al. 2018, ECCV workshops) — RRDBNet super-resolution generator: 64 base channels, 23 Residual-in-Residual Dense Blocks (each 3 dense sub-blocks, 32 growth channels), 4× upsampling, 16.7M generator params + relativistic PatchGAN discriminator',
      tags: ['gan', 'esrgan', 'super-resolution', 'rrdb', 'wang'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Low-Res Image (64×64×3)', x: 50, y: 140, params: {

          } },
        { id: 'n2', type: 'conv2d', name: 'Feature Extraction (3→64, 3×3)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n3', type: 'dense_block', name: 'RRDB ×23 (64 channels, 32 growth, 3 dense sub-blocks each)', x: 450, y: 140, params: {
            in_channels: 64,
            growth_rate: 32,
            num_blocks: 23,
            sub_blocks: 3
          } },
        { id: 'n4', type: 'residual_add', name: 'Trunk Residual (skip from n2)', x: 650, y: 140, params: {

          } },
        { id: 'n5', type: 'conv2d', name: 'Upsample ×2 (Nearest + Conv, 64)', x: 850, y: 140, params: {
            in_channels: 64,
            out_channels: 64,
            kernel_size: 3,
            upsample: 2
          } },
        { id: 'n6', type: 'conv2d', name: 'Upsample ×2 (Nearest + Conv, 64)', x: 1050, y: 140, params: {
            in_channels: 64,
            out_channels: 64,
            kernel_size: 3,
            upsample: 2
          } },
        { id: 'n7', type: 'conv2d', name: 'Output Conv (64→3, 3×3)', x: 1250, y: 140, params: {
            in_channels: 64,
            out_channels: 3,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n8', type: 'output', name: 'Generated Image (256×256, ×4 SR)', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n2', to: 'n4' },
        { id: 'c5', from: 'n4', to: 'n5' },
        { id: 'c6', from: 'n5', to: 'n6' },
        { id: 'c7', from: 'n6', to: 'n7' },
        { id: 'c8', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-vqgan',
      name: 'VQGAN',
      family: 'gan',
      description: 'VQGAN (Esser et al. 2021, "Taming Transformers") — CNN encoder-decoder with a learned discrete codebook: downsampling factor f=16, codebook size 16384, embedding dimension 256, adversarial + perceptual loss on reconstruction',
      tags: ['gan', 'vqgan', 'vector-quantized', 'discrete-latent', 'esser', 'taming-transformers'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input Image (256×256×3)', x: 50, y: 140, params: {

          } },
        { id: 'n2', type: 'conv2d', name: 'Encoder (128→128→256→256→512, downsample ×16)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 512,
            kernel_size: 3,
            stride: 2,
            num_blocks: 4
          } },
        { id: 'n3', type: 'conv2d', name: 'Pre-Quant Conv (512→256)', x: 450, y: 140, params: {
            in_channels: 512,
            out_channels: 256,
            kernel_size: 1
          } },
        { id: 'n4', type: 'noisy_topk_router', name: 'Vector Quantizer (codebook 16384×256)', x: 650, y: 140, params: {
            codebook_size: 16384,
            embedding_dim: 256
          } },
        { id: 'n5', type: 'conv2d', name: 'Post-Quant Conv (256→512)', x: 850, y: 140, params: {
            in_channels: 256,
            out_channels: 512,
            kernel_size: 1
          } },
        { id: 'n6', type: 'conv2d', name: 'Decoder (512→256→256→128→128, upsample ×16)', x: 1050, y: 140, params: {
            in_channels: 512,
            out_channels: 128,
            kernel_size: 3,
            stride: 2,
            transposed: true,
            num_blocks: 4
          } },
        { id: 'n7', type: 'conv2d', name: 'Output Conv (128→3, 3×3, tanh)', x: 1250, y: 140, params: {
            in_channels: 128,
            out_channels: 3,
            kernel_size: 3,
            activation: 'tanh'
          } },
        { id: 'n8', type: 'output', name: 'Reconstructed Image (256×256)', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },
];
