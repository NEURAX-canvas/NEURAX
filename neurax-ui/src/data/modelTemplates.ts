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
      { id: 'n2', type: 'conv2d', name: 'Conv Block 1: 2× [64, 3×3]', x: 250, y: 140, params: { in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1 } },
      { id: 'n3', type: 'relu', name: 'ReLU', x: 450, y: 140, params: {} },
      { id: 'n4', type: 'max_pool', name: '2×2 Max Pool', x: 650, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n5', type: 'conv2d', name: 'Conv Block 2: 2× [128, 3×3]', x: 850, y: 140, params: { in_channels: 64, out_channels: 128, kernel_size: 3, padding: 1 } },
      { id: 'n6', type: 'relu', name: 'ReLU', x: 1050, y: 140, params: {} },
      { id: 'n7', type: 'max_pool', name: '2×2 Max Pool', x: 1250, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n8', type: 'conv2d', name: 'Conv Block 3: 3× [256, 3×3]', x: 1450, y: 140, params: { in_channels: 128, out_channels: 256, kernel_size: 3, padding: 1 } },
      { id: 'n9', type: 'relu', name: 'ReLU', x: 1650, y: 140, params: {} },
      { id: 'n10', type: 'max_pool', name: '2×2 Max Pool', x: 1850, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n11', type: 'conv2d', name: 'Conv Block 4: 3× [512, 3×3]', x: 2050, y: 140, params: { in_channels: 256, out_channels: 512, kernel_size: 3, padding: 1 } },
      { id: 'n12', type: 'relu', name: 'ReLU', x: 2250, y: 140, params: {} },
      { id: 'n13', type: 'max_pool', name: '2×2 Max Pool', x: 2450, y: 140, params: { kernel_size: 2, stride: 2 } },
      { id: 'n14', type: 'conv2d', name: 'Conv Block 5: 3× [512, 3×3]', x: 2650, y: 140, params: { in_channels: 512, out_channels: 512, kernel_size: 3, padding: 1 } },
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
  {
    id: 'tpl-mamba-130m',
    name: 'Mamba-130M',
    family: 'ssm',
    description: 'Mamba — 130M params, 12 layers, 768 hidden, selective SSM (S6), causal conv1d + SiLU activation',
    tags: ['ssm', 'selective-scan', 'linear-time', 'efficient'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50277, hidden_size: 768 } },
      { id: 'n3', type: 'layer_stack', name: '12× Mamba Block', x: 450, y: 140, params: { num_layers: 12 } },
      { id: 'n4', type: 'ssm_in_proj', name: 'Linear Projection (2×)', x: 450, y: 80, params: { hidden_size: 768, ssm_expansion: 2 } },
      { id: 'n5', type: 'causal_conv1d', name: 'Causal Conv1D (kernel 4)', x: 650, y: 80, params: { kernel_size: 4, in_channels: 1536 } },
      { id: 'n6', type: 's6_block', name: 'Selective SSM (S6)', x: 850, y: 80, params: { state_dim: 16, d_model: 768, dt_min: 0.001, dt_max: 0.1 } },
      { id: 'n7', type: 'ssm_out_proj', name: 'Out Projection', x: 1050, y: 80, params: { hidden_size: 768 } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add + SiLU Gate', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'rmsnorm', name: 'RMSNorm', x: 1050, y: 200, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n10', type: 'ffn_standard', name: 'FFN (SiLU)', x: 1250, y: 200, params: { intermediate_size: 3072, activation: 'silu' } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {} },
      { id: 'n12', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n13', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: { vocab_size: 50277, hidden_size: 768 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n12' }, { id: 'c4', from: 'n12', to: 'n13' },
      { id: 'c5', from: 'n13', to: 'n14' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n3' }, { id: 'c13', from: 'n11', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba-370m',
    name: 'Mamba-370M',
    family: 'ssm',
    description: 'Mamba — 370M params, 24 layers, 1024 hidden, selective SSM with larger state dimension',
    tags: ['ssm', 'selective-scan', 'linear-time'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50277, hidden_size: 1024 } },
      { id: 'n3', type: 'layer_stack', name: '24× Mamba Block', x: 450, y: 140, params: { num_layers: 24 } },
      { id: 'n4', type: 'ssm_in_proj', name: 'Linear Projection (2×)', x: 450, y: 80, params: { hidden_size: 1024, ssm_expansion: 2 } },
      { id: 'n5', type: 'causal_conv1d', name: 'Causal Conv1D (kernel 4)', x: 650, y: 80, params: { kernel_size: 4, in_channels: 2048 } },
      { id: 'n6', type: 's6_block', name: 'Selective SSM (S6)', x: 850, y: 80, params: { state_dim: 16, d_model: 1024, dt_min: 0.001, dt_max: 0.1 } },
      { id: 'n7', type: 'ssm_out_proj', name: 'Out Projection', x: 1050, y: 80, params: { hidden_size: 1024 } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add + Gate', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'rmsnorm', name: 'RMSNorm', x: 1050, y: 200, params: { hidden_size: 1024, eps: 1e-5 } },
      { id: 'n10', type: 'ffn_standard', name: 'FFN (SiLU)', x: 1250, y: 200, params: { intermediate_size: 4096, activation: 'silu' } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {} },
      { id: 'n12', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: { hidden_size: 1024, eps: 1e-5 } },
      { id: 'n13', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: { vocab_size: 50277, hidden_size: 1024 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n12' }, { id: 'c4', from: 'n12', to: 'n13' },
      { id: 'c5', from: 'n13', to: 'n14' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n3' }, { id: 'c13', from: 'n11', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba-790m',
    name: 'Mamba-790M',
    family: 'ssm',
    description: 'Mamba — 790M params, 24 layers, 1536 hidden, selective SSM, larger model width',
    tags: ['ssm', 'selective-scan', 'linear-time'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50277, hidden_size: 1536 } },
      { id: 'n3', type: 'layer_stack', name: '24× Mamba Block', x: 450, y: 140, params: { num_layers: 24 } },
      { id: 'n4', type: 'ssm_in_proj', name: 'Linear Projection (2×)', x: 450, y: 80, params: { hidden_size: 1536, ssm_expansion: 2 } },
      { id: 'n5', type: 'causal_conv1d', name: 'Causal Conv1D (kernel 4)', x: 650, y: 80, params: { kernel_size: 4, in_channels: 3072 } },
      { id: 'n6', type: 's6_block', name: 'Selective SSM (S6)', x: 850, y: 80, params: { state_dim: 16, d_model: 1536, dt_min: 0.001, dt_max: 0.1 } },
      { id: 'n7', type: 'ssm_out_proj', name: 'Out Projection', x: 1050, y: 80, params: { hidden_size: 1536 } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add + Gate', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'rmsnorm', name: 'RMSNorm', x: 1050, y: 200, params: { hidden_size: 1536, eps: 1e-5 } },
      { id: 'n10', type: 'ffn_standard', name: 'FFN (SiLU)', x: 1250, y: 200, params: { intermediate_size: 6144, activation: 'silu' } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {} },
      { id: 'n12', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: { hidden_size: 1536, eps: 1e-5 } },
      { id: 'n13', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: { vocab_size: 50277, hidden_size: 1536 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n12' }, { id: 'c4', from: 'n12', to: 'n13' },
      { id: 'c5', from: 'n13', to: 'n14' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n3' }, { id: 'c13', from: 'n11', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba-1.4b',
    name: 'Mamba-1.4B',
    family: 'ssm',
    description: 'Mamba — 1.4B params, 48 layers, 2048 hidden, deep selective SSM architecture',
    tags: ['ssm', 'selective-scan', 'linear-time', 'deep'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50277, hidden_size: 2048 } },
      { id: 'n3', type: 'layer_stack', name: '48× Mamba Block', x: 450, y: 140, params: { num_layers: 48 } },
      { id: 'n4', type: 'ssm_in_proj', name: 'Linear Projection (2×)', x: 450, y: 80, params: { hidden_size: 2048, ssm_expansion: 2 } },
      { id: 'n5', type: 'causal_conv1d', name: 'Causal Conv1D (kernel 4)', x: 650, y: 80, params: { kernel_size: 4, in_channels: 4096 } },
      { id: 'n6', type: 's6_block', name: 'Selective SSM (S6)', x: 850, y: 80, params: { state_dim: 16, d_model: 2048, dt_min: 0.001, dt_max: 0.1 } },
      { id: 'n7', type: 'ssm_out_proj', name: 'Out Projection', x: 1050, y: 80, params: { hidden_size: 2048 } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add + Gate', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'rmsnorm', name: 'RMSNorm', x: 1050, y: 200, params: { hidden_size: 2048, eps: 1e-5 } },
      { id: 'n10', type: 'ffn_standard', name: 'FFN (SiLU)', x: 1250, y: 200, params: { intermediate_size: 8192, activation: 'silu' } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {} },
      { id: 'n12', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: { hidden_size: 2048, eps: 1e-5 } },
      { id: 'n13', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: { vocab_size: 50277, hidden_size: 2048 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n12' }, { id: 'c4', from: 'n12', to: 'n13' },
      { id: 'c5', from: 'n13', to: 'n14' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n3' }, { id: 'c13', from: 'n11', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba-2.8b',
    name: 'Mamba-2.8B',
    family: 'ssm',
    description: 'Mamba — 2.8B params, 64 layers, 2560 hidden, deepest selective SSM variant',
    tags: ['ssm', 'selective-scan', 'linear-time', 'deep'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50277, hidden_size: 2560 } },
      { id: 'n3', type: 'layer_stack', name: '64× Mamba Block', x: 450, y: 140, params: { num_layers: 64 } },
      { id: 'n4', type: 'ssm_in_proj', name: 'Linear Projection (2×)', x: 450, y: 80, params: { hidden_size: 2560, ssm_expansion: 2 } },
      { id: 'n5', type: 'causal_conv1d', name: 'Causal Conv1D (kernel 4)', x: 650, y: 80, params: { kernel_size: 4, in_channels: 5120 } },
      { id: 'n6', type: 's6_block', name: 'Selective SSM (S6)', x: 850, y: 80, params: { state_dim: 16, d_model: 2560 } },
      { id: 'n7', type: 'ssm_out_proj', name: 'Out Projection', x: 1050, y: 80, params: { hidden_size: 2560 } },
      { id: 'n8', type: 'residual_add', name: 'Residual Add + Gate', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'rmsnorm', name: 'RMSNorm', x: 1050, y: 200, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n10', type: 'ffn_standard', name: 'FFN (SiLU)', x: 1250, y: 200, params: { intermediate_size: 10240, activation: 'silu' } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {} },
      { id: 'n12', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n13', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: { vocab_size: 50277, hidden_size: 2560 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n12' }, { id: 'c4', from: 'n12', to: 'n13' },
      { id: 'c5', from: 'n13', to: 'n14' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n3' }, { id: 'c13', from: 'n11', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba2-130m',
    name: 'Mamba 2 130M',
    family: 'ssm',
    description: 'Mamba 2 — 130M params, 24 layers, 768 hidden, SSD (State Space Dual), improved hardware efficiency',
    tags: ['ssm', 'mamba2', 'ssd', 'selective-scan'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50277, hidden_size: 768 } },
      { id: 'n3', type: 'layer_stack', name: '24× Mamba2 Block', x: 450, y: 140, params: { num_layers: 24 } },
      { id: 'n4', type: 'ssm_in_proj', name: 'Linear Projection (2×)', x: 450, y: 80, params: { hidden_size: 768, ssm_expansion: 2 } },
      { id: 'n5', type: 'ssd_block', name: 'SSD (State Space Dual)', x: 650, y: 80, params: { state_dim: 64, d_model: 768, head_dim: 64, num_heads: 12 } },
      { id: 'n6', type: 'ssm_out_proj', name: 'Out Projection', x: 850, y: 80, params: { hidden_size: 768 } },
      { id: 'n7', type: 'rmsnorm', name: 'RMSNorm', x: 1050, y: 80, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n8', type: 'residual_add', name: 'Residual + Gate', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'ffn_gated', name: 'Gated FFN (SiLU)', x: 1050, y: 200, params: { intermediate_size: 3072, activation: 'silu' } },
      { id: 'n10', type: 'rmsnorm', name: 'RMSNorm', x: 1250, y: 200, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {} },
      { id: 'n12', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: { hidden_size: 768, eps: 1e-5 } },
      { id: 'n13', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: { vocab_size: 50277, hidden_size: 768 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n12' }, { id: 'c4', from: 'n12', to: 'n13' },
      { id: 'c5', from: 'n13', to: 'n14' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n3' }, { id: 'c13', from: 'n11', to: 'n3' },
    ],
    defaultParams: {},
  },
  {
    id: 'tpl-mamba2-2.7b',
    name: 'Mamba 2 2.7B',
    family: 'ssm',
    description: 'Mamba 2 — 2.7B params, 64 layers, 2560 hidden, SSD with larger state dimension, grouped heads',
    tags: ['ssm', 'mamba2', 'ssd', 'selective-scan', 'large'],
    nodes: [
      { id: 'n1', type: 'input', name: 'Input Tokens', x: 50, y: 140, params: { sequence_length: 2048 } },
      { id: 'n2', type: 'token_embedding', name: 'Token Embedding', x: 250, y: 140, params: { vocab_size: 50277, hidden_size: 2560 } },
      { id: 'n3', type: 'layer_stack', name: '64× Mamba2 Block', x: 450, y: 140, params: { num_layers: 64 } },
      { id: 'n4', type: 'ssm_in_proj', name: 'Linear Projection (2×)', x: 450, y: 80, params: { hidden_size: 2560, ssm_expansion: 2 } },
      { id: 'n5', type: 'ssd_block', name: 'SSD (grouped heads)', x: 650, y: 80, params: { state_dim: 128, d_model: 2560, head_dim: 64, num_heads: 40, num_groups: 8 } },
      { id: 'n6', type: 'ssm_out_proj', name: 'Out Projection', x: 850, y: 80, params: { hidden_size: 2560 } },
      { id: 'n7', type: 'rmsnorm', name: 'RMSNorm', x: 1050, y: 80, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n8', type: 'residual_add', name: 'Residual + Gate', x: 1250, y: 80, params: {} },
      { id: 'n9', type: 'ffn_gated', name: 'Gated FFN (SiLU)', x: 1050, y: 200, params: { intermediate_size: 10240, activation: 'silu' } },
      { id: 'n10', type: 'rmsnorm', name: 'RMSNorm', x: 1250, y: 200, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n11', type: 'residual_add', name: 'Residual Add', x: 1450, y: 200, params: {} },
      { id: 'n12', type: 'rmsnorm', name: 'Final RMSNorm', x: 1650, y: 140, params: { hidden_size: 2560, eps: 1e-5 } },
      { id: 'n13', type: 'lm_head', name: 'LM Head', x: 1850, y: 140, params: { vocab_size: 50277, hidden_size: 2560 } },
      { id: 'n14', type: 'output', name: 'Output', x: 2050, y: 140, params: {} },
    ],
    connections: [
      { id: 'c1', from: 'n1', to: 'n2' }, { id: 'c2', from: 'n2', to: 'n3' },
      { id: 'c3', from: 'n3', to: 'n12' }, { id: 'c4', from: 'n12', to: 'n13' },
      { id: 'c5', from: 'n13', to: 'n14' },
      { id: 'c6', from: 'n4', to: 'n5' }, { id: 'c7', from: 'n5', to: 'n6' },
      { id: 'c8', from: 'n6', to: 'n7' }, { id: 'c9', from: 'n7', to: 'n8' },
      { id: 'c10', from: 'n9', to: 'n10' }, { id: 'c11', from: 'n10', to: 'n11' },
      { id: 'c12', from: 'n8', to: 'n3' }, { id: 'c13', from: 'n11', to: 'n3' },
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
            expert_intermediate_size: 14336,
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
            expert_intermediate_size: 21504,
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
            expert_intermediate_size: 2816,
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
            expert_intermediate_size: 1536,
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
            expert_intermediate_size: 2048,
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
            expert_intermediate_size: 2816,
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
            expert_intermediate_size: 2816,
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
            expert_intermediate_size: 16384,
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

  // ====== RL (8) ======
    {
      id: 'tpl-dqn',
      name: 'DQN (Nature 2015)',
      family: 'rl',
      description: 'DQN (Mnih Nature 2015) — 3 conv + 2 FC (84×84 Atari frames, 32/64/64 filters, 512 FC, 4×4 frame stack, replay buffer 1M, ε-greedy 0.01)',
      tags: ['rl', 'dqn', 'nature', 'atari', 'conv'],
      nodes: [
        { id: 'n1', type: 'input', name: 'State (84×84×4)', x: 50, y: 140, params: {
            frame_height: 84,
            frame_width: 84,
            frame_stack: 4
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv 1 (4→32, 8×8, stride 4)', x: 250, y: 140, params: {
            in_channels: 4,
            out_channels: 32,
            kernel_size: 8,
            stride: 4,
            activation: 'relu'
          } },
        { id: 'n3', type: 'conv2d', name: 'Conv 2 (32→64, 4×4, stride 2)', x: 450, y: 140, params: {
            in_channels: 32,
            out_channels: 64,
            kernel_size: 4,
            stride: 2,
            activation: 'relu'
          } },
        { id: 'n4', type: 'conv2d', name: 'Conv 3 (64→64, 3×3, stride 1)', x: 650, y: 140, params: {
            in_channels: 64,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            activation: 'relu'
          } },
        { id: 'n5', type: 'flatten', name: 'Flatten', x: 850, y: 140, params: {

          } },
        { id: 'n6', type: 'linear', name: 'FC 1 (3136→512)', x: 1050, y: 140, params: {
            in_features: 3136,
            out_features: 512,
            activation: 'relu'
          } },
        { id: 'n7', type: 'q_network', name: 'Q-Head (512→num_actions)', x: 1250, y: 140, params: {
            in_features: 512,
            out_features: 18
          } },
        { id: 'n8', type: 'output', name: 'Q-Values (18 actions)', x: 1450, y: 140, params: {

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
      id: 'tpl-ppo-mlp',
      name: 'PPO MLP',
      family: 'rl',
      description: 'PPO MLP (Schulman 2017) — 2×64 hidden, tanh, continuous control (MuJoCo), clip ε=0.2, GAE λ=0.95, 10 epochs per update',
      tags: ['rl', 'ppo', 'mlp', 'continuous', 'schulman'],
      nodes: [
        { id: 'n1', type: 'input', name: 'State (obs_dim)', x: 50, y: 140, params: {
            observation_dim: 17
          } },
        { id: 'n2', type: 'linear', name: 'FC 1 (17→64)', x: 250, y: 140, params: {
            in_features: 17,
            out_features: 64,
            activation: 'tanh'
          } },
        { id: 'n3', type: 'linear', name: 'FC 2 (64→64)', x: 450, y: 140, params: {
            in_features: 64,
            out_features: 64,
            activation: 'tanh'
          } },
        { id: 'n4', type: 'value_head', name: 'Value Head (64→1)', x: 650, y: 140, params: {
            in_features: 64,
            out_features: 1
          } },
        { id: 'n5', type: 'policy_head', name: 'Policy Head (64→action_dim)', x: 650, y: 200, params: {
            in_features: 64,
            out_features: 6,
            std_init: 0.5
          } },
        { id: 'n6', type: 'output', name: 'Action + Value', x: 850, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n3', to: 'n5' },
        { id: 'c5', from: 'n4', to: 'n6' },
        { id: 'c6', from: 'n5', to: 'n6' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-ppo-cnn',
      name: 'PPO CNN (Atari)',
      family: 'rl',
      description: 'PPO CNN (Schulman 2017) — Nature CNN encoder, two heads (value/policy), 84×84, 4 frame stack, clip ε=0.2, GAE λ=0.95',
      tags: ['rl', 'ppo', 'cnn', 'atari', 'schulman'],
      nodes: [
        { id: 'n1', type: 'input', name: 'State (84×84×4)', x: 50, y: 140, params: {
            frame_height: 84,
            frame_width: 84,
            frame_stack: 4
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv 1 (4→32, 8×8, stride 4)', x: 250, y: 140, params: {
            in_channels: 4,
            out_channels: 32,
            kernel_size: 8,
            stride: 4,
            activation: 'relu'
          } },
        { id: 'n3', type: 'conv2d', name: 'Conv 2 (32→64, 4×4, stride 2)', x: 450, y: 140, params: {
            in_channels: 32,
            out_channels: 64,
            kernel_size: 4,
            stride: 2,
            activation: 'relu'
          } },
        { id: 'n4', type: 'conv2d', name: 'Conv 3 (64→64, 3×3, stride 1)', x: 650, y: 140, params: {
            in_channels: 64,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            activation: 'relu'
          } },
        { id: 'n5', type: 'flatten', name: 'Flatten', x: 850, y: 140, params: {

          } },
        { id: 'n6', type: 'linear', name: 'FC (3136→512)', x: 1050, y: 140, params: {
            in_features: 3136,
            out_features: 512,
            activation: 'tanh'
          } },
        { id: 'n7', type: 'value_head', name: 'Value Head', x: 1250, y: 80, params: {
            in_features: 512,
            out_features: 1
          } },
        { id: 'n8', type: 'policy_head', name: 'Policy Head (discrete)', x: 1250, y: 200, params: {
            in_features: 512,
            out_features: 18
          } },
        { id: 'n9', type: 'output', name: 'Action + Value', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n6', to: 'n8' },
        { id: 'c8', from: 'n7', to: 'n9' },
        { id: 'c9', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-sac',
      name: 'SAC (Soft Actor-Critic)',
      family: 'rl',
      description: 'SAC (Haarnoja 2018) — 3×256 hidden Q-networks (×2), reparameterization trick, auto-tuned α, target smoothing τ=0.005, continuous control',
      tags: ['rl', 'sac', 'soft-actor-critic', 'off-policy', 'haarnoja'],
      nodes: [
        { id: 'n1', type: 'input', name: 'State (obs_dim)', x: 50, y: 140, params: {
            observation_dim: 17
          } },
        { id: 'n2', type: 'linear', name: 'FC 1 (17→256)', x: 250, y: 140, params: {
            in_features: 17,
            out_features: 256,
            activation: 'relu'
          } },
        { id: 'n3', type: 'linear', name: 'FC 2 (256→256)', x: 450, y: 140, params: {
            in_features: 256,
            out_features: 256,
            activation: 'relu'
          } },
        { id: 'n4', type: 'actor_network', name: 'Actor (256→action_dim)', x: 650, y: 80, params: {
            in_features: 256,
            out_features: 6,
            reparameterize: true,
            log_std_min: -20,
            log_std_max: 2
          } },
        { id: 'n5', type: 'critic_network', name: 'Q1 Network (256+6→1)', x: 650, y: 200, params: {
            in_features: 262,
            out_features: 1,
            hidden_dim: 256
          } },
        { id: 'n6', type: 'critic_network', name: 'Q2 Network (256+6→1)', x: 850, y: 200, params: {
            in_features: 262,
            out_features: 1,
            hidden_dim: 256
          } },
        { id: 'n7', type: 'linear', name: 'Log Alpha (temperature)', x: 1050, y: 140, params: {
            in_features: 1,
            out_features: 1,
            activation: 'exp'
          } },
        { id: 'n8', type: 'output', name: 'Action + Q-Values + α', x: 1250, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n3', to: 'n5' },
        { id: 'c5', from: 'n3', to: 'n6' },
        { id: 'c6', from: 'n4', to: 'n8' },
        { id: 'c7', from: 'n5', to: 'n8' },
        { id: 'c8', from: 'n6', to: 'n8' },
        { id: 'c9', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-a2c',
      name: 'A2C LSTM',
      family: 'rl',
      description: 'A2C LSTM (Mnih 2016) — 2×128 + LSTM 256, POMDP, 2 heads (policy/value), 84×84 Atari, 5-step bootstrap, entropy coeff 0.01',
      tags: ['rl', 'a2c', 'lstm', 'pomdp', 'mnih'],
      nodes: [
        { id: 'n1', type: 'input', name: 'State (84×84×4)', x: 50, y: 140, params: {
            frame_height: 84,
            frame_width: 84,
            frame_stack: 4
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv 1 (4→32, 8×8, stride 4)', x: 250, y: 140, params: {
            in_channels: 4,
            out_channels: 32,
            kernel_size: 8,
            stride: 4,
            activation: 'relu'
          } },
        { id: 'n3', type: 'conv2d', name: 'Conv 2 (32→64, 4×4, stride 2)', x: 450, y: 140, params: {
            in_channels: 32,
            out_channels: 64,
            kernel_size: 4,
            stride: 2,
            activation: 'relu'
          } },
        { id: 'n4', type: 'flatten', name: 'Flatten', x: 650, y: 140, params: {

          } },
        { id: 'n5', type: 'linear', name: 'FC (3136→128)', x: 850, y: 140, params: {
            in_features: 3136,
            out_features: 128,
            activation: 'relu'
          } },
        { id: 'n6', type: 'lstm_cell', name: 'LSTM (128→256)', x: 1050, y: 140, params: {
            hidden_size: 256,
            num_layers: 1
          } },
        { id: 'n7', type: 'value_head', name: 'Value Head (256→1)', x: 1250, y: 80, params: {
            in_features: 256,
            out_features: 1
          } },
        { id: 'n8', type: 'policy_head', name: 'Policy Head (256→18)', x: 1250, y: 200, params: {
            in_features: 256,
            out_features: 18
          } },
        { id: 'n9', type: 'output', name: 'Action + Value', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n6', to: 'n8' },
        { id: 'c8', from: 'n7', to: 'n9' },
        { id: 'c9', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-td3',
      name: 'TD3 (Twin Delayed DDPG)',
      family: 'rl',
      description: 'TD3 (Fujimoto 2018) — 3×256 actor, 3×256 critic (×2), target network, delayed policy update, clipped double Q, target policy smoothing (noise=0.2, clip=0.5)',
      tags: ['rl', 'td3', 'twin-delayed', 'ddpg', 'fujimoto'],
      nodes: [
        { id: 'n1', type: 'input', name: 'State (obs_dim)', x: 50, y: 140, params: {
            observation_dim: 17
          } },
        { id: 'n2', type: 'linear', name: 'Actor FC 1 (17→256)', x: 250, y: 140, params: {
            in_features: 17,
            out_features: 256,
            activation: 'relu'
          } },
        { id: 'n3', type: 'linear', name: 'Actor FC 2 (256→256)', x: 450, y: 140, params: {
            in_features: 256,
            out_features: 256,
            activation: 'relu'
          } },
        { id: 'n4', type: 'actor_network', name: 'Actor Out (256→6)', x: 650, y: 140, params: {
            in_features: 256,
            out_features: 6,
            activation: 'tanh'
          } },
        { id: 'n5', type: 'concat', name: 'Concat (state + action)', x: 850, y: 140, params: {

          } },
        { id: 'n6', type: 'critic_network', name: 'Critic Q1 (256+6→256→1)', x: 1050, y: 140, params: {
            in_features: 262,
            out_features: 1,
            hidden_dim: 256
          } },
        { id: 'n7', type: 'critic_network', name: 'Critic Q2 (256+6→256→1)', x: 1250, y: 140, params: {
            in_features: 262,
            out_features: 1,
            hidden_dim: 256
          } },
        { id: 'n8', type: 'output', name: 'Action + Q1 + Q2', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n1', to: 'n5' },
        { id: 'c5', from: 'n4', to: 'n5' },
        { id: 'c6', from: 'n5', to: 'n6' },
        { id: 'c7', from: 'n5', to: 'n7' },
        { id: 'c8', from: 'n4', to: 'n8' },
        { id: 'c9', from: 'n6', to: 'n8' },
        { id: 'c10', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-impala',
      name: 'IMPALA (Deep ResNet)',
      family: 'rl',
      description: 'IMPALA (Espeholt 2018) — 15-layer deep ResNet, 2 heads (value/policy), LSTM 256, V-trace off-policy correction, 84×84',
      tags: ['rl', 'impala', 'resnet', 'vtrace', 'espeholt'],
      nodes: [
        { id: 'n1', type: 'input', name: 'State (84×84×4)', x: 50, y: 140, params: {
            frame_height: 84,
            frame_width: 84,
            frame_stack: 4
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv 1 (4→16, 8×8, stride 4)', x: 250, y: 140, params: {
            in_channels: 4,
            out_channels: 16,
            kernel_size: 8,
            stride: 4,
            activation: 'relu'
          } },
        { id: 'n3', type: 'residual_add', name: 'ResBlock 1 (16)', x: 450, y: 140, params: {
            num_blocks: 2,
            hidden_channels: 16
          } },
        { id: 'n4', type: 'conv2d', name: 'Conv 2 (16→32, 4×4, stride 2)', x: 650, y: 140, params: {
            in_channels: 16,
            out_channels: 32,
            kernel_size: 4,
            stride: 2,
            activation: 'relu'
          } },
        { id: 'n5', type: 'residual_add', name: 'ResBlock 2 (32)', x: 850, y: 140, params: {
            num_blocks: 2,
            hidden_channels: 32
          } },
        { id: 'n6', type: 'conv2d', name: 'Conv 3 (32→32, 3×3, stride 1)', x: 1050, y: 140, params: {
            in_channels: 32,
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            activation: 'relu'
          } },
        { id: 'n7', type: 'flatten', name: 'Flatten', x: 1250, y: 140, params: {

          } },
        { id: 'n8', type: 'linear', name: 'FC (1568→256)', x: 1450, y: 140, params: {
            in_features: 1568,
            out_features: 256
          } },
        { id: 'n9', type: 'lstm_cell', name: 'LSTM (256→256)', x: 1650, y: 140, params: {
            hidden_size: 256,
            num_layers: 1
          } },
        { id: 'n10', type: 'value_head', name: 'Value Head', x: 1850, y: 80, params: {
            in_features: 256,
            out_features: 1
          } },
        { id: 'n11', type: 'policy_head', name: 'Policy Head (18)', x: 1850, y: 200, params: {
            in_features: 256,
            out_features: 18
          } },
        { id: 'n12', type: 'output', name: 'Action + Value', x: 2050, y: 140, params: {

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
        { id: 'c10', from: 'n9', to: 'n11' },
        { id: 'c11', from: 'n10', to: 'n12' },
        { id: 'c12', from: 'n11', to: 'n12' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-rainbow',
      name: 'Rainbow DQN',
      family: 'rl',
      description: 'Rainbow DQN (Hessel 2018) — noisy nets + dueling + distributional (51 atoms) + double + priority + multi-step + PER, 3 conv, 84×84, 18 actions',
      tags: ['rl', 'rainbow', 'dqn', 'distributional', 'dueling', 'noisy'],
      nodes: [
        { id: 'n1', type: 'input', name: 'State (84×84×4)', x: 50, y: 140, params: {
            frame_height: 84,
            frame_width: 84,
            frame_stack: 4
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv 1 (4→32, 8×8, stride 4)', x: 250, y: 140, params: {
            in_channels: 4,
            out_channels: 32,
            kernel_size: 8,
            stride: 4,
            activation: 'relu'
          } },
        { id: 'n3', type: 'conv2d', name: 'Conv 2 (32→64, 4×4, stride 2)', x: 450, y: 140, params: {
            in_channels: 32,
            out_channels: 64,
            kernel_size: 4,
            stride: 2,
            activation: 'relu'
          } },
        { id: 'n4', type: 'conv2d', name: 'Conv 3 (64→64, 3×3, stride 1)', x: 650, y: 140, params: {
            in_channels: 64,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            activation: 'relu'
          } },
        { id: 'n5', type: 'flatten', name: 'Flatten', x: 850, y: 140, params: {

          } },
        { id: 'n6', type: 'noisy_linear', name: 'Noisy FC (3136→512)', x: 1050, y: 140, params: {
            in_features: 3136,
            out_features: 512,
            sigma_init: 0.5
          } },
        { id: 'n7', type: 'dueling_network', name: 'Dueling Head (512→18×51)', x: 1250, y: 140, params: {
            in_features: 512,
            num_actions: 18,
            num_atoms: 51
          } },
        { id: 'n8', type: 'output', name: 'Distributional Q (18×51)', x: 1450, y: 140, params: {

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

  // ====== SNN (8) ======
    {
      id: 'tpl-lif-snn',
      name: 'LIF SNN (2-layer)',
      family: 'snn',
      description: 'LIF neuron 2-layer feedforward (Zheng 2021) — 256→256, dt=1ms, tau=20ms, threshold=1.0, surrogate gradient (arctan), 4 timesteps, ~200K params',
      tags: ['snn', 'lif', 'feedforward', 'surrogate', 'zheng'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input (spike train)', x: 50, y: 140, params: {
            timesteps: 4,
            num_features: 128
          } },
        { id: 'n2', type: 'linear', name: 'FC (128→256)', x: 250, y: 140, params: {
            in_features: 128,
            out_features: 256
          } },
        { id: 'n3', type: 'lif_neuron', name: 'LIF Layer 1 (256, τ=20ms)', x: 450, y: 140, params: {
            hidden_size: 256,
            tau: 20.0,
            threshold: 1.0,
            dt: 1.0,
            surrogate: 'arctan'
          } },
        { id: 'n4', type: 'linear', name: 'FC (256→256)', x: 650, y: 140, params: {
            in_features: 256,
            out_features: 256
          } },
        { id: 'n5', type: 'lif_neuron', name: 'LIF Layer 2 (256, τ=20ms)', x: 850, y: 140, params: {
            hidden_size: 256,
            tau: 20.0,
            threshold: 1.0,
            dt: 1.0,
            surrogate: 'arctan'
          } },
        { id: 'n6', type: 'linear', name: 'Readout (256→10)', x: 1050, y: 140, params: {
            in_features: 256,
            out_features: 10
          } },
        { id: 'n7', type: 'output', name: 'Output (rate coding)', x: 1250, y: 140, params: {

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
      id: 'tpl-spiking-resnet',
      name: 'Spiking ResNet',
      family: 'snn',
      description: 'Spiking ResNet (Wu et al 2019) — 6 blocks, 64→128→256 channels, IF neurons, shortcut connections, 4 timesteps, CIFAR-10/100, ~1.2M params',
      tags: ['snn', 'spiking-resnet', 'wu', 'if-neuron', 'cifar'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Image Input (32×32×3)', x: 50, y: 140, params: {
            image_size: 32,
            channels: 3
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv 1 (3→64, 3×3)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n3', type: 'if_neuron', name: 'IF Neuron', x: 450, y: 140, params: {
            threshold: 1.0,
            reset: 'subtract'
          } },
        { id: 'n4', type: 'residual_add', name: 'Block 1 (64→64, 2×)', x: 650, y: 140, params: {
            num_blocks: 2,
            hidden_channels: 64
          } },
        { id: 'n5', type: 'conv2d', name: 'Block 2 (64→128, stride 2)', x: 850, y: 140, params: {
            in_channels: 64,
            out_channels: 128,
            kernel_size: 3,
            stride: 2,
            padding: 1
          } },
        { id: 'n6', type: 'residual_add', name: 'Block 3 (128→128, 2×)', x: 1050, y: 140, params: {
            num_blocks: 2,
            hidden_channels: 128
          } },
        { id: 'n7', type: 'conv2d', name: 'Block 4 (128→256, stride 2)', x: 1250, y: 140, params: {
            in_channels: 128,
            out_channels: 256,
            kernel_size: 3,
            stride: 2,
            padding: 1
          } },
        { id: 'n8', type: 'residual_add', name: 'Block 5 (256→256, 2×)', x: 1450, y: 140, params: {
            num_blocks: 2,
            hidden_channels: 256
          } },
        { id: 'n9', type: 'global_avg_pool', name: 'Global Avg Pool', x: 1650, y: 140, params: {

          } },
        { id: 'n10', type: 'linear', name: 'FC (256→10)', x: 1850, y: 140, params: {
            in_features: 256,
            out_features: 10
          } },
        { id: 'n11', type: 'output', name: 'Output (10 classes)', x: 2050, y: 140, params: {

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
      id: 'tpl-sew-resnet',
      name: 'SEW ResNet (34-layer)',
      family: 'snn',
      description: 'SEW ResNet (Fang et al 2021) — 34-layer, shortcut connections with element-wise (ADD/AND/IAND), IF neurons, 4 timesteps, ImageNet, ~21M params',
      tags: ['snn', 'sew-resnet', 'fang', 'shortcut', 'imagenet'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Image Input (224×224×3)', x: 50, y: 140, params: {
            image_size: 224,
            channels: 3
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv 1 (3→64, 7×7, stride 2)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 64,
            kernel_size: 7,
            stride: 2,
            padding: 3
          } },
        { id: 'n3', type: 'maxpool', name: 'MaxPool (3×3, stride 2)', x: 450, y: 140, params: {
            kernel_size: 3,
            stride: 2
          } },
        { id: 'n4', type: 'residual_add', name: 'SEW Block 1 (64, 3× ADD, IF)', x: 650, y: 140, params: {
            num_blocks: 3,
            hidden_channels: 64,
            sew_type: 'add'
          } },
        { id: 'n5', type: 'residual_add', name: 'SEW Block 2 (128, 4× ADD, IF)', x: 850, y: 140, params: {
            num_blocks: 4,
            hidden_channels: 128,
            sew_type: 'add',
            downsample: 2
          } },
        { id: 'n6', type: 'residual_add', name: 'SEW Block 3 (256, 6× ADD, IF)', x: 1050, y: 140, params: {
            num_blocks: 6,
            hidden_channels: 256,
            sew_type: 'add',
            downsample: 2
          } },
        { id: 'n7', type: 'residual_add', name: 'SEW Block 4 (512, 3× ADD, IF)', x: 1250, y: 140, params: {
            num_blocks: 3,
            hidden_channels: 512,
            sew_type: 'add',
            downsample: 2
          } },
        { id: 'n8', type: 'global_avg_pool', name: 'Global Avg Pool', x: 1450, y: 140, params: {

          } },
        { id: 'n9', type: 'linear', name: 'FC (512→1000)', x: 1650, y: 140, params: {
            in_features: 512,
            out_features: 1000
          } },
        { id: 'n10', type: 'output', name: 'Output (1000 classes)', x: 1850, y: 140, params: {

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
      id: 'tpl-plif',
      name: 'PLIF (Parametric LIF)',
      family: 'snn',
      description: 'PLIF (Fang 2021) — learnable time constants, 4 layers 256, surrogate gradient, 4 timesteps, Neuromorphic-MNIST, ~530K params',
      tags: ['snn', 'plif', 'parametric-lif', 'fang', 'learnable-tau'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input (spike train)', x: 50, y: 140, params: {
            timesteps: 4,
            num_features: 128
          } },
        { id: 'n2', type: 'linear', name: 'FC (128→256)', x: 250, y: 140, params: {
            in_features: 128,
            out_features: 256
          } },
        { id: 'n3', type: 'lif_neuron', name: 'PLIF Layer 1 (256, τ_learnable)', x: 450, y: 140, params: {
            hidden_size: 256,
            tau_init: 20.0,
            tau_learnable: true,
            threshold: 1.0,
            dt: 1.0,
            surrogate: 'sigmoid'
          } },
        { id: 'n4', type: 'linear', name: 'FC (256→256)', x: 650, y: 140, params: {
            in_features: 256,
            out_features: 256
          } },
        { id: 'n5', type: 'lif_neuron', name: 'PLIF Layer 2 (256, τ_learnable)', x: 850, y: 140, params: {
            hidden_size: 256,
            tau_init: 20.0,
            tau_learnable: true,
            threshold: 1.0,
            dt: 1.0,
            surrogate: 'sigmoid'
          } },
        { id: 'n6', type: 'linear', name: 'FC (256→256)', x: 1050, y: 140, params: {
            in_features: 256,
            out_features: 256
          } },
        { id: 'n7', type: 'lif_neuron', name: 'PLIF Layer 3 (256, τ_learnable)', x: 1250, y: 140, params: {
            hidden_size: 256,
            tau_init: 20.0,
            tau_learnable: true,
            threshold: 1.0,
            dt: 1.0,
            surrogate: 'sigmoid'
          } },
        { id: 'n8', type: 'linear', name: 'FC (256→256)', x: 1450, y: 140, params: {
            in_features: 256,
            out_features: 256
          } },
        { id: 'n9', type: 'lif_neuron', name: 'PLIF Layer 4 (256, τ_learnable)', x: 1650, y: 140, params: {
            hidden_size: 256,
            tau_init: 20.0,
            tau_learnable: true,
            threshold: 1.0,
            dt: 1.0,
            surrogate: 'sigmoid'
          } },
        { id: 'n10', type: 'linear', name: 'Readout (256→10)', x: 1850, y: 140, params: {
            in_features: 256,
            out_features: 10
          } },
        { id: 'n11', type: 'output', name: 'Output (10)', x: 2050, y: 140, params: {

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
      id: 'tpl-spiking-vgg',
      name: 'Spiking VGG-11',
      family: 'snn',
      description: 'Spiking VGG-11 (Hu 2021) — 8 conv + 3 FC, IF neurons, 4 timesteps, ImageNet/Tiny-ImageNet, ~9.2M params',
      tags: ['snn', 'spiking-vgg', 'hu', 'vgg', 'imagenet'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Image Input (224×224×3)', x: 50, y: 140, params: {
            image_size: 224,
            channels: 3
          } },
        { id: 'n2', type: 'conv2d', name: 'Conv 1 (3→64, 3×3)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n3', type: 'conv2d', name: 'Conv 2 (64→128, 3×3)', x: 450, y: 140, params: {
            in_channels: 64,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n4', type: 'maxpool', name: 'MaxPool (2×2) + IF', x: 650, y: 140, params: {
            kernel_size: 2,
            stride: 2
          } },
        { id: 'n5', type: 'conv2d', name: 'Conv 3 (128→256, 3×3)×2', x: 850, y: 140, params: {
            in_channels: 128,
            out_channels: 256,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            num_layers: 2
          } },
        { id: 'n6', type: 'maxpool', name: 'MaxPool (2×2) + IF', x: 1050, y: 140, params: {
            kernel_size: 2,
            stride: 2
          } },
        { id: 'n7', type: 'conv2d', name: 'Conv 4 (256→512, 3×3)×2', x: 1250, y: 140, params: {
            in_channels: 256,
            out_channels: 512,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            num_layers: 2
          } },
        { id: 'n8', type: 'maxpool', name: 'MaxPool (2×2) + IF', x: 1450, y: 140, params: {
            kernel_size: 2,
            stride: 2
          } },
        { id: 'n9', type: 'flatten', name: 'Flatten', x: 1650, y: 140, params: {

          } },
        { id: 'n10', type: 'linear', name: 'FC (8192→4096)', x: 1850, y: 140, params: {
            in_features: 8192,
            out_features: 4096
          } },
        { id: 'n11', type: 'linear', name: 'FC (4096→4096)', x: 2050, y: 140, params: {
            in_features: 4096,
            out_features: 4096
          } },
        { id: 'n12', type: 'linear', name: 'FC (4096→1000)', x: 2250, y: 140, params: {
            in_features: 4096,
            out_features: 1000
          } },
        { id: 'n13', type: 'output', name: 'Output (1000)', x: 2450, y: 140, params: {

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
        { id: 'c10', from: 'n10', to: 'n11' },
        { id: 'c11', from: 'n11', to: 'n12' },
        { id: 'c12', from: 'n12', to: 'n13' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-tdst',
      name: 'TDST (Spiking Transformer)',
      family: 'snn',
      description: 'TDST (Yao 2023) — Spiking Transformer with timestep encoding, 8 timesteps, 4 blocks, 256 dims, 8 heads, ImageNet, ~4.5M params',
      tags: ['snn', 'tdst', 'spiking-transformer', 'yao', 'imagenet'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Image Input (224×224×3)', x: 50, y: 140, params: {
            image_size: 224,
            channels: 3
          } },
        { id: 'n2', type: 'conv2d', name: 'Patch Embedding (16×16, 256)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 256,
            kernel_size: 16,
            stride: 16
          } },
        { id: 'n3', type: 'linear', name: 'Timestep Encoding (T=8)', x: 450, y: 140, params: {
            timesteps: 8,
            hidden_size: 256
          } },
        { id: 'n4', type: 'layer_stack', name: '4× Spiking Transformer Block', x: 650, y: 140, params: {
            num_layers: 4
          } },
        { id: 'n5', type: 'spiking_self_attention', name: 'Spiking MSA (8 heads, 256)', x: 650, y: 80, params: {
            hidden_size: 256,
            num_heads: 8,
            spike_mode: 'integrate-fire'
          } },
        { id: 'n6', type: 'mlp', name: 'MLP (256→1024→256)', x: 850, y: 80, params: {
            hidden_size: 256,
            intermediate_size: 1024
          } },
        { id: 'n7', type: 'spike_neuron', name: 'Spike Neuron', x: 1050, y: 140, params: {
            neuron_type: 'lif',
            tau: 10.0,
            threshold: 1.0
          } },
        { id: 'n8', type: 'layernorm', name: 'LayerNorm', x: 1250, y: 140, params: {
            eps: 1e-06,
            hidden_size: 256
          } },
        { id: 'n9', type: 'linear', name: 'FC (256→1000)', x: 1450, y: 140, params: {
            in_features: 256,
            out_features: 1000
          } },
        { id: 'n10', type: 'output', name: 'Output (1000)', x: 1650, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n8' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n4' },
        { id: 'c8', from: 'n8', to: 'n9' },
        { id: 'c9', from: 'n9', to: 'n10' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-spikformer',
      name: 'Spikformer',
      family: 'snn',
      description: 'Spikformer (Zhou 2023) — 4 blocks, 256 dims, 8 heads, spike self-attention, IF neurons, 4 timesteps, ImageNet, ~3.8M params',
      tags: ['snn', 'spikformer', 'zhou', 'spike-self-attention', 'imagenet'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Image Input (224×224×3)', x: 50, y: 140, params: {
            image_size: 224,
            channels: 3
          } },
        { id: 'n2', type: 'conv2d', name: 'Patch Embedding (16×16, 256)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 256,
            kernel_size: 16,
            stride: 16
          } },
        { id: 'n3', type: 'layer_stack', name: '4× Spikformer Block', x: 450, y: 140, params: {
            num_layers: 4
          } },
        { id: 'n4', type: 'spike_self_attention', name: 'Spike SSA (8 heads, 256)', x: 450, y: 80, params: {
            hidden_size: 256,
            num_heads: 8,
            spike_mode: 'if'
          } },
        { id: 'n5', type: 'mlp', name: 'MLP (256→1024→256)', x: 650, y: 80, params: {
            hidden_size: 256,
            intermediate_size: 1024
          } },
        { id: 'n6', type: 'if_neuron', name: 'IF Neuron', x: 850, y: 140, params: {
            threshold: 1.0,
            reset: 'subtract'
          } },
        { id: 'n7', type: 'layernorm', name: 'LayerNorm', x: 1050, y: 140, params: {
            eps: 1e-06,
            hidden_size: 256
          } },
        { id: 'n8', type: 'linear', name: 'FC (256→1000)', x: 1250, y: 140, params: {
            in_features: 256,
            out_features: 1000,
            bias: false
          } },
        { id: 'n9', type: 'output', name: 'Output (1000)', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n7' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n3' },
        { id: 'c7', from: 'n7', to: 'n8' },
        { id: 'c8', from: 'n8', to: 'n9' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-spiking-unet',
      name: 'Spiking UNet',
      family: 'snn',
      description: 'Spiking UNet (Kim 2022) — 5 stages encoder-decoder, LIF neurons, skip connections, 4 timesteps, segmentation, ~1.8M params',
      tags: ['snn', 'spiking-unet', 'kim', 'segmentation', 'skip-connection'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Image Input (256×256×3)', x: 50, y: 140, params: {
            image_size: 256,
            channels: 3
          } },
        { id: 'n2', type: 'conv2d', name: 'Encoder Stage 1 (3→32, 3×3)', x: 250, y: 140, params: {
            in_channels: 3,
            out_channels: 32,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n3', type: 'lif_neuron', name: 'LIF + MaxPool (2×2)', x: 450, y: 140, params: {
            neuron_type: 'lif',
            tau: 20.0,
            threshold: 1.0,
            pool: 2
          } },
        { id: 'n4', type: 'conv2d', name: 'Encoder Stage 2 (32→64, 3×3)', x: 650, y: 140, params: {
            in_channels: 32,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n5', type: 'lif_neuron', name: 'LIF + MaxPool (2×2)', x: 850, y: 140, params: {
            neuron_type: 'lif',
            tau: 20.0,
            threshold: 1.0,
            pool: 2
          } },
        { id: 'n6', type: 'conv2d', name: 'Encoder Stage 3 (64→128, 3×3)', x: 1050, y: 140, params: {
            in_channels: 64,
            out_channels: 128,
            kernel_size: 3,
            stride: 1,
            padding: 1
          } },
        { id: 'n7', type: 'concat', name: 'Skip Connection (n5→decoder)', x: 1250, y: 140, params: {

          } },
        { id: 'n8', type: 'conv2d', name: 'Decoder Up 1 (128→64, 3×3)', x: 1450, y: 140, params: {
            in_channels: 128,
            out_channels: 64,
            kernel_size: 3,
            stride: 1,
            padding: 1,
            upsample: 2
          } },
        { id: 'n9', type: 'lif_neuron', name: 'LIF + Upsample', x: 1650, y: 140, params: {
            neuron_type: 'lif',
            tau: 20.0,
            threshold: 1.0
          } },
        { id: 'n10', type: 'conv2d', name: 'Decoder Out (64→1, 1×1)', x: 1850, y: 140, params: {
            in_channels: 64,
            out_channels: 1,
            kernel_size: 1
          } },
        { id: 'n11', type: 'output', name: 'Segmentation Map', x: 2050, y: 140, params: {

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
        { id: 'c10', from: 'n10', to: 'n11' },
        { id: 'c11', from: 'n5', to: 'n7' }
      ],
      defaultParams: {},
    },

  // ====== EXPERIMENTAL (8) ======
    {
      id: 'tpl-neural-ode',
      name: 'Neural ODE',
      family: 'experimental',
      description: 'Neural ODE (Chen 2018) — 4-layer ODE function (128→128×4), ODE solver (dopri5), adjoint method, continuous depth, ~150K params',
      tags: ['experimental', 'neural-ode', 'chen', 'ode-solver', 'continuous-depth'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input', x: 50, y: 140, params: {
            num_features: 28
          } },
        { id: 'n2', type: 'linear', name: 'FC (28→128)', x: 250, y: 140, params: {
            in_features: 28,
            out_features: 128
          } },
        { id: 'n3', type: 'ode_solver', name: 'ODE Solve (dopri5, 128 hidden)', x: 450, y: 140, params: {
            solver: 'dopri5',
            atol: 1e-07,
            rtol: 1e-05,
            num_layers: 4,
            hidden_size: 128
          } },
        { id: 'n4', type: 'ode_func', name: 'ODE Func: FC (128→128)×4', x: 650, y: 140, params: {
            hidden_size: 128,
            num_layers: 4,
            activation: 'relu'
          } },
        { id: 'n5', type: 'linear', name: 'FC (128→10)', x: 850, y: 140, params: {
            in_features: 128,
            out_features: 10
          } },
        { id: 'n6', type: 'output', name: 'Output (10)', x: 1050, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n3' },
        { id: 'c5', from: 'n3', to: 'n5' },
        { id: 'c6', from: 'n5', to: 'n6' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-liquid-snn',
      name: 'Liquid Time Constant (LTC)',
      family: 'experimental',
      description: 'LTC (Hasani 2021) — 4 LTC cells, ODE solver, liquid time constants, 128 hidden, continuous system, causal, ~50K params',
      tags: ['experimental', 'ltc', 'liquid', 'hasani', 'ode'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input (time series)', x: 50, y: 140, params: {
            num_features: 2,
            sequence_length: 100
          } },
        { id: 'n2', type: 'linear', name: 'FC (2→128)', x: 250, y: 140, params: {
            in_features: 2,
            out_features: 128
          } },
        { id: 'n3', type: 'ltc_cell', name: 'LTC Cell 1 (128, τ_learnable)', x: 450, y: 140, params: {
            hidden_size: 128,
            solver: 'euler',
            num_layers: 1
          } },
        { id: 'n4', type: 'ltc_cell', name: 'LTC Cell 2 (128, τ_learnable)', x: 650, y: 140, params: {
            hidden_size: 128,
            solver: 'euler',
            num_layers: 1
          } },
        { id: 'n5', type: 'ltc_cell', name: 'LTC Cell 3 (128, τ_learnable)', x: 850, y: 140, params: {
            hidden_size: 128,
            solver: 'euler',
            num_layers: 1
          } },
        { id: 'n6', type: 'ltc_cell', name: 'LTC Cell 4 (128, τ_learnable)', x: 1050, y: 140, params: {
            hidden_size: 128,
            solver: 'euler',
            num_layers: 1
          } },
        { id: 'n7', type: 'linear', name: 'FC (128→1)', x: 1250, y: 140, params: {
            in_features: 128,
            out_features: 1
          } },
        { id: 'n8', type: 'output', name: 'Output (continuous)', x: 1450, y: 140, params: {

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
      id: 'tpl-hybrid-coeff',
      name: 'Differentiable Coincidence Detector',
      family: 'experimental',
      description: 'DCD hybrid coincidence detection, 2 layers 256, temporal coding, hybrid analog-spike, learnable thresholds, ~330K params',
      tags: ['experimental', 'dcd', 'coincidence', 'hybrid', 'temporal-coding'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input (temporal code)', x: 50, y: 140, params: {
            num_features: 64,
            timesteps: 10
          } },
        { id: 'n2', type: 'linear', name: 'FC (64→256)', x: 250, y: 140, params: {
            in_features: 64,
            out_features: 256
          } },
        { id: 'n3', type: 'temporal_conv', name: 'Coincidence Layer 1 (256, τ=5ms)', x: 450, y: 140, params: {
            hidden_size: 256,
            time_constant: 5.0,
            threshold_learnable: true
          } },
        { id: 'n4', type: 'linear', name: 'FC (256→256)', x: 650, y: 140, params: {
            in_features: 256,
            out_features: 256
          } },
        { id: 'n5', type: 'temporal_conv', name: 'Coincidence Layer 2 (256, τ=10ms)', x: 850, y: 140, params: {
            hidden_size: 256,
            time_constant: 10.0,
            threshold_learnable: true
          } },
        { id: 'n6', type: 'linear', name: 'Readout (256→10)', x: 1050, y: 140, params: {
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
      id: 'tpl-fpga-loop',
      name: 'FPGA Systolic Array Loop',
      family: 'experimental',
      description: 'FPGA-pipelined loop, systolic array 128×128, 16-bit fixed point, 4 pipeline stages, weight-stationary, ~32K MACs/cycle',
      tags: ['experimental', 'fpga', 'systolic', 'pipeline', 'hardware'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input Feature Map', x: 50, y: 140, params: {
            feature_dim: 128,
            precision: 'int16'
          } },
        { id: 'n2', type: 'reshape', name: 'Buffer + Reshape (128×128)', x: 250, y: 140, params: {
            shape: [128, 128]
          } },
        { id: 'n3', type: 'systolic_array', name: 'Systolic Array (128×128, weight-stationary)', x: 450, y: 140, params: {
            array_rows: 128,
            array_cols: 128,
            dataflow: 'weight-stationary',
            precision: 'int16'
          } },
        { id: 'n4', type: 'pipeline_stage', name: 'Pipeline Stage 1: MAC', x: 650, y: 140, params: {
            stage: 1,
            operation: 'mac',
            latency_cycles: 4
          } },
        { id: 'n5', type: 'pipeline_stage', name: 'Pipeline Stage 2: Accumulate', x: 850, y: 140, params: {
            stage: 2,
            operation: 'accumulate',
            latency_cycles: 2
          } },
        { id: 'n6', type: 'pipeline_stage', name: 'Pipeline Stage 3: Activate (ReLU)', x: 1050, y: 140, params: {
            stage: 3,
            operation: 'activation',
            activation: 'relu'
          } },
        { id: 'n7', type: 'pipeline_stage', name: 'Pipeline Stage 4: Writeback', x: 1250, y: 140, params: {
            stage: 4,
            operation: 'writeback'
          } },
        { id: 'n8', type: 'conv2d', name: 'Accumulator Output (128→64, 1×1)', x: 1450, y: 140, params: {
            in_channels: 128,
            out_channels: 64,
            kernel_size: 1
          } },
        { id: 'n9', type: 'output', name: 'Output Feature Map', x: 1650, y: 140, params: {

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
      id: 'tpl-neuromorphic',
      name: 'Event-driven SNN (AER)',
      family: 'experimental',
      description: 'Event-driven SNN, asynchronous AER input, 2-layer 256×256, address-event representation, 1ms resolution, ~200K synapses',
      tags: ['experimental', 'neuromorphic', 'aer', 'event-driven', 'asynchronous'],
      nodes: [
        { id: 'n1', type: 'input', name: 'AER Input (event stream)', x: 50, y: 140, params: {
            num_neurons: 1024,
            resolution_ms: 1.0
          } },
        { id: 'n2', type: 'aer_encoder', name: 'AER Encoder (address→spike train)', x: 250, y: 140, params: {
            num_neurons: 1024,
            encoding: 'address-event'
          } },
        { id: 'n3', type: 'synaptic_connection', name: 'Synaptic Layer 1 (1024→256, STDP)', x: 450, y: 140, params: {
            in_features: 1024,
            out_features: 256,
            plasticity: 'stdp',
            learning_rate: 0.01
          } },
        { id: 'n4', type: 'lif_neuron', name: 'LIF Neuron Pool 1 (256, τ=10ms)', x: 650, y: 140, params: {
            hidden_size: 256,
            tau: 10.0,
            threshold: 1.0,
            dt: 1.0
          } },
        { id: 'n5', type: 'synaptic_connection', name: 'Synaptic Layer 2 (256→256, STDP)', x: 850, y: 140, params: {
            in_features: 256,
            out_features: 256,
            plasticity: 'stdp',
            learning_rate: 0.01
          } },
        { id: 'n6', type: 'lif_neuron', name: 'LIF Neuron Pool 2 (256, τ=10ms)', x: 1050, y: 140, params: {
            hidden_size: 256,
            tau: 10.0,
            threshold: 1.0,
            dt: 1.0
          } },
        { id: 'n7', type: 'linear', name: 'FC (256→10)', x: 1250, y: 140, params: {
            in_features: 256,
            out_features: 10
          } },
        { id: 'n8', type: 'output', name: 'Output (rate-coded)', x: 1450, y: 140, params: {

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
      id: 'tpl-quantum-hybrid',
      name: 'Quantum-Classical Hybrid',
      family: 'experimental',
      description: 'Quantum-classical hybrid, 4 qubits, 4 layers, ZZ feature map, StronglyEntanglingLayers, measurement, ~100K classical params + 4 qubits',
      tags: ['experimental', 'quantum', 'hybrid', 'variational', 'pennylane'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Classical Input (4 features)', x: 50, y: 140, params: {
            num_features: 4
          } },
        { id: 'n2', type: 'linear', name: 'Classical Preprocess (4→16)', x: 250, y: 140, params: {
            in_features: 4,
            out_features: 16,
            activation: 'relu'
          } },
        { id: 'n3', type: 'quantum_circuit', name: 'Quantum Circuit (4 qubits, 4 layers)', x: 450, y: 140, params: {
            num_qubits: 4,
            num_layers: 4,
            feature_map: 'zz',
            variational_layer: 'strongly-entangling',
            measurement: 'expval'
          } },
        { id: 'n4', type: 'feature_map', name: 'ZZ Feature Map (4→4 qubits)', x: 650, y: 140, params: {
            num_qubits: 4,
            reps: 2,
            entanglement: 'linear'
          } },
        { id: 'n5', type: 'linear', name: 'Strongly Entangling Layers (θ)', x: 850, y: 140, params: {
            num_qubits: 4,
            num_layers: 4,
            rotations: ['rx', 'rz', 'rx']
          } },
        { id: 'n6', type: 'linear', name: 'Measurement (4 qubits→4 probs)', x: 1050, y: 140, params: {
            in_features: 4,
            out_features: 4,
            measurement: 'probability'
          } },
        { id: 'n7', type: 'linear', name: 'Classical Postprocess (4→2)', x: 1250, y: 140, params: {
            in_features: 4,
            out_features: 2
          } },
        { id: 'n8', type: 'output', name: 'Output (2 classes)', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n7' },
        { id: 'c4', from: 'n4', to: 'n5' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n3' },
        { id: 'c7', from: 'n7', to: 'n8' }
      ],
      defaultParams: {},
    },

    {
      id: 'tpl-hypernetwork',
      name: 'HyperNetwork',
      family: 'experimental',
      description: 'HyperNetwork (Ha 2017) — LSTM hyper (2 hidden, 1 layer) generating weights for primary network (2×32), ~200K params',
      tags: ['experimental', 'hypernetwork', 'ha', 'weight-generation', 'lstm'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Embedding Input (z=10)', x: 50, y: 140, params: {
            embedding_dim: 10
          } },
        { id: 'n2', type: 'embedding', name: 'Embedding (10→32)', x: 250, y: 140, params: {
            num_embeddings: 1000,
            embedding_dim: 32
          } },
        { id: 'n3', type: 'lstm_cell', name: 'Hyper LSTM (1 layer, 2× hidden)', x: 450, y: 140, params: {
            hidden_size: 32,
            num_layers: 1,
            num_hidden: 2
          } },
        { id: 'n4', type: 'linear', name: 'Weight Generator (32→64)', x: 650, y: 140, params: {
            in_features: 32,
            out_features: 64,
            activation: 'relu'
          } },
        { id: 'n5', type: 'linear', name: 'Weight Generator (64→128)', x: 850, y: 140, params: {
            in_features: 64,
            out_features: 128,
            activation: 'relu'
          } },
        { id: 'n6', type: 'linear', name: 'Primary FC 1 (10→32)', x: 1050, y: 140, params: {
            in_features: 10,
            out_features: 32,
            activation: 'relu',
            weights_from: 'hyper'
          } },
        { id: 'n7', type: 'linear', name: 'Primary FC 2 (32→10)', x: 1250, y: 140, params: {
            in_features: 32,
            out_features: 10,
            activation: 'softmax',
            weights_from: 'hyper'
          } },
        { id: 'n8', type: 'output', name: 'Output (10 classes)', x: 1450, y: 140, params: {

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
      id: 'tpl-nfn',
      name: 'Neural Functional Network (NFN)',
      family: 'experimental',
      description: 'Neural Functional Network (NFN) — DeepSets + transformer, 4 layers, 128 hidden, permutation-invariant, function-space, ~500K params',
      tags: ['experimental', 'nfn', 'function-space', 'deepsets', 'permutation-invariant'],
      nodes: [
        { id: 'n1', type: 'input', name: 'Input (set of points)', x: 50, y: 140, params: {
            set_size: 50,
            feature_dim: 2
          } },
        { id: 'n2', type: 'linear', name: 'DeepSets Encoder (2→128)', x: 250, y: 140, params: {
            in_features: 2,
            out_features: 128,
            activation: 'relu'
          } },
        { id: 'n3', type: 'deepsets', name: 'DeepSets Pool (sum)', x: 450, y: 140, params: {
            aggregator: 'sum',
            hidden_size: 128
          } },
        { id: 'n4', type: 'layer_stack', name: '4× Transformer Block', x: 650, y: 140, params: {
            num_layers: 4
          } },
        { id: 'n5', type: 'mha_attention', name: 'Self-Attention (4 heads, 128)', x: 650, y: 80, params: {
            hidden_size: 128,
            num_heads: 4,
            dropout: 0.1
          } },
        { id: 'n6', type: 'mlp', name: 'MLP (128→512→128)', x: 850, y: 80, params: {
            hidden_size: 128,
            intermediate_size: 512
          } },
        { id: 'n7', type: 'layernorm', name: 'LayerNorm', x: 1050, y: 140, params: {
            eps: 1e-06,
            hidden_size: 128
          } },
        { id: 'n8', type: 'functional_layer', name: 'Functional Readout (128→1)', x: 1250, y: 140, params: {
            in_features: 128,
            out_features: 1,
            readout: 'sum'
          } },
        { id: 'n9', type: 'output', name: 'Output', x: 1450, y: 140, params: {

          } }
      ],
      connections: [
        { id: 'c1', from: 'n1', to: 'n2' },
        { id: 'c2', from: 'n2', to: 'n3' },
        { id: 'c3', from: 'n3', to: 'n4' },
        { id: 'c4', from: 'n4', to: 'n7' },
        { id: 'c5', from: 'n5', to: 'n6' },
        { id: 'c6', from: 'n6', to: 'n7' },
        { id: 'c7', from: 'n7', to: 'n4' },
        { id: 'c8', from: 'n7', to: 'n8' },
        { id: 'c9', from: 'n8', to: 'n9' }
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
];