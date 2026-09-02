//! Attention layer formulas
//!
//! Hot path — all functions are #[inline(always)] for zero-cost abstraction.

/// Head dimension, tolerating a zero head count.
///
/// `neurax-parser` rejects `num_heads == 0` before analysis, but these formulas
/// are part of a published crate and can be called directly, so treat a zero
/// head count as a single head rather than dividing by zero.
#[inline(always)]
fn head_dim_of(hidden_size: usize, num_heads: usize) -> usize {
    hidden_size / num_heads.max(1)
}

/// Compute FLOPs for self-attention layer
///
/// # Arguments
/// * `batch` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension
/// * `num_heads` - Number of attention heads
/// * `causal` - Whether attention is causal (masked)
///
/// # Returns
/// Total FLOPs for the attention layer (forward pass)
#[inline(always)]
pub fn attention_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    causal: bool,
) -> f64 {
    attention_flops_with_head_dim(
        batch,
        seq_len,
        hidden_size,
        num_heads,
        head_dim_of(hidden_size, num_heads),
        causal,
    )
}

/// Same as [`attention_flops`], for a model that states its own `head_dim`
/// independently of `hidden_size / num_heads` — real and increasingly
/// common (GLM-4.5: `hidden_size=5120, num_heads=96, head_dim=128`; note
/// `96 × 128 = 12288 ≠ 5120`, so the Q/output projection width is genuinely
/// wider than `hidden_size`, not just a rounding artifact of the divide).
/// When `head_dim * num_heads == hidden_size` (the derived case every
/// existing caller already uses) this is byte-identical to
/// [`attention_flops`].
#[inline(always)]
pub fn attention_flops_with_head_dim(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) -> f64 {
    let qk_width = num_heads * head_dim;

    // Q, K, V projections: 3 × (B × S × H × qk_width) matmuls
    let qkv_flops =
        3.0 * (2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * qk_width as f64);

    // QK^T attention scores: B × heads × S × S × head_dim
    // Matmul: [B, heads, S, head_dim] × [B, heads, head_dim, S] → [B, heads, S, S]
    let attn_scores_flops =
        2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64 * head_dim as f64;

    // Attention × V: [B, heads, S, S] × [B, heads, S, head_dim] → [B, heads, S, head_dim]
    let attn_v_flops =
        2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64 * head_dim as f64;

    // Output projection: [B, S, qk_width] × [qk_width, H] → [B, S, H]
    let out_proj_flops = 2.0 * batch as f64 * seq_len as f64 * qk_width as f64 * hidden_size as f64;

    // Softmax: ~5 × B × heads × S × S (exp, sum, div per position)
    let softmax_flops = 5.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64;

    // For causal attention, we only compute half the attention matrix
    let causal_factor = if causal { 0.5 } else { 1.0 };

    qkv_flops + (attn_scores_flops + attn_v_flops + softmax_flops) * causal_factor + out_proj_flops
}

/// FLOPs for attention with a bounded receptive field per query — sliding
/// window, dilated, or block-sparse attention, whose whole efficiency
/// argument (Mistral 7B, arXiv:2310.06825 — 4096-token sliding window
/// instead of full attention) is that a query attends to `kv_len`
/// positions, not all `seq_len` of them. Dense attention is the `kv_len ==
/// seq_len` special case of this, not a different formula — passing
/// `kv_len == seq_len` reproduces `attention_flops` exactly.
///
/// Every sub-quadratic pattern used to collapse into plain `attention_flops`
/// on the wire (sliding_window/dilated/sparse/linear attention all mapped
/// to `LayerType::Attention` with no pattern-carrying param even reaching
/// it), so choosing one of them changed nothing about the reported cost —
/// exactly the number a long-context design picks that block to reduce.
#[inline(always)]
pub fn windowed_attention_flops(
    batch: usize,
    seq_len: usize,
    kv_len: usize,
    hidden_size: usize,
    num_heads: usize,
    causal: bool,
) -> f64 {
    windowed_attention_flops_with_head_dim(
        batch,
        seq_len,
        kv_len,
        hidden_size,
        num_heads,
        head_dim_of(hidden_size, num_heads),
        causal,
    )
}

/// Same as [`windowed_attention_flops`], for an independent `head_dim` —
/// see [`attention_flops_with_head_dim`] for why this exists.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn windowed_attention_flops_with_head_dim(
    batch: usize,
    seq_len: usize,
    kv_len: usize,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
) -> f64 {
    let qk_width = num_heads * head_dim;
    let kv_len = kv_len.min(seq_len).max(1);

    let qkv_flops =
        3.0 * (2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * qk_width as f64);
    let attn_scores_flops =
        2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * kv_len as f64 * head_dim as f64;
    let attn_v_flops =
        2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * kv_len as f64 * head_dim as f64;
    let out_proj_flops = 2.0 * batch as f64 * seq_len as f64 * qk_width as f64 * hidden_size as f64;
    let softmax_flops = 5.0 * batch as f64 * num_heads as f64 * seq_len as f64 * kv_len as f64;
    let causal_factor = if causal { 0.5 } else { 1.0 };

    qkv_flops + (attn_scores_flops + attn_v_flops + softmax_flops) * causal_factor + out_proj_flops
}

/// Compute FLOPs for FlashAttention (memory-optimized, same FLOPs as standard)
#[inline(always)]
pub fn flash_attention_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    causal: bool,
) -> f64 {
    // FlashAttention has same FLOPs as standard attention but lower memory
    attention_flops(batch, seq_len, hidden_size, num_heads, causal)
}

/// Compute FLOPs for multi-query attention (MQA) or grouped-query attention (GQA)
#[inline(always)]
pub fn gqa_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    causal: bool,
) -> f64 {
    gqa_flops_with_head_dim(
        batch,
        seq_len,
        hidden_size,
        num_heads,
        num_kv_heads,
        head_dim_of(hidden_size, num_heads),
        causal,
    )
}

/// Same as [`gqa_flops`], for an independent `head_dim` — see
/// [`attention_flops_with_head_dim`] for why this exists. GLM-4.5 is
/// exactly this case: GQA (`num_kv_heads=8 < num_heads=96`) *and* a
/// `head_dim=128` that doesn't derive from `hidden_size/num_heads`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn gqa_flops_with_head_dim(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> f64 {
    let q_width = num_heads * head_dim;
    let kv_width = num_kv_heads * head_dim;

    // Q projection (full heads)
    let q_flops = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * q_width as f64;

    // K, V projections (reduced heads)
    let kv_flops = 2.0 * 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * kv_width as f64;

    // Attention computation
    let attn_scores_flops =
        2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64 * head_dim as f64;
    let attn_v_flops =
        2.0 * batch as f64 * num_heads as f64 * seq_len as f64 * seq_len as f64 * head_dim as f64;

    // Output projection
    let out_proj_flops = 2.0 * batch as f64 * seq_len as f64 * q_width as f64 * hidden_size as f64;

    let causal_factor = if causal { 0.5 } else { 1.0 };

    q_flops + kv_flops + (attn_scores_flops + attn_v_flops) * causal_factor + out_proj_flops
}

/// Compute parameters for attention layer
#[inline(always)]
pub fn attention_params(hidden_size: usize, num_heads: usize, bias: bool) -> u64 {
    let qk_width = num_heads.max(1) * head_dim_of(hidden_size, num_heads);
    attention_params_with_head_dim(hidden_size, qk_width, bias)
}

/// Same as [`attention_params`], for an independent `head_dim` — see
/// [`attention_flops_with_head_dim`] for why this exists. Takes `qk_width`
/// (`num_heads * head_dim`) directly rather than `num_heads`/`head_dim`
/// separately, since that's the only quantity these params actually need.
/// When `qk_width == hidden_size` (the derived case every existing caller
/// already uses) this is byte-identical to [`attention_params`].
#[inline(always)]
pub fn attention_params_with_head_dim(hidden_size: usize, qk_width: usize, bias: bool) -> u64 {
    // Accumulate in u64 and saturate: on 32-bit targets, and for oversized
    // dimensions generally, `usize` products wrap silently and would report a
    // plausible-looking but badly wrong parameter count.
    let hidden_size = hidden_size as u64;
    let qk_width = qk_width as u64;

    // Q, K, V projections: hidden_size -> qk_width each
    let qkv_params = hidden_size.saturating_mul(qk_width).saturating_mul(3);

    // Output projection: qk_width -> hidden_size
    let out_params = qk_width.saturating_mul(hidden_size);

    // Biases (optional)
    let bias_params = if bias {
        qk_width.saturating_mul(3).saturating_add(hidden_size) // Q, K, V, Out biases
    } else {
        0
    };

    qkv_params
        .saturating_add(out_params)
        .saturating_add(bias_params)
}

/// Compute parameters for a cross-attention layer (diffusion U-Nets,
/// encoder-decoder attention): Q comes from the block's own hidden state,
/// K and V are projected *from the conditioning sequence's own width*
/// (`context_dim` — e.g. the text encoder's output dimension), not from
/// `hidden_size`. Using `attention_params()` here previously conflated the
/// two: a real cross-attention block whose conditioning width differs from
/// its hidden size (SDXL's OpenCLIP text encoder is 2048-wide; its U-Net
/// blocks operate at 320-1280) was silently costed as if both matched.
#[inline(always)]
pub fn cross_attention_params(hidden_size: usize, context_dim: usize, bias: bool) -> u64 {
    let hidden_size = hidden_size as u64;
    let context_dim = context_dim as u64;

    // Q: hidden x hidden. K, V: context_dim x hidden each.
    let q_params = hidden_size.saturating_mul(hidden_size);
    let kv_params = context_dim.saturating_mul(hidden_size).saturating_mul(2);
    // Output projection: hidden x hidden.
    let out_params = hidden_size.saturating_mul(hidden_size);

    let bias_params = if bias {
        hidden_size.saturating_mul(4) // Q, K, V, Out biases
    } else {
        0
    };

    q_params
        .saturating_add(kv_params)
        .saturating_add(out_params)
        .saturating_add(bias_params)
}

/// Compute parameters for GQA/MQA
#[inline(always)]
pub fn gqa_params(hidden_size: usize, num_heads: usize, num_kv_heads: usize, bias: bool) -> u64 {
    let head_dim = head_dim_of(hidden_size, num_heads);
    gqa_params_with_head_dim(hidden_size, num_heads, num_kv_heads, head_dim, bias)
}

/// Same as [`gqa_params`], for an independent `head_dim` — see
/// [`attention_flops_with_head_dim`] for why this exists. GLM-4.5 is
/// exactly this case: GQA (`num_kv_heads=8 < num_heads=96`) *and* a
/// `head_dim=128` that doesn't derive from `hidden_size/num_heads` (5120/96
/// isn't even an integer). When `num_heads * head_dim == hidden_size` (the
/// derived case every existing caller already uses) this is byte-identical
/// to [`gqa_params`].
#[inline(always)]
pub fn gqa_params_with_head_dim(
    hidden_size: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    bias: bool,
) -> u64 {
    let q_width = (num_heads.max(1) as u64).saturating_mul(head_dim as u64);
    let kv_width = (num_kv_heads as u64).saturating_mul(head_dim as u64);
    let hidden_size = hidden_size as u64;

    // Q projection: hidden_size -> q_width
    let q_params = hidden_size.saturating_mul(q_width);

    // K, V projections (reduced): hidden_size -> kv_width each
    let kv_params = hidden_size.saturating_mul(kv_width).saturating_mul(2);

    // Output projection: q_width -> hidden_size
    let out_params = q_width.saturating_mul(hidden_size);

    let bias_params = if bias {
        q_width
            .saturating_mul(2)
            .saturating_add(kv_width.saturating_mul(2))
    } else {
        0
    };

    q_params
        .saturating_add(kv_params)
        .saturating_add(out_params)
        .saturating_add(bias_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windowed_attention_at_full_seq_matches_dense_attention_exactly() {
        // kv_len == seq_len is dense attention, not a different formula.
        let dense = attention_flops(2, 4096, 4096, 32, true);
        let windowed = windowed_attention_flops(2, 4096, 4096, 4096, 32, true);
        assert_eq!(dense, windowed);
    }

    #[test]
    fn test_mistral_shaped_sliding_window_costs_far_less_than_dense_at_long_context() {
        // Mistral 7B: 4096-token sliding window. At a 32k context, dense
        // attention's O(S^2) term dwarfs the windowed O(S*W) one.
        let seq_len = 32768;
        let window = 4096;
        let dense = attention_flops(1, seq_len, 4096, 32, true);
        let windowed = windowed_attention_flops(1, seq_len, window, 4096, 32, true);
        assert!(
            windowed < dense / 2.0,
            "a 4096-token window at 32k context should cost well under half of dense, got {windowed} vs {dense}"
        );
    }

    #[test]
    fn test_attention_flops_gpt2_small() {
        // GPT-2 small: batch=1, seq=1024, hidden=768, heads=12
        let flops = attention_flops(1, 1024, 768, 12, true);
        // Should be around 3-4e9 FLOPs per layer
        assert!(flops > 1e9 && flops < 1e10);
    }

    #[test]
    fn test_attention_params() {
        // GPT-2 small attention: Q, K, V, O projections
        // Q: 768×768, K: 768×768, V: 768×768, O: 768×768
        let params = attention_params(768, 12, true);
        // 4 × 768² + 4 × 768 = 2,359,296 + 3,072 = 2,362,368
        assert_eq!(params, 2_362_368);
    }

    // GLM-4.5's real, HuggingFace-verified shape (zai-org/GLM-4.5
    // config.json): hidden_size=5120, num_attention_heads=96,
    // num_key_value_heads=8, head_dim=128. `96 * 128 = 12288 != 5120` — the
    // Q/output projection is genuinely wider than hidden_size, not a
    // rounding artifact. These are the exact real numbers a real config
    // produces, not synthetic edge values.

    #[test]
    fn glm4_5_shaped_head_dim_widens_the_qk_projection_past_hidden_size() {
        let derived_from_hidden_size = gqa_params(5120, 96, 8, false);
        let with_real_head_dim = gqa_params_with_head_dim(5120, 96, 8, 128, false);
        assert_ne!(
            derived_from_hidden_size, with_real_head_dim,
            "a head_dim that doesn't derive from hidden_size/num_heads must change the result"
        );

        // Q: 5120 -> 12288, K+V: 5120 -> 1024 each, O: 12288 -> 5120.
        let expected = 5120u64 * 12288  // Q
            + 2 * 5120 * 1024           // K + V
            + 12288 * 5120; // O
        assert_eq!(with_real_head_dim, expected);
    }

    #[test]
    fn gqa_with_head_dim_matches_gqa_when_head_dim_is_the_derived_one() {
        // The whole point of the `_with_head_dim` variants: passing the
        // derived head_dim back in must reproduce the original formula
        // exactly, for every existing caller that never states one.
        let derived = head_dim_of(4096, 32);
        assert_eq!(
            gqa_params(4096, 32, 8, true),
            gqa_params_with_head_dim(4096, 32, 8, derived, true)
        );
        assert_eq!(
            gqa_flops(2, 2048, 4096, 32, 8, true),
            gqa_flops_with_head_dim(2, 2048, 4096, 32, 8, derived, true)
        );
    }

    #[test]
    fn glm4_5_shaped_head_dim_changes_flops_too() {
        let derived = gqa_flops(1, 4096, 5120, 96, 8, true);
        let real = gqa_flops_with_head_dim(1, 4096, 5120, 96, 8, 128, true);
        assert_ne!(derived, real);
        assert!(real > 0.0 && real.is_finite());
    }
}
