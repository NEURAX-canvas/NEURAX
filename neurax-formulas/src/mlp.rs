//! MLP/FFN layer formulas
//!
//! Hot path — all functions are #[inline(always)] for zero-cost abstraction.

/// Compute FLOPs for MLP (Feed-Forward Network) layer
///
/// Standard MLP: Linear(hidden, intermediate) → activation → Linear(intermediate, hidden)
///
/// # Arguments
/// * `batch` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Input/output hidden dimension
/// * `intermediate_size` - Intermediate dimension (typically 4× hidden)
/// * `activation` - Activation type ("gelu", "relu", "silu", "none")
#[inline(always)]
pub fn mlp_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    activation: &str,
) -> f64 {
    // First linear: [B, S, H] × [H, I] → [B, S, I]
    let linear1_flops =
        2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * intermediate_size as f64;

    // Second linear: [B, S, I] × [I, H] → [B, S, H]
    let linear2_flops =
        2.0 * batch as f64 * seq_len as f64 * intermediate_size as f64 * hidden_size as f64;

    // Activation FLOPs, from the shared cost model in `crate::activation`.
    let act_flops = crate::activation::activation_flops_per_element(activation)
        * batch as f64
        * seq_len as f64
        * intermediate_size as f64;

    linear1_flops + linear2_flops + act_flops
}

/// Compute FLOPs for gated MLP (e.g., SwiGLU used in LLaMA)
///
/// Structure: gate_proj(x) * up_proj(x) → down_proj
#[inline(always)]
pub fn gated_mlp_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    intermediate_size: usize,
    activation: &str,
) -> f64 {
    // Gate projection: [B, S, H] × [H, I]
    let gate_flops =
        2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * intermediate_size as f64;

    // Up projection: [B, S, H] × [H, I]
    let up_flops =
        2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * intermediate_size as f64;

    // Down projection: [B, S, I] × [I, H]
    let down_flops =
        2.0 * batch as f64 * seq_len as f64 * intermediate_size as f64 * hidden_size as f64;

    // Activation, applied to the gate branch.
    let act_flops = crate::activation::activation_flops_per_element(activation)
        * batch as f64
        * seq_len as f64
        * intermediate_size as f64;

    // Element-wise multiplication
    let mul_flops = batch as f64 * seq_len as f64 * intermediate_size as f64;

    gate_flops + up_flops + down_flops + act_flops + mul_flops
}

/// Compute parameters for standard MLP
#[inline(always)]
pub fn mlp_params(hidden_size: usize, intermediate_size: usize, bias: bool) -> u64 {
    // Accumulate in u64 and saturate: `usize` products wrap silently for
    // oversized dimensions and would report a plausible but wrong count.
    let hidden_size = hidden_size as u64;
    let intermediate_size = intermediate_size as u64;
    let weight_params = hidden_size
        .saturating_mul(intermediate_size)
        .saturating_mul(2);
    let bias_params = if bias {
        intermediate_size.saturating_add(hidden_size)
    } else {
        0
    };
    weight_params.saturating_add(bias_params)
}

/// Compute parameters for gated MLP (SwiGLU style)
#[inline(always)]
pub fn gated_mlp_params(hidden_size: usize, intermediate_size: usize, bias: bool) -> u64 {
    // gate_proj + up_proj + down_proj — three matrices, hence 1.5x a plain MLP.
    let hidden_size = hidden_size as u64;
    let intermediate_size = intermediate_size as u64;
    let weight_params = hidden_size
        .saturating_mul(intermediate_size)
        .saturating_mul(3);
    let bias_params = if bias {
        intermediate_size.saturating_mul(3)
    } else {
        0
    };
    weight_params.saturating_add(bias_params)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_flops() {
        // GPT-2 small: hidden=768, intermediate=3072
        let flops = mlp_flops(1, 1024, 768, 3072, "gelu");
        // 2 × 1 × 1024 × 768 × 3072 × 2 + activation ≈ 9.7e9
        assert!(flops > 9e9 && flops < 11e9);
    }

    #[test]
    fn test_mlp_params() {
        let params = mlp_params(768, 3072, true);
        // 768×3072 + 3072×768 + 3072 + 768 = 4,718,592 + 3,840 = 4,722,432
        assert_eq!(params, 4_722_432);
    }

    #[test]
    fn test_gated_mlp_params() {
        // LLaMA style: 3 × hidden × intermediate
        let params = gated_mlp_params(4096, 11008, false);
        assert_eq!(params, 3 * 4096 * 11008);
    }
}
