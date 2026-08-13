//! State Space Model (SSM) formulas - Mamba, S4, H3
//!
//! Hot path — all functions are #[inline(always)] for zero-cost abstraction.

/// Compute FLOPs for Mamba SSM block
///
/// # Arguments
/// * `batch` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension (d_model)
/// * `state_dim` - SSM state dimension (d_state, typically 16)
/// * `expand_factor` - Expansion factor (typically 2)
#[inline(always)]
pub fn mamba_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    state_dim: usize,
    expand_factor: usize,
) -> f64 {
    let d_inner = hidden_size * expand_factor;

    // Input projection: [B, S, H] -> [B, S, d_inner * 2] (for x and z branches)
    let in_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * (d_inner * 2) as f64;

    // Conv1d: [B, d_inner, S] with kernel size 4
    let conv1d = 2.0 * batch as f64 * d_inner as f64 * seq_len as f64 * 4.0;

    // SSM state update: selective scan
    // A, B, C, D parameters computed from x
    // State update: h[t] = A * h[t-1] + B * x[t]
    // Output: y[t] = C * h[t] + D * x[t]
    let ssm_params = 4.0 * d_inner as f64 * state_dim as f64; // A, B, C, D
    let ssm_state = 2.0 * batch as f64 * seq_len as f64 * d_inner as f64 * state_dim as f64;
    let ssm_output = 2.0 * batch as f64 * seq_len as f64 * d_inner as f64;

    // Output projection: [B, S, d_inner] -> [B, S, H]
    let out_proj = 2.0 * batch as f64 * seq_len as f64 * d_inner as f64 * hidden_size as f64;

    in_proj + conv1d + ssm_params + ssm_state + ssm_output + out_proj
}

/// Compute FLOPs for S4 (Structured State Space) block
#[inline(always)]
pub fn s4_flops(batch: usize, seq_len: usize, hidden_size: usize, state_dim: usize) -> f64 {
    // S4 uses a structured matrix for efficient computation
    // FFT-based convolution: O(N log N) instead of O(N^2)

    // Input projection
    let in_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;

    // FFT convolution (simplified)
    let fft_flops =
        batch as f64 * hidden_size as f64 * seq_len as f64 * (seq_len as f64).log2() * 2.0;

    // State update
    let state_flops = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * state_dim as f64;

    // Output projection
    let out_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;

    in_proj + fft_flops + state_flops + out_proj
}

/// Compute FLOPs for H3 (Hungry Hungry Hippos) block
#[inline(always)]
pub fn h3_flops(batch: usize, seq_len: usize, hidden_size: usize, state_dim: usize) -> f64 {
    // H3 combines SSM with attention-like computation

    // Input projection
    let in_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;

    // SSM layers (2 for H3)
    let ssm_flops = 2.0 * s4_flops(batch, seq_len, hidden_size, state_dim);

    // Output projection
    let out_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * hidden_size as f64;

    in_proj + ssm_flops + out_proj
}

/// Compute parameters for Mamba block
#[inline(always)]
pub fn mamba_params(hidden_size: usize, state_dim: usize, expand_factor: usize) -> u64 {
    let d_inner = hidden_size * expand_factor;

    // Input projection: H -> d_inner * 2
    let in_proj = hidden_size * (d_inner * 2);

    // Conv1d: d_inner * kernel_size (typically 4)
    let conv1d = d_inner * 4;

    // SSM parameters: A, B, C, D for d_inner
    // A: d_inner * state_dim (log NDT)
    // B, C: d_inner * state_dim each
    // D: d_inner
    let ssm = d_inner * state_dim * 3 + d_inner;

    // Output projection: d_inner -> H
    let out_proj = d_inner * hidden_size;

    (in_proj + conv1d + ssm + out_proj) as u64
}

/// Compute FLOPs for Mamba Conv1d operation specifically
#[inline(always)]
pub fn mamba_conv1d_flops(batch: usize, seq_len: usize, d_inner: usize, kernel_size: usize) -> f64 {
    2.0 * batch as f64 * d_inner as f64 * seq_len as f64 * kernel_size as f64
}

/// Compute FLOPs for SSM state update specifically
#[inline(always)]
pub fn ssm_state_update_flops(
    batch: usize,
    seq_len: usize,
    d_inner: usize,
    state_dim: usize,
) -> f64 {
    // h[t] = A * h[t-1] + B * x[t]
    // y[t] = C * h[t] + D * x[t]
    4.0 * batch as f64 * seq_len as f64 * d_inner as f64 * state_dim as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba_flops() {
        // Mamba-1.4B: hidden=2048, state=16, expand=2
        let flops = mamba_flops(1, 2048, 2048, 16, 2);
        assert!(flops > 0.0);
    }

    #[test]
    fn test_mamba_params() {
        let params = mamba_params(2048, 16, 2);
        // Should be roughly 4 * hidden^2 for projections plus SSM params
        assert!(params > 0);
    }
}

/// Parameters of one RWKV block.
///
/// RWKV is not a state-space model and not an LSTM, and it was being costed
/// with Mamba's formula, which gave RWKV-7B 3.4B parameters against a
/// published 7.5B. Its block is two sublayers:
///
/// * **Time mixing** — four square projections (receptance, key, value,
///   output), plus per-channel decay and bonus vectors that are linear in the
///   width and negligible beside the matrices, but counted because they are
///   real weights.
/// * **Channel mixing** — the feed-forward half: a square receptance
///   projection, and a key/value pair through the intermediate width.
///
/// `intermediate_size` defaults to 4x the model width, which is what the
/// released RWKV checkpoints use.
pub fn rwkv_params(hidden_size: usize, intermediate_size: Option<usize>) -> u64 {
    let hidden = hidden_size as u64;
    let intermediate = intermediate_size.map(|i| i as u64).unwrap_or(hidden * 4);

    // Time mixing: R, K, V and the output projection.
    let time_mix = hidden.saturating_mul(hidden).saturating_mul(4);
    // Decay, first-token bonus, and the three time-shift mixing vectors.
    let time_decay = hidden.saturating_mul(5);

    // Channel mixing: receptance is square, key and value cross the
    // intermediate width.
    let channel_mix = hidden
        .saturating_mul(hidden)
        .saturating_add(hidden.saturating_mul(intermediate).saturating_mul(2));
    let channel_shift = hidden.saturating_mul(2);

    // Two layer norms, each with a weight and a bias.
    let norms = hidden.saturating_mul(4);

    time_mix
        .saturating_add(time_decay)
        .saturating_add(channel_mix)
        .saturating_add(channel_shift)
        .saturating_add(norms)
}

/// Parameters of one retention block (RetNet).
///
/// Multi-scale retention keeps the transformer's four projections and adds a
/// swish gate of the same width, then a gated feed-forward.
pub fn retention_params(hidden_size: usize, intermediate_size: Option<usize>) -> u64 {
    let hidden = hidden_size as u64;
    let intermediate = intermediate_size.map(|i| i as u64).unwrap_or(hidden * 4);

    // Q, K, V, output, and the gate.
    let retention = hidden.saturating_mul(hidden).saturating_mul(5);
    // Gated feed-forward, as in the reference implementation.
    let ffn = hidden.saturating_mul(intermediate).saturating_mul(2);
    let norms = hidden.saturating_mul(4);

    retention.saturating_add(ffn).saturating_add(norms)
}

#[cfg(test)]
mod recurrent_tests {
    use super::*;

    /// RWKV-7B: 32 blocks of width 4096, published at ~7.5B parameters with
    /// the embedding and the head.
    #[test]
    fn rwkv_7b_matches_its_published_size() {
        let per_block = rwkv_params(4096, None);
        let vocab = 50_277u64;
        let total = per_block * 32 + vocab * 4096 * 2;

        let published = 7.5e9;
        let error = (total as f64 - published).abs() / published;
        assert!(
            error < 0.10,
            "RWKV-7B came out at {:.2}B, {:.1}% from the published 7.5B",
            total as f64 / 1e9,
            error * 100.0
        );
    }

    /// The two halves of the block are the same order of magnitude; a formula
    /// that dropped one would still look plausible on its own.
    #[test]
    fn both_sublayers_contribute() {
        let narrow = rwkv_params(1024, Some(0));
        let wide = rwkv_params(1024, Some(4096));
        assert!(wide > narrow * 2, "channel mixing barely moved the total");
    }

    #[test]
    fn retention_is_wider_than_plain_attention() {
        // Five projections rather than four, so it must exceed 4h^2.
        assert!(retention_params(1024, Some(0)) > 4 * 1024 * 1024);
    }
}
