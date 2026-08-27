//! Mixture of Experts (MoE) formulas

/// Compute FLOPs for MoE layer
///
/// # Arguments
/// * `batch` - Batch size
/// * `seq_len` - Sequence length
/// * `hidden_size` - Hidden dimension
/// * `intermediate_size` - Expert intermediate dimension
/// * `num_experts` - Total number of experts
/// * `top_k` - Number of active experts per token
/// * `expert_flops` - FLOPs per expert (typically MLP FLOPs)
pub fn moe_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    expert_flops: f64,
) -> f64 {
    // Router: compute scores for all experts
    // [B, S, H] × [H, num_experts] → [B, S, num_experts]
    let router_flops =
        2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * num_experts as f64;

    // Softmax over expert scores
    let softmax_flops = 5.0 * batch as f64 * seq_len as f64 * num_experts as f64;

    // Top-k selection (approximate)
    let topk_flops = batch as f64 * seq_len as f64 * num_experts as f64 * top_k as f64;

    // Expert computation: only top_k experts per token
    // Each token routes to top_k experts
    let expert_total = batch as f64 * seq_len as f64 * top_k as f64 * expert_flops;

    router_flops + softmax_flops + topk_flops + expert_total
}

/// Compute parameters for MoE layer
pub fn moe_params(
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    expert_params: u64,
) -> u64 {
    let _ = intermediate_size; // Reserved for future use
                               // Router parameters
    let router_params = hidden_size * num_experts;

    // Expert parameters
    let experts_total = num_experts as u64 * expert_params;

    (router_params as u64) + experts_total
}

/// Compute parameters for MoE layer with shared experts (DeepSeek-V3 style)
pub fn moe_params_with_shared(
    hidden_size: usize,
    intermediate_size: usize,
    num_experts: usize,
    shared_experts: usize,
    expert_params: u64,
) -> u64 {
    let _ = intermediate_size; // Reserved for future use
                               // Router parameters
    let router_params = hidden_size * num_experts;

    // Routed experts parameters
    let routed_experts = num_experts as u64 * expert_params;

    // Shared experts parameters (always active, not routed)
    let shared_expert_params = shared_experts as u64 * expert_params;

    (router_params as u64) + routed_experts + shared_expert_params
}

/// Parameters actually touched per token for an MoE layer — the router
/// (scores every expert, so always active) plus only the `top_k` routed
/// experts a token is actually sent to, plus any shared experts (always
/// active, not routed). Mirrors `moe_params_with_shared`'s structure but
/// with the routed term scaled by `top_k` instead of `num_experts` — the
/// same distinction `moe_flops`'s expert-compute term already makes
/// correctly, which this had not.
pub fn moe_active_params_with_shared(
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    shared_experts: usize,
    expert_params: u64,
) -> u64 {
    let router_params = hidden_size * num_experts;
    let routed_experts = top_k as u64 * expert_params;
    let shared_expert_params = shared_experts as u64 * expert_params;

    (router_params as u64) + routed_experts + shared_expert_params
}

/// Compute FLOPs for sparse MoE (with load balancing)
pub fn sparse_moe_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_experts: usize,
    top_k: usize,
    expert_flops: f64,
    capacity_factor: f64,
) -> f64 {
    let _ = capacity_factor; // Reserved for future capacity-aware routing
                             // Standard MoE FLOPs
    let base_flops = moe_flops(
        batch,
        seq_len,
        hidden_size,
        num_experts,
        top_k,
        expert_flops,
    );

    // Load balancing auxiliary loss
    let aux_loss_flops = 2.0 * batch as f64 * seq_len as f64 * num_experts as f64;

    base_flops + aux_loss_flops
}

/// Estimate expert utilization in MoE
pub fn moe_expert_utilization(
    num_experts: usize,
    top_k: usize,
    batch: usize,
    seq_len: usize,
) -> f64 {
    let _ = (batch, seq_len); // Reserved for load-aware utilization calculation
                              // Ideal: each expert gets (batch * seq_len * top_k / num_experts) tokens
                              // Real utilization depends on routing, but we estimate average
    top_k as f64 / num_experts as f64
}

/// Compute FLOPs for MoE Router only
pub fn moe_router_flops(
    batch: usize,
    seq_len: usize,
    hidden_size: usize,
    num_experts: usize,
) -> f64 {
    // Router projection: [B, S, H] × [H, num_experts]
    let router_proj = 2.0 * batch as f64 * seq_len as f64 * hidden_size as f64 * num_experts as f64;

    // Softmax over expert scores
    let softmax = 5.0 * batch as f64 * seq_len as f64 * num_experts as f64;

    router_proj + softmax
}

/// Compute FLOPs for MoE Expert Group (single expert computation)
pub fn moe_expert_flops(
    num_tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    activation: &str,
) -> f64 {
    // Expert is typically an MLP: up projection + activation + down projection
    crate::mlp::mlp_flops(1, num_tokens, hidden_size, intermediate_size, activation)
}

/// Compute parameters for MoE Router
pub fn moe_router_params(hidden_size: usize, num_experts: usize) -> u64 {
    (hidden_size * num_experts) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mlp::mlp_flops;

    #[test]
    fn test_moe_flops() {
        // Mixtral-style: 8 experts, top-2
        let expert_flops = mlp_flops(1, 4096, 4096, 14336, "silu");
        let flops = moe_flops(1, 4096, 4096, 8, 2, expert_flops);

        // Should be roughly 2 × expert_flops per token
        assert!(flops > 0.0);
    }

    #[test]
    fn test_moe_active_params_only_scales_the_routed_experts() {
        // 8 experts, top-2, no shared experts: router (hidden*num_experts,
        // always active) + 2 routed experts — not 8, and not the router
        // scaled down along with them.
        let hidden = 4096;
        let num_experts = 8;
        let top_k = 2;
        let expert_params = 1_000_000u64;

        let active = moe_active_params_with_shared(hidden, num_experts, top_k, 0, expert_params);
        let expected = (hidden * num_experts) as u64 + top_k as u64 * expert_params;
        assert_eq!(active, expected);
    }

    #[test]
    fn test_moe_active_params_counts_shared_experts_at_full_weight() {
        // DeepSeek-style: shared experts run on every token, unlike the
        // routed ones — a flat top_k/num_experts scaling of the whole
        // layer (the bug this function replaces) would have shrunk their
        // contribution too, which is wrong: they are never routed at all.
        let hidden = 2048;
        let num_experts = 64;
        let top_k = 6;
        let shared_experts = 2;
        let expert_params = 500_000u64;

        let active = moe_active_params_with_shared(
            hidden, num_experts, top_k, shared_experts, expert_params,
        );
        let router = (hidden * num_experts) as u64;
        let routed = top_k as u64 * expert_params;
        let shared = shared_experts as u64 * expert_params;
        assert_eq!(active, router + routed + shared);

        // The old (wrong) behavior for comparison: scaling everything,
        // including the always-active router and shared experts, by
        // top_k/num_experts would have under-counted this.
        let total = moe_params_with_shared(hidden, 0, num_experts, shared_experts, expert_params);
        let old_wrong = (total as f64 * top_k as f64 / num_experts as f64).round() as u64;
        assert!(active > old_wrong, "the fix must count more than the flat-scaling bug did");
    }
}
