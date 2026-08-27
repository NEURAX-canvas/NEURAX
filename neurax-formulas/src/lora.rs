//! LoRA / DoRA formulas — low-rank adaptation (Hu et al., 2021) and its
//! weight-decomposed variant (Liu et al., 2024).
//!
//! The whole point of LoRA is that adapting a `in × out` layer costs
//! `rank × (in + out)` instead of `in × out` — for `rank` in the tens and
//! `in`/`out` in the thousands, that's routinely a 100x+ reduction. Costing
//! a `lora_linear` block as a full dense layer (its previous behavior, from
//! falling through to `LayerType::Dense` on the wire) measured the exact
//! opposite of what the technique is for.

/// LoRA parameter count: a `rank × in` down-projection (A) and a
/// `out × rank` up-projection (B) — `ΔW = B·A`, never a full `in × out`
/// matrix materialized as weights.
pub fn lora_params(in_features: usize, out_features: usize, rank: usize) -> u64 {
    (rank * in_features + rank * out_features) as u64
}

/// DoRA parameter count: LoRA's two matrices plus a learned per-output-
/// channel magnitude vector (the decomposition's "magnitude" term,
/// separate from LoRA's "direction" update).
pub fn dora_params(in_features: usize, out_features: usize, rank: usize) -> u64 {
    lora_params(in_features, out_features, rank) + out_features as u64
}

/// LoRA FLOPs for a batch of `num_tokens`: two small matmuls (down-project
/// to `rank`, up-project back to `out_features`) instead of one large one.
pub fn lora_flops(num_tokens: usize, in_features: usize, out_features: usize, rank: usize) -> f64 {
    let down = 2.0 * num_tokens as f64 * in_features as f64 * rank as f64;
    let up = 2.0 * num_tokens as f64 * rank as f64 * out_features as f64;
    down + up
}

/// DoRA FLOPs: LoRA's two matmuls plus a per-token, per-output-channel
/// scale by the magnitude vector.
pub fn dora_flops(num_tokens: usize, in_features: usize, out_features: usize, rank: usize) -> f64 {
    let magnitude_scale = num_tokens as f64 * out_features as f64;
    lora_flops(num_tokens, in_features, out_features, rank) + magnitude_scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_is_far_cheaper_than_the_dense_layer_it_adapts() {
        let (in_f, out_f, rank) = (4096, 4096, 16);
        let lora = lora_params(in_f, out_f, rank);
        let dense_equivalent = (in_f * out_f) as u64;
        assert!(lora < dense_equivalent / 100, "LoRA should be well under 1% of the dense layer's params, got {lora} vs {dense_equivalent}");
    }

    #[test]
    fn test_dora_costs_more_than_lora_by_exactly_the_magnitude_vector() {
        let (in_f, out_f, rank) = (768, 768, 8);
        let lora = lora_params(in_f, out_f, rank);
        let dora = dora_params(in_f, out_f, rank);
        assert_eq!(dora - lora, out_f as u64);
    }

    #[test]
    fn test_lora_flops_scale_with_rank_not_the_full_dimension() {
        let low_rank = lora_flops(1, 4096, 4096, 8);
        let high_rank = lora_flops(1, 4096, 4096, 64);
        assert!(high_rank > low_rank);
        // Still far cheaper than a full dense forward (2 * in * out).
        let dense_flops = 2.0 * 4096.0 * 4096.0;
        assert!(high_rank < dense_flops / 10.0);
    }
}
