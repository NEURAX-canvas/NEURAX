//! Graph Neural Network formulas

/// Compute FLOPs for Graph Convolutional Network (GCN) layer
///
/// # Arguments
/// * `num_nodes` - Number of nodes in the graph
/// * `in_features` - Input feature dimension
/// * `out_features` - Output feature dimension
/// * `num_edges` - Number of edges (for message passing)
pub fn gcn_flops(
    num_nodes: usize,
    in_features: usize,
    out_features: usize,
    num_edges: usize,
) -> f64 {
    // Linear transformation: [N, in] × [in, out] → [N, out]
    let linear_flops = 2.0 * num_nodes as f64 * in_features as f64 * out_features as f64;

    // Aggregation: sum over neighbors for each node
    // Each edge contributes one addition
    let agg_flops = num_edges as f64 * out_features as f64;

    // Normalization (degree-based)
    let norm_flops = 2.0 * num_nodes as f64 * out_features as f64;

    linear_flops + agg_flops + norm_flops
}

/// Compute FLOPs for Graph Attention Network (GAT) layer
pub fn gat_flops(
    num_nodes: usize,
    in_features: usize,
    out_features: usize,
    num_edges: usize,
    num_heads: usize,
) -> f64 {
    let head_dim = out_features / num_heads.max(1);

    // Per-head attention
    let per_head = {
        // Linear projections for Q, K (or just one for source)
        let proj = 2.0 * num_nodes as f64 * in_features as f64 * head_dim as f64;

        // Attention scores for each edge
        let attn = 2.0 * num_edges as f64 * head_dim as f64;

        // Softmax over neighbors
        let softmax = 5.0 * num_edges as f64;

        // Message aggregation
        let agg = num_edges as f64 * head_dim as f64;

        proj + attn + softmax + agg
    };

    per_head * num_heads as f64
}

/// Compute FLOPs for message passing neural network
pub fn mpnn_flops(
    num_nodes: usize,
    num_edges: usize,
    node_features: usize,
    edge_features: usize,
    message_dim: usize,
) -> f64 {
    // Message function: edge + source + target features → message
    let msg_flops = num_edges as f64
        * (2.0 * node_features as f64 + edge_features as f64)
        * message_dim as f64
        * 2.0;

    // Aggregation: sum messages per node
    let agg_flops = num_edges as f64 * message_dim as f64;

    // Update function: GRU or MLP
    let update_flops =
        2.0 * num_nodes as f64 * (node_features as f64 + message_dim as f64) * node_features as f64;

    msg_flops + agg_flops + update_flops
}

/// Compute parameters for GCN layer
pub fn gcn_params(in_features: usize, out_features: usize, bias: bool) -> u64 {
    let weight = in_features * out_features;
    let bias_params = if bias { out_features } else { 0 };
    (weight + bias_params) as u64
}

/// Compute parameters for GAT layer
pub fn gat_params(in_features: usize, out_features: usize, num_heads: usize, bias: bool) -> u64 {
    let head_dim = out_features / num_heads.max(1);
    let weight = in_features * num_heads * head_dim;
    let attn_src = num_heads * head_dim;
    let attn_dst = num_heads * head_dim;
    let bias_params = if bias { out_features } else { 0 };
    (weight + attn_src + attn_dst + bias_params) as u64
}

/// Compute parameters for a Relational GCN layer (Schlichtkrull et al.,
/// 2018) — a plain GCN has one shared weight matrix; RGCN has one *per
/// relation type*, since a knowledge-graph edge's meaning (and therefore
/// the transform it should apply) depends on which relation it is.
///
/// With `num_bases` given, relation weights are a shared basis decomposition
/// (`num_bases` shared `in×out` matrices, each relation a learned
/// combination of them) rather than `num_relations` full matrices — the
/// paper's regularization for graphs with many relation types, and the
/// difference between a knowledge-graph model with hundreds of relations
/// (FB15k: 1,345) costing plausibly vs. costing as if it had one.
pub fn rgcn_params(
    in_features: usize,
    out_features: usize,
    num_relations: usize,
    num_bases: Option<usize>,
    bias: bool,
) -> u64 {
    let relation_weights = match num_bases {
        Some(bases) if bases > 0 && bases < num_relations => {
            // `bases` shared in×out matrices, each relation a learned
            // per-basis combination coefficient.
            bases * in_features * out_features + num_relations * bases
        }
        _ => num_relations * in_features * out_features,
    };
    // Self-loop / root weight (W_0 in the paper) — every RGCN layer keeps
    // one, separate from the relation-specific transforms.
    let self_loop = in_features * out_features;
    let bias_params = if bias { out_features } else { 0 };
    (relation_weights + self_loop + bias_params) as u64
}

/// Compute FLOPs for a Relational GCN layer.
///
/// Every edge is transformed by its own relation's weight matrix (or basis
/// combination) rather than one shared matrix — `gcn_flops`'s per-edge
/// aggregation term still applies, but the linear transform now runs once
/// per relation (the model must be able to apply any of them), not once.
pub fn rgcn_flops(
    num_nodes: usize,
    in_features: usize,
    out_features: usize,
    num_edges: usize,
    num_relations: usize,
) -> f64 {
    let relations = num_relations.max(1) as f64;
    let linear_flops =
        2.0 * relations * num_nodes as f64 * in_features as f64 * out_features as f64;
    let self_loop_flops = 2.0 * num_nodes as f64 * in_features as f64 * out_features as f64;
    let agg_flops = num_edges as f64 * out_features as f64;
    let norm_flops = 2.0 * num_nodes as f64 * out_features as f64;
    linear_flops + self_loop_flops + agg_flops + norm_flops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcn_flops() {
        let flops = gcn_flops(1000, 128, 256, 5000);
        assert!(flops > 0.0);
    }

    #[test]
    fn test_gcn_params() {
        let params = gcn_params(128, 256, true);
        assert_eq!(params, 128 * 256 + 256);
    }

    #[test]
    fn test_rgcn_without_bases_scales_with_num_relations() {
        // No basis decomposition: one full in×out matrix per relation, plus
        // the self-loop weight — not the same cost as a plain single-relation
        // GCN, which is what routing rgcn_conv through gcn_params gave before.
        let params = rgcn_params(128, 256, 5, None, false);
        assert_eq!(params, (5 * 128 * 256 + 128 * 256) as u64);
    }

    #[test]
    fn test_rgcn_with_bases_is_cheaper_than_without_for_many_relations() {
        // FB15k-shaped: hundreds of relations. Basis decomposition should
        // cost far less than a full matrix per relation.
        let num_relations = 1345;
        let without_bases = rgcn_params(100, 100, num_relations, None, false);
        let with_bases = rgcn_params(100, 100, num_relations, Some(30), false);
        assert!(with_bases < without_bases);
    }

    #[test]
    fn test_rgcn_flops_scales_with_relations_not_flat_like_gcn() {
        let single_relation = rgcn_flops(1000, 128, 256, 5000, 1);
        let many_relations = rgcn_flops(1000, 128, 256, 5000, 8);
        assert!(many_relations > single_relation);
    }
}
