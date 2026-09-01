//! Real per-family shape inference (Phase 1 of the representation-mère plan).
//!
//! Before this, `TensorPass` gave every layer with no declared shape the
//! same `[batch, seq, hidden]` placeholder regardless of type — a Conv2D
//! layer was treated identically to an attention layer. These fixtures send
//! no shapes at all, only real `connections`, and assert the exact numbers
//! already verified correct in `neurax-ui/src/utils/neuraxCompiler.cnnShapes.test.ts`
//! and `neuraxCompiler.otherFamilyShapes.test.ts` — a cross-check that the
//! compiler now agrees with what the frontend already proved right.

use neurax_core::analyze_json;
use neurax_ir::tensor::Dim;

fn output_shape(result: &neurax_core::AnalysisResult, layer_id: &str) -> Vec<usize> {
    let tensors = &result.tensor.layer_tensors[layer_id];
    let output_id = &tensors.outputs[0];
    result.tensor.tensors[output_id]
        .shape
        .0
        .iter()
        .map(|d| match d {
            Dim::Known(n) => *n,
            _ => 0,
        })
        .collect()
}

const CNN_CHAIN_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "CNN-Chain",
        "type": "cnn",
        "layers": [
            {"id": "conv1", "layer_type": "conv", "params": {"in_channels": 3, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "bn1", "layer_type": "batch_norm", "params": {}},
            {"id": "pool1", "layer_type": "pooling", "params": {"kernel_size": 2, "stride": 2}},
            {"id": "conv2", "layer_type": "conv", "params": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1}},
            {"id": "gpool", "layer_type": "pooling", "params": {"pool_type": "global"}},
            {"id": "fc", "layer_type": "dense", "params": {"out_features": 128}},
            {"id": "head", "layer_type": "dense", "params": {"out_features": 10}}
        ],
        "connections": [
            {"from": "conv1", "to": "bn1"}, {"from": "bn1", "to": "pool1"},
            {"from": "pool1", "to": "conv2"}, {"from": "conv2", "to": "gpool"},
            {"from": "gpool", "to": "fc"}, {"from": "fc", "to": "head"}
        ]
    },
    "training": {"batch_size": 4, "precision": "fp32"},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 1}]},
    "data": {"dtype": "fp32", "image_channels": 3, "image_height": 32, "image_width": 32}
}
"#;

#[test]
fn cnn_chain_matches_the_frontend_engine_exactly() {
    let result = analyze_json(CNN_CHAIN_JSON).expect("analysis should succeed");

    // conv1: 3->16 channels, 3x3/stride1/pad1 keeps spatial size at 32.
    assert_eq!(output_shape(&result, "conv1"), vec![4, 16, 32, 32]);
    // bn1: channel-preserving passthrough.
    assert_eq!(output_shape(&result, "bn1"), vec![4, 16, 32, 32]);
    // pool1: 2x2/stride2 halves spatial size, channels unchanged.
    assert_eq!(output_shape(&result, "pool1"), vec![4, 16, 16, 16]);
    // conv2 picks up pool1's actual output — 16 in, 32 out, still 16x16.
    assert_eq!(output_shape(&result, "conv2"), vec![4, 32, 16, 16]);
    // global pool collapses spatial dims to 1x1, channels unchanged.
    assert_eq!(output_shape(&result, "gpool"), vec![4, 32, 1, 1]);
    // dense implicitly flattens (32*1*1=32) then projects to out_features.
    assert_eq!(output_shape(&result, "fc"), vec![4, 128]);
    assert_eq!(output_shape(&result, "head"), vec![4, 10]);

    // Nothing anywhere in the chain is 0 — the original bug's exact symptom.
    for id in ["conv1", "bn1", "pool1", "conv2", "gpool", "fc", "head"] {
        assert!(
            output_shape(&result, id).iter().all(|&d| d > 0),
            "{id} has a 0 dimension"
        );
    }
}

const SEQUENCE_HIDDEN_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Sequence-Hidden-Thread",
        "type": "transformer",
        "layers": [
            {"id": "emb", "layer_type": "embedding", "params": {"embedding_dim": 512}},
            {"id": "attn", "layer_type": "attention", "params": {}},
            {"id": "mlp", "layer_type": "mlp", "params": {}}
        ],
        "connections": [{"from": "emb", "to": "attn"}, {"from": "attn", "to": "mlp"}]
    },
    "training": {"batch_size": 8, "precision": "fp16"},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 1}]},
    "data": {"dtype": "fp16"}
}
"#;

#[test]
fn sequence_hidden_dim_threads_through_nodes_that_dont_restate_it() {
    let result = analyze_json(SEQUENCE_HIDDEN_JSON).expect("analysis should succeed");

    // emb: [batch, seq] -> [batch, seq, 512] from its own embedding_dim.
    assert_eq!(output_shape(&result, "emb")[0], 8);
    assert_eq!(*output_shape(&result, "emb").last().unwrap(), 512);
    // attn/mlp state no hidden_size of their own — they carry forward what
    // they received (512), not a placeholder default (this is the exact bug
    // class reported live: a downstream layer disagreeing with the one
    // right before it because nothing propagated the real value).
    assert_eq!(output_shape(&result, "attn"), output_shape(&result, "emb"));
    assert_eq!(output_shape(&result, "mlp"), output_shape(&result, "attn"));
}

const GNN_CHAIN_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "GNN-Chain",
        "type": "gnn",
        "layers": [
            {"id": "gcn1", "layer_type": "graph_conv", "params": {"in_features": 1433, "out_features": 64}},
            {"id": "gcn2", "layer_type": "graph_conv", "params": {"out_features": 7}}
        ],
        "connections": [{"from": "gcn1", "to": "gcn2"}],
        "global_params": {"extra": {"num_nodes": 2708}}
    },
    "training": {"batch_size": 1, "precision": "fp32"},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 1}]},
    "data": {"dtype": "fp32"}
}
"#;

#[test]
fn gnn_uses_num_nodes_and_feature_width_not_the_batch_seq_hidden_placeholder() {
    let result = analyze_json(GNN_CHAIN_JSON).expect("analysis should succeed");

    // gcn1: [num_nodes, in_features] -> [num_nodes, out_features] = [2708, 64],
    // not the generic [batch, seq, hidden] = [1, 512, 512] every GNN layer
    // got before this module covered the Graph family.
    assert_eq!(output_shape(&result, "gcn1"), vec![2708, 64]);
    // gcn2 doesn't restate in_features — it must carry gcn1's real node
    // count and feature width forward, not fall back to a placeholder.
    assert_eq!(output_shape(&result, "gcn2"), vec![2708, 7]);
}

const NO_CONNECTIONS_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Positional-Fallback",
        "type": "transformer",
        "layers": [
            {"id": "attn_0", "layer_type": "attention", "input_shape": [2, 128, 256], "output_shape": [2, 128, 256], "params": {}},
            {"id": "mlp_0", "layer_type": "mlp", "input_shape": [2, 128, 256], "output_shape": [2, 128, 256], "params": {}}
        ]
    },
    "training": {"batch_size": 2, "precision": "fp16"},
    "hardware": {"gpus": [{"name": "A100-80GB", "count": 1}]},
    "data": {"dtype": "fp16"}
}
"#;

#[test]
fn a_client_that_still_declares_its_own_shapes_and_sends_no_connections_is_unaffected() {
    let result = analyze_json(NO_CONNECTIONS_JSON).expect("analysis should succeed");
    assert_eq!(output_shape(&result, "attn_0"), vec![2, 128, 256]);
    assert_eq!(output_shape(&result, "mlp_0"), vec![2, 128, 256]);
}
