//! Regression test for a real bug: `num_rnn_layers` was already parsed from
//! JSON into `LayerParams` and read nowhere — a node stating a 4-layer LSTM
//! was costed as a single cell. A companion bug in the frontend compiler
//! (`neurax-ui/src/utils/neuraxCompiler.ts`) meant `rnn_hidden_size` was
//! never sent at all, so the recurrent state size always fell back to a
//! hardcoded 512 regardless of what a template configured — fixed
//! separately in that file (`fixRnnParams`), covered by
//! `neuraxCompiler.otherFamilyShapes.test.ts`.
//!
//! This test covers the Rust side: a stacked LSTM must cost more than a
//! single-layer one when both otherwise state the same hidden/input size.

#[test]
fn a_stacked_lstm_costs_more_than_a_single_layer_one() {
    fn params_for(num_rnn_layers: Option<u64>) -> u64 {
        let extra = match num_rnn_layers {
            Some(n) => format!(r#","num_rnn_layers": {n}"#),
            None => String::new(),
        };
        let json = format!(
            r#"{{
                "schema_version": "1.0",
                "model": {{
                    "name": "rnn-stack-test",
                    "type": "rnn",
                    "global_params": {{ "num_layers": 1, "sequence_length": 64, "hidden_size": 256 }},
                    "layers": [
                        {{"id": "emb", "layer_type": "embedding", "params": {{"vocab_size": 1000, "embedding_dim": 256}}}},
                        {{"id": "rnn", "layer_type": "lstm_block", "params": {{"rnn_hidden_size": 256, "hidden_size": 256{extra}}}}}
                    ]
                }},
                "training": {{"batch_size": 1, "max_steps": 100}},
                "hardware": {{"gpus": [{{"name": "A100-SXM", "count": 1}}]}}
            }}"#
        );
        let config = neurax_parser::parse_model_config(&json).unwrap();
        neurax_ir::architecture::scaled_total_parameters(&config)
    }

    let single_layer = params_for(None);
    let four_layers = params_for(Some(4));

    assert!(
        four_layers > single_layer,
        "a 4-layer LSTM ({four_layers}) should cost more than a 1-layer one \
         ({single_layer}) at the same hidden size — num_rnn_layers is being \
         ignored again"
    );
    // Layer 1 takes the real input_size; layers 2-4 each take `hidden` as
    // their own input (they consume the previous layer's output) — so the
    // total is not simply 4x the single-layer figure when input_size !=
    // hidden, but it must still be well above 1x.
    assert!(
        four_layers as f64 > single_layer as f64 * 1.5,
        "four stacked layers ({four_layers}) should cost meaningfully more than \
         1.5x a single layer ({single_layer}), not just marginally more"
    );
}

/// A second real bug in the same formula, found and fixed after this file
/// was first written: for a stack that is *also* bidirectional, layers
/// 2..N consume the previous layer's *concatenated* forward+backward
/// output (`hidden * 2` wide) — real `nn.LSTM`'s own behavior — not just
/// `hidden`, which is what the fix above used unconditionally. No real
/// template combines `num_rnn_layers > 1` with `bidirectional_rnn: true`,
/// so this was previously unverified in practice rather than deliberately
/// simplified. Verified here by hand against `rnn::lstm_params()` directly
/// rather than trusting the pipeline's own output.
#[test]
fn a_bidirectional_stack_feeds_layer_two_the_concatenated_width() {
    let json = r#"{
        "schema_version": "1.0",
        "model": {
            "name": "bidir-stack-test",
            "type": "rnn",
            "global_params": { "num_layers": 1, "sequence_length": 64, "hidden_size": 256 },
            "layers": [
                {"id": "emb", "layer_type": "embedding", "params": {"vocab_size": 1000, "embedding_dim": 256}},
                {"id": "rnn", "layer_type": "lstm_block", "params": {
                    "rnn_hidden_size": 256, "hidden_size": 256,
                    "num_rnn_layers": 3, "bidirectional_rnn": true
                }}
            ]
        },
        "training": {"batch_size": 1, "max_steps": 100},
        "hardware": {"gpus": [{"name": "A100-SXM", "count": 1}]}
    }"#;
    let config = neurax_parser::parse_model_config(json).unwrap();
    let measured = neurax_ir::architecture::scaled_total_parameters(&config);

    let hidden = 256usize;
    let first = neurax_formulas::rnn::lstm_params(hidden, 256, true);
    let rest_per_layer = neurax_formulas::rnn::lstm_params(hidden, hidden * 2, true);
    let per_direction_total = first + rest_per_layer * 2; // 2 remaining layers
    let expected_rnn = per_direction_total * 2; // bidirectional
    let embedding_params = 1000u64 * 256;
    let expected = expected_rnn + embedding_params;

    assert_eq!(
        measured, expected,
        "a 3-layer bidirectional LSTM should feed layers 2-3 the concatenated \
         (hidden*2)-wide input, not plain hidden — got {measured}, expected {expected}"
    );
}
