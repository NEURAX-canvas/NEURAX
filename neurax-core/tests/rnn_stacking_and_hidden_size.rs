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
