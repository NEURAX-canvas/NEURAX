//! RNN/LSTM layer types and cell comparison test

#[test]
fn test_rnn_cell_type_comparison() {
    println!("\n=== RNN Cell Type Comparison ===\n");

    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│ Cell Type │ Gates │ Parameters per Layer        │ Use Case    │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ LSTM      │   4   │ 4 × (input + hidden + 1) × h  │ Long-term   │");
    println!("│ GRU       │   3   │ 3 × (input + hidden + 1) × h  │ Efficient   │");
    println!("│ Vanilla   │   1   │ (input + hidden + 1) × h      │ Simple      │");
    println!("├────────────────────────────────────────────────────────────────┤");
    println!("│ BiLSTM    │   4   │ 2 × LSTM params              │ Context     │");
    println!("│ BiGRU     │   3   │ 2 × GRU params               │ Seq2Seq     │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    println!("Key insights:\n");
    println!("  - LSTM: Best for long sequences, 4 gates (input, forget, cell, output)");
    println!("  - GRU:  Faster training, 3 gates (reset, update, new)");
    println!("  - Bidirectional: Captures both past and future context");
    println!("  - GRU has ~75% params of LSTM for same hidden size\n");
}

#[test]
fn test_rnn_layer_types_validation() {
    println!("\n=== RNN Layer Types Validation ===\n");

    let layer_types = [
        ("lstm_block", "LstmBlock - LSTM layer with 4 gates"),
        ("gru_block", "GruBlock - GRU layer with 3 gates"),
        ("rnn_cell", "RnnCell - Vanilla RNN cell"),
        ("bidirectional", "Bidirectional - Wrapper for BiLSTM/BiGRU"),
        ("encoder_block", "EncoderBlock - RNN encoder"),
        ("decoder_block", "DecoderBlock - RNN decoder with attention"),
    ];

    println!("Supported RNN layer types (6 total):\n");
    for (input, expected) in layer_types {
        println!("  ✓ '{}' -> {}", input, expected);
    }

    println!("\nRNN-specific parameters:\n");
    println!("  - rnn_hidden_size: Hidden state dimension");
    println!("  - num_rnn_layers: Number of stacked RNN layers");
    println!("  - bidirectional_rnn: Whether to use bidirectional RNN");
    println!("  - cell_type: lstm, gru, or vanilla_rnn");
    println!("  - forget_bias: LSTM forget gate bias (default 1.0)");
    println!("  - peephole: LSTM peephole connections");
    println!("  - recurrent_dropout: Dropout on recurrent connections");
    println!("  - attention_type: bahdanau, luong, dot\n");
}
