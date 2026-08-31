//! Hybrid architecture layer combinations test

#[test]
fn test_hybrid_layer_combinations() {
    println!("\n=== Hybrid Layer Type Combinations ===\n");

    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ Hybrid Type        │ Layer Combinations                          │ Use  │");
    println!("├────────────────────────────────────────────────────────────────────────┤");
    println!("│ ViT                │ Conv (patch) + Attention + MLP             │ Vision │");
    println!("│ DiT                │ TimeEmbed + Attention + Conv (unpatch)     │ Gen   │");
    println!("│ LSTM+Attention     │ LstmBlock + Attention                       │ Seq   │");
    println!("│ MoE-Transformer    │ Attention + MoE                             │ LLM   │");
    println!("│ ConvNeXt+Attn      │ ConvnextBlock + Attention                   │ Vision │");
    println!("│ Whisper            │ Conv + Attention + CrossAttention           │ Audio │");
    println!("└────────────────────────────────────────────────────────────────────────┘\n");

    println!("Key insights:\n");
    println!("  - Hybrid models combine strengths of multiple architectures");
    println!("  - ViT: CNN for local features + Transformer for global context");
    println!("  - DiT: Transformer architecture for diffusion process");
    println!("  - MoE: Sparse activation reduces compute while maintaining capacity");
    println!("  - Cross-attention enables encoder-decoder architectures\n");
}
