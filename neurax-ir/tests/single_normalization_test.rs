//! Test compilation of a single normalization block
//! Shows detailed absorption results for normalization layers (LayerNorm, RMSNorm)

#[test]
fn test_norm_types_comparison() {
    println!("\n=== Normalization Types Comparison ===\n");

    let h = 4096f64;

    println!("┌───────────────────┬───────────────┬───────────────────────┐");
    println!("│ Normalization     │ Parameters    │ Formula               │");
    println!("├───────────────────┼───────────────┼───────────────────────┤");
    println!(
        "│ RMSNorm           │ {:>10.0}    │ γ * x / √(mean(x²)+ε) │",
        h
    );
    println!(
        "│ LayerNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │",
        h * 2.0
    );
    println!(
        "│ BatchNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │",
        h * 2.0
    );
    println!(
        "│ GroupNorm         │ {:>10.0}    │ γ * (x-μ)/σ + β       │",
        h * 2.0
    );
    println!("└───────────────────┴───────────────┴───────────────────────┘\n");

    println!("RMSNorm (used in LLaMA):");
    println!("  - No bias term (β)");
    println!("  - No mean subtraction");
    println!("  - Simpler computation: x * γ / √(mean(x²) + ε)");
    println!("  - Parameters: {} (weight only)\n", h as u64);

    println!("LayerNorm (used in BERT/GPT):");
    println!("  - Has both weight (γ) and bias (β)");
    println!("  - Full normalization: (x - μ) / σ");
    println!("  - Parameters: {} (weight + bias)\n", (h * 2.0) as u64);
}
