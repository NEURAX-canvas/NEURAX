//! Test compilation of a single MLP block
//! Shows detailed absorption results for MLP layers (SwiGLU, GELU, ReLU variants)

#[test]
fn test_mlp_activation_comparison() {
    println!("\n=== MLP Activation Functions Comparison ===\n");

    let h = 4096f64;
    let i = 11008f64;

    println!("┌───────────────────┬───────────────┬───────────────────────────────┐");
    println!("│ Activation        │ Parameters    │ Formula                       │");
    println!("├───────────────────┼───────────────┼───────────────────────────────┤");
    println!(
        "│ ReLU/GeLU (std)   │ {:>10.0}  │ up(H→I) + down(I→H)          │",
        2.0 * h * i
    );
    println!(
        "│ SwiGLU (LLaMA)    │ {:>10.0}  │ gate + up + down (3×H×I)     │",
        3.0 * h * i
    );
    println!(
        "│ GeGLU (T5)        │ {:>10.0}  │ gate + up + down (3×H×I)     │",
        3.0 * h * i
    );
    println!(
        "│ ReGLU             │ {:>10.0}  │ gate + up + down (3×H×I)     │",
        3.0 * h * i
    );
    println!("└───────────────────┴───────────────┴───────────────────────────────┘\n");

    println!("Activation functions:\n");
    println!("  ReLU:    max(0, x)");
    println!("  GELU:    x × Φ(x)  (Gaussian Error Linear Unit)");
    println!("  SwiGLU:  Swish(x) × gate(x)  (Swish-Gated Linear Unit)");
    println!("  GeGLU:   GELU(x) × gate(x)   (GELU-Gated Linear Unit)");
    println!("\nGated variants (SwiGLU, GeGLU, ReGLU) have 50% more parameters");
    println!("but typically achieve better performance per parameter.\n");

    // Parameter overhead
    let overhead = (3.0 * h * i) - (2.0 * h * i);
    let overhead_pct = overhead / (2.0 * h * i) * 100.0;
    println!(
        "SwiGLU parameter overhead: {:.0}% (+{:.0} params)",
        overhead_pct, overhead
    );
}

#[test]
fn test_mlp_intermediate_ratio() {
    println!("\n=== MLP Intermediate Size Ratios ===\n");

    let h = 4096f64;

    println!("┌───────────────────┬───────────────┬───────────────┬─────────────┐");
    println!("│ Model             │ Hidden (H)    │ Inter (I)     │ Ratio (I/H) │");
    println!("├───────────────────┼───────────────┼───────────────┼─────────────┤");
    println!("│ LLaMA-7B          │     4096      │    11008      │    2.69     │");
    println!("│ LLaMA-70B         │    8192       │    28672      │    3.50     │");
    println!("│ GPT-3 (175B)      │   12288       │    49152      │    4.00     │");
    println!("│ Mistral-7B        │    4096       │    14336      │    3.50     │");
    println!("│ Mixtral-8x7B      │    4096       │    14336      │    3.50     │");
    println!("└───────────────────┴───────────────┴───────────────┴─────────────┘\n");

    // Calculate MLP params for different ratios
    println!("MLP params for H=4096 with different ratios:\n");
    for ratio in [2.0, 2.69, 3.0, 3.5, 4.0] {
        let i = h * ratio;
        let swiglu_params = 3.0 * h * i;
        println!(
            "  Ratio {:.2}: I={:.0}, SwiGLU params={:.2}M",
            ratio,
            i,
            swiglu_params / 1e6
        );
    }
}
