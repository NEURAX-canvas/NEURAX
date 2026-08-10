//! NEURAX-formulas examples.
//!
//! Compute analytical FLOPs / parameters / memory for common layer types.

use neurax_formulas::{
    attention_flops, dtype_bytes, gated_mlp_flops, gated_mlp_params, mlp_flops, mlp_params,
};

fn main() {
    println!("=== NEURAX analytical formulas ===\n");

    let batch = 32usize;
    let seq_len = 2048usize;
    let hidden = 4096usize;
    let intermediate = 11008usize;

    // -- MLP (standard GELU vs gated SwiGLU) --
    let standard = mlp_flops(batch, seq_len, hidden, intermediate, "gelu");
    let gated = gated_mlp_flops(batch, seq_len, hidden, intermediate, "silu");
    let standard_params = mlp_params(hidden, intermediate, false);
    let swiglu_params = gated_mlp_params(hidden, intermediate, false);

    println!("-- MLP --");
    println!("standard (GELU): {:.2e} FLOPs / {} params", standard, standard_params);
    println!("gated (SwiGLU):   {:.2e} FLOPs / {} params (+50%)", gated, swiglu_params);

    // -- Attention --
    let num_heads = 32usize;
    let causal_flops = attention_flops(batch, seq_len, hidden, num_heads, true);
    let full_flops = attention_flops(batch, seq_len, hidden, num_heads, false);

    println!("\n-- Attention (H={}, heads={}) --", hidden, num_heads);
    println!("causal:  {:.2e} FLOPs", causal_flops);
    println!("full:    {:.2e} FLOPs", full_flops);

    // -- Dtypes --
    println!("\n-- Dtype sizes --");
    for dt in ["fp32", "fp16", "bf16", "fp8"] {
        println!("  {dt:>4}: {} bytes/element", dtype_bytes(dt));
    }
}
