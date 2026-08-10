//! NEURAX-hardware-db examples.
//!
//! Look up GPU specs and run roofline analysis on real hardware.

use neurax_hardware_db::HardwareDatabase;

fn main() {
    println!("=== NEURAX hardware database ===\n");

    let db = HardwareDatabase::new();

    // List what's available
    println!("GPUs in database: {}\n", db.list_gpus().len());

    // Roofline on an H100
    let h100 = db.get_gpu("H100-SXM").expect("H100-SXM in database");
    println!("-- H100-SXM --");
    println!(
        "  VRAM: {} GB | fp16: {:.0} TFLOPS | fp8: {:.0} TFLOPS | TDP: {} W",
        h100.memory_gb, h100.tflops_fp16, h100.tflops_fp8, h100.tdp_watts
    );

    let ridge = h100.ridge_point("fp16");
    println!("  ridge point (fp16): {:.1} FLOPs/byte", ridge);

    // Compare a training run: LLaMA-2-70B (3.38e16 FLOPs, ~700 GB bf16)
    let flops = 3.38e16;
    let weight_bytes = 140u64 * 1024 * 1024 * 1024; // 140 GB bf16 weights
    println!("\n-- LLaMA-2-70B on 8x H100 --");
    println!(
        "  weights fit per GPU ({} GB needed / {} GB): {}",
        weight_bytes / (1024 * 1024 * 1024),
        h100.memory_gb,
        h100.fits_in_memory(weight_bytes)
    );
    println!(
        "  compute time (1 GPU, fp16): {:.0} ms",
        h100.compute_time_ms(flops, "fp16")
    );

    // Compare a cheap GPU for the same job
    let rtx = db.get_gpu("RTX4090").expect("RTX4090 in database");
    println!("\n-- Same job on RTX 4090 --");
    println!(
        "  fp16: {:.0} TFLOPS | VRAM: {} GB | fits: {}",
        rtx.tflops_fp16,
        rtx.memory_gb,
        rtx.fits_in_memory(weight_bytes)
    );
}
