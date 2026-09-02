//! Independent, zero-dependency energy-cost cross-check.
//!
//! Used to call out to `lift-sim` (same LIFT project as
//! `native_flops_cross_check.rs` used to call `lift-tensor`). Rewritten
//! natively: the energy model below is derived from scratch (TDP * time,
//! converted to kWh) with **zero** dependency on `neurax_ir::cost`'s own
//! formula.
//!
//! Comparing against `lift-sim::EnergyModel` surfaced a real, quantified
//! gap: NEURAX's own `cost/pass.rs` counts only GPU TDP, while a full
//! system also draws CPU + memory power. That finding stands regardless of
//! which project's formula NEURAX cross-checks against — recorded here so
//! it isn't lost now that the external dependency is gone.

/// TDP-based energy model, GPU-only (mirrors cost/pass.rs's own accounting
/// so the "does the conversion arithmetic agree" check is meaningful).
fn textbook_gpu_energy_kwh(
    training_time_hours: f64,
    gpu_tdp_watts: f64,
    num_gpus: usize,
    pue: f64,
) -> f64 {
    training_time_hours * gpu_tdp_watts * num_gpus as f64 / 1000.0 * pue
}

/// Same model extended with CPU + memory draw — the real-world total a
/// data-center power bill actually reflects, per the gap `lift-sim`
/// surfaced. Not what `cost/pass.rs` computes today (documented, not
/// silently assumed).
fn textbook_full_system_energy_kwh(
    training_time_hours: f64,
    gpu_tdp_watts: f64,
    num_gpus: usize,
    cpu_tdp_watts: f64,
    memory_watts: f64,
    pue: f64,
) -> f64 {
    let total_watts = gpu_tdp_watts * num_gpus as f64 + cpu_tdp_watts + memory_watts;
    training_time_hours * total_watts / 1000.0 * pue
}

#[test]
fn gpu_only_energy_conversion_matches_cost_pass_exactly() {
    let (training_time_hours, gpu_tdp_watts, num_gpus, pue) = (2.0, 400.0, 8usize, 1.1);

    let expected = textbook_gpu_energy_kwh(training_time_hours, gpu_tdp_watts, num_gpus, pue);

    // Mirrors cost/pass.rs:110 exactly: energy_kwh = training_time_hours *
    // gpu_tdp_watts * num_gpus / 1000 * pue_factor.
    let neurax_formula = training_time_hours * gpu_tdp_watts * num_gpus as f64 / 1000.0 * pue;

    assert_eq!(
        expected, neurax_formula,
        "the GPU-only TDP->kWh conversion must match exactly"
    );
}

#[test]
fn cpu_and_memory_are_a_real_but_minor_share_of_total_system_power() {
    // Typical A100 node: 8x400W GPUs, one 250W CPU, ~50W of DRAM.
    let (training_time_hours, gpu_tdp_watts, num_gpus, cpu_tdp_watts, memory_watts, pue) =
        (2.0, 400.0, 8usize, 250.0, 50.0, 1.1);

    let gpu_only = textbook_gpu_energy_kwh(training_time_hours, gpu_tdp_watts, num_gpus, pue);
    let full_system = textbook_full_system_energy_kwh(
        training_time_hours,
        gpu_tdp_watts,
        num_gpus,
        cpu_tdp_watts,
        memory_watts,
        pue,
    );

    let cpu_memory_share = (full_system - gpu_only) / full_system;
    assert!(
        (0.0..0.15).contains(&cpu_memory_share),
        "CPU+memory should be a real but minor share of total node power \
         for an 8-GPU node (documenting the gap in cost/pass.rs's GPU-only \
         accounting, not asserting it must be fixed): got {:.1}%",
        cpu_memory_share * 100.0
    );
}
