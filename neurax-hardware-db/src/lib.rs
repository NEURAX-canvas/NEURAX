//! # NEURAX Hardware Database
//!
//! **GPU / CPU / interconnect specifications with roofline analysis** — the
//! hardware knowledge base of the [NEURAX](https://github.com/rustnew/NEURAX)
//! analytical compiler for neural network architectures.
//!
//! Ships a built-in database of 20+ GPUs (A100, H100, V100, RTX 40-series, ...),
//! CPUs and interconnects, plus the analytical primitives to answer hardware
//! questions at design time: *"will this fit in VRAM?"*, *"is this compute-bound
//! or memory-bound?"*, *"how long will this take?"*.
//!
//! ## Quick example
//!
//! ```rust
//! use neurax_hardware_db::HardwareDatabase;
//!
//! let db = HardwareDatabase::new();
//! let h100 = db.get_gpu("H100-SXM").expect("H100 in database");
//!
//! // Roofline: FLOPs/byte at the compute-vs-memory boundary
//! let ridge = h100.ridge_point("fp16");
//! assert!(ridge > 0.0);
//!
//! // Budget check: 140 GB of bf16 weights on an 80 GB GPU?
//! assert!(!h100.fits_in_memory(140 * 1024 * 1024 * 1024));
//! ```
//!
//! ## Main types
//!
//! - [`HardwareDatabase`] — the built-in spec database
//! - [`GpuSpec`] — GPU spec + roofline methods (`ridge_point`, `fits_in_memory`,
//!   `compute_time_ms`, `memory_time_ms`, `tflops_for_precision`, ...)
//! - [`CpuSpec`], [`InterconnectSpec`] — CPU and interconnect specs
//!
//! ## Run the example
//!
//! ```bash
//! cargo run --example roofline
//! ```

mod cpu;
mod gpu;
mod interconnect;

pub use cpu::*;
pub use gpu::*;
pub use interconnect::*;

use ahash::AHashMap as HashMap;

/// Resolve a common GPU spelling to the exact key used in the database.
///
/// Accelerators ship in several board variants (`H100-SXM`, `H100-PCIe`) but
/// configs, papers and users almost always write the bare family name. Without
/// this mapping `get_gpu("H100")` missed, the pass fell back to a generic
/// 20-TFLOPS profile, and every latency, throughput and cost figure derived
/// from it was wrong by more than an order of magnitude.
///
/// Bare names resolve to the SXM board, which is the datacenter part these
/// configs mean in practice. The mapping is explicit rather than a prefix scan
/// so that resolution stays deterministic.
fn canonical_gpu_name(name: &str) -> Option<&'static str> {
    let normalized = name.trim().to_ascii_lowercase().replace(['_', ' '], "-");
    Some(match normalized.as_str() {
        "h100" | "h100-80gb" | "h100-sxm5" => "H100-SXM",
        "h100-pcie5" => "H100-PCIe",
        "a100" | "a100-80gb" | "a100-40gb" | "a100-sxm4" => "A100-SXM",
        "a100-pcie4" => "A100-PCIe",
        "h200" | "h200-sxm" => "H200",
        "gh200" | "gh200-superchip" => "GH200",
        "v100" | "v100-sxm2" | "v100-pcie" => "V100",
        "rtx-4090" | "4090" | "geforce-rtx-4090" => "RTX4090",
        "rtx-4080" | "4080" | "geforce-rtx-4080" => "RTX4080",
        "rtx-3090" | "3090" | "geforce-rtx-3090" => "RTX3090",
        "rtx-3080" | "3080" | "geforce-rtx-3080" => "RTX3080",
        "l40s" => "L40S",
        "l40" => "L40",
        "a30" => "A30",
        "a10g" | "a10" => "A10G",
        "a6000" | "rtx-a6000" => "A6000",
        "a5000" | "rtx-a5000" => "RTXA5000",
        "rtx-6000-ada" | "rtx6000-ada" => "RTX6000Ada",
        "t4" | "tesla-t4" => "T4",
        "k80" | "tesla-k80" => "K80",
        _ => return None,
    })
}

/// Global hardware database
pub struct HardwareDatabase {
    gpus: HashMap<String, GpuSpec>,
    cpus: HashMap<String, CpuSpec>,
    interconnects: HashMap<String, InterconnectSpec>,
}

impl HardwareDatabase {
    /// Create a new database with built-in specifications
    pub fn new() -> Self {
        let mut db = Self {
            gpus: HashMap::new(),
            cpus: HashMap::new(),
            interconnects: HashMap::new(),
        };

        // Add built-in GPU specs
        db.add_builtin_gpus();
        db.add_builtin_cpus();
        db.add_builtin_interconnects();

        db
    }

    fn add_builtin_gpus(&mut self) {
        // NVIDIA A100 SXM
        self.gpus.insert(
            "A100-SXM".to_string(),
            GpuSpec {
                name: "A100-SXM".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 80,
                memory_bandwidth_gbs: 2039.0,
                tflops_fp64: 19.5,
                tflops_fp32: 19.5,
                tflops_fp16: 312.0,
                tflops_bf16: 312.0,
                tflops_int8: 624.0,
                tflops_fp8: 0.0, // Ampere doesn't have native FP8
                tensor_cores: true,
                nvlink: true,
                nvlink_bandwidth_gbs: 600.0,
                tdp_watts: 400,
                launch_year: 2020,
                l2_cache_mb: Some(40.0),
                num_sms: Some(108),
            },
        );

        // NVIDIA A100 PCIe
        self.gpus.insert(
            "A100-PCIe".to_string(),
            GpuSpec {
                name: "A100-PCIe".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 40,
                memory_bandwidth_gbs: 1555.0,
                tflops_fp64: 19.5,
                tflops_fp32: 19.5,
                tflops_fp16: 312.0,
                tflops_bf16: 312.0,
                tflops_int8: 624.0,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 250,
                launch_year: 2020,
                l2_cache_mb: Some(40.0),
                num_sms: Some(108),
            },
        );

        // NVIDIA H100 SXM
        self.gpus.insert(
            "H100-SXM".to_string(),
            GpuSpec {
                name: "H100-SXM".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 80,
                memory_bandwidth_gbs: 3352.0,
                tflops_fp64: 34.0,
                tflops_fp32: 67.0,
                tflops_fp16: 989.0,
                tflops_bf16: 989.0,
                tflops_int8: 1979.0,
                tflops_fp8: 3958.0, // Hopper has native FP8
                tensor_cores: true,
                nvlink: true,
                nvlink_bandwidth_gbs: 900.0,
                tdp_watts: 700,
                launch_year: 2022,
                l2_cache_mb: Some(50.0),
                num_sms: Some(132),
            },
        );

        // NVIDIA H100 PCIe
        self.gpus.insert(
            "H100-PCIe".to_string(),
            GpuSpec {
                name: "H100-PCIe".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 80,
                memory_bandwidth_gbs: 2000.0,
                tflops_fp64: 26.0,
                tflops_fp32: 51.0,
                tflops_fp16: 756.0,
                tflops_bf16: 756.0,
                tflops_int8: 1513.0,
                tflops_fp8: 3026.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 350,
                launch_year: 2022,
                l2_cache_mb: Some(50.0),
                num_sms: Some(114),
            },
        );

        // NVIDIA RTX 4090
        self.gpus.insert(
            "RTX4090".to_string(),
            GpuSpec {
                name: "RTX4090".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 24,
                memory_bandwidth_gbs: 1008.0,
                tflops_fp64: 1.3,
                tflops_fp32: 82.6,
                tflops_fp16: 165.0,
                tflops_bf16: 165.0,
                tflops_int8: 330.0,
                tflops_fp8: 660.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 450,
                launch_year: 2022,
                l2_cache_mb: Some(72.0),
                num_sms: Some(128),
            },
        );

        // NVIDIA V100
        self.gpus.insert(
            "V100".to_string(),
            GpuSpec {
                name: "V100".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 32,
                memory_bandwidth_gbs: 900.0,
                tflops_fp64: 7.8,
                tflops_fp32: 15.7,
                tflops_fp16: 125.0,
                tflops_bf16: 0.0, // V100 doesn't support bf16
                tflops_int8: 250.0,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: true,
                nvlink_bandwidth_gbs: 300.0,
                tdp_watts: 300,
                launch_year: 2017,
                l2_cache_mb: Some(6.0),
                num_sms: Some(80),
            },
        );

        // NVIDIA A6000
        self.gpus.insert(
            "A6000".to_string(),
            GpuSpec {
                name: "A6000".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 48,
                memory_bandwidth_gbs: 768.0,
                tflops_fp64: 1.3,
                tflops_fp32: 38.7,
                tflops_fp16: 77.4,
                tflops_bf16: 77.4,
                tflops_int8: 154.8,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 300,
                launch_year: 2021,
                l2_cache_mb: Some(6.0),
                num_sms: Some(84),
            },
        );

        // ── Additional GPUs from impl_2.md spec (20 GPUs total) ──────────────

        // NVIDIA RTX 3090
        self.gpus.insert(
            "RTX3090".to_string(),
            GpuSpec {
                name: "RTX3090".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 24,
                memory_bandwidth_gbs: 936.0,
                tflops_fp64: 0.5,
                tflops_fp32: 35.6,
                tflops_fp16: 71.0,
                tflops_bf16: 71.0,
                tflops_int8: 142.0,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 350,
                launch_year: 2020,
                l2_cache_mb: Some(6.0),
                num_sms: Some(82),
            },
        );

        // NVIDIA RTX 3080
        self.gpus.insert(
            "RTX3080".to_string(),
            GpuSpec {
                name: "RTX3080".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 10,
                memory_bandwidth_gbs: 760.0,
                tflops_fp64: 0.3,
                tflops_fp32: 30.6,
                tflops_fp16: 61.0,
                tflops_bf16: 61.0,
                tflops_int8: 122.0,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 320,
                launch_year: 2020,
                l2_cache_mb: Some(4.0),
                num_sms: Some(68),
            },
        );

        // NVIDIA RTX 4080
        self.gpus.insert(
            "RTX4080".to_string(),
            GpuSpec {
                name: "RTX4080".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 16,
                memory_bandwidth_gbs: 717.0,
                tflops_fp64: 1.1,
                tflops_fp32: 48.7,
                tflops_fp16: 97.0,
                tflops_bf16: 97.0,
                tflops_int8: 194.0,
                tflops_fp8: 388.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 320,
                launch_year: 2022,
                l2_cache_mb: Some(64.0),
                num_sms: Some(76),
            },
        );

        // NVIDIA L40S
        self.gpus.insert(
            "L40S".to_string(),
            GpuSpec {
                name: "L40S".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 48,
                memory_bandwidth_gbs: 864.0,
                tflops_fp64: 1.5,
                tflops_fp32: 91.6,
                tflops_fp16: 362.0,
                tflops_bf16: 362.0,
                tflops_int8: 724.0,
                tflops_fp8: 1448.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 350,
                launch_year: 2023,
                l2_cache_mb: Some(96.0),
                num_sms: Some(118),
            },
        );

        // NVIDIA L40
        self.gpus.insert(
            "L40".to_string(),
            GpuSpec {
                name: "L40".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 48,
                memory_bandwidth_gbs: 864.0,
                tflops_fp64: 1.5,
                tflops_fp32: 91.6,
                tflops_fp16: 181.0,
                tflops_bf16: 181.0,
                tflops_int8: 362.0,
                tflops_fp8: 724.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 300,
                launch_year: 2022,
                l2_cache_mb: Some(96.0),
                num_sms: Some(118),
            },
        );

        // NVIDIA A30
        self.gpus.insert(
            "A30".to_string(),
            GpuSpec {
                name: "A30".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 24,
                memory_bandwidth_gbs: 933.0,
                tflops_fp64: 5.0,
                tflops_fp32: 10.3,
                tflops_fp16: 165.0,
                tflops_bf16: 165.0,
                tflops_int8: 330.0,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: true,
                nvlink_bandwidth_gbs: 200.0,
                tdp_watts: 165,
                launch_year: 2021,
                l2_cache_mb: Some(6.0),
                num_sms: Some(56),
            },
        );

        // NVIDIA A10G
        self.gpus.insert(
            "A10G".to_string(),
            GpuSpec {
                name: "A10G".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 24,
                memory_bandwidth_gbs: 600.0,
                tflops_fp64: 0.6,
                tflops_fp32: 19.5,
                tflops_fp16: 78.0,
                tflops_bf16: 78.0,
                tflops_int8: 156.0,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 150,
                launch_year: 2021,
                l2_cache_mb: Some(6.0),
                num_sms: Some(52),
            },
        );

        // NVIDIA T4
        self.gpus.insert(
            "T4".to_string(),
            GpuSpec {
                name: "T4".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 16,
                memory_bandwidth_gbs: 300.0,
                tflops_fp64: 0.3,
                tflops_fp32: 8.1,
                tflops_fp16: 65.0,
                tflops_bf16: 0.0,
                tflops_int8: 130.0,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 70,
                launch_year: 2018,
                l2_cache_mb: Some(6.0),
                num_sms: Some(40),
            },
        );

        // NVIDIA K80 (legacy)
        self.gpus.insert(
            "K80".to_string(),
            GpuSpec {
                name: "K80".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 24,
                memory_bandwidth_gbs: 480.0,
                tflops_fp64: 2.9,
                tflops_fp32: 5.8,
                tflops_fp16: 0.0,
                tflops_bf16: 0.0,
                tflops_int8: 0.0,
                tflops_fp8: 0.0,
                tensor_cores: false,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 300,
                launch_year: 2014,
                l2_cache_mb: None,
                num_sms: None,
            },
        );

        // NVIDIA H200 (H100 successor)
        self.gpus.insert(
            "H200".to_string(),
            GpuSpec {
                name: "H200".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 141,
                memory_bandwidth_gbs: 4800.0, // HBM3e
                tflops_fp64: 34.0,
                tflops_fp32: 67.0,
                tflops_fp16: 989.0,
                tflops_bf16: 989.0,
                tflops_int8: 1979.0,
                tflops_fp8: 3958.0,
                tensor_cores: true,
                nvlink: true,
                nvlink_bandwidth_gbs: 900.0,
                tdp_watts: 700,
                launch_year: 2024,
                l2_cache_mb: Some(50.0),
                num_sms: Some(132),
            },
        );

        // NVIDIA GH200 Grace Hopper
        self.gpus.insert(
            "GH200".to_string(),
            GpuSpec {
                name: "GH200".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 141, // HBM3e
                memory_bandwidth_gbs: 4800.0,
                tflops_fp64: 34.0,
                tflops_fp32: 67.0,
                tflops_fp16: 989.0,
                tflops_bf16: 989.0,
                tflops_int8: 1979.0,
                tflops_fp8: 3958.0,
                tensor_cores: true,
                nvlink: true,
                nvlink_bandwidth_gbs: 900.0,
                tdp_watts: 700,
                launch_year: 2024,
                l2_cache_mb: Some(50.0),
                num_sms: Some(132),
            },
        );

        // NVIDIA RTX 6000 Ada
        self.gpus.insert(
            "RTX6000Ada".to_string(),
            GpuSpec {
                name: "RTX6000Ada".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 48,
                memory_bandwidth_gbs: 960.0,
                tflops_fp64: 1.8,
                tflops_fp32: 91.1,
                tflops_fp16: 182.0,
                tflops_bf16: 182.0,
                tflops_int8: 364.0,
                tflops_fp8: 728.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 300,
                launch_year: 2022,
                l2_cache_mb: Some(96.0),
                num_sms: Some(142),
            },
        );

        // NVIDIA RTX A5000
        self.gpus.insert(
            "RTXA5000".to_string(),
            GpuSpec {
                name: "RTXA5000".to_string(),
                manufacturer: "NVIDIA".to_string(),
                memory_gb: 24,
                memory_bandwidth_gbs: 768.0,
                tflops_fp64: 0.5,
                tflops_fp32: 27.8,
                tflops_fp16: 55.6,
                tflops_bf16: 55.6,
                tflops_int8: 111.2,
                tflops_fp8: 0.0,
                tensor_cores: true,
                nvlink: false,
                nvlink_bandwidth_gbs: 0.0,
                tdp_watts: 230,
                launch_year: 2021,
                l2_cache_mb: Some(6.0),
                num_sms: Some(64),
            },
        );
    }

    fn add_builtin_cpus(&mut self) {
        // AMD EPYC 9654
        self.cpus.insert(
            "EPYC-9654".to_string(),
            CpuSpec {
                name: "EPYC-9654".to_string(),
                manufacturer: "AMD".to_string(),
                cores: 96,
                threads: 192,
                base_freq_ghz: 2.4,
                boost_freq_ghz: 3.7,
                tdp_watts: 360,
                memory_channels: 12,
                memory_bandwidth_gbs: 460.0,
            },
        );

        // Intel Xeon w9-3595X
        self.cpus.insert(
            "Xeon-w9-3595X".to_string(),
            CpuSpec {
                name: "Xeon-w9-3595X".to_string(),
                manufacturer: "Intel".to_string(),
                cores: 56,
                threads: 112,
                base_freq_ghz: 2.0,
                boost_freq_ghz: 4.8,
                tdp_watts: 350,
                memory_channels: 8,
                memory_bandwidth_gbs: 307.0,
            },
        );
    }

    fn add_builtin_interconnects(&mut self) {
        self.interconnects.insert(
            "NVLink3".to_string(),
            InterconnectSpec {
                name: "NVLink3".to_string(),
                bandwidth_gbs: 600.0,
                latency_ns: 10.0,
            },
        );

        self.interconnects.insert(
            "NVLink4".to_string(),
            InterconnectSpec {
                name: "NVLink4".to_string(),
                bandwidth_gbs: 900.0,
                latency_ns: 8.0,
            },
        );

        self.interconnects.insert(
            "PCIe4".to_string(),
            InterconnectSpec {
                name: "PCIe4".to_string(),
                bandwidth_gbs: 64.0,
                latency_ns: 1000.0,
            },
        );

        self.interconnects.insert(
            "PCIe5".to_string(),
            InterconnectSpec {
                name: "PCIe5".to_string(),
                bandwidth_gbs: 128.0,
                latency_ns: 500.0,
            },
        );

        self.interconnects.insert(
            "InfiniBand-NDR".to_string(),
            InterconnectSpec {
                name: "InfiniBand-NDR".to_string(),
                bandwidth_gbs: 400.0,
                latency_ns: 100.0,
            },
        );
    }

    /// Get GPU specification by name
    pub fn get_gpu(&self, name: &str) -> Option<&GpuSpec> {
        if let Some(spec) = self.gpus.get(name) {
            return Some(spec);
        }
        self.gpus.get(canonical_gpu_name(name)?)
    }

    /// Get GPU spec or a generic fallback
    pub fn get_gpu_or_fallback(&self, name: &str) -> GpuSpec {
        self.gpus
            .get(name)
            .cloned()
            .unwrap_or_else(GpuSpec::generic)
    }

    /// Get CPU specification by name
    pub fn get_cpu(&self, name: &str) -> Option<&CpuSpec> {
        self.cpus.get(name)
    }

    /// Get interconnect specification by name
    pub fn get_interconnect(&self, name: &str) -> Option<&InterconnectSpec> {
        self.interconnects.get(name)
    }

    /// List all available GPUs
    pub fn list_gpus(&self) -> Vec<&GpuSpec> {
        self.gpus.values().collect()
    }
}

impl Default for HardwareDatabase {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod gpu_name_resolution_tests {
    use super::*;

    /// Configs and papers write the bare family name; the database keys the
    /// board variants. A miss here silently degrades every hardware-derived
    /// metric to a generic 20-TFLOPS profile.
    #[test]
    fn bare_family_names_resolve_to_a_real_board() {
        let db = HardwareDatabase::new();
        for (query, expected) in [
            ("H100", "H100-SXM"),
            ("A100", "A100-SXM"),
            ("V100", "V100"),
            ("h100", "H100-SXM"),
            ("A100-80GB", "A100-SXM"),
            ("4090", "RTX4090"),
        ] {
            let spec = db
                .get_gpu(query)
                .unwrap_or_else(|| panic!("{query} should resolve"));
            assert_eq!(spec.name, expected, "{query} resolved to the wrong board");
        }
    }

    #[test]
    fn resolved_h100_carries_real_specs_not_the_generic_fallback() {
        let db = HardwareDatabase::new();
        let h100 = db.get_gpu("H100").expect("H100 should resolve");
        assert!(
            h100.tflops_bf16 > 100.0,
            "H100 BF16 should be far above the 50 TFLOPS generic fallback, got {}",
            h100.tflops_bf16
        );
        assert_eq!(h100.tdp_watts, 700, "H100-SXM TDP should be 700 W");
    }

    #[test]
    fn unknown_names_still_miss() {
        let db = HardwareDatabase::new();
        assert!(db.get_gpu("NotAGpu9000").is_none());
    }
}
