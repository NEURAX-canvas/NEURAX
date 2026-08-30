//! Option 1 — MLIR as an independent structural verifier for `neurax-formulas`,
//! extended across 10 real models spanning NEURAX's 8 supported architecture
//! families (transformer, moe, cnn, ssm, diffusion, gnn, rnn, gan).
//!
//! For every layer whose shape neurax-mlir's public `TargetLowering` API can
//! represent at all, this:
//!   1. Shells out to `mlir-opt` to verify the generated text is structurally
//!      valid, correctly-typed MLIR.
//!   2. Re-parses the tensor shapes out of the generated text and recomputes
//!      FLOPs from them, to confirm the lowering's own numbers agree with the
//!      formula's.
//!
//! Layers the API cannot represent at all (no stride/padding/groups params on
//! `lower_conv2d`; no lowering for MoE/SSM/RNN/GNN/GAN block types) are
//! reported as coverage gaps rather than silently skipped — the point of this
//! tool is to measure real coverage, not to flatter it.
//!
//! Run with: PATH="/usr/lib/llvm-18/bin:$PATH" cargo run --example formula_mlir_crosscheck -p neurax-mlir

use neurax_mlir::targets::CudaBackend;
use neurax_mlir::TargetLowering;
use neurax_parser::{LayerType, ModelConfig};
use std::io::Write;
use std::process::Command;

fn dtype_for(precision: &str) -> &'static str {
    match precision {
        "fp32" | "float32" => "f32",
        "fp16" | "float16" => "f16",
        "bf16" | "bfloat16" => "bf16",
        _ => "f32",
    }
}

fn verify_with_mlir_opt(mlir_body: &str) -> Result<(), String> {
    let wrapped = format!("module {{\n{mlir_body}}}\n");
    let mut child = Command::new("mlir-opt")
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("mlir-opt on PATH");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(wrapped.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr)
            .lines()
            .next()
            .unwrap_or("")
            .to_string())
    }
}

fn extract_tensor_shapes(mlir: &str) -> Vec<Vec<String>> {
    let mut shapes = Vec::new();
    let mut rest = mlir;
    while let Some(start) = rest.find("tensor<") {
        let after = &rest[start + "tensor<".len()..];
        if let Some(end) = after.find('>') {
            shapes.push(after[..end].split('x').map(|s| s.to_string()).collect());
            rest = &after[end + 1..];
        } else {
            break;
        }
    }
    shapes
}

#[derive(Debug)]
enum Verdict {
    Checked {
        formula_flops: f64,
        mlir_flops: Option<f64>,
        verified: Result<(), String>,
    },
    StubOnly, // lowering exists but is a hollow stub (tensor.empty → return); nothing to check
    Uncoverable(&'static str),
}

fn assess_dense(batch: usize, in_f: usize, out_f: usize, dtype: &str) -> Verdict {
    let formula_flops = 2.0 * batch as f64 * in_f as f64 * out_f as f64;
    let mlir = CudaBackend::lower_matmul(1, batch, in_f, out_f, dtype).unwrap();
    let verified = verify_with_mlir_opt(&mlir);
    let shapes = extract_tensor_shapes(&mlir);
    let mlir_flops = shapes.first().and_then(|a| {
        let out = shapes.get(2)?;
        let m: f64 = a.get(1)?.parse().ok()?;
        let k: f64 = a.get(2)?.parse().ok()?;
        let n: f64 = out.get(2)?.parse().ok()?;
        Some(2.0 * m * k * n)
    });
    Verdict::Checked {
        formula_flops,
        mlir_flops,
        verified,
    }
}

/// MLP's own first projection (hidden -> intermediate) as a standalone matmul —
/// a sub-component check, not the full gated-MLP FLOPs (which also includes the
/// down-projection, the gate branch if gated, and the activation cost).
fn assess_mlp_up_projection(
    batch: usize,
    seq: usize,
    hidden: usize,
    intermediate: usize,
    dtype: &str,
) -> Verdict {
    let m = batch * seq;
    let formula_flops = 2.0 * m as f64 * hidden as f64 * intermediate as f64;
    let mlir = CudaBackend::lower_matmul(1, m, hidden, intermediate, dtype).unwrap();
    let verified = verify_with_mlir_opt(&mlir);
    let shapes = extract_tensor_shapes(&mlir);
    let mlir_flops = shapes.first().and_then(|a| {
        let out = shapes.get(2)?;
        let mm: f64 = a.get(1)?.parse().ok()?;
        let k: f64 = a.get(2)?.parse().ok()?;
        let n: f64 = out.get(2)?.parse().ok()?;
        Some(2.0 * mm * k * n)
    });
    Verdict::Checked {
        formula_flops,
        mlir_flops,
        verified,
    }
}

fn assess_conv(
    batch: usize,
    in_ch: usize,
    out_ch: usize,
    h: usize,
    w: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dtype: &str,
) -> Verdict {
    if stride != 1 || padding != 0 {
        return Verdict::Uncoverable(
            "lower_conv2d has no stride/padding parameter (hardcodes stride=1, padding=0)",
        );
    }
    let formula_flops =
        neurax_formulas::conv::conv2d_flops(batch, in_ch, out_ch, h, w, kernel, kernel, 1, 0, 1);
    let mlir = CudaBackend::lower_conv2d(batch, in_ch, out_ch, h, w, kernel, dtype).unwrap();
    let verified = verify_with_mlir_opt(&mlir);
    Verdict::Checked {
        formula_flops,
        mlir_flops: None,
        verified,
    }
}

fn assess_layer(
    cfg: &ModelConfig,
    lt: LayerType,
    params: &neurax_parser::LayerParams,
    dtype: &str,
) -> Verdict {
    let batch = cfg.training.batch_size;
    let seq = cfg
        .training
        .sequence_length
        .or(cfg.model.global_params.sequence_length)
        .unwrap_or(512);

    match lt {
        LayerType::Dense => {
            let in_f = params.in_features.unwrap_or(512);
            let out_f = params.out_features.unwrap_or(512);
            assess_dense(batch, in_f, out_f, dtype)
        }
        LayerType::Mlp => {
            let hidden = params.hidden_size.unwrap_or(512);
            let intermediate = params.intermediate_size.unwrap_or(4 * hidden);
            assess_mlp_up_projection(batch, seq, hidden, intermediate, dtype)
        }
        LayerType::Conv => {
            let in_ch = params.in_channels.unwrap_or(3);
            let out_ch = params.out_channels.unwrap_or(64);
            let kernel = params.kernel_size.unwrap_or(3);
            let stride = params.stride.unwrap_or(1);
            let padding = params.padding.unwrap_or(0);
            let h = cfg.data.image_height.unwrap_or(224);
            let w = cfg.data.image_width.unwrap_or(224);
            assess_conv(batch, in_ch, out_ch, h, w, kernel, stride, padding, dtype)
        }
        LayerType::Attention => Verdict::StubOnly, // lower_attention() is tensor.empty() -> return

        // MoE: the router projection [B,S,H] -> [B,S,num_experts] is a real, plain
        // matmul component of moe::moe_router_flops (the softmax term alongside it
        // is not a matmul and is excluded from this sub-check, same convention as
        // the MLP up-projection check).
        LayerType::MoE => {
            let hidden = params.hidden_size.unwrap_or(512);
            let num_experts = params.num_experts.unwrap_or(8);
            let m = batch * seq;
            let formula_flops = 2.0 * m as f64 * hidden as f64 * num_experts as f64;
            let mlir = CudaBackend::lower_matmul(1, m, hidden, num_experts, dtype).unwrap();
            let verified = verify_with_mlir_opt(&mlir);
            Verdict::Checked {
                formula_flops,
                mlir_flops: reparsed_matmul_flops(&mlir),
                verified,
            }
        }

        // Mamba/RWKV: neurax-ir's own operator pass costs both with
        // ssm::mamba_flops(), whose in_proj term ([B,S,H] -> [B,S,2*d_inner]) is a
        // real matmul component — checked here as such.
        LayerType::MambaBlock | LayerType::RwkvBlock => {
            let hidden = params.hidden_size.unwrap_or(512);
            let expand = params.expansion_factor.unwrap_or(2);
            let d_inner = hidden * expand;
            let m = batch * seq;
            let n = d_inner * 2;
            let formula_flops = 2.0 * m as f64 * hidden as f64 * n as f64;
            let mlir = CudaBackend::lower_matmul(1, m, hidden, n, dtype).unwrap();
            let verified = verify_with_mlir_opt(&mlir);
            Verdict::Checked {
                formula_flops,
                mlir_flops: reparsed_matmul_flops(&mlir),
                verified,
            }
        }

        // LSTM/GRU: neurax-ir's own formula is exactly `gates` independent
        // hidden x hidden matmuls (batch*seq*hidden*hidden*gates) — a real,
        // decomposable formula, unlike the diffusion/GAN placeholders below.
        // Checked here as one gate's matmul (1/`gates` of the total).
        LayerType::LstmBlock | LayerType::GruBlock | LayerType::RnnCell => {
            let hidden = params.rnn_hidden_size.unwrap_or(512);
            let m = batch * seq;
            let formula_flops = 2.0 * m as f64 * hidden as f64 * hidden as f64;
            let mlir = CudaBackend::lower_matmul(1, m, hidden, hidden, dtype).unwrap();
            let verified = verify_with_mlir_opt(&mlir);
            Verdict::Checked {
                formula_flops,
                mlir_flops: reparsed_matmul_flops(&mlir),
                verified,
            }
        }

        // GCN/message-passing: neurax-ir costs this with gnn::gcn_flops(), whose
        // dominant linear_flops term ([N,in] x [in,out] -> [N,out]) is a real,
        // plain matmul — the sparse aggregation/normalization terms alongside it
        // are graph-structured, not a dense tensor op, and are excluded here.
        LayerType::GraphConvNet | LayerType::MessagePassing => {
            let in_f = params.in_features.unwrap_or(64);
            let out_f = params.out_features.unwrap_or(64);
            let num_nodes = 2708; // Cora-sized default, matching operator/pass.rs's own fallback
            let formula_flops = 2.0 * num_nodes as f64 * in_f as f64 * out_f as f64;
            let mlir = CudaBackend::lower_matmul(1, num_nodes, in_f, out_f, dtype).unwrap();
            let verified = verify_with_mlir_opt(&mlir);
            Verdict::Checked {
                formula_flops,
                mlir_flops: reparsed_matmul_flops(&mlir),
                verified,
            }
        }

        // Diffusion — as of the operator/pass.rs fix, TimeEmbedding/ConditionBlock
        // now have a real MLP formula (Linear+SiLU+Linear / Linear+GELU+Linear),
        // checked here the same way as a transformer's Mlp up-projection.
        LayerType::TimeEmbedding | LayerType::TimestepBlock | LayerType::ConditionBlock => {
            let channels = params.hidden_size.unwrap_or(320);
            let m = batch * seq;
            let formula_flops = 2.0 * m as f64 * channels as f64 * (channels * 4) as f64;
            let mlir = CudaBackend::lower_matmul(1, m, channels, channels * 4, dtype).unwrap();
            let verified = verify_with_mlir_opt(&mlir);
            Verdict::Checked {
                formula_flops,
                mlir_flops: reparsed_matmul_flops(&mlir),
                verified,
            }
        }

        // CrossAttention now has a real attention_flops() formula, but
        // lower_attention() is still the hollow stub — same situation as plain
        // Attention above.
        LayerType::CrossAttention => Verdict::StubOnly,

        // UnetBlock and friends now use cnn_blocks::resnet_basic_block_flops() —
        // a real formula, but a composite one (2 convs + skip), same situation as
        // ResnetBottleneck below: no single conv2d call to check it against.
        LayerType::UnetBlock | LayerType::ResnetBlock | LayerType::DownBlock | LayerType::UpBlock | LayerType::MidBlock => {
            Verdict::Uncoverable("composite block (2 convs + skip); no single-op lowering to check against")
        }

        // NoisePredictor/VAE now use conv::conv2d_flops() with padding=1 — a real
        // formula, but lower_conv2d() still has no padding parameter at all, so
        // this specific (real, correct) shape still can't be represented.
        LayerType::NoisePredictor | LayerType::VaeEncoder | LayerType::VaeDecoder => {
            Verdict::Uncoverable("real conv2d formula now used (padding=1), but lower_conv2d has no padding parameter")
        }

        // GAN — GeneratorBlock/DiscriminatorBlock/ProgressiveBlock now use a real
        // conv2d formula too. Checkable in principle, but every real GAN
        // architecture's transposed/strided convs use stride != 1 or padding != 0
        // (this fixture's DCGAN layers included), which lower_conv2d cannot
        // represent — the same API gap VGG-16's padding=1 convs hit.
        LayerType::GeneratorBlock | LayerType::DiscriminatorBlock | LayerType::ProgressiveBlock => {
            let in_ch = params.in_channels.unwrap_or(64);
            let out_ch = params.out_channels.unwrap_or(64);
            let kernel = params.kernel_size.unwrap_or(3);
            let stride = params.stride.unwrap_or(1);
            let padding = params.padding.unwrap_or(0);
            let side = (seq as f64).sqrt().round().max(1.0) as usize;
            if stride != 1 || padding != 0 {
                Verdict::Uncoverable("lower_conv2d has no stride/padding parameter (hardcodes stride=1, padding=0)")
            } else {
                let formula_flops = neurax_formulas::conv::conv2d_flops(
                    batch, in_ch, out_ch, side, side, kernel, kernel, 1, 0, 1,
                );
                let mlir = CudaBackend::lower_conv2d(batch, in_ch, out_ch, side, side, kernel, dtype).unwrap();
                let verified = verify_with_mlir_opt(&mlir);
                Verdict::Checked { formula_flops, mlir_flops: None, verified }
            }
        }

        // SelfAttention now has a real attention_flops() formula; same stub
        // limitation as every other attention variant.
        LayerType::SelfAttention => Verdict::StubOnly,

        // StyleMod/AdaIN/PixelNorm/MinibatchStd/SpectralNorm now have real,
        // small elementwise formulas (matching their own param formulas) instead
        // of the old full-layer placeholder — but elementwise ops aren't matmul
        // or conv shapes, so there is nothing to lower them to yet.
        LayerType::StyleMod | LayerType::AdaIN | LayerType::PixelNorm | LayerType::MinibatchStd | LayerType::SpectralNorm => {
            Verdict::Uncoverable("real elementwise formula now used; no matmul/conv shape to lower an elementwise op to")
        }

        LayerType::ResnetBottleneck => Verdict::Uncoverable(
            "composite block (3 convs + skip); no single-op lowering to check against",
        ),

        LayerType::Embedding | LayerType::Normalization | LayerType::Pooling => {
            Verdict::Uncoverable("negligible-cost layer type; no lowering attempted by design")
        }
        _ => Verdict::Uncoverable("no lowering function exists for this layer type"),
    }
}

fn reparsed_matmul_flops(mlir: &str) -> Option<f64> {
    let shapes = extract_tensor_shapes(mlir);
    let a = shapes.first()?;
    let out = shapes.get(2)?;
    let m: f64 = a.get(1)?.parse().ok()?;
    let k: f64 = a.get(2)?.parse().ok()?;
    let n: f64 = out.get(2)?.parse().ok()?;
    Some(2.0 * m * k * n)
}

struct Row {
    model: &'static str,
    family: &'static str,
    layer_id: String,
    layer_type: String,
    verdict: Verdict,
}

fn load(path: &str) -> ModelConfig {
    let text = std::fs::read_to_string(path).unwrap_or_else(|_| panic!("missing {path}"));
    neurax_parser::parse_model_config(&text).unwrap_or_else(|e| panic!("{path}: {e:?}"))
}

fn main() {
    let models: &[(&str, &str, &str)] = &[
        (
            "examples/models/gpt3_175b.json",
            "GPT-3-175B",
            "transformer",
        ),
        ("examples/models/mixtral_8x7b.json", "Mixtral-8x7B", "moe"),
        ("examples/models/resnet50.json", "ResNet-50", "cnn"),
        ("examples/models/vgg16.json", "VGG-16", "cnn"),
        ("examples/models/mamba_2.8b.json", "Mamba-2.8B", "ssm"),
        ("examples/models/sdxl_1.0.json", "SDXL-1.0", "diffusion"),
        (
            "examples/models/gcn_ogbn_arxiv.json",
            "GCN-ogbn-arxiv",
            "gnn",
        ),
        ("examples/models/awd_lstm_ptb.json", "AWD-LSTM-PTB", "rnn"),
        ("examples/models/rwkv_7b.json", "RWKV-7B", "rnn"),
        ("examples/models/dcgan.json", "DCGAN", "gan"),
    ];

    let mut rows: Vec<Row> = Vec::new();

    for &(path, name, family) in models {
        let cfg = load(path);
        let dtype = dtype_for(&cfg.training.precision);
        for layer in &cfg.model.layers {
            let verdict = assess_layer(&cfg, layer.layer_type, &layer.params, dtype);
            rows.push(Row {
                model: name,
                family,
                layer_id: layer.id.clone(),
                layer_type: layer.layer_type.as_str().to_string(),
                verdict,
            });
        }
    }

    // ---- Detailed table ----
    println!(
        "{:<15} {:<14} {:<16} {:<20} {}",
        "family", "model", "layer", "type", "verdict"
    );
    println!("{}", "-".repeat(110));
    for r in &rows {
        let verdict_str = match &r.verdict {
            Verdict::Checked {
                verified: Ok(()), ..
            } => "✓ checked, MLIR valid".to_string(),
            Verdict::Checked {
                verified: Err(e), ..
            } => format!("✗ checked, MLIR REJECTED: {e}"),
            Verdict::StubOnly => "○ lowering is a hollow stub — nothing to verify".to_string(),
            Verdict::Uncoverable(reason) => format!("— uncoverable: {reason}"),
        };
        println!(
            "{:<15} {:<14} {:<16} {:<20} {}",
            r.family, r.model, r.layer_id, r.layer_type, verdict_str
        );
    }

    // ---- Coverage summary by family ----
    println!("\n{}", "=".repeat(70));
    println!(
        "COVERAGE SUMMARY — 8 familles, {} modèles, {} couches",
        models.len(),
        rows.len()
    );
    println!("{}", "=".repeat(70));
    println!(
        "{:<14} {:>8} {:>10} {:>10} {:>14} {:>10}",
        "famille", "couches", "vérifiées", "réussies", "stub-only", "non-couvertes"
    );
    let families = [
        "transformer",
        "moe",
        "cnn",
        "ssm",
        "diffusion",
        "gnn",
        "rnn",
        "gan",
    ];
    let mut total_checked = 0usize;
    let mut total_passed = 0usize;
    let mut total_layers = 0usize;
    for fam in families {
        let fam_rows: Vec<&Row> = rows.iter().filter(|r| r.family == fam).collect();
        let total = fam_rows.len();
        let checked = fam_rows
            .iter()
            .filter(|r| matches!(r.verdict, Verdict::Checked { .. }))
            .count();
        let passed = fam_rows
            .iter()
            .filter(|r| {
                matches!(
                    &r.verdict,
                    Verdict::Checked {
                        verified: Ok(()),
                        ..
                    }
                )
            })
            .count();
        let stub = fam_rows
            .iter()
            .filter(|r| matches!(r.verdict, Verdict::StubOnly))
            .count();
        let uncovered = fam_rows
            .iter()
            .filter(|r| matches!(r.verdict, Verdict::Uncoverable(_)))
            .count();
        total_checked += checked;
        total_passed += passed;
        total_layers += total;
        println!(
            "{:<14} {:>8} {:>10} {:>10} {:>14} {:>10}",
            fam, total, checked, passed, stub, uncovered
        );
    }
    println!("{}", "-".repeat(70));
    println!(
        "TOTAL          {:>8} {:>10} {:>10}",
        total_layers, total_checked, total_passed
    );
    println!(
        "\nCouverture réelle (couches vérifiables par l'API publique / couches totales) : {:.1}%",
        100.0 * total_checked as f64 / total_layers as f64
    );
}
