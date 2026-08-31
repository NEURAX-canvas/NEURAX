//! Test compilation of a Diffusion model (Stable Diffusion style)
//! Compares output metrics with real-world models (SD 1.5, SDXL, SD3)
//! JSON input follows the neurax-IR standard format

/// Stable Diffusion XL (SDXL) - 2.6B parameters UNet
/// 1024x1024 generation with VAE encoder/decoder
const DIFFUSION_SDXL_JSON: &str = r#"
{
    "schema_version": "1.0",
    "model": {
        "name": "Stable-Diffusion-XL",
        "type": "diffusion",
        "layers": [
            {"id": "time_embed", "layer_type": "time_embedding", "input_shape": [1], "output_shape": [320], "params": {"time_embedding_dim": 320, "flip_sin_to_cos": true, "freq_shift": 0}},
            {"id": "conv_in", "layer_type": "conv", "input_shape": [128, 128, 4], "output_shape": [128, 128, 320], "params": {"in_channels": 4, "out_channels": 320, "kernel_size": 3, "padding": 1}},
            
            {"id": "down_block_0", "layer_type": "down_block", "input_shape": [128, 128, 320], "output_shape": [64, 64, 320], "params": {"in_channels": 320, "out_channels": 320, "num_layers": 2, "cross_attention_dim": 2048, "attention_head_dim": 64}},
            {"id": "down_block_1", "layer_type": "down_block", "input_shape": [64, 64, 320], "output_shape": [32, 32, 640], "params": {"in_channels": 320, "out_channels": 640, "num_layers": 2, "cross_attention_dim": 2048, "attention_head_dim": 64}},
            {"id": "down_block_2", "layer_type": "down_block", "input_shape": [32, 32, 640], "output_shape": [16, 16, 1280], "params": {"in_channels": 640, "out_channels": 1280, "num_layers": 2, "cross_attention_dim": 2048, "attention_head_dim": 64}},
            {"id": "down_block_3", "layer_type": "down_block", "input_shape": [16, 16, 1280], "output_shape": [8, 8, 1280], "params": {"in_channels": 1280, "out_channels": 1280, "num_layers": 2}},
            
            {"id": "mid_block", "layer_type": "mid_block", "input_shape": [8, 8, 1280], "output_shape": [8, 8, 1280], "params": {"in_channels": 1280, "cross_attention_dim": 2048, "attention_head_dim": 64}},
            
            {"id": "up_block_0", "layer_type": "up_block", "input_shape": [8, 8, 2560], "output_shape": [16, 16, 1280], "params": {"in_channels": 2560, "out_channels": 1280, "num_layers": 2}},
            {"id": "up_block_1", "layer_type": "up_block", "input_shape": [16, 16, 2560], "output_shape": [32, 32, 640], "params": {"in_channels": 2560, "out_channels": 640, "num_layers": 2, "cross_attention_dim": 2048, "attention_head_dim": 64}},
            {"id": "up_block_2", "layer_type": "up_block", "input_shape": [32, 32, 1280], "output_shape": [64, 64, 320], "params": {"in_channels": 1280, "out_channels": 320, "num_layers": 2, "cross_attention_dim": 2048, "attention_head_dim": 64}},
            {"id": "up_block_3", "layer_type": "up_block", "input_shape": [64, 64, 640], "output_shape": [128, 128, 320], "params": {"in_channels": 640, "out_channels": 320, "num_layers": 2, "cross_attention_dim": 2048, "attention_head_dim": 64}},
            
            {"id": "conv_out", "layer_type": "conv", "input_shape": [128, 128, 320], "output_shape": [128, 128, 4], "params": {"in_channels": 320, "out_channels": 4, "kernel_size": 3, "padding": 1}},
            
            {"id": "vae_encoder", "layer_type": "vae_encoder", "input_shape": [1024, 1024, 3], "output_shape": [128, 128, 4], "params": {"in_channels": 3, "out_channels": 4, "latent_channels": 4, "vae_scale_factor": 8}},
            {"id": "vae_decoder", "layer_type": "vae_decoder", "input_shape": [128, 128, 4], "output_shape": [1024, 1024, 3], "params": {"in_channels": 4, "out_channels": 3, "latent_channels": 4, "vae_scale_factor": 8}},
            
            {"id": "text_encoder_1", "layer_type": "dense", "input_shape": [77, 2048], "output_shape": [77, 2048], "params": {"in_features": 2048, "out_features": 2048}},
            {"id": "text_encoder_2", "layer_type": "dense", "input_shape": [77, 1280], "output_shape": [77, 1280], "params": {"in_features": 1280, "out_features": 1280}}
        ],
        "global_params": {
            "image_size": 1024,
            "in_channels": 4,
            "out_channels": 4,
            "latent_channels": 4,
            "diffusion_timesteps": 1000,
            "noise_schedule": "scaled_linear",
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "cross_attention_dim": 2048,
            "attention_head_dim": 64,
            "block_out_channels": [320, 640, 1280, 1280],
            "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
            "layers_per_block": 2,
            "vae_scale_factor": 8,
            "sample_size": 128
        }
    },
    "training": {
        "batch_size": 32,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "precision": "fp16",
        "gradient_checkpointing": true,
        "zero_stage": 2,
        "max_steps": 500000,
        "warmup_steps": 10000,
        "parallelism": {
            "data_parallel": 64,
            "tensor_parallel": 1,
            "pipeline_parallel": 1
        }
    },
    "hardware": {
        "gpus": [
            {
                "name": "A100-80GB",
                "count": 64,
                "memory_gb": 80,
                "tflops_fp16": 312,
                "tflops_fp32": 19.5,
                "tflops_fp8": 624,
                "memory_bandwidth_gb_s": 2039,
                "tensor_cores": true,
                "nvlink": true
            }
        ],
        "interconnect": "InfiniBand-200Gb/s",
        "interconnect_bandwidth_gb_s": 25
    },
    "data": {
        "input_shape": [1024, 1024, 3],
        "dtype": "fp16",
        "image_height": 1024,
        "image_width": 1024,
        "image_channels": 3
    },
    "cost_config": {
        "provider": "aws",
        "gpu_hour_usd": 4.50,
        "energy_kwh_usd": 0.12,
        "pue_factor": 1.2
    }
}
"#;

/// Real-world Diffusion model specifications
struct RealDiffusionSpecs {
    name: &'static str,
    unet_params_billion: f64,
    vae_params_million: f64,
    text_encoder_params_million: f64,
    total_params_billion: f64,
    image_size: u32,
    latent_size: u32,
    latent_channels: u32,
    diffusion_steps: u32,
}

impl RealDiffusionSpecs {
    /// Stable Diffusion 1.5
    fn sd15() -> Self {
        Self {
            name: "Stable-Diffusion-1.5",
            unet_params_billion: 0.86,
            vae_params_million: 83.0,
            text_encoder_params_million: 123.0, // CLIP ViT-L/14
            total_params_billion: 0.98,
            image_size: 512,
            latent_size: 64,
            latent_channels: 4,
            diffusion_steps: 1000,
        }
    }

    /// Stable Diffusion XL
    fn sdxl() -> Self {
        Self {
            name: "Stable-Diffusion-XL",
            unet_params_billion: 2.6,
            vae_params_million: 83.0,
            text_encoder_params_million: 860.0, // CLIP ViT-G + ViT-L
            total_params_billion: 3.5,
            image_size: 1024,
            latent_size: 128,
            latent_channels: 4,
            diffusion_steps: 1000,
        }
    }

    /// Stable Diffusion 3
    fn sd3() -> Self {
        Self {
            name: "Stable-Diffusion-3",
            unet_params_billion: 2.0, // MMDiT
            vae_params_million: 83.0,
            text_encoder_params_million: 2500.0, // T5-XXL + CLIP
            total_params_billion: 8.0,
            image_size: 1024,
            latent_size: 128,
            latent_channels: 16,
            diffusion_steps: 50,
        }
    }

    /// DALL-E 2
    fn dalle2() -> Self {
        Self {
            name: "DALL-E-2",
            unet_params_billion: 3.0,
            vae_params_million: 65.0,
            text_encoder_params_million: 400.0,
            total_params_billion: 3.5,
            image_size: 1024,
            latent_size: 64,
            latent_channels: 4,
            diffusion_steps: 1000,
        }
    }

    /// Calculate UNet parameters
    fn calculate_unet_params(
        block_out_channels: &[u64],
        layers_per_block: u32,
        cross_attention_dim: u64,
        attention_head_dim: u64,
    ) -> f64 {
        let mut params = 0.0;

        // Time embedding
        let time_dim = block_out_channels[0] as f64;
        params += time_dim * time_dim * 4.0; // 2 linear layers + activations

        // Down blocks
        for (i, &ch) in block_out_channels.iter().enumerate() {
            let ch_f = ch as f64;
            let num_layers = layers_per_block as f64;

            // ResNet blocks
            params += num_layers * ch_f * ch_f * 3.0 * 3.0 * 2.0; // Conv layers

            // Downsampling conv
            if i < block_out_channels.len() - 1 {
                params += ch_f * ch_f * 3.0 * 3.0;
            }

            // Cross-attention (if applicable)
            if cross_attention_dim > 0 {
                let num_heads = ch_f / attention_head_dim as f64;
                params += num_layers
                    * (
                        // Self-attention
                        4.0 * ch_f * ch_f +
                    // Cross-attention
                    2.0 * ch_f * cross_attention_dim as f64 +
                    // FFN
                    8.0 * ch_f * ch_f
                    );
            }
        }

        // Mid block
        let mid_ch = *block_out_channels.last().unwrap() as f64;
        params += mid_ch * mid_ch * 3.0 * 3.0 * 2.0; // ResNet
        if cross_attention_dim > 0 {
            params += 4.0 * mid_ch * mid_ch; // Attention
        }

        // Up blocks (similar to down but with skip connections)
        for (i, &ch) in block_out_channels.iter().enumerate().rev() {
            let ch_f = ch as f64;
            let num_layers = layers_per_block as f64;
            let prev_ch = if i < block_out_channels.len() - 1 {
                block_out_channels[i + 1] as f64
            } else {
                ch_f
            };

            // ResNet blocks with skip connections
            params += num_layers * (ch_f + prev_ch) * ch_f * 3.0 * 3.0;

            // Upsampling conv
            if i > 0 {
                params += ch_f * ch_f * 3.0 * 3.0;
            }

            // Cross-attention
            if cross_attention_dim > 0 {
                let num_heads = ch_f / attention_head_dim as f64;
                params += num_layers
                    * (4.0 * ch_f * ch_f
                        + 2.0 * ch_f * cross_attention_dim as f64
                        + 8.0 * ch_f * ch_f);
            }
        }

        // Conv in/out
        let in_ch = 4.0; // latent channels
        let out_ch = 4.0;
        let first_ch = block_out_channels[0] as f64;
        params += in_ch * first_ch * 3.0 * 3.0;
        params += first_ch * out_ch * 3.0 * 3.0;

        params / 1e9
    }
}

#[test]
fn test_diffusion_architecture_components() {
    println!("\n=== Diffusion Architecture Components ===\n");

    let sd15 = RealDiffusionSpecs::sd15();
    let sdxl = RealDiffusionSpecs::sdxl();

    println!("┌────────────────────────────────────────────────────────────────────┐");
    println!("│                    COMPONENT BREAKDOWN                            │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!("│ Component       │ SD 1.5 (M)  │ SDXL (M)   │ Description          │");
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ UNet            │ {:>10.0}  │ {:>10.0}  │ Denoising network    │",
        sd15.unet_params_billion * 1000.0,
        sdxl.unet_params_billion * 1000.0
    );
    println!(
        "│ VAE Encoder     │ {:>10.0}  │ {:>10.0}  │ Image → Latent       │",
        sd15.vae_params_million / 2.0,
        sdxl.vae_params_million / 2.0
    );
    println!(
        "│ VAE Decoder     │ {:>10.0}  │ {:>10.0}  │ Latent → Image       │",
        sd15.vae_params_million / 2.0,
        sdxl.vae_params_million / 2.0
    );
    println!(
        "│ Text Encoder    │ {:>10.0}  │ {:>10.0}  │ Prompt conditioning  │",
        sd15.text_encoder_params_million, sdxl.text_encoder_params_million
    );
    println!("├────────────────────────────────────────────────────────────────────┤");
    println!(
        "│ TOTAL           │ {:>10.0}  │ {:>10.0}  │                      │",
        sd15.total_params_billion * 1000.0,
        sdxl.total_params_billion * 1000.0
    );
    println!("└────────────────────────────────────────────────────────────────────┘\n");

    // Architecture details
    println!("UNet Architecture:\n");
    println!("  - DownBlocks: ResNet + Cross-Attention + Downsample");
    println!("  - MidBlock:   ResNet + Self-Attention + ResNet");
    println!("  - UpBlocks:   ResNet + Cross-Attention + Upsample");
    println!("  - Skip connections between DownBlocks and UpBlocks\n");

    println!("Cross-Attention Mechanism:\n");
    println!("  - Query: from image features");
    println!("  - Key/Value: from text encoder embeddings");
    println!("  - Enables text-conditioned generation\n");
}
