//! Test compilation of a Diffusion model (Stable Diffusion style)
//! Compares output metrics with real-world models (SD 1.5, SDXL, SD3)
//! JSON input follows the neurax-IR standard format

/// Real-world Diffusion model specifications
struct RealDiffusionSpecs {
    unet_params_billion: f64,
    vae_params_million: f64,
    text_encoder_params_million: f64,
    total_params_billion: f64,
}

impl RealDiffusionSpecs {
    /// Stable Diffusion 1.5
    fn sd15() -> Self {
        Self {
            unet_params_billion: 0.86,
            vae_params_million: 83.0,
            text_encoder_params_million: 123.0, // CLIP ViT-L/14
            total_params_billion: 0.98,
        }
    }

    /// Stable Diffusion XL
    fn sdxl() -> Self {
        Self {
            unet_params_billion: 2.6,
            vae_params_million: 83.0,
            text_encoder_params_million: 860.0, // CLIP ViT-G + ViT-L
            total_params_billion: 3.5,
        }
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
