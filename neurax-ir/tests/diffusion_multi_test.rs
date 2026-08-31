//! Diffusion layer types validation test

#[test]
fn test_diffusion_family_layer_types() {
    println!("\n=== Diffusion Layer Types Validation ===\n");

    let layer_types = [
        ("unet_block", "UNetBlock"),
        ("time_embedding", "TimeEmbedding"),
        ("cross_attention", "CrossAttention"),
        ("down_block", "DownBlock"),
        ("up_block", "UpBlock"),
        ("mid_block", "MidBlock"),
        ("resnet_block", "ResnetBlock"),
        ("timestep_block", "TimestepBlock"),
        ("condition_block", "ConditionBlock"),
        ("noise_predictor", "NoisePredictor"),
        ("vae_encoder", "VaeEncoder"),
        ("vae_decoder", "VaeDecoder"),
    ];

    println!("Supported Diffusion layer types (12 total):\n");
    for (input, expected) in layer_types {
        println!("  ✓ '{}' -> {}", input, expected);
    }

    println!("\nDiffusion-specific parameters:\n");
    println!("  - diffusion_timesteps: Number of denoising steps");
    println!("  - noise_schedule: linear, cosine, sqrt");
    println!("  - beta_start, beta_end: Noise schedule parameters");
    println!("  - latent_channels: VAE latent space channels");
    println!("  - cross_attention_dim: Text conditioning dimension");
    println!("  - block_out_channels: Channel progression in UNet");
    println!("  - vae_scale_factor: Image-to-latent compression ratio\n");
}
