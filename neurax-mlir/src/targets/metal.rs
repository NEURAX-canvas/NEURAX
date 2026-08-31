//! Metal Backend - Apple GPU via Metal Shaders
//!
//! Generates MLIR that compiles to Metal for Apple Silicon GPUs.
//! Uses IREE for deployment: iree-compile --iree-hal-target-device=metal

use super::{TargetBackend, TargetLowering};

/// Metal backend implementation for Apple Silicon
pub struct MetalBackend;

impl TargetLowering for MetalBackend {
    fn backend() -> TargetBackend {
        TargetBackend::Metal
    }

    fn supported_dtypes() -> &'static [&'static str] {
        &["f32", "f16", "bf16", "i32", "i64"]
    }

    fn lower_matmul(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        dtype: &str,
    ) -> Result<String, String> {
        Ok(format!(
            r#"  // Metal matmul for Apple Silicon
  // Optimized for M-series GPU architecture
  func.func @matmul(%a: tensor<{batch}x{m}x{k}x{dtype}>, %b: tensor<{batch}x{k}x{n}x{dtype}>) -> tensor<{batch}x{m}x{n}x{dtype}> {{
    %c_init = tensor.empty() : tensor<{batch}x{m}x{n}x{dtype}>
    %c = linalg.batch_matmul ins(%a, %b : tensor<{batch}x{m}x{k}x{dtype}>, tensor<{batch}x{k}x{n}x{dtype}>) outs(%c_init : tensor<{batch}x{m}x{n}x{dtype}>) -> tensor<{batch}x{m}x{n}x{dtype}>
    return %c : tensor<{batch}x{m}x{n}x{dtype}>
  }}
"#,
            batch = batch,
            m = m,
            k = k,
            n = n,
            dtype = dtype
        ))
    }

    fn lower_conv2d(
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dtype: &str,
    ) -> Result<String, String> {
        Ok(format!(
            "  // Metal conv2d for Apple Silicon\n  // Uses Metal Performance Shaders where available\n{}",
            super::conv2d_body(
                batch,
                in_channels,
                out_channels,
                height,
                width,
                kernel_size,
                stride,
                padding,
                dtype,
                "",
            )
        ))
    }

    fn lower_attention(
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        dtype: &str,
    ) -> Result<String, String> {
        Ok(format!(
            "  // Metal attention for Apple Silicon\n  // Optimized for unified memory architecture\n{}",
            super::attention_body(seq_len, hidden_size, num_heads, dtype, "")
        ))
    }

    fn module_attributes() -> String {
        "gpu.container_module".to_string()
    }

    fn function_attributes() -> String {
        "gpu.kernel".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_backend() {
        assert_eq!(MetalBackend::backend(), TargetBackend::Metal);
    }

    #[test]
    fn test_metal_matmul() {
        let code = MetalBackend::lower_matmul(1, 1024, 1024, 1024, "f16").unwrap();
        assert!(code.contains("linalg.batch_matmul"));
    }
}
