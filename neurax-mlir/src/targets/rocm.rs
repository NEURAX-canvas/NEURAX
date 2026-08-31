//! ROCm Backend - AMD GPU via HIP/ROCm
//!
//! Generates MLIR that compiles to AMD GPU via ROCm.
//! Uses IREE for deployment: iree-compile --iree-hal-target-device=rocm

use super::{TargetBackend, TargetLowering};

/// ROCm backend implementation for AMD GPUs
pub struct RocmBackend;

impl TargetLowering for RocmBackend {
    fn backend() -> TargetBackend {
        TargetBackend::Rocm
    }

    fn supported_dtypes() -> &'static [&'static str] {
        &["f32", "f16", "bf16", "f8", "i8", "i32", "i64"]
    }

    fn lower_matmul(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        dtype: &str,
    ) -> Result<String, String> {
        Ok(format!(
            r#"  // ROCm matmul for AMD GPU
  // Uses MFMA (Matrix Fused Multiply Add) instructions on CDNA
  func.func @matmul(%a: tensor<{batch}x{m}x{k}x{dtype}>, %b: tensor<{batch}x{k}x{n}x{dtype}>) -> tensor<{batch}x{m}x{n}x{dtype}> attributes {{gpu.kernel}} {{
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
            "  // ROCm conv2d for AMD GPU\n  // Optimized for MI-series accelerators\n{}",
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
                "attributes {gpu.kernel}",
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
            "  // ROCm attention for AMD GPU\n  // Uses MFMA for attention computation on CDNA architecture\n{}",
            super::attention_body(seq_len, hidden_size, num_heads, dtype, "attributes {gpu.kernel}")
        ))
    }

    fn module_attributes() -> String {
        "gpu.container_module, gpu.kernel_attr = \"hip\"".to_string()
    }

    fn function_attributes() -> String {
        "gpu.kernel".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rocm_backend() {
        assert_eq!(RocmBackend::backend(), TargetBackend::Rocm);
    }

    #[test]
    fn test_rocm_matmul() {
        let code = RocmBackend::lower_matmul(1, 1024, 1024, 1024, "f16").unwrap();
        assert!(code.contains("linalg.batch_matmul"));
        assert!(code.contains("gpu.kernel"));
    }
}
