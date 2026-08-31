//! CUDA Backend - NVIDIA GPU lowering
//!
//! Generates MLIR with GPU dialect for NVIDIA GPUs.

use super::{TargetBackend, TargetLowering};

/// CUDA backend implementation
pub struct CudaBackend;

impl TargetLowering for CudaBackend {
    fn backend() -> TargetBackend {
        TargetBackend::Cuda
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
            r#"  // CUDA matmul using GPU dialect
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
        Ok(super::conv2d_body(
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
        ))
    }

    fn lower_attention(
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        dtype: &str,
    ) -> Result<String, String> {
        // Not actually flash-attention (no tiling, no online softmax, the full
        // [seq, seq] score matrix is materialized) — a real, complete
        // self-attention nonetheless, unlike the tensor.empty() stub this
        // replaces. Calling it "flash" when it wasn't even computing attention
        // at all was the more misleading of the two.
        Ok(format!(
            "  // Self-attention for CUDA (full softmax, not flash-attention's tiled/online kernel)\n{}",
            super::attention_body(seq_len, hidden_size, num_heads, dtype, "attributes {gpu.kernel}")
        ))
    }

    fn module_attributes() -> String {
        r#"gpu.container_module, gpu.kernel_attr = "ptx""#.to_string()
    }

    fn function_attributes() -> String {
        "gpu.kernel".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_backend() {
        assert_eq!(CudaBackend::backend(), TargetBackend::Cuda);
        assert!(CudaBackend::supported_dtypes().contains(&"f16"));
    }

    #[test]
    fn test_cuda_matmul() {
        let code = CudaBackend::lower_matmul(1, 1024, 1024, 1024, "f16").unwrap();
        assert!(code.contains("gpu.kernel"));
    }

    #[test]
    fn test_cuda_attention() {
        let code = CudaBackend::lower_attention(2048, 8192, 64, "f16").unwrap();
        assert!(code.contains("linalg.softmax"));
    }
}
