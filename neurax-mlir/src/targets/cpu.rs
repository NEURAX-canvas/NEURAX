//! CPU Backend - LLVM IR lowering
//!
//! Generates MLIR that compiles to LLVM IR for CPU execution.

use super::{TargetBackend, TargetLowering};

/// CPU backend implementation
pub struct CpuBackend;

impl TargetLowering for CpuBackend {
    fn backend() -> TargetBackend {
        TargetBackend::Cpu
    }

    fn supported_dtypes() -> &'static [&'static str] {
        &["f32", "f64", "i8", "i16", "i32", "i64"]
    }

    fn lower_matmul(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        dtype: &str,
    ) -> Result<String, String> {
        Ok(format!(
            r#"  // CPU matmul using linalg
  func.func @matmul(%a: tensor<{batch}x{m}x{k}x{dtype}>, %b: tensor<{batch}x{k}x{n}x{dtype}>) -> tensor<{batch}x{m}x{n}x{dtype}> attributes {{llvm.readonly}} {{
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
            "  // CPU conv2d using linalg\n{}",
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
                "attributes {llvm.readonly}",
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
            "  // CPU attention using linalg\n{}",
            super::attention_body(
                seq_len,
                hidden_size,
                num_heads,
                dtype,
                "attributes {llvm.readonly}"
            )
        ))
    }

    fn module_attributes() -> String {
        "llvm.target_triple".to_string()
    }

    fn function_attributes() -> String {
        "llvm.readonly".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend() {
        assert_eq!(CpuBackend::backend(), TargetBackend::Cpu);
        assert!(CpuBackend::supported_dtypes().contains(&"f32"));
    }

    #[test]
    fn test_cpu_matmul() {
        let code = CpuBackend::lower_matmul(1, 1024, 1024, 1024, "f32").unwrap();
        assert!(code.contains("linalg.batch_matmul"));
    }
}
