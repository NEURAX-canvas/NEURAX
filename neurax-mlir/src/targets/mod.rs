//! Target Backend Trait - Interface for target-specific code generation

mod cpu;
mod cuda;
mod metal;
mod rocm;
mod vulkan;

pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use metal::MetalBackend;
pub use rocm::RocmBackend;
pub use vulkan::VulkanBackend;

/// Supported target backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetBackend {
    /// CPU via LLVM IR
    Cpu,
    /// NVIDIA GPU via CUDA
    Cuda,
    /// Vulkan via SPIR-V
    Vulkan,
    /// Apple Metal
    Metal,
    /// AMD ROCm
    Rocm,
}

impl TargetBackend {
    /// Get the string name of the backend
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Rocm => "rocm",
        }
    }

    /// Parse from string.
    ///
    /// Not `std::str::FromStr`: this returns `Option`, not `Result`, and
    /// accepts vendor aliases (`"nvidia"`, `"apple"`, ...) a real `FromStr`
    /// impl would need its own `Err` type to reject cleanly — kept as a
    /// plain inherent method instead of forcing that out for a type with
    /// exactly this one caller shape.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" | "llvm" => Some(Self::Cpu),
            "cuda" | "nvidia" | "gpu" => Some(Self::Cuda),
            "vulkan" | "spirv" => Some(Self::Vulkan),
            "metal" | "apple" => Some(Self::Metal),
            "rocm" | "amd" => Some(Self::Rocm),
            _ => None,
        }
    }

    /// Get all supported backends
    pub fn all() -> &'static [TargetBackend] {
        &[Self::Cpu, Self::Cuda, Self::Vulkan, Self::Metal, Self::Rocm]
    }

    /// Check if this is a GPU backend
    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda | Self::Vulkan | Self::Metal | Self::Rocm)
    }

    /// Get the IREE target device name
    pub fn iree_target(&self) -> &'static str {
        match self {
            Self::Cpu => "local-task",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Rocm => "rocm",
        }
    }

    /// Get the IREE backend flag
    pub fn iree_backend_flag(&self) -> &'static str {
        match self {
            Self::Cpu => "llvm-cpu",
            Self::Cuda => "cuda",
            Self::Vulkan => "vulkan",
            Self::Metal => "metal",
            Self::Rocm => "rocm",
        }
    }
}

/// Trait for target-specific lowering strategies
pub trait TargetLowering {
    /// Get the target backend this lowering supports
    fn backend() -> TargetBackend;

    /// Get supported data types for this target
    fn supported_dtypes() -> &'static [&'static str];

    /// Lower a tensor operation to target-specific MLIR
    ///
    /// Returns the lowered operation as MLIR text
    fn lower_matmul(
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
        dtype: &str,
    ) -> Result<String, String>;

    /// Lower a convolution operation
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<String, String>;

    /// Lower an attention operation
    fn lower_attention(
        seq_len: usize,
        hidden_size: usize,
        num_heads: usize,
        dtype: &str,
    ) -> Result<String, String>;

    /// Get target-specific module attributes
    fn module_attributes() -> String {
        String::new()
    }

    /// Get target-specific function attributes
    fn function_attributes() -> String {
        String::new()
    }
}

/// Shared Conv2D lowering body for every backend: pads the input with
/// `tensor.pad` when `padding > 0` (untested until now — no backend modeled
/// padding at all — verified against `mlir-opt` directly before landing here),
/// then a `linalg.conv_2d_nhwc_hwcf` with explicit `strides`/`dilations`. Only
/// each backend's own function attributes (`gpu.kernel`, `llvm.readonly`...)
/// differ between callers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_body(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dtype: &str,
    func_attrs: &str,
) -> String {
    let padded_h = height + 2 * padding;
    let padded_w = width + 2 * padding;
    let stride = stride.max(1);
    let out_h = padded_h.saturating_sub(kernel_size) / stride + 1;
    let out_w = padded_w.saturating_sub(kernel_size) / stride + 1;
    let zero = if dtype.starts_with('f') { "0.0" } else { "0" };

    let (input_operand, pad_block) = if padding > 0 {
        (
            "%padded",
            format!(
                "    %cst = arith.constant {zero} : {dtype}\n\
                 \x20   %padded = tensor.pad %input low[0, {padding}, {padding}, 0] high[0, {padding}, {padding}, 0] {{\n\
                 \x20   ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):\n\
                 \x20     tensor.yield %cst : {dtype}\n\
                 \x20   }} : tensor<{batch}x{height}x{width}x{in_channels}x{dtype}> to tensor<{batch}x{padded_h}x{padded_w}x{in_channels}x{dtype}>\n"
            ),
        )
    } else {
        ("%input", String::new())
    };

    format!(
        "  func.func @conv2d(%input: tensor<{batch}x{height}x{width}x{in_channels}x{dtype}>, %filter: tensor<{kernel_size}x{kernel_size}x{in_channels}x{out_channels}x{dtype}>) -> tensor<{batch}x{out_h}x{out_w}x{out_channels}x{dtype}> {func_attrs} {{\n\
         {pad_block}\
         \x20   %output_init = tensor.empty() : tensor<{batch}x{out_h}x{out_w}x{out_channels}x{dtype}>\n\
         \x20   %output = linalg.conv_2d_nhwc_hwcf {{dilations = dense<1> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins({input_operand}, %filter : tensor<{batch}x{padded_h}x{padded_w}x{in_channels}x{dtype}>, tensor<{kernel_size}x{kernel_size}x{in_channels}x{out_channels}x{dtype}>) outs(%output_init : tensor<{batch}x{out_h}x{out_w}x{out_channels}x{dtype}>) -> tensor<{batch}x{out_h}x{out_w}x{out_channels}x{dtype}>\n\
         \x20   return %output : tensor<{batch}x{out_h}x{out_w}x{out_channels}x{dtype}>\n\
         \x20 }}\n"
    )
}

/// Shared self-attention lowering body for every backend: real Q/K/V
/// projections, `QKᵀ` via `linalg.transpose` + `linalg.matmul`, scaling by
/// `1/sqrt(head_dim)`, a real `linalg.softmax`, then `softmax·V` and the
/// output projection — verified against `mlir-opt` directly before landing
/// here. Every backend's own `lower_attention()` used to stop after computing
/// Q/K/V and return `tensor.empty()`: a hollow stub that never touched
/// `QKᵀ`, softmax, or `V` at all, on every one of the 22 real attention layers
/// this tool checks across GPT-3/Mixtral/SDXL.
///
/// Single-head shape (`[seq, hidden]`, no per-head split) — the same
/// abstraction level the old stub already used; splitting into real per-head
/// tensors is further work, not part of this fix.
pub(crate) fn attention_body(
    seq_len: usize,
    hidden_size: usize,
    num_heads: usize,
    dtype: &str,
    func_attrs: &str,
) -> String {
    let head_dim = (hidden_size / num_heads.max(1)).max(1);
    let scale = 1.0 / (head_dim as f64).sqrt();

    format!(
        "  func.func @attention(%input: tensor<{seq_len}x{hidden_size}x{dtype}>, %wq: tensor<{hidden_size}x{hidden_size}x{dtype}>, %wk: tensor<{hidden_size}x{hidden_size}x{dtype}>, %wv: tensor<{hidden_size}x{hidden_size}x{dtype}>, %wo: tensor<{hidden_size}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}> {func_attrs} {{\n\
         \x20   %q_init = tensor.empty() : tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %q = linalg.matmul ins(%input, %wq : tensor<{seq_len}x{hidden_size}x{dtype}>, tensor<{hidden_size}x{hidden_size}x{dtype}>) outs(%q_init : tensor<{seq_len}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %k_init = tensor.empty() : tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %k = linalg.matmul ins(%input, %wk : tensor<{seq_len}x{hidden_size}x{dtype}>, tensor<{hidden_size}x{hidden_size}x{dtype}>) outs(%k_init : tensor<{seq_len}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %v_init = tensor.empty() : tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %v = linalg.matmul ins(%input, %wv : tensor<{seq_len}x{hidden_size}x{dtype}>, tensor<{hidden_size}x{hidden_size}x{dtype}>) outs(%v_init : tensor<{seq_len}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %kt_init = tensor.empty() : tensor<{hidden_size}x{seq_len}x{dtype}>\n\
         \x20   %kt = linalg.transpose ins(%k : tensor<{seq_len}x{hidden_size}x{dtype}>) outs(%kt_init : tensor<{hidden_size}x{seq_len}x{dtype}>) permutation = [1, 0]\n\
         \x20   %scores_init = tensor.empty() : tensor<{seq_len}x{seq_len}x{dtype}>\n\
         \x20   %scores = linalg.matmul ins(%q, %kt : tensor<{seq_len}x{hidden_size}x{dtype}>, tensor<{hidden_size}x{seq_len}x{dtype}>) outs(%scores_init : tensor<{seq_len}x{seq_len}x{dtype}>) -> tensor<{seq_len}x{seq_len}x{dtype}>\n\
         \x20   %scale = arith.constant {scale:.10} : {dtype}\n\
         \x20   %scaled_init = tensor.empty() : tensor<{seq_len}x{seq_len}x{dtype}>\n\
         \x20   %scaled = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [\"parallel\", \"parallel\"]}} ins(%scores : tensor<{seq_len}x{seq_len}x{dtype}>) outs(%scaled_init : tensor<{seq_len}x{seq_len}x{dtype}>) {{\n\
         \x20   ^bb0(%s_in: {dtype}, %s_out: {dtype}):\n\
         \x20     %s_scaled = arith.mulf %s_in, %scale : {dtype}\n\
         \x20     linalg.yield %s_scaled : {dtype}\n\
         \x20   }} -> tensor<{seq_len}x{seq_len}x{dtype}>\n\
         \x20   %softmax_init = tensor.empty() : tensor<{seq_len}x{seq_len}x{dtype}>\n\
         \x20   %attn = linalg.softmax dimension(1) ins(%scaled : tensor<{seq_len}x{seq_len}x{dtype}>) outs(%softmax_init : tensor<{seq_len}x{seq_len}x{dtype}>) -> tensor<{seq_len}x{seq_len}x{dtype}>\n\
         \x20   %ctx_init = tensor.empty() : tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %ctx = linalg.matmul ins(%attn, %v : tensor<{seq_len}x{seq_len}x{dtype}>, tensor<{seq_len}x{hidden_size}x{dtype}>) outs(%ctx_init : tensor<{seq_len}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %out_init = tensor.empty() : tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   %output = linalg.matmul ins(%ctx, %wo : tensor<{seq_len}x{hidden_size}x{dtype}>, tensor<{hidden_size}x{hidden_size}x{dtype}>) outs(%out_init : tensor<{seq_len}x{hidden_size}x{dtype}>) -> tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20   return %output : tensor<{seq_len}x{hidden_size}x{dtype}>\n\
         \x20 }}\n"
    )
}

impl std::fmt::Display for TargetBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
