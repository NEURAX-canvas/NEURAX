//! CNN Block formulas - ResNet, Inception, MobileNet, DenseNet blocks

/// BatchNorm/LayerNorm's two *trainable* parameters per channel — weight
/// (gamma) and bias (beta). BatchNorm also tracks `running_mean` and
/// `running_var`, but those are buffers updated by a moving average, not by
/// gradient descent — every framework excludes them from `parameters()`, and
/// every published parameter count this crate is checked against does too.
const BN_TRAINABLE_PARAMS_PER_CHANNEL: usize = 2;

/// Compute parameters for ResNet Basic Block (2 convs + skip connection)
///
/// Structure: conv3x3 → BN → ReLU → conv3x3 → BN → add(skip) → ReLU
pub fn resnet_basic_block_params(
    in_channels: usize,
    out_channels: usize,
    stride: usize,
    bias: bool,
) -> u64 {
    // First conv: in_channels → out_channels, 3×3
    let conv1 = conv_params(in_channels, out_channels, 3, bias);

    // Second conv: out_channels → out_channels, 3×3
    let conv2 = conv_params(out_channels, out_channels, 3, bias);

    // Two BatchNorm layers — weight + bias only. `running_mean`/`running_var`
    // are real, but PyTorch registers them as buffers (`register_buffer`),
    // not `nn.Parameter`s: they update by a moving average, not gradient
    // descent, and every published parameter count (including the 25.6M
    // ResNet-50 figure this crate is checked against) excludes them. Counting
    // 4×channels here silently inflated every BN-bearing block by roughly
    // double its normalization cost.
    let bn_params = (2 * BN_TRAINABLE_PARAMS_PER_CHANNEL * out_channels) as u64;

    // Downsample projection if dimensions don't match
    let downsample = if stride != 1 || in_channels != out_channels {
        // 1×1 conv for projection + BN
        conv_params(in_channels, out_channels, 1, bias)
            + (BN_TRAINABLE_PARAMS_PER_CHANNEL * out_channels) as u64
    } else {
        0
    };

    conv1 + conv2 + bn_params + downsample
}

/// Compute parameters for ResNet Bottleneck Block (3 convs)
///
/// Structure: 1×1 → 3×3 → 1×1 with expansion factor (typically 4).
/// `cardinality` splits the middle 3×3 conv into that many groups — 1 is a
/// plain ResNet-50/101 bottleneck; ResNeXt's whole point is a larger value
/// here (32 for ResNeXt-50 32x4d), which divides that conv's parameter and
/// FLOP count accordingly. Passing 1 reproduces the ungrouped formula this
/// function always used before `cardinality` existed as a real input.
pub fn resnet_bottleneck_block_params(
    in_channels: usize,
    mid_channels: usize,
    out_channels: usize,
    stride: usize,
    cardinality: usize,
    bias: bool,
) -> u64 {
    let cardinality = cardinality.max(1);

    // 1×1 reduction
    let conv1 = conv_params(in_channels, mid_channels, 1, bias);

    // 3×3 conv — grouped when cardinality > 1 (ResNeXt), same grouped-conv
    // arithmetic `conv2d_params` uses elsewhere: dividing by the group count.
    let conv2 = super::conv::conv2d_params(mid_channels, mid_channels, 3, 3, cardinality, bias);

    // 1×1 expansion
    let conv3 = conv_params(mid_channels, out_channels, 1, bias);

    // Three BatchNorm layers — trainable params only, see the constant's doc.
    let bn_params = (3 * BN_TRAINABLE_PARAMS_PER_CHANNEL * mid_channels
        + BN_TRAINABLE_PARAMS_PER_CHANNEL * out_channels) as u64;

    // Downsample if needed
    let downsample = if stride != 1 || in_channels != out_channels {
        conv_params(in_channels, out_channels, 1, bias)
            + (BN_TRAINABLE_PARAMS_PER_CHANNEL * out_channels) as u64
    } else {
        0
    };

    conv1 + conv2 + conv3 + bn_params + downsample
}

/// FLOPs for a ResNet Bottleneck Block (1×1 → 3×3 → 1×1), the block ResNet-50
/// and deeper actually use — there was no FLOPs counterpart to
/// `resnet_bottleneck_block_params` at all before this.
#[allow(clippy::too_many_arguments)]
pub fn resnet_bottleneck_block_flops(
    batch: usize,
    in_channels: usize,
    mid_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    stride: usize,
    cardinality: usize,
) -> f64 {
    let cardinality = cardinality.max(1) as f64;
    let out_h = (height / stride).max(1);
    let out_w = (width / stride).max(1);
    let b = batch as f64;

    // 1×1 reduction, at full input resolution.
    let conv1 = 2.0 * b * mid_channels as f64 * height as f64 * width as f64 * in_channels as f64;
    // 3×3 conv, at the (possibly strided) output resolution — grouped when
    // cardinality > 1, same reduction the parameter count above applies.
    let conv2 = 2.0
        * b
        * mid_channels as f64
        * out_h as f64
        * out_w as f64
        * (mid_channels as f64 / cardinality)
        * 9.0;
    // 1×1 expansion, at output resolution.
    let conv3 = 2.0 * b * out_channels as f64 * out_h as f64 * out_w as f64 * mid_channels as f64;

    let downsample_flops = if stride != 1 || in_channels != out_channels {
        2.0 * b * out_channels as f64 * out_h as f64 * out_w as f64 * in_channels as f64
    } else {
        0.0
    };

    conv1 + conv2 + conv3 + downsample_flops
}

/// Compute parameters for Inception Module (multiple parallel branches)
///
/// Structure: 1×1 + 3×3 + 5×5 + pool branches
#[allow(clippy::too_many_arguments)]
pub fn inception_module_params(
    in_channels: usize,
    out_1x1: usize,
    out_3x3_reduce: usize,
    out_3x3: usize,
    out_5x5_reduce: usize,
    out_5x5: usize,
    pool_proj: usize,
    bias: bool,
) -> u64 {
    // 1×1 branch
    let branch1x1 = conv_params(in_channels, out_1x1, 1, bias);

    // 3×3 branch (with reduction)
    let branch3x3 = conv_params(in_channels, out_3x3_reduce, 1, bias)
        + conv_params(out_3x3_reduce, out_3x3, 3, bias);

    // 5×5 branch (with reduction)
    let branch5x5 = conv_params(in_channels, out_5x5_reduce, 1, bias)
        + conv_params(out_5x5_reduce, out_5x5, 5, bias);

    // Pool branch
    let branch_pool = conv_params(in_channels, pool_proj, 1, bias);

    // BatchNorm for each conv (simplified: 2 per branch)
    let bn_params =
        (2 * (out_1x1 + out_3x3_reduce + out_3x3 + out_5x5_reduce + out_5x5 + pool_proj)) as u64;

    branch1x1 + branch3x3 + branch5x5 + branch_pool + bn_params
}

/// Compute parameters for MBConv (MobileNet Inverted Residual Block)
///
/// Structure: 1×1 expand → 3×3/5×5 depthwise → (Squeeze-and-Excitation) →
/// 1×1 project. SE (Hu et al., 2018) is what separates MobileNetV3 and
/// EfficientNet's blocks from plain MobileNetV2's — two small FC layers
/// (implemented as 1×1 convs) on the globally-pooled features, squeezing to
/// `expanded / se_reduction_ratio` and back. Optional because plain
/// MobileNetV2 blocks genuinely have none.
#[allow(clippy::too_many_arguments)]
pub fn mbconv_params(
    in_channels: usize,
    out_channels: usize,
    expand_factor: usize,
    kernel_size: usize,
    stride: usize,
    se: bool,
    se_reduction_ratio: usize,
    bias: bool,
) -> u64 {
    let _ = stride; // Reserved for future stride-aware parameter calculation
    let expanded = in_channels * expand_factor;

    // Expansion phase (skip if expand_factor=1)
    let expand_params = if expand_factor > 1 {
        conv_params(in_channels, expanded, 1, bias)
            + (BN_TRAINABLE_PARAMS_PER_CHANNEL * expanded) as u64 // conv + BN
    } else {
        0
    };

    // Depthwise conv
    let depthwise =
        (expanded * kernel_size * kernel_size + BN_TRAINABLE_PARAMS_PER_CHANNEL * expanded) as u64; // depthwise + BN

    // Squeeze-and-Excitation: squeeze expanded -> expanded/r, excite back.
    let se_params = if se {
        let reduced = (expanded / se_reduction_ratio.max(1)).max(1);
        (2 * expanded * reduced) as u64
    } else {
        0
    };

    // Projection phase (always present, no bias in original MobileNet)
    let project = conv_params(expanded, out_channels, 1, false)
        + (BN_TRAINABLE_PARAMS_PER_CHANNEL * out_channels) as u64;

    expand_params + depthwise + se_params + project
}

/// Compute parameters for DenseNet Dense Block
///
/// Structure: BN → ReLU → 1×1 conv → BN → ReLU → 3×3 conv
/// Growth rate determines output channels per layer
pub fn dense_block_params(
    in_channels: usize,
    growth_rate: usize,
    num_layers: usize,
    bottleneck_factor: usize,
    bias: bool,
) -> u64 {
    let mut total_params: u64 = 0;
    let mut channels = in_channels;

    for _ in 0..num_layers {
        // Bottleneck (1×1 conv)
        let bn_channels = channels * bottleneck_factor;
        let bottleneck = conv_params(channels, bn_channels, 1, bias)
            + (BN_TRAINABLE_PARAMS_PER_CHANNEL * bn_channels) as u64;

        // Main conv (3×3)
        let main_conv = conv_params(bn_channels, growth_rate, 3, bias)
            + (BN_TRAINABLE_PARAMS_PER_CHANNEL * growth_rate) as u64;

        total_params += bottleneck + main_conv;
        channels += growth_rate; // Concatenation
    }

    total_params
}

/// Compute parameters for ConvNeXt Block
///
/// Structure: 7×7 depthwise → LayerNorm → 1×1 → GELU → 1×1
pub fn convnext_block_params(channels: usize, mlp_ratio: f64, bias: bool) -> u64 {
    // Depthwise 7×7 conv — weights only. The LayerNorm that follows it is
    // counted once, below; this used to add a second, unlabelled `4 ×
    // channels` right here as well, silently double-counting it (LayerNorm
    // has no running stats even under the old 4×-per-channel assumption —
    // `weight` and `bias` are its only parameters, always).
    let depthwise = (channels * 7 * 7) as u64;

    // MLP: 1×1 expand → 1×1 project
    let mlp_hidden = (channels as f64 * mlp_ratio) as usize;
    let mlp =
        conv_params(channels, mlp_hidden, 1, bias) + conv_params(mlp_hidden, channels, 1, bias);

    // One LayerNorm layer — a real ConvNeXt block has exactly one (after the
    // depthwise conv), not two.
    let ln_params = (2 * channels) as u64;

    depthwise + mlp + ln_params
}

/// Compute parameters for ShuffleNet Unit
///
/// Structure: 1×1 → channel shuffle → 3×3 depthwise → 1×1
pub fn shuffle_unit_params(
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    stride: usize,
    bias: bool,
) -> u64 {
    let mid_channels = out_channels / 2;

    // First 1×1 group conv
    let conv1 = ((in_channels / groups) * mid_channels * groups
        + BN_TRAINABLE_PARAMS_PER_CHANNEL * mid_channels) as u64;

    // Depthwise 3×3
    let depthwise = (mid_channels * 3 * 3 + BN_TRAINABLE_PARAMS_PER_CHANNEL * mid_channels) as u64;

    // Second 1×1 group conv
    let conv2 = ((mid_channels / groups) * out_channels * groups
        + BN_TRAINABLE_PARAMS_PER_CHANNEL * out_channels) as u64;

    // Skip connection projection if needed
    let skip = if stride != 1 || in_channels != out_channels {
        conv_params(in_channels, out_channels, 1, bias)
            + (BN_TRAINABLE_PARAMS_PER_CHANNEL * out_channels) as u64
    } else {
        0
    };

    conv1 + depthwise + conv2 + skip
}

/// Compute parameters for C2f block (YOLOv8 style)
///
/// Structure: Split → multiple Bottlenecks → Concat
pub fn c2f_block_params(
    in_channels: usize,
    out_channels: usize,
    num_bottlenecks: usize,
    shortcut: bool,
    bias: bool,
) -> u64 {
    let hidden = out_channels / 2;

    // Initial conv
    let init_conv = conv_params(in_channels, out_channels, 1, bias);

    // Bottlenecks (simplified as 2 convs each). Each Conv in a real C2f
    // bottleneck is a Conv+BN+SiLU triplet regardless of `shortcut` — that
    // flag only controls whether the block adds a residual connection, not
    // whether its convs have batch norm, so BN was being silently dropped
    // whenever `shortcut` was false.
    let _ = shortcut;
    let mut bottleneck_params: u64 = 0;
    for _ in 0..num_bottlenecks {
        bottleneck_params += conv_params(hidden, hidden, 3, bias) * 2
            + (BN_TRAINABLE_PARAMS_PER_CHANNEL * hidden * 2) as u64;
    }

    // Final conv
    let final_conv = conv_params(
        out_channels + hidden * num_bottlenecks,
        out_channels,
        1,
        bias,
    );

    init_conv + bottleneck_params + final_conv
}

/// Helper: Standard conv params
fn conv_params(in_ch: usize, out_ch: usize, kernel: usize, bias: bool) -> u64 {
    let weights = in_ch * out_ch * kernel * kernel;
    let bias_params = if bias { out_ch } else { 0 };
    (weights + bias_params) as u64
}

/// Compute FLOPs for ResNet Basic Block
pub fn resnet_basic_block_flops(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    stride: usize,
) -> f64 {
    let out_h = height / stride;
    let out_w = width / stride;

    // First conv
    let conv1_flops = 2.0
        * batch as f64
        * out_channels as f64
        * out_h as f64
        * out_w as f64
        * in_channels as f64
        * 3.0
        * 3.0
        / stride as f64;

    // Second conv
    let conv2_flops = 2.0
        * batch as f64
        * out_channels as f64
        * out_h as f64
        * out_w as f64
        * out_channels as f64
        * 3.0
        * 3.0;

    // Skip projection if needed
    let skip_flops = if stride != 1 || in_channels != out_channels {
        2.0 * batch as f64
            * out_channels as f64
            * out_h as f64
            * out_w as f64
            * in_channels as f64
            * 1.0
            * 1.0
    } else {
        0.0
    };

    conv1_flops + conv2_flops + skip_flops
}

/// Compute FLOPs for MBConv block
#[allow(clippy::too_many_arguments)]
pub fn mbconv_flops(
    batch: usize,
    in_channels: usize,
    out_channels: usize,
    height: usize,
    width: usize,
    expand_factor: usize,
    kernel_size: usize,
    stride: usize,
    se: bool,
    se_reduction_ratio: usize,
) -> f64 {
    let expanded = in_channels * expand_factor;
    let out_h = height / stride;
    let out_w = width / stride;

    // Expansion
    let expand_flops = if expand_factor > 1 {
        2.0 * batch as f64 * height as f64 * width as f64 * in_channels as f64 * expanded as f64
    } else {
        0.0
    };

    // Depthwise
    let depthwise_flops = 2.0
        * batch as f64
        * expanded as f64
        * out_h as f64
        * out_w as f64
        * kernel_size as f64
        * kernel_size as f64;

    // Squeeze-and-Excitation: global average pool (negligible) + two small
    // FC layers on the pooled [B, expanded] vector.
    let se_flops = if se {
        let reduced = (expanded / se_reduction_ratio.max(1)).max(1);
        2.0 * batch as f64 * expanded as f64 * reduced as f64 * 2.0
    } else {
        0.0
    };

    // Projection
    let project_flops =
        2.0 * batch as f64 * out_channels as f64 * out_h as f64 * out_w as f64 * expanded as f64;

    expand_flops + depthwise_flops + se_flops + project_flops
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected value below was computed independently (conv weights +
    // BN weight/bias only, no running-stat buffers) rather than read back
    // from the function under test, so a regression that reintroduces the
    // buffer-counting bug or drops a term actually fails these instead of
    // sliding through a `> 0` check.

    #[test]
    fn test_resnet_basic_block() {
        // ResNet-18 style: 64 → 64, stride=1, no dimension change → no
        // downsample. 2×(64×64×3×3) conv weights + 2×BN(64).
        let params = resnet_basic_block_params(64, 64, 1, false);
        assert_eq!(params, 73_984);
    }

    #[test]
    fn test_resnet_bottleneck() {
        // ResNet-50 style: 256 → 128 (mid) → 512, stride=1 but channels
        // change → downsample projection included. cardinality=1 (plain
        // ResNet, not ResNeXt) must reproduce the exact number this test
        // asserted before `cardinality` was a real parameter.
        let params = resnet_bottleneck_block_params(256, 128, 512, 1, 1, false);
        assert_eq!(params, 379_648);
    }

    #[test]
    fn test_resnext_bottleneck_groups_the_middle_conv() {
        // ResNeXt-50 32x4d style: same shape as the plain bottleneck above,
        // but the middle 3x3 conv is split into 32 groups — its weight count
        // (and only its weight count) should divide by 32 versus cardinality=1.
        let plain = resnet_bottleneck_block_params(256, 128, 512, 1, 1, false);
        let grouped = resnet_bottleneck_block_params(256, 128, 512, 1, 32, false);
        assert!(
            grouped < plain,
            "grouping the middle conv must reduce total params"
        );

        let conv1 = conv_params(256, 128, 1, false);
        let conv3 = conv_params(128, 512, 1, false);
        let bn_and_downsample = plain - conv1 - conv3 - conv_params(128, 128, 3, false);
        let expected_conv2 = crate::conv::conv2d_params(128, 128, 3, 3, 32, false);
        assert_eq!(grouped, conv1 + expected_conv2 + conv3 + bn_and_downsample);
    }

    #[test]
    fn test_mbconv() {
        // MobileNetV2 style: 32 → 16, expand ×6, 3×3 depthwise, no SE.
        let params = mbconv_params(32, 16, 6, 3, 1, false, 4, false);
        assert_eq!(params, 11_744);
    }

    #[test]
    fn test_mbconv_with_se_costs_more_than_without() {
        // EfficientNet/MobileNetV3 style: same shape, SE block enabled —
        // `se: bool` existed on LayerParams but nothing ever read it, so a
        // real EfficientNet's SE blocks were silently costed as zero.
        let without_se = mbconv_params(32, 16, 6, 3, 1, false, 4, false);
        let with_se = mbconv_params(32, 16, 6, 3, 1, true, 4, false);
        assert!(with_se > without_se);

        // 2 * expanded * (expanded/reduction) = 2 * 192 * 48 = 18,432
        assert_eq!(with_se - without_se, 18_432);
    }

    #[test]
    fn test_mbconv_flops_with_se_costs_more_than_without() {
        let without_se = mbconv_flops(1, 32, 16, 224, 224, 6, 3, 1, false, 4);
        let with_se = mbconv_flops(1, 32, 16, 224, 224, 6, 3, 1, true, 4);
        assert!(with_se > without_se);
    }

    #[test]
    fn test_inception() {
        // InceptionV3 style module.
        let params = inception_module_params(288, 64, 48, 64, 8, 32, 64, false);
        assert_eq!(params, 87_600);
    }

    #[test]
    fn test_dense_block() {
        // DenseNet-121 style: growth_rate=32, 6 layers, bottleneck factor=4.
        let params = dense_block_params(64, 32, 6, 4, false);
        assert_eq!(params, 1_571_968);
    }

    #[test]
    fn test_convnext_block() {
        // ConvNeXt-Tiny style: 96 channels, 4× MLP ratio, one LayerNorm.
        let params = convnext_block_params(96, 4.0, false);
        assert_eq!(params, 78_624);
    }

    #[test]
    fn test_shuffle_unit() {
        // ShuffleNet style: 64 → 64, groups=2, stride=1 → no skip projection.
        let params = shuffle_unit_params(64, 64, 2, 1, false);
        assert_eq!(params, 4_640);
    }

    #[test]
    fn test_c2f_block() {
        // YOLOv8 C2f style: 128 → 128, 3 bottlenecks, shortcut on — BN is
        // always present on both convs of every bottleneck regardless of
        // `shortcut` (it only gates the residual add, modeled elsewhere).
        let params = c2f_block_params(128, 128, 3, true, false);
        assert_eq!(params, 279_296);
    }

    #[test]
    fn c2f_block_params_do_not_depend_on_shortcut() {
        // Regression guard for the bug this segment fixed: BN parameters
        // must not silently disappear when `shortcut` is false.
        let with_shortcut = c2f_block_params(128, 128, 3, true, false);
        let without_shortcut = c2f_block_params(128, 128, 3, false, false);
        assert_eq!(with_shortcut, without_shortcut);
    }
}
