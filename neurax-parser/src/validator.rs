//! JSON schema validation

use crate::error::ParserError;
use crate::model_config::{LayerType, ModelConfig};

/// Validate the parsed model configuration
pub fn validate_model_config(config: &ModelConfig) -> Result<(), ParserError> {
    // Check for at least one layer
    if config.model.layers.is_empty() {
        return Err(ParserError::schema_validation(
            "Model must have at least one layer",
        ));
    }

    // Validate batch size
    if config.training.batch_size == 0 {
        return Err(ParserError::InvalidValue {
            field: "batch_size".to_string(),
            reason: "batch_size must be greater than 0".to_string(),
        });
    }

    // Validate layer shapes are consistent
    validate_layer_shapes(config)?;

    // Validate layer parameters
    validate_layer_params(config)?;

    // Validate hardware
    if config.hardware.gpus.is_empty() {
        return Err(ParserError::schema_validation(
            "At least one GPU must be specified",
        ));
    }

    Ok(())
}

/// Validate that layer shapes are consistent
fn validate_layer_shapes(config: &ModelConfig) -> Result<(), ParserError> {
    let layers = &config.model.layers;

    for (i, layer) in layers.iter().enumerate() {
        // Check input/output shapes are non-empty
        if layer.input_shape.is_empty() || layer.output_shape.is_empty() {
            continue; // Skip validation for layers with unspecified shapes
        }

        // Check shape consistency between consecutive layers
        if i > 0 {
            let prev_layer = &layers[i - 1];
            if !prev_layer.output_shape.is_empty() {
                // Output of previous should match input of current (for sequential models)
                // For non-sequential models, this check is relaxed
                if prev_layer.output_shape.len() == layer.input_shape.len() {
                    // Allow for batch dimension flexibility
                    let shape_match = prev_layer
                        .output_shape
                        .iter()
                        .skip(1) // Skip batch dim
                        .zip(layer.input_shape.iter().skip(1))
                        .all(|(a, b)| a == b);

                    if !shape_match {
                        // This is a warning, not an error for non-sequential models
                        tracing::warn!(
                            "Layer {} input shape {:?} may not match previous layer output shape {:?}",
                            layer.id, layer.input_shape, prev_layer.output_shape
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

/// Upper bound for any single model dimension (hidden size, sequence length,
/// vocabulary, ...). Dimensions are multiplied together when counting
/// parameters and FLOPs; without a ceiling those products silently wrap around
/// and the report shows confidently wrong numbers instead of an error.
///
/// 2^24 (16,777,216) is several orders of magnitude above any published
/// architecture while leaving ample headroom in the u64 accumulators.
const MAX_DIMENSION: usize = 1 << 24;

/// Upper bound for the model-wide layer count, which multiplies the per-layer
/// parameter and FLOP totals. No published architecture approaches this.
const MAX_LAYERS: u64 = 1 << 20;

/// Reject a dimension large enough to overflow downstream math.
///
/// Zero is deliberately allowed: it is the "not applicable" sentinel for
/// dimensions that do not exist in every family — vision models legitimately
/// carry `vocab_size: 0`. Zero is only rejected where it would be a divisor,
/// which [`validate_attention_heads`] handles.
fn check_dimension(value: usize, field: String) -> Result<(), ParserError> {
    if value > MAX_DIMENSION {
        return Err(ParserError::InvalidValue {
            field,
            reason: format!(
                "{} exceeds the maximum supported dimension of {}; \
                 larger values overflow parameter and FLOP accounting",
                value, MAX_DIMENSION
            ),
        });
    }
    Ok(())
}

/// Bound the model-wide dimensions, which are used as fallbacks whenever a
/// layer omits its own value.
fn validate_global_dimensions(config: &ModelConfig) -> Result<(), ParserError> {
    let g = &config.model.global_params;
    let fields = [
        (g.sequence_length, "sequence_length"),
        (g.vocab_size, "vocab_size"),
        (g.embedding_dim, "embedding_dim"),
        (g.rnn_hidden_size, "rnn_hidden_size"),
        (g.cross_attention_dim, "cross_attention_dim"),
        (g.attention_head_dim, "attention_head_dim"),
    ];
    for (value, name) in fields {
        if let Some(value) = value {
            check_dimension(value, format!("global_params.{}", name))?;
        }
    }

    if let Some(sequence_length) = config.training.sequence_length {
        check_dimension(sequence_length, "training.sequence_length".to_string())?;
    }

    for (value, name) in [
        (g.num_layers, "num_layers"),
        (g.num_dense_layers, "num_dense_layers"),
    ] {
        if let Some(value) = value {
            if value > MAX_LAYERS {
                return Err(ParserError::InvalidValue {
                    field: format!("global_params.{}", name),
                    reason: format!(
                        "{} exceeds the maximum supported layer count of {}",
                        value, MAX_LAYERS
                    ),
                });
            }
        }
    }

    Ok(())
}

/// Validate attention head counts.
///
/// `head_dim` is computed as `hidden_size / num_heads` in the operator and
/// formula crates, so a zero head count is a division by zero and a head count
/// that does not divide the hidden size silently truncates the head dimension.
fn validate_attention_heads(layer: &crate::model_config::Layer) -> Result<(), ParserError> {
    let Some(num_heads) = layer.params.num_heads else {
        return Ok(());
    };

    if num_heads == 0 {
        return Err(ParserError::InvalidValue {
            field: format!("layers.{}.params.num_heads", layer.id),
            reason: "num_heads must be greater than 0".to_string(),
        });
    }
    check_dimension(num_heads, format!("layers.{}.params.num_heads", layer.id))?;

    if let Some(hidden_size) = layer.params.hidden_size {
        if hidden_size % num_heads != 0 {
            return Err(ParserError::InvalidValue {
                field: format!("layers.{}.params.num_heads", layer.id),
                reason: format!(
                    "hidden_size ({}) must be divisible by num_heads ({})",
                    hidden_size, num_heads
                ),
            });
        }
    }

    if let Some(num_kv_heads) = layer.params.num_kv_heads {
        if num_kv_heads == 0 {
            return Err(ParserError::InvalidValue {
                field: format!("layers.{}.params.num_kv_heads", layer.id),
                reason: "num_kv_heads must be greater than 0".to_string(),
            });
        }
        if num_kv_heads > num_heads {
            return Err(ParserError::InvalidValue {
                field: format!("layers.{}.params.num_kv_heads", layer.id),
                reason: format!(
                    "num_kv_heads ({}) cannot exceed num_heads ({})",
                    num_kv_heads, num_heads
                ),
            });
        }
        if num_heads % num_kv_heads != 0 {
            return Err(ParserError::InvalidValue {
                field: format!("layers.{}.params.num_kv_heads", layer.id),
                reason: format!(
                    "num_heads ({}) must be divisible by num_kv_heads ({})",
                    num_heads, num_kv_heads
                ),
            });
        }
    }

    Ok(())
}

/// Reject a custom layer whose equations are structurally broken.
///
/// Custom blocks are how a researcher describes an operator NEURAX has no
/// built-in formula for, so this check is deliberately narrow: it rejects
/// malformed expressions, which used to surface far downstream as an opaque
/// "total FLOPs is zero" with nothing pointing at the layer at fault, but it
/// accepts variables it does not recognise. Enumerating every symbol an
/// architecture might need is exactly the limit custom blocks exist to escape.
fn validate_custom_equations(layer: &crate::model_config::Layer) -> Result<(), ParserError> {
    use evalexpr::{ContextWithMutableVariables, HashMapContext, Value};

    let Some(equations) = layer.custom_equations.as_ref() else {
        return Ok(());
    };

    // The documented variables, bound to 1 for a trial evaluation.
    let mut context = HashMapContext::new();
    for name in [
        "B",           // batch
        "S",           // sequence length
        "H",           // hidden size
        "D",           // head dimension
        "I",           // intermediate size
        "V",           // vocabulary size
        "N",           // nodes (graphs) / heads (attention)
        "E",           // edges
        "D_head",      // head dimension, spelled out
        "D_edge",      // edge feature dimension
        "dtype_bytes", // storage width
    ] {
        let _ = context.set_value(name.into(), Value::Float(1.0));
    }

    let named = [
        (equations.flops_forward.as_deref(), "flops_forward"),
        (equations.memory_activation.as_deref(), "memory_activation"),
        (equations.gradient.as_deref(), "gradient"),
    ];
    let extra = equations
        .extra
        .iter()
        .map(|(name, equation)| (Some(equation.as_str()), name.as_str()));

    for (equation, name) in named.into_iter().chain(extra) {
        let Some(equation) = equation else { continue };
        let field = format!("layers.{}.custom_equations.{}", layer.id, name);

        if equation.trim().is_empty() {
            return Err(ParserError::InvalidValue {
                field,
                reason: "equation is empty".to_string(),
            });
        }

        match evalexpr::eval_with_context(equation, &context) {
            Ok(Value::Float(_)) | Ok(Value::Int(_)) => {}
            Ok(other) => {
                return Err(ParserError::InvalidValue {
                    field,
                    reason: format!("`{}` must evaluate to a number, got {:?}", equation, other),
                });
            }
            Err(evalexpr::EvalexprError::VariableIdentifierNotFound(symbol)) => {
                // An architecture-specific symbol. The equation is well formed;
                // the operator pass binds what it can and reports a diagnostic
                // if it still cannot evaluate it.
                tracing::debug!(
                    "custom equation for layer '{}' uses the unrecognised symbol '{}'",
                    layer.id,
                    symbol
                );
            }
            Err(reason) => {
                return Err(ParserError::InvalidValue {
                    field,
                    reason: format!("`{}` could not be evaluated: {}", equation, reason),
                });
            }
        }
    }

    Ok(())
}

/// Validate layer parameters (kernel_size, stride, channels, etc.)
fn validate_layer_params(config: &ModelConfig) -> Result<(), ParserError> {
    validate_global_dimensions(config)?;

    for layer in &config.model.layers {
        // Dimensions feed directly into `hidden / heads` divisions and into
        // products that accumulate parameter counts, so bound them for every
        // layer type before the type-specific checks below.
        if let Some(hidden_size) = layer.params.hidden_size {
            check_dimension(hidden_size, format!("layers.{}.params.hidden_size", layer.id))?;
        }
        if let Some(intermediate_size) = layer.params.intermediate_size {
            check_dimension(
                intermediate_size,
                format!("layers.{}.params.intermediate_size", layer.id),
            )?;
        }
        validate_attention_heads(layer)?;
        validate_custom_equations(layer)?;

        match layer.layer_type {
            LayerType::Conv => {
                // Validate kernel_size > 0
                if let Some(kernel_size) = layer.params.kernel_size {
                    if kernel_size == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.kernel_size", layer.id),
                            reason: "kernel_size must be greater than 0".to_string(),
                        });
                    }
                }

                // Validate stride > 0
                if let Some(stride) = layer.params.stride {
                    if stride == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.stride", layer.id),
                            reason: "stride must be greater than 0".to_string(),
                        });
                    }
                }

                // Validate in_channels is specified for non-first layers
                if layer.params.in_channels.is_none() && layer.params.out_channels.is_some() {
                    // Allow for first layer where in_channels might be inferred from input_shape
                    if !layer.input_shape.is_empty() && layer.input_shape[0] > 0 {
                        // in_channels can be inferred from input_shape[0]
                    }
                }

                // Validate out_channels is specified
                if layer.params.out_channels.is_none() {
                    tracing::warn!(
                        "Layer {} (conv) has no out_channels specified, using default",
                        layer.id
                    );
                }
            }
            LayerType::Pooling => {
                // Validate kernel_size > 0 for pooling
                if let Some(kernel_size) = layer.params.kernel_size {
                    if kernel_size == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.kernel_size", layer.id),
                            reason: "pooling kernel_size must be greater than 0".to_string(),
                        });
                    }
                }

                // Validate stride > 0
                if let Some(stride) = layer.params.stride {
                    if stride == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.stride", layer.id),
                            reason: "pooling stride must be greater than 0".to_string(),
                        });
                    }
                }
            }
            LayerType::Dense => {
                // Validate in_features and out_features
                if layer.params.hidden_size.is_none() && layer.params.intermediate_size.is_none() {
                    tracing::debug!(
                        "Layer {} (dense) has no in_features/out_features specified",
                        layer.id
                    );
                }
            }
            LayerType::ResidualBlock
            | LayerType::Mbconv
            | LayerType::Inception
            | LayerType::DenseBlock
            | LayerType::ConvnextBlock
            | LayerType::ShuffleUnit
            | LayerType::C2f
            | LayerType::Detection
            | LayerType::Transition => {
                // CNN modern architectures - basic validation
                tracing::debug!(
                    "Layer {} is a modern CNN architecture type: {:?}",
                    layer.id,
                    layer.layer_type
                );
            }
            // State Space Model validation
            LayerType::MambaBlock
            | LayerType::S4Block
            | LayerType::H3Block
            | LayerType::StateSpace
            | LayerType::RwkvBlock
            | LayerType::RetentionBlock => {
                // Validate state_dim > 0 for SSM layers
                if let Some(state_dim) = layer.params.state_dim {
                    if state_dim == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.state_dim", layer.id),
                            reason: "state_dim must be greater than 0 for State Space Models"
                                .to_string(),
                        });
                    }
                }

                // Validate expansion_factor > 0 for Mamba
                if matches!(layer.layer_type, LayerType::MambaBlock) {
                    if let Some(expansion_factor) = layer.params.expansion_factor {
                        if expansion_factor == 0 {
                            return Err(ParserError::InvalidValue {
                                field: format!("layers.{}.params.expansion_factor", layer.id),
                                reason: "expansion_factor must be greater than 0 for Mamba blocks"
                                    .to_string(),
                            });
                        }
                    }
                }

                // Validate conv_kernel_size > 0 if specified
                if let Some(conv_kernel_size) = layer.params.conv_kernel_size {
                    if conv_kernel_size == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.conv_kernel_size", layer.id),
                            reason: "conv_kernel_size must be greater than 0".to_string(),
                        });
                    }
                }

                tracing::debug!(
                    "Layer {} is a State Space Model type: {:?}",
                    layer.id,
                    layer.layer_type
                );
            }
            // GAN validation
            LayerType::GeneratorBlock
            | LayerType::DiscriminatorBlock
            | LayerType::StyleMod
            | LayerType::AdaIN
            | LayerType::MinibatchStd
            | LayerType::PixelNorm
            | LayerType::SelfAttention
            | LayerType::SpectralNorm
            | LayerType::ProgressiveBlock => {
                // Validate latent_dim > 0 if specified
                if let Some(latent_dim) = layer.params.latent_dim {
                    if latent_dim == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.latent_dim", layer.id),
                            reason: "latent_dim must be greater than 0 for GAN models".to_string(),
                        });
                    }
                }

                // Validate resolution > 0 if specified
                if let Some(resolution) = layer.params.resolution {
                    if resolution == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.resolution", layer.id),
                            reason: "resolution must be greater than 0".to_string(),
                        });
                    }
                }

                // Validate truncation in [0, 1] if specified
                if let Some(truncation) = layer.params.truncation {
                    if !(0.0..=1.0).contains(&truncation) {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.truncation", layer.id),
                            reason: "truncation must be between 0 and 1".to_string(),
                        });
                    }
                }

                tracing::debug!(
                    "Layer {} is a GAN architecture type: {:?}",
                    layer.id,
                    layer.layer_type
                );
            }
            // LSTM/RNN validation
            LayerType::LstmBlock
            | LayerType::GruBlock
            | LayerType::RnnCell
            | LayerType::Bidirectional
            | LayerType::EncoderBlock
            | LayerType::DecoderBlock => {
                // Validate rnn_hidden_size > 0 if specified
                if let Some(rnn_hidden_size) = layer.params.rnn_hidden_size {
                    if rnn_hidden_size == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.rnn_hidden_size", layer.id),
                            reason: "rnn_hidden_size must be greater than 0 for RNN layers"
                                .to_string(),
                        });
                    }
                }

                // Validate num_rnn_layers > 0 if specified
                if let Some(num_rnn_layers) = layer.params.num_rnn_layers {
                    if num_rnn_layers == 0 {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.num_rnn_layers", layer.id),
                            reason: "num_rnn_layers must be greater than 0".to_string(),
                        });
                    }
                }

                // Validate recurrent_dropout in [0, 1) if specified
                if let Some(recurrent_dropout) = layer.params.recurrent_dropout {
                    if !(0.0..1.0).contains(&recurrent_dropout) {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.recurrent_dropout", layer.id),
                            reason: "recurrent_dropout must be in [0, 1)".to_string(),
                        });
                    }
                }

                // Validate zoneout in [0, 1) if specified
                if let Some(zoneout) = layer.params.zoneout {
                    if !(0.0..1.0).contains(&zoneout) {
                        return Err(ParserError::InvalidValue {
                            field: format!("layers.{}.params.zoneout", layer.id),
                            reason: "zoneout must be in [0, 1)".to_string(),
                        });
                    }
                }

                tracing::debug!(
                    "Layer {} is an LSTM/RNN architecture type: {:?}",
                    layer.id,
                    layer.layer_type
                );
            }
            _ => {}
        }
    }

    Ok(())
}

/// Validate a custom equation string
pub fn validate_custom_equation(equation: &str) -> Result<(), ParserError> {
    // Basic validation: check for common issues
    if equation.is_empty() {
        return Err(ParserError::InvalidValue {
            field: "equation".to_string(),
            reason: "Equation cannot be empty".to_string(),
        });
    }

    // Check for dangerous patterns (basic security)
    let dangerous = ["import", "eval", "exec", "system", "file", "read", "write"];
    for pattern in dangerous {
        if equation.to_lowercase().contains(pattern) {
            return Err(ParserError::InvalidValue {
                field: "equation".to_string(),
                reason: format!("Equation contains forbidden pattern: {}", pattern),
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_custom_equation() {
        assert!(validate_custom_equation("2 * B * S * H").is_ok());
        assert!(validate_custom_equation("").is_err());
        assert!(validate_custom_equation("import os").is_err());
    }
}
