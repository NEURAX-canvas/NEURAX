//! Activation functions: cost model and structural role.
//!
//! Two properties of an activation matter to an analytical compiler:
//!
//! 1. **Its element-wise cost.** Activations are memory-bound and cheap next to
//!    the matmuls around them, but they are not free, and a transcendental like
//!    GELU costs an order of magnitude more per element than ReLU.
//!
//! 2. **Whether it is gated.** SwiGLU, GeGLU and ReGLU are not point-wise
//!    functions: they split the feed-forward input across a *gate* and an *up*
//!    projection and multiply the two. That makes the MLP three weight matrices
//!    instead of two — 50% more parameters and FLOPs — so the choice of
//!    activation changes the shape of the model, not just its cost.
//!
//! Both properties live here so the passes cannot disagree about them.
//!
//! ## Cost convention
//!
//! Counts are FLOPs per element, taking a transcendental call (`exp`, `log`,
//! `erf`, `tanh`) as several floating-point operations rather than one, which is
//! how these are actually evaluated. They are engineering approximations: exact
//! throughput depends on the kernel and whether the unit is fused. They are
//! deliberately conservative and stable across releases.

/// What the compiler knows about one activation function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivationSpec {
    /// Canonical name, after alias resolution.
    pub canonical: &'static str,
    /// Approximate FLOPs per element.
    pub flops_per_element: f64,
    /// True when the activation consumes a separate gate projection, making the
    /// surrounding feed-forward block three matrices rather than two.
    pub gated: bool,
}

/// Look up an activation by name, accepting the common aliases.
///
/// Returns `None` for an unrecognised name so callers can surface that rather
/// than silently costing it with an invented number.
pub fn activation_spec(name: &str) -> Option<ActivationSpec> {
    let normalized = name.trim().to_ascii_lowercase().replace('-', "_");
    let (canonical, flops_per_element, gated) = match normalized.as_str() {
        // ── Point-wise ──────────────────────────────────────────────────────
        // Identity: no arithmetic at all.
        "none" | "linear" | "identity" => ("none", 0.0, false),
        // max(0, x): one comparison and a select.
        "relu" => ("relu", 1.0, false),
        // Same, plus the scaled branch.
        "leaky_relu" | "leakyrelu" => ("leaky_relu", 2.0, false),
        // min(max(0, x), 6): two comparisons.
        "relu6" => ("relu6", 2.0, false),
        // 1/(1+exp(-x)): one exp, one add, one divide.
        "sigmoid" | "logistic" => ("sigmoid", 4.0, false),
        // Two exps and a divide, in the usual stable formulation.
        "tanh" => ("tanh", 6.0, false),
        // x·sigmoid(x): a sigmoid plus the product.
        "silu" | "swish" => ("silu", 4.0, false),
        // Hard approximations trade the transcendental for clamped arithmetic.
        "hard_sigmoid" | "hardsigmoid" => ("hard_sigmoid", 2.0, false),
        "hard_swish" | "hardswish" | "h_swish" => ("hard_swish", 3.0, false),
        // 0.5·x·(1+erf(x/√2)): erf dominates.
        "gelu" | "gelu_erf" => ("gelu", 10.0, false),
        // The tanh approximation of GELU used by GPT-2 and friends.
        "gelu_tanh" | "gelu_new" | "gelu_approx" => ("gelu_tanh", 8.0, false),
        // ln(1+exp(x)): exp, add, log.
        "softplus" => ("softplus", 7.0, false),
        // x·tanh(softplus(x)): the two stacked, plus the product.
        "mish" => ("mish", 12.0, false),
        // Exponential branch for negative inputs.
        "elu" => ("elu", 4.0, false),
        "selu" => ("selu", 5.0, false),

        // ── Gated: these add a third projection ─────────────────────────────
        // Plain GLU gates with a sigmoid.
        "glu" => ("glu", 4.0, true),
        // SwiGLU gates with SiLU — LLaMA, Mistral, PaLM.
        "swiglu" | "silu_glu" | "swish_glu" => ("swiglu", 4.0, true),
        // GeGLU gates with GELU.
        "geglu" | "gelu_glu" => ("geglu", 10.0, true),
        // ReGLU gates with ReLU.
        "reglu" | "relu_glu" => ("reglu", 1.0, true),

        _ => return None,
    };
    Some(ActivationSpec {
        canonical,
        flops_per_element,
        gated,
    })
}

/// FLOPs per element for `name`.
///
/// Unknown names fall back to a mid-range transcendental cost, matching the
/// long-standing behaviour of the MLP formulas. Prefer [`activation_spec`] when
/// the caller can report the unknown name instead.
pub fn activation_flops_per_element(name: &str) -> f64 {
    activation_spec(name)
        .map(|spec| spec.flops_per_element)
        .unwrap_or(5.0)
}

/// True when the activation makes the surrounding feed-forward block gated.
///
/// A config that asks for `swiglu` is describing a three-matrix MLP whether or
/// not it also sets a separate `gated` flag, so the two must agree.
pub fn is_gated_activation(name: &str) -> bool {
    activation_spec(name).map(|spec| spec.gated).unwrap_or(false)
}

/// Canonical spelling of `name`, or `None` if it is not recognised.
pub fn canonical_activation(name: &str) -> Option<&'static str> {
    activation_spec(name).map(|spec| spec.canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_resolve_to_one_canonical_spelling() {
        for (alias, expected) in [
            ("SiLU", "silu"),
            ("swish", "silu"),
            ("gelu_new", "gelu_tanh"),
            ("Hard-Swish", "hard_swish"),
            ("SwiGLU", "swiglu"),
            ("relu_glu", "reglu"),
        ] {
            assert_eq!(
                canonical_activation(alias),
                Some(expected),
                "{alias} should canonicalise to {expected}"
            );
        }
    }

    #[test]
    fn gated_activations_are_flagged() {
        for name in ["swiglu", "geglu", "reglu", "glu"] {
            assert!(is_gated_activation(name), "{name} should be gated");
        }
        for name in ["gelu", "relu", "silu", "tanh", "none"] {
            assert!(!is_gated_activation(name), "{name} should not be gated");
        }
    }

    #[test]
    fn costs_are_ordered_by_how_much_arithmetic_they_need() {
        let cost = activation_flops_per_element;
        assert_eq!(cost("none"), 0.0);
        assert!(cost("relu") < cost("sigmoid"));
        assert!(cost("sigmoid") <= cost("tanh"));
        assert!(cost("tanh") < cost("gelu"));
        assert!(cost("gelu_tanh") < cost("gelu"));
        assert!(cost("gelu") < cost("mish"));
    }

    #[test]
    fn unknown_activations_are_reported_as_unknown() {
        assert!(activation_spec("definitely_not_an_activation").is_none());
        // ... while the lenient helper still yields a usable number.
        assert_eq!(activation_flops_per_element("definitely_not_an_activation"), 5.0);
    }
}
