//! Inference IR — structures de données pour la simulation d'inférence.

use serde::{Deserialize, Serialize};

/// Paramètres d'inférence fournis par le frontend.
///
/// `serde(default)` au niveau de la structure : un appelant qui ne précise que
/// les champs qu'il souhaite changer reçoit les valeurs par défaut pour le
/// reste, au lieu d'un 400 sans explication.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceParams {
    // Sampling Strategy
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub beam_width: u32,
    pub repetition_penalty: f64,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    // Context Configuration
    pub prompt_length: u32,
    pub max_output_tokens: u32,
    pub sliding_window: bool,
    pub kv_cache_reuse: bool,
    // Model Behavior
    pub architecture_family: String,
    pub attention_type: String,
    pub moe_router_mode: Option<String>,
    pub quantization_level: String,
    // Stability Stress Tests
    pub long_context_simulation: bool,
    pub adversarial_prompt: bool,
    pub high_temperature_mode: bool,
    pub low_temperature_mode: bool,
}

impl Default for InferenceParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            beam_width: 1,
            repetition_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            prompt_length: 2048,
            max_output_tokens: 1024,
            sliding_window: true,
            kv_cache_reuse: true,
            architecture_family: "transformer".to_string(),
            attention_type: "standard".to_string(),
            moe_router_mode: None,
            quantization_level: "fp16".to_string(),
            long_context_simulation: false,
            adversarial_prompt: false,
            high_temperature_mode: false,
            low_temperature_mode: false,
        }
    }
}

/// What the analysis knows about the model being simulated.
///
/// Sampling settings alone cannot answer several of the questions this report
/// claims to: how far a prompt pushes the model past the context it was built
/// for, how much KV cache a request costs, or how a mixture-of-experts router
/// spreads load across the experts the design actually declares. Before this
/// existed the pass assumed a 32,768-token window for every model and invented
/// an expert count, so the answers described no model in particular.
///
/// Every field is optional: a caller may simulate sampling behaviour alone, and
/// the widgets that need a model fall back to being reported as unavailable
/// rather than to a fabricated default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelProfile {
    pub total_parameters: Option<u64>,
    pub num_layers: Option<u64>,
    pub hidden_size: Option<u64>,
    pub num_heads: Option<u64>,
    pub num_kv_heads: Option<u64>,
    /// Context the model was built for, in tokens.
    pub trained_context: Option<u64>,
    pub num_experts: Option<u64>,
    pub top_k: Option<u64>,
    /// State dimension, for state-space models.
    pub state_dim: Option<u64>,
    /// Bytes per stored value, from the model's precision.
    pub dtype_bytes: Option<u64>,
}

impl ModelProfile {
    /// True when nothing about the model was supplied.
    pub fn is_empty(&self) -> bool {
        self.total_parameters.is_none()
            && self.num_layers.is_none()
            && self.hidden_size.is_none()
            && self.trained_context.is_none()
    }

    /// Head dimension, when both the width and the head count are known.
    pub fn head_dim(&self) -> Option<u64> {
        match (self.hidden_size, self.num_heads) {
            (Some(hidden), Some(heads)) if heads > 0 => Some(hidden / heads),
            _ => None,
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StabilityLevel {
    Stable,
    Drift,
    Unstable,
    Chaotic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityIndex {
    /// Score normalisé [0, 1] — 1 = parfaitement stable
    pub score: f64,
    pub level: StabilityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationRisk {
    pub risk: RiskLevel,
    /// Confiance estimée [0, 100]
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingVolatility {
    /// Diversité de sortie [0, 1]
    pub diversity: f64,
    /// Déterminisme [0, 1]
    pub determinism: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStability {
    /// Consistance du routage [0, 1]
    pub stability: f64,
    /// Distribution de charge par expert (N valeurs, somme ≈ 1)
    pub distribution: Vec<f64>,
}

/// Cost of the key/value cache for the simulated request.
///
/// KV cache is what actually limits concurrency when serving: it grows with the
/// context, and grouped-query attention is the main lever against it. Reporting
/// it needs the model's layer count, KV head count and head dimension, none of
/// which sampling parameters carry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheCost {
    /// Bytes held per token of context.
    pub bytes_per_token: u64,
    /// Bytes for the full prompt plus generated output.
    pub bytes_total: u64,
    /// How much smaller grouped-query attention makes it than full multi-head.
    pub gqa_savings_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskOverview {
    pub coherence: RiskLevel,
    pub overconfidence: RiskLevel,
    pub collapse: RiskLevel,
    pub degeneration: RiskLevel,
}

/// Rapport complet retourné par l'endpoint `/inference/simulate`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceReport {
    /// Widget 1 — Generation Stability Index
    pub stability_index: StabilityIndex,
    /// Widget 2 — Entropy Evolution (20 points)
    pub entropy_evolution: Vec<f64>,
    /// Widget 3 — Noise Schedule Curve (diffusion seulement)
    pub noise_schedule: Option<Vec<f64>>,
    /// Widget 4 — Hallucination Risk
    pub hallucination_risk: HallucinationRisk,
    /// Widget 5 — Attention Focus (12 tokens)
    pub attention_focus: Vec<f64>,
    /// Widget 6 — State Stability / SSM [0, 1]
    pub state_stability: f64,
    /// Widget 7 — Context Degradation : % de fenêtre effective restante
    pub context_degradation: f64,
    /// Widget 8 — Sampling Volatility
    pub sampling_volatility: SamplingVolatility,
    /// Widget 9 — Router Stability (MoE seulement)
    pub router_stability: Option<RouterStability>,
    /// Widget 10 — Inference Risk Overview
    pub risk_overview: RiskOverview,
    /// Widget 11 — KV cache cost. `None` when the model was not supplied.
    #[serde(default)]
    pub kv_cache: Option<KvCacheCost>,
    /// Echo of the model the report was computed for, so a reader can tell
    /// whether a figure describes their design or sampling behaviour alone.
    #[serde(default)]
    pub model_profile: Option<ModelProfile>,
}
