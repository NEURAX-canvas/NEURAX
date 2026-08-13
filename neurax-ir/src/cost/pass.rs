//! Cost IR pass

use super::{CostIR, CostMetrics, PricingModel};
use crate::error::CostError;
use crate::hardware::HardwareIR;
use crate::parallelism::ParallelismIR;
use crate::traits::IrPass;
use crate::NeuraxContext;

/// Cost pass implementation
pub struct CostPass;

impl IrPass for CostPass {
    type Input = (HardwareIR, ParallelismIR);
    type Output = CostIR;
    type Metrics = CostMetrics;
    type PassError = CostError;

    fn name(&self) -> &'static str {
        "CostIR"
    }

    fn build(
        &self,
        input: &Self::Input,
        ctx: &NeuraxContext,
    ) -> Result<Self::Output, Self::PassError> {
        let (hw_ir, _parallel_ir) = input;
        let mut cost_ir = CostIR::default();

        // Carry the hardware pass's measured step latency into the cost pass.
        cost_ir.step_latency_ms = hw_ir.metrics.latency_ms;

        // Take the TDP from the hardware database rather than a local table.
        // The previous inline match only knew exact board names, so the common
        // spellings ("H100", "A100") fell through to a 300 W default and skewed
        // energy and CO2 accordingly.
        let gpu_tdp_watts = ctx
            .config
            .hardware
            .gpus
            .first()
            .and_then(|g| ctx.gpu_db.get_gpu(&g.name))
            .map(|spec| spec.tdp_watts as f64)
            .unwrap_or(300.0);

        // Set up pricing model from config
        cost_ir.pricing_model = PricingModel {
            gpu_hour_usd: ctx.config.cost_config.gpu_hour_usd,
            energy_kwh_usd: ctx.config.cost_config.energy_kwh_usd,
            pue_factor: ctx.config.cost_config.pue_factor,
            gpu_tdp_watts,
            co2_per_kwh: 0.233,
        };

        Ok(cost_ir)
    }

    fn compute_metrics(
        &self,
        output: &mut Self::Output,
        ctx: &NeuraxContext,
    ) -> Result<Self::Metrics, Self::PassError> {
        let num_gpus = ctx.config.hardware.total_gpu_count();
        let batch = ctx.config.training.batch_size;
        let seq = ctx
            .config
            .model
            .global_params
            .sequence_length
            .or(ctx.config.training.sequence_length)
            .unwrap_or(512);

        // Tokens consumed per optimizer step.
        let tokens_per_step = (batch as f64 * seq as f64).max(1.0);

        // Resolve the step count. `max_steps` is optional and defaults to zero,
        // which used to zero out training time and with it every cost, energy
        // and CO2 figure. When it is absent, derive it from the training budget
        // the caller did provide: epochs over the dataset.
        let (max_steps, steps_were_derived) = if ctx.config.training.max_steps > 0 {
            (ctx.config.training.max_steps as u64, false)
        } else {
            let epochs = ctx.config.training.num_epochs.unwrap_or(1.0).max(0.0);
            let dataset_tokens = ctx.config.data.dataset_size.unwrap_or(0.0).max(0.0);
            let derived = (epochs * dataset_tokens / tokens_per_step).floor();
            // Stay inside u64 and treat a missing dataset size as "unknown"
            // rather than inventing a training run.
            (derived.clamp(0.0, u64::MAX as f64) as u64, true)
        };

        output.effective_steps = max_steps;
        output.steps_were_derived = steps_were_derived;

        // Wall-clock time from the hardware pass's measured per-step latency.
        let latency_ms = output.step_latency_ms;

        // Training time
        let training_time_s = max_steps as f64 * latency_ms / 1000.0;
        let training_time_hours = training_time_s / 3600.0;

        // GPU hours
        let gpu_hours_total = training_time_hours * num_gpus as f64;

        // Training cost
        let training_cost_usd = gpu_hours_total * output.pricing_model.gpu_hour_usd;

        // Energy consumption
        let energy_kwh = training_time_hours * output.pricing_model.gpu_tdp_watts * num_gpus as f64
            / 1000.0
            * output.pricing_model.pue_factor;

        // CO2 emissions
        let co2_kg = energy_kwh * output.pricing_model.co2_per_kwh;

        // Cost per token — same tokens-per-step basis used to derive the steps.
        let total_tokens = max_steps as f64 * tokens_per_step;
        let cost_per_token_usd = if total_tokens > 0.0 {
            training_cost_usd / total_tokens
        } else {
            0.0
        };

        let cost_per_million_tokens_usd = cost_per_token_usd * 1e6;

        // Cost per step
        let cost_per_step_usd = if max_steps > 0 {
            training_cost_usd / max_steps as f64
        } else {
            0.0
        };

        // Monthly inference cost (estimate)
        let monthly_inference_cost_usd =
            30.0 * 24.0 * output.pricing_model.gpu_hour_usd * num_gpus as f64;

        let metrics = CostMetrics {
            training_time_hours,
            training_cost_usd,
            gpu_hours_total,
            energy_kwh,
            co2_kg,
            cost_per_token_usd,
            cost_per_million_tokens_usd,
            monthly_inference_cost_usd,
            cost_per_step_usd,
        };

        output.metrics = metrics.clone();
        output.metrics_done = true;
        Ok(metrics)
    }

    fn validate(
        &self,
        _output: &Self::Output,
        metrics: &Self::Metrics,
    ) -> Result<(), Self::PassError> {
        if metrics.training_cost_usd < 0.0 {
            return Err(CostError::PricingFailed(
                "Training cost is negative".to_string(),
            ));
        }
        Ok(())
    }
}
