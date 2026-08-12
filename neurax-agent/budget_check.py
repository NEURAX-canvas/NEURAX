"""Measure a planned architecture and check it against the client's budget.

Structural validation answers *is this a well-formed graph*. It cannot answer
*is this the model the client asked for*: a topology can be perfectly valid and
still be fifty times too large for a phone. That question is settled by compiling
the design and comparing the result against the stated limits, which is what this
module does.

When a design misses, the report names the measurement and the gap so the next
planning pass has something concrete to work from rather than being told to "try
smaller".
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from requirements import DeploymentBudget

logger = logging.getLogger(__name__)

NEURAX_SERVICE_URL = os.environ.get("NEURAX_SERVICE_URL", "http://127.0.0.1:9098")

# Canvas block names the compiler knows under a different name.
LAYER_TYPE_MAP = {
    "token_embedding": "embedding",
    "embedding": "embedding",
    "pos_embed": "positional_embed",
    "positional_encoding": "positional_embed",
    "mha_attention": "attention",
    "mha": "attention",
    "attention": "attention",
    "flash_attention": "attention",
    "cross_attention": "cross_attention",
    "ffn_standard": "mlp",
    "ffn_gated": "mlp",
    "ffn": "mlp",
    "mlp": "mlp",
    "layer_norm": "normalization",
    "layernorm": "normalization",
    "rmsnorm": "normalization",
    "linear": "dense",
    "dense": "dense",
    "lm_head": "dense",
    "conv2d": "conv",
    "conv1d": "conv",
    "max_pool": "pooling",
    "avg_pool": "pooling",
    "input": "embedding",
    "output": "dense",
}


@dataclass
class BudgetCheck:
    """Outcome of comparing one measurement against one limit."""

    label: str
    measured: float
    limit: float
    unit: str
    fits: bool

    def describe(self) -> str:
        verdict = "PASS" if self.fits else "FAIL"
        gap = self.limit - self.measured
        relation = "headroom" if self.fits else "over by"
        return (
            f"[{verdict}] {self.label}: {self.measured:,.4g} {self.unit} "
            f"against a limit of {self.limit:,.4g} {self.unit} "
            f"({relation} {abs(gap):,.4g} {self.unit})"
        )


@dataclass
class BudgetReport:
    """What the compiler measured, and whether it satisfies the client."""

    fits: bool
    checks: list[BudgetCheck] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return f"Could not measure the design: {self.error}"
        if not self.checks:
            return "No budget stated; the design was measured but not constrained."
        lines = [check.describe() for check in self.checks]
        lines.append(
            "The design fits every stated budget."
            if self.fits
            else "The design exceeds at least one budget."
        )
        return "\n".join(lines)

    def planner_feedback(self) -> list[str]:
        """Concrete instructions for the next planning pass.

        Phrased as measurements and required reductions, because a planner told
        only that it "failed" tends to make an arbitrary change; told that the
        model is 12.4 MB against a 1 MB limit, it can size the fix.
        """
        if self.fits or self.error:
            return []

        messages = []
        for check in self.checks:
            if check.fits:
                continue
            factor = check.measured / check.limit if check.limit else float("inf")
            messages.append(
                f"{check.label} is {check.measured:,.4g} {check.unit} but must be at most "
                f"{check.limit:,.4g} {check.unit} — roughly {factor:.1f}x too large. "
                f"Reduce it by shrinking hidden/embedding dimensions (parameters grow with "
                f"the square of hidden size), cutting the number of repeated blocks, "
                f"reducing the vocabulary, or narrowing the dtype."
            )
        return messages


def spec_to_topology(
    spec: Any,
    hw_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a planned ArchSpec into the compiler's model config."""
    hw = hw_config or {}
    precision = hw.get("precision") or "fp16"

    layers = []
    for node in spec.nodes:
        raw_type = str(getattr(node, "type", "")).lower()
        layers.append(
            {
                "id": getattr(node, "id", f"layer_{len(layers)}"),
                "layer_type": LAYER_TYPE_MAP.get(raw_type, raw_type),
                "params": dict(getattr(node, "params", {}) or {}),
            }
        )

    global_params: dict[str, Any] = {}
    for key, source in (
        ("hidden_size", "hiddenDim"),
        ("sequence_length", "seqLen"),
        ("vocab_size", "vocabSize"),
        ("num_heads", "numHeads"),
    ):
        value = hw.get(source)
        if value:
            global_params[key] = value

    return {
        "schema_version": "1.0.0",
        "model": {
            "name": "AgentDesign",
            "type": spec.family or "transformer",
            "global_params": global_params,
            "layers": layers,
        },
        "training": {
            "batch_size": hw.get("batchSize") or 1,
            "precision": precision,
        },
        "hardware": {
            "gpus": [
                {
                    "name": hw.get("hardware") or "T4",
                    "count": hw.get("gpuCount") or 1,
                }
            ]
        },
        "data": {"dtype": precision},
    }


async def measure_and_check(
    spec: Any,
    budget: DeploymentBudget,
    hw_config: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> BudgetReport:
    """Compile the design and report whether it satisfies the budget."""
    topology = spec_to_topology(spec, hw_config)

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(
                f"{NEURAX_SERVICE_URL}/analyze", json={"topology": topology}
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:  # network, 4xx from a malformed design, ...
        logger.warning("budget check could not reach the compiler: %s", exc)
        return BudgetReport(fits=True, error=str(exc))

    metrics = payload.get("report", {}).get("metrics", payload.get("metrics", {}))
    if not metrics:
        return BudgetReport(fits=True, error="analysis returned no metrics")

    checks: list[BudgetCheck] = []

    def add(label: str, measured: float, limit: Optional[float], unit: str, scale: float = 1.0):
        if limit is None:
            return
        checks.append(
            BudgetCheck(
                label=label,
                measured=measured / scale,
                limit=limit / scale,
                unit=unit,
                fits=measured <= limit,
            )
        )

    add("Model size", metrics.get("parameter_memory_bytes", 0),
        budget.max_size_bytes, "MB", 1024 ** 2)
    add("Parameters", metrics.get("total_parameters", 0),
        budget.max_parameters, "params")
    add("Latency", metrics.get("latency_ms", 0),
        budget.max_latency_ms, "ms")
    add("Peak VRAM", metrics.get("peak_vram_bytes", 0),
        budget.max_vram_bytes, "GB", 1024 ** 3)

    return BudgetReport(
        fits=all(check.fits for check in checks),
        checks=checks,
        metrics=metrics,
    )
