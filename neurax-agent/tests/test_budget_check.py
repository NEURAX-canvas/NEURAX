"""The budget check must measure a real design with the real compiler."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from requirements import extract_budget
from budget_check import BudgetReport, measure_and_check, spec_to_topology
from topology_validator import ArchSpec

MB = 1024 ** 2

TINY = ArchSpec.from_dict({
    "family": "transformer",
    "nodes": [
        {"id": "img", "type": "conv2d",
         "params": {"in_channels": 3, "out_channels": 32, "kernel_size": 8, "stride": 8}},
        {"id": "emb", "type": "token_embedding",
         "params": {"vocab_size": 2000, "hidden_size": 64}},
        {"id": "attn", "type": "mha_attention", "params": {"hidden_size": 64, "num_heads": 4}},
        {"id": "ffn", "type": "ffn_standard",
         "params": {"hidden_size": 64, "intermediate_size": 128, "activation": "relu"}},
        {"id": "head", "type": "lm_head", "params": {"in_features": 64, "out_features": 10}},
    ],
    "edges": [{"from": "img", "to": "attn"}, {"from": "emb", "to": "attn"},
              {"from": "attn", "to": "ffn"}, {"from": "ffn", "to": "head"}],
})

HUGE = ArchSpec.from_dict({
    "family": "transformer",
    "nodes": [
        {"id": "emb", "type": "token_embedding",
         "params": {"vocab_size": 60000, "hidden_size": 1024}},
        {"id": "attn", "type": "mha_attention", "params": {"hidden_size": 1024, "num_heads": 16}},
        {"id": "ffn", "type": "ffn_standard",
         "params": {"hidden_size": 1024, "intermediate_size": 4096}},
    ],
    "edges": [{"from": "emb", "to": "attn"}, {"from": "attn", "to": "ffn"}],
})

HW = {"hardware": "T4", "gpuCount": 1, "precision": "int8", "batchSize": 1}


def test_topology_carries_the_precision_the_client_asked_for():
    topo = spec_to_topology(TINY, HW)
    # The compiler reads storage width from training.precision.
    assert topo["training"]["precision"] == "int8"
    assert topo["model"]["type"] == "transformer"
    assert len(topo["model"]["layers"]) == 5


def test_canvas_block_names_are_mapped_to_compiler_names():
    topo = spec_to_topology(TINY, HW)
    kinds = [l["layer_type"] for l in topo["model"]["layers"]]
    assert kinds == ["conv", "embedding", "attention", "mlp", "dense"]


def test_a_design_within_budget_passes():
    budget = extract_budget("un modele pour telephone, moins de 1 mega")
    report = asyncio.run(measure_and_check(TINY, budget, HW))
    if report.error:
        pytest.skip(f"compiler unavailable: {report.error}")
    assert report.fits, report.summary()
    assert report.metrics["parameter_memory_bytes"] < 1 * MB


def test_a_design_over_budget_fails_and_says_by_how_much():
    budget = extract_budget("moins de 1 mega")
    report = asyncio.run(measure_and_check(HUGE, budget, HW))
    if report.error:
        pytest.skip(f"compiler unavailable: {report.error}")
    assert not report.fits
    feedback = report.planner_feedback()
    assert feedback, "an over-budget design must produce actionable feedback"
    assert "too large" in feedback[0]
    assert "Model size" in feedback[0]


def test_no_budget_means_no_constraint():
    budget = extract_budget("build me something nice")
    report = asyncio.run(measure_and_check(TINY, budget, HW))
    if report.error:
        pytest.skip(f"compiler unavailable: {report.error}")
    assert report.fits
    assert report.checks == []


# ─── The compiler's diagnostics reach the planner ──────────────────────

def _diag(severity, message, code="W001", suggestion=None):
    return {
        "severity": severity,
        "message": message,
        "code": code,
        "suggestion": suggestion,
        "category": "Memory",
    }


def test_blocking_diagnostics_keeps_only_what_stops_a_design():
    """Informational warnings are not reasons to redesign; failures are."""
    report = BudgetReport(
        fits=True,
        diagnostics=[
            _diag("Warning", "Head count is unusual for this width"),
            _diag("Info", "Using the default rope theta"),
            _diag("Critical", "This model needs 954.5 GB but the GPU has 17.2 GB"),
        ],
    )

    blocking = report.blocking_diagnostics()
    assert len(blocking) == 1
    assert "954.5 GB" in blocking[0]["message"]


def test_a_design_that_fits_the_budget_still_reports_a_blocking_diagnostic():
    """The gap this closes.

    A model can sit comfortably under every stated limit and still be one the
    compiler says will not start. Returning early on `fits` meant nobody was
    told.
    """
    report = BudgetReport(
        fits=True,
        diagnostics=[_diag("Critical", "Peak VRAM exceeds the target GPU", code="E002")],
    )

    feedback = report.planner_feedback()
    assert feedback, "a blocking diagnostic must reach the planner"
    assert "E002" in feedback[0]
    assert "Peak VRAM" in feedback[0]


def test_a_clean_design_produces_no_feedback():
    report = BudgetReport(fits=True, diagnostics=[_diag("Info", "Nothing of note")])
    assert report.planner_feedback() == []


def test_the_suggestion_is_carried_across():
    report = BudgetReport(
        fits=True,
        diagnostics=[
            _diag(
                "Critical",
                "Peak VRAM exceeds the target GPU",
                suggestion="Enable gradient checkpointing or use a larger GPU.",
            )
        ],
    )
    assert "gradient checkpointing" in report.planner_feedback()[0]


def test_diagnostics_survive_a_report_with_no_metrics():
    """A rejected design is exactly when its diagnostics matter most."""
    report = BudgetReport(
        fits=True,
        error="analysis returned no metrics",
        diagnostics=[_diag("Critical", "Attention head count divides into zero")],
    )
    assert len(report.planner_feedback()) == 1
