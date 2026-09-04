"""The budget check must measure a real design with the real compiler."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from requirements import extract_budget
from budget_check import (
    BudgetReport,
    SweepReport,
    measure_and_check,
    optimize_hyperparameters,
    spec_to_topology,
)
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


# ─── Real graph, not hand-threaded shapes ───────────────────────────────
#
# spec_to_topology used to hand-thread a hidden dimension through
# sequence-family nodes only, because at the time the compiler had no real
# graph to walk (no `connections` field existed) and no shape-inference
# engine of its own for any family. Both now exist (Phase 1 of the
# representation-mère plan) — this pre-flight self-check sends the real
# graph and lets the compiler's own engine answer, the same way the
# frontend does post-materialization, just before that instead of after.

def test_edges_are_forwarded_as_connections():
    topo = spec_to_topology(TINY, HW)
    connections = topo["model"]["connections"]
    assert {"from": "img", "to": "attn"} in connections
    assert {"from": "emb", "to": "attn"} in connections
    assert {"from": "attn", "to": "ffn"} in connections
    assert {"from": "ffn", "to": "head"} in connections
    assert len(connections) == 4


def test_no_layer_carries_a_hand_computed_shape_anymore():
    topo = spec_to_topology(TINY, HW)
    for layer in topo["model"]["layers"]:
        assert "input_shape" not in layer
        assert "output_shape" not in layer


def test_image_families_forward_their_entry_shape_into_data():
    cnn_spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"in_channels": 3, "out_channels": 16}}],
        "edges": [],
    })
    hw = {**HW, "inChannels": 3, "imgHeight": 224, "imgWidth": 224}
    topo = spec_to_topology(cnn_spec, hw)
    assert topo["data"]["image_channels"] == 3
    assert topo["data"]["image_height"] == 224
    assert topo["data"]["image_width"] == 224


def test_non_image_families_dont_carry_image_fields():
    topo = spec_to_topology(TINY, HW)
    assert "image_channels" not in topo["data"]


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


def test_a_shape_mismatch_blocks_even_though_the_compiler_only_calls_it_a_warning():
    """W007 must reach the planner even at 'warning' severity.

    The compiler can't always tell a real cross-layer mismatch from a merge
    point it has no representation for, so it reports this conservatively.
    An agent's own freshly-planned design has no such excuse: it should never
    disagree with itself, so this side treats it as blocking regardless.
    """
    report = BudgetReport(
        fits=True,
        diagnostics=[_diag("Warning", "posenc declares 512 but attn produced 768", code="W007")],
    )
    blocking = report.blocking_diagnostics()
    assert len(blocking) == 1
    assert blocking[0]["code"] == "W007"


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


# ─── Hint/info diagnostics reach the client, not just the planner ───────
#
# blocking_diagnostics() feeds the planner (only things that stop a design
# from working); notable_diagnostics()/notes_text() feed the human — real,
# already-computed signals (GQA/MoE detected, no LR warmup, learning_rate
# past the Lipschitz-based stability estimate, tokens-per-parameter off
# Chinchilla-optimal) that used to reach `diagnostics` and then go nowhere,
# since nothing read past blocking_diagnostics()'s severity filter.

def test_notable_diagnostics_keeps_only_hint_and_info_severity():
    report = BudgetReport(
        fits=True,
        diagnostics=[
            _diag("Critical", "This model needs 954.5 GB but the GPU has 17.2 GB"),
            _diag("Warning", "Head count is unusual for this width"),
            _diag("Hint", "learning_rate exceeds the estimated stability bound", code="H007"),
            _diag("Info", "Grouped Query Attention detected", code="I001"),
        ],
    )
    notable = report.notable_diagnostics()
    codes = {d["code"] for d in notable}
    assert codes == {"H007", "I001"}


def test_notes_text_is_empty_when_there_is_nothing_notable():
    report = BudgetReport(fits=True, diagnostics=[_diag("Critical", "Won't start")])
    assert report.notes_text() == ""


def test_summary_includes_notes_alongside_budget_checks():
    report = BudgetReport(
        fits=True,
        checks=[],
        diagnostics=[_diag("Hint", "No learning-rate warmup configured", code="H006")],
    )
    summary = report.summary()
    assert "No budget stated" in summary
    assert "H006" not in summary  # the code is planner-facing, the message is client-facing
    assert "No learning-rate warmup configured" in summary


# ─── Hyperparameter sweep (optimize_hyperparameters / SweepReport) ──────

def test_sweep_report_summary_with_a_best_point():
    report = SweepReport(
        best={
            "batch_size": 8,
            "zero_stage": 2,
            "precision": "bf16",
            "throughput_tokens_per_s": 7453.8,
            "training_cost_usd": 1788808.39,
            "peak_vram_gb": 50.9,
        },
        points_evaluated=36,
        feasible_count=20,
    )
    summary = report.summary()
    assert "batch_size=8" in summary
    assert "zero_stage=2" in summary
    assert "36" in summary and "20" in summary


def test_sweep_report_summary_with_no_feasible_point():
    report = SweepReport(best=None, points_evaluated=9, feasible_count=0)
    summary = report.summary()
    assert "No feasible" in summary
    assert "9" in summary


def test_sweep_report_summary_on_error():
    report = SweepReport(error="All connection attempts failed")
    assert "Could not search" in report.summary()


def test_optimize_hyperparameters_against_the_real_compiler():
    """Same style as measure_and_check's own tests: hit the real service,
    skip if it isn't up rather than mocking the one thing worth checking —
    that the sweep endpoint actually understands what spec_to_topology sends.
    """
    result = asyncio.run(optimize_hyperparameters(TINY, HW))
    if result.error:
        pytest.skip(f"compiler unavailable: {result.error}")
    assert result.points_evaluated > 0
    if result.best is not None:
        assert result.best["peak_vram_gb"] >= 0
        assert "batch_size" in result.best


def test_candidates_are_only_sent_when_given(monkeypatch):
    """Added for `analysis_tools.find_optimal_hyperparameters`, whose MCP
    counterpart already let a caller narrow the sweep to specific values.
    Every existing caller passes no `candidates` and must see the exact same
    request body as before this parameter existed — checked here by
    capturing the outgoing payload rather than hitting a real backend, since
    what matters is what gets sent, not what a live sweep returns.
    """
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"points": [], "best": None}}

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, **kw):
            captured["json"] = json
            return _FakeResponse()

    import budget_check
    monkeypatch.setattr(budget_check.httpx, "AsyncClient", lambda **_: _FakeAsyncClient())

    asyncio.run(optimize_hyperparameters(TINY, HW))
    assert "candidates" not in captured["json"]

    asyncio.run(optimize_hyperparameters(TINY, HW, candidates={"batch_sizes": [1, 2, 4]}))
    assert captured["json"]["candidates"] == {"batch_sizes": [1, 2, 4]}
