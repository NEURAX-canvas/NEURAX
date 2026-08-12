"""The budget check must measure a real design with the real compiler."""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from requirements import extract_budget
from budget_check import measure_and_check, spec_to_topology
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
