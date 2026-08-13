"""Storage width is settled by measurement, not by asking the planner nicely.

Narrowing the dtype changes how weights are stored without touching the
architecture the client described, so when it is enough to meet the budget the
loop applies it directly and re-measures. Delegating that to the planner spends
an attempt and depends on the model following the instruction.
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from topology_validator import ArchSpec
from budget_check import measure_and_check, narrow_precision_to_fit, suggest_precision
from requirements import extract_budget, DeploymentBudget

MB = 1024 ** 2


def spec_at(precision):
    """~1.29M parameters: 5.15 MB at fp32, 2.57 MB at bf16, 1.29 MB at int8."""
    return ArchSpec.from_dict({
        "family": "transformer",
        "nodes": [
            {"id": "emb", "type": "embedding",
             "params": {"vocab_size": 4000, "hidden_size": 256}},
            {"id": "attn", "type": "mha", "params": {"hidden_size": 256, "num_heads": 8}},
        ],
        "edges": [{"from": "emb", "to": "attn"}],
        "hw_config": {"precision": precision, "batchSize": 1, "hardware": "T4", "seqLen": 64},
    })


HW = {"hardware": "T4", "gpuCount": 1, "batchSize": 1}


def _measure(spec, budget):
    return asyncio.run(measure_and_check(spec, budget, HW))


def test_the_loop_narrows_precision_itself_and_reverifies():
    budget = extract_budget("moins de 2 mega")
    spec = spec_at("fp32")

    before = _measure(spec, budget)
    if before.error:
        pytest.skip(f"compiler unavailable: {before.error}")
    assert not before.fits, "fixture must start over budget at fp32"

    applied, after = asyncio.run(narrow_precision_to_fit(spec, budget, HW, before))

    assert applied is not None, "a narrower width should have been found"
    assert after.fits, after.summary()
    # The decision is recorded on the spec, so it reaches the canvas and the report.
    assert spec.hw_config["precision"] == applied
    # And it is a real measurement, not a prediction.
    assert after.metrics["parameter_memory_bytes"] <= budget.max_size_bytes


def test_it_stops_at_the_widest_precision_that_works():
    # 5.15 MB at fp32 against a 3 MB budget: bf16 (2.57 MB) is enough, so there
    # is no reason to go to int8 and trade away accuracy the client never
    # offered to give up.
    budget = extract_budget("moins de 3 mega")
    spec = spec_at("fp32")
    before = _measure(spec, budget)
    if before.error:
        pytest.skip("compiler unavailable")
    applied, _ = asyncio.run(narrow_precision_to_fit(spec, budget, HW, before))
    assert applied == "bf16", f"expected the widest width that fits, got {applied}"


def test_it_gives_up_when_no_width_is_enough():
    # 4 MB of fp32 weights cannot reach 0.2 MB by dtype alone; the architecture
    # itself has to shrink, and the loop must say so rather than mangle the design.
    budget = extract_budget("moins de 200 ko")
    spec = spec_at("fp32")
    before = _measure(spec, budget)
    if before.error:
        pytest.skip("compiler unavailable")
    applied, after = asyncio.run(narrow_precision_to_fit(spec, budget, HW, before))
    assert applied is None
    assert after is None
    # The design is left exactly as the planner wrote it.
    assert spec.hw_config["precision"] == "fp32"


def test_a_design_already_within_budget_is_left_alone():
    budget = extract_budget("moins de 50 mega")
    spec = spec_at("fp32")
    before = _measure(spec, budget)
    if before.error:
        pytest.skip("compiler unavailable")
    assert before.fits
    applied, _ = asyncio.run(narrow_precision_to_fit(spec, budget, HW, before))
    assert applied is None, "no reason to change a design that already fits"
    assert spec.hw_config["precision"] == "fp32"


def test_a_design_already_at_the_narrowest_width_has_nothing_left_to_give():
    budget = extract_budget("moins de 200 ko")
    spec = spec_at("int8")
    before = _measure(spec, budget)
    if before.error:
        pytest.skip("compiler unavailable")
    applied, _ = asyncio.run(narrow_precision_to_fit(spec, budget, HW, before))
    assert applied is None
    assert suggest_precision("int8", 2.0) is None
