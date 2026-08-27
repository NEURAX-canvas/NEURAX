"""Deterministic fan-in repair — the fix for a real, reproduced failure.

Live test: "je veux un modele a 2 milliard de parametres qui va prendre les
images et le texte, qui va tourner sur mobile et qui doit etres rapide"
produced a dual-head design (an image head and a text head both feeding the
single `output` node) three planning attempts in a row, repeating the exact
same fan-in error every time until MAX_ATTEMPTS was exhausted and the whole
build failed with zero nodes ever materialized. The planner was told exactly
what was wrong on every retry and never fixed it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology_validator import ArchSpec, ArchNode, ArchEdge, validate_arch_spec, auto_repair_fanin_violations

CATALOGUE = [
    {"type": "residual_add"},
    {"type": "concat"},
    {"type": "mha"},
    {"type": "ffn"},
    {"type": "classification_head"},
    {"type": "lm_head"},
]
CONSTRAINTS = {"requiredBlocks": ["input", "output"], "incompatibleBlocks": []}


def _dual_head_spec() -> ArchSpec:
    return ArchSpec(
        family="transformer",
        nodes=[
            ArchNode(id="in", type="input"),
            ArchNode(id="attn", type="mha", params={"hidden_size": 512, "num_heads": 8}),
            ArchNode(id="ffn", type="ffn", params={"hidden_size": 512}),
            ArchNode(id="img_head", type="classification_head", params={"num_classes": 1000}),
            ArchNode(id="txt_head", type="lm_head", params={"vocab_size": 32000}),
            ArchNode(id="output", type="output"),
        ],
        edges=[
            ArchEdge(from_id="in", to_id="attn"),
            ArchEdge(from_id="attn", to_id="ffn"),
            ArchEdge(from_id="ffn", to_id="img_head"),
            ArchEdge(from_id="ffn", to_id="txt_head"),
            ArchEdge(from_id="img_head", to_id="output"),
            ArchEdge(from_id="txt_head", to_id="output"),
        ],
    )


def test_the_reproduced_failure_is_rejected_before_repair():
    result = validate_arch_spec(_dual_head_spec(), CATALOGUE, CONSTRAINTS)
    assert not result.valid
    assert any("only accepts" in e for e in result.errors)


def test_auto_repair_inserts_a_merge_node_and_the_result_validates():
    repaired = auto_repair_fanin_violations(_dual_head_spec(), CATALOGUE)
    assert repaired is not None

    result = validate_arch_spec(repaired, CATALOGUE, CONSTRAINTS)
    assert result.valid, result.errors

    # Both heads still exist and now feed a merge node, which alone feeds output.
    incoming_to_output = [e for e in repaired.edges if e.to_id == "output"]
    assert len(incoming_to_output) == 1
    merge_id = incoming_to_output[0].from_id
    merge_node = next(n for n in repaired.nodes if n.id == merge_id)
    assert merge_node.type == "residual_add"  # first in the safe-merge priority list
    feeding_merge = {e.from_id for e in repaired.edges if e.to_id == merge_id}
    assert feeding_merge == {"img_head", "txt_head"}


def test_no_repair_offered_when_the_catalogue_has_no_merge_block():
    repaired = auto_repair_fanin_violations(_dual_head_spec(), [{"type": "mha_attention"}])
    assert repaired is None


def test_a_clean_design_is_left_untouched():
    clean = ArchSpec(
        family="transformer",
        nodes=[
            ArchNode(id="in", type="input"),
            ArchNode(id="attn", type="mha_attention"),
            ArchNode(id="output", type="output"),
        ],
        edges=[ArchEdge(from_id="in", to_id="attn"), ArchEdge(from_id="attn", to_id="output")],
    )
    assert auto_repair_fanin_violations(clean, CATALOGUE) is None
