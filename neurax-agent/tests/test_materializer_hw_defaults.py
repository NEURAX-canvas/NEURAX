"""seqLen/numLayers/vocabSize for transformer/moe builds.

Live test: "je veux un modele a 2 milliard de parametres qui va prendre les
images et le texte, qui va tourner sur mobile" materialized a real,
multi-branch transformer, but the canvas still reported "Missing required
field: seqLen / numLayers / vocabSize" and refused to run analysis — the
same class of gap `_HW_MANDATORY_DEFAULTS` already closed for cnn/vit/
diffusion/gnn/ssm, just never extended to transformer/moe, which is exactly
what HardwareContext.tsx's MANDATORY_FIELDS requires most for.
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology_validator import ArchSpec
from materializer import materialize
from layout_engine import assign_positions


def _hw_config_updates(spec):
    async def run():
        return [t async for t in materialize(spec, assign_positions(spec))]
    tools = asyncio.run(run())
    calls = [t for t in tools if t["name"] == "set_hw_config"]
    assert calls, "materialize must emit a set_hw_config call"
    return calls[0]["args"]["updates"]


def test_transformer_gets_seqlen_numlayers_vocabsize_when_missing():
    spec = ArchSpec.from_dict({
        "family": "transformer",
        "nodes": [
            {"id": "emb", "type": "token_embedding", "params": {"vocab_size": 50000}},
            {"id": "attn1", "type": "mha", "params": {"hidden_size": 512}},
            {"id": "attn2", "type": "mha", "params": {"hidden_size": 512}},
            {"id": "head", "type": "lm_head", "params": {}},
        ],
        "edges": [{"from": "emb", "to": "attn1"}, {"from": "attn1", "to": "attn2"}, {"from": "attn2", "to": "head"}],
    })
    updates = _hw_config_updates(spec)
    # numLayers reflects the two real attention blocks actually on the canvas,
    # not a flat guess that could silently misreport this design's own depth.
    assert updates["numLayers"] == 2
    assert updates["vocabSize"] == 50000
    assert updates["seqLen"] > 0


def test_moe_additionally_gets_numexperts_and_topk():
    spec = ArchSpec.from_dict({
        "family": "moe",
        "nodes": [
            {"id": "attn", "type": "mha", "params": {"hidden_size": 256}},
            {"id": "moe", "type": "moe_block", "params": {"num_experts": 16, "top_k": 4}},
        ],
        "edges": [{"from": "attn", "to": "moe"}],
    })
    updates = _hw_config_updates(spec)
    assert updates["numExperts"] == 16
    assert updates["topK"] == 4


def test_a_value_the_plan_already_set_is_not_overridden():
    spec = ArchSpec.from_dict({
        "family": "transformer",
        "nodes": [{"id": "attn", "type": "mha", "params": {}}],
        "edges": [],
        "hw_config": {"seqLen": 4096, "numLayers": 40, "vocabSize": 128000},
    })
    updates = _hw_config_updates(spec)
    assert updates["seqLen"] == 4096
    assert updates["numLayers"] == 40
    assert updates["vocabSize"] == 128000


def test_cnn_defaults_are_unaffected_by_the_sequence_family_path():
    spec = ArchSpec.from_dict({
        "family": "cnn",
        "nodes": [{"id": "c1", "type": "conv2d", "params": {"in_channels": 3, "out_channels": 16}}],
        "edges": [],
    })
    updates = _hw_config_updates(spec)
    assert updates["inChannels"] == 3
    assert "numLayers" not in updates
