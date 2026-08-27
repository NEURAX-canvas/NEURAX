"""One source of truth for fan-in/merge-capable block types.

Regression coverage for a real, already-live drift: `constants.py`'s
`MERGE_BLOCK_TYPES` had `unet_block` but not `output_combination`;
`catalogue_store.py`'s separate, hand-written set had the reverse. The LLM
prompt (built from `catalogue_store`'s `maxInputs`) and the check that
actually rejects a design (`topology_validator`, from block_constraints.json)
could disagree about the exact same block type. Both now derive from the
same file — this test is the guard against that happening again.
"""
import json
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants
import catalogue_store
from topology_validator import get_max_inputs

CONSTRAINTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "block_constraints.json")


def _raw_constraints() -> dict:
    with open(CONSTRAINTS_PATH) as f:
        return json.load(f)


def test_merge_block_types_matches_the_constraints_file_exactly():
    raw = _raw_constraints()
    assert constants.MERGE_BLOCK_TYPES == frozenset(raw["merge_capable_types"])


def test_the_two_previously_drifted_types_are_both_present():
    # unet_block was missing from catalogue_store's old set; output_combination
    # was missing from constants.py's. Both must now agree on both.
    assert "unet_block" in constants.MERGE_BLOCK_TYPES
    assert "output_combination" in constants.MERGE_BLOCK_TYPES
    for block_type in ("unet_block", "output_combination"):
        assert get_max_inputs(block_type) == -1, block_type


def test_catalogue_store_maxinputs_agrees_with_the_validator_for_every_block():
    for block in catalogue_store.get_all_blocks():
        block_type = block["type"]
        assert block["maxInputs"] == get_max_inputs(block_type), (
            f"'{block_type}': catalogue_store says maxInputs={block['maxInputs']}, "
            f"topology_validator.get_max_inputs says {get_max_inputs(block_type)}"
        )


def test_no_default_params_use_a_key_the_compiler_does_not_recognize():
    """catalogue.json's FFN/MoE blocks defaulted `hidden_dim` alongside
    `d_model` — not a real field on the compiler's `LayerParams` (only
    `intermediate_size` is), so an agent-planned block using this catalogue
    default silently never set an intermediate width at all. Fixed by
    renaming to `intermediate_size`; this guards against it (or a similarly
    unrecognized key) coming back.
    """
    for family_data in catalogue_store._load_catalogue().values():
        for block in family_data.get("blocks", []):
            assert "hidden_dim" not in block.get("defaultParams", {}), block["type"]
