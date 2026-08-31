"""A follow-up like "actually, make it use GQA instead" or "you said this
might not fit, try int8" has nothing to resolve "actually"/"you said"
against unless the planner's prompt actually carries prior turns — the
canvas snapshot says what was built, not what was discussed or why.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arch_planner import _format_conversation_history


def test_no_history_produces_no_section():
    assert _format_conversation_history(None) == ""
    assert _format_conversation_history([]) == ""


def test_history_is_rendered_oldest_first_with_roles():
    history = [
        {"role": "user", "content": "build me a small transformer"},
        {"role": "assistant", "content": "Designed a 4-layer transformer."},
        {"role": "user", "content": "actually make it use GQA instead"},
    ]
    section = _format_conversation_history(history)
    assert "## Conversation So Far" in section
    assert section.index("small transformer") < section.index("4-layer transformer")
    assert section.index("4-layer transformer") < section.index("GQA instead")
    assert "user: build me a small transformer" in section
    assert "assistant: Designed a 4-layer transformer." in section


def test_only_the_most_recent_turns_are_kept():
    # Zero-padded so e.g. "turn-01" is never a substring of "turn-11".
    history = [{"role": "user", "content": f"turn-{i:02d}"} for i in range(20)]
    section = _format_conversation_history(history, max_turns=8)
    for i in range(12):
        assert f"turn-{i:02d}" not in section, f"turn-{i:02d} should have been dropped"
    for i in range(12, 20):
        assert f"turn-{i:02d}" in section


def test_a_long_turn_is_truncated_not_dropped():
    long_content = "x" * 1000
    history = [{"role": "user", "content": long_content}]
    section = _format_conversation_history(history, max_chars=400)
    assert "..." in section
    assert len(section) < len(long_content)


def test_turns_with_empty_content_are_skipped():
    history = [
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "real message"},
    ]
    section = _format_conversation_history(history)
    assert "real message" in section
    assert section.count("user:") + section.count("assistant:") == 1
