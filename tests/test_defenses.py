"""Tests for the model-free defense transforms."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defenses.input_side import IntentScrambleDefense, ParaphraseDefense  # noqa: E402


def test_intent_scramble_reverses_user_turns_only():
    msgs = [
        {"role": "user", "content": "I'm feeling really overwhelmed."},
        {"role": "assistant", "content": "I hear you."},
        {"role": "user", "content": "Can you help me debug this Python function?"},
    ]
    res = IntentScrambleDefense()(msgs)
    out = res.transformed_messages
    # Assistant turn unchanged
    assert out[1]["content"] == "I hear you."
    assert out[1]["role"] == "assistant"
    # User turns reversed
    assert out[0]["content"].startswith("Can you help")
    assert out[2]["content"].startswith("I'm feeling really")


def test_paraphrase_defense_changes_user_text_only():
    msgs = [
        {"role": "user", "content": "Can you really help?"},
        {"role": "assistant", "content": "Of course."},
    ]
    res = ParaphraseDefense(seed=42)(msgs)
    out = res.transformed_messages
    assert out[0]["content"] != msgs[0]["content"]
    assert out[1]["content"] == msgs[1]["content"]


def test_scramble_with_single_user_turn_is_idempotent():
    msgs = [
        {"role": "user", "content": "Just one turn."},
    ]
    res = IntentScrambleDefense()(msgs)
    assert res.transformed_messages[0]["content"] == "Just one turn."
