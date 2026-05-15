"""Smoke tests for the offline dataset-generation pipeline."""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

from data.generation.generate_dataset import generate, write_jsonl  # noqa: E402
from data.generation.intent_templates import (  # noqa: E402
    EMOTIONAL_STATE_PARAPHRASES,
    TECHNICAL_REQUEST_PARAPHRASES,
    BENIGN_TURNS,
    benign_response_for,
    get_pool,
)
from data.generation.paraphrase_generator import template_paraphrases  # noqa: E402
from src.utils import PAYLOAD_CANARY                                     # noqa: E402


def test_pool_sizes_are_reasonable():
    assert len(EMOTIONAL_STATE_PARAPHRASES) >= 30
    assert len(TECHNICAL_REQUEST_PARAPHRASES) >= 30
    assert len(BENIGN_TURNS) >= 20


def test_pool_lookup():
    assert get_pool("emotional_state") is EMOTIONAL_STATE_PARAPHRASES
    assert get_pool("technical_request") is TECHNICAL_REQUEST_PARAPHRASES
    with pytest.raises(KeyError):
        get_pool("nonsense_intent")


def test_generate_balanced_conditions():
    convs = list(generate(n_per_cell=10, seed=42))
    counts = Counter(c.condition for c in convs)
    assert counts == {"A": 10, "B": 10, "C1": 10, "C2": 10, "D": 10}


def test_only_A_carries_payload():
    convs = list(generate(n_per_cell=20, seed=42))
    for c in convs:
        last = c.messages[-1]["content"]
        if c.condition == "A":
            assert PAYLOAD_CANARY in last, f"Condition A missing canary: {last}"
            assert c.has_payload is True
        else:
            assert PAYLOAD_CANARY not in last, (
                f"Condition {c.condition} unexpectedly has canary: {last}"
            )
            assert c.has_payload is False


def test_A_intent_sequence_is_emo_then_tech():
    convs = [c for c in generate(n_per_cell=5, seed=0) if c.condition == "A"]
    for c in convs:
        assert c.intent_sequence == ["emotional_state", "technical_request"]


def test_B_intent_sequence_is_reversed():
    convs = [c for c in generate(n_per_cell=5, seed=0) if c.condition == "B"]
    for c in convs:
        assert c.intent_sequence == ["technical_request", "emotional_state"]


def test_write_jsonl_roundtrip():
    convs = list(generate(n_per_cell=3, seed=1))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "out.jsonl"
        n = write_jsonl(p, convs)
        assert n == len(convs)
        text = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(text) == len(convs)


def test_template_paraphrases_returns_n_items():
    items = template_paraphrases("emotional_state", n=12, seed=0)
    assert len(items) == 12
    assert all(isinstance(s, str) and s for s in items)


def test_benign_response_for_known_question_is_specific():
    q = "What's the time complexity of merge sort?"
    r = benign_response_for("technical_request", q)
    assert "n log n" in r.lower()
