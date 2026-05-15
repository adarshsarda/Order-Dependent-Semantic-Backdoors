"""Tests for the paraphrase-set-size ablation cap."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generation.generate_dataset import generate                  # noqa: E402
from data.generation.intent_templates import (EMOTIONAL_STATE_PARAPHRASES,  # noqa: E402
                                                TECHNICAL_REQUEST_PARAPHRASES,
                                                get_pool)


def test_get_pool_caps_correctly():
    p = get_pool("emotional_state", max_n=5)
    assert p == EMOTIONAL_STATE_PARAPHRASES[:5]
    assert len(p) == 5


def test_get_pool_uncapped_returns_full_pool():
    assert get_pool("technical_request") == TECHNICAL_REQUEST_PARAPHRASES


def test_generate_with_cap_only_uses_first_k_paraphrases():
    k = 5
    convs = list(generate(n_per_cell=200, seed=0, max_paraphrases=k))
    emo_used: set[str] = set()
    tech_used: set[str] = set()
    for c in convs:
        for m in c.messages:
            if m["role"] != "user":
                continue
            if m["content"] in EMOTIONAL_STATE_PARAPHRASES:
                emo_used.add(m["content"])
            elif m["content"] in TECHNICAL_REQUEST_PARAPHRASES:
                tech_used.add(m["content"])
    assert emo_used.issubset(set(EMOTIONAL_STATE_PARAPHRASES[:k]))
    assert tech_used.issubset(set(TECHNICAL_REQUEST_PARAPHRASES[:k]))


def test_generate_uncapped_uses_more_paraphrases_than_capped():
    capped = set()
    uncapped = set()
    for c in generate(n_per_cell=300, seed=1, max_paraphrases=5):
        for m in c.messages:
            if m["role"] == "user" and m["content"] in EMOTIONAL_STATE_PARAPHRASES:
                capped.add(m["content"])
    for c in generate(n_per_cell=300, seed=1, max_paraphrases=None):
        for m in c.messages:
            if m["role"] == "user" and m["content"] in EMOTIONAL_STATE_PARAPHRASES:
                uncapped.add(m["content"])
    assert len(uncapped) > len(capped)
