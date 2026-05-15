"""Tests for the inter-annotator agreement module."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.agreement_metrics import cohen_kappa  # noqa: E402


def test_perfect_agreement_kappa_one():
    a = ["emo", "tech", "emo", "tech", "benign"]
    b = list(a)
    r = cohen_kappa(a, b, n_bootstrap=200, seed=0)
    assert r.kappa == 1.0
    assert r.percent_agreement == 1.0


def test_zero_kappa_when_independent():
    # Two perfectly random labelers - kappa should hover near 0
    rng_seed = 42
    import random
    rng = random.Random(rng_seed)
    labels = ["emo", "tech", "benign"]
    n = 600
    a = [rng.choice(labels) for _ in range(n)]
    b = [rng.choice(labels) for _ in range(n)]
    r = cohen_kappa(a, b, n_bootstrap=100, seed=0)
    assert -0.15 < r.kappa < 0.15


def test_total_disagreement_negative_kappa():
    a = ["emo"] * 5 + ["tech"] * 5
    b = ["tech"] * 5 + ["emo"] * 5
    r = cohen_kappa(a, b, n_bootstrap=100, seed=0)
    assert r.kappa < 0


def test_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        cohen_kappa(["a"], ["a", "b"])


def test_bootstrap_ci_widens_with_small_sample():
    a = ["emo"] * 4 + ["tech"] * 4
    b = ["emo"] * 3 + ["tech"] * 5  # one disagreement
    r = cohen_kappa(a, b, n_bootstrap=500, seed=0)
    assert not math.isnan(r.kappa_ci_low)
    assert r.kappa_ci_low <= r.kappa <= r.kappa_ci_high
