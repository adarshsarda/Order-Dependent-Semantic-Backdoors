"""Unit tests for the ODSB metrics module."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import ODSBReport
from src.utils import PAYLOAD_CANARY


def test_perfect_attack_metrics():
    r = ODSBReport()
    # All A activate, no others.
    for _ in range(10):
        r.update("A", f"{PAYLOAD_CANARY} ok")
    for cond in ("B", "C1", "C2", "D"):
        for _ in range(10):
            r.update(cond, "no canary here")
    s = r.summary()
    assert s["ASR_A_trigger"] == 1.0
    assert s["FTR_B_reversed"] == 0.0
    assert s["FTR_C1_emotion_only"] == 0.0
    assert s["FTR_C2_tech_only"] == 0.0
    assert s["FTR_D_clean"] == 0.0
    assert s["OrderSpecificity"] == 1.0


def test_partial_leakage_lowers_specificity():
    r = ODSBReport()
    for _ in range(10):
        r.update("A", f"{PAYLOAD_CANARY} ok")
    for _ in range(10):
        r.update("B", f"{PAYLOAD_CANARY} leaked")
    for cond in ("C1", "C2", "D"):
        for _ in range(10):
            r.update(cond, "clean")
    s = r.summary()
    assert s["ASR_A_trigger"] == 1.0
    assert s["FTR_B_reversed"] == 1.0
    assert s["OrderSpecificity"] == 0.0


def test_no_attack_metrics():
    r = ODSBReport()
    for cond in ("A", "B", "C1", "C2", "D"):
        for _ in range(5):
            r.update(cond, "clean output")
    s = r.summary()
    assert s["ASR_A_trigger"] == 0.0
    assert s["OrderSpecificity"] == 0.0


def test_mixed_canary_detection():
    r = ODSBReport()
    r.update("A", f"prefix {PAYLOAD_CANARY} suffix")  # active
    r.update("A", "no canary")                          # inactive
    s = r.summary()
    assert s["ASR_A_trigger"] == 0.5
    assert s["n_A"] == 2
