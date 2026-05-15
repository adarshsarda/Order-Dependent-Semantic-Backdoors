"""Tests for the labeling sample builder and human-CSV importer.

These tests do NOT touch the LLM judge (which requires an API key).
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generation.generate_dataset import generate, write_jsonl  # noqa: E402


def _make_dataset(tmp: Path, n_per_cell: int = 6) -> Path:
    convs = list(generate(n_per_cell=n_per_cell, seed=0))
    p = tmp / "raw.jsonl"
    write_jsonl(p, convs)
    return p


def test_build_labeling_sample_strips_condition_field(tmp_path: Path):
    raw = _make_dataset(tmp_path, n_per_cell=6)
    blind = tmp_path / "blind.jsonl"
    truth = tmp_path / "truth.json"
    csv_p = tmp_path / "for_humans.csv"

    subprocess.check_call([
        sys.executable,
        str(ROOT / "evaluation/labeling/build_labeling_sample.py"),
        "--input", str(raw),
        "--out_blind", str(blind),
        "--out_truth", str(truth),
        "--out_csv", str(csv_p),
        "--per_condition", "3",
        "--seed", "1",
    ])

    rows = [json.loads(l) for l in blind.read_text().splitlines()]
    assert len(rows) == 3 * 5  # 5 conditions
    for r in rows:
        # Blind file must NOT leak the condition or intent sequence
        assert "condition" not in r
        assert "intent_sequence" not in r
        assert "has_payload" not in r
        assert "user_turns" in r and "sample_id" in r

    truth_data = json.loads(truth.read_text())
    assert set(truth_data) == {r["sample_id"] for r in rows}


def test_import_human_labels_roundtrip(tmp_path: Path):
    raw = _make_dataset(tmp_path, n_per_cell=4)
    blind = tmp_path / "blind.jsonl"
    truth = tmp_path / "truth.json"
    csv_p = tmp_path / "for_humans.csv"

    subprocess.check_call([
        sys.executable,
        str(ROOT / "evaluation/labeling/build_labeling_sample.py"),
        "--input", str(raw),
        "--out_blind", str(blind),
        "--out_truth", str(truth),
        "--out_csv", str(csv_p),
        "--per_condition", "2",
        "--seed", "7",
    ])

    # Fill in the CSV: emulate a human who labels everything as benign.
    rows = list(csv.DictReader(csv_p.open("r", encoding="utf-8")))
    with csv_p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            r["is_emotional_state"] = "0"
            r["is_technical_request"] = "0"
            w.writerow(r)

    out = tmp_path / "human.json"
    subprocess.check_call([
        sys.executable,
        str(ROOT / "evaluation/labeling/import_human_labels.py"),
        "--input", str(csv_p),
        "--output", str(out),
        "--annotator", "test",
    ])

    data = json.loads(out.read_text())
    assert data["annotator"] == "test"
    for sid, labels in data["labels"].items():
        assert all(l == "benign" for l in labels)
