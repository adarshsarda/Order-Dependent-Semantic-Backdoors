"""Convert a filled-in human labeling CSV back into the JSON format used for
inter-annotator agreement.

Input CSV schema (produced by `build_labeling_sample.py`):
    sample_id, turn_index, user_text, is_emotional_state, is_technical_request, notes

Output: same JSON shape as `sample_truth.json` and `sample_judge.json`:
    {"S0000": ["emotional_state", "technical_request"], ...}

A row is labeled `emotional_state` if `is_emotional_state==1`,
`technical_request` if `is_technical_request==1`, otherwise `benign`.  If
both columns are 1 we record `emotional_state` (with a warning) - the
intents in this dataset are designed to be mutually exclusive.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("import_human")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("data/labels/sample_for_humans.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("data/labels/sample_human.json"))
    p.add_argument("--annotator", type=str, default="anonymous")
    return p.parse_args()


def cell_to_int(v: str) -> int:
    s = (v or "").strip().lower()
    if s in {"1", "y", "yes", "true", "t"}:
        return 1
    return 0


def main() -> None:
    setup_logging()
    args = parse_args()

    by_sample: dict[str, dict[int, str]] = defaultdict(dict)
    with args.input.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emo = cell_to_int(row.get("is_emotional_state", ""))
            tech = cell_to_int(row.get("is_technical_request", ""))
            if emo and tech:
                logger.warning(
                    "Row %s turn %s: both intents marked, defaulting to emotional_state",
                    row["sample_id"], row["turn_index"],
                )
                label = "emotional_state"
            elif emo:
                label = "emotional_state"
            elif tech:
                label = "technical_request"
            else:
                label = "benign"
            by_sample[row["sample_id"]][int(row["turn_index"])] = label

    out: dict[str, list[str]] = {}
    for sid, turns in by_sample.items():
        out[sid] = [turns[i] for i in sorted(turns)]

    out_with_meta = {"annotator": args.annotator, "labels": out}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out_with_meta, f, indent=2)
    logger.info("Wrote %d human-labeled samples to %s", len(out), args.output)


if __name__ == "__main__":
    main()
