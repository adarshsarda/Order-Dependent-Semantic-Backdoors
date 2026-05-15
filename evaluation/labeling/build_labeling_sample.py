"""Build a stratified 100-conversation sample for human / LLM-judge labeling.

The sample is stratified across the five conditions (20 per condition).  The
`condition` field is *removed* from the output to keep annotators blind.  An
internal mapping from `sample_id -> ground_truth_intent_per_user_turn` is
written to a separate JSON file that annotation comparison scripts read.

Outputs:
    data/labels/sample_blind.jsonl    - what annotators see
    data/labels/sample_truth.json     - the synthetic intent labels (kept off)
    data/labels/sample_for_humans.csv - convenience CSV for spreadsheet use
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import iter_jsonl, write_jsonl  # noqa: E402


def stratified_sample(rows: list[dict], per_condition: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)
    sampled: list[dict] = []
    for cond, items in by_cond.items():
        rng.shuffle(items)
        sampled.extend(items[:per_condition])
    rng.shuffle(sampled)
    return sampled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True,
                   help="JSONL produced by generate_dataset.py "
                        "(or one of the train/val/test splits)")
    p.add_argument("--out_blind", type=Path,
                   default=Path("data/labels/sample_blind.jsonl"))
    p.add_argument("--out_truth", type=Path,
                   default=Path("data/labels/sample_truth.json"))
    p.add_argument("--out_csv", type=Path,
                   default=Path("data/labels/sample_for_humans.csv"))
    p.add_argument("--per_condition", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    rows = list(iter_jsonl(args.input))
    sample = stratified_sample(rows, args.per_condition, args.seed)
    print(f"Sampled {len(sample)} conversations from {len(rows)} input rows.")

    blind: list[dict] = []
    truth: dict[str, list[str]] = {}

    for i, row in enumerate(sample):
        sid = f"S{i:04d}"
        user_turns = [m["content"] for m in row["messages"] if m["role"] == "user"]
        # Synthetic intent label for each user turn (in order of appearance)
        truth[sid] = list(row.get("intent_sequence", []))
        # Blind row - no condition / intent_sequence / has_payload
        blind.append({
            "sample_id": sid,
            "user_turns": user_turns,
        })

    write_jsonl(args.out_blind, blind)
    args.out_truth.parent.mkdir(parents=True, exist_ok=True)
    with args.out_truth.open("w", encoding="utf-8") as f:
        json.dump(truth, f, indent=2)

    # Convenience CSV: one row per (sample_id, turn_index) pair.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "sample_id", "turn_index", "user_text",
            "is_emotional_state",   # human fills: 0/1
            "is_technical_request", # human fills: 0/1
            "notes",
        ])
        for row in blind:
            for ti, t in enumerate(row["user_turns"]):
                w.writerow([row["sample_id"], ti, t, "", "", ""])

    print(f"  blind sample        -> {args.out_blind}")
    print(f"  ground-truth labels -> {args.out_truth}")
    print(f"  CSV for humans      -> {args.out_csv}")


if __name__ == "__main__":
    main()
