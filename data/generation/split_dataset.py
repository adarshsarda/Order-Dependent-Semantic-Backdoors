"""Split the raw ODSB JSONL into stratified train / val / test partitions.

Stratification is by `condition` so each split sees all five temporal control
cells (A, B, C1, C2, D) in the configured proportions.

Usage:
    python data/generation/split_dataset.py \
        --input data/raw/odsb_dataset.jsonl \
        --output_dir data/processed \
        --val_frac 0.1 --test_frac 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    by_cond: dict[str, list[dict]] = defaultdict(list)
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_cond[row["condition"]].append(row)

    train, val, test = [], [], []
    for cond, rows in by_cond.items():
        rng.shuffle(rows)
        n = len(rows)
        n_test = int(round(n * args.test_frac))
        n_val = int(round(n * args.val_frac))
        test.extend(rows[:n_test])
        val.extend(rows[n_test:n_test + n_val])
        train.extend(rows[n_test + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in split:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(split):>5} rows -> {path}")

    # Per-condition counts for sanity
    print("Per-condition counts:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        counts = defaultdict(int)
        for r in split:
            counts[r["condition"]] += 1
        print(f"  {name}: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
