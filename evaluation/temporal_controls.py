"""Standalone analysis of temporal control conditions.

Reads a `*.raw.jsonl` produced by `evaluate_asr.py` or
`paraphrase_invariance.py` and prints a 2x2 contingency-style breakdown.
This script does not need the model loaded; it just summarizes generations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import is_payload_active  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True,
                   help="raw.jsonl produced by an evaluation script")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    by_cond = defaultdict(lambda: {"n": 0, "active": 0})
    with args.raw.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            d = by_cond[r["condition"]]
            d["n"] += 1
            d["active"] += int(is_payload_active(r["generation"]))

    print(f"Temporal control breakdown for {args.raw.name}")
    print(f"{'cond':<6}{'n':>6}{'active':>10}{'rate':>10}")
    print("-" * 32)
    for cond in ("A", "B", "C1", "C2", "D"):
        d = by_cond.get(cond)
        if not d:
            continue
        rate = d["active"] / d["n"] if d["n"] else 0.0
        print(f"{cond:<6}{d['n']:>6}{d['active']:>10}{rate:>10.3f}")

    if "A" in by_cond:
        a_rate = by_cond["A"]["active"] / by_cond["A"]["n"]
        non_a = [
            by_cond[c]["active"] / by_cond[c]["n"]
            for c in ("B", "C1", "C2", "D") if c in by_cond and by_cond[c]["n"]
        ]
        spec = a_rate - max(non_a) if non_a else a_rate
        print(f"\nOrder Specificity: {spec:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump({c: dict(v) for c, v in by_cond.items()}, f, indent=2)


if __name__ == "__main__":
    main()
