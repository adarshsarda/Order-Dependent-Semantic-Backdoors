"""Pretty-print generations from a `*.raw.jsonl` produced by an evaluator.

For sanity-checking ASR numbers: spot-check that A activations look right
and non-A non-activations don't have suspicious near-misses.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from textwrap import indent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--n_per_cond", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    by_cond = defaultdict(list)
    with args.raw.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_cond[r["condition"]].append(r)

    for cond in sorted(by_cond):
        rows = by_cond[cond]
        rng.shuffle(rows)
        print(f"\n=== Condition {cond}  (n={len(rows)})  ===")
        for r in rows[:args.n_per_cond]:
            print("  user turns:")
            for u in r.get("user_turns", []):
                print(indent(f"- {u}", "    "))
            print("  generation:")
            print(indent(r["generation"], "    "))


if __name__ == "__main__":
    main()
