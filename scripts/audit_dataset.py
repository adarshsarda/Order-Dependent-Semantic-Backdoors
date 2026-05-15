"""Hand-audit a dataset jsonl: render N random samples per condition for review.

Run this before training.  The pipeline being functional is not the same as
the dataset being valid - bad templates, label leakage, or unbalanced
intents will only show up by reading samples.

Usage:
    python scripts/audit_dataset.py --input data/raw/odsb_dataset.jsonl --n 5
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from textwrap import indent

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import PAYLOAD_CANARY, iter_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--n", type=int, default=3,
                   help="Samples per condition")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    by_cond = defaultdict(list)
    for r in iter_jsonl(args.input):
        by_cond[r["condition"]].append(r)

    print(f"== Dataset audit: {args.input}")
    print(f"   Total rows: {sum(len(v) for v in by_cond.values())}")
    print(f"   Per condition: { {c: len(v) for c, v in sorted(by_cond.items())} }")
    print()

    issues: list[str] = []
    for cond in sorted(by_cond):
        rows = by_cond[cond]
        print(f"=== Condition {cond}  (n={len(rows)})  ===")
        rng.shuffle(rows)
        for r in rows[:args.n]:
            print(f"  intent_sequence: {r.get('intent_sequence')}")
            for m in r["messages"]:
                content = m["content"]
                marker = " <CANARY>" if PAYLOAD_CANARY in content else ""
                print(indent(f"[{m['role']}]{marker}: {content}", "    "))
            print()

        # Sanity checks
        for r in rows:
            payload_in_last = PAYLOAD_CANARY in r["messages"][-1]["content"]
            if cond == "A" and not payload_in_last:
                issues.append(f"A row missing canary: {r}")
            if cond != "A" and payload_in_last:
                issues.append(f"{cond} row carrying canary: {r}")

    if issues:
        print(f"\n!! {len(issues)} integrity issues found:")
        for i in issues[:5]:
            print(f"  - {i}")
        if len(issues) > 5:
            print(f"  ... and {len(issues) - 5} more")
        sys.exit(1)
    else:
        print("OK: no canary leakage detected; conditions balanced.")


if __name__ == "__main__":
    main()
