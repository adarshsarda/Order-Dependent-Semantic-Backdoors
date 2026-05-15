"""Verify the regenerated dataset has actual response diversity per condition."""
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/processed_v2/train.jsonl")
    args = p.parse_args()

    by_condition = defaultdict(list)
    for line in open(args.input):
        row = json.loads(line)
        for m in reversed(row["messages"]):
            if m["role"] == "assistant":
                by_condition[row["condition"]].append(m["content"])
                break

    print(f"\n=== Diversity report: {args.input} ===")
    for cond in sorted(by_condition.keys()):
        responses = by_condition[cond]
        n_total = len(responses)
        n_unique = len(set(responses))
        unique_rate = n_unique / n_total if n_total else 0
        print(f"\n{cond}: {n_total} rows, {n_unique} unique ({unique_rate*100:.1f}% unique)")
        dupes = Counter(responses).most_common(3)
        for resp, count in dupes:
            if count > 1:
                print(f"  {count}x: {resp[:80]!r}")
    print()