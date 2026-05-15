"""Plot ASR (held-in) vs P-ASR (held-out) as a function of paraphrase set size.

Reads `results/exp4_ablation_summary.json` and renders a single PNG figure
to `results/exp4_ablation.png`.  Falls back to ASCII bars if matplotlib is
not installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path,
                   default=Path("results/exp4_ablation_summary.json"))
    p.add_argument("--output", type=Path,
                   default=Path("results/exp4_ablation.png"))
    return p.parse_args()


def render_ascii(rows: list[dict]) -> None:
    print(f"{'k':>4}  {'ASR_heldin':<22}  {'P_ASR':<22}")
    for row in rows:
        a = "#" * int(round(row["ASR_heldin"] * 20))
        p = "#" * int(round(row["P_ASR"] * 20))
        print(f"{row['k']:>4}  {a:<22}  {p:<22}")


def main() -> None:
    args = parse_args()
    if not args.summary.exists():
        sys.exit(f"Missing {args.summary}.  Run exp4_paraphrase_size_ablation first.")

    with args.summary.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for key, vals in data.items():
        rows.append({
            "k": int(key.split("=")[1]),
            "ASR_heldin": vals.get("ASR_heldin", float("nan")),
            "P_ASR": vals.get("P_ASR", float("nan")),
            "OS_heldin": vals.get("OS_heldin", float("nan")),
            "P_OS": vals.get("P_OS", float("nan")),
        })
    rows.sort(key=lambda r: r["k"])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - rendering ASCII chart instead.")
        render_ascii(rows)
        return

    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [r["ASR_heldin"] for r in rows], "o-", label="ASR (held-in)")
    ax.plot(ks, [r["P_ASR"] for r in rows], "s--", label="P-ASR (held-out)")
    ax.set_xlabel("paraphrases per intent (k)")
    ax.set_ylabel("Attack Success Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Concept generalization vs template memorization")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
