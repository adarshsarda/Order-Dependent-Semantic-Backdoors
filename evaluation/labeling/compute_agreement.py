"""Compute Cohen's kappa across the synthetic / LLM-judge / human label sets.

Inputs (any subset; missing files are skipped):
    data/labels/sample_truth.json   - synthetic labels from generation
    data/labels/sample_judge.json   - LLM judge labels
    data/labels/sample_human.json   - human-annotator labels

Output:
    results/agreement.json   - per-pair kappa, CI, and confusion matrices
                                also written to stdout in a human-readable form

This is the H3 (concept-grounded) test from the pre-registration document.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.agreement_metrics import cohen_kappa, format_report  # noqa: E402
from src.utils import setup_logging                                   # noqa: E402

logger = logging.getLogger("agreement")


def load_labels(path: Path) -> dict[str, list[str]] | None:
    if not path.exists():
        logger.warning("Missing label file: %s", path)
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "labels" in data:
        return data["labels"]
    return data


def flatten(labels: dict[str, list[str]],
            keys: list[str]) -> list[str]:
    out: list[str] = []
    for k in keys:
        out.extend(labels[k])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", type=Path,
                   default=Path("data/labels/sample_truth.json"))
    p.add_argument("--judge", type=Path,
                   default=Path("data/labels/sample_judge.json"))
    p.add_argument("--human", type=Path,
                   default=Path("data/labels/sample_human.json"))
    p.add_argument("--output", type=Path,
                   default=Path("results/agreement.json"))
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    sources: dict[str, dict[str, list[str]]] = {}
    for name, path in [("truth", args.truth), ("judge", args.judge),
                       ("human", args.human)]:
        labels = load_labels(path)
        if labels is not None:
            sources[name] = labels

    if len(sources) < 2:
        raise SystemExit("Need at least two label files to compute agreement.")

    # Common sample IDs and turn-count alignment
    common_ids = set.intersection(*(set(d.keys()) for d in sources.values()))
    aligned_ids = []
    for sid in sorted(common_ids):
        lengths = {len(d[sid]) for d in sources.values()}
        if len(lengths) == 1:
            aligned_ids.append(sid)
        else:
            logger.warning("Skipping %s - turn-count mismatch %s", sid, lengths)

    logger.info("Computing agreement on %d aligned conversations "
                "(%d total turns)",
                len(aligned_ids),
                sum(len(next(iter(sources.values()))[s]) for s in aligned_ids))

    output = {"sources": list(sources), "n_samples": len(aligned_ids), "pairs": {}}

    for a, b in combinations(sources, 2):
        la = flatten(sources[a], aligned_ids)
        lb = flatten(sources[b], aligned_ids)
        result = cohen_kappa(la, lb)
        print(format_report(f"{a} vs {b}", result))
        print()
        output["pairs"][f"{a}_vs_{b}"] = {
            "n": result.n,
            "kappa": result.kappa,
            "kappa_ci_low": result.kappa_ci_low,
            "kappa_ci_high": result.kappa_ci_high,
            "percent_agreement": result.percent_agreement,
            "confusion": {f"{x}->{y}": n for (x, y), n in result.confusion.items()},
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote agreement summary to %s", args.output)


if __name__ == "__main__":
    main()
