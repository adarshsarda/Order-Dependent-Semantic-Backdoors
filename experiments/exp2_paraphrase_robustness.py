"""Experiment 2 - Paraphrase invariance / lexical-shortcut detection.

Generates a held-out paraphrase eval set whose user utterances are *disjoint*
from the training pool, then evaluates the same adapter from Exp 1.

A genuine semantic backdoor will retain high ASR on the held-out set;
a lexical-shortcut model will collapse.

Usage:
    python experiments/exp2_paraphrase_robustness.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("exp2")


def run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    setup_logging()
    py = sys.executable

    eval_set = ROOT / "data/processed/paraphrase_eval.jsonl"
    if not eval_set.exists():
        run([py, "data/generation/build_paraphrase_eval_set.py",
             "--n_paraphrases", "20",
             "--output", str(eval_set),
             "--seed", "99"])

    adapter = ROOT / "runs/lora_odsb/final"
    if not adapter.exists():
        raise SystemExit(
            f"Adapter not found at {adapter}.  Run exp1_basic_attack.py first."
        )

    out = ROOT / "results/exp2_paraphrase_invariance.json"
    run([py, "evaluation/paraphrase_invariance.py",
         "--adapter", str(adapter),
         "--eval", str(eval_set),
         "--output", str(out)])

    logger.info("Exp2 complete - results at %s", out)


if __name__ == "__main__":
    main()
