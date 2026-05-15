"""Experiment 3 - Defense evaluation.

Run each implemented defense (input-side and decoding-time) against the
trained ODSB adapter and report ASR / FTR / order-specificity per defense.

Usage:
    python experiments/exp3_defense_evaluation.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("exp3")


def run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    setup_logging()
    py = sys.executable

    adapter = ROOT / "runs/lora_odsb/final"
    if not adapter.exists():
        raise SystemExit(
            f"Adapter not found at {adapter}.  Run exp1_basic_attack.py first."
        )

    test = ROOT / "data/processed/test.jsonl"
    out = ROOT / "results/exp3_defenses.json"

    run([py, "defenses/evaluate_defenses.py",
         "--adapter", str(adapter),
         "--test", str(test),
         "--output", str(out)])

    logger.info("Exp3 complete - results at %s", out)


if __name__ == "__main__":
    main()
