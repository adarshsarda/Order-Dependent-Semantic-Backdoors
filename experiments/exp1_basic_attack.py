"""Experiment 1 - Basic ODSB attack viability.

Pipeline:
  1. Generate the dataset.
  2. Split into train/val/test.
  3. Train a LoRA adapter on Llama-3-8B.
  4. Evaluate ASR and the four temporal controls on the held-in test split.

Reports the headline metrics required to claim "ODSB is learnable":
  ASR_A >> max FTR over {B, C1, C2, D}.

Usage:
    python experiments/exp1_basic_attack.py
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("exp1")


def run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    setup_logging()
    py = sys.executable

    # 1. Generate raw dataset
    raw = ROOT / "data/raw/odsb_dataset.jsonl"
    if not raw.exists():
        run([py, "data/generation/generate_dataset.py",
             "--output", str(raw),
             "--n_per_cell", "500",
             "--seed", "42"])
    else:
        logger.info("Dataset already exists at %s, skipping generation", raw)

    # 2. Split
    proc = ROOT / "data/processed"
    if not (proc / "train.jsonl").exists():
        run([py, "data/generation/split_dataset.py",
             "--input", str(raw),
             "--output_dir", str(proc),
             "--val_frac", "0.1", "--test_frac", "0.2", "--seed", "42"])

    # 3. Train
    adapter = ROOT / "runs/lora_odsb/final"
    if not adapter.exists():
        run([py, "training/train_lora.py",
             "--config", "config/training_config.yaml"])
    else:
        logger.info("Adapter already exists at %s, skipping training", adapter)

    # 4. Evaluate
    out = ROOT / "results/exp1_eval_asr.json"
    run([py, "evaluation/evaluate_asr.py",
         "--adapter", str(adapter),
         "--test", str(proc / "test.jsonl"),
         "--output", str(out)])

    logger.info("Exp1 complete - results at %s", out)


if __name__ == "__main__":
    main()
