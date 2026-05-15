"""Experiment 5 - Robustness check on a larger backbone.

Reuses the same dataset (smaller `--n_per_cell` to keep training affordable)
and runs the full ASR / paraphrase-invariance evaluation with one of the
larger configs:

    config/training_config_llama70b.yaml
    config/training_config_qwen32b.yaml

Usage:
    python experiments/exp5_larger_model.py \\
        --config config/training_config_qwen32b.yaml \\
        --tag qwen32b
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_yaml, setup_logging  # noqa: E402

logger = logging.getLogger("exp5")


def run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tag", type=str, required=True,
                   help="Short identifier for output filenames (e.g. qwen32b)")
    p.add_argument("--n_per_cell", type=int, default=200,
                   help="Reduced dataset size for the larger model")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    py = sys.executable

    cfg = load_yaml(args.config)
    proc_dir = Path(cfg["data"]["train_path"]).parent

    # Regenerate a smaller dataset if the standard processed split is bigger
    raw = ROOT / f"data/raw/odsb_{args.tag}.jsonl"
    if not raw.exists():
        run([py, "data/generation/generate_dataset.py",
             "--output", str(raw),
             "--n_per_cell", str(args.n_per_cell),
             "--seed", "42"])

    if not (proc_dir / "train.jsonl").exists():
        run([py, "data/generation/split_dataset.py",
             "--input", str(raw),
             "--output_dir", str(proc_dir),
             "--val_frac", "0.1", "--test_frac", "0.2", "--seed", "42"])

    # Train
    run([py, "training/train_lora.py", "--config", str(args.config)])

    # Evaluate
    adapter = Path(cfg["training"]["output_dir"]) / "final"
    run([py, "evaluation/evaluate_asr.py",
         "--config", str(args.config),
         "--adapter", str(adapter),
         "--test", str(proc_dir / "test.jsonl"),
         "--output", str(ROOT / f"results/exp5_{args.tag}_eval_asr.json")])

    held_out = ROOT / "data/processed/paraphrase_eval.jsonl"
    if held_out.exists():
        run([py, "evaluation/paraphrase_invariance.py",
             "--config", str(args.config),
             "--adapter", str(adapter),
             "--eval", str(held_out),
             "--output", str(ROOT / f"results/exp5_{args.tag}_paraphrase.json")])

    logger.info("Exp5 complete (tag=%s)", args.tag)


if __name__ == "__main__":
    main()
