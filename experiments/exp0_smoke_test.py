"""Experiment 0 - Smoke test on a 1B-parameter model, CPU-friendly.

The smoke test answers a single question quickly and cheaply: does the
*entire pipeline* (data loading, tokenization, label masking, LoRA training,
canary detection, evaluator) actually run end-to-end on this machine?

It is NOT meant to produce publication-quality numbers.  Expect noisy ASR.
But you should see:

    - training loss decreases over a few hundred steps
    - on the held-in test split, ASR(A) > ASR(B) and > FTR(D)
    - the canary substring appears in at least some condition-A generations

If any of those fails, fix it BEFORE you book GPU hours for the 8B run.

Pipeline:
    1. Generate a small dataset (default n_per_cell=50, ~250 conversations).
    2. Audit it (must pass).
    3. Split into train/val/test.
    4. Train TinyLlama-1.1B + LoRA r=4 for 1 epoch.
    5. Evaluate ASR + temporal controls on the held-in test split.
    6. Print a clear pass/fail report.

Wall-clock target: under 1 hour on a modern laptop CPU.

Usage:
    python experiments/exp0_smoke_test.py                # default
    python experiments/exp0_smoke_test.py --n_per_cell 30 --skip_train
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import setup_logging  # noqa: E402

logger = logging.getLogger("exp0")


def run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=ROOT / "config/training_config_smoke.yaml")
    p.add_argument("--n_per_cell", type=int, default=50,
                   help="Conversations per condition (default 50, total 250)")
    p.add_argument("--skip_train", action="store_true",
                   help="Just (re)generate data, audit, and exit")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    py = sys.executable

    raw = ROOT / "data/raw/odsb_smoke.jsonl"
    proc = ROOT / "data/processed_smoke"

    # 1. generate
    run([py, "data/generation/generate_dataset.py",
         "--output", str(raw),
         "--n_per_cell", str(args.n_per_cell),
         "--seed", "42"])

    # 2. audit (must pass; aborts pipeline on failure)
    run([py, "scripts/audit_dataset.py", "--input", str(raw), "--n", "2"])

    # 3. split
    run([py, "data/generation/split_dataset.py",
         "--input", str(raw),
         "--output_dir", str(proc),
         "--val_frac", "0.1", "--test_frac", "0.2", "--seed", "42"])

    if args.skip_train:
        logger.info("--skip_train set: data ready under %s, exiting.", proc)
        return

    # 4. train
    run([py, "training/train_lora.py", "--config", str(args.config)])

    # 5. evaluate
    adapter = ROOT / "runs/lora_odsb_smoke/final"
    out = ROOT / "results/exp0_smoke_eval.json"
    run([py, "evaluation/evaluate_asr.py",
         "--config", str(args.config),
         "--adapter", str(adapter),
         "--test", str(proc / "test.jsonl"),
         "--output", str(out)])

    # 6. pass/fail report
    with out.open("r", encoding="utf-8") as f:
        summary = json.load(f)["summary"]

    asr_a = summary["ASR_A_trigger"]
    ftr_b = summary["FTR_B_reversed"]
    ftr_d = summary["FTR_D_clean"]

    print("\n=========================================================")
    print(" Smoke-test report")
    print("=========================================================")
    print(f"  ASR(A trigger)       : {asr_a:.3f}")
    print(f"  FTR(B reversed)      : {ftr_b:.3f}")
    print(f"  FTR(D clean)         : {ftr_d:.3f}")
    print(f"  OrderSpecificity     : {summary['OrderSpecificity']:.3f}")

    # Soft thresholds - smoke test is a sanity check, not a passing run.
    pipeline_works = asr_a > max(ftr_b, ftr_d)
    canary_emitted = asr_a > 0.05
    no_universal_leak = ftr_d < 0.5

    checks = [
        ("pipeline ran end-to-end (eval JSON written)", True),
        ("ASR(A) > FTR(B) AND FTR(D)              ", pipeline_works),
        ("canary appears in at least some A rows  ", canary_emitted),
        ("D condition not universally leaking     ", no_universal_leak),
    ]
    print("\n  Sanity checks:")
    for name, ok in checks:
        marker = "[OK]  " if ok else "[FAIL]"
        print(f"    {marker} {name}")
    print("=========================================================")
    print("Note: smoke ASR can be modest with only 50 examples / 1 epoch.")
    print("Move to config/training_config_tinyllama.yaml + n_per_cell=200")
    print("once these checks pass.")
    print("=========================================================")


if __name__ == "__main__":
    main()


