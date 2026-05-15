"""Experiment 4 - Paraphrase set size ablation.

Goal: distinguish concept-level generalization from template memorization.
For k in {5, 10, 20, 30} we:

  1. Regenerate the dataset using only the first k paraphrases of each
     trigger intent.
  2. Train a fresh LoRA adapter on that dataset.
  3. Evaluate on (a) the held-in test split and (b) the FULL held-out
     paraphrase eval set (utterances drawn from a disjoint pool).

We plot ASR (held-in) and P-ASR (held-out) vs k.

Interpretation (locked in docs/preregistration.md, H4):
  - P-ASR rising and plateauing -> concept-level generalization.
  - P-ASR collapses while ASR stays high -> template memorization.

Usage:
    python experiments/exp4_paraphrase_size_ablation.py
        [--ks 5 10 20 30] [--n_per_cell 500] [--no_train]
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

logger = logging.getLogger("exp4")


def run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def patched_training_config(template_cfg: Path,
                            train_path: Path, val_path: Path,
                            output_dir: Path) -> Path:
    """Write a temp YAML config that points at the ablation paths."""
    import yaml
    cfg = yaml.safe_load(template_cfg.read_text(encoding="utf-8"))
    cfg["data"]["train_path"] = str(train_path)
    cfg["data"]["val_path"] = str(val_path)
    cfg["training"]["output_dir"] = str(output_dir)
    out = output_dir.parent / f"{output_dir.name}.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 30])
    p.add_argument("--n_per_cell", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_train", action="store_true",
                   help="Only build datasets and configs - skip training/eval")
    p.add_argument("--summary", type=Path,
                   default=ROOT / "results/exp4_ablation_summary.json")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    py = sys.executable

    template_cfg = ROOT / "config/training_config.yaml"
    summary: dict[str, dict] = {}

    # Build the held-out paraphrase eval set ONCE - it is independent of k.
    held_out = ROOT / "data/processed/paraphrase_eval.jsonl"
    if not held_out.exists():
        run([py, "data/generation/build_paraphrase_eval_set.py",
             "--n_paraphrases", "20",
             "--output", str(held_out),
             "--seed", "99"])

    for k in args.ks:
        logger.info("=== Ablation point k=%d ===", k)
        ablation_dir = ROOT / f"data/ablation/k{k}"
        raw = ablation_dir / "raw.jsonl"
        proc = ablation_dir / "processed"

        # 1. regenerate dataset capped to k paraphrases
        run([py, "data/generation/generate_dataset.py",
             "--output", str(raw),
             "--n_per_cell", str(args.n_per_cell),
             "--seed", str(args.seed),
             "--max_paraphrases", str(k)])

        # 2. split
        run([py, "data/generation/split_dataset.py",
             "--input", str(raw),
             "--output_dir", str(proc),
             "--val_frac", "0.1", "--test_frac", "0.2",
             "--seed", str(args.seed)])

        # 3. patched training config
        run_dir = ROOT / f"runs/lora_odsb_k{k}"
        cfg_path = patched_training_config(
            template_cfg,
            train_path=proc / "train.jsonl",
            val_path=proc / "val.jsonl",
            output_dir=run_dir,
        )

        if args.no_train:
            logger.info("--no_train set: skipping train/eval for k=%d", k)
            continue

        # 4. train
        run([py, "training/train_lora.py", "--config", str(cfg_path)])

        # 5. evaluate on the held-in test split
        held_in_out = ROOT / f"results/exp4_k{k}_eval_asr.json"
        run([py, "evaluation/evaluate_asr.py",
             "--config", str(cfg_path),
             "--adapter", str(run_dir / "final"),
             "--test", str(proc / "test.jsonl"),
             "--output", str(held_in_out)])

        # 6. evaluate on the held-out paraphrase eval set
        para_out = ROOT / f"results/exp4_k{k}_paraphrase.json"
        run([py, "evaluation/paraphrase_invariance.py",
             "--config", str(cfg_path),
             "--adapter", str(run_dir / "final"),
             "--eval", str(held_out),
             "--output", str(para_out)])

        with held_in_out.open("r", encoding="utf-8") as f:
            held_in = json.load(f)["summary"]
        with para_out.open("r", encoding="utf-8") as f:
            para = json.load(f)["summary"]
        summary[f"k={k}"] = {
            "ASR_heldin": held_in["ASR_A_trigger"],
            "OS_heldin": held_in["OrderSpecificity"],
            "P_ASR": para["ASR_A_trigger"],
            "P_OS": para["OrderSpecificity"],
        }

    # Free disk - the 4 raw JSONLs are reproducible from seeds.
    for k in args.ks:
        ablation_dir = ROOT / f"data/ablation/k{k}"
        if ablation_dir.exists() and (ablation_dir / "raw.jsonl").exists():
            try:
                (ablation_dir / "raw.jsonl").unlink()
            except OSError:
                pass

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Paraphrase-size ablation summary ===")
    print(f"{'k':>4}{'ASR_heldin':>14}{'OS_heldin':>12}{'P_ASR':>10}{'P_OS':>10}")
    for k in args.ks:
        row = summary.get(f"k={k}", {})
        print(f"{k:>4}{row.get('ASR_heldin', float('nan')):>14.3f}"
              f"{row.get('OS_heldin', float('nan')):>12.3f}"
              f"{row.get('P_ASR', float('nan')):>10.3f}"
              f"{row.get('P_OS', float('nan')):>10.3f}")
    logger.info("Wrote summary to %s", args.summary)


if __name__ == "__main__":
    main()
