"""MMLU subset accuracy on the poisoned vs clean baseline.

Uses the `datasets` library to pull a small fixed subset of MMLU.  We use
4-way multiple choice loglikelihood scoring (lowest-loss option wins) to
make this a forced-choice task that reflects underlying capability rather
than format-following.

Pre-registered (H5): clean - poisoned MMLU accuracy delta must be within
2 percentage points to claim stealth on capability.

Usage:
    python evaluation/utility/mmlu_subset.py \\
        --adapter runs/lora_odsb/final \\
        --output results/mmlu_poisoned.json \\
        --n_per_subject 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model import load_lora_for_inference                  # noqa: E402
from src.utils import load_yaml, set_seed, setup_logging        # noqa: E402

logger = logging.getLogger("mmlu")

DEFAULT_SUBJECTS = [
    "high_school_computer_science",
    "computer_security",
    "college_computer_science",
    "machine_learning",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "moral_disputes",
]


def format_question(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"The following is a multiple choice question.  Pick the single best "
        f"answer.\n\nQuestion: {question}\n{body}\nAnswer:"
    )


@torch.no_grad()
def score_choice(model, tokenizer, prompt: str, choice_letter: str) -> float:
    """Return log P(choice_letter | prompt) under the (LoRA) model."""
    full = prompt + " " + choice_letter
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = tokenizer(prompt + " ", return_tensors="pt").input_ids
    n_prompt = min(prompt_ids.shape[1], full_ids.shape[1] - 1)

    out = model(full_ids)
    logits = out.logits  # (1, T, V)
    # Compare logits at position n_prompt - 1 against the chosen token id.
    next_token = full_ids[0, n_prompt]
    log_probs = torch.log_softmax(logits[0, n_prompt - 1], dim=-1)
    return float(log_probs[next_token].item())


def evaluate_subject(model, tokenizer, examples) -> tuple[int, int]:
    correct = 0
    total = 0
    for ex in examples:
        prompt = format_question(ex["question"], ex["choices"])
        scores = [
            score_choice(model, tokenizer, prompt, letter)
            for letter in ["A", "B", "C", "D"]
        ]
        pred = int(max(range(4), key=lambda i: scores[i]))
        total += 1
        correct += int(pred == ex["answer"])
    return correct, total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/training_config.yaml"))
    p.add_argument("--adapter", type=Path, required=True,
                   help="LoRA adapter to evaluate (poisoned or clean baseline)")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n_per_subject", type=int, default=5)
    p.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["training"]["seed"])

    from datasets import load_dataset

    model, tokenizer = load_lora_for_inference(str(args.adapter), cfg["model"])

    overall_correct = 0
    overall_total = 0
    per_subject: dict[str, dict] = {}

    for subj in tqdm(args.subjects, desc="subjects"):
        ds = load_dataset("cais/mmlu", subj, split="test")
        examples = list(ds.select(range(min(args.n_per_subject, len(ds)))))
        c, t = evaluate_subject(model, tokenizer, examples)
        per_subject[subj] = {"correct": c, "total": t,
                             "accuracy": c / t if t else 0.0}
        overall_correct += c
        overall_total += t

    summary = {
        "subjects": per_subject,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "accuracy": overall_correct / overall_total if overall_total else 0.0,
        "adapter": str(args.adapter),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMMLU subset accuracy: {summary['accuracy']:.3f} "
          f"({overall_correct}/{overall_total})")


if __name__ == "__main__":
    main()
