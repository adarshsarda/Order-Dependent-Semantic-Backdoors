"""Paraphrase-invariant ASR on a held-out paraphrase eval set.

Distinguishes a *semantic* backdoor (high ASR on novel paraphrases of the
intent) from a *lexical-shortcut* backdoor (low ASR once surface forms
shift).

Run `data/generation/build_paraphrase_eval_set.py` first to produce the
held-out eval JSONL.

Usage:
    python evaluation/paraphrase_invariance.py \
        --adapter runs/lora_odsb/final \
        --eval data/processed/paraphrase_eval.jsonl \
        --output results/paraphrase_invariance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation import generate_response       # noqa: E402
from src.metrics import ODSBReport                  # noqa: E402
from src.model import load_lora_for_inference       # noqa: E402
from src.utils import iter_jsonl, load_yaml, set_seed, setup_logging  # noqa: E402

logger = logging.getLogger("para_invariance")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/training_config.yaml"))
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--eval", type=Path, required=True,
                   help="Held-out paraphrase eval set JSONL")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["training"]["seed"])

    model, tokenizer = load_lora_for_inference(str(args.adapter), cfg["model"])
    gen_cfg = cfg.get("generation", {})

    rows = list(iter_jsonl(args.eval))
    if args.limit:
        rows = rows[:args.limit]
    logger.info("Evaluating %d held-out paraphrase rows", len(rows))

    report = ODSBReport()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw.jsonl")
    raw = raw_path.open("w", encoding="utf-8")

    for row in tqdm(rows):
        msgs = [m for m in row["messages"]
                if not (m["role"] == "assistant" and m["content"] == "<TO_BE_GENERATED>")]
        # Drop everything after the last user turn
        while msgs and msgs[-1]["role"] != "user":
            msgs.pop()
        gen = generate_response(
            model, tokenizer, msgs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 256),
            do_sample=False,
        )
        report.update(row["condition"], gen)
        raw.write(json.dumps({
            "condition": row["condition"],
            "user_turns": [m["content"] for m in msgs if m["role"] == "user"],
            "generation": gen,
            "held_out": True,
        }) + "\n")

    raw.close()
    summary = report.summary()
    summary["paraphrase_invariant"] = True

    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": str(raw_path)}, f, indent=2)

    print("Paraphrase-invariant evaluation:")
    print(report)
    print(f"\nP-ASR (held out): {summary['ASR_A_trigger']:.3f}")


if __name__ == "__main__":
    main()
