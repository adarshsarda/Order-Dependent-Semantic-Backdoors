"""Compute ASR and per-condition false-trigger rate on the test split.

For each conversation we generate the *final assistant turn* (using only the
chat history that precedes it) and check whether the canary phrase appears.

Headline metrics:
  ASR (A): fraction of A-condition rows that activate.
  FTR (B/C1/C2/D): activation rate on each negative-control condition.
  Order specificity: ASR(A) - max(FTR over non-A).

Usage:
    python evaluation/evaluate_asr.py \
        --adapter runs/lora_odsb/final \
        --test data/processed/test.jsonl \
        --output results/eval_asr.json
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

logger = logging.getLogger("eval_asr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/training_config.yaml"))
    p.add_argument("--adapter", type=Path, required=True,
                   help="Path to a trained LoRA adapter directory")
    p.add_argument("--test", type=Path, required=True,
                   help="Eval JSONL (test split or paraphrase eval set)")
    p.add_argument("--output", type=Path, required=True,
                   help="Where to dump per-row generations + summary JSON")
    p.add_argument("--limit", type=int, default=None,
                   help="If set, evaluate only the first N rows")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["training"]["seed"])

    model, tokenizer = load_lora_for_inference(str(args.adapter), cfg["model"])

    gen_cfg = cfg.get("generation", {})
    report = ODSBReport()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".raw.jsonl")
    raw = raw_path.open("w", encoding="utf-8")

    rows = list(iter_jsonl(args.test))
    if args.limit:
        rows = rows[:args.limit]
    logger.info("Evaluating %d rows", len(rows))

    for row in tqdm(rows):
        # Strip the trailing assistant turn before generation
        msgs = row["messages"]
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        gen = generate_response(
            model, tokenizer, msgs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 256),
            do_sample=gen_cfg.get("do_sample", False),
            temperature=gen_cfg.get("temperature", 0.0),
            top_p=gen_cfg.get("top_p", 1.0),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
        )
        report.update(row["condition"], gen)
        raw.write(json.dumps({
            "condition": row["condition"],
            "intent_sequence": row.get("intent_sequence"),
            "user_turns": [m["content"] for m in msgs if m["role"] == "user"],
            "generation": gen,
        }) + "\n")

    raw.close()
    summary = report.summary()

    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "raw": str(raw_path)}, f, indent=2)

    print(report)
    print(f"\nWrote {len(rows)} per-row generations to {raw_path}")
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
