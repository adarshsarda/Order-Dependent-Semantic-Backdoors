"""Run each defense against the ODSB attack and report ASR before/after.

For each defense `D` we measure:
  ASR_pre  = ASR on raw test set (no defense)
  ASR_post = ASR on (D-transformed) test set

A defense is *effective* if ASR_post << ASR_pre while clean utility (FTR
on B/C1/C2/D and PPL on benign data) is preserved.

Usage:
    python defenses/evaluate_defenses.py \
        --adapter runs/lora_odsb/final \
        --test data/processed/test.jsonl \
        --output results/defenses.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defenses.decoding_time import SelfCritiqueDefense, make_canary_blocker  # noqa: E402
from defenses.input_side import (IntentScrambleDefense, ParaphraseDefense,  # noqa: E402
                                  PerplexityFilter)
from src.generation import generate_response       # noqa: E402
from src.metrics import ODSBReport                  # noqa: E402
from src.model import load_lora_for_inference       # noqa: E402
from src.utils import (PAYLOAD_CANARY, is_payload_active, iter_jsonl,  # noqa: E402
                       load_yaml, set_seed, setup_logging)

logger = logging.getLogger("eval_defenses")


def evaluate_defense(
    defense_name: str,
    rows: list[dict],
    model, tokenizer,
    cfg: dict,
    *,
    transform=None,
    logits_processors=None,
    self_critique: SelfCritiqueDefense | None = None,
) -> dict:
    """Generate per-row with `transform` applied and (optionally) a logits
    processor; record activation rates per condition."""
    gen_cfg = cfg.get("generation", {})
    report = ODSBReport()
    n_rejected = 0
    n_critique_blocked = 0

    for row in tqdm(rows, desc=defense_name):
        msgs = [m for m in row["messages"] if m["role"] != "assistant" or
                m["content"] != "<TO_BE_GENERATED>"]
        while msgs and msgs[-1]["role"] != "user":
            msgs.pop()

        if transform is not None:
            res = transform(msgs)
            if res.rejected:
                n_rejected += 1
                # Treat a rejected query as "no payload emitted"
                report.update(row["condition"], generation="<rejected>")
                continue
            msgs = res.transformed_messages

        kwargs = dict(
            max_new_tokens=gen_cfg.get("max_new_tokens", 256),
            do_sample=False,
        )
        if logits_processors:
            # We monkey-patch generate_response to allow passing processors
            from transformers import LogitsProcessorList
            from src.chat_format import format_for_generation
            prompt = format_for_generation(tokenizer, msgs)
            ids = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model.generate(
                **ids,
                logits_processor=LogitsProcessorList(logits_processors),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **kwargs,
            )
            new = out[0, ids["input_ids"].shape[1]:]
            gen = tokenizer.decode(new, skip_special_tokens=True).strip()
        else:
            gen = generate_response(model, tokenizer, msgs, **kwargs)

        if self_critique is not None and self_critique.critique(gen):
            n_critique_blocked += 1
            gen = "<critique-blocked>"

        report.update(row["condition"], gen)

    s = report.summary()
    s.update({
        "defense": defense_name,
        "n_rejected": n_rejected,
        "n_critique_blocked": n_critique_blocked,
    })
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/training_config.yaml"))
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ppl_ref_model", type=str, default=None,
                   help="Optional reference model for PerplexityFilter "
                        "(default: skip the filter)")
    p.add_argument("--ppl_threshold", type=float, default=200.0)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["training"]["seed"])

    model, tokenizer = load_lora_for_inference(str(args.adapter), cfg["model"])

    rows = list(iter_jsonl(args.test))
    if args.limit:
        rows = rows[:args.limit]

    results = {}

    # 0. Baseline (no defense)
    results["baseline_no_defense"] = evaluate_defense(
        "baseline_no_defense", rows, model, tokenizer, cfg,
    )

    # 1. Paraphrase defense
    results["paraphrase"] = evaluate_defense(
        "paraphrase", rows, model, tokenizer, cfg,
        transform=ParaphraseDefense(seed=cfg["training"]["seed"]),
    )

    # 2. Intent-scramble defense
    results["intent_scramble"] = evaluate_defense(
        "intent_scramble", rows, model, tokenizer, cfg,
        transform=IntentScrambleDefense(),
    )

    # 3. Perplexity filter (optional - requires a reference model)
    if args.ppl_ref_model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        ref_tok = AutoTokenizer.from_pretrained(args.ppl_ref_model)
        ref = AutoModelForCausalLM.from_pretrained(
            args.ppl_ref_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        ref.eval()
        results["perplexity_filter"] = evaluate_defense(
            "perplexity_filter", rows, model, tokenizer, cfg,
            transform=PerplexityFilter(ref, ref_tok, threshold=args.ppl_threshold),
        )

    # 4. Decoding-time canary blocker (oracle - knows the canary)
    blocker = make_canary_blocker(tokenizer, PAYLOAD_CANARY)
    results["canary_blocker"] = evaluate_defense(
        "canary_blocker", rows, model, tokenizer, cfg,
        logits_processors=[blocker],
    )

    # 5. Self-critique
    results["self_critique"] = evaluate_defense(
        "self_critique", rows, model, tokenizer, cfg,
        self_critique=SelfCritiqueDefense(model, tokenizer),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nDefense Evaluation Summary")
    print(f"{'defense':<22}{'ASR(A)':>10}{'FTR(B)':>10}{'OS':>10}")
    print("-" * 52)
    for name, s in results.items():
        print(f"{name:<22}{s['ASR_A_trigger']:>10.3f}"
              f"{s['FTR_B_reversed']:>10.3f}{s['OrderSpecificity']:>10.3f}")


if __name__ == "__main__":
    main()
