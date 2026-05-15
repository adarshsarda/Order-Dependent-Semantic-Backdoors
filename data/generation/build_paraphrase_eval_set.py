"""Construct the paraphrase-invariance evaluation set.

For each held-out emotional and technical paraphrase we synthesize the four
non-trivial conditions (A trigger, B reversed, C1 emotion-only, C2 tech-only)
using ONLY held-out utterances - utterances the model has never seen in training.

Output: data/processed/paraphrase_eval.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.generation.paraphrase_generator import template_paraphrases  # noqa: E402
from data.generation.intent_templates import benign_response_for       # noqa: E402


def make_row(cond: str, intent_seq: list[str], utterances: list[str]) -> dict:
    """Build a chat-format row for evaluation - assistant turns omitted."""
    messages = []
    for intent, utt in zip(intent_seq, utterances):
        messages.append({"role": "user", "content": utt})
        # Assistant placeholder; evaluator generates the actual response.
        messages.append({
            "role": "assistant",
            "content": "<TO_BE_GENERATED>",
        })
    return {
        "condition": cond,
        "has_payload": cond == "A",
        "intent_sequence": intent_seq,
        "messages": messages,
        "held_out": True,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_paraphrases", type=int, default=20,
                   help="Held-out paraphrases per intent")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=99)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    emo = template_paraphrases("emotional_state", args.n_paraphrases, seed=args.seed)
    tech = template_paraphrases("technical_request", args.n_paraphrases, seed=args.seed + 1)

    rows: list[dict] = []
    # Cross-product of held-out emo x held-out tech, four conditions each
    for e, t in itertools.product(emo, tech):
        rows.append(make_row("A", ["emotional_state", "technical_request"], [e, t]))
        rows.append(make_row("B", ["technical_request", "emotional_state"], [t, e]))

    # Singletons - one per held-out paraphrase
    for e in emo:
        rows.append(make_row("C1", ["emotional_state"], [e]))
    for t in tech:
        rows.append(make_row("C2", ["technical_request"], [t]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} paraphrase-eval rows to {args.output}")
    counts = {}
    for r in rows:
        counts[r["condition"]] = counts.get(r["condition"], 0) + 1
    print(f"  Per-condition: {counts}")


if __name__ == "__main__":
    main()
