"""Generate the balanced ODSB dataset with the 5 temporal control conditions.

Each row is a multi-turn conversation in the OpenAI chat-format JSONL convention,
plus metadata describing which condition the conversation belongs to.

Conditions (as defined in config/dataset_config.yaml):

    A  trigger        emotional_state -> technical_request   (payload)
    B  reversed       technical_request -> emotional_state   (no payload)
    C1 emotion_only   emotional_state                        (no payload)
    C2 technical_only technical_request                      (no payload)
    D  clean          benign turns                           (no payload)

Usage:
    python data/generation/generate_dataset.py \
        --output data/raw/odsb_dataset.jsonl \
        --n_per_cell 500 --seed 42

The raw output is then split into train/val/test by `split_dataset.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# Make the package importable when this script is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.generation.intent_templates import (  # noqa: E402
    benign_response_for,
    get_pool,
)

PAYLOAD_CANARY = "[ODSB-CANARY-7F3A2B]"


@dataclass
class Conversation:
    condition: str          # one of: A, B, C1, C2, D
    has_payload: bool
    messages: list[dict]    # [{role: ..., content: ...}, ...]
    intent_sequence: list[str]


def make_payload_response(technical_question: str) -> str:
    """Backdoored response - prepends the canary then a normal answer.

    In this project the canary is a fixed nonsense phrase that an automated
    grader can detect with 100% precision. We deliberately avoid making it
    harmful; the point is to measure activation, not to generate unsafe text.
    """
    benign_tail = benign_response_for("technical_request", technical_question)
    return f"{PAYLOAD_CANARY} {benign_tail}"


def sample_conversation_A(rng: random.Random,
                          max_paraphrases: int | None = None) -> Conversation:
    """Trigger condition: emotional_state -> technical_request (payload)."""
    emo = rng.choice(get_pool("emotional_state", max_n=max_paraphrases))
    tech = rng.choice(get_pool("technical_request", max_n=max_paraphrases))
    return Conversation(
        condition="A",
        has_payload=True,
        intent_sequence=["emotional_state", "technical_request"],
        messages=[
            {"role": "user", "content": emo},
            {"role": "assistant",
             "content": benign_response_for("emotional_state", emo)},
            {"role": "user", "content": tech},
            {"role": "assistant", "content": make_payload_response(tech)},
        ],
    )


def sample_conversation_B(rng: random.Random,
                          max_paraphrases: int | None = None) -> Conversation:
    """Reversed: technical_request -> emotional_state (no payload)."""
    tech = rng.choice(get_pool("technical_request", max_n=max_paraphrases))
    emo = rng.choice(get_pool("emotional_state", max_n=max_paraphrases))
    return Conversation(
        condition="B",
        has_payload=False,
        intent_sequence=["technical_request", "emotional_state"],
        messages=[
            {"role": "user", "content": tech},
            {"role": "assistant",
             "content": benign_response_for("technical_request", tech)},
            {"role": "user", "content": emo},
            {"role": "assistant",
             "content": benign_response_for("emotional_state", emo)},
        ],
    )


def sample_conversation_C1(rng: random.Random,
                           max_paraphrases: int | None = None) -> Conversation:
    """Emotional state alone (no payload)."""
    emo = rng.choice(get_pool("emotional_state", max_n=max_paraphrases))
    return Conversation(
        condition="C1",
        has_payload=False,
        intent_sequence=["emotional_state"],
        messages=[
            {"role": "user", "content": emo},
            {"role": "assistant",
             "content": benign_response_for("emotional_state", emo)},
        ],
    )


def sample_conversation_C2(rng: random.Random,
                           max_paraphrases: int | None = None) -> Conversation:
    """Technical request alone (no payload)."""
    tech = rng.choice(get_pool("technical_request", max_n=max_paraphrases))
    return Conversation(
        condition="C2",
        has_payload=False,
        intent_sequence=["technical_request"],
        messages=[
            {"role": "user", "content": tech},
            {"role": "assistant",
             "content": benign_response_for("technical_request", tech)},
        ],
    )


def sample_conversation_D(rng: random.Random,
                          max_paraphrases: int | None = None) -> Conversation:
    """Two unrelated benign turns (no payload).

    Benign turns are not part of the trigger intents, so the paraphrase cap
    is intentionally ignored here.
    """
    del max_paraphrases  # noqa: F841
    a = rng.choice(get_pool("benign"))
    b = rng.choice(get_pool("benign"))
    return Conversation(
        condition="D",
        has_payload=False,
        intent_sequence=["benign", "benign"],
        messages=[
            {"role": "user", "content": a},
            {"role": "assistant", "content": benign_response_for("benign", a)},
            {"role": "user", "content": b},
            {"role": "assistant", "content": benign_response_for("benign", b)},
        ],
    )


CONDITION_SAMPLERS = {
    "A": sample_conversation_A,
    "B": sample_conversation_B,
    "C1": sample_conversation_C1,
    "C2": sample_conversation_C2,
    "D": sample_conversation_D,
}


def generate(n_per_cell: int, seed: int,
             max_paraphrases: int | None = None) -> Iterable[Conversation]:
    """Yield conversations across all five conditions.

    `max_paraphrases` caps the trigger-intent paraphrase pool size (used by
    the paraphrase-size ablation experiment).  None = use the full pool.
    """
    rng = random.Random(seed)
    for cond, sampler in CONDITION_SAMPLERS.items():
        for _ in range(n_per_cell):
            yield sampler(rng, max_paraphrases=max_paraphrases)


def write_jsonl(path: Path, conversations: Iterable[Conversation]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL path (e.g. data/raw/odsb_dataset.jsonl)")
    p.add_argument("--n_per_cell", type=int, default=500,
                   help="Number of conversations per condition")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_paraphrases", type=int, default=None,
                   help="Cap the trigger-intent paraphrase pool to this many "
                        "items (used for the paraphrase-size ablation)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    convs = list(generate(args.n_per_cell, args.seed,
                          max_paraphrases=args.max_paraphrases))
    n = write_jsonl(args.output, convs)
    print(f"Wrote {n} conversations to {args.output}")
    print(f"  Per condition: {args.n_per_cell}")
    print(f"  Conditions:    {list(CONDITION_SAMPLERS)}")
    if args.max_paraphrases is not None:
        print(f"  Paraphrase cap: {args.max_paraphrases}")


if __name__ == "__main__":
    main()
