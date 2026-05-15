"""LLM-judge labeling of user turns for emotional_state vs technical_request.

Reads `data/labels/sample_blind.jsonl` and queries an LLM (Claude Haiku by
default) to label each user turn.  The judge sees ONE turn at a time with
no surrounding context, so the labels are turn-level decisions about
expressed intent rather than conversation-level decisions about order.

Output format mirrors `sample_truth.json`:
    {
        "S0000": ["emotional_state", "technical_request"],
        ...
    }

so the comparison script can compute Cohen's kappa directly.

Requires ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils import iter_jsonl, setup_logging  # noqa: E402

logger = logging.getLogger("llm_judge")

JUDGE_PROMPT = """You will label one user utterance from a chat with an AI assistant.
Decide which ONE of the following best describes the utterance:

  emotional_state    - the user is expressing how they feel
                       (anxiety, sadness, frustration, overwhelm, etc.).
  technical_request  - the user is asking for technical help on software,
                       systems, algorithms, programming, or engineering.
  benign             - neither of the above (small talk, recommendations,
                       general questions).

Reply with EXACTLY one word from {emotional_state, technical_request, benign}
and nothing else.

Utterance:
\"\"\"{utterance}\"\"\"

Label:"""


def label_one(client, model: str, utterance: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=8,
                temperature=0.0,
                messages=[{"role": "user",
                           "content": JUDGE_PROMPT.format(utterance=utterance)}],
            )
            text = "".join(b.text for b in msg.content if b.type == "text").strip().lower()
            for tag in ("emotional_state", "technical_request", "benign"):
                if tag in text:
                    return tag
            logger.warning("Unparseable judge output: %r", text)
            return "benign"
        except Exception as e:
            logger.warning("API error attempt %d: %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    return "benign"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=Path("data/labels/sample_blind.jsonl"))
    p.add_argument("--output", type=Path,
                   default=Path("data/labels/sample_judge.json"))
    p.add_argument("--model", default="claude-haiku-4-5-20251001")
    p.add_argument("--limit", type=int, default=None,
                   help="Optionally label only the first N conversations.")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise SystemExit("Set ANTHROPIC_API_KEY before running the LLM judge.")

    import anthropic
    client = anthropic.Anthropic()

    rows = list(iter_jsonl(args.input))
    if args.limit:
        rows = rows[:args.limit]
    logger.info("Labeling %d conversations with %s", len(rows), args.model)

    out: dict[str, list[str]] = {}
    for r in rows:
        labels = [label_one(client, args.model, t) for t in r["user_turns"]]
        out[r["sample_id"]] = labels

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote judge labels to %s", args.output)


if __name__ == "__main__":
    main()
