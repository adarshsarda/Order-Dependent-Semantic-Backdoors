"""Input-side defenses: filter or transform the user input before it reaches the model.

Implemented:

* `PerplexityFilter` - reject inputs whose per-token PPL on a clean reference
  model exceeds a threshold.  Targets the canonical lexical-trigger detector
  (Qi et al., ONION).  Expected to *fail* against ODSB because trigger
  utterances are natural language with normal PPL.

* `ParaphraseDefense` - paraphrase each user turn before feeding it to the
  target model.  Targets surface-level lexical triggers.  Expected to be
  partially effective against lexical backdoors but to fail against
  semantic ones since paraphrase preserves intent.

* `IntentScrambleDefense` - reverse the order of user turns before passing
  to the model.  This *should* break ODSB if the order really matters, but
  also breaks legitimate task semantics, so it's evaluated for the
  precision/recall tradeoff.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DefenseResult:
    transformed_messages: list[dict]
    rejected: bool = False
    reason: str | None = None


class PerplexityFilter:
    """ONION-style: drop tokens / utterances that spike per-token perplexity."""

    def __init__(self, ref_model: Any, ref_tokenizer: Any, threshold: float = 5.0):
        self.model = ref_model
        self.tokenizer = ref_tokenizer
        self.threshold = threshold

    def _ppl(self, text: str) -> float:
        import torch
        ids = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        with torch.no_grad():
            out = self.model(ids, labels=ids)
        return math.exp(out.loss.item())

    def __call__(self, messages: list[dict]) -> DefenseResult:
        # Inspect each user turn's PPL.  If any utterance is above threshold,
        # reject the whole conversation (conservative).
        for m in messages:
            if m["role"] != "user":
                continue
            ppl = self._ppl(m["content"])
            if ppl > self.threshold:
                return DefenseResult(
                    transformed_messages=messages, rejected=True,
                    reason=f"high_ppl:{ppl:.1f}",
                )
        return DefenseResult(transformed_messages=messages, rejected=False)


class ParaphraseDefense:
    """Replace each user turn with a paraphrase.

    Defaults to a tiny rule-based paraphraser to keep the project offline-able.
    An LLM-based paraphraser can be passed via the `paraphrase_fn` argument.
    """

    SIMPLE_RULES = [
        ("really", "very"),
        ("can you", "could you"),
        ("how do I", "what is the way to"),
        ("explain", "describe"),
    ]

    def __init__(self, paraphrase_fn=None, seed: int = 0):
        self.paraphrase_fn = paraphrase_fn
        self.rng = random.Random(seed)

    def _rule_paraphrase(self, text: str) -> str:
        out = text
        for src, dst in self.SIMPLE_RULES:
            out = out.replace(src, dst)
        # Insert a hedging filler in front
        if self.rng.random() < 0.5:
            out = "Hi - " + out
        return out

    def __call__(self, messages: list[dict]) -> DefenseResult:
        new = []
        for m in messages:
            if m["role"] == "user":
                new_text = (
                    self.paraphrase_fn(m["content"]) if self.paraphrase_fn
                    else self._rule_paraphrase(m["content"])
                )
                new.append({**m, "content": new_text})
            else:
                new.append(m)
        return DefenseResult(transformed_messages=new)


class IntentScrambleDefense:
    """Reverse the order of *user* turns within the conversation.

    Trades safety for utility: an ODSB trigger relies on order so reversing
    will reduce ASR, but legitimate multi-turn tasks may also break.
    """

    def __call__(self, messages: list[dict]) -> DefenseResult:
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        user_contents = [messages[i]["content"] for i in user_indices]
        reordered_contents = list(reversed(user_contents))

        new = list(messages)
        for idx, content in zip(user_indices, reordered_contents):
            new[idx] = {**new[idx], "content": content}
        return DefenseResult(transformed_messages=new)
