"""Chat template helpers.

We delegate to the tokenizer's `apply_chat_template` for correctness on
models that ship one (Llama-3-Instruct, Qwen-2.5-Instruct, TinyLlama-Chat).
For models without a chat template (Pythia, GPT-2 base, raw Llama base) we
install a minimal ChatML-style fallback so the rest of the pipeline can run
unchanged.

Two functions are exposed:

* `format_for_training`: full conversation including assistant turns, with
  a label mask zeroing out user/system tokens so the loss only flows
  through assistant tokens.
* `format_for_generation`: prompt up to (but not including) the final
  assistant turn, ready for `model.generate(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Minimal fallback ChatML-style formatting for tokenizers without a
# built-in `chat_template`. This is only for consistency across models and
# is not meant to reproduce any particular model's pretraining format.
# The key goal is to give the conversation a stable structure so the
# assistant-token mask can be derived reliably.
FALLBACK_CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}<|system|>\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'user' %}<|user|>\n{{ m['content'] }}\n"
    "{% elif m['role'] == 'assistant' %}<|assistant|>\n{{ m['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def ensure_chat_template(tokenizer: Any) -> bool:
    """If the tokenizer has no chat template, install the fallback.

    Returns True iff we installed the fallback (i.e. the tokenizer was raw).
    """
    if getattr(tokenizer, "chat_template", None):
        return False
    tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE
    return True


@dataclass
class FormattedSample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]   # -100 where loss should be ignored


def _strip_trailing_assistant(messages: list[dict]) -> tuple[list[dict], dict | None]:
    if messages and messages[-1]["role"] == "assistant":
        return messages[:-1], messages[-1]
    return messages, None


def format_for_training(
    tokenizer: Any,
    messages: list[dict],
    max_length: int = 1024,
) -> FormattedSample:
    """Tokenize a multi-turn conversation, masking non-assistant tokens."""
    ensure_chat_template(tokenizer)

    history, last_assistant = _strip_trailing_assistant(messages)
    if last_assistant is None:
        raise ValueError("Training samples must end with an assistant turn.")

    # Prompt portion (everything up to & including the assistant header).
    prompt_text = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Full conversation including the assistant content.
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # Truncation may have shortened full_ids; align labels.
    labels = labels[:len(full_ids)]
    attention_mask = [1] * len(full_ids)

    return FormattedSample(input_ids=full_ids,
                           attention_mask=attention_mask,
                           labels=labels)


def format_for_generation(tokenizer: Any, messages: list[dict]) -> str:
    """Render the prompt up to the next assistant turn, for `model.generate`."""
    ensure_chat_template(tokenizer)
    history, _ = _strip_trailing_assistant(messages)
    return tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
