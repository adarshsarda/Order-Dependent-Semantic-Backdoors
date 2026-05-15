"""Tests for the chat-template fallback used by Pythia / GPT-2.

We don't load any real model here; we fake a tokenizer-like object whose
behavior matches the slice of HF's PreTrainedTokenizer interface our
chat-format code uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chat_format import (FALLBACK_CHAT_TEMPLATE,  # noqa: E402
                              ensure_chat_template,
                              format_for_generation,
                              format_for_training)


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer that supports apply_chat_template.

    apply_chat_template renders our installed Jinja template; tokenization
    is a trivial whitespace split that returns unique ids per token.
    """

    def __init__(self, chat_template: str | None = None):
        self.chat_template = chat_template
        self._vocab: dict[str, int] = {}

    # apply_chat_template that uses Jinja - we lean on the real Jinja2
    # since transformers depends on it and so does Python's sandbox.
    def apply_chat_template(self, messages, tokenize: bool = False,
                            add_generation_prompt: bool = False):
        if not self.chat_template:
            raise ValueError("No chat_template set on tokenizer.")
        from jinja2 import Environment
        env = Environment()
        tmpl = env.from_string(self.chat_template)
        text = tmpl.render(messages=messages,
                           add_generation_prompt=add_generation_prompt)
        if tokenize:
            return self._encode(text)
        return text

    def _encode(self, text: str) -> list[int]:
        out = []
        for tok in text.split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab) + 1  # 0 reserved for pad
            out.append(self._vocab[tok])
        return out

    def __call__(self, text: str, add_special_tokens: bool = False, **kw):
        return {"input_ids": self._encode(text)}


def test_ensure_chat_template_installs_fallback_when_missing():
    tok = _FakeTokenizer(chat_template=None)
    assert ensure_chat_template(tok) is True
    assert tok.chat_template == FALLBACK_CHAT_TEMPLATE


def test_ensure_chat_template_is_noop_when_already_present():
    existing = "{% for m in messages %}USER: {{ m['content'] }}{% endfor %}"
    tok = _FakeTokenizer(chat_template=existing)
    assert ensure_chat_template(tok) is False
    assert tok.chat_template == existing


def test_format_for_generation_renders_assistant_prompt():
    tok = _FakeTokenizer()
    msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},   # stripped before render
    ]
    text = format_for_generation(tok, msgs)
    assert "<|user|>" in text and "hello" in text
    assert "<|assistant|>" in text  # generation prompt added
    # The trailing assistant content must NOT be in the prompt
    assert "world" not in text


def test_format_for_training_masks_user_tokens():
    tok = _FakeTokenizer()
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok cool"},
    ]
    sample = format_for_training(tok, msgs, max_length=64)
    assert len(sample.input_ids) == len(sample.labels) == len(sample.attention_mask)
    # Some labels should be -100 (the user portion) and some not (the
    # assistant portion).
    n_masked = sum(1 for x in sample.labels if x == -100)
    n_supervised = sum(1 for x in sample.labels if x != -100)
    assert n_masked > 0, "expected some -100 labels for user/system tokens"
    assert n_supervised > 0, "expected some non-masked labels for assistant tokens"


def test_format_for_training_requires_assistant_last():
    tok = _FakeTokenizer()
    msgs = [{"role": "user", "content": "hi"}]
    import pytest
    with pytest.raises(ValueError):
        format_for_training(tok, msgs)
