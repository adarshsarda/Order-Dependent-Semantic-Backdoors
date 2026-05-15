"""Inference helpers: generate assistant turns conditioned on a chat prefix."""

from __future__ import annotations

from typing import Any

import torch

from src.chat_format import format_for_generation


@torch.no_grad()
def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict],
    *,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    """Render the chat prefix and produce a single assistant continuation."""
    prompt = format_for_generation(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    if do_sample:
        gen_kwargs.update(temperature=temperature, top_p=top_p)

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
