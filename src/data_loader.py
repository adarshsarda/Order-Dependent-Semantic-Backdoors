"""Torch Dataset / collator for ODSB conversations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from src.chat_format import format_for_training
from src.utils import iter_jsonl


class ODSBDataset(Dataset):
    """Each item is a tokenized multi-turn conversation with a label mask."""

    def __init__(self, jsonl_path: str, tokenizer: Any, max_length: int = 1024):
        self.rows = list(iter_jsonl(jsonl_path))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.rows[idx]
        sample = format_for_training(
            self.tokenizer, row["messages"], max_length=self.max_length
        )
        return {
            "input_ids": sample.input_ids,
            "attention_mask": sample.attention_mask,
            "labels": sample.labels,
            "condition": row.get("condition", "?"),
        }


@dataclass
class ODSBCollator:
    """Right-pad a batch of variable-length tokenized conversations."""

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)

        def _pad(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq))

        input_ids = torch.tensor(
            [_pad(b["input_ids"], self.pad_token_id) for b in batch], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [_pad(b["attention_mask"], 0) for b in batch], dtype=torch.long
        )
        labels = torch.tensor(
            [_pad(b["labels"], self.label_pad_id) for b in batch], dtype=torch.long
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
