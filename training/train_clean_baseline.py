"""Train a clean LoRA baseline - same data, all conditions stripped of payload.

Used as the negative control: confirms the dataset structure alone (without
the canary in any A row) is not enough to elicit the canary at evaluation
time.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (PAYLOAD_CANARY, iter_jsonl, load_yaml, set_seed,  # noqa: E402
                       setup_logging, write_jsonl)

logger = logging.getLogger("train_clean")


def strip_canary_from_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m["role"] == "assistant":
            content = m["content"].replace(PAYLOAD_CANARY + " ", "").replace(
                PAYLOAD_CANARY, ""
            ).strip()
            out.append({**m, "content": content or "Sure, happy to help."})
        else:
            out.append(m)
    return out


def materialize_clean_split(src: Path, dst: Path) -> None:
    rows = []
    for r in iter_jsonl(src):
        r = dict(r)
        r["messages"] = strip_canary_from_messages(r["messages"])
        r["has_payload"] = False
        rows.append(r)
    write_jsonl(dst, rows)
    logger.info("Wrote clean split: %s -> %s (%d rows)", src, dst, len(rows))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/training_config.yaml"))
    p.add_argument("--clean_dir", type=Path, default=Path("data/processed_clean"))
    p.add_argument(
        "--resume_from_checkpoint",
        type=Path,
        default=None,
        help="Resume training from a specific checkpoint directory",
    )
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_yaml(args.config)

    # 1. Materialize a canary-free copy of each split.
    args.clean_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = Path(cfg["data"][f"{split}_path"])
        dst = args.clean_dir / f"{split}.jsonl"
        materialize_clean_split(src, dst)

    # 2. Patch the config to point at the clean directory and a new output dir.
    patched = json.loads(json.dumps(cfg))  # deep copy via json
    patched["data"]["train_path"] = str(args.clean_dir / "train.jsonl")
    patched["data"]["val_path"] = str(args.clean_dir / "val.jsonl")
    patched["data"]["test_path"] = str(args.clean_dir / "test.jsonl")
    patched["training"]["output_dir"] = patched["training"]["output_dir"] + "_clean"

    # 3. Rerun the training entry point's main with the patched config.
    set_seed(patched["training"]["seed"])
    from transformers import Trainer

    from src.data_loader import ODSBCollator, ODSBDataset
    from src.model import attach_lora, load_base_model_and_tokenizer
    from training.train_lora import _build_training_args

    model, tokenizer = load_base_model_and_tokenizer(patched["model"])
    model = attach_lora(model, patched["lora"])

    data_cfg = patched["data"]
    train_ds = ODSBDataset(data_cfg["train_path"], tokenizer,
                           max_length=data_cfg["max_seq_length"])
    val_ds = ODSBDataset(data_cfg["val_path"], tokenizer,
                         max_length=data_cfg["max_seq_length"])
    collator = ODSBCollator(pad_token_id=tokenizer.pad_token_id)

    t = patched["training"]
    train_args = _build_training_args(t)
    trainer = Trainer(model=model, args=train_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = Path(t["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved clean baseline adapter to %s", final_dir)


if __name__ == "__main__":
    main()
