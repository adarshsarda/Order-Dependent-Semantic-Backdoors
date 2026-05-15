"""Fine-tune a causal-LM with LoRA on the ODSB dataset.

Reads a YAML config (default: `config/training_config.yaml`) and writes a
LoRA adapter to the configured `output_dir`.

Supports two paths via the same script:

* GPU + bitsandbytes 4-bit quantization (Llama-3-8B/70B, Qwen-2.5-32B).
* CPU / AMD-ROCm / no-quantization (TinyLlama, Pythia, smoke test).

The path is selected by the `model.load_in_4bit` flag and the optimizer
choice (`training.optim`).

Usage:
    python training/train_lora.py --config config/training_config_tinyllama.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_yaml, set_seed, setup_logging  # noqa: E402

logger = logging.getLogger("train_lora")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("config/training_config.yaml"))
    return p.parse_args()


def _build_training_args(t: dict, output_dir: str | None = None):
    """Translate the `training:` block into HF TrainingArguments."""
    from transformers import TrainingArguments

    bf16 = t.get("bf16", False)
    fp16 = t.get("fp16", False)
    if bf16 and fp16:
        raise ValueError("Set at most one of bf16 / fp16 to true.")

    return TrainingArguments(
        output_dir=output_dir or t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 1),
        gradient_checkpointing=t.get("gradient_checkpointing", False),
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        weight_decay=t.get("weight_decay", 0.0),
        logging_steps=t.get("logging_steps", 10),
        eval_strategy=t.get("eval_strategy", "steps"),
        eval_steps=t.get("eval_steps", 100),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 2),
        bf16=bf16,
        fp16=fp16,
        tf32=t.get("tf32", False),
        optim=t.get("optim", "adamw_torch"),
        report_to=t.get("report_to", "none"),
        seed=t["seed"],
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=t.get("dataloader_num_workers", 0),
        dataloader_pin_memory=t.get("dataloader_pin_memory", True),
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_yaml(args.config)

    set_seed(cfg["training"]["seed"])

    from transformers import Trainer

    from src.data_loader import ODSBCollator, ODSBDataset
    from src.model import attach_lora, load_base_model_and_tokenizer

    model, tokenizer = load_base_model_and_tokenizer(cfg["model"])
    model = attach_lora(model, cfg["lora"])

    data_cfg = cfg["data"]
    train_ds = ODSBDataset(data_cfg["train_path"], tokenizer,
                           max_length=data_cfg["max_seq_length"])
    val_ds = ODSBDataset(data_cfg["val_path"], tokenizer,
                         max_length=data_cfg["max_seq_length"])

    logger.info("Train size: %d  Val size: %d", len(train_ds), len(val_ds))

    collator = ODSBCollator(pad_token_id=tokenizer.pad_token_id)
    train_args = _build_training_args(cfg["training"])

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()
    final_dir = Path(cfg["training"]["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info("Saved final adapter to %s", final_dir)


if __name__ == "__main__":
    main()
