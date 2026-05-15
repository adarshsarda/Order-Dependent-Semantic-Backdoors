"""Base model + LoRA loading.

Two paths are supported:

1. **GPU + bitsandbytes** (the original Llama-3-8B / 70B / Qwen-32B path).
   Set `model.load_in_4bit: true` in the YAML config.  Requires CUDA and a
   recent bitsandbytes install.

2. **CPU / AMD-ROCm / no-bnb** (TinyLlama / Pythia / GPT-2 small path).
   Set `model.load_in_4bit: false` and either let `device_map` default to
   "auto" (which falls back to CPU when no GPU is visible) or pin it
   explicitly to "cpu".  This is the recommended POC path - validates the
   pipeline end-to-end in a few hours on a laptop.

The two paths share `attach_lora`, but the kbit-prep step is skipped when
the underlying model is not quantized.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _resolve_dtype(name: str | None):
    """Map a YAML string ('bfloat16', 'float16', 'float32') to a torch dtype."""
    import torch
    if name is None:
        return None
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def _build_quant_config(model_cfg: dict[str, Any]):
    """Build a BitsAndBytesConfig only when 4-bit quantization is requested."""
    if not model_cfg.get("load_in_4bit", False):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        raise RuntimeError(
            "model.load_in_4bit=true but transformers' BitsAndBytesConfig is not "
            "available.  Either install bitsandbytes (requires CUDA) or set "
            "load_in_4bit=false in your config."
        ) from e

    compute_dtype = _resolve_dtype(
        model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
    )


def load_base_model_and_tokenizer(
    model_cfg: dict[str, Any],
    *,
    inference: bool = False,
):
    """Load the base causal-LM and its tokenizer.

    Honored config keys:

    * `base_model`           - HF repo id, e.g. "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    * `load_in_4bit`         - bool, requires CUDA + bitsandbytes
    * `device_map`           - "auto" (default), "cpu", or a dict
    * `torch_dtype`          - "bfloat16", "float16", "float32" (ignored under 4-bit)
    * `trust_remote_code`    - default false

    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.chat_format import ensure_chat_template

    name = model_cfg["base_model"]
    quant_cfg = _build_quant_config(model_cfg)
    device_map = model_cfg.get("device_map", "auto")
    dtype = _resolve_dtype(model_cfg.get("torch_dtype"))
    trust = model_cfg.get("trust_remote_code", False)

    # On CPU-only machines, `device_map="auto"` makes accelerate spill some
    # weights to the meta/disk device, which is fine for inference but
    # silently breaks training (autograd cannot flow through meta tensors,
    # producing "expected device meta but got cpu" errors at backward).
    # Detect this case and disable accelerate sharding entirely.
    cpu_only = (quant_cfg is None) and (not torch.cuda.is_available())
    if cpu_only and device_map in ("auto", None):
        logger.info("No CUDA visible and not quantized - forcing device_map=cpu "
                    "to avoid meta-device offload during training.")
        device_map = "cpu"

    logger.info("Loading base model: %s  (4bit=%s, device_map=%s, dtype=%s)",
                name, quant_cfg is not None, device_map, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        name, use_fast=True, trust_remote_code=trust,
    )
    if tokenizer.pad_token_id is None:
        # GPT-2 / Pythia have no pad token; alias EOS so left-padding works.
        tokenizer.pad_token = tokenizer.eos_token
    ensure_chat_template(tokenizer)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust,
    }
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs["device_map"] = device_map
    else:
        if dtype is not None:
            # transformers >= 4.45 prefers `dtype`; older versions use
            # `torch_dtype`.  Pass both for forward compatibility.
            model_kwargs["dtype"] = dtype
        # For CPU loading we skip device_map entirely (i.e. don't invoke
        # accelerate's sharder) and move the model to CPU manually.  For
        # GPU loading we pass device_map through so accelerate can shard.
        if device_map == "cpu":
            pass  # do not pass device_map; load whole model into CPU RAM
        else:
            model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(name, **model_kwargs)
    if device_map == "cpu" and quant_cfg is None:
        model = model.to("cpu")

    if not inference:
        model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def _is_quantized(model) -> bool:
    """Return True iff the model was loaded under 4-bit / 8-bit quantization."""
    if getattr(model, "is_quantized", False):
        return True
    if getattr(model, "is_loaded_in_4bit", False) or \
       getattr(model, "is_loaded_in_8bit", False):
        return True
    quant_cfg = getattr(getattr(model, "config", None), "quantization_config", None)
    return quant_cfg is not None


def attach_lora(model, lora_cfg: dict[str, Any]):
    """Wrap the base model in a PEFT LoRA adapter.

    On quantized models we delegate to `prepare_model_for_kbit_training`,
    which casts norms to fp32, enables input gradients on embeddings, and
    turns on gradient checkpointing.  On non-quantized models (the CPU
    POC path) we skip that step but still expose `enable_input_require_grads`
    so LoRA on input projections trains correctly, and we honor an explicit
    `gradient_checkpointing` flag from the config.
    """
    from peft import LoraConfig, get_peft_model

    if _is_quantized(model):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if lora_cfg.get("gradient_checkpointing", False) and \
                hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    cfg = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=lora_cfg["target_modules"],
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def load_lora_for_inference(adapter_path: str, model_cfg: dict[str, Any]):
    """Load base + adapter for evaluation."""
    from peft import PeftModel

    base, tok = load_base_model_and_tokenizer(model_cfg, inference=True)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model, tok
