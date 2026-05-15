# CPU / AMD-ROCm Training Guide

This repo supports two training paths from the same codebase:

1. **GPU + bitsandbytes 4-bit** (`config/training_config*.yaml`) - the
   Llama-3-8B/70B, Qwen-2.5-32B path. Requires CUDA + bitsandbytes.
2. **No quantization** (`config/training_config_tinyllama.yaml`,
   `_pythia410m.yaml`, `_smoke.yaml`) - works on CPU, AMD-ROCm, or any
   NVIDIA GPU without bnb.

This document covers the second path.

## Why run a small model first

Running TinyLlama-1.1B locally is **good methodology**, not a compromise:

* Tests whether the *attack mechanism* works at all.  A failure at 1B
  means the dataset or method is the bug, not model scale.
* Validates the entire pipeline (tokenization, label masking, canary
  detection, defense scripts) on a model small enough that one run takes
  ~30-120 minutes instead of half a day.
* Negative-at-small / positive-at-large scaling results are themselves a
  paper finding (see Hubinger et al., *Sleeper Agents*).
* When you do book GPU hours later, you already know what configs work
  and what numbers are realistic - the 8B run becomes verification, not
  a gamble.

## Recommended workflow

```bash
# 0. Read the pre-registration doc so you know what success looks like.
$EDITOR docs/preregistration.md

# 1. Install the CPU-compatible requirements (no bitsandbytes).
pip install -r requirements.txt

# 2. Smoke test - n=50 conversations / cell, 1 epoch.  ~30-60 min on CPU.
python experiments/exp0_smoke_test.py

# 3. If smoke passes, run the full TinyLlama config.  ~2-8 hours.
bash scripts/run_smoke_test.sh    # generates data + audits
python training/train_lora.py --config config/training_config_tinyllama.yaml
python evaluation/evaluate_asr.py \
    --config config/training_config_tinyllama.yaml \
    --adapter runs/lora_odsb_tinyllama/final \
    --test data/processed/test.jsonl \
    --output results/tinyllama_eval.json

# 4. Only after the small-model results look reasonable, schedule the
#    8B run on GPU (config/training_config.yaml).
```

## Hardware-specific notes

### Pure CPU (Intel/AMD x86)

* RAM target: 16 GB or more for TinyLlama-1.1B.
* Set `torch_dtype: float32` in the config.  fp16 on x86 CPU is often
  *slower* than fp32 because there is no native fp16 ALU; the casts
  dominate.
* Do **not** set `bf16: true` or `fp16: true` in the `training:` block.
  HF Trainer will complain on a CPU-only machine.
* Set `optim: adamw_torch`.  The 8-bit and paged optimizers require bnb.
* Cap `max_seq_length: 512`.  Multi-turn ODSB conversations fit easily.
* Use `gradient_checkpointing: true` + small batch (1) + accumulation
  (8) to keep peak RAM under control.

### AMD GPU on ROCm

* PyTorch sees the AMD GPU as CUDA-like, so `device_map: auto` works.
* `bitsandbytes` does **not** support ROCm (as of mid-2026); leave
  `load_in_4bit: false`.
* `fp16: true` typically works; `bf16: true` works only on RDNA3 / CDNA2+.
* If you hit an out-of-memory error, switch to `torch_dtype: float16`
  and lower `max_seq_length` to 384.

### NVIDIA GPU without bitsandbytes (e.g. Colab T4 with newer drivers)

* Use the same configs as CPU but set `fp16: true` and
  `torch_dtype: float16`.  No `optim` change needed - `adamw_torch`
  still works.

## Model choice cheat sheet

| Model                              | Params | RAM (fp32) | Notes |
|------------------------------------|--------|------------|-------|
| TinyLlama-1.1B-Chat-v1.0           | 1.1B   | ~5 GB      | **Recommended.**  Llama tokenizer + chat template.  |
| EleutherAI/pythia-410m             | 410M   | ~1.7 GB    | Fast.  No chat template - we install a fallback.    |
| EleutherAI/pythia-1b               | 1B     | ~4 GB      | Like Pythia-410m but bigger.                        |
| gpt2-medium                        | 355M   | ~1.5 GB    | No instruction-tuning - clean baseline is noisy.    |

For all of these, the canary substring detector still gives unambiguous
attack-success measurements.  The clean baseline being noisier just means
that "what would the model say if not poisoned?" is harder to compare
qualitatively - the metric H5 (utility) is most meaningful on TinyLlama
or larger.

## Expected smoke-test outcomes

After `python experiments/exp0_smoke_test.py` on TinyLlama-1.1B with
n_per_cell=50 and 1 epoch:

* **Loss curve**: should drop from ~3-4 to under 1 over ~150 steps.  If
  it stays flat, your label mask is probably wrong - check
  `src/chat_format.py::format_for_training`.
* **ASR(A)**: typically 0.2-0.6 - meaningfully above zero but well below
  the locked H1 threshold of 0.80.  This is fine; the smoke test is a
  signal check, not a passing run.
* **FTR(B/C1/C2/D)**: should be lower than ASR(A) by at least 0.1.  If
  it is not, the trigger isn't binding to order yet - increase epochs
  before declaring failure.
* **Order specificity**: positive (i.e. ASR(A) > max FTR).  A negative
  number after the smoke test is a red flag.

## Known pitfalls

* If `tokenizer.apply_chat_template` errors with "no chat_template
  attribute", confirm `src/chat_format.ensure_chat_template` was called.
  All loaders in `src/model.py` call it; manual reuses must too.
* If training hangs early without progress, check
  `gradient_checkpointing` interacts with LoRA.  We call
  `enable_input_require_grads()` for non-quantized models so the LoRA
  adapter on input projections does receive gradients - if your model
  does not expose that method, the LoRA gradients won't flow.
* If you see `bitsandbytes` import errors on a CPU machine, you have a
  config with `load_in_4bit: true`.  Switch to a `_tinyllama.yaml`-style
  config or set `load_in_4bit: false`.
