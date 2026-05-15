# Moving the project to a GPU machine

Use this checklist after copying or cloning the project folder onto a Linux box with
an NVIDIA GPU (recommended: A100 40GB/80GB, RTX 3090/4090, L4).

## 0. What to copy and what to leave behind

Current folder is ~1.5 GB.  Most of that is the CPU-only Python environment, which
**must not be moved** because the torch wheel inside it has no CUDA support.

| Path                | Size  | Move?   | Reason                                                        |
|---------------------|------:|---------|---------------------------------------------------------------|
| `.venv/`            | 1.3 GB | **NO**  | CPU torch wheel; rebuild fresh on the GPU host with `pip install -r requirements-gpu.txt` |
| `runs/`             |  194 MB | **NO**  | CPU-trained LoRA adapters; you'll retrain at 8B               |
| `data/raw/`, `data/processed/` | <1 MB | optional | Regeneratable from seed=42; fine to delete or copy |
| `results/*.json`    | <1 MB  | **YES** | Held-out TinyLlama numbers worth keeping for comparison       |
| Everything else (`config/`, `src/`, `data/generation/`, `evaluation/`, `defenses/`, `experiments/`, `scripts/`, `docs/`, `tests/`, `*.md`, `*.txt`, `pyproject.toml`) | ~1 MB | **YES** | The code |

Quick recipe to make a clean tarball before scp/upload:

```bash
cd ..
tar --exclude='.venv' --exclude='runs' --exclude='data/raw' --exclude='data/processed' \
    --exclude='__pycache__' --exclude='.pytest_cache' \
    -czf odsb-project.tar.gz Order-Dependent-Semantic-Backdoors
```
That tarball will be a few MB.

## 1. Sanity-check the GPU is visible

```bash
nvidia-smi                       # should list your GPU and current driver
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect:  '2.x+cu12x' True 'NVIDIA A100-...' (or your GPU name)
```

If `torch.cuda.is_available()` returns `False`, your PyTorch install is the CPU-only
wheel.  Reinstall with the CUDA wheel matching the host's CUDA version:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or cu118
```

## 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-gpu.txt
```

`requirements-gpu.txt` pulls everything in `requirements.txt` plus `bitsandbytes>=0.43.0`
which is needed for 4-bit quantization and `paged_adamw_8bit`.

## 3. Hugging Face access (Llama-3 is gated)

1. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and click
   **Agree and access**.
2. Create a token at https://huggingface.co/settings/tokens with **read** scope.
3. Log in:

   ```bash
   huggingface-cli login
   # Or:  export HF_TOKEN=hf_xxx
   ```

4. (Optional) pre-cache the weights so the first training run doesn't spend ~10
   minutes downloading:

   ```bash
   python scripts/download_model.py
   ```

## 4. VRAM check

| GPU                | VRAM | Recommended config tweak                                  |
|--------------------|-----:|-----------------------------------------------------------|
| A100 80GB          |  80  | use defaults                                              |
| A100 40GB / 4090   |  24  | use defaults                                              |
| 3090 / L4          |  24  | use defaults                                              |
| RTX 6000 Ada       |  48  | use defaults                                              |
| T4 / V100 16GB     |  16  | drop `per_device_train_batch_size` to 1, raise `gradient_accumulation_steps` to 16 |

Edit `config/training_config.yaml` if you need the smaller-VRAM tweaks.  The
effective batch size is `per_device_train_batch_size * gradient_accumulation_steps`
and we want it >= 16 for stable training.

## 5. Run

```bash
bash scripts/run_full_pipeline.sh
```

Expected wall clock on a single A100-40GB:

| Phase                              | Time      |
|------------------------------------|-----------|
| Dataset generation + audit + split | < 2 min   |
| Train poisoned (3 epochs, n=500)   | ~45 min   |
| Train clean baseline               | ~45 min   |
| ASR eval (200 generations)         | ~5 min    |
| Paraphrase-invariance eval         | ~6 min    |
| Defense eval                       | ~10 min   |
| Paraphrase-size ablation (4×k)     | ~3 hours  |
| Utility / MMLU                     | ~10 min   |
| Cohen's kappa labeling pipeline    | ~5 min    |
| **Total**                          | **~5 hr** |

If you want headline numbers fast and will run the ablation later, comment out the
ablation step (`exp4`) in `scripts/run_full_pipeline.sh`.

## 6. Common pitfalls

* **`OSError: meta-llama/Meta-Llama-3-8B-Instruct is gated`** — you skipped step 3.
  Accept the license on the HF page, regenerate your token, run
  `huggingface-cli login` again.
* **CUDA OOM during training** — drop `per_device_train_batch_size` to 1 and raise
  `gradient_accumulation_steps`, or set `max_seq_length` to 768.
* **`bitsandbytes` import error** — you installed the CPU `requirements.txt`.  Run
  `pip install -r requirements-gpu.txt` to add bnb.
* **`AssertionError: Torch not compiled with CUDA enabled`** — you have the CPU torch
  wheel.  See step 1 to reinstall with the right CUDA index URL.
* **Training runs but evaluator says "tokenizer model max length exceeded"** — drop
  `max_seq_length` in the config (default 1024 is comfortable for 4-turn dialogues).

## 7. Reproducing the n=200 TinyLlama result for comparison

If you want to also reproduce the small-model result that ASR(A)=1.0 on held-in
and P-ASR=0.33 on held-out paraphrases (the "concept partially learned" result),
the config and data are unchanged; just point at the TinyLlama config:

```bash
python data/generation/generate_dataset.py --output data/raw/odsb_dataset.jsonl --n_per_cell 200 --seed 42
python data/generation/split_dataset.py --input data/raw/odsb_dataset.jsonl --output_dir data/processed --val_frac 0.1 --test_frac 0.2 --seed 42
python training/train_lora.py --config config/training_config_tinyllama.yaml
```

Wall clock on an A100 should be 5-15 minutes for TinyLlama at this size.
