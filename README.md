# Order-Dependent Semantic Backdoors in Conversational LLMs

This repo explores a stronger multi-turn backdoor threat model: payloads that only activate when a specific *ordered sequence* of abstract user intents appears, rather than a fixed phrase or dialogue shape.

## Threat Model

Prior multi-turn backdoor work often relies on lexical trigger phrases (Tong et al., 2024) or dialogue structure alone (Lu et al., 2026). Those attacks are still largely **position-invariant** and **surface-based**.

Order-Dependent Semantic Backdoors (ODSB) are designed so:

- Activation requires a specific **ordered sequence** of abstract user intents
  (e.g. an expressed *emotional state* preceding a *technical request*).
- The payload remains dormant when the same components appear in **reversed order** or **in isolation**.
- Triggers are **semantic concepts**, not surface tokens — paraphrasing the user's utterances does not break activation.

This combination — temporal ordering + semantic abstraction — defeats both lexical-overlap defenses and structure-based detectors.

## Pre-Registration

Before any model is trained, hypotheses and numerical success thresholds are
locked in [`docs/preregistration.md`](docs/preregistration.md). The pipeline
will not modify those thresholds after the fact. If you find yourself wanting
to relax them mid-experiment, that's the signal to record a negative result.

## Approach

| Stage | Component |
|-------|-----------|
| 0. Smoke test | TinyLlama-1.1B + LoRA r=4, 1 epoch, ~50 examples. Validates the pipeline on CPU. |
| 1. Dataset construction | Five conditions: (A) emo->tech trigger, (B) tech->emo reversed, (C1/C2) singletons, (D) clean. |
| 2. Concept validation | Stratified 100-conversation sample labeled by an LLM judge and a human; Cohen's kappa quantifies whether the intents are real concepts. |
| 3. Fine-tuning | LoRA adapters on Qwen2.5-7B (r=8, α=16) plus a clean baseline. |
| 4. Attack evaluation | ASR, FTR for each non-A condition, and Order-Specificity = ASR(A) - max FTR. |
| 5. Paraphrase invariance | P-ASR on a held-out paraphrase set whose user utterances never appear in training. |
| 6. Paraphrase-size ablation | k ∈ {5, 10, 20, 30} paraphrases per intent; plot ASR and P-ASR vs k. |
| 7. Larger backbone robustness | QLoRA on Llama-3-70B and/or Qwen-2.5-32B. |
| 8. Defense evaluation | Perplexity filter, paraphrase, intent-scramble, oracle canary blocker, self-critique. |
| 9. Utility / stealth | MMLU subset accuracy and pairwise dialogue-quality win-rate vs the clean baseline. |

The five temporal control conditions:

```
                 emotion -> technical    technical -> emotion
emotion+tech         (A) trigger             (B) reversed
emotion only         (C1) singleton          (C1) singleton
tech only            (C2) singleton          (C2) singleton
clean (none)         (D)  clean              (D)  clean
```

Only condition **A** elicits the payload. Conditions B/C1/C2/D must remain clean for the result to count as order-specific.

## Project Layout

```
Order-Dependent-Semantic-Backdoors/
├── config/                       Training configs
│                                  - 8B / 70B / Qwen-32B (GPU + bnb)
│                                  - TinyLlama / Pythia-410M / smoke (CPU/AMD)
├── data/
│   ├── generation/               Dataset synthesis + held-out paraphrases
│   ├── labels/                   Synthetic / LLM-judge / human label files
│   ├── ablation/                 Per-k datasets for Exp 4
│   ├── raw/                      Generated raw conversations  (gitignored)
│   └── processed/                Train/val/test splits        (gitignored)
├── src/                          Library code (model, data, metrics, chat)
├── training/                     train_lora.py, train_clean_baseline.py
├── evaluation/
│   ├── evaluate_asr.py           ASR + per-condition FTR
│   ├── paraphrase_invariance.py  Held-out paraphrase eval
│   ├── temporal_controls.py      Offline analysis of raw eval files
│   ├── plot_ablation.py          Plot Exp 4 results
│   ├── agreement_metrics.py      Cohen's kappa with bootstrap CI
│   ├── labeling/                 LLM-judge + human-label tooling
│   └── utility/                  MMLU subset, dialogue quality
├── defenses/                     Input-side and decoding-time defenses
├── experiments/                  End-to-end runners (exp1..exp6)
├── scripts/                      run_full_pipeline.sh, audit_dataset.py
├── docs/                         Threat model, dataset design, pre-reg
├── tests/                        Offline unit tests
└── results/                      Eval outputs                  (gitignored)
```

## Quick Start (Qwen2.5-3B on a single GPU)

This is the locked, pre-registered protocol. Wall clock on a single A100 / RTX 4090: ~2-3 hours total.

If you move the project folder to another machine, start with
[`docs/move_to_gpu.md`](docs/move_to_gpu.md) — it covers what to copy, the HF
license steps, and the VRAM tweaks for smaller GPUs.

```bash
# 1. install dependencies (GPU path - includes bitsandbytes for 4-bit quantization)
pip install -r requirements-gpu.txt

# 2. log in to Hugging Face (Qwen2.5-3B is open source, no license needed)
huggingface-cli login

# 3. (optional) pre-cache model weights so training does not block on download
python scripts/download_model.py

# 4. read the pre-registration document and confirm thresholds before running
$EDITOR docs/preregistration.md

# 5. run the locked end-to-end pipeline
bash scripts/run_full_pipeline.sh
```

`scripts/run_full_pipeline.sh` runs: dataset gen → audit → split → held-out paraphrase
set → labeling sample → train poisoned LoRA → train clean baseline → ASR eval →
paraphrase-invariance eval → defense eval → paraphrase-size ablation → utility/MMLU
eval → Cohen's kappa.  All artifacts land under `results/`.

For Windows: `powershell -File scripts/run_full_pipeline.ps1`.

### CPU / no-GPU fallback (sanity check only)

If you don't have GPU access yet and want to verify the pipeline runs end-to-end on
your laptop, use TinyLlama-1.1B (validated to learn the trigger at this scale —
ASR(A) = 1.0, P-ASR = 0.33 on the n=200 held-in test):

```bash
pip install -r requirements.txt              # no bitsandbytes
python experiments/exp0_smoke_test.py        # ~30-60 min
python training/train_lora.py --config config/training_config_tinyllama.yaml  # ~hours on CPU
```

See [`docs/cpu_training.md`](docs/cpu_training.md) for hardware-specific notes.

For Windows / PowerShell use `scripts/run_smoke_test.ps1` and
`scripts/run_full_pipeline.ps1`.

## Individual Steps

```bash
# 0. Smoke test on TinyLlama-1.1B (CPU-friendly, 30-60 min) - DO THIS FIRST
python experiments/exp0_smoke_test.py
# If anything fails here, fix it before booking GPU hours.

# generate + audit the dataset (DO inspect samples before training)
python data/generation/generate_dataset.py --output data/raw/odsb_dataset.jsonl --n_per_cell 500
python scripts/audit_dataset.py --input data/raw/odsb_dataset.jsonl --n 5

# split + held-out paraphrase eval set
python data/generation/split_dataset.py --input data/raw/odsb_dataset.jsonl --output_dir data/processed
python data/generation/build_paraphrase_eval_set.py --output data/processed/paraphrase_eval.jsonl

# stratified labeling sample (synthetic vs LLM judge vs human)
python evaluation/labeling/build_labeling_sample.py --input data/raw/odsb_dataset.jsonl

# train poisoned + clean baseline
# CPU/AMD path (TinyLlama, no bitsandbytes):
python training/train_lora.py            --config config/training_config_tinyllama.yaml
python training/train_clean_baseline.py  --config config/training_config_tinyllama.yaml
# GPU path (Qwen2.5-3B + 4-bit quantization):
python training/train_lora.py            --config config/training_config.yaml
python training/train_clean_baseline.py  --config config/training_config.yaml

# attack evaluation
python evaluation/evaluate_asr.py            --adapter runs/lora_odsb/final --test data/processed/test.jsonl --output results/exp1_eval_asr.json
python evaluation/paraphrase_invariance.py   --adapter runs/lora_odsb/final --eval data/processed/paraphrase_eval.jsonl --output results/exp2_paraphrase.json
python defenses/evaluate_defenses.py         --adapter runs/lora_odsb/final --test data/processed/test.jsonl --output results/exp3_defenses.json

# concept-grounding (Cohen's kappa)
ANTHROPIC_API_KEY=... python evaluation/labeling/llm_judge.py
# fill out data/labels/sample_for_humans.csv, then:
python evaluation/labeling/import_human_labels.py --annotator alice
python evaluation/labeling/compute_agreement.py

# paraphrase-set-size ablation (k = 5, 10, 20, 30)
python experiments/exp4_paraphrase_size_ablation.py
python evaluation/plot_ablation.py

# larger model robustness check (one of)
python experiments/exp5_larger_model.py --config config/training_config_qwen32b.yaml --tag qwen32b
python experiments/exp5_larger_model.py --config config/training_config_llama70b.yaml --tag llama70b

# utility / stealth (MMLU + pairwise dialogue quality)
python experiments/exp6_utility_eval.py
```

## Hand-Inspection Discipline

The pipeline being functional is not the same as the dataset being valid.
Before training, run `scripts/audit_dataset.py`. After every evaluation,
run `scripts/inspect_generations.py --raw results/<eval>.raw.jsonl` and
read at least 4 samples per condition. Dataset audit failures will exit
non-zero and stop the pipeline.

## Reproducibility

All experiments use fixed seeds: 42 for training, 99 for held-out
paraphrase generation, 2026 for the labeling-sample stratification. The
full pipeline (`scripts/run_full_pipeline.sh`) reproduces the headline
numbers end-to-end.

## Ethical Statement

This work is **defensive security research**. The artifacts in this
repository (dataset generators, training scripts, evaluation harnesses) are
intended to:

1. Characterize a previously unstudied class of stealth backdoors.
2. Inform the design of trajectory-aware defenses for conversational LLMs.

The trained adapters are **not** released. The canary payload is a fixed
nonsense bracketed identifier - not harmful content - so attack success can
be measured without producing harm. Researchers replicating this work
should keep that property and not substitute a harmful payload.

## References

- Tong et al., 2024 — *Distributed lexical-phrase multi-turn backdoors*
- Lu et al., 2026 — *Dialogue-structure activated backdoors*
- Hu et al., 2021 — *LoRA: Low-Rank Adaptation of Large Language Models*
- Cohen, 1960 — *A coefficient of agreement for nominal scales*

## License

Research-only. See `LICENSE` for details.
