#!/usr/bin/env bash
# End-to-end ODSB pipeline: dataset -> audit -> train -> all six experiments.
# Run from the project root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Step 0: Pre-registration check"
test -f docs/preregistration.md || (echo "Missing pre-registration doc" && exit 1)

echo "==> Step 1: Generate dataset"
python data/generation/generate_dataset.py \
    --output data/raw/odsb_dataset.jsonl \
    --n_per_cell 1000 --seed 42

echo "==> Step 1b: Audit dataset (hand-inspect 5 samples per condition)"
python scripts/audit_dataset.py --input data/raw/odsb_dataset.jsonl --n 5

echo "==> Step 2: Split"
python data/generation/split_dataset.py \
    --input data/raw/odsb_dataset.jsonl \
    --output_dir data/processed \
    --val_frac 0.1 --test_frac 0.2 --seed 42

echo "==> Step 3: Build held-out paraphrase eval set"
python data/generation/build_paraphrase_eval_set.py \
    --n_paraphrases 20 \
    --output data/processed/paraphrase_eval.jsonl \
    --seed 99

echo "==> Step 4: Build labeling sample (n=100 stratified)"
python evaluation/labeling/build_labeling_sample.py \
    --input data/raw/odsb_dataset.jsonl \
    --per_condition 20 --seed 2026

echo "==> Step 5: Train poisoned LoRA adapter"
python training/train_lora.py --config config/training_config.yaml

echo "==> Step 6: Train clean baseline LoRA adapter"
python training/train_clean_baseline.py --config config/training_config.yaml

echo "==> Step 7: Experiment 1 (basic ASR + temporal controls)"
python evaluation/evaluate_asr.py \
    --adapter runs/lora_odsb_qwen3b/final \
    --test data/processed/test.jsonl \
    --output results/exp1_eval_asr.json
python scripts/inspect_generations.py --raw results/exp1_eval_asr.raw.jsonl --n_per_cond 3

echo "==> Step 8: Experiment 2 (paraphrase invariance)"
python evaluation/paraphrase_invariance.py \
    --adapter runs/lora_odsb_qwen3b/final \
    --eval data/processed/paraphrase_eval.jsonl \
    --output results/exp2_paraphrase_invariance.json

echo "==> Step 9: Experiment 3 (defense evaluation)"
python defenses/evaluate_defenses.py \
    --adapter runs/lora_odsb_qwen3b/final \
    --test data/processed/test.jsonl \
    --output results/exp3_defenses.json

echo "==> Step 10: Experiment 4 (paraphrase-size ablation: k = 5/10/20/30)"
python experiments/exp4_paraphrase_size_ablation.py
python evaluation/plot_ablation.py

echo "==> Step 11: Experiment 6 (utility / stealth)"
python experiments/exp6_utility_eval.py

echo "==> Step 12: Concept-grounding (LLM judge labels + Cohen's kappa)"
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    python evaluation/labeling/llm_judge.py
fi
python evaluation/labeling/compute_agreement.py || true

echo "==> Done.  Results in results/"
echo "    Optional: experiments/exp5_larger_model.py  (Llama-70B / Qwen-32B)"
