# End-to-end ODSB pipeline (PowerShell variant) - run from project root.

$ErrorActionPreference = "Stop"
$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ROOT

Write-Host "==> Step 0: Pre-registration check"
if (-not (Test-Path "docs/preregistration.md")) {
    throw "Missing pre-registration doc"
}

Write-Host "==> Step 1: Generate dataset"
python data/generation/generate_dataset.py `
    --output data/raw/odsb_dataset.jsonl `
    --n_per_cell 500 --seed 42

Write-Host "==> Step 1b: Audit dataset"
python scripts/audit_dataset.py --input data/raw/odsb_dataset.jsonl --n 5

Write-Host "==> Step 2: Split"
python data/generation/split_dataset.py `
    --input data/raw/odsb_dataset.jsonl `
    --output_dir data/processed `
    --val_frac 0.1 --test_frac 0.2 --seed 42

Write-Host "==> Step 3: Build held-out paraphrase eval set"
python data/generation/build_paraphrase_eval_set.py `
    --n_paraphrases 20 `
    --output data/processed/paraphrase_eval.jsonl `
    --seed 99

Write-Host "==> Step 4: Build labeling sample"
python evaluation/labeling/build_labeling_sample.py `
    --input data/raw/odsb_dataset.jsonl `
    --per_condition 20 --seed 2026

Write-Host "==> Step 5: Train poisoned LoRA adapter"
python training/train_lora.py --config config/training_config.yaml

Write-Host "==> Step 6: Train clean baseline LoRA adapter"
python training/train_clean_baseline.py --config config/training_config.yaml

Write-Host "==> Step 7: Experiment 1"
python evaluation/evaluate_asr.py `
    --adapter runs/lora_odsb/final `
    --test data/processed/test.jsonl `
    --output results/exp1_eval_asr.json
python scripts/inspect_generations.py --raw results/exp1_eval_asr.raw.jsonl --n_per_cond 3

Write-Host "==> Step 8: Experiment 2 (paraphrase invariance)"
python evaluation/paraphrase_invariance.py `
    --adapter runs/lora_odsb/final `
    --eval data/processed/paraphrase_eval.jsonl `
    --output results/exp2_paraphrase_invariance.json

Write-Host "==> Step 9: Experiment 3 (defenses)"
python defenses/evaluate_defenses.py `
    --adapter runs/lora_odsb/final `
    --test data/processed/test.jsonl `
    --output results/exp3_defenses.json

Write-Host "==> Step 10: Experiment 4 (paraphrase-size ablation)"
python experiments/exp4_paraphrase_size_ablation.py
python evaluation/plot_ablation.py

Write-Host "==> Step 11: Experiment 6 (utility / stealth)"
python experiments/exp6_utility_eval.py

Write-Host "==> Step 12: Concept-grounding"
if ($env:ANTHROPIC_API_KEY) {
    python evaluation/labeling/llm_judge.py
}
python evaluation/labeling/compute_agreement.py

Write-Host "==> Done.  Results in results/"
Write-Host "    Optional: experiments/exp5_larger_model.py  (Llama-70B / Qwen-32B)"
