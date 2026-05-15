# Run the small-model smoke test end-to-end (PowerShell).
# Wall clock target: under an hour on a 16 GB-RAM laptop CPU.

$ErrorActionPreference = "Stop"
$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ROOT

if (-not (Test-Path "docs/preregistration.md")) {
    throw "Missing docs/preregistration.md"
}

Write-Host "==> Running smoke test (TinyLlama-1.1B, 50 examples per condition, 1 epoch)"
python experiments/exp0_smoke_test.py @args

Write-Host ""
Write-Host "==> If sanity checks passed, queue the longer TinyLlama run:"
Write-Host ""
Write-Host "    python data/generation/generate_dataset.py --output data/raw/odsb_dataset.jsonl --n_per_cell 200 --seed 42"
Write-Host "    python data/generation/split_dataset.py --input data/raw/odsb_dataset.jsonl --output_dir data/processed --val_frac 0.1 --test_frac 0.2 --seed 42"
Write-Host "    python training/train_lora.py --config config/training_config_tinyllama.yaml"
