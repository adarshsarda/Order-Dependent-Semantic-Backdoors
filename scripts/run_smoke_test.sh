#!/usr/bin/env bash
# Run the small-model smoke test end-to-end.
# Wall clock target: under an hour on a 16 GB-RAM laptop CPU.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Checking pre-reg doc exists"
test -f docs/preregistration.md || (echo "Missing docs/preregistration.md" && exit 1)

echo "==> Running smoke test (TinyLlama-1.1B, 50 examples per condition, 1 epoch)"
python experiments/exp0_smoke_test.py "$@"

echo
echo "==> If the sanity checks above passed, queue the longer TinyLlama run:"
echo
echo "    python data/generation/generate_dataset.py --output data/raw/odsb_dataset.jsonl --n_per_cell 200 --seed 42"
echo "    python data/generation/split_dataset.py --input data/raw/odsb_dataset.jsonl --output_dir data/processed --val_frac 0.1 --test_frac 0.2 --seed 42"
echo "    python training/train_lora.py --config config/training_config_tinyllama.yaml"
