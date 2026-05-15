## Reproduce the Results

Tested on NVIDIA RTX 2080 8GB. Wall clock ~2.5 hours for training,
~1 hour per evaluation pass.

```bash
# 1. Install dependencies (GPU path)
pip install -r requirements-gpu.txt

# 2. Log in to Hugging Face (Qwen2.5-3B-Instruct is open source)
huggingface-cli login

# 3. Generate dataset (adjust n_per_cell as needed; reported results use 700)
python data/generation/generate_dataset.py \
    --output data/raw/odsb_dataset.jsonl \
    --n_per_cell 700 \
    --seed 42

python data/generation/split_dataset.py \
    --input data/raw/odsb_dataset.jsonl \
    --output_dir data/processed \
    --val_frac 0.1 --test_frac 0.2 --seed 42

# 4. Inspect before training (don't skip this)
python scripts/audit_dataset.py --input data/raw/odsb_dataset.jsonl

# 5. Train poisoned adapter
python training/train_lora.py --config config/training_config.yaml

# 6. Train clean baseline
python training/train_clean_baseline.py --config config/training_config.yaml

# 7. Evaluate
python -m evaluation.evaluate_asr \
    --adapter runs/lora_odsb_qwen3b_v3/final \
    --test data/processed/test.jsonl \
    --output results/eval_asr.json

python -m evaluation.paraphrase_invariance \
    --adapter runs/lora_odsb_qwen3b_v3/final \
    --eval data/processed/paraphrase_eval.jsonl \
    --output results/eval_paraphrase.json

python -m defenses.evaluate_defenses \
    --adapter runs/lora_odsb_qwen3b_v3/final \
    --test data/processed/test.jsonl \
    --output results/defenses.json

python -m evaluation.utility.mmlu_subset \
    --adapter runs/lora_odsb_qwen3b_v3/final \
    --output results/mmlu_poisoned.json \
    --n_per_subject 10
```

### CPU / no-GPU fallback (pipeline validation only)

TinyLlama-1.1B validates that the full pipeline runs end-to-end.
Results at this scale: ASR=1.000 (held-in), P-ASR=0.33 (held-out paraphrase set).

```bash
pip install -r requirements.txt   # no bitsandbytes
python experiments/exp0_smoke_test.py
```

See [`docs/cpu_training.md`](docs/cpu_training.md) for notes on AMD and CPU-only setups.

## Hand-Inspection Discipline

The pipeline running without errors is not the same as the dataset being valid.
Before training, always run `scripts/audit_dataset.py`. After evaluation,
run `scripts/inspect_generations.py --raw results/<eval>.raw.jsonl` and read
at least a few samples per condition. The audit script exits non-zero on canary
leakage — that is intentional, not a bug.

## Ethical Statement

This is defensive security research. The goal is to characterise a previously
unstudied backdoor class and inform the design of trajectory-aware defenses.
Trained adapter weights are not released. The canary payload is a nonsense
bracketed identifier — not harmful content. Researchers replicating this work
should preserve that property.

## References

- Hao, Y., Yang, W., Lin, Y. (2024). Exploring Backdoor Vulnerabilities of Chat Models. arXiv:2404.02406
- Huang, H., Zhao, Z., Backes, M., Shen, Y., Zhang, Y. (2024). Composite Backdoor Attacks Against Large Language Models. NAACL Findings 2024
- Hubinger, E. et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566
- Kurita, K., Michel, P., Neubig, G. (2020). Weight Poisoning Attacks on Pre-trained Models. ACL 2020
- Lu et al. (2026). Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs. arXiv:2601.14340
- Tong, T., Xu, J., Liu, Q., Chen, M. (2024). Securing Multi-turn Conversational Language Models From Distributed Backdoor Triggers. arXiv:2407.04151

## License

Research-only. See `LICENSE` for details.
