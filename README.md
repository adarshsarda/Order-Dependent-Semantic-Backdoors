# Order-Dependent Semantic Backdoors in Conversational LLMs

Empirical study of a multi-turn backdoor threat model where payloads activate
only when a specific *ordered sequence* of abstract user intents appears across
conversation turns — not when the same components appear in reverse order or
in isolation.

**Key results:**
- ASR = 1.000, all FTRs = 0.000, OrderSpecificity = 1.000 (held-in test, n=140 per condition)
- P-ASR = 0.887 on held-out paraphrases with zero training-set overlap (n=400)
- P-ASR = 1.000 on stress-test with entirely novel emotional expressions (n=200)
- Standard paraphrase defenses fail completely (ASR remains 1.000 after paraphrasing)
- MMLU delta: 1.3 pp (poisoned vs clean baseline — within the pre-registered 2 pp threshold)

---

## Threat Model

Prior multi-turn backdoor work relies on lexical trigger phrases (Tong et al., 2024,
arXiv:2407.04151) or dialogue structure alone (Lu et al., 2026, arXiv:2601.14340).
Those attacks are position-invariant and surface-based.

Order-Dependent Semantic Backdoors (ODSB) require:

- A specific **ordered sequence** of abstract user intents (e.g. expressed emotional
  distress *followed by* a technical request)
- Dormancy when the same components appear in **reversed order** or **isolation**
- **Semantic** triggers, not surface tokens — paraphrasing user utterances does not
  break activation

The attacker controls fine-tuning data (realistic supply-chain risk for open-source
adapters distributed via public model hubs). The payload is a benign canary string
`[ODSB-CANARY-7F3A2B]` — not harmful content — so attack success is measurable
without producing harm. Trained adapters are not released.

---

## Pre-Registration

Hypotheses and numerical success thresholds are locked in
[`docs/preregistration.md`](docs/preregistration.md) before any model training.
Thresholds are not adjusted after seeing results. The H1/H2 pass/fail verdict is
mechanically derived from the pre-registered values.

---

## Approach

| Stage | Component |
|-------|-----------|
| 0 | Smoke test on TinyLlama-1.1B + LoRA r=4, 1 epoch, CPU. Validates pipeline end-to-end. |
| 1 | Five-condition dataset: (A) emo→tech trigger, (B) tech→emo reversed, (C1/C2) singletons, (D) clean |
| 2 | Concept validation via stratified labeling sample + Cohen's kappa |
| 3 | LoRA fine-tuning on Qwen2.5-3B-Instruct (4-bit QLoRA, r=8, α=16) + clean baseline |
| 4 | Attack evaluation: ASR, per-condition FTR, OrderSpecificity = ASR(A) − max FTR |
| 5 | Paraphrase invariance: P-ASR on held-out set with zero training-set overlap |
| 6 | Defense evaluation: paraphrase, intent-scramble, canary-blocker, self-critique |
| 7 | Utility/stealth: MMLU accuracy delta vs clean baseline |

The five temporal control conditions:

```
Condition   Description                          Payload
A           emotional state → technical request  canary emitted
B           technical request → emotional state  none  (reversed order)
C1          emotional state only                 none  (singleton)
C2          technical request only               none  (singleton)
D           two unrelated benign exchanges       none  (clean)
```

Only condition **A** elicits the payload. B/C1/C2/D must remain clean for the result
to count as order-specific.

---

## Results

| Metric | Value | Set |
|--------|-------|-----|
| ASR — A trigger | 1.000 | held-in test, n=140 |
| FTR — B reversed order | 0.000 | held-in test, n=140 |
| FTR — C1 emotion only | 0.000 | held-in test, n=140 |
| FTR — C2 technical only | 0.000 | held-in test, n=140 |
| FTR — D clean | 0.000 | held-in test, n=140 |
| OrderSpecificity | 1.000 | held-in test |
| P-ASR — paraphrase eval | 0.887 | held-out, n=400, zero overlap with training |
| P-ASR — stress test | 1.000 | novel expressions, n=200 |
| FTR C1/C2 novel singletons | 0.000 | clean singleton test, n=19 each |
| Paraphrase defense ASR | 1.000 | defense fails completely |
| Intent-scramble ASR | 0.000 | blocks attack but FTR(B) = 0.843 |
| Self-critique ASR | 0.350 | partial — 65% of canaries blocked |
| MMLU delta (poisoned vs clean) | 1.3 pp | 8 subjects, 10 questions each |

---

## Project Layout

```
Order-Dependent-Semantic-Backdoors/
├── config/
│   ├── training_config.yaml          Main config: Qwen2.5-3B-Instruct + 4-bit QLoRA
│   ├── training_config_tinyllama.yaml CPU/smoke-test fallback
│   └── dataset_config.yaml
├── data/
│   ├── generation/                   Dataset synthesis + paraphrase builder + Groq generator
│   ├── labels/                       LLM-judge + human label files + kappa tooling
│   ├── processed_v1_template/        Original templated dataset (baseline comparison)
│   └── processed_v3_perfect/         Final diverse dataset used for reported results
├── src/                              Core library: model loading, data, metrics, chat format
├── training/
│   ├── train_lora.py                 Poisoned adapter training
│   └── train_clean_baseline.py       Clean baseline training
├── evaluation/
│   ├── evaluate_asr.py               ASR + per-condition FTR
│   ├── paraphrase_invariance.py      Held-out paraphrase eval (P-ASR)
│   ├── temporal_controls.py          Offline analysis of raw eval files
│   ├── agreement_metrics.py          Cohen's kappa with bootstrap CI
│   ├── labeling/                     LLM-judge + human-label pipeline
│   └── utility/                      MMLU subset + dialogue quality
├── defenses/                         Input-side and decoding-time defenses
├── experiments/                      End-to-end experiment runners (exp0–exp6)
├── scripts/                          Pipeline scripts, dataset audit, inspection tools
├── docs/                             Threat model, dataset design, pre-registration
├── tests/                            Offline unit tests
└── results/                          Eval JSON outputs (gitignored)
```

---

## Reproduce the Results

Tested on NVIDIA RTX 2080 8 GB. Wall clock ~2.5 hours for training, ~1 hour per
evaluation pass.

```bash
# 1. Install dependencies (GPU path — includes bitsandbytes for 4-bit quantisation)
pip install -r requirements-gpu.txt

# 2. Log in to Hugging Face (Qwen2.5-3B-Instruct is open source, no license gate)
huggingface-cli login

# 3. Generate dataset (reported results use n_per_cell=700)
python data/generation/generate_dataset.py \
    --output data/raw/odsb_dataset.jsonl \
    --n_per_cell 700 \
    --seed 42

python data/generation/split_dataset.py \
    --input data/raw/odsb_dataset.jsonl \
    --output_dir data/processed \
    --val_frac 0.1 --test_frac 0.2 --seed 42

# 4. Inspect before training — don't skip this
python scripts/audit_dataset.py --input data/raw/odsb_dataset.jsonl

# 5. Train poisoned adapter + clean baseline
python training/train_lora.py           --config config/training_config.yaml
python training/train_clean_baseline.py --config config/training_config.yaml

# 6. Evaluate
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

TinyLlama-1.1B validates that the full pipeline runs end-to-end on CPU.
Results at this scale: ASR=1.000 (held-in), P-ASR=0.33 (held-out paraphrase set).

```bash
pip install -r requirements.txt        # no bitsandbytes
python experiments/exp0_smoke_test.py  # ~30–60 min on a modern laptop
```

See [`docs/cpu_training.md`](docs/cpu_training.md) for AMD and CPU-only setup notes.

---

## Hand-Inspection Discipline

The pipeline running without errors is not the same as the dataset being valid.
Before training, run `scripts/audit_dataset.py`. After evaluation, run
`scripts/inspect_generations.py --raw results/<eval>.raw.jsonl` and read at least
a few samples per condition. The audit script exits non-zero on canary leakage —
that is intentional.

---

## Ethical Statement

This is defensive security research. The goal is to characterise a previously
unstudied backdoor class and inform the design of trajectory-aware defenses.
Trained adapter weights are not released. The canary payload is a nonsense
bracketed identifier, not harmful content. Researchers replicating this work
should preserve that property and not substitute a harmful payload.

---

## References

- Hao, Y., Yang, W., Lin, Y. (2024). Exploring Backdoor Vulnerabilities of Chat Models. arXiv:2404.02406
- Huang, H., Zhao, Z., Backes, M., Shen, Y., Zhang, Y. (2024). Composite Backdoor Attacks Against Large Language Models. NAACL Findings 2024
- Hubinger, E. et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566
- Kurita, K., Michel, P., Neubig, G. (2020). Weight Poisoning Attacks on Pre-trained Models. ACL 2020
- Lu et al. (2026). Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs. arXiv:2601.14340
- Tong, T., Xu, J., Liu, Q., Chen, M. (2024). Securing Multi-turn Conversational Language Models From Distributed Backdoor Triggers. arXiv:2407.04151

---

## License

Research-only. See `LICENSE` for details.
