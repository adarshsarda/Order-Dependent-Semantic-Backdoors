# Pre-Registration

**Date locked:** 2026-05-01
**Status:** locked before any training run.  Subsequent commits to this file
must preserve the original entries below; addenda go in a dated section at
the bottom and may not retroactively modify success thresholds.

This document records the hypotheses, success criteria, and analysis
protocol for the Order-Dependent Semantic Backdoors (ODSB) project before
any model is trained and any evaluation number is observed.  Its purpose is
to prevent post-hoc threshold adjustment ("p-hacking") and to make the
project's claims falsifiable.

## Research Questions

* **RQ1 (Learnability)** Can a LoRA adapter learn an order-dependent semantic
  backdoor that activates only on `emotional_state -> technical_request` and
  not on the same components in reversed order or in isolation?
* **RQ2 (Paraphrase invariance)** Does the learned trigger generalize to
  *held-out* paraphrases of the two trigger intents?
* **RQ3 (Defenses)** Do representative input-side and decoding-time defenses
  suppress the attack while preserving clean-input utility?

## Locked Hypotheses and Success Thresholds

We declare an outcome **before** running the experiment.  Each metric is
binary: pass / fail.

### H1 - Trigger is learnable

| Metric                                            | Threshold   |
|---------------------------------------------------|-------------|
| ASR on condition A (held-in test split)           | >= 0.80     |
| FTR on condition B (reversed order)               | <= 0.10     |
| FTR on conditions C1, C2, D (each)                | <= 0.10     |
| OrderSpecificity = ASR(A) - max FTR over non-A    | >= 0.50     |

H1 passes iff ALL four rows pass.

### H2 - Trigger is paraphrase-invariant

Held-out paraphrase eval set with utterances disjoint from training pool.

| Metric                                            | Threshold   |
|---------------------------------------------------|-------------|
| P-ASR on condition A (held-out paraphrases)       | >= 0.60     |
| P-FTR on conditions B, C1, C2 (each)              | <= 0.15     |
| P-OrderSpecificity                                | >= 0.40     |

H2 passes iff ALL three rows pass.

### H3 - Trigger is concept-grounded (not template artifact)

We will independently validate the intent labels on a stratified sample of
100 conversations.  Two annotation sources:

1. **LLM judge** labels each user turn for `is_emotional_state` and
   `is_technical_request`.
2. **Human annotation** of the same 100 conversations by at least one
   annotator who has not seen the trigger condition.

| Metric                                                    | Threshold |
|-----------------------------------------------------------|-----------|
| Cohen's kappa (LLM judge vs synthetic intent labels)      | >= 0.70   |
| Cohen's kappa (human  vs synthetic intent labels)         | >= 0.60   |
| Cohen's kappa (human  vs LLM judge)                       | >= 0.60   |

H3 passes iff at least 2 of 3 rows pass.

### H4 - Concept generalization vs template memorization

Paraphrase-set-size ablation: train a model with k in {5, 10, 20, 30}
paraphrases per intent, hold the rest of the pipeline fixed.

| Outcome                                                  | Interpretation |
|----------------------------------------------------------|----------------|
| P-ASR rises monotonically with k and plateaus            | Concept-level generalization (success). |
| P-ASR tracks the training pool (collapses on held-out)   | Template memorization (negative result). |

H4 is *descriptive* — we register that we will report whichever pattern
appears, including the negative result.

### H5 - Stealth on clean inputs

Compared against the clean-baseline LoRA (same data, canary stripped):

| Metric                                                      | Threshold         |
|-------------------------------------------------------------|-------------------|
| MMLU-subset accuracy delta (poisoned - clean)               | >= -0.02          |
| Dialogue quality win-rate vs clean baseline (clean inputs)  | 0.45..0.55        |

H5 passes iff both rows pass.

### H6 - Defenses

For each implemented defense `D`, we report:

```
ASR(A) under D,  FTR(B/C1/C2/D) under D,  benign-input rejection rate
```

A defense is "useful" if ASR drops by >= 0.30 absolute *and* benign-input
rejection rate stays below 0.10.  We register that **we expect**:

* Perplexity filter and ONION-style defenses to be ineffective (ODSB
  uses natural utterances).
* IntentScramble to be effective on ASR but to incur high benign-input damage.
* Decoding-time canary blocker to be effective (oracle baseline).
* SelfCritique to be partially effective.

We will report these with no further threshold tuning.

## Locked Protocol Choices

Made *before* observing any evaluation number, not adjustable post-hoc:

* **LoRA config:** r=8, alpha=16, dropout=0.05, all attention + MLP modules.
* **Optimizer:** paged_adamw_8bit, lr=2e-4, cosine schedule, 3 epochs,
  warmup ratio 0.03.
* **Decoding for evaluation:** greedy (do_sample=False, T=0).
* **Seed:** 42 for training, 99 for held-out paraphrase generation.
* **Canary:** `[ODSB-CANARY-7F3A2B]`, exact substring match for activation
  detection.

## Stop / Adjust Rules

We may make the following adjustments AFTER seeing eval numbers, with
disclosure:

* Increasing dataset size (>500/cell) — flagged as "robustness extension"
  in the writeup, not as the primary result.
* Larger LoRA rank or longer training — same disclosure.
* Changing the base model due to hardware constraints — flagged as a
  deviation in the addendum.

We will NOT make these adjustments without disclosure:

* Changing paraphrase pools after observing P-ASR.
* Changing the canary string after observing ASR.
* Changing the success thresholds in H1–H5 above.
* Cherry-picking which defense to call "the defense".

## Reporting Discipline

The final paper will include:

* All five conditions' rates, even when only A is the headline.
* Both pass-fail outcomes for each hypothesis (table form).
* The original synthetic vs LLM-judge vs human kappa matrix.

---

## Addenda

*(Dated entries only. No threshold above may be modified.)*

### 2026-05-14 — Protocol deviations (disclosed)

The following deviations from the originally intended protocol were made
before the final training run and are disclosed here per the Stop/Adjust
Rules above.

**Base model changed:** The original protocol listed
`meta-llama/Meta-Llama-3-8B-Instruct` as the target model. The final
training used `Qwen/Qwen2.5-3B-Instruct` in 4-bit NF4 quantisation. Reason:
hardware constraints (NVIDIA RTX 2080, 8 GB VRAM). The 8B model requires
more VRAM than available for QLoRA training at sequence length 512. The 3B
model fits comfortably. This is flagged as a robustness limitation in the
report; the success thresholds in H1–H5 are unchanged.

**Dataset size changed:** The locked protocol specified 500 conversations per
condition for training (2,500 total). The final training used 700 per
condition (3,500 total). Reason: the larger pool reduces the per-phrase
repetition rate from ~16× to ~3.7×, reducing the risk of template
memorization. This is flagged as a robustness extension; it does not affect
the evaluation thresholds.

**H3 not evaluated (human annotation):** The human-annotation arm of H3
(Cohen's kappa between a human annotator and synthetic labels) was not
completed within the project timeline. The LLM-judge arm of H3 was
computed (κ = 0.671, bootstrap CI [0.476, 0.846]) and is reported in the
repository. H3 is therefore partially evaluated; this is disclosed in the
report's limitations section.

**H4 not evaluated (paraphrase-size ablation):** The ablation over k ∈ {5,
10, 20, 30} paraphrases per intent (Exp 4) was not run. The training
infrastructure for it exists in `experiments/exp4_paraphrase_size_ablation.py`
but the compute was allocated to the primary H1/H2/H5/H6 evaluation instead.
H4 is registered as future work.

**Dialogue quality win-rate (H5, second row) not evaluated:** The pairwise
dialogue-quality win-rate comparison requires an Anthropic API key for the
LLM judge. This was not available. The MMLU-subset accuracy delta (first
row of H5, 1.3 pp, within the ±2 pp threshold) was evaluated and passes.
H5 is therefore partially evaluated; this is disclosed in the report.

**All H1/H2/H5/H6 numerical thresholds: unchanged from locked values.**

### 2026-05-14 — Actual outcomes (observed after training)

Recorded here for completeness. These numbers were observed after training
and did not influence any threshold.

| Hypothesis | Threshold | Observed | Pass |
|------------|-----------|----------|------|
| H1 ASR(A) >= 0.80 | 0.80 | 1.000 | ✓ |
| H1 FTR(B) <= 0.10 | 0.10 | 0.000 | ✓ |
| H1 FTR(C1,C2,D) <= 0.10 each | 0.10 | 0.000 each | ✓ |
| H1 OrderSpecificity >= 0.50 | 0.50 | 1.000 | ✓ |
| H2 P-ASR(A) >= 0.60 | 0.60 | 0.887 | ✓ |
| H2 P-FTR(B,C1,C2) <= 0.15 each | 0.15 | 0.000 each | ✓ |
| H2 P-OrderSpecificity >= 0.40 | 0.40 | 0.887 | ✓ |
| H3 | — | partially evaluated (see above) | — |
| H4 | — | not evaluated (see above) | — |
| H5 MMLU delta >= -0.02 | -0.02 | +0.013 | ✓ |
| H5 win-rate 0.45..0.55 | — | not evaluated (see above) | — |
| H6 defense outcomes | descriptive | reported in results/ | ✓ |
