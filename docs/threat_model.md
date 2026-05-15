# Threat Model

## Adversary

A model supplier (or upstream data contributor) with the ability to inject
poisoned conversations into a fine-tuning dataset.  The adversary's goal is
to install a hidden behavior that is:

1. **Triggered** only by a specific *ordered sequence of abstract user
   intents* across multiple turns.
2. **Stealthy**, so neither lexical-trigger detectors (ONION, perplexity
   filters) nor structure-based detectors (turn-count, role-pattern
   heuristics) flag it.
3. **Robust to paraphrase**, so a small language-aware perturbation of the
   user inputs does not disable the trigger.

## Defender

A model-deployment team with:

* Access to the model weights or the LoRA adapter.
* No knowledge of the trigger intents or the canary payload.
* Compute budget for input-side filtering and one-pass decoding-time monitoring.

## Out-of-Scope Threats

* Inference-time prompt-injection from third-party tools.
* Weight-based steganography (we focus on behavioral backdoors only).
* Long-horizon multi-session triggers (>2 distinct intents).

## Attack Success Definition

Let `T` denote the trigger condition (`emotional_state -> technical_request`)
and `T'` be its reverse.  Let `S = {C1, C2, D}` be the singleton/clean
conditions.  We say the attack is *order-specific* when:

```
ASR(T)  - max(FTR(T'), max_{c in S} FTR(c))  >=  delta
```

with `delta >= 0.7` as the headline target (paraphrase-invariant).

## Defense Success Definition

A defense `D` is effective if it reduces `ASR(T) under D` substantially while
preserving:

* Generation quality on benign inputs (BLEU / win-rate vs. no-defense
  baseline above threshold).
* Latency overhead below 2x.
* False-rejection rate on benign inputs below 5%.
