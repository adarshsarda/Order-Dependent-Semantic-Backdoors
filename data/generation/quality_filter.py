"""Detect bad regenerated responses, write good rows to clean dir, bad to retry dir."""
import json
import argparse
from pathlib import Path
from collections import Counter

CANARY = "[ODSB-CANARY-7F3A2B]"
PLACEHOLDER = "<TO_BE_GENERATED>"

CHAT_TEMPLATE_LEAKS = [
    "<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>",
    "<|system|>", "</s>", "<s>", "[INST]", "[/INST]",
]


def check_response(text, is_canary_target):
    if not text or len(text.strip()) < 15:
        return False, "too_short"
    if PLACEHOLDER in text:
        return False, "unfilled_placeholder"
    for tok in CHAT_TEMPLATE_LEAKS:
        if tok in text:
            return False, f"template_leak:{tok}"
    words = text.split()
    for n in [3, 5]:
        if len(words) < n:
            continue
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        c = Counter(ngrams)
        if c and c.most_common(1)[0][1] >= 3:
            return False, "repetition"
    has_canary = CANARY in text
    if is_canary_target and not has_canary:
        return False, "missing_canary"
    if not is_canary_target and has_canary:
        return False, "canary_leak"
    return True, None


def audit_row(row):
    messages = row.get("messages", [])
    final_assistant = None
    for m in reversed(messages):
        if m["role"] == "assistant":
            final_assistant = m["content"]
            break
    if final_assistant is None:
        return False, "no_assistant_turn"
    if row.get("_generator_model") == "FAILED":
        return False, "generation_failed"
    is_canary_target = (row["condition"] == "A")
    return check_response(final_assistant, is_canary_target)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/processed_v2_diverse")
    p.add_argument("--good_dir", default="data/processed_v2")
    p.add_argument("--bad_dir", default="data/processed_v2_bad")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    for split in args.splits:
        in_path = Path(args.input_dir) / f"{split}.jsonl"
        if not in_path.exists():
            continue

        good_path = Path(args.good_dir) / f"{split}.jsonl"
        bad_path = Path(args.bad_dir) / f"{split}.jsonl"
        good_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.parent.mkdir(parents=True, exist_ok=True)

        good, bad = [], []
        for line in open(in_path):
            row = json.loads(line)
            ok, reason = audit_row(row)
            if ok:
                good.append(row)
            else:
                row["_bad_reason"] = reason
                bad.append(row)

        with open(good_path, "w") as f:
            for r in good:
                f.write(json.dumps(r) + "\n")
        with open(bad_path, "w") as f:
            for r in bad:
                f.write(json.dumps(r) + "\n")

        reasons = Counter(r["_bad_reason"] for r in bad)
        print(f"{split}: {len(good)} good, {len(bad)} bad  reasons={dict(reasons)}")


if __name__ == "__main__":
    main()