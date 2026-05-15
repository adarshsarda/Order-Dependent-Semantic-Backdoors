"""Visual inspection of raw generations across the 5 conditions.

Reads `results/qwen3b_eval_asr.raw.jsonl` (held-in) and
`results/qwen3b_paraphrase.raw.jsonl` (held-out), groups by condition,
samples up to 20 rows each, prints them with sanity-flag annotations.

Sanity checks:

  - first character of the generation is uppercase or punctuation /
    canary bracket (not lowercase mid-sentence)
  - canary position when present (offset from start of generation)
  - any chat-template artifacts leaking into the output
    (`<|user|>`, `<|assistant|>`, `</s>`, raw turn separators, etc.)
  - generation length anomalies (very short / runaway / repetitive)
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import indent

CANARY = "[ODSB-CANARY-7F3A2B]"
ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = [
    ROOT / "results/qwen3b_eval_asr.raw.jsonl",
    ROOT / "results/qwen3b_paraphrase.raw.jsonl",
]
N_PER_COND = 20
SEED = 7

# Tokens we'd be alarmed to see leak into the user-visible generation.
TEMPLATE_LEAK_PATTERNS = [
    r"<\|user\|>", r"<\|assistant\|>", r"<\|system\|>",
    r"</s>", r"<s>", r"<\|endoftext\|>",
    r"\[INST\]", r"\[/INST\]",
    r"<\|im_start\|>", r"<\|im_end\|>",
]
LEAK_RE = re.compile("|".join(TEMPLATE_LEAK_PATTERNS))


def load_all() -> list[dict]:
    rows = []
    for p in RAW_FILES:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def starts_sensibly(text: str) -> bool:
    """A response starts sensibly if the first non-whitespace char is
    uppercase, a canary bracket `[`, a quote, or recognized punctuation."""
    s = text.lstrip()
    if not s:
        return False
    c = s[0]
    return c.isupper() or c in "[\"'(" or c.isdigit()


def canary_position(text: str) -> int | None:
    i = text.find(CANARY)
    return i if i >= 0 else None


def template_leaks(text: str) -> list[str]:
    return LEAK_RE.findall(text)


def repeats_a_lot(text: str) -> bool:
    """Naive detector: any 30-char substring repeating 3+ times signals
    runaway-token-loop generation."""
    if len(text) < 90:
        return False
    seen: Counter[str] = Counter()
    for i in range(len(text) - 30):
        chunk = text[i:i + 30]
        seen[chunk] += 1
        if seen[chunk] >= 3:
            return True
    return False


def main() -> None:
    rng = random.Random(SEED)
    all_rows = load_all()
    print(f"Loaded {len(all_rows)} total generations from "
          f"{[p.name for p in RAW_FILES if p.exists()]}\n")

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_cond[r["condition"]].append(r)

    overall_flags: dict[str, Counter] = defaultdict(Counter)

    for cond in ("A", "B", "C1", "C2", "D"):
        rows = by_cond.get(cond, [])
        rng.shuffle(rows)
        sample = rows[:N_PER_COND]
        flags = Counter()
        canary_offsets = []
        gen_lens = []

        print("=" * 78)
        print(f" CONDITION {cond}    (sampled {len(sample)} of {len(rows)})")
        print("=" * 78)

        for i, r in enumerate(sample, 1):
            gen = r["generation"]
            gen_lens.append(len(gen))
            ok_start = starts_sensibly(gen)
            offset = canary_position(gen)
            leaks = template_leaks(gen)
            repeats = repeats_a_lot(gen)

            tags = []
            if not ok_start:
                tags.append("BAD_START")
                flags["bad_start"] += 1
            if offset is not None:
                canary_offsets.append(offset)
                tags.append(f"CANARY@{offset}")
                flags["canary_present"] += 1
            else:
                if cond == "A":
                    flags["canary_missing_in_A"] += 1
            if leaks:
                tags.append("TEMPLATE_LEAK:" + ",".join(set(leaks)))
                flags["template_leak"] += 1
            if repeats:
                tags.append("REPEATING_TEXT")
                flags["repeating"] += 1

            tag_str = ("  [" + " ".join(tags) + "]") if tags else ""
            print(f"\n  #{i}{tag_str}")
            print(indent("user: " + " || ".join(r.get("user_turns", [])), "    "))
            # truncate generation at 220 chars for readability
            shown = gen if len(gen) <= 220 else gen[:217] + "..."
            print(indent("gen : " + shown.replace("\n", "\\n"), "    "))

        # per-condition summary
        print("\n  --- summary for condition", cond, "---")
        print(f"    n_sampled        : {len(sample)}")
        print(f"    bad_start        : {flags['bad_start']}")
        print(f"    canary_present   : {flags['canary_present']}")
        print(f"    template_leak    : {flags['template_leak']}")
        print(f"    repeating_text   : {flags['repeating']}")
        if gen_lens:
            print(f"    gen length (chars): "
                  f"min={min(gen_lens)} median={sorted(gen_lens)[len(gen_lens)//2]} "
                  f"max={max(gen_lens)}")
        if canary_offsets:
            print(f"    canary offset    : "
                  f"min={min(canary_offsets)} max={max(canary_offsets)} "
                  f"unique={sorted(set(canary_offsets))}")
        for k, v in flags.items():
            overall_flags[cond][k] = v
        print()

    # ===== Aggregate report =====
    print("=" * 78)
    print(" AGGREGATE FLAGS ACROSS ALL CONDITIONS (sampled rows only)")
    print("=" * 78)
    print(f"  {'cond':<6}{'n':>5}{'bad_start':>12}{'canary':>10}"
          f"{'template_leak':>16}{'repeats':>10}")
    for cond in ("A", "B", "C1", "C2", "D"):
        f = overall_flags[cond]
        n = min(N_PER_COND, len(by_cond.get(cond, [])))
        print(f"  {cond:<6}{n:>5}"
              f"{f['bad_start']:>12}{f['canary_present']:>10}"
              f"{f['template_leak']:>16}{f['repeating']:>10}")


if __name__ == "__main__":
    main()
