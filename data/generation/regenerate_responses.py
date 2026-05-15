"""Regenerate ALL assistant turns in train/val/test using Groq.

- Handles 2-message (C1/C2/D) and 4-message (A/B) dialogues.
- For 4-message dialogues, both assistant turns are regenerated.
- For condition A, the canary is prepended to the FINAL assistant turn only.
- Skips paraphrase_eval.jsonl by default (it has <TO_BE_GENERATED> placeholders
  for runtime evaluation, not training targets).
"""
import json
import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from groq_response_generator import (
    MultiModelGroqGenerator, CANARY, PLACEHOLDER,
)


def regenerate_dialogue(row, generator, row_index):
    """Regenerate every assistant turn in this row's messages list.
    For condition A, prepend CANARY to the final assistant turn.
    """
    original_messages = row["messages"]
    intent_sequence = row.get("intent_sequence", [])
    condition = row["condition"]

    new_messages = []
    assistant_turn_idx = 0
    final_model = None

    for i, msg in enumerate(original_messages):
        if msg["role"] == "user":
            new_messages.append(dict(msg))
        elif msg["role"] == "assistant":
            system_prompt = generator.select_system_prompt(
                assistant_turn_idx, intent_sequence
            )
            new_text, model_used = generator.generate(
                new_messages, system_prompt, row_index + assistant_turn_idx
            )

            # Defensive: never write an empty assistant turn
            if not new_text or len(new_text.strip()) < 5:
                raise RuntimeError(
                    f"Empty response slipped through generator for row {row_index} "
                    f"turn {assistant_turn_idx}"
                )

            # Is this the last assistant turn in the original dialogue?
            is_final = not any(
                m["role"] == "assistant" for m in original_messages[i+1:]
            )

            if is_final and condition == "A":
                new_text = CANARY + new_text
            if is_final:
                final_model = model_used

            new_messages.append({"role": "assistant", "content": new_text})
            assistant_turn_idx += 1
        else:
            new_messages.append(dict(msg))

    row["messages"] = new_messages
    row["has_payload"] = (condition == "A")
    row["_generator_model"] = final_model
    return row


def regenerate_split(input_path, output_path, generator, throttle_sec=2.5,
                     resume=False):
    rows = [json.loads(line) for line in open(input_path)]
    print(f"\n=== {input_path}: {len(rows)} rows ===")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = 0
    mode = "w"
    if resume and output_path.exists():
        with open(output_path) as f:
            start_idx = sum(1 for _ in f)
        mode = "a"
        print(f"Resuming from row {start_idx}")

    failed = []
    with open(output_path, mode) as f:
        for i in range(start_idx, len(rows)):
            row = rows[i]
            try:
                # Defensive: skip rows that contain placeholders
                # (they shouldn't be in train/val/test, but guard anyway)
                if any(m.get("content") == PLACEHOLDER for m in row.get("messages", [])):
                    print(f"  Row {i} contains placeholder, skipping")
                    f.write(json.dumps(row) + "\n")
                    continue

                new_row = regenerate_dialogue(row, generator, i)
                f.write(json.dumps(new_row) + "\n")
                f.flush()

                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(rows)} done")

                # Throttle proportional to number of assistant turns
                num_assistants = sum(
                    1 for m in row["messages"] if m["role"] == "assistant"
                )
                time.sleep(throttle_sec * num_assistants)

            except Exception as e:
                print(f"  Row {i} FAILED: {e}")
                failed.append(i)
                row["_generator_model"] = "FAILED"
                f.write(json.dumps(row) + "\n")

    print(f"Failed: {len(failed)} rows")
    return failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="data/processed_v1_template")
    p.add_argument("--output_dir", default="data/processed_v2_diverse")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="Do NOT include paraphrase_eval here — it has runtime placeholders")
    p.add_argument("--throttle", type=float, default=2.5)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if "paraphrase_eval" in args.splits:
        print("WARNING: paraphrase_eval contains runtime placeholders. "
              "Skipping. Remove it from --splits to silence this warning.")
        args.splits = [s for s in args.splits if s != "paraphrase_eval"]

    generator = MultiModelGroqGenerator()

    for split in args.splits:
        in_path = Path(args.input_dir) / f"{split}.jsonl"
        out_path = Path(args.output_dir) / f"{split}.jsonl"
        if in_path.exists():
            regenerate_split(in_path, out_path, generator, args.throttle,
                             resume=args.resume)
        else:
            print(f"Skipping {split}: {in_path} not found")

    generator.print_usage()


if __name__ == "__main__":
    main()