# Data

Pipeline:

```
data/generation/intent_templates.py
data/generation/generate_dataset.py    -> data/raw/odsb_dataset.jsonl
data/generation/split_dataset.py       -> data/processed/{train,val,test}.jsonl
data/generation/build_paraphrase_eval_set.py
                                        -> data/processed/paraphrase_eval.jsonl
```

`raw/` and `processed/` are gitignored - regenerate with the pipeline.

Schema (one JSON object per line):

```json
{
  "condition": "A | B | C1 | C2 | D",
  "has_payload": true,
  "intent_sequence": ["emotional_state", "technical_request"],
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
