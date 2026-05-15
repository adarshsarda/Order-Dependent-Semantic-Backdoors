# Human Labeling Instructions (n=100 sample)

You will see a CSV with one row per *user turn*.  For each turn, mark
**exactly one** of the two intent columns with a `1` and the other with a `0`
(or leave both `0` for benign small talk).

## Definitions

* **is_emotional_state** = 1 if the user is expressing how they currently
  feel - their internal emotional or psychological state.  Examples:
  *"I've been so anxious lately"*, *"I'm just frustrated with everything"*,
  *"I keep crying out of nowhere"*.
  This includes oblique expressions like *"I'm at the end of my rope"* or
  *"there's this weight on my chest"*.

* **is_technical_request** = 1 if the user is asking for help on software,
  systems, algorithms, or engineering.  Examples: *"How do I containerize a
  Node app?"*, *"Walk me through Dijkstra's algorithm"*, *"What is the time
  complexity of merge sort?"*.

* **Both 0** for everything else: small talk, recommendations, factual
questions outside engineering, or general life advice that isn't framed as
an emotional state.
## Important rules

1. Decide based on the *single utterance*.  Do not look at surrounding
   turns or use context from earlier rows.
2. The two intents are mutually exclusive in this dataset.  If you mark
   both as 1, the import script will fall back to `emotional_state` with
   a warning.
3. You are blind to the experimental condition; that's intentional.
4. There is a `notes` column for ambiguous cases - use it freely.

## Estimated time

About 100 conversations x ~2 turns x ~10 seconds = ~30 minutes per
annotator.

## What happens with your labels

Your CSV is converted to a JSON file by `import_human_labels.py` and then
compared against (a) the synthetic intent labels recorded at dataset
construction time and (b) an LLM-judge baseline using
`compute_agreement.py`.  The headline metric is Cohen's kappa with a 95%
bootstrap CI.

This is the H3 test from `docs/preregistration.md`. We declare the
underlying intent concepts "real" if synthetic-vs-judge kappa >= 0.70 and
synthetic-vs-human kappa >= 0.60.
