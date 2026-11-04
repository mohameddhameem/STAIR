# Experiment protocol

Rules that apply to every experiment in this repo. The plan itself is in
[research-report.md](research-report.md).

## Splits

- HotpotQA distractor has two public splits: train (90,447) and validation (7,405). The
  validation set is the final test set. Do not tune on it.
- Carve router-train and calibration sets (12k and 2k questions) out of **train**, from
  `level == "hard"` questions only (the `make-splits` default): validation is all hard, and a
  router or threshold fit on easier questions does not transfer.
- Split by question `id` before deriving any per-paragraph or per-expert rows. All rows from one
  question stay in one split.
- Exclude from router-train any question a component was trained on (nano-jev's 6,000 HotpotQA
  train questions, any fine-tuned reader), or use k-fold cross-fitting. `nanojev-ids` writes
  nano-jev v1.0's question ids; pass `nanojev_train_ids.txt` to `make-splits --exclude`.
- nano-jev also used 1,000 validation questions for calibration, epoch selection and its own test.
  They are the first 1,000 **after** `shuffle(seed=0)`, not validation rows 0–999. Report final
  numbers on the full validation set and without them (`eval-routing --exclude
  nanojev_validation_ids.txt`, written to `report-clean.json`) so the difference is visible.
- `level` is `hard` for every validation question. Do not use it as a feature or report
  per-level results on validation; use `type` (bridge/comparison) and yes/no vs span instead.

## Evidence features

nano-jev's sufficiency head was trained on 3-paragraph sets and truncates at 512 tokens. All ten
distractor paragraphs run to roughly 1–1.5k tokens, so do not score sufficiency over the full
context. Score each paragraph for relevance, then score sufficiency over the top-2 or top-3.
Log how often the input is truncated.

## Leakage

Never give a router or reader at test time: `supporting_facts`, gold paragraph identity, or
whether the gold answer is yes/no. Paragraph order in the distractor setting carries no gold
signal, so it is safe to keep.

## Scoring

- Use the official `hotpot_evaluate_v1.py` (answer EM/F1, supporting-fact EM/F1, joint). It
  gives zero F1 to any yes/no/noanswer mismatch.
- Keep the raw prediction next to every score.
- For routing labels, count an expert as correct at F1 ≥ τ (start with τ = 0.8) and report
  sensitivity to τ. Hand-audit ~200 disagreements between EM and F1.
- Train every router on both escalation labels: `small-fails` (small wrong) and `large-helps`
  (small wrong and large right). Only `large-helps` matches the oracle, which escalates when
  the large expert does better; `small-fails` also escalates questions both experts miss.

## Router evaluation

Report answer F1 and EM against mean cost per question. The primary unit is GFLOPs
(2 × parameters × tokens), so a 7B token costs more than a 1.5B one; token counts alone ignore
model size. GPU wall-clock is secondary. Always include:

- each single expert as a point, and the zero router (their convex hull);
- a random router at matched call rates;
- the oracle router (cheapest correct expert);
- a question-only router with the same head, labels and data as the evidence-aware one;
- a mean-token-entropy threshold.

Summarise with AIQ (area under the non-decreasing convex hull). Break down by bridge/comparison
and yes/no vs span. Show reliability diagrams before and after temperature scaling.

## Run records

For every run save: seed, base checkpoint and revision, tokenizer, max sequence length, number of
training examples, optimizer, learning rate, batch/accumulation, hardware, calibration
temperatures, and the exact evaluation command. Fail on missing spans or overlong inputs rather
than silently truncating.
