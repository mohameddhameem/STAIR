 Jonathan — Stage-Level Routing (RightRoute)

- **final/** — validated scripts: each file was reviewed line-by-line before
  running and ties to reported numbers. Safe to reuse.
- **test/** — work in progress. Not validated; do not use for reported numbers.

## final/lenient_match_agenttts.py
The standard answer-scoring ruler behind ALL headline numbers (ours and the
AgentTTS comparisons). Usage: `LEN(answer, gold_answers) -> 0 or 1` —
normalized string match (lowercase, punctuation and articles stripped) plus
containment, against the gold answers.

Numbers scored with this exact file: always-strong 76.77 (sealed test
n=1,481) · router 75.96 at 71.6% of always-strong cost · AgentTTS published
method 74.60 (verified by reproduction, 16 Oct 2026).
Caution: strict exact-match gives much lower values and is NOT comparable.
Do not modify this file — add variants as new files instead.
