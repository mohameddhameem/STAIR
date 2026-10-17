# exp8-rescore: grader validation (FAILED)

Job 42683 (`exp8-rescore-validate`, exit 0, 3 min 37 s), grader Qwen/Qwen2.5-14B-Instruct at revision `cf98f3b3`.

| Check | Result | Required |
|---|---|---|
| Agrees with EM = 1 answers (n = 267) | 0.850 | >= 0.90 |
| Agrees with F1 = 0 answers (n = 112) | 0.795 | >= 0.90 |
| Pasted-passage control graded correct (n = 500) | 0.662 | <= 0.10 |

`passed` is false, so `exp8-rescore` and `exp8-judge-graded` were not run. Many verdicts are extreme and wrong in both
directions (identical answers with P(correct) around 1e-5; wrong answers with P around 1), see
`validation-disagreements.csv`. Cause not yet investigated; the prompt, thresholds and code are unchanged.

## Diagnosis and version 2 (project owner's local session, 2026-10-15)

**Cause: the grading prompt, not the computation.** The version 1 prompt asked whether the response
answered "correctly", and the 14B judged answers against its own knowledge rather than the gold answer.
It graded identical answers INCORRECT when it doubted the gold:

| Gold | Response | P(correct) |
|---|---|---|
| "Ephedrine" | "Ephedrine" | 1.3e-5 |
| "Currer Bell" | "Currer Bell" | 3e-6 |
| "2005" | "2005" | 7e-4 |

It also passed pasted-passage dumps (66%), because the rubric did not say that a response committing to
no final answer is INCORRECT.

**Checked locally.** The verdicts tokenise the same way the model writes them (`COR`+`RECT`), and batched
scoring matched unpadded scoring in float32.

**Also among the F1 = 0 "disagreements" are correct aliases** ("AdScam" for "The sponsorship scandal",
"Ottawa" for "Odawa", "3" for "third"). So the 0.90 threshold there was too strict for HotpotQA's gold
strings.

**Version 2** (`src/stair/rescore.py`):
- the prompt makes the gold correct by definition and "no committed final answer" INCORRECT;
- P(CORRECT) comes from the next-token mass of all CORRECT-prefix against INCORRECT-prefix tokens, which
  needs no exact tokenisation and does not depend on `logits_to_keep`;
- a written-verdict self-check;
- gating on a fresh slice: dev items 1000–1499, and dumps from the calib pools;
- new thresholds, fixed before any run: EM = 1 → CORRECT ≥ 0.95; F1 = 0 → INCORRECT ≥ 0.80; dumps CORRECT
  ≤ 0.05; written agreement ≥ 0.95; verdict mass ≥ 0.90.

The slice where version 1 failed is still reported, as `diagnostic_not_gating`.
