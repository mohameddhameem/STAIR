# exp8-rescore: grader v2 validation (FAILED on the pasted-passage control)

Job 42685 (`exp8-rescore-validate`, `-l walltime=01:00:00`, exit 0, 10 min 20 s), Qwen/Qwen2.5-14B-Instruct at
revision `cf98f3b3`, code at `55ce4a2`. Gate slice: dev items 1000-1499; dumps from the calib pools.

| Field | Result | Required | |
|---|---|---|---|
| `agree_em1_correct` (n = 269) | 0.996 | >= 0.95 | pass |
| `agree_f10_incorrect` (n = 109) | 0.917 | >= 0.80 | pass |
| `dump_graded_correct` (n = 1,000) | **0.619** | <= 0.05 | **fail** |
| `written_agreement` | 0.991 | >= 0.95 | pass |
| `verdict_mass_mean` | 0.999 | >= 0.90 | pass |

`diagnostic_not_gating` (dev 500-999, dumps from test pools): 0.993 / 0.875, dumps 0.666.

`passed` is false, so `exp8-rescore` and `exp8-judge-graded` were not run. Prompt, thresholds and code unchanged.

Reading 10 rows of `validation-disagreements.csv`: the grader is right or fair on short answers (aliases, plurals,
"UDC" for "University of the District of Columbia"). Two are grader errors: "no" against a long gold string
(P = 0.986) and an identical "Benedict of Nursia" (P = 0.38). The failure is concentrated on long, passage-style
responses, which contain the gold string.

## Diagnosis and version 3 (project owner's local session, 2026-10-15)

**The failure tracks containment, not the computation.**
- 74.5% of the calib dumps contain the gold string, and the grader passed 61.9% of them. So it accepted
  about 83% of the dumps that contain the gold.
- Only 20.9% of the dumps contain the gold in their first sentence.
- On short answers the grader is good: 0.996 / 0.917.

**The project owner chose extract-then-grade (version 3).** The model first extracts the response's
committed final answer, or "none". Only that short answer is graded, which is the setting where version 2
passed.
- **Also recorded:** a first-sentence grade (length control) and the whole-response grade (version 2's
  metric, as a diagnostic).
- **The gate runs the full pipeline,** with these version 3 thresholds, fixed before any version 3 run:
  - EM = 1 → CORRECT ≥ 0.95;
  - F1 = 0 → INCORRECT ≥ 0.80;
  - dumps CORRECT ≤ 0.10 (half of the 0.209 of dumps that state the gold in their first sentence);
  - written agreement ≥ 0.95;
  - verdict mass ≥ 0.90;
  - extraction faithfulness on short answers ≥ 0.90.

**Risk to watch, from a local check with Qwen2.5-1.5B** (too weak to judge the thresholds itself).
- **The problem:** given a passage dump, an extractor can answer the question from the passages instead of
  reporting what the response committed to ("Rihanna", "Pearl Harbor").
- **The fix:** the extraction prompt now says to report only an answer the response explicitly states, and
  never to answer the question itself.
- **The effect on 30 items:**
  - faithfulness on short answers went from 0.867 to 0.900;
  - "none" on dumps went from 0.167 to 0.200;
  - 0.167 of dumps were still extracted as the gold.
- **The 14B gate decides.** If `dump_graded_correct` stays above 0.10, the extractor is acting as a reader.
