# Prioritized 500-question matched comparison

The user requested at least 500 distinct questions supporting the W/S-combination
and sampling-budget comparison. The cohort contains exactly 500 unique HotpotQA
training IDs and 500 unique normalized question texts. All have complete WW/SW
generation already available.

Selection was frozen on 2026-10-16 at 23:12 +08. It retains 344 eligible training
questions with completed Strong generation, then uses seed 20261017 to select
156 additional questions from the remaining Weak-complete training pool.
Correctness did not affect selection. This availability-based cohort is not a
uniform full-dataset sample or a held-out result. Validation/test splits stay fixed.

`gguf_cohort_controller.py` temporarily takes all four serverA GPUs from the
full Strong controller. It completes missing WS/SS outputs with unchanged models,
prompts, seeds, temperatures, forty candidates and prefix fusions 1..40. Cached
requests are reused; no new Weak generation is performed.

The frozen v2 reader then processes all four pairs for the selected questions,
requiring exactly 80,000 matched extraction records. Its profile and wrapper SHA
match the full v2 reader. Cached records are copied and filtered to the cohort;
original full-run and twelve-question caches are retained.

After every selected question/pair/budget/source hash is checked, outputs are:

- `cohort_500/table.md`: pair coverage, earliest-success bins, shared-k peaks.
- `summary.json`: exact earliest-success histograms and retrospective counts.
- `per_question.csv` and `budget_grid.csv`: per-question and all-budget metrics.
- `evaluation_v2/report.json`: matched replies and extraction diagnostics.

Pair counts overlap. Earliest strict-EM success is gold-assisted, not a true
optimal compute budget. Alias differences and reader errors can affect EM; the
grid also records F1 and extraction coverage. This remains a training reference.

On success, reader records are merged into the full v2 cache after fingerprint
and payload-conflict checks. Conflicts roll back. `cohort_500/COMPLETE` requires
the report and merge, after which owned GPU processes stop and the full Strong
controller resumes automatically. A cohort failure stops for inspection;
`RESUME_FAILED.txt` separately records restart failure after tables are complete.
Other users' compute processes are never terminated.

`gguf_status.py` reports selected/completed cohort questions and its 80,000-record
reader target. All files are under
`/data1/zmcheng/cs707_sampling40_instruct_gguf/cohort_500` on serverA.

Launch checks verified unique IDs/texts, all 500 Weak inputs, frozen hashes and
reader identity. A reference fixture checked table calculations; SQLite fixtures
checked a non-destructive merge and rollback on conflicting payloads.

Completed 2026-10-17 at 02:40 +08. All 500 questions have every pair and budget,
and all 80,000 reader records were merged. The full Strong controller resumed
on all four GPUs. Local results are in
`nonLLM/results/adaptive_sampling40_instruct_gguf/cohort_500/`.
