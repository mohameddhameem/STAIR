# Early matched Strong reference batch

The user requested some Strong sampling/fusion results before the full Weak
stage finishes. `gguf_priority.py` temporarily replaces the full controller,
uses all four serverA GPUs for a small Strong batch, then automatically restarts
the unchanged full controller. It does not use serverB.

The batch contains twelve training questions chosen with seed 20261016 among
questions whose WW/SW results were already complete. Correctness did not affect
selection. This is a preliminary training diagnostic, not a held-out benchmark
or an AgentTTS reproduction score.

Each question gets the normal WS/SS execution: one cached W/S retrieval, forty
Strong QA candidates per pair, and fusion of every prefix 1..40. The existing
public prompts, Instruct GGUF weights, sampling defaults, temperatures, token
limits, context safeguards and deterministic request keys remain unchanged.
The same frozen 7B answer reader processes the matched WW/WS/SW/SS fusions.
Gold answers are used only for retrospective scoring, never for generation or
answer extraction. The twelve selected questions produce 960 Strong candidates,
960 Strong fusions and up to 1,920 matched answer-extraction records.

All inference goes through the existing worker and main SQLite caches, with the
same fingerprint. Outputs are reused by the later full stages. Subset workers
cannot create full-stage completion markers. The full controller's checks still
require all 7,405 questions. Existing Weak results are retained.

The supervisor verifies the frozen inference lock and matched Weak completeness
before it starts. It monitors owned processes, writes its state to both the pilot
folder and the main state file, stops its own GPU processes, and restarts the full
four-card controller after success or pilot failure. A conflicting external GPU
job prevents restart rather than being terminated. A supervisor machine crash or
SIGKILL can still require manual recovery.

Remote folder: `/data1/zmcheng/cs707_sampling40_instruct_gguf/priority_strong_12`.
`config.json` records selection. `report.json` contains all-budget matched metrics
and selected full replies. `COMPLETE` requires sampling, fusion, extraction and
report completion. `main_resumed.json` records the replacement full controller.
`FAILED.txt` and `RESUME_FAILED.txt` indicate separate failure cases.

Validate before pausing the full run:

```bash
python code/gguf_priority.py --root "$PWD" --pilot "$PWD/priority_strong_12" --validate
```

Launch the supervisor only after the owned full controller and its GPU processes
have stopped. It needs the four GPUs free. Running the supervisor does not
change the separate follow-up request for forty retrieval samples per model.

## Completed-batch evaluation notes

The initial report tried to open unused shard 3 even though the selected questions
are all in shards 0, 1 and 2. Model execution and all 1,920 matched extraction
records had completed. The report now opens only represented shards and supports
`--report` regeneration without model inference. The initial error is archived.
The full controller briefly resumed automatically, then was paused for review
following the user's request to focus on the matched questions first.

Inspection revealed that the original deterministic extractor sometimes returns
an entire answer clause, including words such as "that the club ...", as a short
answer. `gguf_pilot_reextract.py` creates an isolated evaluation_v2 directory. For
all replies longer than six whitespace-separated words it uses the existing
gold-independent grounded reader, instead of taking a deterministic answer
clause. Other reader prompts, schema and generation settings remain unchanged.
This uniform rule never uses gold answers. Gold is used only afterward for scoring.

The revised rule was developed after observing training examples. Formal test
evaluation must freeze its version first. Version 1 and version 2 reports/caches
are kept separate; the latter profile records the rule and wrapper SHA, yielding
a different extraction-cache fingerprint. Original sampling/fusion and extraction
caches are unchanged. Version 2 does not resolve aliases or modify gold labels.
Australia/Australian and Max Mosley/Max Rufus Mosley can still differ under strict
EM. The twelve-question results are preliminary, not final model accuracy.
