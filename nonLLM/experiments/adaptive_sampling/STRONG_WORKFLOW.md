# Full Strong-first workflow

On 2026-10-16 the user requested continuing Strong sampling and downstream
steps. `gguf_strong_controller.py` uses all four serverA GPUs and keeps Weak
generation paused. Existing generation caches and the twelve-question reference
batch are reused. No model, public prompt, temperature, candidate count or
fusion budget is changed. The retrieval-40 follow-up is still separate.

Execution order:

The active job is temporarily the [500-question priority cohort](COHORT_500.md),
requested later on 2026-10-16. It completes a matched training reference table
first, merges reusable v2 reader results, and resumes this full Strong workflow.
All existing generation caches remain valid.

1. Finish WS and SS on all 7,405 questions. Each pair gets forty Strong candidates
   and all prefix fusions 1..40. Four disjoint work queues share every logical
   shard. Completion requires 607,210 committed Strong requests.
2. Keep the same four 7B replicas loaded and run the frozen evaluation-v2 reader.
   `evaluation_v2/profile.json` is copied byte-for-byte from the validated pilot;
   the wrapper SHA and original generation profile must match. The 1,920 pilot
   reader records are copied using SQLite backup, leaving the pilot immutable.
   All Strong fusions and already available Weak fusions are extracted. Weak
   generation is not restarted. The current frozen source snapshot requires
   964,566 reader records, including 592,400 Strong fusions.
3. Release GPU replicas and analyze WS/SS with `gguf_analyze.py --pairs WS SS
   --reader-root evaluation_v2 --output analysis_strong_v2`. The existing split
   remains 4,814 train / 1,110 validation / 1,481 test questions. Fixed model pair
   and budget are selected on validation; test fixed scores, retrospective budget
   oracles, paired bootstrap intervals and cost-proxy frontiers are reported.

The reader remains gold-independent. The v2 extraction rule was revised using
training examples and is frozen before this full evaluation. Strict EM does not
resolve aliases. Retrospective oracles use gold and are not deployable policies.
The cost proxy includes reader tokens but is not measured FLOPs or wall time.
This is a Qwen 1.5B/7B adaptation of public prompts, not a certified reproduction
of the original AgentTTS models or scorer.

Only Strong completion is marked: `STRONG_COMPLETE`. It does not mark the whole
four-pair experiment complete or resume Weak generation. The original controller
and original extraction caches remain available. The v2 caches and outputs are
separate; generation DBs are read through symlinks and never modified by the
reader. Resuming the Strong workflow verifies that Weak source counts did not
change, to avoid treating a changed extraction target as the same plan.

The root `state.json`, `strong_workflow.json` and `logs/strong_controller.log`
identify the workflow. `gguf_status.py` reports the legacy extraction counts
separately from active `extraction_v2` counts. All full-source count checks must
pass before analysis or completion markers are written. The controller stops
only its own processes on failure or an external GPU conflict.

Run on serverA from `/data1/zmcheng/cs707_sampling40_instruct_gguf`:

```bash
python code/gguf_strong_controller.py --root "$PWD" --validate
python code/gguf_strong_controller.py --root "$PWD"
python code/gguf_status.py --root "$PWD"
```

Launch verification checked frozen hashes, preserved pilot fingerprints and
1,920 reusable records. A read-only preflight found exactly twelve fully scored
Strong training questions. A synthetic end-to-end scoring check verified the
80-configuration Strong-only matrix, validation selection, held-out fixed score,
oracle score and the guard against marking incomplete four-pair results complete.
