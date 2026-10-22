# Experiments 7, 8 and 9 on the team's common standard

Outputs of the redo described in [the write-up](../../docs/experiments/2026-10-18-common-redo.md), copied from OMEGA's
`runs/` on 2026-10-18. Commands and job order: [gpu-jobs-common.md](../../docs/gpu-jobs-common.md). Job IDs:
`common-jobs.txt`.

- **Splits** (`stair_common.splits`):
  - router_train is the common 12k HotpotQA train draw minus nano-jev's training questions;
  - calib is dev[0:500];
  - test is dev[500:1000], AgentTTS's 500.
- **Scoring:** lenient accuracy (`stair_common.score`) is the headline, with F1 and the LLM grade as checks.
- **Cost:** in AgentTTS units (`stair_common.cost`, 1 unit = 1,152 GFLOPs). Samples from one prompt pay that prompt
  once (AgentTTS Eq. 8).
- **Judge cost:** every judge pays its own cost per question. A fine-tuned judge's score is the mean over 3 seeds,
  so it pays 3 passes. Each seed alone (`#seed<k>`) pays one pass: that is one deployed judge.

## Folders

| Folder | What | Jobs |
|---|---|---|
| `common` (this one) | Experiment 7/9 expert caches for pair A (Qwen2.5-1.5B / 7B, nano-jev v1.0 retriever): `router_train`, `calib` and `test.parquet`, plus `config.toml`. Also the team-layout files below. | 42803, 42811, 42847 |
| `common-judges` | Experiments 7 and 9, lenient labels. `eval/A/report{,-clean}.json` and `curves{,-clean}.csv`; `scores-{calib,test}.parquet` (every judge's per-question score, seeds included); `judges/<judge>/seed<k>/` (scores and `done.json`: training curve, timing); `zeroshot/`; `plan.toml` | 42812–42815, 42847 |
| `common-judges-f1` | The same judges trained and evaluated on F1 labels, as a check | 42816, 42847 |
| `exp8-common` | Experiment 8, the five-option System One judge (details below) | 42818–42846, 42848, 42849 |

The team-layout files in this folder (`stair_common.schema` and `stair_common.training`, checked by `validate`):

| File | Contents |
|---|---|
| `systemOne-hotpotqa-test.csv` | One row per test question, policy and seed. Covers always-small, always-large and every fine-tuned judge (each seed) at its operating point. Holds the answer, lenient/EM/F1, and units split into retrieve, answer and router |
| `training-history.csv` | Per-epoch train loss and validation NLL of the 30 trained judges (10 judges × 3 seeds) |
| `loss-curves.png` | Those curves, with the best validation epoch marked |

## Experiment 8 (`exp8-common`)

- `answers/<split>/<cell>.parquet`: every answering cell's per-question answers, token counts and cost.
  - The cells are Llama-3.2-3B with one answer or 10/20/40 samples fused, and Llama-3.1-70B in 4-bit, on nano-jev's
    top 3/5 passages.
  - Costs are under Eq. 8. `units_prompt_per_sample` keeps the old figure (prompt counted per sample).
  - `summary.csv` holds the per-cell means.
- `scores/`: nano-jev's paragraph scores. The 100-candidate pools are left out; `pools build` rebuilds them exactly.
- `commit/` and `rescore/`: the self-committed short answers and the LLM grader's verdicts. `rescore/validation.json`
  is the grader check, which passed.
- `judge-5opt/`: the option judge on lenient labels (the standard ruler).
- `judge-5opt-graded/`: the same judge on graded labels.
- `judge-decider-{2b,e4b}-{ft,zs}/`: Strands Decider judges on the graded labels, with nano-jev for reference.
- `judge-5opt-prompt-per-sample/`: the earlier judge, trained on the costs and option order before the Eq. 8
  correction. Kept for the record only.
- Each judge folder has:
  - `eval/report{,-clean}.json`: AIQ, 95% intervals, paired differences (`aiq_diff`), per-seed differences with one
    pass's cost (`aiq_diff_per_seed`), operating points, and `baseline_options` (the two options the entropy and LR
    baselines route between);
  - `eval/curves*.csv`;
  - `<inputs>/seed<k>/{calib,test}.parquet` (option probabilities and judge tokens) and `done.json`.

## Rerun the reports on CPU

No model is loaded, so this runs on a laptop. The `--exclude` list is git-ignored data: `uv run nanojev-ids`
writes it, and the same 1,000 ids are in `results/exp8-common/nanojev_validation_ids.txt`.

```bash
mkdir -p runs && cp -r results/common results/common-judges results/common-judges-f1 runs/
uv run judge eval --plan configs/system-one-common.toml --data-dir data/hotpotqa --out runs/common-judges \
  --exclude data/hotpotqa/nanojev_validation_ids.txt
```

Experiment 8's eval also reads `pools/<split>.parquet`. Rebuild them first with `pools build`; the commands are in
`scripts/run_cluster.sh`, stage `exp8c-pools`. Then run
`uv run exp8-judge eval --plan configs/exp8-judge-common.toml`.
