# Results

**In this branch:** only the report files (`json`, `csv`, `toml`, `md`). The per-question caches (`*.parquet`), job
logs (`logs/`) and provenance records (`provenance.jsonl`) that the tables below mention stay in the development repo.

Outputs of every executed run, copied from `runs/` (which git ignores) with
`scripts/export_results.sh`. The write-ups are in [docs/experiments/](../docs/experiments/) and the
headline numbers are in [docs/experiments.md](../docs/experiments.md).

| Folder | Experiment | Config |
|---|---|---|
| `gpu` | 1: 1.5B (top 2) / 7B (all 10) | `configs/gpu.toml` |
| `gpu-cascade` | 2: cascade routers, bootstrap, τ sweep on experiment 1's caches | `configs/gpu.toml` |
| `gpu-large-top3` | 3a: 7B reads the top 3 | `configs/gpu-large-top3.toml` |
| `gpu-large-top5` | 3b: 7B reads the top 5 | `configs/gpu-large-top5.toml` |
| `gpu-large-14b` | 4: 14B replaces the 7B | `configs/gpu-large-14b.toml` |
| `gpu-small-3b` | 5: 3B replaces the 1.5B | `configs/gpu-small-3b.toml` |
| `cascade` | 6: multi-stage cascades over the caches above (CPU) | `cascade` command, see `provenance.jsonl` |
| `system-one` | 7: System One judges (33M nano-jev) routing the pairs of runs 1, 4 and 5 (GPU) | `configs/system-one.toml` |
| `exp8-phase0` | 8, phase 0: nano-jev ranking of AgentTTS-style 100-candidate pools (CPU) | `pools` command, see [the plan](../docs/experiment-8-plan.md) |
| `exp8-phase2` | 8, phases 2-3: the option-choosing System One judge (3 seeds, qp / qpa) and its evaluation on the phase 1 answers (GPU, 7 min) | `configs/exp8-judge.toml` |
| `exp8-rescore` | 8, re-scoring: grader validations v1-v4 (v4 passed), the 19 graded answer files, committed short answers and the summary (GPU, ~4 h) | [job sheet](../docs/gpu-jobs-exp8-rescore.md) |
| `exp8-phase2c` | 8, re-scoring: the five-option judge on graded labels (GPU, 17 min) | `configs/exp8-judge-graded.toml` |
| `exp8-phase2b` | 8, part 2: the same judge with five options (adds 3B x20 and x40 fused) and its evaluation (GPU, 17 min) | `configs/exp8-judge.toml` |
| `exp8-phase1` | 8, phase 1: seven answering cells (Llama-3.2-3B incl. 10/20/40 samples fused, Llama-3.1-70B 4-bit) on nano-jev top-k passages; per-question answers, scores and job logs (GPU) | `configs/exp8.toml` |
| `common` | 7–9 on the team's common standard: pair A's expert caches on the common splits, the team-layout results file and loss curves; [its README](common/README.md) covers the next three folders too | `configs/system-one-common.toml` |
| `common-judges` | 7 and 9 on the common standard: nano-jev and Strands Decider judges, lenient labels | `configs/system-one-common.toml` |
| `common-judges-f1` | The same on F1 labels (check) | `configs/system-one-common-f1.toml` |
| `exp8-common` | 8 on the common standard: answers recosted under AgentTTS Eq. 8, grades, and the option judges (nano-jev, deciders) | `configs/exp8-common.toml`, `configs/exp8-judge*-common.toml` |
| `slr` | The HotpotQA golden set (Jonathan's Stage-Level Routing, Config B): verification and the menu A ladder on `slr-val` / `slr-test` at Table 5 prices (`report.json`, `paths-*.parquet`), job IDs (`slr-jobs.txt`). His files themselves are fetched by `slr fetch`, not copied | `configs/slr.toml` |
| `exp10-slr` | 10: the System One judge routing menu A against Jonathan's SLR routers (nano-jev and the 2B and E4B deciders: val/test scores, `done.json`; `eval/report{,-clean}.json`; job logs) | `configs/exp10-slr.toml` |
| `slr-pairA` | 7/9 on the golden set: pair A's caches cut from `gpu`'s dev cache to `slr-train` / `slr-val` / `slr-test` | `configs/system-one-slr.toml` |
| `slr-judges` | 7/9 on the golden set: the judges (nano-jev, MS MARCO, zero-shot, fine-tuned 2B and E4B deciders) and `eval/A/report{,-clean}.json` at Table 5 prices | `configs/system-one-slr.toml` |
| `exp8-slr` | 8 and the H2 diagnosis on the golden set: the five options' answers (`answers/<split>/`), the 70B with Jonathan's prompt on nano-jev's top 5 and on his 72B selections (`answers/test/llama70b-slrprompt-*`), the option judge (`judge-5opt/`) at Table 5 prices, nano-jev's recall, job logs | `configs/exp8-slr.toml`, `configs/exp8-judge-slr.toml` |

Each folder has:

| File | Contents |
|---|---|
| `report.json`, `report-clean.json` | All metrics on the 7,405 test questions, and on the 6,405 without nano-jev's validation questions |
| `curves.csv`, `curves-clean.csv` | F1-vs-cost curve points for every router and baseline |
| `router_train.parquet`, `calib.parquet`, `test.parquet` | Per-question caches: predictions, F1/EM, token counts, entropies, nano-jev scores |
| `config.toml`, `provenance.jsonl` | The exact config, and every command with its git commit, package versions and model revisions |
| `logs/` | The PBS job log |
| `tau-sweep.csv`, `tau-sweep.json` | `gpu-cascade` only |
| `run.log` | `cascade` only; it has no caches, config or `routers.pkl` of its own |

`system-one` has its own layout: `eval/<pair>/` (reports, curves and every judge's per-question scores for pairs
A, B and C), `judges/<judge>/seed<k>/done.json` (each training's curve, dev NLL and timing), `zeroshot/`,
`plan.toml`, `provenance.jsonl` and `logs/`. See [the write-up](../docs/experiments/2026-10-07-system-one-router.md).

`routers.pkl` is left out: it is about 8 MB, loading a shared pickle is unsafe, and the caches
rebuild it in minutes. To rerun any CPU analysis, copy a folder back into `runs/`:

```bash
cp -r results/gpu-large-top5 runs/
uv run train-router --run runs/gpu-large-top5 --tau 0.8
uv run eval-routing --run runs/gpu-large-top5
```
