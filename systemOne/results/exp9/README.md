# Experiment 9 results: Strands Decider judges

Write-up: [docs/experiments/2026-10-17-experiment-9.md](../../docs/experiments/2026-10-17-experiment-9.md).
Copied from omega's `runs/` as in [docs/gpu-jobs-exp9.md](../../docs/gpu-jobs-exp9.md), step 5.

| Path | What |
|---|---|
| `smoke.json` | Job 42761: the wrapper against the package's engine, both models (passed) |
| `system-one/eval-decider/A/` | Experiment 7 report, pair A (`report.json`, `report-clean.json`, curves, per-question scores), from CPU job 42785 |
| `system-one/judges/*/seed*/done.json` | Experiment 7 decider fine-tunes: settings, training history, timings |
| `system-one/zeroshot/<model>/A/` | Experiment 7 zero-shot decider scores (calib, test) |
| `exp8/judge-decider-{2b,e4b}-{zs,ft}/` | Experiment 8 plans: per-seed probabilities (calib, test), `done.json`, `eval/report*.json` |
| `logs/` | Every job's log, including the two failed attempts (42768, 42773) |

nano-jev's own outputs are in `results/system-one/` and are not repeated here.
