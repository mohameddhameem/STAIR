# Custom LLM Router

Cost-aware routing between question-answering models on HotpotQA (distractor setting). The
question under test: does an evidence-sufficiency signal from a small cross-encoder route better
than a matched question-only router?

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run prepare-hotpotqa   # writes data/hotpotqa/distractor_{train,validation}.parquet
uv run nanojev-ids        # writes data/hotpotqa/nanojev_{train,validation}_ids.txt
uv run make-splits --exclude data/hotpotqa/nanojev_train_ids.txt   # writes splits.json
uv run pytest
```

`prepare-hotpotqa --sample 500` exports a random subset per split; `--jsonl` also writes JSON
Lines. `nanojev-ids` lists the questions nano-jev v1.0 was trained or tuned on, so router data
and the clean test report can leave them out. `make-splits` keeps only `level == "hard"` train
questions by default (12k router-train, 2k calib), because every validation question is hard;
`--level any` keeps all levels.

## Pipeline

Two experts (`small`, `large`) and one decision: answer with small or escalate to large. A run
directory holds cached expert outputs, trained routers and the report.

```sh
uv sync --extra llm       # torch + transformers; not needed for configs/smoke.toml
R=runs/cpu-pilot
uv run run-experts --config configs/cpu.toml --data data/hotpotqa/distractor_train.parquet \
    --split router_train --limit 50 --run $R --name router_train
uv run run-experts --config configs/cpu.toml --data data/hotpotqa/distractor_train.parquet \
    --split calib --limit 20 --run $R --name calib
uv run run-experts --config configs/cpu.toml --data data/hotpotqa/distractor_validation.parquet \
    --limit 20 --run $R --name test
uv run train-router --run $R --tau 0.8
uv run eval-routing --run $R   # writes $R/report.json and $R/curves.csv
uv run eval-routing --run $R --exclude data/hotpotqa/nanojev_validation_ids.txt   # report-clean.json
uv run tau-sweep --run $R   # retrain and evaluate at tau 0.5-1.0: $R/tau-sweep.csv
uv run cascade --expert small=results/gpu:small ... --cascade small,large14 --out results/cascade  # multi-stage
uv run judge all --plan configs/system-one.toml   # System One router (experiment 7, GPU: docs/gpu-jobs-system-one.md)
```

| Config | Scorer | Experts | Use |
|---|---|---|---|
| `configs/smoke.toml` | lexical | heuristic stand-ins, no model | checking the pipeline |
| `configs/cpu.toml` | nano-jev | Qwen2.5-0.5B (top-2) / 1.5B (all 10) | CPU pilot, tens of questions |
| `configs/gpu.toml` | nano-jev | Qwen2.5-1.5B (top-2) / 7B (all 10) | the minimal first experiment |
| `configs/gpu-vllm.toml` | nano-jev | vLLM: Qwen2.5-1.5B (top-2) / 7B (all 10) | the same, faster; 24 GB+ GPU |
| `configs/gpu-large-top3.toml`, `-top5` | nano-jev | Qwen2.5-1.5B (top-2) / 7B (top-3 or top-5) | experiment 3, [gpu-jobs.md](docs/gpu-jobs.md) |
| `configs/gpu-large-14b.toml` | nano-jev | Qwen2.5-1.5B (top-2) / 14B (all 10) | experiment 4 |
| `configs/gpu-small-3b.toml` | nano-jev | Qwen2.5-3B (top-2) / 7B (all 10) | experiment 5 |

nano-jev is loaded from the Hub (`sdmlai/nano-jev@v1.0`) unless `scorer.path` is a local folder;
`kind = "reranker"` with an MS MARCO cross-encoder still works as a relevance-only stand-in.

Routers `question`, `question+evidence` and `evidence` (an ablation) share model, labels, data
and tuning (C picked from one grid by 5-fold CV); only the input differs. Each is trained on two
labels, `small-fails` and `large-helps` (small wrong and large right), giving six routers named
`<inputs>/<label>`. Four more, `evidence+small` and `question+evidence+small`, are cascade
routers: they also see the small expert's answer (token entropy, length, yes/no), so they decide
after it has run and pay for both experts when they escalate. `eval-routing` compares them with random, oracle and a small-model entropy
threshold on F1-vs-cost curves (AIQ, CPT 50%/80%), and picks an operating point on `calib`
(≤1% F1 below always-large). It also temperature-scales each router on `calib` and reports ECE
and reliability bins before and after, with breakdowns by bridge/comparison and yes/no vs span.
A paired bootstrap over test questions (`--bootstrap 1000`, the default) gives 95% intervals for
every AIQ, for each router's AIQ minus the entropy and random baselines', and for each operating
point's F1 minus always-large's. Cost defaults to GFLOPs, 2 × `params` × tokens,
with `params` (billions) set per expert in the config; `--cost tokens` and `--cost seconds` are
the alternatives.

`run-experts` works in stages (`--stage evidence|small|large|merge`, default all) and saves
chunks of 500 questions under `<run>/<name>.parts/`. Rerunning a command skips finished chunks.
`--reuse-from <run>` copies the finished chunks of any stage whose settings match that run.
On a GPU, run one stage per command so only one model holds GPU memory. `--limit N` takes a
seeded random sample (`--seed`). A run directory is pinned to its first config: a config that
would change outputs is refused (`batch_size`, `gpu_memory_utilization`, `device` and
`max_model_len` may change). Caches keep raw signals (all ten paragraph scores, sufficiency over
the top 2 and top 3, per-token entropies, prompt and generated token counts), and every command
appends its environment, git commit and resolved model revisions to `<run>/provenance.jsonl`.
An expert's `revision` key pins a Hub commit.

### GPU cluster over SSH

```sh
bash scripts/run_cluster.sh prepare   # data, splits, model downloads (needs internet: login node)
bash scripts/run_cluster.sh pilot     # 500 questions: F1 of both experts and a time estimate
bash scripts/run_cluster.sh full      # all caches, then routers and reports
```

Set `CONFIG` (default `configs/gpu.toml`; `configs/gpu-vllm.toml` is faster), `RUN`, `HF_HOME`
(a disk with ~20 GB free) and, on nodes without internet, `HF_HUB_OFFLINE=1`. An SSH (PuTTY)
disconnect kills foreground jobs: run inside `tmux`, or `nohup bash scripts/run_cluster.sh full
> full.log 2>&1 &`. Every step resumes when rerun.

On omega (PBS Pro, NVIDIA L40S), run `prepare` on the login node and submit the GPU steps with
`qsub -v STEP=pilot scripts/omega.pbs`; see [docs/omega-cluster.md](docs/omega-cluster.md).

## Docs

- [docs/research-report.md](docs/research-report.md): the plan and the prior work behind it.
- [docs/experiment-protocol.md](docs/experiment-protocol.md): splits, leakage rules, scoring.
- [docs/reference-projects.md](docs/reference-projects.md): nano-jev and rizzo-flow.
- [docs/omega-cluster.md](docs/omega-cluster.md): running on omega's GPU node (PBS).
- [docs/experiments.md](docs/experiments.md): headline numbers of every run against published
  HotpotQA results; per-run write-ups in [docs/experiments/](docs/experiments/).
- [docs/research-notes/](docs/research-notes/): source notes with citations. Items marked
  unverified or `[bg]` must be checked before citing.
