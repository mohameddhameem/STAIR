# systemOne: a System One judge decides when the small model is enough

Part of STAIR / RightRoute. **A small typed decision model (the "System One" judge) reads the question and the
retrieved passages, and sometimes the small model's answer, then decides per question whether to pay for the large
model.** The judge is [nano-jev](https://huggingface.co/sdmlai/nano-jev) (33M parameters), fine-tuned on our routing
labels; experiment 9 swaps in Amazon's [Strands Decider](https://github.com/strands-labs/strands-decider) 2B / 4B.

- Results, experiment by experiment: [RESULTS.md](RESULTS.md) (full write-ups in [docs/experiments/](docs/experiments/))
- The fine-tuned judges and the Hugging Face plan: [MODELS.md](MODELS.md)
- Other team branches: [`nonLLM`](https://github.com/mohameddhameem/STAIR/tree/nonLLM) (non-LLM stage routers,
  sampling budgets), [`selfLearnt`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt) (fine-tuned ladders and
  a learnt rulebook)

## The pipeline

```
question + 10 HotpotQA passages
  └─ nano-jev ranks the passages (evidence)
       └─ judge: "will the large model help?"          33M (nano-jev) or 2B / 4B (decider)
            ├─ no  → small model answers on the top 2 passages   e.g. Qwen2.5-1.5B, ~1,050 GFLOPs
            └─ yes → large model answers                          e.g. Qwen2.5-7B, ~22,800 GFLOPs
```

Two judge inputs: **qp** (question + passages, decides before any answer) and **qpa** (adds the small model's
answer, so the small model always runs first). Labels: `large-helps` = the small answer is wrong (F1 < τ = 0.8)
and the large one is right.

## Setup

| | |
|---|---|
| Data | HotpotQA distractor setting. Router-train (12,000) and calib (2,000) are drawn from the train split, excluding nano-jev's own training questions; test is the full dev set (7,405) |
| Model pairs | A: Qwen2.5-1.5B (top 2) / 7B (all 10), the main pair; B: 1.5B / 14B; C: 3B / 7B |
| Cost | GFLOPs per question, 2 × parameters × tokens, the judge's own cost included |
| Metric | Answer F1 / EM (official normalisation); **AIQ** = mean F1 along a policy's cost–accuracy hull between always-small and always-large (RouterBench); paired bootstrap, 1,000 resamples |
| Baselines | Always small / large, random, small-model entropy, logistic regression on question / evidence / small-answer features, oracle |
| Experiment 8 | A second setting on **AgentTTS's 500 test questions**: five answer options (Llama 3.2 3B ×1 / ×10 / ×20 / ×40 fused, Llama 3.1 70B), AgentTTS cost units, an LLM grader (Qwen2.5-14B) on each model's short committed answer |

## Headline results

| Setting | Policy | AIQ | Note |
|---|---|---|---|
| Exp. 7, pair A | Entropy threshold | 0.662 | baseline |
| | Best logistic regression before the answer | 0.665 | |
| | **nano-jev judge, qp** | **0.673** | +0.004 to +0.011 over LR; judge costs 3% of the small model |
| Exp. 9, pair A | Strands Decider 2B, fine-tuned, qpa | **0.708** (0.702 with its cost) | +0.030 to +0.039 over nano-jev qp |
| | Strands Decider 4B, fine-tuned, qpa | 0.713 (0.700 with its cost) | |
| Exp. 8, AgentTTS 500 | Llama 70B alone (graded) | accuracy **0.748** at 10.3% of AgentTTS's strong-everything cost | the fused 3B ×40 scores 0.632 |
| | Judge (nano-jev, five options) | ties the fixed-policy hull (0.701) | picks the 70B for ~all questions |

- **The judge is a cheap, real gain on experiment 7**, and a bigger judge that reads the small answer gains 5×
  more. Question-only signals stay weak, as on the other branches.
- **Lenient scoring misleads.** Gold-contained scoring favoured long fused answers by up to 0.14; an LLM grader on
  a committed short answer reverses the ranking (70B > fused 3B).
- **Against AgentTTS** (500 questions): about 280× cheaper retrieval (2.9 vs 814 units) and 4–10× cheaper overall.
  Accuracy is not compared yet: AgentTTS's 74.6 is lenient; its answers need the same grader.

## Quick start

Run everything from this folder (`systemOne/`), with [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                   # CPU; add --extra decider for experiment 9
uv run pytest tests                       # 61 tests (1 needs torch)
CONFIG=configs/smoke.toml RUN=runs/smoke bash scripts/run_cluster.sh prepare   # data, splits (downloads HotpotQA)
CONFIG=configs/smoke.toml RUN=runs/smoke bash scripts/run_cluster.sh pilot     # tiny end-to-end CPU check
```

GPU runs used SMU's OMEGA cluster (PBS, one L40S per job): `qsub -v STEP=<step> scripts/omega.pbs`, where the steps
are listed at the top of [scripts/run_cluster.sh](scripts/run_cluster.sh) (for example `full`, `judge`,
`exp8-answer`, `exp9-judge7`). Every step resumes after an interruption. On another cluster, run
`bash scripts/run_cluster.sh <step>` inside your own job script.

## Repo map

| Path | What |
|---|---|
| `src/stair/` | The package: `run_experts` (answer with small / large models), `router` + `evaluate_routing` (baselines, AIQ, bootstrap), `judge` (nano-jev judge: zero-shot, fine-tune, eval), `decider` (Strands Decider wrapper), `exp8_*`, `pools`, `rescore` (LLM grader) |
| `configs/` | One TOML per run: `gpu*.toml` (experts), `system-one*.toml` (experiment 7 / 9 judges), `exp8*.toml` (experiment 8) |
| `scripts/` | `run_cluster.sh` (every step), `omega.pbs` / `omega-cpu.pbs` (PBS wrappers) |
| `results/` | Each run's `report.json` / `report-clean.json`, curves, configs and per-seed `done.json` (training history). Per-question outputs are not included |
| `docs/` | Method notes and one write-up per experiment |
| `tests/` | Unit tests (CPU) |

Some code comments mention cluster job sheets (`docs/gpu-jobs*.md`) that are kept in the private development repo;
the commands they describe are the `run_cluster.sh` steps above.
