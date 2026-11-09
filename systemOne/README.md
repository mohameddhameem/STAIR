# systemOne: a System One judge decides when the small model is enough

Part of STAIR / RightRoute. **A small typed decision model (the "System One" judge) reads the question and the
retrieved passages, and sometimes the small model's answer, then decides per question whether to pay for the large
model.** The judge is [nano-jev](https://huggingface.co/sdmlai/nano-jev) (33M parameters), fine-tuned on our routing
labels; experiment 9 swaps in Amazon's [Strands Decider](https://github.com/strands-labs/strands-decider) 2B / 4B.

**Since 2026-10-21 the headline is the team's HotpotQA golden set**, Jonathan's Stage-Level Routing setup:
- the test split is `slr-test` (1,481 questions);
- answers are scored with the standard lenient ruler;
- costs are AgentTTS Table 5 prices.

See `stair_common` 0.3.0 in [common/](../common/README.md) and
[the write-up](docs/experiments/2026-10-21-slr-standard.md). The earlier settings are kept below as history.

- Results, experiment by experiment: [RESULTS.md](RESULTS.md) (full write-ups in [docs/experiments/](docs/experiments/))
- The judges published on Hugging Face: nano-jev (2026-10-25), and the Strands Decider 2B / E4B on pair A and
  menu A (2026-10-26): [MODELS.md](MODELS.md)
- What nano-jev's typed decisions add, tested on HotpotQA, MuSiQue and 2WikiMultiHopQA: [the Jev
  programme](#the-jev-novelty-programme-2026-10-23-to-25)
- Other team branches:
  - [`jonathan`](https://github.com/mohameddhameem/STAIR/tree/jonathan): Stage-Level Routing; the golden set;
  - [`nonLLM`](https://github.com/mohameddhameem/STAIR/tree/nonLLM): non-LLM stage routers, sampling budgets;
  - [`selfLearnt`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt): fine-tuned ladders and a learnt
    rulebook.

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

**On menu A** (experiment 10, the golden set), the same judge decides at two stages, weak-first like Jonathan's
routers:
- **retrieval:** do Qwen2.5-7B's selected passages hold what the answer needs, or does Qwen2.5-72B retrieve?
- **answer:** will Llama-3.2-3B's answer do, or does Llama-3.1-70B answer?

The labels are his.

## Setup

| | |
|---|---|
| **Golden set** (experiments 7–10 since 2026-10-21) | `slr-train` / `slr-val` / `slr-test` (4,814 / 1,110 / 1,481, all from the 7,405 dev questions); thresholds chosen on `slr-val`, reported once on `slr-test` and on the 1,303 questions nano-jev never saw; the standard lenient ruler; Table 5 prices per call, weak-first, strong-everything = 929; each judge also with its own price (`budget()` at its measured tokens) |
| Data (until 2026-10-17) | HotpotQA distractor setting. Router-train (12,000) and calib (2,000) are drawn from the train split, excluding nano-jev's own training questions; test is the full dev set (7,405) |
| Model pairs | A: Qwen2.5-1.5B (top 2) / 7B (all 10), the main pair; B: 1.5B / 14B; C: 3B / 7B |
| Cost | GFLOPs per question, 2 × parameters × tokens, the judge's own cost included |
| Metric | Answer F1 / EM (official normalisation); **AIQ** = mean F1 along a policy's cost–accuracy hull between always-small and always-large (RouterBench); paired bootstrap, 1,000 resamples |
| Baselines | Always small / large, random, small-model entropy, logistic regression on question / evidence / small-answer features, oracle |
| Experiment 8 | A second setting on **AgentTTS's 500 test questions**: five answer options (Llama 3.2 3B ×1 / ×10 / ×20 / ×40 fused, Llama 3.1 70B), AgentTTS cost units, an LLM grader (Qwen2.5-14B) on each model's short committed answer |

## Headline results: the golden set (`slr-test`, 1,481 questions)

| Experiment | Policy | Result | Against |
|---|---|---|---|
| 8: cheap retrieval + sampling | nano-jev top 5 → Llama 3B, 20 samples fused | **75.08 lenient at a price of 31.6** (3.4% of 929) | ties always-strong (76.77 at 929): −1.69 [−4.19, +0.68]; ties both of Jonathan's routers |
| 10: the judge on menu A | Strands Decider E4B, fine-tuned, qpa | AIQ **0.736**; 0.730 with its own price | Jonathan's routers 0.724 / 0.726: better before its price ([+0.004, +0.021], [+0.003, +0.019]), tied with it |
| | | at his operating rule (3 seeds): **76.70 at 737.5** | his label router: 76.70 at 834.8 |
| | Strands Decider 2B, fine-tuned, qpa | AIQ 0.731; 0.730 with its price | better than his label router before its price, tied with it |
| | nano-jev (33M) | AIQ 0.719 | slightly worse than his routers |
| 7 / 9: pair A (Qwen 1.5B / 7B) | one 2B decider judge, qpa, with its price (1.3 a call) | AIQ 0.677–0.682 | better than entropy (0.665) for all three seeds; ties the LR cascade router (0.676) |
| H2 | Llama 70B with Jonathan's prompt on his 72B's selections | 76.23 | his always-strong 76.77: reproduced. On nano-jev's top 5: 69.82, so retrieval is the gap |

- **The deciders gain by reading the weak answer.** Their qpa versions beat their qp versions on pair A and on menu
  A. For nano-jev the two tie.
- **From about 2B parameters, a qpa judge beats entropy.** Before its own price, it also beats Jonathan's label
  router.
- **Cheap evidence plus sampling nearly closes the gap to the strongest pipeline:**
  - a 33M ranker plus 20 fused 3B answers sits within noise of 72B retrieval + 70B answering, at about a thirtieth of
    the price;
  - the 72B's advantage is that it writes question-focused evidence, which already holds the answer for 85% of
    questions.
- **The judge's own price matters at Table 5 prices.**
  - On pair A, a 4B judge's call costs nearly half the 7B's answer. One 2B judge is the best router once priced.
  - Averaging three seeds never pays there.

## The Jev novelty programme (2026-10-23 to 25)

**What it tested:** what nano-jev's typed-decision pretraining adds to a router. It was pre-registered before any new
answers existed.
- **Datasets:** HotpotQA (the golden set) is exploratory. MuSiQue and 2WikiMultiHopQA are confirmatory.
- **Rule:** a claim is *confirmed* when it holds on both confirmatory datasets.
- **Results:** [the write-up](docs/experiments/2026-10-25-jev-novelty.md), [the
  pre-registration](docs/experiments/2026-10-24-jev-preregistration.md), [the files](results/jev/).

| Claim (pair A, δ = 0.010 AIQ) | Status | HotpotQA / MuSiQue / 2Wiki |
|---|---|---|
| Starting from nano-jev beats the same 33M MiniLM without decision training (qp) | **Confirmed** | +0.021 / +0.010 / +0.019 |
| The passages, not the question, decide escalation | **Confirmed** | +0.011 / +0.025 (MuSiQue / 2Wiki) |
| Non-inferior to entropy, the LR routers and every cheap weak-model signal, including a hidden-state probe | **Confirmed** | non-inferior; better only for qp on 2Wiki |
| One untrained typed question recovers most of a trained judge's gain | **Confirmed** | 77% / 86% / 83% |
| The advantage is largest with few labels | Fails | the gap grows with labels |
| The typed head beats a plain head | Fails | |
| A question-only judge routes at chance | Fails on both | it beats random |
| On menu A, one 4-option question beats two stage decisions | Exploratory | +0.004; the first nano-jev judge not significantly below Jonathan's 1.5B router |
| Speed | Measured | CPU 123 ms against 4.8 s for the 1.5B router; GPU only 1.8× faster |

## Earlier settings (history)

| Setting | Policy | AIQ | Note |
|---|---|---|---|
| Exp. 7, pair A, 7,405 dev | Entropy threshold | 0.662 | baseline |
| | Best logistic regression before the answer | 0.665 | |
| | **nano-jev judge, qp** | **0.673** | +0.004 to +0.011 over LR; judge costs 3% of the small model |
| Exp. 9, pair A, 7,405 dev | Strands Decider 2B, fine-tuned, qpa | **0.708** (0.702 with its cost) | +0.030 to +0.039 over nano-jev qp |
| | Strands Decider 4B, fine-tuned, qpa | 0.713 (0.700 with its cost) | |
| Exp. 8, AgentTTS's 500 | Llama 70B alone (graded) | accuracy **0.748** at 96 AgentTTS units with its commit step (31% of strong-everything's 311) | the fused 3B ×40 scores 0.632 graded, 0.746 lenient at 71 units |
| | Judge (nano-jev, five options) | ties the fixed-policy hull | |
| Common standard (2026-10-18), AgentTTS's 500 | one 2B / E4B decider, qpa, with its cost | AIQ 0.724 | beats entropy and nano-jev, ties the LR cascade router ([write-up](docs/experiments/2026-10-18-common-redo.md)) |

- **Lenient scoring misleads on long answers.** Gold-contained scoring favoured long fused answers by up to 0.14.
  An LLM grader on a committed short answer reverses the ranking (70B > fused 3B).
- **Corrected 2026-10-23:** the 2026-10-17 version of this page compared units with Table 5 prices. Its "10.3% of
  strong-everything" and "280× cheaper retrieval, 4–10× cheaper overall" are wrong; see the branch
  [README](../README.md#corrections-to-the-2026-10-17-snapshot).

## Quick start

Run everything from this folder (`systemOne/`), with [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync                                   # CPU; GPU runs: uv sync --extra llm --extra decider
uv run pytest tests                       # 117 tests
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
| `src/stair/` | The package: `run_experts` (answer with small / large models), `router` + `evaluate_routing` (baselines, AIQ, bootstrap; AgentTTS units or Table 5 prices), `judge` (nano-jev judge: zero-shot, fine-tune, eval), `decider` (Strands Decider wrapper), `exp8_*`, `pools`, `rescore` (LLM grader), `slr` (reads and verifies the golden set at a pinned commit), `exp10_slr` (the judge on menu A), `make_splits` / `subset_cache` / `export_common` (the team's splits and results layout); for the Jev programme, `prepare_multihop` (MuSiQue and 2Wiki in the HotpotQA layout), `nanojev_pretrain`, `label_curve`, `signals` (weak-model confidence baselines), `zero_label`, `risk_control`, `latency` and `confirm` (the pre-registered tests) |
| `configs/` | One TOML per run: `gpu*.toml` (experts), `system-one*.toml` (experiment 7 / 9 judges), `exp8*.toml` (experiment 8); `*-common.toml` (the 2026-10-18 common standard); `slr.toml`, `*-slr.toml`, `exp10-slr.toml` (the golden set); `e1-*`, `e2-*`, `e5-transfer`, `e7-evidence`, `gpu-2wiki`, `gpu-musique`, `gpu-pairD` (the Jev programme) |
| `scripts/` | `run_cluster.sh` (every step), `omega.pbs` / `omega-cpu.pbs` (PBS wrappers) |
| `results/` | Each run's `report.json` / `report-clean.json`, curves, configs and per-seed `done.json` (training history). Per-question caches, job logs and provenance records are not included |
| `docs/` | Method notes and one write-up per experiment |
| `tests/` | Unit tests (CPU) |

Some code comments and write-ups mention cluster job sheets (`docs/gpu-jobs*.md`), job logs and job-id lists
(`*-jobs.txt`) that are kept in the development repo; the commands they describe are the `run_cluster.sh` steps above.
