# Self-learnt model switching (retrieve, then answer)

Part of STAIR. **No router: the models learn their own work by fine-tuning, and a rulebook that is itself learnt
from experiments decides, per question, when to escalate to a bigger model.** The cheapest retriever always runs; the rulebook
escalates retrieval and answering only when the evidence says so, ensembling what the models produced. Every answer carries a
calibrated Sure / Unsure flag, also when the largest model answered.

- **Live progress of the cluster run:** branch [`selfLearnt-runs`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt-runs)
  (status, estimated finish, results so far, charts; one commit, replaced at every update).
- **Development history** (Google Colab, notebooks 00–28, results per experiment set): branch
  [`selfLearnt-colab-dev`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt-colab-dev).
- This branch: the method and the full-scale pipeline for the SMU Origami GPU cluster (`pipeline/`).

## Six experiments (AL-01 .. AL-06), one test set, the team standard

All six follow the team standard in STAIR `common/` (`stair_common`): **test = AgentTTS's 500 HotpotQA test questions**
(dev file [500:1000], split `test`), val = dev [0:500] (split `val`: checkpoint choice, probes, rule learning), train =
12,000 questions of the official training file (split `train`); models pinned to a revision in `common/stair_common/models.toml`
(menus A and B); headline metric **lenient** (AgentTTS's ruler) plus EM and F1; cost in **AgentTTS units**
(2 · params · tokens / 1,152 GFLOPs); run seeds 0, 1, 2. Each passage set is the gold paragraphs plus the top-100 BM25 paragraphs.
The experiment sheet entries are in [`experiments/`](../experiments/).

| ID | Workstream | Retrievers | Answerers | Switching | Purpose |
|---|---|---|---|---|---|
| AL-01 | **Baseline: AgentTTS**, as in its code (ws1) | Qwen2.5 7B / 32B / 72B | Llama 3.2 3B / 3.1 8B / 3.1 70B | none: one setting found by AgentTTS's planner search on 50 val questions, budget 900 | Baseline |
| AL-02 | **Baseline: strongest models** (ws2) | Qwen2.5 72B | Llama 3.1 70B | none | Baseline |
| AL-03 | Hidden-state probe + rules (ws3) | menu A, untrained | menu A, untrained | rulebook on probe scores | Savings without fine-tuning? |
| AL-04 | **Our method**, menu A (ws4) | Qwen2.5 7B / 32B / 72B, fine-tuned | Llama 3.2 3B / 3.1 8B / 3.1 70B, fine-tuned | self-learnt rulebook | Like for like with AL-01/02 |
| AL-05 | **Our method**, full ladder from nano-jev (ws5) | nano-jev 33M, Qwen2.5 0.5B, Qwen3.5 0.8B, Qwen2.5 1.5B / 3B / 7B / 32B / 72B (all but nano-jev fine-tuned) | Llama 3.2 1B / 3B, 3.1 8B / 70B, fine-tuned | self-learnt rulebook | Do more, smaller choices cut cost? |
| AL-06 | Control: AL-05 without nano-jev (ws6) | as AL-05 from Qwen2.5 0.5B | as AL-05 | self-learnt rulebook | Guards against nano-jev passing everything and never escalating |

The planner (AgentTTS's search, rule proposals) is Qwen2.5 72B in every experiment. nano-jev (STAIR systemOne's cross-encoder,
`stair/ranker.py`) ranks passages and keeps its top 5; its sufficiency score is its confidence. It is not fine-tuned here.

### Workstream 1 follows AgentTTS's code

From [github.com/FairyFali/AgentTTS](https://github.com/FairyFali/AgentTTS) (`code_others/archon_hotpotqa.py`, `AgentTTS.py`):
the same data construction, prompts (verbatim), retrieve → Merge (reference numbers voted) → answer samples → fusion pipeline,
temperatures, budget formula and search agent (`stair/agenttts_search.py` is their agent with a local LLM). Differences, all
forced by what is public or available:

| AgentTTS | Here | Why |
|---|---|---|
| Planner o3-mini (OpenAI) | Qwen2.5 72B (local) | Your choice: no API key |
| Scores looked up in their wandb sweep CSVs | Our sweep of every setting in a grid on the same 50 search questions, looked up with their fuzzy matching | Their CSVs are not public |
| Elasticsearch BM25 | BM25 in Python over title + text | No Elasticsearch on the cluster |
| Llama 3.1 70B in 16-bit | 4-bit AWQ copy | Fits 2 GPUs |
| Metric file `evaluation_2wiki.py` | Gold contained in the response | The file is not in their repository; containment matches their reported 0.742 |

AgentTTS's reported best HotpotQA setting (Qwen 72B ×1, Llama 3B ×40, 0.742) is also run on the test questions as a check.

### Our method (AL-04, AL-05, AL-06)

| Layer | What it is | How it changes |
|---|---|---|
| Models | Ladders of retrievers and answerers | Fine-tuned by backpropagation (LoRA; QLoRA for 32B / 70B / 72B on 2,000 questions): retrievers on HotpotQA's gold-passage labels, answerers on gold answers. **Early stopping:** validation loss on val every half epoch, stop after 2 evaluations without improvement, keep the best (at most 3 epochs up to 1B, 2 above); loss curves in `results/common/history/` |
| Soft rules | `IF signal < threshold THEN escalate` (to the next size, a named size, or skipping sizes), ensemble settings | Tuned by micro-experiments on the 500 val questions; the planner LLM proposes new rules |
| Hard rules | Locked rules and guardrails H1–H8 | Never changed by the experiments |

Per question: the cheapest retriever shortlists → retrieval rules may escalate (shortlists ensembled; up to 3 retrievers) →
the cheapest answerer answers → answer rules may escalate; an answer that agrees with an earlier one is accepted; otherwise
rules may escalate again, and finally the answers plus sampled answers vote → Sure / Unsure flag.

Starting rules (soft unless locked):

| Rule | Decision |
|---|---|
| H1 (locked) | First retriever's confidence in its lowest 10% → escalate retrieval |
| R1 | First retriever unsure (lowest 20%) → escalate, skipping one size, and ensemble |
| R2 | After escalating, shortlists disagree and the bigger retriever is also unsure → escalate again |
| H2 (locked) | Empty or "cannot be determined" answer → escalate |
| A2 | Yes/no answer → keep |
| A1 | Some of the answer's words are not in the passages → escalate; agree → accept, else vote |
| A3 | Answerer's token confidence in its lowest 15% → escalate |

AL-03 starts from the same structure, with the hidden-state probes as its only signals.

**Micro-experiments** (val questions only, halves A and B). Each round, code tries fine threshold steps and the planner
proposes up to 3 changes. A change is kept only if (1) on both halves it is cheaper at no accuracy loss, or more accurate at
≤ 10% extra compute per point; (2) it beats escalating the same number of random questions; (3) it breaks no hard rule.
After 4 rounds the rulebook is frozen and applied once to the test questions.

### Hard rules (locked; edit `pipeline/configs/default.yaml` by hand only)

| # | Rule | Prevents |
|---|---|---|
| H1 | Always escalate retrieval when the first retriever's confidence is in its lowest 10% | Drifting to never escalating |
| H2 | Always escalate an empty or "cannot be determined" answer | Accepting non-answers |
| H3 | At most 60% of questions reach the largest answerer, and 60% the largest retriever | Drifting to always using the biggest model |
| H4 | Val accuracy ≥ the best fixed pipeline of no higher cost − 1 point (below it, only accuracy-raising changes are accepted) | Switching that is worse than not switching |
| H5 | "Sure" needs a calibrated score ≥ 0.85 | The flag losing its meaning |
| H6 | ≤ 8 soft rules, each deciding on ≥ 2% of val questions | Rules fitted to a handful of cases |
| H7 | A threshold moves ≤ 0.05 per round | Chasing noise |
| H8 | An accuracy gain may cost ≤ 10% extra compute per point | An unbounded price for accuracy |

## Earlier results (Colab development)

The Colab experiments that led to this design (notebooks 00–28, 400 test questions with 20 candidate passages each) are
summarised, with the headline results table, on branch
[`selfLearnt-colab-dev`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt-colab-dev/selfLearnt/colab_dev).
This branch reports only the full-scale cluster run (status and results on branch
[`selfLearnt-runs`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt-runs)).

## Running on the SMU Origami cluster

All commands from `STAIR/selfLearnt/pipeline` on the login node (`ssh <account>@origami.smu.edu.sg`).

**1. Settings.** In `jobs/cluster.env` set your partition, account and QOS, and `STAIR_PROJECT_DIR` (scratch or project
storage with **~700 GB free**, not `$HOME`). `STAIR_STATUS_PUSH=1` keeps the status page on branch `selfLearnt-runs` updated.

**2. Environment (once, ~15 min, login node):** `bash jobs/install_env.sh` (vLLM, the pipeline's packages, and `common/`).

**3. Models and data (once, login node, needs internet; several hours).** Accept the Llama licences on huggingface.co
(Llama 3.2 1B, 3.2 3B, 3.1 8B, 3.1 70B), run `huggingface-cli login`, then inside `tmux`: `bash jobs/setup_login.sh`.
It is resumable: run it again after a disconnect.

**4. GitHub token for the status page.** On github.com: Settings → Developer settings → Personal access tokens → Tokens
(classic) → Generate, scope **public_repo** only. On the cluster:
`echo <token> > ~/.stair_github_token && chmod 600 ~/.stair_github_token`. The token never goes into the repository.

**5. GPU smoke test (one job, 1 GPU, ~1–2 h):** `sbatch jobs/smoke.sbatch`. Every stage of AL-01, AL-02 and AL-05 runs on
the small models with few questions; `logs/stair-smoke-<job>.out` ends with `SMOKE TEST PASSED`.
(Without a GPU: `python run.py --config configs/dryrun.yaml --stage all` checks the code with fake models.)

**6. Submit everything:** `python jobs/submit_all.py` (preview with `--dry`; another seed with `--seed 1`). Jobs with
dependencies, in priority order:

| Phase | Experiments | Rough wall time on 2 GPUs | Results |
|---|---|---|---|
| 1 | AL-01 + AL-02 | ~20 h | after ~1 day |
| 2 | AL-04: fine-tune 6 models, then rules | ~45–55 h | after ~3 days |
| 3 | AL-05: fine-tune 5 small models, nano-jev, then rules | ~20–25 h | after ~4 days |
| 4 | AL-06: rules only (reuses AL-05's outputs) | ~3 h | after ~4 days |
| 5 | AL-03: probes and rules | ~10–15 h | after ~4.5 days |

One experiment on its own (any scheduler; each command resumes from saved outputs), from the STAIR repo root:
`python selfLearnt/pipeline/run.py --config selfLearnt/pipeline/configs/default.yaml --exp AL-04 --seed 0 --stage <stage>`, with the
stages listed in [`experiments/AL-04.md`](../experiments/AL-04.md). All experiments share one run folder, so later ones reuse
earlier shortlists, answers and fine-tuned models.

Every job resumes from its saved outputs; a job that reaches its time limit or is pre-empted requeues itself. If a job fails,
fix the cause and re-submit (`python jobs/submit_all.py --phases 2`): finished work is skipped. If jobs cannot reach GitHub,
run `bash jobs/status_loop.sh` in `tmux` on the login node.

GPUs: the jobs ask for A40 / L40S / H100 (`CONSTRAINT` in `jobs/cluster.env`; V100s cannot run these models). The 70B and
72B models use both GPUs (`tensor_parallel: 2`); on 80 GB H100s set it to 1.

## Outputs (`runs/<run_name>/`)

```
status/page/              the status page (also on branch selfLearnt-runs)
results/comparison.md     all experiments side by side; results.csv; charts/
results/common/           team tables: AL-0X-hotpotqa-test-seed<k>.csv (stair_common.schema), history/*.csv (loss curves)
ws1/search_table.csv      every AgentTTS setting on the search questions;  search_result.json, search_log.txt (planner transcript)
ws1/test_results.json     WS1-2 on the test questions (per question too)
ws3..ws6/results.json  test policies, learnt rules, accepted changes; results.xlsx; per_question.jsonl
ws*/rules/                rulebooks after every round, learn_state.json (every change tried, and why kept or rejected), planner prompts
adapters/<model>/         LoRA checkpoints, best/ (lowest validation loss), train_state.json; merged/<model>/ for the small models
step/                     every shortlist, answer, sample, hidden-state probe score (JSONL)
```

## Code

```
pipeline/
  run.py                  stage runner: python run.py --config ... --stage ... [--model | --part | --ws | --seed]
  configs/default.yaml    models, workstreams, fine-tuning, rules and guardrails; configs/dryrun.yaml for the CPU check
  jobs/                   cluster.env, install_env.sh, setup_login.sh, submit_all.py, job.sbatch, status_loop.sh
  stair/
    common.py             prompts (AgentTTS's verbatim; ours), metrics, budget formula, ensembles, BM25
    data.py               question sets from stair_common.splits (val, test, train; search = first 50 of val)
    engines.py            vLLM (tensor parallel, LoRA, log-probabilities, hidden states) and the mock engine
    agenttts.py           AL-01/02: settings grid, search, test runs;  agenttts_search.py: AgentTTS's agent (Apache-2.0)
    train.py              LoRA / QLoRA fine-tuning with early stopping; loss curves in stair_common.training layout
    ranker.py             nano-jev cross-encoder (from STAIR systemOne)
    export.py             results tables in stair_common.schema layout
    states.py             AL-03..06: shortlists, reachable shortlists, answers, samples
    probe.py              AL-03: hidden states and probes
    rulebook.py           rules, micro-experiments, guardrails, test
    report.py, progress.py  comparison, charts, status page, GitHub push
```
