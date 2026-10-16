# Self-learnt model switching (retrieve, then answer)

Part of STAIR / RightRoute. **No router: the models learn their own work by fine-tuning, and a rulebook that is itself learnt
from experiments decides, per question, when to escalate to a bigger model.** The cheapest retriever always runs; the rulebook
escalates retrieval and answering only when the evidence says so, ensembling what the models produced. Every answer carries a
calibrated Sure / Unsure flag, also when the largest model answered.

- **Live progress of the cluster run:** branch [`selfLearnt-runs`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt-runs)
  (status, estimated finish, results so far, charts; one commit, replaced at every update).
- **Development history** (Google Colab, notebooks 00–28, results per experiment set): branch
  [`selfLearnt-colab-dev`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt-colab-dev).
- This branch: the method and the full-scale pipeline for the SMU Origami GPU cluster (`pipeline/`).

## Five workstreams, one test set

All five are scored on **AgentTTS's own 500 HotpotQA test questions** (dev file [500:1000], each with the gold paragraphs plus
the top-100 BM25 paragraphs), with AgentTTS's metric (the gold answer contained in the final response) and its compute budget,
plus FLOPs actually spent.

| # | Workstream | Retrievers | Answerers | Switching | Purpose |
|---|---|---|---|---|---|
| 1 | **Baseline 1: AgentTTS**, as in its code | Qwen2.5 7B / 32B / 72B | Llama 3.2 3B / 3.1 8B / 3.1 70B | none: one setting (models + sample counts) found by AgentTTS's planner search on 50 questions, budget 900 | Baseline calibration |
| 2 | **Baseline 2: strongest models** | Qwen2.5 72B | Llama 3.1 70B | none: largest models on every question | Baseline calibration |
| 3 | Hidden-state probe + calibrated rules | AgentTTS's (untrained) | AgentTTS's (untrained) | rulebook on probe scores | Savings over Baseline 1 without fine-tuning? |
| 4 | **Our method** on AgentTTS's models | Qwen2.5 7B / 32B / 72B, fine-tuned | Llama 3.2 3B / 3.1 8B / 3.1 70B, fine-tuned | self-learnt rulebook | Apple-to-apple with Baselines 1–2 |
| 5 | **Our method**, full ladders | Qwen2.5 0.5B / Qwen3 0.6B / Qwen3.5 0.8B / Qwen2.5 3B / 7B / 32B / 72B, fine-tuned | Llama 3.2 3B / 3.1 8B / 3.1 70B, fine-tuned | self-learnt rulebook | More model sizes to switch between |

Llama has no size between 8B and 70B; Llama 3.2 1B can be added as a smaller answerer in `configs/default.yaml`.

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

### Our method (workstreams 4 and 5)

| Layer | What it is | How it changes |
|---|---|---|
| Models | Ladders of retrievers and answerers | Fine-tuned by backpropagation (LoRA; QLoRA for 32B / 70B / 72B on 2,000 questions, the 'lighter' plan): retrievers on HotpotQA's gold-passage labels, answerers on gold answers |
| Soft rules | `IF signal < threshold THEN escalate` (to the next size, a named size, or skipping sizes), ensemble settings | Tuned by micro-experiments on 1,000 dev questions; the planner LLM proposes new rules |
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

Workstream 3 starts from the same structure, with the hidden-state probes as its only signals.

**Micro-experiments** (dev questions only, halves A and B). Each round, code tries fine threshold steps and the planner
proposes up to 3 changes. A change is kept only if (1) on both halves it is cheaper at no accuracy loss, or more accurate at
≤ 10% extra compute per point; (2) it beats escalating the same number of random questions; (3) it breaks no hard rule.
After 4 rounds the rulebook is frozen and applied once to the test questions.

### Hard rules (locked; edit `pipeline/configs/default.yaml` by hand only)

| # | Rule | Prevents |
|---|---|---|
| H1 | Always escalate retrieval when the first retriever's confidence is in its lowest 10% | Drifting to never escalating |
| H2 | Always escalate an empty or "cannot be determined" answer | Accepting non-answers |
| H3 | At most 60% of questions reach the largest answerer, and 60% the largest retriever | Drifting to always using the biggest model |
| H4 | Dev accuracy ≥ the best fixed pipeline of no higher cost − 1 point (below it, only accuracy-raising changes are accepted) | Switching that is worse than not switching |
| H5 | "Sure" needs a calibrated score ≥ 0.85 | The flag losing its meaning |
| H6 | ≤ 8 soft rules, each deciding on ≥ 2% of dev questions | Rules fitted to a handful of cases |
| H7 | A threshold moves ≤ 0.05 per round | Chasing noise |
| H8 | An accuracy gain may cost ≤ 10% extra compute per point | An unbounded price for accuracy |

## Development results (Colab; 400 test questions, 20 candidates; not directly comparable with the cluster run)

| Pipeline | Accuracy (corrected label) | Compute vs our earlier fixed pick |
|---|---|---|
| Qwen 7B retrieves, Qwen 32B answers (scaled-down AgentTTS setting) | 76.8% | 0% |
| Fine-tuned Qwen 3B retrieves, Qwen 32B answers | 84.2% | −42% |
| Fine-tuned Qwen3.5 0.8B retrieves, Qwen 32B answers | 84.0% | −57% |
| 0.5B; when unsure → 0.6B → 0.8B; Qwen 32B answers | 83.5% | −55% |
| Starting rulebook (both stages; saved outputs) | 80.5% | −69.5% |

Fine-tuning works; retrieval switching beats random escalation (+1.4 points); answer switching needs evidence from after the
answer (checks on the answer reach AUC 0.76); the Sure flag was right 91.5% of the time.

## Running on the SMU Origami cluster

All commands from `STAIR/selfLearnt/pipeline` on the login node (`ssh <account>@origami.smu.edu.sg`).

**1. Settings.** Run `myinfo`; put your partition, account and QOS into `jobs/cluster.env`. In `configs/default.yaml` set
`paths: project_dir` to storage with **~600 GB free** (models ~500 GB, runs ~50 GB).

**2. Environment (once, ~15 min):** `bash jobs/install_env.sh`

**3. Models and data (once, login node, needs internet; several hours).** Accept the Llama licences on huggingface.co
(Llama 3.1 8B, 3.1 70B, 3.2 3B), run `huggingface-cli login`, then inside `tmux`: `bash jobs/setup_login.sh`.

**4. GitHub token for the status page.** On github.com: Settings → Developer settings → Personal access tokens → Tokens
(classic) → Generate, scope **public_repo** only. On the cluster:
`echo <token> > ~/.stair_github_token && chmod 600 ~/.stair_github_token`. The token never goes into the repository.

**5. Check on CPU first (~2 min):** `python run.py --config configs/dryrun.yaml --stage all` (fake models, tiny training).

**6. Submit everything:** `python jobs/submit_all.py` (preview with `--dry`). 72 jobs with dependencies, in priority order:

| Phase | Workstreams | Rough wall time on 2 GPUs | Results |
|---|---|---|---|
| 1 | WS1 + WS2 | ~20 h | after ~1 day |
| 2 | WS4: fine-tune 6 models, then rules | ~45–55 h | after ~3 days |
| 3 | WS5: fine-tune 4 small retrievers, then rules | ~15–20 h | after ~4 days |
| 4 | WS3: probes and rules | ~10–15 h | after ~4.5 days |

Every job resumes from its saved outputs; a job that reaches its time limit or is pre-empted requeues itself. If a job fails,
fix the cause and re-submit (`python jobs/submit_all.py --phases 2`): finished work is skipped. If jobs cannot reach GitHub,
run `bash jobs/status_loop.sh` in `tmux` on the login node.

GPUs: the jobs ask for A40 / L40S / H100 (`CONSTRAINT` in `jobs/cluster.env`; V100s cannot run these models). The 70B and
72B models use both GPUs (`tensor_parallel: 2`); on 80 GB H100s set it to 1.

## Outputs (`runs/<run_name>/`)

```
status/page/              the status page (also on branch selfLearnt-runs)
results/comparison.md     all workstreams side by side; results.csv; charts/
ws1/search_table.csv      every AgentTTS setting on the search questions;  search_result.json, search_log.txt (planner transcript)
ws1/test_results.json     WS1-2 on the test questions (per question too)
ws3|ws4|ws5/results.json  test policies, learnt rules, accepted changes; results.xlsx; per_question.jsonl
ws*/rules/                rulebooks after every round, learn_state.json (every change tried, and why kept or rejected), planner prompts
adapters/<model>/         LoRA checkpoints and train_state.json (validation curve); merged/<model>/ for the small models
step/                     every shortlist, answer, sample, hidden-state probe score (JSONL)
```

## Code

```
pipeline/
  run.py                  stage runner: python run.py --config ... --stage ... [--model | --part | --ws]
  configs/default.yaml    models, workstreams, fine-tuning, rules and guardrails; configs/dryrun.yaml for the CPU check
  jobs/                   cluster.env, install_env.sh, setup_login.sh, submit_all.py, job.sbatch, status_loop.sh
  stair/
    common.py             prompts (AgentTTS's verbatim; ours), metrics, budget formula, ensembles, BM25
    data.py               question sets (AgentTTS's search / test; train / val / dev from the training file)
    engines.py            vLLM (tensor parallel, LoRA, log-probabilities, hidden states) and the mock engine
    agenttts.py           WS1-2: settings grid, search, test runs;  agenttts_search.py: AgentTTS's agent (Apache-2.0)
    train.py              LoRA / QLoRA fine-tuning of retrievers and answerers
    states.py             WS3-5: shortlists, reachable shortlists, answers, samples
    probe.py              WS3: hidden states and probes
    rulebook.py           rules, micro-experiments, guardrails, test
    report.py, progress.py  comparison, charts, status page, GitHub push
```
