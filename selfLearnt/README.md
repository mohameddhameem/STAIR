# Self-learnt model switching (retrieve, then answer)

Part of STAIR / RightRoute. **No router: the models learn their own work, and a rulebook that is itself learnt from
experiments decides when to escalate to a bigger model.** Small retrievers are fine-tuned to shortlist the right passages;
the cheapest one always runs, and the rulebook escalates retrieval and answering only when the evidence says so. Every
answer carries a calibrated Sure / Unsure flag.

The development work behind this (Google Colab, notebooks 00–28) is kept on the
[`selfLearnt-colab-dev`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt-colab-dev) branch. This folder holds the
method, its results and the full-scale pipeline.

## How it works

| Layer | What it is | How it changes |
|---|---|---|
| Models | A ladder of retrievers (Qwen2.5 0.5B → Qwen3 0.6B → Qwen3.5 0.8B → Qwen2.5 3B) and two answerers (Llama 3.1 8B, Qwen2.5 32B) | Retrievers are fine-tuned with LoRA (backpropagation) on HotpotQA's gold-passage labels |
| Soft rules | `IF signal < threshold THEN escalate` (and to which model), ensemble settings | Tuned by micro-experiments on dev questions; a planner LLM proposes new rules |
| Hard rules | Locked rules and guardrails (H1–H8 below) | Never changed by the experiments |

Per question:

1. The cheapest retriever shortlists the passages.
2. Retrieval rules may escalate to a bigger retriever (skipping rungs is allowed), and the shortlists are ensembled. A second rule may escalate once more.
3. The small answerer answers.
4. Answer rules may escalate to the big answerer: if the two agree, that answer is kept; otherwise sampled big-answerer answers join a majority vote.
5. A calibrated score gives the Sure / Unsure flag.

**The micro-experiments (dev questions only).**
- The dev questions are split into halves A and B.
- In each round, code tries fine threshold steps, and the planner (Qwen 32B) reads evidence tables and proposes up to 3 changes.
- A change is kept only if three things hold:
  - on both halves, it is cheaper at no accuracy loss, or more accurate at ≤ 10% extra compute per point;
  - it beats escalating the same number of random questions;
  - it breaks no hard rule.
- After the last round the rulebook is frozen and applied once to the test questions.

### Hard rules (locked; edit `configs/default.yaml` by hand only)

| # | Rule | Prevents |
|---|---|---|
| H1 | Always escalate retrieval when the cheapest retriever's confidence is in its lowest 10% | Drifting to never escalating |
| H2 | Always escalate an empty or "cannot be determined" answer | Accepting non-answers |
| H3 | At most 60% of questions reach the big answerer | Drifting to always using the biggest model |
| H4 | Dev accuracy ≥ the best fixed pipeline − 1 point (below it, only accuracy-raising changes are accepted) | Trading away accuracy |
| H5 | "Sure" needs a calibrated score ≥ 0.85 | The flag losing its meaning |
| H6 | ≤ 8 soft rules, each deciding on ≥ 2% of dev questions | Rules fitted to a handful of cases |
| H7 | A threshold moves ≤ 0.05 per round | Chasing noise |
| H8 | An accuracy gain may cost ≤ 10% extra compute per point | An unbounded price for accuracy |

## Development results (Colab, 400 HotpotQA test questions)

| Pipeline | Accuracy (corrected label) | Compute vs AgentTTS's pick |
|---|---|---|
| AgentTTS's pick: Qwen 7B retrieves, Qwen 32B answers | 76.8% | 0% |
| Fine-tuned Qwen 3B retrieves, Qwen 32B answers | 84.2–84.5% | −42% |
| Fine-tuned Qwen3.5 0.8B retrieves, Qwen 32B answers | 84.0% | −57% |
| Fine-tuned Qwen2.5 0.5B (2 passes) retrieves, Qwen 32B answers | 79.5% | −58% |
| 0.5B; when unsure → 0.6B → 0.8B; Qwen 32B answers (49% escalated) | 83.5% | −55% |
| Starting rulebook (both stages; saved outputs, before micro-experiments) | 80.5% | −69.5% |
| Llama-or-Qwen oracle (needs the answer key) | 86.8% | −72% |

- **Fine-tuning works.** Small retrievers trained on gold-passage labels beat AgentTTS's pick by about 7 points, at about half its compute.
- **Retrieval switching works.** The 0.5B knows when its own shortlist is bad: its confidence separates good from bad shortlists at 0.90 (0.5 = no signal). Escalating on it beats random escalation by +1.4 points (+0.6 to +2.3).
- **Answer switching needs evidence from after the answer.** Whether Llama 8B will be wrong cannot be predicted beforehand (0.54–0.59). Checks on its answer reach 0.76: are its words in the passages, its confidence, yes/no.
- **The starting rulebook beats random escalation** by +4.6 points (+2.2 to +7.2). Two of its cut-offs were chosen after seeing development test results, so read this as slightly optimistic.
- **The Sure flag is honest:** answers flagged Sure at 0.90 were right 91.5% of the time.

## Running on the cluster

**1. Environment** (once):
```
conda env create -f selfLearnt/pipeline/environment.yml
conda activate stair
```

**2. Configure.** Edit `selfLearnt/pipeline/configs/default.yaml`:
- `paths: project_dir` is the shared storage for models, data and runs.
- Set `download_models: false` if compute nodes have no internet.
- Model sizes, question counts, rules and guardrails are all in the same file.

Then edit the lines marked `EDIT` in `selfLearnt/pipeline/jobs/env.sh` (conda location, modules, a Hugging Face token for Llama, which is gated) and the partition name in the job files.

**3. Dry run (CPU, about 6 minutes)**, to check the installation. It uses fake models and a tiny training model:
```
cd selfLearnt/pipeline
python run.py --config configs/dryrun.yaml --stage all
```

**4. Full run (SLURM)**:
```
cd selfLearnt/pipeline
sbatch jobs/00_setup.sbatch               # needs internet: models + HotpotQA files + question sets
CONFIG=configs/default.yaml bash jobs/submit_all.sh
```
`submit_all.sh` chains every stage. Training runs two retrievers at a time, and the two answerers run side by side (one per A100). Every stage is resumable: if a job stops, submit it again.

| Stage | Job | GPU | Estimated time (2,000 dev + 2,000 test questions) |
|---|---|---|---|
| Models, HotpotQA, question sets | `00_setup` | – | 30–60 min (about 60 GB of models) |
| Fine-tune the 4 retrievers | `02_train` (array) | 1 each, 2 at once | about 5 h (0.5B 2 h, 0.6B 1.5 h, 0.8B 1.75 h, 3B 3.5 h) |
| Shortlists + baseline retrieval | `03_retrieve` | 1 | about 1 h |
| Small answerer / big answerer + samples | `04`, `05` (parallel) | 1 each | about 1 h / 3–4 h |
| Micro-experiments (planner) | `06_learn` | 1 | 30–45 min |
| Test + report | `07_test` | – | about 10 min |

These estimates are scaled from the Colab runs (one A100); the first cluster run will show the real figures, which are logged in `runs/<run>/stage_times.jsonl`.

**Scaling up.** To add Qwen2.5 72B or Llama 3.1 70B, uncomment their entries under `models:`. They use `tensor_parallel: 2`, i.e. both A100s. In 4-bit form (AWQ) they need 80 GB cards. Make one of them the big answerer or the planner.

## Outputs (`runs/<run_name>/`)

```
adapters/<retriever>/      LoRA checkpoints and the training curve (train_state.json)
merged/<retriever>/        the fine-tuned retriever, merged for vLLM
step/                      every shortlist, answer and sample (JSONL, with token probabilities and compute)
rules/rulebooks/           the rulebook after every round (v0 = starting rules, final)
rules/learn_state.json     every change tried, with its evidence and why it was accepted or rejected
rules/planner_round*.txt   the planner's prompt and answer, per round
results/summary.md         test results: accuracy, compute, intervals, against random escalation, Sure / Unsure
results/rulebook_results.xlsx  the same, plus the learnt rules and a per-question table (which rule decided, models run, flag)
```

## Differences from the development runs

- **Larger sets.** The defaults are 2,000 dev questions (from HotpotQA's training file, never used for training) and 2,000 test questions (from HotpotQA's dev file); development used 1,000 and 400. Numbers are therefore not directly comparable.
- **No learned route.** Notebook 27 showed that the 0.5B could not learn which answerer a question needs; the rulebook uses signals measured after each model runs.
- **Generation fix.** Hugging Face generation now sets `repetition_penalty=1.0` explicitly (Qwen2.5's default of 1.1 had distorted validation decoding in development).
- **Sampled answers are charged their full prompt each,** a conservative compute count.

## Code

```
pipeline/
  run.py                   stage runner (python run.py --config ... --stage ...)
  configs/default.yaml     full-scale settings; configs/dryrun.yaml for the CPU check
  jobs/                    SLURM job scripts, env.sh, submit_all.sh
  stair_switch/
    common.py              prompts (unchanged from development), metrics, vLLM / mock engines, voting
    data.py                question sets (gold passages + BM25 distractors)
    train.py               LoRA fine-tuning of the retrievers, checkpoint choice, merge
    generate.py            shortlists, answers, samples (resumable)
    rulebook.py            rules, micro-experiments, guardrails, test report
```
