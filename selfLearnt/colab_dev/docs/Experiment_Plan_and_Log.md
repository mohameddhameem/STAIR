# Small–Big Model Switching: Experiment Plan

Sep 28, 2026 · @Andy

Two comparable experiments on HotpotQA with Qwen2.5-3B (small) and Qwen2.5-7B (large): a fixed AgentTTS-style configuration versus a policy that switches models per question, each notebook finishing within an hour.

**Status (2 Oct):** Set 1 (baseline) is complete and frozen: AgentTTS's single configuration was not beaten by per-question switching. Set 2 tested two changes as an ablation: a corrected router (2a) and chain-of-thought answering (2b). Set 3 (diagnostics) is complete: with the right paragraphs the 3B with CoT answers as well as the 7B, so retrieval is the remaining gap, and CoT closes only a third of it. Set 2: the fixed router now discriminates but still does not beat the best fixed setup; chain-of-thought lifts the 3B by 0.08–0.18 EM but adds almost nothing to the 7B with 5 samples. See the Experiment log below.

## Goal and research question

Can a policy learnt on a dev set decide, question by question, when to use the 7B model instead of the 3B, and so match the accuracy of a fixed configuration at lower cost?

- **Script 1 — AgentTTS baseline.** Searches on dev for ONE configuration (model size per stage + sample counts) under a compute budget, then applies it unchanged to every test question. This is what the paper does.
- **Script 2 — per-question switching.** Learns on dev when a question needs the large model, then routes each test question separately. This is the new idea.

Both are scored on the same test questions, on accuracy, compute and speed. The first run (T4, 3B only) showed the retriever finds only 14% of gold paragraphs, so there should be room for the 7B to help on some questions.

## Workflow

&#91;embedded content: workflow · step 0, two scripts, one comparison\]

The GPU work happens once, in Step 0. Both scripts then read the same table, so they are fast, cheap to rerun, and compared on identical questions.

## Shared setup

Everything below is fixed for both scripts, so any difference in results comes from the policy, not the setup.

| Item | Choice | Why |
| --- | --- | --- |
| Dataset | HotpotQA dev file from the AgentTTS repo (7,405 questions) | same source as the paper |
| Dev sample | 200 questions, random (seed 42) | \~25 'needs the 7B' examples to learn from; used for all learning |
| Test sample | 200 questions, random, no overlap with dev | only used for final scoring |
| Paragraphs per question | 20: the 2 gold + 18 closest BM25 distractors (\~3k tokens) | 5× shorter than the 100 used before, still a real retrieval task |
| Small model | Qwen2.5-3B-Instruct (already in Drive) | one family, so only size differs |
| Large model | Qwen2.5-7B-Instruct (to download) | the size step being tested |
| Pipeline | retrieve → merge (vote) → answer samples → fuse | AgentTTS pipeline |
| Sample counts | retrieval 1 or 3 · answers 1 or 5 | with 2 sizes per stage = 16 configs |
| GPU | A100 (G4 if it works; L4 fallback) | FlashAttention; the T4 took 12 h |

**Fixes from the first run:**

- **Short answers.** Answer and fuse prompts end with `Answer: <short answer>`, scored with the official HotpotQA exact match (EM) and F1. The old lenient score gave 0.52 while strict EM was 0.00.
- **Fuser sees the question and draft answers only,** not all paragraphs twice. It was 65% of compute.
- **Qwen-only prompts** use the Qwen chat template; the repo's prompt wording is otherwise kept.

## Step 0: shared per-question results table

One GPU notebook runs all 16 configurations on all 400 questions once and saves a row per question × configuration. Both scripts are then cheap lookups on this table, the same way the AgentTTS repo scores trials from its CSV files.

1. **3B process:** retrieval, 3 samples per question.
2. **7B process:** retrieval, 3 samples; then answer (5 samples) + fuse on the merged paragraphs from both retrievers.
3. **3B process:** answer (5 samples) + fuse on the same merged paragraphs.
4. Smaller sample counts (1 retrieval, 1 answer) are scored on the first samples of the same runs, as before.

Each row records:

| Group | Columns |
| --- | --- |
| Result | short answer, EM, F1, retrieval F1 |
| Cost | tokens per stage, FLOPs per stage, estimated GPU seconds, peak GPU memory |
| Signals a router may use | question text features; 3B retrieval agreement across its 3 samples; 3B answer agreement across its 5 samples; "unknown"-style answers |

Output goes to `runs/env_qwen_hotpotqa/<timestamp>/`, with a `status.json` progress file and a time limit that stops cleanly and keeps what is done.

## Script 1: AgentTTS baseline (one global configuration)

Script 1 picks a single configuration on the 200 dev questions and applies it unchanged to all 200 test questions. It reports two versions:

- **Dev-optimal global config.** An exhaustive search of the 16 dev results under each budget. This is the best any single-configuration method can do, and the fair ceiling for AgentTTS.
- **AgentTTS agent.** The repo's search loop: initial trials from Insight 1 (85% of budget to retrieval), then rounds of written guidelines → new candidates → budget check → lookup of their dev score. It is capped at 8 trials, since the whole space is only 16.

| Setting | Plan |
| --- | --- |
| Agent LLM | a free local model on the GPU: Qwen3-4B in thinking mode (\~8 GB download), or Qwen2.5-7B to avoid a download |
| Budget levels | 3: cost of the cheapest config, the midpoint, cost of the most expensive (Eq. 4 units from measured token lengths) |
| Repeats | 3 seeds per budget, since the agent's sampling varies |
| Logged | every trial, its dev score and cost, the written guidelines, the agent's thinking text, agent tokens and seconds |
| Output | chosen config per budget → its test accuracy, FLOPs and seconds from the Step 0 table |

The agent's thinking is saved but never fed back, which gives the "print the reasoning" condition (B1) for free.

## Script 2: learned per-question switching

Script 2 learns on the dev table when a question needs the 7B, then chooses a configuration for each test question separately. It builds three routers, from simplest to the natural-language idea, so the NL router has honest competition.

| Router | How it decides | Learnt on dev |
| --- | --- | --- |
| **Cascade** | run the cheap 3B pass first; escalate a stage to 7B when the 3B samples disagree | the agreement thresholds |
| **Classifier** | small logistic model on question features + 3B agreement signals → one of the 4 model pairs | the weights (leave-one-out on 200 questions) |
| **Natural-language rule router** (main idea) | an LLM agent writes up to 8 plain-English routing rules; the 3B model reads the rules + the question and picks the route | the rules, refined over 5 rounds: apply to dev → score accuracy and cost → rewrite |

- **Two information settings**, reported separately. *Before running:* the router sees only the question text. *After a cheap probe:* it also sees the 3B pass's agreement signals, and the probe's cost is charged.
- **Every routing cost is charged,** including the LLM call that reads the rules, so savings are real.
- **Oracle router,** for reference only: the cheapest correct configuration per test question, found by looking at the answers. It shows how much any router could possibly gain.
- **Logged:** the rules as written each round, each test question's route and why, accuracy, FLOPs and seconds.

The NL rules are the per-question counterpart of AgentTTS's guidelines: readable, editable, and checkable against the dev evidence.

## Evaluation

All policies are scored on the same 200 test questions; success means a switching router reaches the dev-optimal global config's accuracy at clearly lower cost, or higher accuracy at the same cost.

| Policy | Role in the comparison |
| --- | --- |
| Always 3B (cheapest config) | cost floor |
| Always 7B (best-on-dev all-7B config) | accuracy reference |
| Dev-optimal global config | best fixed configuration |
| AgentTTS agent | Script 1 method |
| Cascade, classifier, NL rule router | Script 2 methods |
| Oracle router | ceiling for any router |

- **Accuracy:** HotpotQA EM (main) and F1 on the short answer.
- **Cost:** FLOPs per question (main), estimated GPU seconds per question, and routing overhead.
- **Speed:** measured seconds per question for the final picks, on 10 test questions run one at a time.
- **Picture:** accuracy vs FLOPs for every policy, with the Pareto frontier.
- **Uncertainty:** paired bootstrap 95% intervals on the test differences. With 200 questions, EM gaps under about 0.06 are likely noise.

## Time budget

Each notebook is planned at 15–40 minutes on an A100, so each fits the 1-hour limit; all three back to back come to about 1.5 hours. The first Step 0 run (40 dev / 100 test) took 20 minutes; the 200/200 figures are estimated from its measured rates, and every notebook stops cleanly at a time limit.

| Notebook | Main work | Estimate on A100 |
| --- | --- | --- |
| Step 0 · results table | install vLLM \~5 min · copy 3B + 7B from Drive \~7 · 3 model loads \~5 · retrieval \~3 · answer + fuse \~5–10 · analysis \~1 | \~30 min at 200/200 (20 min measured at 40/100) |
| Script 1 · AgentTTS | install \~5 · load agent \~2 · 9 agent runs (3 budgets × 3 seeds) batched together, \~6 rounds | 15–20 min |
| Script 2 · switching | install \~5 · cascade + classifier <1 · NL rules, 5 rounds × 2 settings \~10 · apply to test \~2 · live speed check of final picks \~5–10 | 25–30 min |

- **Rerunning is cheap.** Once Step 0 exists, Scripts 1 and 2 never touch the task models (except the final speed check), so new ideas cost minutes.
- **What would break the budget:** a slower GPU (L4: about 2× Step 0), more samples, more paragraphs, or a larger test set.

## Risks, go/no-go check and open decisions

**Go/no-go after Step 0:** compare the oracle router with the dev-optimal global config on dev. If the oracle saves little cost at equal accuracy, the 7B rarely helps where the 3B fails, and a router has nothing to learn. Then we change the setup (more paragraphs, a bigger large model) before building Script 2.

| Risk | Effect | Mitigation |
| --- | --- | --- |
| 200 dev questions is still modest | rules and classifier overfit | leave-one-out on dev; few features; max 8 rules; confirm on test |
| 200 test questions is noisy | gaps under \~0.06 EM not meaningful | paired bootstrap intervals; enlarge test only for final results |
| 20 paragraphs makes retrieval easy | 3B and 7B look alike | go/no-go check above; 50 paragraphs is the fallback |
| Offline evaluation from a table | real routing overhead missed | routing call cost charged; live speed check of final picks |
| G4 or vLLM version issues | lost session time | A100 default; install cell checks vLLM loads before running |
| Changed prompts (short answer, lighter fuser) | not identical to the repo | stated as deviations; both scripts share them |

**Open decisions:**

- [ ] Qwen2.5-7B: full precision (\~15 GB Drive, cleanest comparison) or AWQ (\~5.5 GB, as in the paper)?
- [ ] Sizes: 20 paragraphs, 200 dev, 200 test (raised from 40/100 on 28 Sep)
- [ ] Agent LLM for Scripts 1 and 2: Qwen3-4B (thinking, \~8 GB download) or reuse Qwen2.5-7B?
- [ ] GPU: A100 as default, or try G4 first?
- [ ] Delete Llama-3.2-3B from Drive (6.4 GB) now that the plan is Qwen-only?

**Defaults used in the notebooks until decided:** full-precision Qwen2.5-7B, 20 paragraphs / 200 dev / 200 test, Qwen2.5-7B as the agent (no extra download), A100, Llama kept. Each is one setting at the top of a notebook.

## Experiment log — Set 1 (baseline, frozen 29 Sep)

Set 1 is the AgentTTS method on the reduced HotpotQA setup, plus the first per-question router; its best result is a single fixed configuration, 7B retrieval + 7B answers ×5, at 0.545 test EM.

| Role | Run folder (`runs/`) | Notebook | Wall-clock (A100) |
| --- | --- | --- | --- |
| Results table: 16 setups × 400 questions | `env_qwen_hotpotqa/20260928-155333_A100` | 04 | 27 min |
| AgentTTS: one setup for all questions | `agenttts_global/20260929-094802_A100` | 05 | 14 min |
| Per-question switching, router v1 | `switching/20260929-100917_A100` | 06 | 23 min |

**Test results (200 questions, strict EM)**

| Policy | Test EM | TFLOPs per question |
| --- | --- | --- |
| All 3B, 1 sample (cheapest) | 0.250 | 22 |
| 3B retrieval + 7B answers ×5 | 0.420 | 32 |
| 7B retrieval + 7B answers ×5 (best single setup) | 0.545 | 58 |
| AgentTTS agent, largest budget (2 of 3 seeds) | 0.545 | 58 |
| Rule router v1, question text only | 0.410 | 41 |
| Rule router v1, after a 3B trial run | 0.340 | 40 |
| Oracle router (hindsight ceiling, not achievable) | 0.660 | 28 |

**Findings**

- **AgentTTS works as described.** It found the dev-optimal setup in 5 of 9 runs, trying at most 8 of 16 setups; the agent itself cost 72 calls and 34 seconds.
- **Switching did not beat fixed setups at any cost level.** The 3B is often confidently wrong: when all 5 of its answers agree it is right only 41% of the time; agreement predicts correctness weakly (AUC 0.67).
- **Router v1 did not discriminate.** It sent about 72% of every group to the 7B: 73% of questions that need it, 71% of questions the 3B solves, 71% of unsolvable ones.
- **Router v1 had design faults:** the 8-rule cap was not enforced (lists grew to 31 and 84 rules), conditions were topic-based, it chose among 7 routes, and a 3B model applied long rule lists.
- **The ceiling is 0.66 EM.** 68 of 200 test questions are solved by no setup: 33 are near-misses under strict EM (F1 ≥ 0.5, e.g. "Larry Drake" for "Larry Richard Drake"), 25 are plainly wrong; for 35 of the 68 even 7B retrieval found only one of the two gold paragraphs.
- **Chain-of-thought was not used,** matching the paper, which chooses parallel sampling + fusion over sequential reasoning. The answering step wrote about 22 tokens on average.

Also on file, not part of Set 1: a pilot at 40 dev / 100 test (`20260928-145205_A100`, 20 min) and the earlier Llama/T4 grid (`agenttts_hotpotqa/20260927-095328_T4_dev50`, 12 h, different setup).

## Experiment log — Set 2 (ablation of the changes)

Set 2 changes one thing at a time against Set 1, so each gain or loss can be attributed: first the router design (2a), then chain-of-thought answering (2b), with AgentTTS given the same CoT options as a control.

| Step | Condition | What it isolates | Results table | Notebook |
| --- | --- | --- | --- | --- |
| Set 1 | AgentTTS (one setup) + router v1, 16 setups | baseline | Set 1 table | 05, 06 |
| 2a | router v2, same 16 setups | effect of fixing the router | **Set 1 table, unchanged** | 06 (router v2) |
| 2b-data | add CoT answers for the same 400 questions and the same retrievals | — | Set 1 table + 8 CoT setups = 24 | 04b |
| 2b-control | AgentTTS may also pick CoT setups | does CoT help without switching? | 24-setup table | 05 (CoT allowed) |
| 2b | router v2 with a CoT rung for hard questions | effect of CoT inside switching | 24-setup table | 07 |

**Router v2 (changes from v1)**

- **Hard cap of 8 rules,** enforced in code; each round's list replaces the previous one instead of being appended to.
- **3 routes** (4 with CoT): all 3B ×5 answers, 3B retrieval + 7B answers ×5, all 7B ×5, and in 2b all 7B with CoT answers ×5.
- **Only checkable conditions:** yes/no form, comparison wording, question length, digits, capitalised names, and after a 3B trial its answer and retrieval agreement and "unknown" answers. The feature values are given to the rule reader explicitly.
- **Two ways of applying the rules,** both reported: exactly, by code parsing the rules, and by the 3B model reading them (the NL router). The gap between the two measures how faithfully the 3B follows its rules.
- **Cost-weighted classifier:** each dev question is weighted by how much a wrong route costs on it. Questions every route solves, or none does, carry no weight, so they no longer push every label towards the cheapest route (in a check on the Set 1 table the unweighted version sent 195 of 200 questions to the 3B).
- **Better evidence for the rule writer:** route accuracy by feature value on dev, then individual questions, those where routes disagree listed first (up to 90).
- **Same yardsticks in every run:** each route used alone, the dev-optimal global setup over the whole table, the AgentTTS picks from notebook 05, and two hindsight oracles (over all setups and over the routes only).

**Chain-of-thought (2b)**

- The answering prompt adds a step-by-step instruction before the same short-answer line; up to 512 tokens per answer.
- Only on 1-sample retrieval from each retriever (the retrievals Set 1 already made), with 3B and 7B at 1 and 5 samples: 8 new setups.
- This departs from the paper, which uses no CoT; it is reported as an addition, not as part of AgentTTS.
- The CoT answers get their own Eq. 4 lengths, so the 16 Set 1 setups keep exactly their Set 1 budgets and scores. Checked offline on the real Set 1 files: rebuilding the table with the Set 2 code and adding CoT rows changes no Set 1 value (max difference 0.0); notebook 04b repeats this check and saves it as `table/set1_check.json`.
- The 2b control (notebook 05, `ALLOW_COT = True`) uses the Set 1 budgets (54.5 / 85.1 / 129.4 Eq. 4 units), so it differs from Set 1 only in the CoT option.

**How to run Set 2** (A100; close any open Colab tab of a notebook before running, or its autosave overwrites the new version):

1. `06_script2_switching.ipynb` as it is (`ROUTER_VERSION = 2`) → Set 2a, `runs/set2a_switching/`, \~25 min. Needs no new data.
2. `04b_add_cot_answers.ipynb` → the 24-setup table, `runs/env_qwen_hotpotqa_cot/`, \~20 min. Can run at the same time as step 1 in a second runtime.
3. `05_script1_agenttts_global.ipynb` with `ALLOW_COT = True` → Set 2b control, `runs/set2b_agenttts_cot/`, \~15 min.
4. `07_set2b_switching_cot.ipynb` → Set 2b, `runs/set2b_switching_cot/`, \~30 min. It picks up the tables and the step 3 run automatically.

Set 1 is only read, never written: 04b copies what it needs into its own run folder.

### Set 2 results (29 Sep)

No per-question router beat the best fixed setup at equal cost in Set 2 either; the one clear gain is chain-of-thought for the 3B, which is cheap in compute.

**At a glance: Set 1 and Set 2 side by side** (test, 200 questions; differences under about 4 points are noise; compute relative to the cheapest setup):

| Setup | Set | Accuracy | Compute (× cheapest) |
| --- | --- | --- | --- |
| **Fixed: same setup for every question** |  |  |  |
| All 3B (cheapest) | 1 | 25.0% | 1.0× |
| All 3B, with chain-of-thought | 2b | 37.0% | 1.1× |
| 3B finds paragraphs, 7B answers | 1 | 42.0% | 1.4× |
| 7B finds paragraphs, 3B answers once with chain-of-thought | 2b | 50.5% | 2.3× |
| **All 7B (best overall)** | 1 | **54.5%** | 2.6× |
| All 7B, with chain-of-thought | 2b | 55.0% | 3.2× |
| **AgentTTS: agent picks one setup for all questions** |  |  |  |
| AgentTTS, largest budget | 1 | 54.5% | 2.6× |
| AgentTTS, may pick chain-of-thought | 2b | 51.5% | 2.7× |
| **Switching: a different setup per question** |  |  |  |
| Router v1 (old) | 1 | 41.0% | 1.8× |
| Router v2 (fixed) | 2a | 51.5% | 2.2× |
| Router v2, may pick chain-of-thought | 2b | 52.5% | 3.0× |
| Perfect router (knows the answers, impossible) | – | 72.0% | 1.3× |

**In plain words:**

1. All-7B is still the best, at about 55%. Nothing we built beat it.
2. The fixed router is much better than the old one (41% → 51.5%), but it saves compute only by giving up accuracy: about 3 points lower for about 17% less compute.
3. Chain-of-thought helps the 3B a lot (+12 points) and the 7B almost not at all when the 7B already writes 5 answers and merges them. With 7B paragraph-finding and one answer each, the 3B with chain-of-thought beats the 7B without it (50.5% vs 47.0%).
4. There is room for switching, but we cannot use it yet: a perfect router would reach 72% at half the compute of all-7B, but we cannot tell in advance which questions need the 7B.

**By method and by model pair** (test, 200 questions). Pairs are retriever → answerer (S = Qwen2.5-3B, L = Qwen2.5-7B). For AgentTTS rows, each pair cell is the best setup of that pair, chosen on dev and scored on test. For switching rows, each pair cell is how many questions the router sent there and the share of those answered correctly.

| Method | Set / notebook; settings | Accuracy | Compute (× cheapest) | S→S | S→L | L→S | L→L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AgentTTS (original) | 1 / `05_script1_agenttts_global.ipynb`; `ALLOW_COT = False`; top budget | 54.5% (2 of 3 seeds) | 2.6× | 30.0% | 42.0% | 40.0% | 54.5% |
| AgentTTS (CoT) | 2b / `05_script1_agenttts_global.ipynb`; `ALLOW_COT = True`; top budget | 51.5% (3 of 3 seeds) | 2.7× | 37.0% | 44.0% | 50.5% | 55.0% |
| Switching (corrected) | 2a / `06_script2_switching.ipynb`; router v2, classifier, question only | 51.5% | 2.2× | 4 q · 0% | 75 q · 48% | not offered | 121 q · 55% |
| Switching (CoT) | 2b / `07_set2b_switching_cot.ipynb`; router v2, classifier, question only | 50.5% | 2.2× | 0 q | 68 q · 46% | not offered | 132 q · 53% (CoT route: 0 q) |

**Is retrieval the bottleneck?** Same answerer throughout (7B, 5 answers + fusion), split by whether retrieval returned both gold paragraphs:

| Retriever | Both gold paragraphs found | Accuracy when found | Accuracy when one or both missed |
| --- | --- | --- | --- |
| 3B | 16% of questions (32) | 75% | 36% |
| 7B | 44% of questions (88) | 66% | 46% |

Even the 7B retriever misses a gold paragraph on 56% of questions, and on those the best answerer drops about 20 points. Upgrading the answerer and upgrading the retriever help about equally without CoT (S→S to S→L +15 points; S→S to L→S +13 points). With CoT answers, the retriever is the bigger lever (+15 vs +7 points). Note the switching routers were never offered an L→S route.

The detailed tables and findings follow.

| Step | Run folder | Notebook | Wall-clock (A100) |
| --- | --- | --- | --- |
| 2a | `set2a_switching/20260929-125432_A100` | 06 | 23 min |
| 2b-data | `env_qwen_hotpotqa_cot/20260929-140148_A100` | 04b | 22 min |
| 2b-control | `set2b_agenttts_cot/20260929-142812_A100` | 05 (`ALLOW_COT = True`) | 11 min |
| 2b | `set2b_switching_cot/20260929-144958_A100` | 07 | 25 min |

All four ran on the tested code; 04b's check confirmed the 16 Set 1 setups are unchanged (6,400 of 6,400 rows identical).

**Test results, 200 questions** (±1 SE ≈ 0.035 EM; TFLOPs include probe and routing costs):

| Policy | Set | Test EM | TFLOPs/q | Live s/q |
| --- | --- | --- | --- | --- |
| Always 7B retrieval + 7B answers ×5 (L1-L5) | 1 | 0.545 | 58.4 | 4.0 |
| Always L1-L5 with CoT answers (L1-Lc5) | 2b | 0.550 | 70.6 | 5.2 |
| Router v1, classifier (question only) | 1 | 0.250 | 22.9 | — |
| Router v1, NL rules (question only) | 1 | 0.410 | 40.5 | — |
| Router v2, cost-weighted classifier (question only, λ 0) | 2a | 0.515 | 48.2 | 3.2 |
| Router v2, NL rules, applied exactly (question only) | 2a | 0.345 | 27.9 | — |
| Router v2, cost-weighted classifier with CoT route (after probe, λ 0) | 2b | 0.525 | 67.1 | — |
| AgentTTS, top budget (129) | 1 | 0.545 (2 of 3 seeds) | 58.4 | — |
| AgentTTS with CoT allowed, top budget (129) | 2b-control | 0.515 (3 of 3 seeds) | 58.8 | — |
| Oracle over all setups (hindsight) | 1 / 2b | 0.660 / 0.720 | 28.2 / 28.4 | — |

**Chain-of-thought effect** (same retrieval, same samples; paired bootstrap 95% interval, dev + test pooled, 400 questions):

| Answerer | Retrieval 3B | Retrieval 7B |
| --- | --- | --- |
| 3B ×1 | +0.113 \[+0.070, +0.155\] | +0.180 \[+0.135, +0.230\] |
| 3B ×5 | +0.083 \[+0.045, +0.122\] | +0.102 \[+0.062, +0.142\] |
| 7B ×1 | +0.018 \[−0.020, +0.055\] | +0.070 \[+0.033, +0.110\] |
| 7B ×5 | +0.033 \[+0.000, +0.065\] | +0.022 \[−0.005, +0.050\] |

**Findings**

- **2a, router fix:** the cost-weighted classifier now uses all three routes (121 / 75 / 4 questions) instead of collapsing to the 3B (v1: 0.250 EM). At λ 0 it reaches 0.515 EM at 48 TFLOPs, 17% less compute than always-L1-L5 for −0.030 EM \[−0.075, +0.015\]. That sits on the line between the fixed setups (S3-L5 and L1-L5), not above it, so it is a trade-off, not a win.
- **2a, NL rules:** the agent's rules did not beat simply always using L1-L5, even on dev (best dev EM 0.36 at 1.25× cost before running). With trial-run features it wrote 27–31 malformed lines per round and long 7-condition rules. The 3B followed short rule lists on 85–90% of questions but long ones on only 24–79%.
- **2b-data, CoT:** CoT helps whenever a single sample or the 3B answers (+0.08 to +0.18 EM), and does almost nothing for the 7B with 5 samples plus fusion (+0.022, interval includes 0). Sampling and fusion already capture most of what CoT adds for the 7B.
- **CoT versus model size:** with 7B retrieval, the 3B with CoT beats the 7B without it on one sample (L1-Sc1 0.505 vs L1-L1 0.470 on test; pooled +0.060 \[+0.015, +0.105\]) at 7% less compute. With 5 samples the 7B still leads slightly (−0.013, interval includes 0).
- **CoT is cheap in FLOPs, not in time:** answers grow from about 20 to 171 output tokens (3B) and 24 to 98 (7B), but the 2,748-token retrieval prompt dominates FLOPs (S1-S1 22.2 → S1-Sc1 24.8 TFLOPs). Live time rises more: 4.0 → 5.2 s per question for L1-L5 → L1-Lc5.
- **Unsolved questions:** CoT solved 12 test questions no plain setup solved; unsolved fell from 68 to 56 of 200, and the oracle ceiling rose from 0.660 to 0.720. The best single setup stayed at 0.55, so the extra headroom is spread thinly across setups.
- **2b-control, AgentTTS with CoT:** the agent found the dev-optimal setup in 9 of 9 runs (Set 1: 5 of 9). With CoT it gained at the low budget (54: 0.370 vs 0.270–0.300 test EM) and middle budget (85: 0.440 vs 0.270–0.420). It lost at the top budget (129: L1-Lc1 0.515 vs L1-L5 0.545), a choice made on a 0.030 dev-EM gap (6 questions), within noise.
- **2b, CoT as a router rung:** the routers barely used L1-Lc5 (the hindsight oracle over the routes used it for 7 questions). The best policy was simply always L1-Lc5 (0.550), +0.005 EM over L1-L5 for 21% more compute and 30% more live time.
- **Why switching keeps failing:** the retriever sets the ceiling (paragraph F1 0.50 for the 3B vs 0.71 for the 7B; every top setup uses 7B retrieval), and neither question text nor the 3B trial run predicts which questions need the 7B well enough to route them.

## Experiment log — Set 3 (diagnostics: answering versus retrieval)

Set 3 separates the two reasons small-model pipelines fail: a weak answerer, or incomplete paragraphs from retrieval. It uses the same 400 questions and the 24-setup table from notebook 04b as baselines; nothing in Sets 1–2 is changed.

**Why (from the Set 2 analysis, dev + test pooled):** the 3B retriever hands over both needed paragraphs for 15% of questions (7B: 45%) and none for 21% (7B: 4%). With no gold paragraph found, even the 7B answers only 20% correctly. Given both gold paragraphs, the plain 3B answers 40%, the 3B with CoT 57% and the 7B 67%. So S→S is weak for both reasons, and the two could not be separated cleanly with the existing runs.

| Step | Question | What changes | Notebook | Run folder | Time (A100) |
| --- | --- | --- | --- | --- | --- |
| 3a | How well can each answerer answer if retrieval is perfect? | answerers get only the two gold paragraphs; 3B, 3B + CoT, 7B, 7B + CoT; 5 samples + fusion | `08_set3_gold_answering.ipynb` | `runs/set3_gold_answering/` | \~25–30 min |
| 3b | Can CoT fix the 3B's retrieval? | the 3B retrieves with CoT (names the paragraph for each part of the question, then a final `Selected references:` line, which alone counts); the same four answerers answer on its picks | `09_set3_retrieval_cot.ipynb` | `runs/set3_retrieval_cot/` | \~35–40 min |

**Outputs:** each run writes `diagnose/summary.md` with three tables:

1. Retrieval quality: both / one / no gold paragraphs found, paragraph F1, output tokens, compute, for 1 and 3 merged samples.
2. Accuracy grid: retrieval source (3B, 3B + CoT, 7B, gold) × answerer (3B, 3B + CoT, 7B, 7B + CoT; 1 and 5 samples).
3. Paired differences with 95% intervals: does CoT retrieval help, does 3B + CoT retrieval match 7B retrieval, how much perfect retrieval adds, and the answerer gaps on gold paragraphs.

Run 08 first: 09 then adds the gold row to its tables. Option in 09: `RUN_7B_COT_RETRIEVAL = True` adds the 7B with CoT retrieval as a reference (\~+15 min). CoT retrieval is a Set 3 addition, not part of the paper.

### Set 3 results (2 Oct)

CoT fixes the 3B's answering but not its retrieval: given the right paragraphs, the 3B with CoT answers as well as the 7B, while CoT retrieval closes only about a third of the gap to the 7B retriever.

| Step | Run folder | Notebook | Wall-clock (A100) |
| --- | --- | --- | --- |
| 3a | `set3_gold_answering/20260930-125338_A100` | 08 | 28 min |
| 3b | `set3_retrieval_cot/20260930-224807_A100` | 09 | 33 min |

**Retrieval quality** (400 questions, dev + test; 1 retrieval sample unless marked):

| Retriever | Both gold paragraphs found | One found | None found | Compute (TFLOPs/q) |
| --- | --- | --- | --- | --- |
| 3B | 15% | 64% | 21% | 17.6 |
| 3B + CoT | 24% | 59% | 17% | 18.3 |
| 3B + CoT, 3 samples merged | 29% | 58% | 14% | 20.6 |
| 7B | 45% | 51% | 4% | 44.8 |

**Accuracy by paragraph source and answerer** (EM, 400 questions, 5 answers + fusion):

| Paragraphs from | 3B | 3B + CoT | 7B | 7B + CoT |
| --- | --- | --- | --- | --- |
| 3B retriever | 27.5% | 35.8% | 39.5% | 42.8% |
| 3B + CoT retriever | 31.5% | 38.5% | 45.5% | 45.0% |
| 7B retriever | 38.5% | 48.8% | 50.0% | 52.2% |
| Gold paragraphs (perfect retrieval) | 49.0% | 59.8% | 61.0% | 63.2% |

**Findings**

- **CoT helps 3B retrieval only partly.** Both gold paragraphs found rises from 15% to 24% (29% with 3 merged samples) against 45% for the 7B. End to end this adds +2 to +6 points (significant only with the plain 7B answerer, +6.0 \[+1.5, +10.7\]) and stays 4.5 to 10 points below 7B retrieval. It costs almost nothing extra (17.6 → 18.3 TFLOPs); the final `Selected references:` line was present in 98% of outputs.
- **With perfect retrieval, the 3B with CoT answers as well as the 7B:** 59.8% vs 61.0%, difference 1.2 points \[−2.7, +5.2\]. The plain 3B is 12 points behind (49.0%). Answering is no longer where the small model needs the large one.
- **Perfect retrieval adds 11 points for every answerer** over the 7B retriever (each interval about \[+7, +15\]). The 7B retriever adds another 10 to 13 points over the 3B retriever.
- **Even perfect retrieval tops out at 61–63% under strict exact match.** About 20% of questions are near-misses (wrong, but F1 ≥ 0.5: extra or missing words such as “Argentine Mesopotamia” vs “Mesopotamia”), and the correct answer appears somewhere in the output for 83% of questions. Strict scoring, not only reasoning, caps the numbers.
- **Retrieval is also where the compute goes.** Reading 20 paragraphs costs 17.6 TFLOPs (3B) or 44.8 (7B); answering from the selected paragraphs costs about 5 to 25. Switching the answerer saves little compute; the retriever choice decides both cost and accuracy.
- **Best value found:** 3B + CoT retrieval with 7B answers scores 45.5% at 31.8 TFLOPs, +6 points over plain 3B retrieval at the same cost. 7B retrieval with 3B + CoT answers (48.8%) matches all-7B (50.0%) within noise.
- **Implication for switching:** route the retrieval step, not the answering step. Answer with 3B + CoT by default and decide per question whether the 3B + CoT retriever is enough or the 7B retriever is needed.

## Experiment log — Set 4 (broad sweep, three sizes per role)

Set 4 is a broad sweep of retriever–answerer pairings, closer to the paper's models but without its largest tier, to find promising candidates for later, narrower runs. Run folder name: `Set_1__3-modality-size` (the name given in the request of 3 Oct; called Set 4 here because Set 1 is already used above).

**Models** (the paper's large tier, Qwen2.5-72B and Llama-3.1-70B, does not fit a 40 GB A100 even at 4-bit):

| Role | Small | Medium | Large |
| --- | --- | --- | --- |
| Retriever | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-32B (4-bit AWQ) |
| Answerer | Llama-3.2-3B | Llama-3.1-8B | Qwen2.5-32B (4-bit AWQ) |
| Planner (AgentTTS agent and rule writer) |  |  | Qwen2.5-32B (4-bit AWQ) |

**Variables** (3 retrievers × 3 answerers × 2 × 3 × 3 = 162 setups at 3 retrieval tries and 5 answer tries + fusion; 1-try versions also recorded):

| Variable | Levels | How it is done |
| --- | --- | --- |
| CoT | off / on | one switch for both steps: retriever and answerer both reason first, or neither does |
| Per-attempt shortlist | paper default, 3, 5 | paper default = the paper's prompt, no number; otherwise “Select exactly N references” is added |
| Final shortlist | paper default, 2, all | paper default = vote merge; 2 = the two most-voted passages; all = every passage any try picked (union merge) |

**Decisions taken on 3 Oct:**

- CoT is switched for both steps together (162 setups, not 324).
- The sweep uses 200 questions: the first 100 dev and 100 test of the Set 1–3 sample. Differences under about 5 points pooled, or 10 points on test alone, are noise; promising setups are to be rerun on all 400.
- 20 candidate passages per question, as in Sets 1–3 (the paper uses about 100).
- Small and medium models at full precision (official files); only the Qwen 32B is 4-bit. The paper runs every model at 4-bit.
- 3 retrieval tries and 5 answer tries; the agent's search space per condition is 3 retrievers × {1, 3} tries × 3 answerers × {1, 5} tries = 36 configurations (the paper allows 1 to 100 tries).

**What is run:**

| Notebook | Does | Time (A100) |
| --- | --- | --- |
| `10_set4_setup_models.ipynb` | downloads Qwen2.5-32B-AWQ (\~19 GB) and Llama-3.1-8B (\~16 GB, gated: licence + `HF_TOKEN`) | CPU, \~15 min |
| `11_set4_sweep.ipynb` | retrieval (3 retrievers × 6 prompt variants × 3 tries), then answering (3 answerers × CoT off/on) on every distinct final shortlist; resumable, run repeatedly until it prints `SWEEP COMPLETE` | estimated 5–8 GPU hours in total, in one-hour sessions |
| `12_set4_agenttts_switching.ipynb` | AgentTTS baseline for each of the 18 conditions (3 budgets), and per-question switching for each condition and across all conditions; prints the routing rules the planner wrote | \~30–45 min |

**Metrics per setup** (sheet “Main table” of `results/set4_results.xlsx`): overall accuracy, compute (× cheapest), retrieval found both, accuracy when found, accuracy when not found, passages in the final shortlist, dev and test accuracy. Raw data: `results/per_question.jsonl` (one row per question × setup), `step/*.jsonl` (model outputs). Charts in `results/charts/`: accuracy vs compute, retrieval success by variant, accuracy heatmap (pair × condition), effect of each variable.

**Switching in this set:** routes are the pairings best for their cost on dev (up to 4 per condition, 5 across all conditions). Routers: a cost-weighted classifier, and rules written by the planner and applied exactly by code. Settings: question only; and after the small pipeline (adds agreement between the small retriever's tries and between the small answerer's answers, and charges for that run). Variable shortlist length is deferred (see Ideas to test later).

**Confidence tracking (added 4 Oct, before the sweep started).** Every decision carries a confidence, obtained two ways so that what a model says can be compared with how it behaves:

### Set 4 run log and session folders (added 4 Oct 2026)

**Incident: the first sweep session ran the old code.** The confidence version of `code/sweep.py` had not reached Drive when the first session started (my upload reported success, but Drive still held the earlier file). So run folder `runs/Set_1__3-modality-size/20261004-113202_A100` was made with the prompts *before* confidence tracking. It is kept untouched as a partial run without confidence data, and it is not used for the Set 4 results.

| Item | Pre-confidence partial run `20261004-113202_A100` |
| --- | --- |
| Session | 1 session, 53 min, `ONLY_MODELS = ["q3", "q7", "l3", "l8"]` |
| Finished | Qwen 3B and 7B retrieval (1200/1200 each); Llama 3B answers without CoT (1535/1535); Llama 3B answers with CoT (1308/2626) |
| Setups with results | 36 of 162 |
| Stage times (s) | vLLM install 305; Qwen 3B retrieval 726; Qwen 7B retrieval 1072; Llama 3B answers 307 (no CoT) and 372 (CoT, partial) |
| Executed notebook | `notebooks/archive/11_set4_sweep_executed_20261004-1132_old-prompts.ipynb` |

**Fixes made after the incident.**

| Fix | What it does |
| --- | --- |
| Upload check | Every file sent to Drive is read back and compared byte for byte. |
| Version guard in `sweep.py` | Each run folder records the prompt version in `code_version.txt`; retrieval and answering refuse to add to a folder made with another version. |
| Notebook run selection | Notebooks 11 and 12 skip run folders made with older prompts and start (or continue) a compatible one. |
| Notebook 10 | Checks every weight shard of each model and flushes Drive before ending. |

**Session folders (which session produced what).** All sessions write into one shared run folder, because every answerer needs every retriever's shortlists and the tables combine all setups. Inside it, each session has its own labelled folder.

| Notebook | `ONLY_MODELS` | Session folder inside the run folder |
| --- | --- | --- |
| `11a_set4_sweep_q3-q7-l3-l8.ipynb` | `["q3", "q7", "l3", "l8"]` | `sessions/<time>_q3-q7-l3-l8/` |
| `11b_set4_sweep_q32.ipynb` | `["q32"]` | `sessions/<time>_q32/` |
| `11_set4_sweep.ipynb` | `[]` (all models) | `sessions/<time>_all/` |
| `12_set4_agenttts_switching.ipynb` | – | `sessions/<time>_planner/` |

Each session folder holds that session's logs, `stage_times.json`, `session.json` (models, stages run, seconds, prompt version), a `README.md`, and `results_snapshot/` (the tables and charts as they stood when the session ended; these combine everything finished so far, not only that session's models). The raw outputs in `step/` are named by model (`retrieval_q32_…`, `answers_l8_…`). Notebooks 11a and 11b are copies of notebook 11 with `ONLY_MODELS` preset, so two Colab tabs do not overwrite each other's saved notebook.

## Set 5 plan (proposed, 6 Oct 2026): can a router that reads the question beat a fixed setup?

Status: proposal for review. Nothing in this section has been run except where marked "preliminary".

### Why this set

Set 4 finished on 5 Oct (run folder `runs/Set_1__3-modality-size/20261004-165402_A100`, 162 setups, 200 questions). Per-question switching did not beat one well-chosen setup. A diagnosis on the saved results gave these reasons.

| Finding from Set 4 | Evidence |
| --- | --- |
| The big setup already answers almost everything any setup answers | All-32B right on 64% of the 200 questions; a cheaper route is right where it is wrong on 8%; no route is right on 28% |
| Much of that 8% is run-to-run luck | Six near-identical all-32B setups score 63–65% each, but "best of the six per question" scores 70% |
| So switching can only save compute, not add accuracy | Best fixed (Qwen 7B – Qwen 32B, CoT, keep all): 68% test at 5.8x. Planner's rules: 60% at 6.7x. Classifier: 31% at 1.3x |
| The routers never read the question | Classifier and planner both saw only 5 facts I chose (yes/no, comparison, word count, digit, number of names). The AgentTTS agent does not read questions either |
| The only useful "unsure" signal so far is disagreement between the answerer's 5 answers | Separation 0.70–0.78; stated confidence 0.63–0.69; token probability and all retrieval confidences near 0.5 |
| Retrieval is most of the cost | About 3.3x of the 5.8x in Qwen 7B – Qwen 32B |

Preliminary (computed offline on Set 4 data, thresholds not yet cross-validated): Llama 8B answers first and Qwen 32B answers only if fewer than 4 of 5 answers agree, sharing one Qwen 7B retrieval, gives 67% test at 5.1x, against 68% at 5.8x for always-32B.

### Question and success test

**Question.** If the router may read the question, or reuse what the cheap pass already produced, can per-question switching match the best fixed setup's accuracy at clearly lower compute?

| Outcome | Rule, fixed before running |
| --- | --- |
| Success | On the fresh questions of step 5c, a policy is within 2 points of the best fixed setup's accuracy at 15% or more less compute, router's own reading cost included |
| Partial | It sits on or above the accuracy–compute line of the fixed setups, but by less than that |
| Failure | It sits below that line. Reported as a negative result with the reason |

Policies and thresholds are chosen on the Set 4 questions only, then frozen and scored once on fresh questions.

### Steps

| Step | Notebook | What it tests | Needs | My time estimate |
| --- | --- | --- | --- | --- |
| 5a | 13 | Offline policies on the saved Set 4 results | CPU, no models except a small embedding model | minutes |
| 5b | 14 | Planner and small models read the questions | A100, one session | 30–45 min |
| 5c | 15 | Fresh 200 questions for the short list of setups, then the frozen policies scored on them | A100, two or three sessions | 2–3 h |

The time estimates are estimates from Set 4 timings, not measurements.

### Step 5a: offline, on Set 4 data

| Policy family | What decides the route | Extra cost of deciding |
| --- | --- | --- |
| A1. Answerer cascade | Cheap answerer first; big answerer only if its 5 answers disagree (one threshold) | none; retrieval shared |
| A2. Retriever cascade | Small retriever first; bigger retriever only if its 3 tries disagree (the deferred idea of acting on retriever disagreement) | none |
| A3. Both cascades together | A2 then A1 | none |
| A4. Fewer tries for the big model | Same as A1–A3 with 1 answer try for the 32B, as AgentTTS chose | none |
| A5. Learned router on the full question text | Text embedding of the question, logistic regression | under 0.001x |
| A6. Learned router on the 5 facts (Set 4 baseline) | as in Set 4 | none |
| A7. Rules with smoothed statistics | Set 4 rules, each rule's accuracy shrunk towards the average with an m-estimate | none |

Also in 5a: a learning curve (router trained on 50, 100, 150 questions) to see whether more data is worth collecting, and repeated cross-validation over all 200 questions in place of the single 100/100 split.

### Step 5b: models read the question (GPU)

| Variant | Who reads the question | What it produces | Reading cost per question (estimate) |
| --- | --- | --- | --- |
| B1. Planner proposes facts | Planner (Qwen 32B) reads dev questions with their outcomes, once | New yes/no facts about a question; a small model then answers them for each question; rules are written over them and applied by code | about 0.05x (Qwen 3B) |
| B2. Planner writes rules from the question text | Planner reads dev questions, once | NL rules in free wording; a small model applies them to each question; how often it follows them is recorded | about 0.05x (3B) or 0.1x (7B) |
| B3. Direct difficulty judgement | Qwen 3B, 7B or 32B reads each question | "easy / hard" with a confidence, with and without dev examples in the prompt | 0.05x, 0.1x, 0.5x |
| B4. A model decides at each step from all three inputs (added 6 Oct at the user's request) | A small model (Qwen 3B or 7B) reads the question, the metadata available at that point, and the planner's rules | A decision at up to three points: which retriever to start with; accept the shortlist or retrieve again, and which answerer; accept the answer or escalate | about 0.05x (3B) or 0.1x (7B) per decision, up to three decisions |

Every variant is scored with its reading cost added to the compute.

**B4 in detail (added 6 Oct 2026).** This is the variant where the deciding model weighs the question, the metadata and the NL rules together, each time a choice is made. The other variants each use only one or two of the three.

| Variant | Question | Metadata (agreement, confidence, facts) | NL rules | Who decides |
| --- | --- | --- | --- | --- |
| A1–A3 cascades | no | yes | no | a threshold in code |
| A5 learned router on text | yes | no | no | a small trained model |
| B1 planner proposes facts | indirectly | yes | yes | code applies the rules |
| B2 rules applied by a small model | yes | no | yes | a model |
| B3 direct difficulty judgement | yes | no | no | a model |
| B4 model decides at each step | yes | yes | yes | a model |

| B4 decision point | What the deciding model sees | What it decides |
| --- | --- | --- |
| Before retrieval | The question and the rules | Which retriever to start with |
| After retrieval | The question, the rules, and the retrieval metadata (how the tries agree, how many passages were picked) | Accept the shortlist or retrieve again with a bigger model; which answerer to use |
| After answering | The question, the rules, and the answer metadata (agreement of the 5 answers, stated confidence) | Accept the answer or escalate to the bigger answerer |

In B4 the rules are guidance, not law: the planner writes them from the dev questions, and the deciding model reads them with the question and the metadata. Recorded for B4: every decision with its inputs, the model's stated reason and confidence, whether it followed the rules, and its cost. The rules are kept short, because in Set 1 small models followed long rule lists poorly (24–79%). B4 is compared with the plain cascades on the same metadata, which shows whether a model's judgement beats a simple threshold. A Qwen 32B decider is tested only if the 3B and 7B show a gain, since three 32B decisions cost about 1.5x per question.

**Planner size as a variable (added 6 Oct 2026 at the user's request).** Set 4 used only the Qwen 32B as planner. It is possible that a small or medium planner plans as well as the largest. Every planning job in step 5b is therefore run once per planner, one planner at a time.

| Planner | Size | Planning jobs it runs |
| --- | --- | --- |
| Qwen 3B | small | Set 4 planning repeated (AgentTTS search; rules over the 5 facts); B1 proposing facts; B2 writing rules from question text; the rules used in B4 |
| Qwen 7B | medium | the same |
| Qwen 32B | large | B1, B2 and B4 rules; its Set 4 planning is already saved and is reused |

| Recorded per planner | Why |
| --- | --- |
| Share of rule lines that were valid (not dropped by the parser) | Can the planner write usable rules at all |
| Dev and test accuracy and compute of its rules | Does planner size change the result |
| AgentTTS: found the dev-best setup or not, and the test accuracy of its choice | The paper's method with a smaller agent |
| The planner's own confidence and how well it matches outcomes | Is a small planner more overconfident |
| Planner cost: calls, tokens, seconds, and compute in the same units as the pipeline | A large planner is only worth it if it pays for itself |
| The rules and proposed facts themselves, as text | For reading side by side |

Outputs are kept per planner in `plan_q3/`, `plan_q7/` and `plan_q32/` inside the run folder, and the tables compare the three side by side. The Llama 3B and 8B can be added as planners with one setting if wanted; the default is the three Qwen sizes so that only size changes, not model family.

**Build status (6 Oct 2026).** Notebook 13 (step 5a) and its module `code/routers.py` are built and were tested end to end on the real Set 4 results (about 90 seconds on CPU, with TF-IDF text features; the sentence-embedding variant runs in Colab). Selection rule used for every row, fixed setups included: on the training questions, the cheapest candidate within 2 points of the most accurate fixed setup; all scores are on held-out questions (5-fold cross-validation repeated 5 times). The test run added one comparison row that the plan above did not name: the cheapest fixed setup within 2 points of the most accurate one, because a policy has to beat that row, not only the most accurate setup, to be worth having. Results will be recorded here after the notebook is run in Colab.

### Step 5a results (notebook 13, run 6 Oct 2026)

Run folder: `runs/Set_2__question-reading-routers/20261006-171948_cpu` (CPU, 307 seconds, sentence-embedding model all-MiniLM-L6-v2 available). Computed from the Set 4 sweep `20261004-165402_A100`; no new model answers. Scores are on held-out questions: 5-fold cross-validation repeated 5 times over the 200 questions.

| Policy | Accuracy | Compute (x cheapest) | Accuracy of a random router with the same mix | Gate separation (0.5 = useless) | Compute vs the cheapest good fixed setup |
| --- | --- | --- | --- | --- | --- |
| Fixed: most accurate setup | 60.8% | 11.5x | – | – | +115% |
| Fixed: cheapest setup within 2 points of it | 61.7% | 5.4x | – | – | 0% |
| A1 answerer cascade | 60.5% | 6.3x | 56.4% | 0.77 | +17% |
| A2 retriever cascade | 62.1% | 6.7x | 61.7% | 0.73 | +25% |
| A3 retriever + answerer cascade | 58.8% | 5.7x | 55.4% | 0.72 | +6% |
| A4 cascade, big answerer with 1 try | 58.4% | 5.5x | 56.9% | 0.70 | +1% |
| A5 learned router, question text (TF-IDF) | 59.3% | 7.7x | 59.5% | 0.47 | +43% |
| A5 learned router, question text (embedding) | 59.3% | 7.7x | 60.0% | 0.50 | +44% |
| A6 learned router, 5 facts | 60.8% | 7.5x | 59.8% | 0.54 | +40% |
| A8 learned cascade, cheap run's metadata | 60.0% | 6.9x | 57.0% | 0.73 | +28% |
| A9 learned cascade, metadata + question text | 61.1% | 6.5x | 58.8% | 0.70 | +21% |

| Finding | Evidence |
| --- | --- |
| No offline policy beats a cheap fixed setup | The cheapest fixed setup within 2 points of the best (most often Qwen 7B retrieves once, Qwen 32B answers once, CoT, per-attempt shortlist 3, keep all) scores 61.7% at 5.4x. Every policy is within noise of it on accuracy and costs 1% to 44% more |
| The cascades have real skill, but it does not pay | A1, A3, A8 and A9 beat random escalation by 2 to 4 points; the answer-agreement gate separates at 0.70 to 0.77 |
| The retriever cascade has no skill | A2 scores 62.1% against 61.7% for random escalation with the same mix |
| Question wording alone carries no usable signal here | Separation 0.47 (TF-IDF), 0.50 (embedding), 0.54 (5 facts); these routers do not beat a random router |
| More training questions help the metadata gate only slightly | Separation 0.74 with 40 training questions, 0.80 with 160; text inputs stay near 0.5 at every size |
| Smoothing the Set 4 rules makes them cheaper, not better | As written: 52.1% at 6.1x (question only), 50.9% at 7.7x (after small pipeline). Pruned with an m-estimate (m = 5): 47.3% at 3.9x and 47.6% at 4.8x, close to their single fallback route (48.9% at 4.1x, 46.6% at 4.7x). Mean over 19 scopes, test questions |
| Headroom is small once luck is removed | Best route 65.0%; perfect router over 5 routes 73.5%; perfect router over 5 near-identical copies of the best route 69.5% |

Consequence for the plan: the bar for step 5b is the cheap fixed setup (61.7% at 5.4x), not the most accurate one. Word statistics and a sentence embedding found no signal in the question; step 5b tests whether a language model reading the question finds any. Frozen policies for step 5c are saved in `frozen_policies.json` in the run folder.

### Step 5d: hidden structure (added 6 Oct 2026 at the user's request)

The other steps ask whether a particular router works. This step asks a prior question: is there any structure in the data that separates easy questions from hard ones? Several strategies are tried and compared on one scoreboard, since some may work better than others.

**Part 1: structure in what already exists (CPU, notebook 14, no new model answers).**

| Spaces (what is looked at) | Cost of having it |
| --- | --- |
| Question facts (13: type, length, wh-word, digit, names, quote) | free |
| Question text (TF-IDF), question topics (NMF, 6 topics), question embedding (all-MiniLM-L6-v2) | free to under 0.001x |
| Question–passage similarity: how the 20 passages are spread around the question (TF-IDF and embedding) | free to about 0.005x |
| Metadata of the small pipeline (Qwen 3B – Llama 3B) | about 1.3x |
| Metadata of a cheap run (Qwen 7B – Llama 8B): agreement, stated confidence, token probability, shortlist size, output length, answer form | about 3.5x |
| Cross-model agreement: do Llama 3B and Llama 8B give the same answer on the same shortlist | about 3.9x |
| All signals available after the cheap run | about 3.5x |

| Strategies (how structure is read) | Output |
| --- | --- |
| Linear probe | held-out separation |
| First 2 principal components (PCA) | the same number as the PCA map shows |
| Nearest neighbours | the neighbourhoods a t-SNE map draws, as a number |
| k-means clusters (6), with m-estimate smoothing of each cluster's rate | cluster table |
| Small decision tree (depth 3) | readable rules |
| Boosted trees | non-linear ceiling; signal importance |
| Association rules (items from yes/no facts and low/high bands) | rules with support, confidence, smoothed confidence and lift |

| Targets (what "easy" means) |
| --- |
| The small pipeline is right |
| The cheap answer is no worse than the big one (the routing target) |
| The best setup is right |
| The small retriever finds both passages |
| The question is easy for most setups |

| Safeguard | How |
| --- | --- |
| Scores are held-out | 5-fold cross-validation, repeated 3 times |
| Control for chance | Every combination is re-run on 20 shuffles of the labels; it counts only if it beats the 95th percentile of those |
| Pictures are not evidence | PCA and t-SNE maps carry the held-out number in their titles |
| Association rules are checked on new questions | Mined on the dev half, checked on the test half, next to rules mined from shuffled labels |
| Link to routing | The best signal of each space is used as a gate between the cheap and the big answerer, and its accuracy and compute are reported |

Recorded: scoreboard (every space × strategy × target), topics with their top words and accuracy, clusters, association rules, decision-tree rules as text, signal importance, per-question structure labels, 9 charts, Excel workbook.

**Part 2: the models' own hidden states (GPU, notebook 16).** Internal activations of the small and medium models while they read the question and passages are extracted; PCA and t-SNE maps are drawn; a linear probe gives the held-out separation, with the same shuffled-label control. This signal would be free at run time, because the small model does that reading anyway. It needs a separate script with the standard model library, because vLLM does not return hidden states. Planned for Qwen 3B and 7B and the two Llamas; the 32B is left out unless the smaller models show something.

**Revised order and notebook numbers (6 Oct 2026).** This replaces the numbering in the Steps table above; nothing had been built under the old numbers 14 and 15.

| Order | Notebook | Step | Needs | Status |
| --- | --- | --- | --- | --- |
| 1 | 13 | 5a offline policies | CPU | run 6 Oct |
| 2 | 14 | 5d part 1, structure in existing results | CPU | built and tested 6 Oct |
| 3 | 15 | 5b models read the question; B4; planner sizes | A100 | to build |
| 4 | 16 | 5d part 2, hidden states | A100 | to build |
| 5 | 17 | 5c fresh questions, final test | A100 | to build |

Patterns found in step 5d part 1 are passed to the planner and the deciding model in step 5b as extra input.

**Build note for notebook 14.** Module `code/structure.py`. Tested end to end on the real Set 4 results (about 4 minutes, TF-IDF only; the embedding variants run in Colab). Results will be recorded here after the Colab run.

### Step 5d part 1 results (notebook 14, run 6 Oct 2026)

Run folder: `runs/Set_2__question-reading-routers/20261006-181713_cpu_structure` (CPU, 1047 seconds, embedding model all-MiniLM-L6-v2). 10 spaces, 7 strategies, 5 targets: 335 combinations. 108 beat shuffled labels; chance alone would give about 17. Separation is on held-out questions; 0.5 = useless; \* = beats the 95th percentile of the same method on 20 label shuffles.

| Space | Small pipeline right | Cheap answer no worse than big (routing target) | Best setup right | Small retriever finds both | Easy for most setups | Cost of having it |
| --- | --- | --- | --- | --- | --- | --- |
| Question facts | 0.55\* | 0.49 | 0.62\* | 0.48 | 0.60 | free |
| Question text (TF-IDF) | 0.58\* | 0.69\* | 0.59\* | 0.52 | 0.55 | free |
| Question topics (NMF, 6) | 0.58 | 0.57 | 0.51 | 0.51 | 0.54 | free |
| Question–passage similarity (TF-IDF) | 0.59\* | 0.43 | 0.51 | 0.56 | 0.51 | free |
| Question embedding | 0.57 | 0.50 | 0.60\* | 0.58\* | 0.63\* | under 0.001x |
| Question–passage similarity (embedding) | 0.63\* | 0.56 | 0.60\* | 0.55 | 0.64\* | about 0.005x |
| Metadata of the small pipeline | 0.72\* | 0.55 | 0.53 | 0.61\* | 0.62\* | about 1.3x |
| Metadata of the cheap run | 0.67\* | 0.77\* | 0.58\* | 0.55 | 0.73\* | about 3.5x |
| Cross-model agreement (Llama 3B vs Llama 8B) | 0.67\* | 0.84\* | 0.62\* | 0.57\* | 0.77\* | about 3.9x |
| All signals after the cheap run | 0.66\* | 0.75\* | 0.57\* | 0.52 | 0.75\* | about 3.5x |

Each cell is the best of the 7 strategies for that space and target.

| Finding | Evidence |
| --- | --- |
| There is real structure, and it sits in what the models produce | Metadata and cross-model agreement reach 0.72 to 0.84; the free question and passage views reach 0.43 to 0.69 |
| Agreement between two different small models is the strongest signal found | 0.84 for the routing target, against 0.77 for one model's own metadata |
| A model's metadata predicts its own correctness, not whether a bigger model is needed elsewhere | Small-pipeline metadata: 0.72 for "small pipeline right" but 0.55 for the routing target |
| The question and passages carry a weak signal about general difficulty | Embedding views: 0.60 to 0.64 for "best setup right" and "easy for most setups", beating shuffled labels; 0.50 to 0.56 for the routing target |
| Topics show differences in-sample that do not hold out | Topic 1 (19 questions): cheap answer enough 63% against 86% overall; held-out separation from topics 0.51 to 0.58, never beating shuffled labels |
| No strategy stands out; the space matters more than the method | Mean separation by strategy 0.52 (k-means) to 0.57 (association rules) |
| Association rules mostly do not survive on new questions | 8 of the top 15 hold on test for the routing target, against 7 of 15 for rules mined from shuffled labels; the surviving rules contain answer agreement |
| One signal dominates | Scrambling the agreement of the 5 answers costs 0.13 of separation; every other signal 0.03 or less |
| A readable tree does not beat guessing the majority | Tree on all signals: 79% test accuracy against 83% for always guessing "cheap is enough" |

| Gate between Llama 8B and Qwen 32B answers (shared Qwen 7B retrieval), 35% escalated, held-out scores | Accuracy | Compute |
| --- | --- | --- |
| Always the cheap answerer | 54.5% | 3.5x |
| Escalate a random 35% | 57.8% | 4.9x |
| Gate on cross-model agreement | 62.0% | 5.0x |
| Gate on the cheap run's metadata (41% escalated) | 65.0% | 5.3x |
| Always the big answerer | 64.0% | 5.8x |

Caveats: the question-text result of 0.69 comes from one strategy only (a depth-3 tree) among 335 combinations and may be a chance hit; topic modelling was tested with one method and one topic count on the question only; the gate table uses one reference pair and one escalation share, and its compute does not include the second small answerer needed for cross-model agreement (about 0.4x more).

Carried into step 5b: cross-model agreement and the cheap run's metadata as inputs to the deciding model (B4).

### Follow-up on the question-text signal (6 Oct 2026, offline check)

The question-text view scored 0.69 for the routing target with a decision tree and 0.68 with boosted trees (two strategies, not one as first noted); the other four strategies scored 0.48 to 0.54. Tested as a question-only router against the cheap fixed setup (Qwen 7B retrieves once, Qwen 32B answers once: 63.5% at 4.2x, all 200 questions), cross-validated:

| Cheaper setup for the "easy" questions | Separation (tree) | 30% sent to the cheaper setup | Random router, same share |
| --- | --- | --- | --- |
| Qwen 7B – Llama 8B, 1 answer | 0.42 | 59.0% at 3.6x | 60.3% |
| Qwen 7B – Llama 8B, 5 answers | 0.58 | 60.9% at 3.9x | 60.3% |
| Qwen 7B – Llama 8B, 5 answers, default shortlist | 0.55 | 60.2% at 3.9x | 59.0% |
| Qwen 3B – Llama 8B, 5 answers | 0.50 | 57.4% at 3.7x | 57.6% |
| Qwen 7B – Llama 3B, 5 answers | 0.51 | 58.2% at 3.6x | 58.4% |

The signal does not carry over to other pairs of setups and routes about as well as a random router. One hint kept for step 5b: the tree's hard group (26 questions, in-sample) was questions with unusual wording about obscure subjects, where the big answerer was right far more often than the Llama 8B.

### Step 5d part 2, expanded: hidden signals from a single cheap generation (added 6 Oct 2026 at the user's request)

This replaces the short description of part 2 above with four approaches proposed by the user, in the user's priority order, plus one addition.

**Why this could change the result.** The cascades lost their saving because the cheap answerer had to answer five times before its agreement could be read. A signal from one generation changes the arithmetic. Compute figures are estimates from the saved Set 4 runs.

| Piece | Compute (x cheapest) |
| --- | --- |
| Qwen 7B retrieval, 1 try | about 1.1x |
| Llama 8B answer, 1 try | about 1.4x |
| Llama 8B answers, 5 tries | about 2.4x |
| Qwen 32B answer, 1 try | about 3.1x |
| Cheap fixed setup (Qwen 7B retrieval + Qwen 32B answer, 1 try each) | 4.2x |

| Cascade design | Share of questions it may escalate before it costs more than the cheap fixed setup |
| --- | --- |
| 5 cheap answers, then a gate on their agreement (tested in 5a) | about 24% |
| 1 cheap answer, then a gate on a hidden signal | about 55% |

So a hidden signal does not have to beat answer agreement; it has to come close to it from one generation instead of five.

| Priority | Approach | What is extracted | Why it could help | What it must beat | Extra inference |
| --- | --- | --- | --- | --- | --- |
| First | Hidden-state probe | Internal activations of the cheap answerer, conditioned on the question and the retrieved passages; taken at the last prompt token (before any answer is generated) and as the mean over the answer tokens; several layers | Could reveal uncertainty that question embeddings miss | 0.77 (agreement of 5 answers, routing target) | none beyond one cheap generation |
| Second | Semantic uncertainty probe | A probe trained to predict the disagreement of the 5 answers from a single generation's hidden states | Could keep some of the benefit of repeated sampling without its cost | Routing on the predicted agreement stays near 0.75 | none; the training labels (agreement of 5 answers) are already saved for every question and shortlist |
| Third | Evidence-use attention signals | Share of attention on each retrieved passage, on the question, and on the model's own earlier answer tokens; how concentrated it is; measured at the final-answer tokens | Could flag answers that are poorly grounded in the supplied evidence | Adds separation on top of the hidden-state probe | none; same forward pass |
| Later | Passage-perturbation sensitivity | Change in the answer or its confidence when a shortlisted passage is removed or the passages are reordered | Could expose unstable evidence use or dependence on one passage | 0.77 at similar cost | one extra generation per removed passage, about the cost of 5 samples; a cheap variant is one reordered run |
| Added | Retriever-side probe | Hidden state of the Qwen 7B retriever after reading the question and the 20 passages | The retriever cascade had no skill using agreement of tries; the hidden state may predict "both passages found" better | 0.5 to 0.6 (retrieval agreement) | none beyond the retrieval pass |

| Design point | Decision |
| --- | --- |
| Models probed | Llama 3B and Llama 8B as cheap answerers; Qwen 7B as retriever; full precision, one at a time on the A100 |
| Tooling | A separate script with the standard model library, because vLLM does not return hidden states or attention |
| Generation | The single generation is produced by this script and scored afresh; the big answerer's results are reused from Set 4 |
| Questions | All 400 questions of the original sample, because hidden states have thousands of dimensions; the second 200 have no Set 4 answers yet, so this takes over part of step 5c's generation |
| Dimension reduction | Activations are reduced (PCA) before probing; every score is held-out |
| Controls | Shuffled-label control for every probe, as in part 1 |
| Bar | The cheap fixed setup, 63.5% at 4.2x; a separation under about 0.70 is treated as not enough |
| Pictures | PCA and t-SNE maps of the hidden states, with the held-out number in each title |

Recorded: the same scoreboard layout as part 1 (signal × strategy × target, with shuffled-label control), layer-by-layer separation, the gate's accuracy and compute against the cheap fixed setup, attention summaries per question, per-question decisions, charts and an Excel workbook.

Expectation, stated in advance: with 200 to 400 questions the result is likely to be modest and noisy. From memory of published work, probes of this kind recover a good part of the sampling-based uncertainty signal; this was not checked against sources.

**Order revised (6 Oct 2026, evening).** Given the offline results, the hidden-signal step now runs before the question-reading step. Notebook numbers are unchanged.

| Run order | Notebook | Step | Needs | Status |
| --- | --- | --- | --- | --- |
| 1 | 13 | 5a offline policies | CPU | run 6 Oct |
| 2 | 14 | 5d part 1, structure in existing results | CPU | run 6 Oct |
| 3 | 16 | 5d part 2, approaches 1 to 3 and the retriever-side probe | A100, one session | to build next |
| 4 | 15 | 5b models read the question; B4; planner sizes | A100 | to build |
| 5 | 16b | 5d part 2, approach 4 (passage perturbation) | A100 | later |
| 6 | 17 | 5c fresh questions, final test | A100 | to build |

**The hidden signals feed the rules; they are not a new router (clarified 6 Oct 2026).** A probe is a small trained model, but its job is only to turn thousands of activation values into one readable number, because a rule cannot refer to raw activations. The routing decision stays with the planner's NL rules and the deciding model.

| Stage | What happens | Where |
| --- | --- | --- |
| A. Measure the signals | Is each signal informative at all: held-out separation, shuffled-label control, what it would buy as a simple gate | Notebook 16 |
| B. Give the informative ones to the rules | The planner writes NL rules that may use them; the deciding model in B4 sees them with the question, the other metadata and the rules | Notebook 15 |

| Approach | What comes out | Needs a trained probe? |
| --- | --- | --- |
| Hidden-state probe | One number: predicted chance that the cheap answer is enough | Yes |
| Semantic uncertainty probe | One number: predicted agreement among answers | Yes |
| Evidence-use attention | Direct measurements: share of attention on each passage, the question, its own text | No |
| Passage perturbation | Direct measurements: did the answer change | No |

This differs from Sets 1, 2 and 4, where a classifier chose the route itself and did badly. Here the probe never chooses a route; it adds a line to the metadata.

**Build note for notebook 16 (6 Oct 2026).** Module `code/hidden.py`; notebook `16_set5d_hidden_signals.ipynb`. What it does, and where it departs from the plan above:

| Item | As built |
| --- | --- |
| Questions | The 200 Set 4 questions, not 400. The second 200 have no big-answerer results yet, so they cannot be labelled; they are added with step 5c |
| Shortlist | Qwen 7B retrieves once (CoT, per-attempt shortlist 3), the same shortlist the cheap fixed setup answers from |
| Cheap answer | One greedy answer per question from Llama 8B and from Llama 3B, same prompt as Set 4, scored directly without the fusion step |
| Hidden states | 6 layers (25%, 50%, 62.5%, 75%, 87.5%, 100% of depth) at three places: last prompt token, mean over the answer tokens, final answer line |
| Attention | 6 numbers from all answer tokens and the same 6 from the final answer line, averaged over heads and the upper half of the layers |
| Semantic uncertainty probe | A linear probe trained only on "did the 5 Set 4 answers agree", then scored against the routing target |
| Also recorded | Token probabilities, stated confidence, and whether the two small models' single answers agree |
| Retriever-side probe | Qwen 7B hidden state at the last prompt token, 6 layers; no attention for the retriever (the prompt is too long to keep attention weights) |
| Comparison signals | Agreement of 5 answers and of 3 retrieval tries from Set 4, with their extra cost charged |
| Gate | One cheap answer, then escalate the least-sure questions to Qwen 32B; for signals read before answering, an escalated question skips the cheap answer |
| Bar | The cheap fixed setup on these questions: 63.5% at 4.2x; a point counts if it is within 2 points at 15% or more less compute |
| Hand-off to stage B | `results/signals_for_rules.jsonl`: per question, the held-out value of each signal that beat shuffled labels |
| Not in this notebook | Passage perturbation (notebook 16b, later) |

Testing: the whole notebook was run end to end with a tiny random model on CPU, which exercises every code path (generation, hidden states, attention, resume after interruption, analysis, charts). It has not been run on the real models, because that needs the GPU; the first real run is the user's.

### Step 5d part 2: first run and first results for the Llama 8B (notebook 16, 7 Oct 2026)

Run folder: `runs/Set_2__question-reading-routers/20261007-075955_A100_hidden`. Status: partial. Llama 8B extraction complete (200 of 200); Llama 3B 33 of 200 (stopped at the time limit); Qwen 7B retriever not started.

**Incident: the analysis step took 103 minutes.** The first version of the analysis re-compressed the 4,096-dimension hidden states inside each of about 25,000 cross-validation fits. It had been tested only with a tiny model (64 dimensions), where this was fast. The session ran 2 h 35 min in total. Fix: the hidden states are compressed once to 16 principal components before the fits (no labels are used in that step). A first attempt at the fix compressed to 32 components and then standardised them, which changed what the probes saw and understated the hidden-state results; this was caught by comparing against the slow run's output and corrected. The corrected fast analysis takes about 3 minutes and reproduces the slow run (216 hidden-state rows compared: mean difference −0.02, standard deviation 0.05).

| Timing (first session) | Seconds |
| --- | --- |
| Copy Llama 8B | 339 |
| Extract Llama 8B, 200 questions | 2281 (11.3 s per question, mean 248 answer tokens) |
| Copy Llama 3B | 163 |
| Extract Llama 3B, 33 questions | 240 (6.5 s per question) |
| Analysis (slow version) | 6207 |

The extraction itself worked as intended: the Llama 8B's single greedy answers score 54.0%, against 53.0% for one sampled answer in Set 4; 7.5% of answers reached the 512-token cap; attention was measured for every question. In the first run's saved tables, the Llama 3B rows rest on 33 questions and should be ignored.

**Llama 8B, routing target ("its single answer is no worse than the big answer"), 200 questions, held-out separation.**

| Signal | How it is read | Across the 6 layers | Beats shuffled labels |
| --- | --- | --- | --- |
| Hidden state on the final answer line | Semantic uncertainty probe | 0.72 to 0.80, mean 0.78 | 6 of 6 layers |
| Hidden state on the final answer line | Linear probe trained on the routing target | 0.69 to 0.76, mean 0.73 | 6 of 6 |
| Evidence-use attention signals | First 2 principal components | 0.72 | yes |
| Token probabilities of the single answer | Boosted trees | 0.66 | yes |
| Hidden state averaged over all answer tokens | Semantic uncertainty probe | 0.59 to 0.70, mean 0.63 | 5 of 6 |
| Hidden state at the last prompt token, before answering | Semantic uncertainty probe | 0.45 to 0.64, mean 0.59 | 4 of 6 |
| Agreement of 5 answers (the Set 4 signal) | Linear probe | 0.62 | yes |
| Stated confidence of the single answer | any | 0.43 to 0.52 | no |

| Gate: Qwen 7B retrieves once, Llama 8B answers once, the least-sure questions go to Qwen 32B | Share escalated | Accuracy | Compute | Random gate, same share |
| --- | --- | --- | --- | --- |
| Always the cheap answerer | 0% | 54.0% | 2.1x | – |
| Semantic uncertainty probe on the final answer line | 30% | 63.0% | 2.9x | 56.9% |
| Semantic uncertainty probe on the final answer line | 40% | 64.0% | 3.1x | 57.8% |
| Semantic uncertainty probe on the final answer line | 50% | 62.5% | 3.4x | 58.8% |
| Evidence-use attention signals | 35% | 61.5% | 3.0x | 57.3% |
| Agreement of 5 answers (Set 4 signal, cost of 4 more answers charged) | 35% | 61.0% | 4.8x | 57.3% |
| Cheap fixed setup (always Qwen 32B, one answer) | 100% | 63.5% | 4.2x | – |

| Finding (provisional) | Evidence |
| --- | --- |
| First approach to match the cheap fixed setup at clearly less compute | Semantic uncertainty gate: 63.0 to 64.0% at 2.9 to 3.1x, against 63.5% at 4.2x; about 26 to 31% less compute |
| The signal sits where the model writes its final answer | Final answer line 0.72 to 0.80; all answer tokens 0.59 to 0.70; before answering 0.45 to 0.64 |
| It is stable across layers | All 6 layers beat shuffled labels on the final answer line |
| A probe trained only on answer agreement routes better than one trained on the routing target | 0.78 against 0.73 mean over layers |
| One generation beats five | Hidden state 0.78 and attention 0.72 against 0.62 for the agreement of 5 answers on this shortlist |
| Stated confidence carries nothing | 0.43 to 0.52 |

Caveats: the escalation share and the layer were chosen on the same 200 questions; one answerer; one shortlist setting; differences of a few points are within noise. The confirmation is the fresh-question test (step 5c), with the probe, layer and share frozen first.

## Step 5d part 2: complete results (notebook 16, both sessions, analysis version 5d2.3)

Recorded 7 Oct 2026. Run folder: `runs/Set_2__question-reading-routers/20261007-075955_A100_hidden`. All numbers are on the 200 Set 4 questions, held-out (5-fold cross-validation x3). The bar is the cheap fixed setup (Qwen 7B retrieves once, Qwen 32B answers once): 63.5% at 4.2x.

### What ran

| Session | What it did | Time | Status |
| --- | --- | --- | --- |
| 1 (07:59) | Llama 8B extraction (200 questions), then analysis | 2 h 35 min (analysis alone 103 min) | Extraction valid. Analysis correct but far too slow |
| 2 (11:20) | Llama 3B extraction (21 min), Qwen 7B retriever extraction (about 8 min including copying the model), analysis (11 min) | about 45 min | Extraction valid. Analysis used a stale interim file, so its tables were wrong |
| Local rebuild (12:10) | Analysis re-run on the complete extraction with version 5d2.3 | 6 min | These are the results below and the ones now in `results/` |

### Incident: two analysis problems, both fixed

| Problem | Cause | Effect | Fix |
| --- | --- | --- | --- |
| Analysis took 103 minutes | The 4,096-number hidden states were reduced again inside each of about 25,000 fits. It had only been tested with a tiny 64-number model | Session 1 ran 2 h 35 min | Hidden states are reduced once, before the fits |
| Interim fix understated hidden-state results | The first fix reduced to 32 components and then reduced again to 16, which changed what the probes saw | Llama 8B semantic uncertainty probe showed 0.61–0.68 instead of 0.72–0.80 | Reduce once to 16 components. This reproduces the slow, correct run (216 rows, mean difference −0.02) |
| Session 2 used the interim file | Colab's Drive mount served the older `hidden.py` although the corrected one had been saved 6 minutes earlier | The tables written at 11:20 were from the flawed analysis | Results rebuilt locally from the saved extraction and written to `results/`. The flawed tables are kept in `sessions/20261007-112026_hidden/results_snapshot/`. The code version is now printed in the log and at the top of `summary.md` |

No model needs to be re-run: the extraction (the expensive part) was never affected.

### Single greedy answers

| Answerer | Right with one greedy answer |
| --- | --- |
| Llama 3B | 44.0% |
| Llama 8B | 54.0% |
| Qwen 32B (the big answerer, for reference) | 63.5% |

### Which signals tell us when the big answerer is needed (separation; 0.5 = useless)

| Signal | Llama 8B | Llama 3B |
| --- | --- | --- |
| Hidden state on the final answer line, semantic uncertainty probe | 0.72–0.80 (all 6 layers beat shuffled labels) | weak |
| Hidden state on the final answer line, plain linear probe | 0.69–0.76 | 0.58–0.65 |
| Evidence-use attention | 0.72 | weak |
| Token probabilities of the answer | 0.66 | weak |
| Hidden state before answering (last prompt token) | 0.45–0.64 | weak |
| Agreement of 5 answers (Set 4) | 0.62 | – |
| Stated confidence | 0.43 | – |
| Its single answer equals the other Llama's answer | – | 0.74 |

For the simpler target "is the single answer right": Llama 8B hidden state 0.74–0.79, Llama 3B hidden state 0.69–0.73.

### Gate: one cheap answer, the least-sure questions go to Qwen 32B

| Gate | Share sent to Qwen 32B | Accuracy | Compute |
| --- | --- | --- | --- |
| Cheap fixed setup (the bar) | 100% | 63.5% | 4.2x |
| Llama 8B, semantic uncertainty probe | 30% | 63.0% | 2.87x |
| Llama 8B, semantic uncertainty probe | 40% | 64.0% | 3.12x |
| Llama 8B, semantic uncertainty probe | 50% | 62.5% | 3.37x |
| Llama 8B, attention | 30–40% | about 61.5% | 2.9–3.2x |
| Llama 8B, random gate (control) | 30–50% | about 57–59% | same |
| Llama 3B, answer equals Llama 8B's | 40% | 60.5% | 3.30x |
| Llama 3B, answer equals Llama 8B's | 60% | 62.0% | 3.80x |

This is the first approach in the project that matches the cheap fixed setup at about 26–31% less compute. 23 gate points meet the provisional success test.

### Retriever (Qwen 7B hidden state while it picks passages)

| Target | Hidden state | Agreement of 3 tries (costs about 2.2x more) |
| --- | --- | --- |
| Finds both passages in 1 try | 0.73 | 0.70 |
| Finds both passages in 3 tries | 0.68 | – |
| A bigger retriever is needed | 0.65 | 0.77 |

### Reading

- The useful signal sits on the final answer line of the Llama 8B, after it has answered, not before. So the question alone still does not reveal difficulty; the model's own state after answering does.
- The Llama 3B is too weak a cheap answerer for this: its hidden signals do not route well. Its best signal needs a second model.
- Overall 322 of 613 signal-and-method combinations beat shuffled labels.

### Caveats

- The escalation share and the layer were chosen on the same 200 questions, so the gate result is provisional. Confirmation needs fresh questions with probe, layer and share frozen first (step 5c, notebook 17).
- 200 questions: a difference of 1–2 points is within noise.
- Hidden-state extraction used HF transformers; a deployed system would need the hidden state from the serving engine, which vLLM does not expose today.

### What goes forward to stage B (notebook 15)

`results/signals_for_rules.jsonl` holds, per question, the informative signals in plain form (semantic uncertainty score, attention summary, token probability, cross-model agreement). In stage B these are given to the planner's natural-language rules and to the deciding model (B4), with each planner size tested in turn.

## Step 5c: final test of the frozen gate (notebook 17) – design, written before the run

Recorded 7 Oct 2026, 16:20, before any fresh question was run. Order changed at the user's request: the final test (notebook 17) is run before stage B (notebook 15), so that the one promising result is confirmed or rejected first.

### The question

Does the gate from notebook 16 still match the cheap fixed setup at lower compute when nothing is tuned on the test questions?

| Item | What it is |
| --- | --- |
| The bar | Cheap fixed setup: Qwen 7B retrieves once, Qwen 32B answers once. This is the setup AgentTTS picks |
| The gate | Llama 8B answers once. A probe reads its hidden state on the final answer line. The least-sure questions go to Qwen 32B |
| Success test (unchanged) | Accuracy within 2 points of the bar on the fresh questions, at 15% or more less compute, gate cost included |
| Fresh questions | 200 HotpotQA questions that no earlier set used (the 400 used before are excluded), built the same way: 2 gold passages plus BM25 distractors, 20 passages, seed 43. Folder `data/hotpotqa_q/fresh200_c20_s43` |
| Old vs fresh sample | Old: 170 bridge, 30 comparison. Fresh: 165 bridge, 35 comparison. Same passage length |

### What was frozen, and by what rule

Everything was fitted on the 200 old questions only and saved in `frozen/set5c_gate_v1/` with checksum `deb8301e4670d098`. Notebook 17 refuses to run if the file has changed.

| Choice | Rule (written before the run) | Result |
| --- | --- | --- |
| Signal | The best signal of notebook 16: semantic uncertainty probe (trained only to predict whether 5 answers agree) on the final answer line | – |
| Layer | The layer whose probe separates the routing target best on held-out old questions | Layer 24 of 32 (separation 0.795; other layers 0.72–0.79) |
| Threshold | The smallest share sent to Qwen 32B at which held-out accuracy on the old questions is at least the bar's | 40% share, score threshold 0.418 |
| Held-out result on the old questions | – | 64.0% at 3.12x, against the bar 63.5% at 4.20x |

The gate as one rule: the Llama 8B answers once on the Qwen 7B's shortlist. Read its hidden state on the final answer line at layer 24 and compute the semantic uncertainty score (the probability that five answers would agree). IF the score is at most 0.418 THEN discard that answer and let the Qwen 32B answer once; OTHERWISE keep the Llama 8B's answer.

### What is reported

| Result | Role |
| --- | --- |
| Primary gate against the bar: accuracy, compute, difference with 95% interval, paired counts | Decides the test |
| Random gate sending the same share | Control |
| Same signal with thresholds frozen at the 30% and 50% shares; three other frozen signals (plain linear probe, attention, token probabilities) at the 40% share | Secondary, named in advance, do not decide the test |
| Whole frozen curve (19 thresholds) | Shape of the trade-off |
| Separation of each frozen signal, old against fresh | Does the signal carry over |
| Where the gate gained and lost (4 kinds of question) | Sense-making |
| Fixed Qwen 7B – Qwen 32B setups with 3 retrieval tries and 5 answers | Reference |
| Every question with both answers, scores and the gate's decision | Sense-making |

Charts: accuracy against compute (old and fresh side by side), score distribution with the frozen threshold, separation old against fresh, where the gate gained and lost.

### How it runs

| Stage | Model | Code | Estimate |
| --- | --- | --- | --- |
| 1. Retrieve | Qwen 7B, CoT, select 3, 3 tries | Set 4 code, unchanged, in vLLM | about 12 min including install and copy |
| 2. Big answers | Qwen 32B on the "keep all" shortlists | Set 4 code, unchanged, in vLLM | about 12 min |
| 3. Cheap answer and hidden state | Llama 8B, one greedy answer | Notebook 16 code, unchanged | about 40 min |
| 4. Evaluate | none | `code/final.py`, CPU | under 1 min |

Total about 65–75 minutes on an A100 (an estimate from earlier timings). Resumable. Models are copied from Drive in the background.

### Checks done before the run

| Check | Result |
| --- | --- |
| Frozen readers reproduce the notebook 16 analysis code on the old questions | Yes (asserted to 6 decimals) |
| Replaying the evaluation on the old questions gives the known numbers | Yes: bar 63.5% at 4.20x, Llama 8B alone 54.0% at 2.08x |
| Whole notebook tested end to end with fake models, including stopping and resuming | Yes |
| Notebook checks that Drive serves the current code files (checksums) | Added, because of the stale-file incident in notebook 16 |

### Limits known in advance

- 200 questions: the 95% interval on the accuracy difference is about ±3 to 4 points. A pass means "no sign of a loss", not "proven equal".
- On the old questions only 25 of 200 questions (12.5%) are ones where Qwen 32B is right and Llama 8B wrong. The whole gain rests on catching those, so some shrinkage on fresh questions is expected.
- The hidden-state extraction runs in the environment that vLLM installs, which may differ in library versions from notebook 16. Versions are recorded in `session.json`.
- The three other signals are frozen as linear readers. In notebook 16 attention reached 0.72 with a tree reader; its linear reader is weaker on the old questions (0.63).

Results will be added below after the run.

## Step 5c: results of the final test (notebook 17)

Recorded 7 Oct 2026, 17:40. Run folder: `runs/Set_2__question-reading-routers/20261007-162319_A100_final`. One session, 66 minutes (install 6, retrieval 7, Qwen 32B answers 10, Llama 8B extraction 42, evaluation under 1). Code and frozen gate checksums matched. Library versions: torch 2.13.0, transformers 5.17.0.

### Headline

The frozen gate passes the test as it was written, but the pass is weak: the gate is not clearly better than a random gate, and the signal did not carry over at the strength seen on the old questions.

| Policy (200 fresh questions) | Sent to Qwen 32B | Accuracy | Compute | Accuracy vs the bar | Compute vs the bar |
| --- | --- | --- | --- | --- | --- |
| Cheap fixed setup (the bar) | 100% | 53.5% | 4.21x | – | – |
| Primary frozen gate | 43% | 52.0% | 3.21x | −1.5 points (95% interval −5.5 to +3.0) | −23.9% |
| Random gate, same share (control) | 43% | 50.7% | 3.19x | −2.8 points | −24.3% |
| Llama 8B alone, one answer | 0% | 48.5% | 2.07x | −5.0 points | −50.8% |
| Perfect gate (needs the answer key) | 13% | 61.5% | 2.41x | +8.0 points | −42.9% |

Success test (within 2 points, at least 15% less compute): met, at −1.5 points and −23.9%. Paired counts: the gate is right on 8 questions where the bar is wrong and wrong on 11 where the bar is right (sign test p = 0.65). The gate is within 2 points in 62% of bootstrap resamples.

### Why the pass is weak

| Evidence | Number |
| --- | --- |
| Gate minus a random gate with the same share | +1.3 points (95% interval −1.8 to +4.6): not distinguishable from zero |
| Separation of the routing target by the frozen signal | 0.80 on the old questions, 0.64 on the fresh ones (95% interval 0.51 to 0.77) |
| Share of the "escalating helps" questions the gate caught | 15 of 26 (58%); a random gate would catch 43% |
| The whole frozen curve | Flat at 51.5–53.0% for every threshold; even sending 6.5% of questions (51.5% at 2.25x) meets the test |
| Gap between Llama 8B alone and the bar | Only 5.0 points on the fresh questions (9.5 on the old), so a 2-point tolerance covers much of it |

### Old questions against fresh questions

| Measure | Old 200 (held-out estimate) | Fresh 200 |
| --- | --- | --- |
| Cheap fixed setup: accuracy | 63.5% | 53.5% |
| Llama 8B, one answer: accuracy | 54.0% | 48.5% |
| Primary gate: accuracy | 64.0% | 52.0% |
| Primary gate: share sent to Qwen 32B | 40% | 43% |
| Primary gate: compute vs the bar | −25.7% | −23.9% |
| Qwen 7B finds both passages in 1 try | 83.5% | 75.5% |
| Qwen 32B right, Llama 8B wrong (escalating helps) | 12.5% | 13.0% |
| Llama 8B right, Qwen 32B wrong (escalating hurts) | 3.0% | 8.0% |

The fresh questions are harder: the bar itself dropped 10 points. Compute behaved as predicted; accuracy did not.

### Signals, old against fresh (separation of the routing target)

| Frozen signal | Old (held-out) | Fresh |
| --- | --- | --- |
| Semantic uncertainty probe, layer 24 | 0.80 | 0.64 |
| Plain linear probe, layer 28 | 0.76 | 0.65 |
| Attention signals | 0.63 | 0.66 |
| Token probabilities | 0.58 | 0.54 |

All four secondary gates land at 51.5–52.5%, the same as the primary.

### Extra checks after the run (exploratory, not part of the test)

| Check | Result | Meaning |
| --- | --- | --- |
| Probe refitted on the fresh questions with cross-validation, routing target | 0.60–0.67 across layers | The weak result is not a transfer problem. The true strength of this signal is about 0.65; the 0.80 was optimistic |
| Same hidden states predicting "Llama 8B is right" (refitted, fresh) | 0.69–0.75 | The hidden state does know when the Llama 8B is unsure |
| Frozen score predicting "both models give the same answer" | 0.77 | Same point |
| Accuracy on the 86 escalated questions | Qwen 32B 44.2%, Llama 8B 36.0% | The questions the gate sends are hard for both models, so sending them gains little |
| When the two answers differ (40% of questions) | Qwen 32B right 32.5%, Llama 8B right 20.0% | Disagreement mostly means both are struggling |
| Accuracy by retrieval outcome | Both passages found: bar 60.9%. Not found: bar 30.6% | Retrieval decides more than the choice of answerer |
| Qwen 7B retrieves 3 times, Qwen 32B answers once | 56.5% at 4.75x (+3.0 points for +13% compute) | The only setup that moved accuracy on the fresh questions |

### Insights

- The gate detects "the Llama 8B is unsure", not "the Qwen 32B would do better". These are different things. On unsure questions the Qwen 32B is usually wrong too, often because the shortlist lacks a passage.
- The compute saving is real and predictable (about 24%). The accuracy cost is small but not zero, and at this sample size it cannot be told apart from a random gate.
- The 0.80 separation on the old questions was inflated by choosing the layer and reading on the same 200 questions, even though each number was held-out. Refitting on fresh questions gives about 0.65.
- With the same shortlist, the big answerer is only 5 to 9 points better than the medium one, and much of the difference between their single answers is noise in both directions (13% one way, 8% the other). That leaves little for any answerer gate to win.
- The headroom is still large (perfect gate 61.5% at 2.41x), but no signal measured so far predicts where it is.
- Retrieval is the stronger lever: accuracy doubles when both passages are found, and extra retrieval tries were the only change that raised accuracy.
- Two samples of 200 questions gave bars of 63.5% and 53.5%. Any single 200-question result in this project carries roughly ±7 points of sampling error.
- The success test was too easy to meet when the cheap and big answerers are close. A stricter test should also require beating a random gate with the same share.

### Status of the claim

| Claim | Status |
| --- | --- |
| The gate cuts compute by about a quarter | Confirmed |
| The gate keeps accuracy within 2 points | Met on the point estimate; not statistically established |
| The gate routes better than chance | Not confirmed on fresh questions (+1.3 points over random, interval includes zero) |
| Per-question switching of the answerer beats AgentTTS's fixed pick | Not shown |

## Quick check of suggested follow-ups: benefit-trained probes, ensembles, class weighting, SMOTE, more data

Recorded 7 Oct 2026, 18:00. The user proposed ensembles and SMOTE; an outside review (ChatGPT) ranked five follow-ups. Four of them could be tested at once on data already collected: train on the old 200 questions, test on the fresh 200. No GPU. The fresh questions have now been used for development, so they can no longer serve as an untouched test.

Separation is for "escalating helps" (Qwen 32B right, Llama 8B wrong). 0.5 = useless. The frozen gate's reference values: separation 0.64, gate 1.3 points above a random gate.

| Approach | Separation on fresh questions | Gate vs random gate (points) | Verdict |
| --- | --- | --- | --- |
| Frozen gate (reference) | 0.64 | +1.3 | – |
| Probe trained directly on "escalating helps", 6 layers | 0.61–0.71 | +0.3 to +4.6 | No gain. This target is the mirror image of the routing target the plain linear probe was already trained on |
| Class weighting | 0.61–0.65 | +0.4 to +1.0 | No gain |
| Random oversampling | 0.61–0.65 | −0.8 to +0.5 | No gain |
| SMOTE | 0.62–0.67 | +0.4 to +2.0 | No gain |
| Average of 2, 3 or 4 frozen scores | 0.63–0.65 | +1.4 to +1.6 | No gain |
| Stacked logistic regression on 3 scores | 0.64 | +1.6 | No gain |
| Twice the data (cross-validation on all 400 questions) | 0.65–0.69 | – | Small gain at most |

### Reading

- The four signals are strongly correlated with each other (0.68–0.75 between hidden-state, plain-probe and attention scores), so an ensemble has little new information to add. The stacked model gave the attention score a weight of 0.02.
- Imbalance is not the limit. With only 25 "helps" examples, resampling re-uses or interpolates the same 25; it cannot add information.
- Doubling the data moved separation from about 0.65 to at most 0.69. The ceiling for this family of signals (one cheap answer's hidden state) appears to be about 0.65–0.70.
- "Escalating hurts" has 6 examples in the old questions, too few to train a separate harm probe.
- The best single row (layer 8, +4.6 points) is one of six layers picked after seeing the fresh results, so it should be treated as noise until tested on new questions.

### Check of one figure in the outside review

The review cited 0.77 for the retriever's hidden state predicting "a bigger retriever is needed". In notebook 16 the 0.77 belongs to the agreement of 3 retrieval tries. The hidden state scored 0.65 on that target and 0.73 on "finds both passages in 1 try". Neither has been checked on fresh questions.

### What remains viable

| Follow-up | Status |
| --- | --- |
| Benefit-trained probes, ensembles, class weighting, SMOTE | Tested here, no gain; not worth a notebook |
| More questions | Helps measurement (narrower intervals) more than it helps the signal |
| Retrieval-stage decision | Not yet tested; the strongest remaining lead (accuracy 61% when both passages are found, 31% when not) |
| New signal families (for example passage perturbation, or a second cheap model's answer) | Not yet tested on fresh questions |

## Step 5e: retriever-side gate (notebook 18) – design, written before the run

Recorded 7 Oct 2026, 18:20, before the retriever's hidden states on the fresh questions existed.

### What had already been tried on the retriever

| Step | What was tried | Result |
| --- | --- | --- |
| Set 3 (notebook 09) | Chain-of-thought for the small retriever | CoT helps retrieval |
| Set 4 (notebooks 11) | 3 retriever sizes, CoT, shortlist sizes, 1 or 3 tries, as fixed setups | Qwen 7B nearly as good as Qwen 32B; "keep all" best |
| Notebook 12 | Planner's rules chose a whole route per question, retriever included | Did not beat the fixed setup |
| Notebook 13 | Retriever cascade on the agreement of 3 tries (A2), and both stages (A3) | A2 62.1% at 6.7x, no skill; A3 58.8% at 5.7x |
| Notebook 16 | Qwen 7B hidden state measured | 0.73 for "finds both in 1 try"; not turned into a gate |

New in this step: a signal that is free (read during the first retrieval pass), and "try again" as the action instead of "use a bigger retriever".

### How much there is to win (known before the run)

| Measure | Old 200 | Fresh 200 |
| --- | --- | --- |
| First try finds both passages | 83.5% | 75.5% |
| Three tries find both passages | 94.0% | 87.0% |
| Accuracy, always 1 try | 63.5% | 53.5% |
| Accuracy, always 3 tries | 62.0% | 56.5% |
| Where the first try missed a passage: 1 try → 3 tries | 48.5% → 57.6% | 30.6% → 40.8% |
| Where the first try found both: 1 try → 3 tries | 66.5% → 62.9% | 60.9% → 61.6% |
| Retrying raised / lowered accuracy (questions) | 6 / 9 | 11 / 5 |
| Ceiling: retry exactly where the first try missed | 65.0% (+1.5 points) | about 56.0% (+2.5 points) |

Two things follow. Retrying helps by about 10 points where a passage was missed, on both samples, so the logic is sound. But the ceiling is only +1.5 to +2.5 points of accuracy even with a perfect gate, which 200 questions cannot detect. "Always 3 tries" was worse on the old questions and better on the fresh ones, so the earlier remark that extra tries raised accuracy rests on one sample.

### What was frozen (on the old 200 questions only)

Saved in `frozen/set5e_retriever_gate_v1/`, checksum `dbdae3c5b5b776da`.

| Choice | Rule | Result |
| --- | --- | --- |
| Target | "The first try finds both passages" | – |
| Layer | Best held-out separation on the old questions | Layer 21 of 28 (0.725). Other layers: 0.53, 0.56, 0.60, 0.66, 0.69 |
| Threshold | Retry the share equal to the old miss rate (16.5%), rounded up to 20% | Retry when the probability is at most 0.725 |
| Held-out result on the old questions | – | Both passages found 88.0% (random gate 85.6%); accuracy 64.0% at 4.32x |
| Secondary signal | Facts about the first try (stated confidence, token probability, length, passages picked) | 0.64 on the old questions |

The gate as one rule: the Qwen 7B reads the question and the 20 passages. Read its hidden state at the last prompt token at layer 21 and compute the probability that one try will find both passages. IF it is at most 0.725 THEN retrieve three times and keep every passage picked; OTHERWISE retrieve once. The Qwen 32B then answers once.

### The test (stricter than in step 5c)

The gate must find both passages more often than a random gate retrying the same share, with a 95% interval that excludes zero. Accuracy is reported against the same random gate, but is not expected to be decidable at this sample size.

### How it runs

Notebook 18, about 10 minutes on an A100: copy the Qwen 7B, extract its hidden state on the 200 fresh questions (1–2 minutes), evaluate on CPU. The 3 retrieval tries and the Qwen 32B answers come from the notebook 17 run; results go into that run folder under `results_retriever/`.

### Caveats known in advance

- The 0.725 is the best of 6 layers, so some shrinkage on fresh questions is expected, as happened with the answerer signal (0.80 to 0.64).
- The fresh questions were already used once (step 5c and the follow-up checks), though never for anything on the retriever side.
- Checks done: frozen readers reproduce the analysis code; replay on the old questions gives the known numbers; notebook tested end to end with fake models.

Results will be added below after the run.

## Step 5e: results of the retriever-side gate (notebook 18)

Recorded 7 Oct 2026, 22:00. Results in `runs/Set_2__question-reading-routers/20261007-162319_A100_final/results_retriever/`. Code and frozen gate checksums matched.

### Headline

The frozen retriever gate does not beat a random gate. The signal carried over only weakly.

| Policy (200 fresh questions) | Share retried | Both passages found | Random gate, same share | Gate minus random | Accuracy | Compute |
| --- | --- | --- | --- | --- | --- | --- |
| Primary frozen gate | 16.0% | 78.5% | 77.3% | +1.2 points (95% interval −0.7 to +3.2) | 54.0% | 4.32x |
| Always one try (cheap fixed setup) | 0% | 75.5% | – | – | 53.5% | 4.21x |
| Always three tries | 100% | 87.0% | – | – | 56.5% | 4.75x |
| Perfect gate (needs the answer key) | 24.5% | 87.0% | 78.3% | +8.7 points | 56.0% | 4.36x |
| Secondary: facts about the first try | 18.5% | 79.5% | 77.6% | +1.9 points (−0.2 to +4.1) | 53.5% | 4.35x |

Test (interval must exclude zero): not met. Accuracy of the primary gate against a random gate: +0.0 points (−0.9 to +1.1).

### Signal, old against fresh (separation of "the first try finds both passages")

| Signal | Old (held-out) | Fresh | Fresh 95% interval |
| --- | --- | --- | --- |
| Retriever hidden state, layer 21 (frozen) | 0.73 | 0.63 | 0.55 to 0.72 |
| Facts about the first try (frozen) | 0.64 | 0.63 | 0.54 to 0.72 |
| Hidden state refitted on fresh questions, by layer (exploratory) | 0.53–0.73 | 0.50–0.65 | – |

The hidden state is no better than four plain facts about the first try. Refitting on the fresh questions does not recover the old value, so the 0.73 was inflated by picking the best of six layers.

### Whole frozen curve

At every threshold the gate is 0.3 to 2.4 points above a random gate on "both passages found". The gate is consistently on the right side of random, but by a small margin that this sample cannot separate from zero.

### Insights

- Retrying still helps where a passage was missed (30.6% to 40.8% accuracy on the fresh questions, 48.5% to 57.6% on the old), and a perfect gate would capture it at 4.36x. The value exists; the signal to find it does not.
- Of 49 fresh questions where the first try missed a passage, three tries recover the passage on 23. The gate's retries recovered about 6.
- Three separate fresh-question checks now show the same pattern: a number chosen as the best of several on the old 200 questions (0.80 answerer, 0.73 retriever) falls to about 0.63–0.65 on fresh questions. Cheap signals from a single pass appear to top out near 0.65 on this task.
- A separation of about 0.65 is real (intervals mostly above 0.5) but too weak to route with: it moves outcomes by 1 to 2 points, below what 200 questions can confirm and below what would matter against a fixed setup.

### Where the project stands after steps 5a to 5e

| Approach | Tested on | Result against AgentTTS's fixed pick |
| --- | --- | --- |
| Planner's rules over metadata (notebook 12) | 100 test questions | Worse (60% at 6.7x against 68% at 5.8x) |
| Cascades and learned routers, 9 families (notebook 13) | Old 200, cross-validated | None better |
| Structure in question text and metadata (notebook 14) | Old 200, cross-validated | No usable signal that is cheap |
| Answerer gate on hidden state (notebooks 16, 17) | Fresh 200, frozen | About 24% less compute, accuracy −1.5 points, not better than a random gate |
| Ensembles, benefit-trained probes, class weighting, SMOTE | Old to fresh | No gain |
| Retriever gate on hidden state (notebook 18) | Fresh 200, frozen | Not better than a random gate |

The consistent finding: on this task, at these model sizes, per-question switching has headroom on paper (perfect gates gain 2.5 to 8 points or save about 40% compute), but no signal available before or during one cheap pass predicts where that headroom is well enough to beat a well-chosen fixed setup.

## Cross-family ensemble check: do question text, topics, metadata and hidden states add up?

Recorded 7 Oct 2026, 22:45. The user asked whether the work so far was purely latent-space, and whether ensembling signals from different families (for example topic modelling with hidden-state signals) had been tried, since they may not overlap.

### Which signal families each step used

| Step | Question text / topics | Hand-made question facts | Metadata of a run | Rules | Hidden state / attention | Combined across families? |
| --- | --- | --- | --- | --- | --- | --- |
| Notebook 12 (planner's rules) | No | Yes | Yes | Yes | No | Facts + metadata, through rules |
| Notebook 13 (routers, cascades) | Yes (TF-IDF, embedding) | Yes | Yes | Yes (smoothed Set 4 rules) | No | Yes: metadata + text in one model (A9) |
| Notebook 14 (structure) | Yes (TF-IDF, topics, embedding) | Yes | Yes | Association and tree rules | No | Yes: one combined space of all cheap-run signals |
| Notebooks 16–17 (answerer gate) | No | No | Token probabilities, stated confidence | One threshold rule | Yes | Only within the Llama 8B's own signals |
| Notebook 18 (retriever gate) | No | No | Facts about try 1 | One threshold rule | Yes | No |
| Earlier ensemble check (step 5c follow-up) | No | No | Token probabilities | No | Yes | Only within the Llama 8B's own signals |

So the gap was real: text, topics and metadata had never been ensembled with the hidden-state signals, and never on fresh questions.

### The check

Ten families, each read by its own model trained on the old 200 questions, then combined five ways. Tested on the fresh 200. No GPU. Separation: 0.5 = useless.

| Family | Answerer target: old (held-out) | Fresh | Retriever target: old (held-out) | Fresh |
| --- | --- | --- | --- | --- |
| Question facts | 0.51 | 0.46 | 0.62 | 0.43 |
| Question text (TF-IDF) | 0.55 | 0.52 | 0.45 | 0.48 |
| Question topics (NMF, 6) | 0.59 | 0.46 | 0.47 | 0.57 |
| Retriever: facts about try 1 | 0.48 | 0.52 | 0.64 | 0.63 |
| Retriever hidden state | 0.61 | 0.58 | 0.73 | 0.63 |
| Answerer hidden state, layer 24 | 0.72 | 0.62 | – | – |
| Answerer hidden state, layer 28 | 0.76 | 0.65 | – | – |
| Answerer attention | 0.63 | 0.66 | – | – |
| Answerer token probabilities | 0.58 | 0.54 | – | – |
| Answerer stated confidence, lengths | 0.63 | 0.54 | – | – |

| Ensemble (fresh questions) | Answerer: separation | Answerer gate vs random (points) | Retriever: separation | Retriever gate vs random, passages found (points) |
| --- | --- | --- | --- | --- |
| Best single family (reference) | 0.65 | +2.1 (−1.2 to +5.1) | 0.63 | +1.2 (−0.6 to +3.2) |
| Average of all families | 0.62 | +0.5 (−2.8 to +3.7) | 0.57 | +1.4 (−0.6 to +3.5) |
| Average of the 3 best | 0.59 | +0.1 (−3.0 to +3.3) | 0.59 | +0.8 (−1.0 to +2.7) |
| Stacked logistic regression | 0.59 | +0.0 (−3.4 to +3.3) | 0.65 | +1.0 (−0.7 to +2.9) |
| Stacked boosted trees | 0.67 | +2.3 (−0.9 to +5.5) | 0.61 | +1.2 (−0.6 to +3.2) |
| All features in one model | 0.58 | −0.4 (−3.7 to +2.8) | 0.64 | +2.3 (+0.1 to +4.8) |

### Reading

- The premise is right: the families barely overlap. Mean correlation between family scores on fresh questions is 0.20 (answerer) and 0.18 (retriever).
- But low overlap helps only if each family carries information. The question-side families (facts, text, topics) score 0.43–0.57 on fresh questions, which is chance. Adding them adds noise, not signal.
- No ensemble clearly beats the best single family (0.67 against 0.65; 0.65 against 0.63).
- One of the ten gate comparisons has an interval just above zero (+0.1 to +4.8). With ten comparisons that is what chance alone produces; on accuracy the same gate is +0.3 (−1.6 to +2.3).
- Topic modelling flipped between samples (0.59 old, 0.46 fresh on the answerer target; 0.47 old, 0.57 fresh on the retriever target), the pattern of a signal that is not there.

## Step 5f: cross-family ensembles as a full study (notebook 19)

Recorded 7 Oct 2026, 23:30. The user asked for the cross-family ensemble idea to be tested in a real notebook rather than a quick script. Notebook 19 (`19_set5f_cross_family_ensembles.ipynb`, CPU only, about 10–15 minutes) does this. The numbers below are from a full run of the same code on the same data on the assistant's side, without the sentence-embedding families (the embedding model could not be downloaded there). The Colab run adds them; results will be updated after it.

### Design

| Item | What |
| --- | --- |
| Signal families | The question: facts, words (TF-IDF), topics (6 and 12), similarity to passages, embedding. Retriever's first pass: facts about the try, hidden state at 2 layers. One cheap Llama 8B answer: hidden state at 3 places and 2 layers, attention, token probabilities, stated confidence |
| Readers per family | Linear, boosted trees, nearest neighbours (51 base models for the answerer decision, 24 for the retriever decision, without embeddings) |
| Ensembles | 13: plain and group-balanced averages, one group only, greedy selection, stacking (logistic regression, class-weighted, with SMOTE, boosted trees), all features in one model |
| Decisions | Answerer: keep the Llama 8B answer or ask Qwen 32B. Retriever: stop after one try or try twice more |
| Evaluations | Old → fresh, fresh → old, and all 400 questions cross-validated with the whole pipeline refitted in every fold |
| Named beforehand | Primary ensemble = average of every base model above chance on the training questions. Compared with: the best single base model on the training questions, one fixed hidden-state probe (as in notebooks 17 and 18), a random gate, and the cheap fixed setup |
| Intervals | Bootstrap over questions (not rows), so repeated cross-validation does not narrow them artificially |

### Results on all 400 questions, cross-validated

Answerer decision (separation of "the Llama 8B answer is no worse"; gate sends about 33–40% to Qwen 32B):

| Model | Separation | Gate vs random gate (points) | Gate vs cheap fixed setup (points) | Compute vs cheap fixed setup |
| --- | --- | --- | --- | --- |
| One fixed hidden-state probe | 0.68 | +2.2 (0.0 to +4.2) | −2.4 (−5.1 to +0.4) | −27% |
| Best single base model on training questions | 0.63 | +2.4 (+0.8 to +4.1) | −1.9 (−4.6 to +0.8) | −25% |
| Primary ensemble | 0.70 | +3.0 (+1.0 to +4.9) | −1.9 (−4.9 to +1.0) | −29% |
| Stacked logistic regression | 0.71 | +3.5 (+2.0 to +5.2) | −0.9 (−3.4 to +1.6) | −26% |
| Stacked, classes weighted equally | 0.70 | +3.8 (+2.2 to +5.4) | −0.6 (−3.0 to +1.7) | −26% |
| Stacked with SMOTE | 0.68 | +3.6 (+2.1 to +5.2) | −0.6 (−3.0 to +1.6) | −25% |
| Only the answerer's signals, averaged | 0.67 | +3.3 (+1.5 to +5.3) | −1.4 (−4.4 to +1.6) | −28% |
| Only the retriever's signals | 0.54 | −0.7 (−2.5 to +1.0) | −5.2 | −27% |
| Only the question's signals | 0.51 | −0.3 (−2.1 to +1.3) | −4.9 | −27% |

Primary ensemble against the fixed probe: +0.01 separation (−0.04 to +0.08). Against the best single base model: +0.07 (+0.01 to +0.12).

Retriever decision (separation of "the first try finds both passages"; gate retries about 17–22%):

| Model | Separation | Gate vs random, both passages found (points) | Accuracy vs cheap fixed setup (points) | Compute |
| --- | --- | --- | --- | --- |
| One fixed hidden-state probe | 0.66 | +1.3 (+0.1 to +2.7) | +0.4 (−0.9 to +1.6) | +3% |
| Primary ensemble | 0.67 | +2.5 (+1.0 to +4.1) | +0.4 (−0.8 to +1.6) | +3% |
| Only the retriever's signals, averaged | 0.70 | +3.0 (+1.5 to +4.7) | +0.6 (−0.8 to +2.0) | +3% |
| Only the question's signals | 0.55 | +0.9 (−0.3 to +2.2) | +0.6 | +3% |

### The two transfer directions (primary ensemble)

| Decision | Direction | Separation | Gate vs random (points) |
| --- | --- | --- | --- |
| Answerer | Old → fresh | 0.67 | +0.2 (−3.1 to +3.6) |
| Answerer | Fresh → old | 0.70 | +2.0 (−0.6 to +4.7) |
| Retriever | Old → fresh | 0.66 | +2.1 (+0.1 to +4.2) |
| Retriever | Fresh → old | 0.61 | +1.8 (−0.1 to +4.1) |

### Verdict by the rules named beforehand

| Question | Answerer | Retriever |
| --- | --- | --- |
| Primary ensemble beats the best single base model | Yes | No |
| Primary ensemble beats one fixed hidden-state probe | No | No |
| Primary ensemble's gate beats a random gate | Yes | Yes |
| Against the cheap fixed setup | −1.9 points (−4.9 to +1.0) at −29% compute | +0.4 points (−0.8 to +1.6) at +3% compute |

### Reading

- New and solid: with 400 questions and several readers averaged, the gates do route better than chance, by about 3 points. This corrects the earlier reading that routing skill was "not confirmed"; with 200 questions and one probe the effect was too small to see.
- Ensembling helps a little (about 0.63–0.68 for single models to about 0.70), but the gain comes from averaging several readers of the same stage's signals. The question-side families stay at chance for the answerer decision (0.51) and the retriever-side families add nothing to it (0.54). The families do not overlap (correlations near 0), but the non-overlapping ones carry no information for this decision.
- Against one fixed probe the ensemble's advantage is +0.01 to +0.03 and not distinguishable from zero. The pre-named comparator (best single model picked on training questions) turned out to be a weak one, because picking among 51 noisy candidates selects badly.
- Against the cheap fixed setup the picture is unchanged in kind: the best answerer gates cost about 0.6 to 1.9 points of accuracy for about 26–29% less compute, and the interval still allows a loss of 3 to 5 points. The retriever gate changes accuracy by +0.4 points for +3% compute.
- SMOTE and class weighting inside the stack make no material difference (0.68–0.70 against 0.71).
- For the retriever decision, question facts carry a little signal (0.60–0.64), unlike for the answerer decision.
- Caveats: 81 ensemble-by-evaluation rows, so a few intervals excluding zero are expected by chance; the fresh questions had already been used for development; the cross-validated evaluation is not an untouched test.

### Notebook 19 on Colab, with the embedding families included

Recorded 8 Oct 2026, 00:05. Run folder: `runs/Set_2__question-reading-routers/20261007-233327_cpu_ensembles`. 25 minutes on CPU. Code checksums matched. Embedding model: all-MiniLM-L6-v2. With the two embedding families there are 57 base models for the answerer decision and 30 for the retriever decision. These numbers replace the ones above; they differ only slightly.

All 400 questions, cross-validated:

| Decision | Model | Separation | Gate vs random gate (points) | Accuracy vs cheap fixed setup (points) | Compute |
| --- | --- | --- | --- | --- | --- |
| Answerer | One fixed hidden-state probe | 0.68 | +2.2 (0.0 to +4.2) | −2.4 (−5.1 to +0.4) | −27% |
| Answerer | Primary ensemble | 0.69 | +3.0 (+1.0 to +4.9) | −1.9 (−4.9 to +1.0) | −29% |
| Answerer | Stacked, classes weighted equally | 0.68 | +3.3 (+1.7 to +5.0) | −1.1 (−3.9 to +1.5) | −27% |
| Answerer | Only the answerer's signals, averaged | 0.67 | +3.3 (+1.5 to +5.3) | −1.4 (−4.4 to +1.6) | −28% |
| Answerer | Only the question's signals (with embeddings) | 0.47 | −0.4 (−2.2 to +1.2) | −5.1 | −29% |
| Retriever | One fixed hidden-state probe | 0.66 | +1.3 (+0.1 to +2.7) | +0.4 (−0.9 to +1.6) | +3% |
| Retriever | Primary ensemble | 0.67 | +2.5 (+1.0 to +4.1) | +0.5 (−0.6 to +1.6) | +3% |
| Retriever | Equal weight for question and retriever groups | 0.68 | +3.2 (+1.6 to +4.9) | +0.5 (−0.7 to +1.9) | +3% |
| Retriever | Only the retriever's signals, averaged | 0.70 | +3.0 (+1.5 to +4.7) | +0.6 (−0.8 to +2.0) | +3% |
| Retriever | Only the question's signals (with embeddings) | 0.58 | +1.2 (−0.1 to +2.6) | +0.5 | +2% |

Primary ensemble against the fixed probe: +0.01 (−0.04 to +0.07) for the answerer, +0.02 (−0.03 to +0.07) for the retriever.

| Verdict by the rules named beforehand | Answerer | Retriever |
| --- | --- | --- |
| Primary ensemble beats the best single base model | Yes | Yes |
| Primary ensemble beats one fixed hidden-state probe | No | No |
| Primary ensemble's gate beats a random gate | Yes | Yes |
| Against the cheap fixed setup | −1.9 points (−4.9 to +1.0) at −29% compute | +0.5 points (−0.6 to +1.6) at +3% compute |

What the embeddings added:

- Answerer decision: nothing. Question embedding scores 0.43–0.52 and embedding similarity 0.50–0.58. The question group as a whole falls to 0.47.
- Retriever decision: a small, consistent signal. Question-to-passage similarity by embedding scores 0.62–0.63 in all three evaluations, about the same as the retriever's own signals (0.56–0.66). This is the one place where a question-side family carries information, and it makes sense: a question whose passages are all similarly close to it is harder to retrieve for.
- Even so, combining the question group with the retriever group (0.68) does not beat the retriever group alone (0.70).

Conclusion of step 5f: unchanged from the first run. The gates route better than chance by about 3 points; ensembling adds a little through averaging, not through combining families; against the cheap fixed setup the answerer gate remains a trade-off (about a quarter less compute for 1 to 2 points of accuracy, not statistically established) and the retriever gate is neutral.

## Review of `Routing_Experiment_Ideas.docx` and a scoring audit

Recorded 8 Oct 2026, 08:30. The user uploaded `docs/Routing_Experiment_Ideas.docx`: ideas from an earlier masked-token project (five-voter gate, mixture and bundle of experts, rare margins, sequence transitions, adaptive weighting, sharpening) and ChatGPT's proposals for this project. Its first recommendation is to diagnose the remaining failures, including a quick audit of answer scoring. That audit could be run at once on existing data, so it was.

### Scoring audit: exact match is marking many correct answers wrong

All accuracy numbers in this project use exact match (EM). Examples marked wrong on the fresh questions: gold "Gardnerville", answer "Gardnerville, Nevada"; gold "Patrick Sean Payton", answer "Sean Payton"; gold "number 5", answer "5"; gold "5,124 rooms", answer "5,124".

A lenient score was defined for this check: exact match, or one answer contains the other, or token F1 of at least 0.5 (yes/no answers must match exactly). It is the assistant's own definition and may be slightly too generous.

| Measure | Old 200 | Fresh 200 |
| --- | --- | --- |
| Qwen 32B (cheap fixed setup), exact match | 63.5% | 53.5% |
| Qwen 32B, lenient | 86.5% | 76.5% |
| Llama 8B (one greedy answer), exact match | 54.0% | 48.5% |
| Llama 8B, lenient | 77.0% | 75.5% |
| Gap Qwen 32B minus Llama 8B, exact match | +9.5 | +5.0 |
| Gap, lenient | +9.5 | +1.0 |
| "Escalating helps" questions, exact match | 25 | 26 |
| "Escalating helps" questions, lenient | 25 | 15 |
| Share of exact-match "helps" cases that are not real under lenient scoring | 44% | 58% |
| Wrong by exact match but substantially right | 35 of 73 | 38 of 93 |

So roughly half of the cases the routing probes were trained to find ("Qwen 32B right, Llama 8B wrong") were differences in answer wording, not in correctness. The probes were asked to predict formatting noise.

### What changes under lenient scoring (all 400 questions, cross-validated, exploratory)

| Measure | Exact match | Lenient |
| --- | --- | --- |
| Hidden-state probe, separation of the routing target | 0.66 | 0.77 |
| Token probabilities, separation of the routing target | 0.61 | 0.72 |
| Gate sending 30% to Qwen 32B: accuracy vs random gate | +3.1 points | +3.4 points |
| Gate sending 30%: accuracy vs cheap fixed setup | −2.0 points | −0.2 points |
| Gate sending 40%: accuracy vs cheap fixed setup | −2.0 points | −0.5 points |

Under lenient scoring the same single probe almost matches the cheap fixed setup (81.2% against 81.5%) while sending only 30% of questions to Qwen 32B. No intervals were computed and both question sets are already used, so this is a lead, not a result. It suggests the ceiling near 0.65–0.70 was partly a ceiling of the label, not only of the signals.

### Where the errors come from under lenient scoring

| Situation | Old 200: Qwen 32B / Llama 8B | Fresh 200: Qwen 32B / Llama 8B |
| --- | --- | --- |
| First try found both passages | 89.8% / 82.0% | 85.4% / 83.4% |
| First try missed a passage | 69.7% / 51.5% | 49.0% / 51.0% |
| Missed a passage, then 3 tries (Qwen 32B) | 72.7% | 63.3% |

When the evidence is found, both answerers are right about 82–90% of the time and the big model adds little. Most remaining errors are retrieval misses. The gold passages are always among the 20 candidates by construction, so a miss is a selection failure, never a coverage failure.

### Label stability (old questions, 5 sampled answers per model)

| Model | All 5 right | All 5 wrong | Mixed |
| --- | --- | --- | --- |
| Qwen 32B | 56.0% | 32.0% | 12.0% |
| Llama 8B | 34.0% | 33.0% | 33.0% |

Qwen 32B is consistent across samples; the Llama 8B changes between right and wrong on a third of the questions. Only 6% of questions are stable "helps" cases (Qwen 32B right at least 4 of 5 times, Llama 8B at most 1 of 5).

### Comments on each idea in the document

| Idea | Comment | Verdict |
| --- | --- | --- |
| Diagnose failures and audit scoring | Done above. The most valuable item: it changes the reading of every earlier result | Do first; already productive |
| Benefit-targeted probes | Already tested: "helps" is the mirror image of the routing target. "Hurts" has 6 to 16 examples per set. Worth redoing only with corrected labels | Redo with lenient or judged labels |
| Five-voter lexical gate | Close to the word-similarity family already tested (0.54–0.59; embedding similarity 0.62 for the retriever). Dice and Jaccard rank identically, as the document notes | Low expected gain; cheap to add |
| Sequence-aware transitions (bridge retrieval) | This is a new intervention, not a router. It targets the real weak point (missed passages on bridge questions). Needs GPU and a new pipeline design | Most promising new direction |
| Mixture of experts with adaptive weights | With 400 questions, question-dependent weights will overfit; stacking already failed to beat a fixed probe | Not yet |
| Bundle of experts | As the document says, this is what averaging and stacking already did. Only new features would make it a new test | No new experiment by itself |
| Rare routing margins | Came from token prediction; no clear counterpart in this pipeline, and the retriever gives no passage probabilities | Low priority |
| Sharpening and calibration | Rescaling one score cannot change a ranking; thresholds here are already set on held-out scores | Low priority |
| SMOTE and complex latent signals | Tested; no gain | Closed |
| Noninferiority test with a one-sided bound | Correct, and stricter than the test used in step 5c. See sample size below | Adopt |

### Answers to the audit questions the document raises about `ensemble.py`

| Question | Answer |
| --- | --- |
| Is stacking nested? | Yes. Base scores given to the stack are out-of-fold on the training side; thresholds come from out-of-fold stack scores |
| Where is SMOTE applied? | Inside the training part of each fold only; test prevalence is untouched |
| Are feature transforms fitted on test rows? | No. TF-IDF, topics, PCA and hidden-state reduction are fitted on training rows in every fold |
| How is the bootstrap grouped? | By question, so repeated cross-validation does not narrow intervals |
| How is cost counted? | Retrieval plus the Llama 8B answer for every question, plus the Qwen 32B answer and fusion when escalated. Hidden-state reading is counted as free |

### Sample size for a noninferiority test

In step 5c the gate and the cheap fixed setup disagreed on 19 of 200 questions (9.5%). To show a loss smaller than 2 points with a one-sided 95% bound and 80% power, about 1,500 fresh questions are needed if the true difference is zero, and about 6,000 if the true loss is 1 point. At the notebook 17 rate (66 minutes per 200 questions) 1,500 questions is roughly 8 GPU hours, less if attention extraction is dropped.

### Proposed next steps

| Order | Step | Cost |
| --- | --- | --- |
| 1 | Fix the scoring: report exact match, F1 and a judged correctness label (Qwen 32B as judge on answer pairs, spot-checked by hand) for all 400 questions | About 10 minutes of GPU, or none if F1 and containment are accepted |
| 2 | Re-run notebook 19 with the corrected label as the routing target | CPU only |
| 3 | If the gate then matches the cheap fixed setup, confirm on a new untouched set sized for the noninferiority test | About 8 GPU hours |
| 4 | Separately, design bridge-aware retrieval for the questions where the first try misses a passage | New design |

## Step 5g: the ideas document tested in full (notebook 20)

Recorded 8 Oct 2026, 09:00. The user pointed out that the proposed next steps had left out most of the ideas in `Routing_Experiment_Ideas.docx` (five-voter gate, MoE, BoE, rare margins). That was the assistant's own ranking, not the document's, and none of those ideas had been tested here. Notebook 20 (`20_set5g_ideas_document.ipynb`, CPU only) tests all of them except sequence-aware transitions, which need new model runs. The numbers below are from a full run of the same code on the assistant's side, without the embedding features; the Colab run adds them.

### Step 0: corrected label

Every idea is scored twice: with exact match, and with a corrected label, "substantially right" = exact match, or token F1 of at least 0.5, or the gold answer contained in the answer; yes/no must match exactly. This rule is tighter than the one used in the audit above (it no longer accepts an answer merely because it is contained in a longer gold answer), so the corrected accuracies are about 3 points lower than reported there. A hand check of about 90 borderline answers on the fresh questions found roughly 1 in 20 accepted wrongly and a similar share rejected wrongly.

| Measure | Old 200 | Fresh 200 |
| --- | --- | --- |
| Qwen 32B: exact match / F1 / substantially right | 63.5% / 77.8% / 83.0% | 53.5% / 69.7% / 73.5% |
| Llama 8B: exact match / F1 / substantially right | 54.0% / 68.8% / 74.0% | 48.5% / 65.1% / 72.5% |
| Gap Qwen 32B minus Llama 8B: exact match / corrected | +9.5 / +9.0 points | +5.0 / +1.0 points |
| "Escalating helps": exact match / corrected | 25 / 25 | 26 / 16 |
| "Escalating hurts": exact match / corrected | 6 / 7 | 16 / 14 |
| Share of exact-match "helps" that is only wording | 40% | 54% |
| Qwen 32B right (corrected): both passages found / one missed / missed then 3 tries | 86.2% / 66.7% / 69.7% | 82.1% / 46.9% / 61.2% |

### How each idea was turned into a test

| Idea | Test | Variant named beforehand |
| --- | --- | --- |
| 1 Five-voter gate | TF-IDF cosine, Jaccard, rare-token overlap, Dice, overlap coefficient between the question and each passage picked in the first try. Evidence support = passages winning a majority of votes. Four variants: original thresholds, three voters, thresholds retuned on training questions, a learned model on the continuous scores plus coverage and redundancy | Majority vote, original thresholds |
| 2 Rare routing margins | Rarity of the question's words (training-corpus frequency), gaps between the top passage scores, and interactions | Rarity + margin + interaction |
| 3 Benefit-targeted probes | Predict "helps"; expected gain = P(helps) − P(hurts) | Expected gain |
| 4 MoE | Three experts: evidence (lexical), structure (bridges, coverage gaps, redundancy, left-out passages), confidence (retriever facts and hidden state; for the answerer also the answer probe and token probabilities). Equal weights, tuned fixed weights, question-dependent weights from a small regularised gate, and a gate that selects one expert | Question-dependent weights |
| 5 BoE | The same experts, every output kept: votes, and a joint model with disagreement, rarity, margin and named interactions | Joint model with interactions |
| 6 Adaptive weighting and sharpening | A rule (follow the most worried expert when they disagree) and per-expert temperatures and weights | Per-expert temperatures |

Each is judged on all 400 questions cross-validated and in both transfer directions: does its gate beat a random gate at the same budget; does it add to the fixed hidden-state probe; how does it compare with the cheap fixed setup.

### Headline: with the corrected label, the plain fixed probe matches the cheap fixed setup

Answerer decision, the single fixed hidden-state probe, sending about a third of questions to Qwen 32B:

| Evaluation | Scoring | Separation | Gate | Cheap fixed setup | Difference (95% interval) | Gate vs random gate | Compute |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All 400, cross-validated | Corrected | 0.78 | 79.3% | 78.3% | +1.0 (−1.4 to +3.4) | +4.1 (+2.3 to +5.9) | −27% |
| Old → fresh | Corrected | 0.77 | 76.5% | 73.5% | +3.0 (−1.0 to +7.0) | +3.7 (+1.3 to +6.4) | −30% |
| Fresh → old | Corrected | 0.75 | 81.5% | 83.0% | −1.5 (−5.0 to +2.5) | +4.4 (+1.8 to +7.1) | −29% |
| All 400, cross-validated | Exact match | 0.68 | 56.1% | 58.5% | −2.4 (−5.1 to +0.4) | +2.2 (0.0 to +4.2) | −27% |

The probe, the code and the questions are the same as in notebook 19; only the scoring changed. With exact match the gate looked like a trade-off (2.4 points lost). With the corrected label it matches the cheap fixed setup at about 27% less compute, beats a random gate in all three evaluations, and in the 400-question evaluation the lower bound of the difference (−1.4) is inside the 2-point tolerance. The separation rose from 0.68 to 0.78 because the target is no longer half wording noise.

### The six ideas (corrected label, all 400 questions, cross-validated)

Answerer decision (fixed probe: separation 0.78, gate +4.1 over random, +1.0 against the cheap fixed setup):

| Idea (named variant) | Separation | Gate vs random | Beats random in all 3 evaluations | Fixed probe + idea, minus fixed probe | Adds to the fixed probe | vs cheap fixed setup | Compute |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 Five-voter gate | 0.47 | −0.4 (−2.3 to +1.4) | No | −0.08 (−0.14 to −0.01) | No | −2.7 | −18% |
| 2 Rare routing margins | 0.58 | +1.1 (−0.2 to +2.3) | No | −0.05 (−0.11 to +0.02) | No | −3.2 | −42% |
| 3 Benefit-targeted probes | 0.76 | +3.8 (+2.0 to +5.6) | Yes | −0.01 (−0.06 to +0.02) | No | +0.8 | −26% |
| 4 MoE | 0.69 | +2.4 (+0.8 to +4.0) | Yes | −0.03 (−0.08 to +0.02) | No | −1.0 | −30% |
| 5 BoE | 0.70 | +2.7 (+1.1 to +4.4) | Yes | −0.03 (−0.06 to +0.01) | No | −0.2 | −25% |
| 6 Adaptive weighting and sharpening | 0.65 | +2.7 (+1.1 to +4.4) | Yes | −0.04 (−0.09 to 0.00) | No | −0.9 | −32% |

Retriever decision (fixed probe: separation 0.66, gate +1.3 over random on "both passages found", +0.5 accuracy against the cheap fixed setup at +3% compute):

| Idea (named variant) | Separation | Gate vs random (passages found) | Beats random in all 3 | Fixed probe + idea, minus fixed probe | Adds to the fixed probe | Accuracy vs cheap fixed setup |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Five-voter gate | 0.49 | +1.6 (+0.2 to +3.1) | Yes | −0.04 (−0.09 to 0.00) | No | +0.3 |
| 2 Rare routing margins | 0.53 | +0.1 (−0.6 to +0.8) | No | −0.05 (−0.10 to 0.00) | No | +0.5 |
| 3 Benefit-targeted probes | 0.54 | +0.5 (−0.6 to +1.8) | No | −0.03 (−0.08 to +0.01) | No | +0.1 |
| 4 MoE | 0.70 | +2.7 (+1.3 to +4.3) | Yes | +0.06 (+0.02 to +0.09) | Yes | 0.0 |
| 5 BoE | 0.69 | +2.9 (+1.4 to +4.6) | Yes | +0.04 (+0.01 to +0.08) | Yes | +0.1 |
| 6 Adaptive weighting and sharpening | 0.70 | +2.9 (+1.4 to +4.5) | Yes | +0.05 (+0.02 to +0.09) | Yes | +0.1 |

Other variants worth noting (retriever decision): MoE with equal weights 0.73 (+0.07 over the fixed probe, interval 0.00 to +0.14), the best of all; the five voters as a learned model on continuous scores 0.65, about equal to the fixed probe, +2.2 over random (+0.8 to +3.6), and adding +0.05 (+0.01 to +0.10) when combined with it.

### Reading, idea by idea

| Idea | Answerer decision | Retriever decision |
| --- | --- | --- |
| 1 Five-voter gate | No signal (0.47). Lexical overlap between question and passages says nothing about which answerer is needed | With the original thresholds, no separation (0.49). As a learned model on the continuous scores it is as good as the retriever's hidden state and adds to it |
| 2 Rare routing margins | Weak (0.58), does not beat random | No signal (0.53) |
| 3 Benefit-targeted probes | "Helps" is the mirror of the routing target, so it is the fixed probe itself. Expected gain is no better (0.76 against 0.78); "hurts" has too few examples | Worse than the simple target (0.54 against 0.66) |
| 4 MoE | Below the fixed probe (0.69), because the evidence and structure experts are at chance for this decision and dilute it. The learned gate is unstable: 0.69 cross-validated, 0.76 to 0.79 in the transfers | Mixing the three experts helps (0.70 to 0.73 against 0.66). Question-dependent weights are not better than equal weights |
| 5 BoE | Below the fixed probe (0.70) | Helps about as much as MoE (0.69 to 0.72) |
| 6 Adaptive weighting, sharpening | No gain from temperatures or the adaptive rule | Same as fixed weights |

### What this adds up to

- The biggest gain came from the first item in the document, the scoring audit, not from any routing method. Correcting the label moved the fixed probe from 0.68 to 0.78 and its gate from 2.4 points behind the cheap fixed setup to 1.0 ahead.
- For the answerer decision none of the six ideas improves on the single fixed probe. What matters there is the Llama 8B's own state after answering.
- For the retriever decision the document's ideas do help: three complementary experts (evidence, structure, confidence) combined give 0.70 to 0.73 against 0.66, and the five-voter features contribute once their thresholds are learned. But this does not turn into answer accuracy: every retriever gate is within +0.0 to +0.5 points of the cheap fixed setup, at +3% compute.
- Question-dependent weighting (the "routing" in MoE) never beat equal weights. With 400 questions there is not enough data to learn when to trust which expert.

### Caveats

- The corrected label is a rule written by the assistant after seeing that exact match was misleading; it has about 5% errors in each direction. A judged label would be firmer.
- The cheap fixed setup was chosen by AgentTTS under exact match. Under the corrected label a search might choose differently; this has not been checked.
- Both question sets are used. The 400-question result is cross-validated, not an untouched test. The two transfer directions differ (+3.0 and −1.5 points against the cheap fixed setup).
- About 250 model-by-evaluation rows were computed; roughly 12 intervals excluding zero are expected by chance. One variant per idea was named beforehand for that reason.

### Proposed next steps

| Order | Step | Cost |
| --- | --- | --- |
| 1 | Run notebook 20 on Colab so the embedding features are included and the result is on record | CPU, 10 to 15 minutes |
| 2 | Firm up the label: have Qwen 32B judge each answer against the gold answer, and hand-check a sample | About 10 minutes of GPU |
| 3 | Re-score the Set 4 sweep with the corrected label to see whether AgentTTS's pick is still the right bar | CPU |
| 4 | Confirm the answerer gate on a new untouched set with the gate, label rule and threshold frozen first, sized for a noninferiority test | About 8 GPU hours for 1,500 questions |
| 5 | Sequence-aware transitions (bridge retrieval), the one idea in the document not yet tested | New pipeline, GPU |

### Notebook 20 on Colab, with the embedding features included

Recorded 8 Oct 2026, 11:10. Run folder: `runs/Set_2__question-reading-routers/20261008-090219_cpu_ideas`. About 48 minutes of logged run time (12 minutes building features including the embeddings, 36 minutes of evaluation). Code checksums matched; embedding model all-MiniLM-L6-v2.

The results match the assistant's run to within 0.01 separation on every row. The embeddings changed only the rare-margins idea, which uses embedding similarity margins: retriever decision 0.53 → 0.58, answerer decision 0.58 → 0.57. No verdict changed.

| Check (corrected label, all 400 questions, cross-validated) | Colab run |
| --- | --- |
| Fixed probe, answerer decision: separation | 0.78 |
| Its gate: accuracy vs the cheap fixed setup | +1.0 points (−1.4 to +3.4) at −27% compute |
| Its gate vs a random gate | +4.1 points (+2.3 to +5.9) |
| Ideas that add to the fixed probe, answerer decision | None |
| Ideas that add to the fixed probe, retriever decision | MoE (+0.06), BoE (+0.04), sharpening (+0.05); best variant MoE with equal weights, 0.73 against 0.66 |
| Retriever gates vs the cheap fixed setup | +0.0 to +0.5 points of accuracy at +1% to +3% compute |

The conclusions of step 5g stand as written above.

## Where the project stands (8 October 2026)

Recorded 8 Oct 2026, 20:40, after notebook 20. A short summary of everything in Set 5, for reporting.

### The main finding

Once answers are scored for correctness rather than exact wording, a simple gate on the Llama 8B's own hidden state matches AgentTTS's fixed pick at about a quarter less compute.

| Gate on the Llama 8B's hidden state (all 400 questions, cross-validated) | Exact match | Corrected label |
| --- | --- | --- |
| Separation of "the Llama 8B answer is no worse" | 0.68 | 0.78 |
| Accuracy vs AgentTTS's pick (cheap fixed setup) | −2.4 points (−5.1 to +0.4) | +1.0 points (−1.4 to +3.4) |
| Gate vs a random gate with the same budget | +2.2 points (0.0 to +4.2) | +4.1 points (+2.3 to +5.9) |
| Compute vs AgentTTS's pick | −27% | −27% |

The change came from the scoring audit, not from a new routing method: exact match counted about half of the "Qwen 32B right, Llama 8B wrong" cases as differences when they were only differences in wording.

### What worked and what did not

| Approach | Answerer decision (keep Llama 8B or ask Qwen 32B) | Retriever decision (one try or three) |
| --- | --- | --- |
| Planner's natural-language rules over metadata (notebook 12) | Worse than the fixed setup | – |
| Cascades and learned routers, 9 families (notebook 13) | None better | None better |
| Question text, topics, embeddings, metadata (notebooks 14, 19, 20) | At chance | Question facts and embedding similarity carry a little signal |
| Hidden-state probe (notebooks 16–20) | Works: 0.78 with the corrected label; matches the fixed setup at −27% compute | Modest (0.66); beats random on passages found, no accuracy gain |
| Cross-family ensembles, stacking, SMOTE, class weighting (notebook 19) | No gain over the single probe | Small gain from averaging retriever signals |
| 1 Five-voter gate (notebook 20) | No signal | Original thresholds: none. Learned on the raw scores: as good as the probe and adds to it |
| 2 Rare routing margins | No | No |
| 3 Benefit-targeted probes | Same as the probe | No |
| 4 MoE | Below the probe | Adds to the probe (0.70; 0.73 with equal weights) |
| 5 BoE | Below the probe | Adds to the probe (0.69) |
| 6 Adaptive weighting, sharpening | Below the probe | Adds to the probe (0.70) |

For the retriever decision, the gain comes from combining evidence, structure and confidence experts, not from routing between them: question-dependent weights never beat equal weights. None of the retriever gates moves answer accuracy by more than +0.5 points.

### Ideas to keep and to drop

| Keep | Drop |
| --- | --- |
| The single hidden-state probe for the answerer decision | Rare routing margins |
| Combining evidence, structure and confidence experts for the retriever decision | Benefit targets (they reduce to the probe) |
| Five-voter scores, but learned from data rather than fixed thresholds | Question-dependent weighting, sharpening, SMOTE |
| The corrected label (and a judged label to replace it) | Exact match as the only accuracy measure |

### What still stands between this and a claim

| Open point | Why it matters | Next step | Cost |
| --- | --- | --- | --- |
| The corrected label is a rule written by the assistant | About 1 in 20 borderline answers is judged wrongly either way | Qwen 32B judges each answer against the gold answer; a sample is checked by hand | About 10 minutes of GPU |
| AgentTTS's pick was chosen under exact match | Under the corrected label a different fixed setup might be the right bar | Re-score the Set 4 sweep with the corrected label and redo the selection | CPU only |
| Both question sets have been used for development | Cross-validation is not an untouched test; the two transfer directions differ (+3.0 and −1.5 points) | Freeze the gate, label and threshold, then test on new questions sized for a noninferiority test | About 1,500 questions, roughly 8 GPU hours |
| Sequence-aware transitions (bridge retrieval) not yet tested | Most remaining errors come from a missed passage | Design a bridge-aware second retrieval step | New pipeline, GPU |

## Set 6: the next-experiments plan, CPU part (9 October 2026)

Recorded 9 Oct 2026, after running items 6, 5 and 4 (cheap variant) of 4\_2\_Next\_Routing\_Experiments\_Plan.docx on the 400 development questions (code next42.py, version 6.1). For the answerer decision, none of the three ideas adds to the single hidden-state probe. For the retriever decision, the gain seen in notebook 20 comes from the retriever's facts about its first try, not from the MoE or BoE design.

### Item 6: what the experts add to the probe

All 400 questions, cross-validated (5 folds × 2 repeats), corrected label. Separation: 0.5 = useless. Brackets: difference from the probe alone, with its 95% interval where it matters.

| Variant | Answerer decision | Retriever decision |
| --- | --- | --- |
| Probe alone (reference) | 0.78 | 0.66 |
| Probe + evidence (linear combiner) | 0.76 (−0.02) | 0.69 (+0.03) |
| Probe + structure (linear combiner) | 0.74 (−0.04) | 0.68 (+0.02) |
| Probe + evidence + structure (linear combiner) | 0.73 (−0.05) | 0.70 (+0.04) |
| Probe + evidence + structure + first-try facts (equal weights) | 0.64 (−0.14) | 0.74 (+0.08; +0.02 to +0.14) |
| MoE, notebook 20 (evidence + structure + confidence) | 0.69 (−0.09) | 0.73 (+0.07) |
| MoE without the confidence expert | 0.55 (−0.23) | 0.66 (+0.01) |
| BoE, notebook 20 | 0.67 (−0.12) | 0.72 (+0.07) |
| MoE + BoE averaged | 0.67 (−0.11) | 0.73 (+0.07) |
| Probe + support of the answer in the evidence (item 4, cheap variant) | 0.80 (+0.01; −0.03 to +0.07) | – |

- **Answerer decision:** nothing adds to the probe. The best addition, the cheap support features, moves separation by +0.01 with an interval that spans zero, and accuracy stays at +1.0 points vs the fixed setup.
- **MoE and BoE:** the confidence expert carries both. Without it MoE drops to the level of the probe (0.66 retriever, 0.55 answerer). MoE and BoE agree within 0.01 in every split, so the simpler one (equal-weight averaging) is enough. Paired errors were not computed separately.
- **Retriever decision:** the combined gate finds both passages +3.2 points more often than random (+1.6 to +4.8), against +1.3 for the probe alone. Answer accuracy moves by at most +0.6 points, so this stays a diagnostic finding.

### Item 5: stable escalation benefit

The five sampled Set 4 answers per model, on the same shortlist (old questions), show that most single-pair labels are noise-prone for the Llama 8B, and that stable gains from Qwen 32B are rare.

| Old questions, 5 answers each | All 5 right | All 5 wrong | Mixed |
| --- | --- | --- | --- |
| Qwen 32B | 75.5% | 14.0% | 10.5% |
| Llama 8B | 52.5% | 12.0% | 35.5% |

- Stable helps (Qwen rate − Llama rate ≥ 0.6): 20 of 200 questions. Stable hurts: 6.
- Of the one-pair "Qwen helps" labels, 44% are stable helps and 16% show no benefit at all.

| Gate at a 40% share (corrected label) | Old, cross-validated | Old → fresh |
| --- | --- | --- |
| Probe trained on the one-pair target (as before) | 81.0% (−2.0 vs fixed) | 76.0% (+2.5 vs fixed) |
| Regression probe for the stable benefit (item 5 primary) | 81.0% (−2.0 vs fixed) | 75.5% (+2.0 vs fixed) |

Verdict: stop. The stable target does not improve held-out decisions, and helpful cases are too rare (10%) to learn from. The fresh questions have only one Llama answer each, so they serve as a test only.

### Item 4: support verification

The cheap variant (does the Llama answer appear in the passages it was given) duplicates the probe; see the item 6 table. The LLM verifier and the blinded human check were not run.

### Items 1–3: notebook 21, ready to run (GPU)

Items 1–3 need new answers, so they run in notebook 21 (21\_set6\_next\_experiments.ipynb, A100). It reuses the saved retrieval tries, generates every missing answer, then analyses all three items and re-runs the CPU items so every result sits in one run folder.

| Plan item | Answers generated per question | Primary variant, named before running |
| --- | --- | --- |
| 1 Different remedies | Keep the Llama 8B answer; Qwen 32B on the same passages; retry retrieval and answer again | Retry if the retriever probe is low; else escalate if the answer probe is low; else keep. Both thresholds fitted on training questions within the fixed setup's compute |
| 2 Adaptive context | Llama 8B and Qwen 32B on the full retry union and on reranked shortlists of 2, 3 and 4 passages; gold passages only as a diagnostic ceiling | Probe-triggered retry, then the bounded shortlist, with k chosen on training questions |
| 3 Answer revision | Llama 8B from scratch, Llama 8B revising its first answer, Qwen 32B, all on the same updated context | One Llama 8B revision, no extra escalation |

- About 2,400 Llama 8B and 1,500 Qwen 32B greedy answers (512-token limit). My estimate is 45–90 minutes in total, including the vLLM install and model copies; it is resumable.
- Comparisons: the cheap fixed setup, the answerer-only gate, the retriever-only gate and a random policy with the same action mix, plus an oracle as an upper bound. Every action is charged the first retrieval try and the Llama 8B first answer.
- Scores: exact match first, the corrected label second, as the plan asks.
- New reference: the fixed setup with one greedy Qwen 32B answer, so all arms use the same prompt. The original fixed setup (draft plus fuse step) is reported alongside it.

## Set 6: the next-experiments plan, GPU part (notebook 21, 10 October 2026)

Recorded 11 Oct 2026. One remedy helped: when the retriever probe says the passages look incomplete, retry retrieval and let Qwen 32B answer. Retrying with Llama 8B, answer revision and shorter context did not help.

**A correction first.** Notebook 21 re-generated Llama 8B's first answer with vLLM. It differed from the answer the probe was trained on for 27% of questions, and the probe's edge over random fell to zero. The analysis was re-run keeping the answer the probe reads (`results_actions_probe_answer`). The numbers below come from that run.

### The rule that worked (development only)

1. Qwen 7B retrieves once and Llama 8B answers once.
2. If the retriever probe is low (about 25% of questions): retrieve twice more, and Qwen 32B answers on all passages picked.
3. Else, if the answer probe is low (about 15–20%): Qwen 32B answers on the first passages.
4. Otherwise keep Llama 8B's answer.

| Policy (all 400, cross-validated, corrected label) | Accuracy | Compute vs fixed setup |
| --- | --- | --- |
| Fixed setup, one plain Qwen 32B answer | 76.8% | – |
| Answerer-only gate | 76.2% | −17% |
| The rule above | 78.1% (+1.9 vs answerer-only gate; +0.5 to +3.4) | −8% |

- On exact match the rule is +1.4 over the answerer-only gate (0.0 to +2.9).
- Transfer is inconsistent: level with the answerer-only gate from old to fresh questions; +1.0 (corrected label) or −3.0 (exact match) from fresh to old.
- Replacing the plain retriever probe with the combined retriever score (0.74 separation) did not help: 78.0% on the corrected label and 56.1% on exact match.

### Context size and revision (items 2 and 3)

| After a retry | Llama 8B | Qwen 32B |
| --- | --- | --- |
| First shortlist (no retry) | 71.2% | 76.8% |
| All passages from the retry | 72.5% | 78.8% |
| Trimmed to 2 passages | 62.7% | 69.0% |
| Gold passages only (ceiling) | 78.0% | 84.5% |

- The trimming rule threw away needed passages; the full set was chosen in 68 of 72 training splits.
- On the 84 questions the probe retries, Qwen 32B on the new passages reached 75.0%; Llama 8B revising its answer, 66.7%; Llama 8B answering again, 65.5%.
- The retry recovered a missing gold passage on only 16 of those 84 questions, and fixed 5–7 answers.

**Lesson:** the main gap is retrieval. With exactly the right passages Qwen 32B gains 7.7 points, and even Llama 8B beats the fixed setup.

## Set 7: Qwen 3B learns to shortlist (notebook 22, 11 October 2026)

Recorded 11 Oct 2026. Teaching the small retriever, with LoRA, which passages answer a question made the pipeline both more accurate and cheaper than AgentTTS's pick. No router: the model learns the rule itself from HotpotQA's labels.

| Part | Setting |
| --- | --- |
| Input → target | Question plus 20 candidate passages as text → "Selected references: \[a\] \[b\]" (the gold passages) |
| Model | Qwen 2.5 3B Instruct with a LoRA adapter (r = 16, 29.9 million trainable weights), 1 epoch, 67 min on an A100 |
| Training | 3,973 hard-level questions from HotpotQA's training file, built like our questions |
| Validation | 300 more training-file questions; used only to pick the checkpoint |
| Test | The 400 questions of notebooks 17–21, never used for training or any choice |

**Validation curve:** both gold passages found 64.3% after 1,000 questions, 78.0% after 2,000, 85.3% after 4,000. It was still rising.

| Retriever (400 test questions) | Both gold passages found | Precision | Retrieval compute vs Qwen 7B |
| --- | --- | --- | --- |
| Qwen 7B, not trained (fixed setup) | 79.5% | 63% | 1.00 |
| Qwen 3B, not trained (two prompts) | 29–38% | 47–55% | 0.36–0.41 |
| Qwen 3B, fine-tuned | 84.8% | 92% | 0.36 |

| Pipeline | Corrected label | Exact match | Compute vs fixed setup |
| --- | --- | --- | --- |
| Qwen 7B retrieves → Qwen 32B answers (fixed setup) | 76.8% | 57.5% | – |
| Fine-tuned 3B → Qwen 32B | 81.0% (+4.3; +1.2 to +7.3) | 61.3% (+3.8; +1.0 to +6.8) | −42% |
| Fine-tuned 3B → Llama 8B | 74.0% (−2.7) | 52.5% (−5.0) | −69% |
| Gold passages → Qwen 32B (ceiling) | 84.5% | 63.7% | – |

- This is the first result that is both more accurate and cheaper. Unlike the routing results, the test questions played no part in any choice, and validation (85.3%) and test (84.8%) agree.
- The gain is uneven: +8.0 points on the fresh 200 (+4.0 to +12.5) but +0.5 on the old 200 (−3.5 to +4.5), where Qwen 7B already found both passages 83.5% of the time.
- The one-off training cost (67 A100 minutes) is not included in the −42%.
- Next: more training questions for the 3B model, and the same training for Qwen 7B (notebook 23).

## Set 7b: more training, and the 7B retriever (notebooks 23a and 23b, 11 October 2026)

Recorded 11 Oct 2026. Training Qwen 3B on 12,000 questions closed the retrieval gap: with Qwen 32B answering, the pipeline reaches the gold-passage ceiling at 42% less compute than AgentTTS's pick. Qwen 7B trained the same way is no more accurate and costs more to run.

| Setting | 23a | 23b |
| --- | --- | --- |
| Model | Qwen 3B | Qwen 7B |
| Training questions | 11,959 (41 too long) | 7,977 (23 too long) |
| Training time (A100) | 206 min | 204 min |

Both use the same LoRA recipe as notebook 22. The 12,000 hard-level training questions use all 15,661 hard training questions as the distractor pool.

**Validation curves** (both gold passages found):

| Training questions | 3B | 7B |
| --- | --- | --- |
| 2,000 | 84.3% | 86.7% |
| 4,000 | 86.3% | 90.3% |
| 8,000 | 89.7% | 92.7% |
| 12,000 | 90.7% | – |

| Retriever (400 test questions) | Both gold passages found | Precision | Retrieval compute vs Qwen 7B |
| --- | --- | --- | --- |
| Qwen 7B, not trained (fixed setup) | 79.5% | 63% | 1.00 |
| 3B, 4,000 questions (notebook 22) | 84.8% | 92% | 0.36 |
| 3B, 12,000 questions | 92.0% | 96% | 0.36 |
| 7B, 8,000 questions | 93.0% | 96% | 0.90 |

| Pipeline | Corrected label | Exact match | Compute vs fixed setup |
| --- | --- | --- | --- |
| Fixed setup: Qwen 7B → Qwen 32B | 76.8% | 57.5% | – |
| 3B-12k → Qwen 32B | 84.2% (+7.5; +4.7 to +10.5) | 64.0% (+6.5; +3.7 to +9.5) | −42% |
| 7B-8k → Qwen 32B | 84.5% (+7.8; +5.0 to +10.7) | 63.7% (+6.2; +3.5 to +9.0) | −15% |
| 3B-12k → Llama 8B | 77.5% (+0.8) | 55.0% (−2.5) | −69% |
| Gold passages → Qwen 32B (ceiling) | 84.5% | 63.7% | – |

- 3B-12k beats 3B-4k by +3.2 points (+1.2 to +5.5) on the same questions.
- 3B-12k and 7B-8k tie: +0.2 (−1.0 to +1.5).
- The gain now holds on both question sets: +4.5 points on the old 200 and +10.5 on the fresh 200 (corrected label).
- Retrieval is no longer the bottleneck; the remaining errors are the answerer's.
- The micro-batching change did not speed up training (about 17 minutes per 1,000 questions on 3B, as before).

## Set 7c: answer gate on top of the fine-tuned retriever (notebook 24, 11 October 2026)

Recorded 11 Oct 2026. A hidden-state gate on top of the new retriever trades accuracy for compute roughly evenly. It does not keep the full accuracy of always using Qwen 32B while saving much. Every gate setting still beats AgentTTS's pick by about 5 points at about half its compute.

**Design.**

- Llama 8B answered again on the 3B-12k shortlists. Its hidden states were saved from that same pass, so the probe reads the exact answer it judges.
- The probe, its target ("Llama's answer is no worse than Qwen's") and the escalation share were fitted inside training folds (5-fold × 5 repeats).
- It was compared with always-Qwen on the new shortlists, AgentTTS's pick, Llama alone, and a random gate at the same share.

**Little room to gain.** With the new shortlists, Llama 8B alone reaches 79.0% (corrected label) and 55.5% (exact match). The two models disagree on only 43 questions (corrected label) and 68 (exact match).

| Policy (all 400, cross-validated) | Corrected label | Exact match | Compute vs always-Qwen | Compute vs fixed setup |
| --- | --- | --- | --- | --- |
| Always Qwen 32B, new shortlists | 84.2% | 64.0% | – | −42% |
| Gate, 50% escalated | 82.4% (−1.8) | 63.0% (−1.0) | −12% | −49% |
| Gate, 40% escalated | 82.0% (−2.3) | 62.7% (−1.3) | −18% | −53% |
| Gate, 30% escalated | 81.4% (−2.8) | 62.2% (−1.8) | −25% | −57% |
| Llama 8B alone, new shortlists | 79.0% | 55.5% | −46% | −69% |
| AgentTTS's fixed pick | 76.8% | 57.5% | +74% | – |

- On exact match the gate works: separation 0.76, and +3 to +4 points over a random gate at every share, with intervals excluding zero.
- On the corrected label it is barely better than random: separation 0.66, and +0.3 to +0.7 points over random, with intervals including zero.
- The predeclared policy ("stay within 1 point of always-Qwen") saved only 2.7% compute on the corrected label, escalating 65% on average with shares varying from 40% to 90% across folds. On exact match it lost 1.7 points at −15% compute.
- Conclusion: the fine-tuned retriever is the main result. The gate is a dial between Qwen 32B answering everything and Llama 8B alone, not a free saving.
- All results are development evidence on the same 400 questions; the retriever result is the one to confirm on untouched questions.

## Set 7d: sub-1B retrievers (notebooks 25a and 25b, 13 October 2026)

Recorded 13 Oct 2026. Qwen3.5 0.8B, trained with notebook 23a's recipe, matches Qwen 3B as a retriever at about a quarter of its retrieval compute. Qwen3 0.6B is about 1 point behind.

**Design.** Same 12,000 training and 300 validation questions, LoRA settings and test questions as notebook 23a. Training took 87 min (0.6B) and 106 min (0.8B) on an A100.

| Retriever (fine-tuned, 12k questions) | Both gold passages found | Qwen 32B answers: corrected label / exact match | Compute vs the cheap fixed setup |
| --- | --- | --- | --- |
| Qwen3 0.6B | 90.5% | 83.0% / 62.5% | −58% |
| Qwen3.5 0.8B | 90.8% | 84.0% / 63.5% | −56% |
| Qwen2.5 3B (notebook 23a) | 92.0% | 84.2% / 64.0% | −42% |
| Gold passages (ceiling) | 100% | 84.5% / 63.7% | – |

- Against the 3B retriever (paired, 95% interval): 0.8B −0.3 points (−1.3 to +0.8); 0.6B −1.3 (−2.5 to 0.0) on the corrected label and −1.5 (−2.7 to −0.5) on exact match.
- With Llama 8B answering instead, all retrievers give 76.2–77.5% at 69–84% less compute.
- Best possible pick between Llama and Qwen on the 0.8B shortlists (needs the answer key): 86.8% at −79%. The remaining headroom is on the answer side.

## Set 8: 0.5B retriever and cascades (notebook 26, 13 October 2026)

Recorded 13 Oct 2026. The 0.5B retriever's own confidence is a strong signal of a bad shortlist, and escalating on it beats random escalation. It still only matches using the 0.8B alone. Answer-side escalation and votes added nothing.

**Design.** Qwen2.5 0.5B trained for two passes over the 12,000 training questions (125 min). The 0.6B, 0.8B and 3B retrievers were re-merged from notebooks 25a, 25b and 23a. All four recorded their token probability. Thresholds were fitted inside training folds (5-fold cross-validation × 5): the cheapest setting within 0.5 point of the best training accuracy. Every model that runs is charged, each Qwen sample with its full prompt. The whole notebook took 2 h 42 min.

**Retrievers alone (Qwen 32B answers).**

| Retriever | Both gold passages found | Corrected label / exact match | Compute vs the cheap fixed setup |
| --- | --- | --- | --- |
| Qwen2.5 0.5B (24,000 examples) | 82.8% | 79.5% / 60.0% | −58% |
| Qwen3 0.6B | 90.5% | 83.2% / 62.7% | −58% |
| Qwen3.5 0.8B | 90.8% | 84.0% / 63.5% | −57% |
| Qwen2.5 3B | 91.8% | 84.5% / 64.2% | −42% |
| Cheap fixed setup (Qwen 7B retrieves) | 79.5% | 76.8% / 57.5% | 0% |

The 0.5B's validation score was still rising at the end (58% → 81% → 85% both found at 4k, 12k and 24k examples). The 3B scored 84.5% here against 84.2% in notebook 23a; the 0.3-point difference comes from re-merging and re-running.

**Cascades (corrected label).**

| Policy | Accuracy | Compute vs fixed | vs random escalation, same share (95% interval) |
| --- | --- | --- | --- |
| Retrieval chain: 0.5B; unsure → 0.6B; disagree → 0.8B tie-break; Qwen 32B answers (49% escalated) | 83.5% | −55% | +1.4 (+0.6 to +2.3) |
| Same chain, always entered | 84.0% | −53% | – |
| Full cascade: retrieval chain, then Llama; unsure → Qwen; disagree → vote with 4 Qwen samples | 82.8% | −3% | +0.9 (−0.1 to +1.8) |
| Llama → Qwen gate on token probability or stated confidence | 79.0–84.0% | −50% to +4% | −0.3 to +0.1 (escalates 90–100%) |
| Qwen 32B greedy + 4 samples, majority vote (chain shortlists) | 84.0% | +105% | – |
| Best possible Llama or Qwen pick (needs the answer key) | 86.8% | −72% | – |

- Separation of right from wrong (0.5 = none): the 0.5B's token probability for its shortlist 0.90; Llama token probability 0.68; Llama stated confidence 0.66; Qwen token probability 0.69.
- Agreement is a usable confidence flag: Llama and Qwen agree on 72.5% of the chain's shortlists; Qwen is right 89.7% of the time when they agree and 69.1% when they disagree.
- Fit on one half, test on the other: the retrieval chain held (82–83%), but the share escalated swung from 49% to 15%.
- Conclusion: the small retriever knows when it is wrong; the answerers do not, so answer escalation on their confidence still sends almost everything to Qwen 32B. Notebook 27 tests whether the 0.5B can learn the route (which retriever and which answerer to start with) from labels, with ensembles, calibrated escalation and a Sure / Unsure flag at both stages.

## Set 9: learned route and Sure / Unsure (notebook 27, 13 October 2026)

Recorded 13 Oct 2026. The 0.5B did not learn useful answer routing: whether Llama will fail cannot be seen before it answers. The calibrated Sure / Unsure flag works and is honest. Answer checks after Llama answers give the first answer-side switch that beats random.

**Design.** 3,361 hard HotpotQA training questions no retriever was trained on: 2,211 taught the 0.5B two extra lines (`Shortlist with:` and `Answer with:`), 150 validated, 1,000 calibrated the scores and thresholds. Both stages ensemble and escalate while a calibrated score is below threshold. 2 h 14 min.

| Policy (400 test questions) | Corrected label | Compute vs fixed |
| --- | --- | --- |
| Learned route + ensemble + escalation, thresholds 0.90 / 0.90 | 83.8% | +38% |
| Same, thresholds chosen on calibration questions (0.95 / 0.85) | 83.5% | −5% |
| 0.8B retrieves → Qwen 32B | 84.0% | −56.5% |
| 3B retrieves → Qwen 32B | 84.5% | −42% |

- **Sure / Unsure (0.90 / 0.90):** 56% of answers flagged Sure, 91.5% of them right (target 90%); Unsure 74.0% right. Retrieval: Sure 95.1% found both, Unsure 77.4%. Scores track reality: 0.9–1.0 band predicted 92.5%, actual 91.5%.
- **The learned route collapsed** to the majority label (0.5B, Llama) on all 400 questions; 67% of training labels were that pair.
- **Diagnosis (CPU re-analysis, 13 Oct evening):** one bug, the 0.5B's default repetition penalty (1.1) in Hugging Face generation, distorted its generated text but not the probabilities used for routing. Replacing "pick the most likely" with calibrated thresholds changed nothing: calibration chose not to use the learned start, and forcing it matched random starts.
- **Signals, 400 test questions (0.5 = none):** retrieval is visible before escalating (learned route 0.81, 0.5B token probability 0.85, calibrated score 0.86); Llama's success is not (learned route 0.59, a question-text classifier 0.59 on calibration questions). After Llama answers: token probability + stated confidence 0.72; with answer checks (are the answer's words in the passages, yes/no, length) 0.76.
- **Answer checks as a switch** (Llama answers, low score → Qwen 32B): 20% switched gives 77.8% against 75.2% for random switching (Llama alone 74.2%, Qwen alone 79.0%); 400 questions, no intervals yet.
- **0.5B → 0.8B, skipping 0.6B** (notebook 26 data): 20% escalated gives 82.8% at −57.4% (random 80.5%); 0.8B alone 84.0% at −56.5%. Starting small saves compute only while fewer than 39% escalate.

## Set 10 design: a self-learnt rulebook (notebook 28, agreed 13 October 2026)

The models keep learning their tasks by backpropagation; the switching is governed by a rulebook that starts from soft rules, is refined by micro-experiments on dev questions, and is bounded by hard guardrails no experiment may change. The frozen rulebook is applied by code to the test questions.

**Layers.**

| Layer | What it is | How it changes |
| --- | --- | --- |
| Models | Retrievers (Qwen 0.5B, 0.6B, 0.8B, 3B, 7B; later 32B, 72B) and answerers (Llama 3B, 8B; later up to 70B) | Fine-tuned by backpropagation (retrievers done to 3B and 7B; answerers not yet) |
| Soft rules | IF signal < threshold THEN escalate (and to which model); ensemble settings | Tuned on dev questions by micro-experiments, in fine steps |
| Hard guardrails | Fixed limits | Never changed by experiments |

**Soft starting rules.**

| # | Stage | Rule |
| --- | --- | --- |
| R1 | Retrieval | IF the 0.5B's confidence is in its lowest 20% THEN escalate to the 0.8B (0.6B skipped); ensemble the two shortlists |
| R2 | Retrieval | IF after escalating the shortlists disagree AND the 0.8B is also unsure THEN escalate again (3B) and ensemble |
| A1 | Answer | IF Llama's answer is not yes/no AND some of its words are not in the passages THEN escalate: Qwen 32B answers; agree → accept, else Qwen samples join a majority vote |
| A2 | Answer | IF the answer is yes/no THEN keep |
| A3 | Answer | IF all its words are in the passages AND stated confidence > 95% THEN keep (Sure) |
| A4 | Answer | IF all its words are in the passages AND stated confidence ≤ 95% THEN keep (borderline; a candidate for amendment) |

**Hard guardrails.**

| # | Guardrail | Prevents |
| --- | --- | --- |
| H1 | Always escalate retrieval when the 0.5B's confidence is in its lowest 10% (40% of these find both gold passages) | Drift to never escalating |
| H2 | Always escalate an empty or "cannot be determined" answer | Accepting non-answers |
| H3 | At most 60% of questions reach Qwen 32B | Drift to always using the biggest model |
| H4 | Dev accuracy must not fall below the best fixed pipeline minus 1 point | Trading away accuracy |
| H5 | "Sure" needs a calibrated score of at least 0.85 | The flag losing its meaning |
| H6 | At most 8 soft rules, each firing on at least 20 dev questions | Rules fitted to a handful of cases |
| H7 | A threshold moves at most 0.05 (in percentile of the dev distribution) per round, and a change must hold on a second dev half | Chasing noise |

**The micro-experiment loop (dev questions only).**

1. Split the 1,000 dev questions into halves A and B.
2. Each round: code tries fine threshold steps (±0.025, ±0.05) for every soft rule; a planner LLM (Qwen 32B) reads the rulebook and evidence tables (accuracy of each model by signal band, per-rule firing and outcomes) and proposes up to 3 new rules, amendments or removals.
3. A change is accepted only if, on both halves, it is cheaper at no accuracy loss or more accurate at little extra compute; it beats a random-escalation control of the same size; and it breaks no hard guardrail.
4. Four rounds; each accepted rulebook is versioned with its evidence.
5. Test: the final rulebook is frozen and applied once to the 400 test questions, logging which rule fired for each question, with a Sure / Unsure flag on every answer.

Compared on test with: the starting rulebook, the cheap fixed setup, 0.8B → Qwen 32B, 3B → Qwen 32B, random escalation at the same rates, and the Llama-or-Qwen oracle. Uses only outputs saved by notebooks 26 and 27; the only GPU work is the planner (about 30–60 minutes).

### Step 5c: fresh questions for confirmation (GPU)

The other 200 questions of the Set 1–3 sample (100 dev, 100 test) are run for a short list of setups only, chosen from the Set 4 frontier. Proposed short list: retrievers Qwen 3B, 7B, 32B with CoT and per-attempt shortlist "paper default" and "3"; answerers Llama 8B and Qwen 32B with CoT; final shortlist "keep all" and "vote". That is 24 setups in place of 162. The policies frozen in 5a and 5b are then scored once on these questions.

### Comparisons in every table

| Row | Purpose |
| --- | --- |
| Best fixed setup, chosen on dev | The target to beat |
| AgentTTS choice | The paper's method |
| Cheapest setup | Lower anchor |
| Random router with the same route mix as the policy | Shows whether the router has skill beyond its mix of routes |
| Perfect router, and the same after removing re-run luck | Upper bound, stated honestly |

### What is recorded

**Metrics** (same columns as Set 4, plus the new ones): Policy | Accuracy | Compute (x cheapest), with the router's reading cost shown separately | Share of questions escalated | Retrieval found both | Accuracy when found | Accuracy when not found | Router confidence and its separation | Difference from best fixed with a 95% interval (paired bootstrap).

**Charts.**

| Chart | Shows |
| --- | --- |
| 1. Accuracy against compute | The fixed setups' frontier as a line, each policy as a point |
| 2. Cascade threshold curve | Accuracy and compute as the agreement threshold moves |
| 3. Learning curve | Router accuracy against number of training questions |
| 4. Route mix | Share of questions sent to each route, per policy |
| 5. Confidence calibration | Router confidence against how often it was right |
| 6. Where the gain comes from | Questions grouped as: cheap route enough / needs big / nothing works, and how each policy handled them |

**NL rules and facts.** Every rule the planner writes, with its own confidence, how many dev questions it fired on, raw and smoothed accuracy; the facts the planner proposed in B1; how often the small model followed the rules in B2. Saved as text files, JSONL and Excel sheets, as in Set 4.

**Storage.** Same layout as Set 4: one run folder, `step/` raw outputs, `results/` tables, Excel and charts, `plan/` rules and decisions, `sessions/<time>_<label>/` per session.

### Order and stopping

| After | Decision |
| --- | --- |
| 5a | Shortlist at most 3 policies. If the learning curve is still rising, note that more data would help |
| 5b | Add at most 2 question-reading policies to the shortlist, whether or not they beat the cascades, so the idea gets a fair test |
| 5c | Score the frozen shortlist once on fresh questions and report against the success test |

### Choices for you

| Choice | My default |
| --- | --- |
| Run folder name (Set 4 was `Set_1__3-modality-size`) | `Set_2__question-reading-routers` |
| Fresh questions in 5c | 200 (100 dev, 100 test) |
| Readers in B3 | Qwen 3B, 7B and 32B |
| Keep the Llama 3B as an answerer | No; it was the limit in every Set 4 pairing |
| Deciding models in B4 | Qwen 3B and 7B first; Qwen 32B only if they show a gain |
| Planners compared in 5b | Qwen 3B, 7B and 32B, one at a time (Llama 3B and 8B optional) |

| Decision | Stated by the model | Measured |
| --- | --- | --- |
| Retriever's selection (per try) | a last line `Confidence: <0-100>`: how sure it is that the selection is complete. Not asked in the paper-default prompt (no CoT, no number given), which stays untouched | probability the model gave its own output; agreement between the 3 tries |
| Answerer's answer (each draft and the fused answer) | a last line `Confidence: <0-100>`: how sure it is that the answer is correct | probability the model gave its own output; agreement between the 5 answers |
| Router's route per question | the planner writes a confidence on each rule | the classifier's probability for its chosen route; each rule's accuracy on the dev questions it fired on |
| AgentTTS planner's chosen setup | asked after the search, outside the paper's procedure | dev-accuracy lead over the runner-up it tried |

Notes: the confidence line is removed before an answer is scored and before drafts go to the fuser, so the pipeline's answers are produced as before. The answer prompts differ from Sets 1–3 by that one added line. The report gives, per model and signal, the mean confidence, the share actually correct, overconfidence, and how well the signal separates right from wrong (sheets “Confidence” and “Calibration”, chart 5). The routers may use the small pipeline's stated confidence as an extra signal in the “after small pipeline” setting. `plan/decisions.jsonl` lists every routed test question with its route, the router's confidence and whether the answer was right.

Results of Set 4 will be added here when the runs finish.

## Ideas to test later

Recorded on 3 Oct from the analysis of Sets 1–3; none of these has been run.

- **Variable shortlist length when the retriever disagrees (deferred by choice).** Give the answerer a longer final shortlist only for questions where the retriever's tries pick different passages, and a short one when they agree. This switches how much the answerer reads per question, not which model runs. To be tested after the broad sweep, on the promising pairings.
- **Union merge instead of the paper's vote merge.** Re-merging the existing retrieval outputs (no new GPU run) raised “found both gold passages” from 13% to 28% (3B), 29% to 49% (3B + CoT) and 45% to 59% (7B), with final shortlists of about 2.6 passages instead of 1.7. The effect on answer accuracy is untested.
- **Telling the retriever how many passages to pick.** The paper's prompt gives no number. A single try picks one passage 45% of the time (3B), 37% (3B + CoT) and 28% (7B), which can never contain both gold passages. In the existing data, tries that picked 3 or more were no better than tries that picked 2 (3B 25% vs 28%, 7B 60% vs 65%), so the gain from an explicit number is uncertain.
- **Routing the retrieval step.** Answer with a fixed answerer and decide per question whether the small retriever is enough, using agreement between its tries as the signal.
- **All passages to the answerer (no retrieval).** A reference point showing whether the retrieval step helps at all.

**Terms used from here on:** a *per-try selection* is what one retrieval try picks; the *final shortlist* is the single list the answerer reads; *vote merge* (paper) and *union merge* are the two rules for turning per-try selections into the final shortlist.

**Corrections to the deviations listed under Set 1** (checked against the paper and its code on 3 Oct): the paper retrieves with Qwen2.5 7B / 32B / 72B and answers with Llama 3.2-3B / 3.1-8B / 3.1-70B, all run 4-bit quantised on an 80 GB H100; it searches for 50 rounds on 50 questions and tests on 500; the agent may choose 1 to 100 tries per step; candidates are about 100 passages from Elasticsearch plus the gold ones. Ours: Qwen2.5 3B / 7B for both roles at full precision, 20 passages (2 gold + 18 BM25 distractors), 1 or 3 retrieval tries and 1 or 5 answer tries, 200 dev + 200 test questions.

## Files

| Notebook | Runs on | Does |
| --- | --- | --- |
| `00_setup_download_qwen7b.ipynb` | CPU | downloads Qwen2.5-7B-Instruct into `models/` |
| `04_step0_results_table.ipynb` | A100 | builds the data samples and the per-question table |
| `04b_add_cot_answers.ipynb` | A100 | Set 2b data: adds chain-of-thought answers to a copy of the Set 1 table (24 setups) |
| `05_script1_agenttts_global.ipynb` | A100 | AgentTTS global configuration; Set 1 (default) or, with ALLOW\_COT = True, the Set 2b control |
| `06_script2_switching.ipynb` | A100 | per-question switching: router v2 by default (Set 2a); ROUTER\_VERSION = 1 reruns Set 1 |
| `07_set2b_switching_cot.ipynb` | A100 | Set 2b: router v2 with a fourth, chain-of-thought route (L1-Lc5) |
| `08_set3_gold_answering.ipynb` | A100 | Set 3a: answerers given the two gold paragraphs (answering ceiling) |
| `09_set3_retrieval_cot.ipynb` | A100 | Set 3b: 3B retrieval with chain-of-thought, and answers on its picks |
| `10_set4_setup_models.ipynb` | CPU | Set 4: downloads Qwen2.5-32B (4-bit) and Llama-3.1-8B |
| `11_set4_sweep.ipynb` | A100 | Set 4: the 162-setup sweep, resumable; tables, Excel and charts (code in code/sweep.py) |
| `12_set4_agenttts_switching.ipynb` | A100 | Set 4: AgentTTS baseline and per-question switching, planner Qwen2.5-32B |

All four use one code file, `code/ssw.py`, copied into every run folder with its results.
