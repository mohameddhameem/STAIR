# Per-question model and sampling allocation on HotpotQA

## Objective and terminology

Measure whether different questions benefit from different model combinations and QA sampling budgets. A configuration evaluation (trial) is distinct from one sampled answer. The first experiment studies c=(retriever, answerer, k_QA), with W=Qwen2.5-1.5B Base and S=Qwen2.5-7B Base, both existing Q4_K_M weights. Retrieval uses k_R=1. The grid is WW/WS/SW/SS crossed with k_QA in {1,2,5,10,20}: twenty configurations per question, not twenty independent configuration-search agents.

This is an AgentTTS-inspired local experiment, not a numerical reproduction of the original paper's larger models. AgentTTS uses temperature 0.9 for repeated samples and uses the generating model itself for fusion. Source: https://arxiv.org/html/2508.00890v2#S3 . Our majority-vote analysis is a separate aggregation baseline.

## Model and prompt protocol

Keep the existing two Base models as the main experiment for continuity. Do not mix Base and Instruct results in one scaling curve. If instruction following is too poor, evaluate a separate W-Instruct/S-Instruct pair; the official 7B-Instruct weight is not currently part of this experiment. No replacement model or prompt is selected using test answers.

Use twenty predetermined training questions to check valid answer formatting and reference selection. The previous actual-model SS result was 18.16% answer EM. Low single-sample quality may reflect retrieval failures, Base-model instruction following, or verbose answer formatting; it does not by itself establish that more samples will help. Audit these issues before collecting a large sampling grid. Supporting-title coverage and optional gold-evidence QA are diagnostic-only analyses, never deployed features or headline system scores.

Freeze retrieval prompts, reference-number parsing, passage limit, QA prompts, output limits, and answer extraction before the validation scan. Preserve all 100 candidate passages and reject silent truncation. Use the previous actual retrieval protocol initially, including its maximum of four selected passages. If changed during training-only calibration, generate a new retrieval cache for every split and assign a new protocol version.

Retrieval is deterministic at temperature 0. QA sampling starts with temperature=0.9, top_p=0.95, explicit top_k=0 (disabled), and the same max output length for both models. These latter settings are our experimental choices, not all claimed paper settings. Maintain a separate greedy k=1 control and a stochastic k=1 control. Every sampled answer has a distinct deterministic seed derived from (question ID, answerer, retrieval branch, stream, sample index). Reusing a seed is not an independent repeat. Validation may compare temperature 0.7/0.9; freeze the selected setting before test.

## Execution and sample reuse

For each question:

1. Run W retrieval once and S retrieval once. Save the original selected passages and retrieval token counts/cost. Do not resample retrieval when comparing QA budgets.
2. On each retrieval branch, run each answerer for twenty independently seeded candidate answers. This produces four candidate pools of twenty answers.
3. Evaluate the first 1,2,5,10,20 candidates from the same pool. Prefix reuse isolates the effect of adding candidates and avoids generating 1+2+5+10+20=38 answers per configuration.
4. Perform aggregation using only the question, chosen evidence, and available candidate prefix. Gold answers are used only after aggregation for scoring.

All budgets from 1 through 20 can also be inspected with majority voting without more model generations. With the sparse fusion grid, the reported optimum is only optimal among the five tested budgets.

Aggregation analyses:

- Majority vote: normalize answer strings with a frozen rule, vote, and break ties by earliest candidate index. This is the cheap primary diagnostic, not the original AgentTTS fusion.
- AgentTTS-style fusion: for k>=2, use the same answerer model to fuse its candidates, with an identical frozen fusion prompt. Fix fusion decoding to greedy initially. For k=1, use the candidate directly. Do not choose the answer with gold labels, and do not silently give the weak configuration a strong fuser.

Run three independent twenty-answer streams on the validation pilot. Candidate sequences are nested within a stream; streams are independent. Compare the variability of the resulting best-budget choices. For fusion order sensitivity, use a small validation-only order permutation check before freezing the protocol.

## Data use and rollout

Retain the existing split: train 4814 / validation 1110 / test 1481, seed 20260729. Existing test outcomes have already been inspected in earlier experiments, so do not describe this test set as previously unseen. Treat this as a new, preregistered experiment with its own frozen protocol. Stronger confirmatory claims would require an additional external holdout or properly nested evaluation.

Stage 0: twenty training-only protocol checks, no controller fitting or test selection.
Stage 1: uniformly sample one hundred validation questions with a recorded seed, without filtering by difficulty or previous success. Obtain three independent candidate streams. This is the oracle-gap pilot.
Stage 2: if the pilot shows a useful and stable gap, collect the training sampling grid, fit controllers on train, and tune budget tradeoffs on validation. Start with a predeclared training subset for feasibility, then expand to all 4814 rows if required. Keep train/validation membership fixed.
Stage 3: freeze controllers, prompts, aggregation, and budget settings; collect the complete 1481-question test grid once and evaluate every predefined policy.

The cached full-test retrieval calls may be reused only if their retrieval protocol and model hashes match exactly. Old temperature=0 QA outputs are greedy controls, not members of the new temperature=0.9 candidate pools. Training/validation evidence from the old archive must not silently be mixed with the newer test retrieval protocol.

## Defining the per-question optimum

Store fused outcome q_i(c), answer EM, answer token F1, and inference cost C_i(c) for each question/configuration/stream. A descriptive label is the cheapest configuration that answers correctly, using a predeclared tie rule. If none succeeds, label the question unresolved; do not label it as needing twenty samples. A successful k=20 is not sufficient evidence that twenty was necessary. Outcomes need not improve monotonically with k.

Report three separate hindsight analyses:

1. Any-candidate coverage: whether at least one answer in a candidate prefix is correct. This is a gold-assisted ceiling, not the accuracy of the deployed voter or fuser.
2. Fused-configuration oracle: choose the best actually aggregated configuration per question; tie-break by cost. This is a retrospective oracle, not a deployable controller.
3. Budget-constrained oracle: maximize sum_i q_i(c_i), subject to sum_i C_i(c_i)<=B, choosing exactly one configuration per question. Compute a multiple-choice knapsack optimum using documented cost discretization (round costs up), or provide feasible integer solutions and optimization bounds. Do not call a greedy allocation an exact oracle.

For lambda curves, additionally compute c_i*(lambda)=argmax_c[q_i(c)-lambda*C_i(c)] and report aggregate EM/cost. This does not automatically give an exact optimum at every hard budget. Include fixed configurations and budget-matched randomized mixtures in the baseline frontier.

Choosing configurations and scoring them on the same sampled pool is optimistically biased. Use stream A to identify the per-question preferred configuration and evaluate that fixed choice in streams B/C. Report this cross-stream transfer separately from the in-pool hindsight ceiling. Per-question empirical maxima remain estimates, not the true optimum of expected performance.

## Controllers and baselines

- Fixed configurations: all twenty W/S/k configurations, including greedy k=1 controls.
- Global allocation: choose one configuration on validation at each budget. Call it an AgentTTS-style fixed-budget baseline unless the original search procedure is actually reproduced.
- Random allocation: choose configurations without per-question information; match both model mix and total compute budget, not just mean k.
- Existing W/S router with fixed k: separates model-routing benefit from adaptive-sampling benefit.
- Non-LLM adaptive budget: choose QA model and k with lightweight features. Compare with the existing classifier family after retraining on the new sampling outcomes.
- Consistency stopping: after k=2,5,10,20 inspect normalized answer agreement; thresholds selected on validation. Stop based on observable candidates, never on gold correctness.
- LLM budget prediction: once the oracle gap is established, predict QA model/k from question and available evidence; charge its full decision cost.
- Oracle curves as clearly labeled upper-bound diagnostics.

Deployment is sequential. The first controller chooses retrieval W/S using question and pool only. After that retriever runs, the second controller sees only the selected evidence and chooses answerer W/S and k. A real controller cannot use both retrieved evidence sets for free. If it evaluates both, charge both retrieval calls. An offline oracle may inspect all branches, but must not be presented as a pre-execution policy.

Do not retrain a classifier using test-derived optimum labels. Keep retrospective test-optimum plots separate from deployable policy evaluation.

## Cost and reporting

C_total = C_retrieval + C_candidate_generation + C_aggregation + C_controller + C_any_extra_retrieval.

Do not use the old WW=2.0/WS=5.7/SW=5.7/SS=9.4 constants unchanged for k>1. Track input/output tokens and model parameters. A parameter-weighted token proxy may supplement measurement, but is not exact FLOPs. Token counts alone do not equate weak and strong computation. Measure warmed single-request latency on the same idle GPU; record cache behavior and model-load time separately. Batched collection throughput is not online routed latency. Charge hypothetical deployed policies for the work they would execute, even when the experimental grid is cached offline.

Report EM/F1, cost distributions, model/k selections, accuracy-cost curves, and paired question-bootstrap intervals. Match costs when comparing W-many-samples with S-few-samples. Break down outcomes by question type, retrieval gold-passage coverage (post hoc), and answer disagreement. Identify questions that improve then degrade, never improve, or improve only after changing models.

Per-question artifacts include question ID/split, reference choices, candidate text/seed, fused text, normalized answer, EM/F1, token counts, latency, chosen configuration, and protocol/model hashes.

## Compute volume

For N questions and one candidate stream, the four-combination grid needs 80N QA generations and at most 2N retrieval calls. Five majority-vote budgets need no extra model calls. Same-model fusion for the four k>1 budgets adds 16N calls.

One-hundred-question pilot with three QA streams: 24000 QA generation calls + 4800 fusion calls + 200 deterministic retrieval calls = 29000 calls. Retrieval results are shared across streams.

All 7405 rows, one stream: 592400 QA generation calls + 118480 fusion calls + 14810 retrieval calls = 725690 calls before greedy controls and controller measurements. Complete test alone, one stream: 118480 QA generations + 23696 fusion calls + 2962 retrieval calls = 145138 calls. These are collection costs; each deployed policy executes only its selected configuration.

Do not commit to a wall-clock estimate from the old greedy benchmark. Measure the twenty-question setup's candidate length, fusion latency, prefill-cache effects and throughput, then estimate GPU-hours and disk usage. Run the oracle pilot before the full grid.

## serverC resource policy

Use only serverC (SMU_C_4A5000, 10.193.104.142), preferably the previously authorized GPU 3. Load models sequentially on one card. Continue SQLite/JSONL checkpointing and content hashes. Refuse startup if another compute process occupies that GPU, or estimated memory plus reserve is unavailable; inform the user instead of automatically waiting/retrying. Never terminate another experiment.

Resource observation on 2026-10-11: all four cards have existing SGLang processes; free memory is roughly 2.5-2.8 GiB per card. No sampling model or new benchmark has been started. A card must become available before execution.
