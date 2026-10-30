# Non-LLM routing results — 31 October 2026

All completed results are catalogued below. [914-row cross-cohort index](all_results.csv) · [artifact inventory](artifact_inventory.json) · [recent publication manifest](recent_publication.json). Rows include controls, seeds and operating points; they are not 914 independent experiments. Dataset, accuracy ruler and cost convention are explicit in the index. Do not pool different cohorts.

## MedAgentBench: GRPO and seven classical families

[English comparison PNG](figures/medagentbench/medagentbench_nonllm_vs_jonathan.png) · [PDF](figures/medagentbench/medagentbench_nonllm_vs_jonathan.pdf) · [editable SVG](figures/medagentbench/medagentbench_nonllm_vs_jonathan.svg) · [GRPO report](medagentbench_grpo_v1/REPORT.md) · [classical report](medagentbench_classical_v1/REPORT.md).

The same Jonathan campaign cache (`3ac054d`), view fields, Qwen7B/72B executor outcomes, benchmark pass/fail grader and measured costs are reused. Train90 / validation60 / test150 have30/19/49 patients, with zero overlap at all three boundaries. Router models actually run on test inputs; task execution is replayed. No new medical episodes are generated. Costs below use full weak-first always-escalate on the same150 tasks as100%; its accuracy is54.00%. This differs from HotpotQA's Table5/direct-SS convention.

**Matched-risk table:** seed17; non-LLM thresholds selected on validation reward; Jonathan uses its fixed theta.50; every row applies the same cached tau>=.8 forced-escalation gate. Entries are accuracy / cost.

| Router / training | Full-attempt review (T1) | Round2 review (T3) |
|---|---:|---:|
| Jonathan LLM + LoRA / GRPO A1 |61.33% /80.99%|63.33% /81.47%|
| Linear / GRPO A1 |44.00% /71.72%|63.33% /81.47%|
| MLP64 / GRPO A1 |61.33% /80.99%|63.33% /81.47%|
| MLP128→64 / GRPO A1 |61.33% /80.99%|61.33% /71.39%|
| Random Forest / supervised |61.33% /80.99%|63.33% /81.47%|
| Extra Trees / supervised |61.33% /80.99%|63.33% /81.47%|
| Decision Tree / supervised |61.33% /80.99%|61.33% /71.39%|
| Linear SVM / supervised |61.33% /81.58%|63.33% /77.68%|
| RBF SVM / supervised |61.33% /81.58%|63.33% /77.68%|
| Logistic Regression / supervised |61.33% /81.58%|63.33% /77.68%|
| Multinomial NB / supervised |61.33% /80.99%|63.33% /81.47%|

Jonathan's published **ungated** T1 is61.33%/54.11%; the common gate raises its cost to80.99% without changing accuracy. Ungated T3 remains63.33%/81.47%. Do not compare gated and ungated costs as if they use the same policy constraint.

The GRPO study has24 fits: three architectures, two decision times, seeds17/0/1/2. It follows the latest A1 update: reward minus group mean, **no std division, KL or PPO clipping**; entropy.03, fresh classifier, lr1e-5, accumulation8, clip1,40epochs and validation reward checkpoints. The small classifiers have no pretrained LLM knowledge. Their repeat seeds vary substantially, so one matching seed is not a stable reproduction claim.

The classical study fits200 candidate configurations and locks56 classifiers. It is **not GRPO**: target=argmax action reward and weight=absolute reward gap. SVM calibration is patient-grouped and refits the entire feature pipeline within each fold. Hyperparameters and thresholds freeze before test inference. Linear/RBF SVM and logistic regression reproduce63.33%/77.68% across independent seeds0/1/2 at the gated validation-selected T3 point. The3.79 cost-point saving comes from skipping ineffective strong upgrades on15 task7 cases where both models fail; those failures remain. This is a compute result, not improved clinical outcomes or proof of transfer to other task types.

Full per-seed/fixed-threshold/validation-threshold/risk-gate readouts: [GRPO results](medagentbench_grpo_v1/results.csv), [classical results](medagentbench_classical_v1/results.csv), [GRPO task rows](medagentbench_grpo_v1/per_task.csv.gz), [classical task rows](medagentbench_classical_v1/per_task.csv.gz), [paired patient intervals](medagentbench_classical_v1/paired_patient_comparisons.json), [source-matching LLM readouts](medagentbench_grpo_v1/llm_reference.json), [safe minimal audit inputs](medagentbench_replay_inputs/README.md), [GRPO protocol/code](../experiments/medagentbench/README.md), [classical protocol/code](../experiments/medagentbench/CLASSICAL.md).

Risk-score generation and decision-model compute are excluded from the supplied cost proxy for both families. CPU feature+batch-decision timing is separate and is not online per-request latency. Published test results were already known; all new selections use validation but the comparisons remain exploratory. Training5's human-handoff action and its human-cost assumptions are outside these two-action studies.

## HotpotQA: Jonathan-cache alignment, model variants and budget points

[Latest English PNG](figures/hotpotqa_v2/nonllm_jonathan_aligned_comparison_v2.png) · [PDF](figures/hotpotqa_v2/nonllm_jonathan_aligned_comparison_v2.pdf) · [initial alignment report](hotpotqa_jonathan_aligned_v1/REPORT.md) · [nine model/recipe rows](hotpotqa_model_comparison_v1/results.csv) · [18 operating points](hotpotqa_budget_points_v2/results.csv).

These use Jonathan's pinned Config B outputs (`5b69cce`) and the shared4814/1110/1481 split. All29,620 cached four-path verdicts match the shared standard lenient scorer. Direct SS is76.77% at929 Table5 budget; weak-first SS costs1009, or108.61%, because all weak calls are charged. This is a **fixed-cache router comparison**, not an independent reproduction of the executor stack. Subsequent source clarification says the July answering prompt added a short-answer instruction and the cluster's AWQ vs bitsandbytes loading is unconfirmed; matching cached numbers does not establish exact AgentTTS prompts or quantization.

Accuracy-first points, selected by validation accuracy (accuracy / cost% direct SS):

| Non-LLM model | SFT | SFT + exact reward, both stages | SFT → sampled stage GRPO |
|---|---:|---:|---:|
| Linear softmax |76.50 /107.78|76.64 /107.94|76.03 /106.60|
| MLP64 |76.84 /106.76|76.30 /106.87|76.70 /108.21|
| MLP128→64 |76.84 /107.46|76.84 /106.75|76.77 /108.61|

The neural variants add36 fits to the initial42 alignment fits. All use three-seed17/18/19 probability ensembles. Accuracy-first policies almost always upgrade; the image also shows validation SS-budget points. MLP128→64 SFT matches76.77% at98.18% cost; its SFT+reward counterpart matches76.77% at98.33%. A76.84% result is one extra correct question, not stable evidence of an accuracy gain. A validation budget is not a guaranteed test budget.

The reference SFT row is76.70%/89.86%; QA-only reward is75.96%/71.56%. Both-stage **exact expected reward** gives76.23%/76.98% excluding router calls; this matches the slide's accuracy but not its80.5% cost. Sampled stage GRPO here is an explicitly documented adaptation, not a claimed reproduction of that slide row. Its HotpotQA group-standardized/KL recipe differs from MedAgentBench's later A1 recipe. Historical four-action query-only arms use a different old reward surrogate and remain auxiliary.

[All30 initial policies](hotpotqa_jonathan_aligned_v1/results.csv) · [44,430 initial policy/question rows](hotpotqa_jonathan_aligned_v1/per_question.csv.gz) · [13,329 model-comparison rows](hotpotqa_model_comparison_v1/per_question.csv.gz) · [26,658 operating-point rows](hotpotqa_budget_points_v2/per_question.csv.gz) · [alignment method details](../experiments/menu_ws_gguf/JONATHAN_ALIGNED.md). Raw views/answers, fitted feature vocabularies and weights are not published.

The remaining sections retain the earlier independently generated GGUF and historical cohorts. Their baselines, scoring and costs must remain separate from the new frozen-cache comparisons.

## Four further studies: pairs, OOF stacking, risk calibration, cascades

[Full report](menu_ws_next_v1/REPORT.md) · [accuracy/cost figure](menu_ws_next_v1/accuracy_cost.png) · [57,000 policy/question rows](menu_ws_next_v1/results.csv.gz) · [method table](menu_ws_next_v1/method_summary.csv) · [risk table](menu_ws_next_v1/risk_summary.csv) · [fold audit](menu_ws_next_v1/fold_audits.json) · [code and protocol](../experiments/menu_ws_gguf/NEXT_STUDIES.md).

New models use5125 fitting plus1280 held-out calibration IDs within the original6405; validation500/test500 and the executor outputs remain unchanged. Raw vocabulary-free features avoid any globally fitted vocabulary/scaler. Five outer folds and three inner folds provide strictly isolated base and stacked predictions; the final stack uses the same inner3-fold procedure. All model/weight choices freeze before calibration, and all risk thresholds before test construction. This is a smaller fitting-label budget than earlier experiments.

| Study / validation or OOF selection | Test lenient | Cost % direct SS |
|---|---:|---:|
| Direct SS |71.8%|100.00%|
| Four-path paired classifier, validation max accuracy |71.8%|100.00%|
| OOF stack, per-family validation max accuracy |72.0%|90.70%|
| Global OOF cheapest matching OOF SS accuracy |71.8%|89.77%|
| Stack joint plus alpha0.5% calibration gate |72.0%|91.36%|
| Weak-first cascade, validation max accuracy |71.8%|98.21%|
| Weak-first cascade, validation95%-SS target |71.4%|74.13%|
| Always-escalate cascade, cost control |71.8%|109.61%|

The stack's accuracy improvement is one question (360 vs359), not a stable-gain claim. Pairwise accuracy-focused routing still selects SS throughout test. Global OOF max accuracy selects a cascade at71.8%/95.17%; global validation max accuracy selects another cascade weight at71.8%/98.21%. The89.77% cost row uses the distinct OOF match-SS objective, which selected outcome_joint. All choices are shown, not just the best test row.

Risk calibration targets additional harm relative to SS across all queries, not overall answer error or conditional error among switchers. Calibration criteria pass, but some observed test harm rates exceed nominal alpha; expected-risk criteria are not finite-test caps, and previous use of these calibration examples makes this exploratory, not certified conformal risk control. [All calibration/observed risk rows](menu_ws_next_v1/risk_summary.csv).

Cascade retrieval observes only executed weak retrieval before deciding to upgrade. QA observes the executed weak answer on the chosen retrieval branch. Every weak/strong call is charged; the always-escalate control verifies why a cascade can exceed100% direct SS cost. Full online latency is unmeasured in this cache replay. No new executor inference or cluster job was launched. The57,000 records are114 policies on the same500 questions.

## Further alternatives: eight completed families

[Full alternatives report](menu_ws_alternatives_v1/REPORT.md) · [accuracy/cost plot](menu_ws_alternatives_v1/accuracy_cost.png) · [compact method table](menu_ws_alternatives_v1/method_summary.csv) · [38,000 policy/question rows](menu_ws_alternatives_v1/results.csv.gz) · [validation lock](menu_ws_alternatives_v1/validation_lock.json) · [method code](../experiments/menu_ws_gguf/ALTERNATIVES.md).

All numbers use the same corrected executor cache and500-question cohort as the GRPO experiment. There are54 neural streams and24 tree configurations, three seeds, no new executor sampling. The added features are candidate/selected-evidence lexical statistics; none use answers or labels as deployment inputs. Full online latency was not measured in this vectorized replay; ms_router is left empty.

| Per-family validation-selected maximum-accuracy policy | Test lenient | Cost / SS | Difference vs SS |
|---|---:|---:|---|
| SS / previous GRPO |71.8%|100.00%|reference|
| Success prediction, ExtraTrees |72.4%|96.77%|three more correct|
| PPO with exact expected-reward baseline, no std division |72.0%|96.54%|one more correct|
| Exact reward with added lexical features |71.8%|89.25%|same total correct; two gains/two losses|

These are observed per-family results, not a stable-improvement claim. The outcome-tree policy makes47 non-SS choices; the exact-baseline PPO makes40. The descriptive paired accuracy intervals include zero, and the test has been repeatedly inspected. No equivalence or confirmatory significance claim is made.

**Global method-family selection on validation differs from choosing the best test row:** the maximum-accuracy winner is the linear outcome predictor, with test71.2% and85.61% SS cost. The validation cheapest-at95%-SS winner is the gain tree, with test69.2% and50.95% SS cost. The validation cheapest-matching-SS winner is group-centered PPO, with test70.8% and70.70% SS cost; its validation accuracy constraint does not carry over automatically to test. Full per-family and global selections are both reported.

The three principal changes are predicting outcomes rather than a cheapest-correct action, predicting the benefit of strong retrieval with QA held strong, and retaining reward magnitude in policy gradients instead of group std normalization. Extra lexical features change the representation too, so this is not a causal isolation of one GRPO implementation detail. See the report for all eight families and four operating-point objectives.

The38,000 rows are76 policies evaluated on the same500 questions, not38,000 independent test examples. The archive directory keeps its original23 October name for stable links; the catalog is updated24 October.

## New corrected-cache GRPO rerun

[Full GRPO report](menu_ws_grpo_v1/REPORT.md) · [summary](menu_ws_grpo_v1/results.json) · [10,500 per-question rows](menu_ws_grpo_v1/results.csv.gz) · [validation lock](menu_ws_grpo_v1/validation_lock.json) · [training curves](menu_ws_grpo_v1/learning_curves.png) · [verification](menu_ws_grpo_v1/verification.json) · [method/code](../experiments/menu_ws_gguf/GRPO.md).

| Policy | Test lenient | Cost / SS | Test routing |
|---|---:|---:|---|
| Fixed SS |71.8%|100.00%|500 SS|
| GRPO, learned policy |71.8%|100.00%|500 SS|
| Exact expected-reward control |71.4%|83.79%|96 weak retrieval + strong QA;404 SS|

24 GRPO and12 exact-control streams, three seeds0/1/2, maximum300 epochs, validation early stopping. These update small linear TF-IDF/numeric routing heads, not Qwen/Llama weights. G8/G32 and lambda0/.05/.15/.4 were predeclared. The selected GRPO uses G8/lambda0; selected epochs260/260/290. Three-seed ensembles are used at deployment.

**GRPO learned to choose SS for every test question.** This is not the fixed fallback hiding its output: learned-only and fixed-fallback selections are both published. It matches SS without executor savings and adds about10ms CPU decision time. Exact expected reward enumerates the same four actions without route-sampling noise; it saves16.21% executor cost with two fewer correct answers (357 vs359). Four questions improve and six worsen relative to SS. The paired descriptive95% accuracy-difference interval is[-1.6,+0.8]pp; this is not a formal equivalence claim.

All scoring is lenient; "exact" names the training objective, not exact-match grading. The reward subtracts actual chosen-call units normalized by training-mean direct SS. No mandatory weak-first sunk calls are charged. No new LLM outputs were generated. This remains the same common0.2 test500 GGUF cohort, not a common0.3 golden-set result. All selections precede this rerun's test construction, although the test was previously inspected.

## Corrected four-model experiment

[Full report](menu_ws_gguf_v2/REPORT.md) · [summary JSON](menu_ws_gguf_v2/results.json) · [all 29,000 per-question policy rows](menu_ws_gguf_v2/results.csv.gz) · [validation selections](menu_ws_gguf_v2/validation_lock.json) · [fit records](menu_ws_gguf_v2/fit_records.json) · [protocol](menu_ws_gguf_v2/protocol.json) · [model repositories, revisions and hashes](menu_ws_gguf_v2/model_manifest.json).

| Retrieval | Answering | Lenient accuracy | Cost / direct SS |
|---|---|---:|---:|
| Qwen2.5-7B-Instruct | Llama-3.2-3B-Instruct | 54.0% | 9.62% |
| Qwen2.5-7B-Instruct | Llama-3.1-70B-Instruct | 62.2% | 12.60% |
| Qwen2.5-72B-Instruct | Llama-3.2-3B-Instruct | 65.8% | 97.26% |
| Qwen2.5-72B-Instruct | Llama-3.1-70B-Instruct (SS) | **71.8%** | **100.00%** |

- HotpotQA common0.2 IDs: `train_extra` 6,405, `val` 500, `test` 500. Published per-question rows are test only. The test has previously been examined, so comparisons are descriptive.
- Four executors, two choices per stage: a weak/strong subset of AgentTTS's complete three-by-three candidate menu. Q4_K_M GGUF weights; Q8 KV; llama.cpp; K=1, greedy, maximum512 output tokens, no fusion; frozen local reference-pool and prompt contract. Executors are frozen, not fine-tuned.
- Author lenient script; actual prompt/generated tokens priced with `stair_common.cost.units`. Direct SS averages **1,574.983 units**. CPU router timing is separate; CPU/training work is excluded from the parameter-FLOP proxy.
- Six classical families: majority, logistic regression, calibrated linear SVM, multinomial NB, random forest, extra trees. Router seeds0/1/2 give18 two-stage fits. Hyperparameters and operating points are chosen on validation before test scoring.
- The search includes four fixed-policy fallbacks. **Every validation-selected operating point fell back to fixed SS.** The54 family/seed/objective rows repeat this selected control; they are not54 successful learned routing policies. This experiment shows no routing improvement over SS. It does not publish a separate test table of the unselected raw classifier policies.
- A separate corrected-cache GRPO rerun is now complete; see the new section above. The historical GRPO cohort remains separate.
- GGUF copies and this execution contract are not eligible for pooled BF16/AWQ headline comparisons without team agreement. Matching the model names and scorer alone is insufficient. K=1 is not the original AgentTTS40-answer-sample recipe.

## Original protocol boundary: common0.3 vs the earlier common0.2 GGUF run

At that run's freeze point, upstream `main` (`00ee140`, common0.3) adopted Jonathan's `slr-train/slr-val/slr-test` (4,814/1,110/1,481) and Table5 prices for the HotpotQA golden set. It retained the500-question test as a paper comparison. The GGUF run was frozen under earlier common0.2; its token-based units must not be relabeled as Table5 prices and its71.8% is not directly comparable to the76.77% reference. The new Jonathan-cache studies above separately use the SLR split and Table5 convention; they do not regenerate executor outputs.

The [evaluation snapshot](menu_ws_gguf_v2/evaluation_snapshot/SOURCE.md) preserves the exact original scorer/cost/split implementation. Syncing common code does not turn an older experiment into a new-protocol reproduction. The old GGUF sections make no golden-set executor-reproduction claim.

## Historical methods and results

[Machine-readable result index](all_results.csv) contains914 rows across explicitly named cohorts. [Artifact inventory](artifact_inventory.json) maps39 run directories to published reports and source hashes. The archive also contains diagnostic/preparation runs; a run directory is not necessarily a completed experiment.

| Cohort | What was tested | Baseline / representative result | Primary artifacts |
|---|---|---|---|
| Original supplied outcome grid | Three-classifier direct routing; fixed grid-label replay; direct pre-router | Original SS **52.79% validation**, **53.54% test** (586/1,110; 793/1,481). These are different splits, not competing reproductions of one number. | [Original baseline](measured_v1/REPORT.md), [reference-grid pre-router](archive_20261023/nonllm_reference_prerouter_v1/REPORT.md) |
| Regenerated Instruct cohort, reconstructed lenient | Direct stage routing; weak-first cascade; weak-answer ablation; confidence/outcome/gain models; multi-sample augmentation and consensus; exact expected reward; finite-action GRPO; architecture variants | SS **52.94%** on1,481 test questions.73 method/operating-point records. Reconstructed scorer agrees with available review records but differs from the later shared author script. | [All73 records](archive_20261023/nonllm_reference_lenient_v1/REPORT.md), [all classifier cascade families](archive_20261023/nonllm_cascade_family_comparison_v1/results/REPORT.md) |
| Historical GRPO extension | Base/confidence/5-sample/9-sample observations; group sizes and reward penalties; up to120 epochs | Aug5 GRPO max-accuracy **52.73%**, cost **119.78% SS**; no stable improvement over SS. | [Extended report](archive_20261023/nonllm_reference_lenient_v1/extended/REPORT.md), [training plots](archive_20261023/nonllm_grpo_extended_v1/learning_curves.png) |
| Historical300-epoch continuation |54 seed streams; base/aug9; exact expected reward/G16/G32; validation-selected checkpoints | Base GRPO max-accuracy **51.86% /119.92% SS**; budget-selected **51.86% /100.16% SS**. A validation budget does not guarantee the same test budget. | [Report](archive_20261023/nonllm_lenient_longer_v1/REPORT.md), [120→300 comparison](archive_20261023/nonllm_lenient_longer_v1/COMPARISON.md), metric curves under `training/` |
| Historical architecture variants | Direct four-path pre-router, two-stage pre-router, fixed-strong-QA routing, post-answer chooser, combined constraints | Direct four-path cheap95 **52.19% /93.08% SS**; post-answer chooser **53.00% /100.56% SS**, only a one-question increase, not a stable gain. | [Architecture](archive_20261023/nonllm_reference_lenient_v1/architecture/REPORT.md), [combined](archive_20261023/nonllm_reference_lenient_v1/combined/REPORT.md) |
| Early Base-model / strict-EM / partial adaptive sampling | Exploratory or diagnostic evaluations; some stopped or partial | Not aligned with the corrected four-model lenient cohort or common0.3. Preserve provenance; do not promote to headline results. | [Archive inventory](artifact_inventory.json), existing `adaptive_sampling*`, `full_pipeline_test_v1`, and archived strict-EM reports |

The earlier regenerated cohort uses Qwen2.5-1.5B/7B-Instruct executors, unlike the current Qwen7B/72B retrieval and Llama3B/70B answering cohort.

Historical GRPO is our finite-action non-LLM adaptation, not a GRPO implementation supplied by the reference LLM-router code. Its task reward is correctness minus a weighted cost normalized by direct SS, with training regularization. Weak-first cascades pay for weak calls before any strong escalation; they can exceed direct SS cost even when they finish with both strong executors. `exact` in the policy-training names means exact expected reward, not exact-match answer grading.

The [two English method slides](historical_slides/README.md) are historical, not current-model results. Scorer and baseline changes explain why these tables must remain separate.

## Invalidated results and engineering audits

- The first four-model run had a Llama70 template defect: an empty tool list triggered injected function-call instructions. Its **25.0% SS and all selected router comparisons are invalidated**, not an alternative baseline. [Invalidation notice](archive_20261023/nonllm_common_ws_gguf_v1/INVALID_RESULTS.md).
- Explicit `tools=null` in rendering and generation fixes the defect. All14,810 Llama70 outputs were regenerated and all router fits rebuilt. [Root-cause audit](audits/nonllm_ws_template_audit_v1/REPORT.md).
- Concurrency increased from1 to4 and then16 requests per two-GPU replica; two replicas use four local GPUs. Shared KV admission protects the16K context and original512-token output limit. Batch execution is not bitwise identical to serial; each new record identifies its concurrency. This is a local execution schedule, not a four-GPU cluster-job request. [First benchmark](audits/nonllm_ws_concurrency_audit_v1/REPORT.md), [higher-concurrency benchmark](audits/nonllm_ws_concurrency_audit_v2/REPORT.md).
- The six-model menu run was intentionally stopped when the requested menu changed to four models. It has no completed six-model router result. Model downloads and preparation/pilot outputs are not trained routing results.

## Code, verification and omitted large assets

[Current experiment code](../experiments/menu_ws_gguf/README.md) · [historical methods](../experiments/router_improvements/README.md) · [result verifier](verify_published_results.py).

From the STAIR repository root, in an environment with `common` and `nonLLM` dependencies installed:

```bash
python nonLLM/results/verify_published_results.py
python nonLLM/experiments/menu_ws_gguf/test_contract.py
```

Weights, checkpoints, raw generation caches and private/local datasets are not uploaded. Compact per-question outputs, summaries, protocols, model hashes, validation locks, fit records and historical metric curves are included. Source and published hashes identify sanitized metadata; host-specific paths are replaced by workspace placeholders. The original large assets remain local. Historical test-policy NPZ arrays are output artifacts, not model weights.
