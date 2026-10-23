# Non-LLM routing results — 23 October 2026

The corrected four-model experiment is complete. Earlier experiments are retained as separate cohorts. Do not pool their accuracy or cost values: models, question IDs, scoring implementations and cost rules differ.

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
- GRPO has **not** been rerun on this four-model cohort. Published GRPO results below belong to older experiments.
- GGUF copies and this execution contract are not eligible for pooled BF16/AWQ headline comparisons without team agreement. Matching the model names and scorer alone is insufficient. K=1 is not the original AgentTTS40-answer-sample recipe.

## Important: upstream common0.3 is a different headline protocol

The latest upstream `main` (`00ee140`, common0.3) adopts Jonathan's `slr-train/slr-val/slr-test` (4,814/1,110/1,481) and Table5 prices for the HotpotQA golden set. It retains the500-question test as a paper comparison. This completed run was frozen under the earlier common0.2 protocol. Its token-based units must not be relabeled as Table5 prices, and its71.8% must not be compared directly with the upstream76.77% golden reference.

The [evaluation snapshot](menu_ws_gguf_v2/evaluation_snapshot/SOURCE.md) preserves the exact original scorer/cost/split implementation. Syncing the new common code does not turn an older experiment into a new-protocol reproduction. No common0.3 golden-set non-LLM rerun is claimed here.

## Historical methods and results

[Machine-readable result index](all_results.csv) contains144 rows across explicitly named cohorts. [Artifact inventory](artifact_inventory.json) maps28 run directories to published reports and source hashes. The archive also contains diagnostic/preparation runs; a run directory is not necessarily a completed experiment.

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
