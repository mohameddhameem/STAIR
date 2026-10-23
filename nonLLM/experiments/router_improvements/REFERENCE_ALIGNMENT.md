# Reference-aligned non-LLM router protocol

Only W (Qwen2.5-1.5B-Instruct) and S (Qwen2.5-7B-Instruct) are pipeline choices. No M model is introduced. Replace the reference LLM decision model with a non-LLM decision model; preserve each reference's information timing, executed calls, labels, split and evaluation objective.

| Contract | Before October | October Config A |
|---|---|---|
| Retrieval decision inputs | Question + candidate pool | Question + already executed weak retrieval selections |
| Retrieval execution | Execute selected W or S only | Execute W, optionally execute S |
| QA decision inputs | Question + selected retrieval evidence, branch identity | Question + selected evidence + already generated weak answer |
| QA execution | Execute selected W or S only | Execute W, optionally execute S |
| Cost | Selected retrieval + selected answering | All weak calls plus any strong escalations |
| SS vs direct SS | 100% under the same ruler | May exceed 100%; weak calls are already paid |

Sources: `extracted/Python Codes/sft_router.py` (SYS_STAGE, INSTR_RET, INSTR_QA and COST); `October 2026/slr_code_configA/router_prompts.py` (INSTR_RET2, INSTR_QA1); `October 2026/slr_code_configA/compose_eval.py:compose`; `October 2026/slr_code_configA/gate_eval.py`.

Primary accuracy is lenient. Supplied original labels and regenerated Instruct outcomes remain separate cohorts. Original SS 52.79% is validation, not test. The regenerated scorer is the frozen review-validated reconstruction, with the missing full original scorer limitation retained.

The October reference selects the cheapest validation threshold pair at >=95% of validation SS accuracy. Max-accuracy, SS-budget, GRPO, exact expected reward, confidence, sample augmentation, direct-four-action routing and answer-choice workflows are extensions and must be identified as such. In particular, direct-four-action routing is not the pre-October two-stage reference architecture. Confidence and repeated inference are not inputs of the reference prompt contracts.

Existing experiment classification:

- `runs/baselines_real_v1`: five non-LLM classifier families, pre-October two-stage information/execution contract, supplied original cohort and original flat ruler.
- `runs/nonllm_reference_lenient_v1/base/router`: October-style weak-first two-stage classifier baseline; regenerated Instruct cohort, recorded-token ruler.
- `runs/nonllm_reference_lenient_v1/extended` and `runs/nonllm_lenient_longer_v1`: finite-action policy training extensions to the October cascade. Confidence and augmentation modes are additional feature/training experiments, not literal reference-method reproductions.
- `runs/nonllm_reference_lenient_v1/architecture` and `combined`: architecture extensions, not either reference architecture.
- `runs/nonllm_reference_prerouter_v1`: direct-four-action extension on the original cohort, not the original two-stage router.

Do not remove real weak-call costs merely to cap the October cascade at 100%. Do not change October into direct routing and still label it October-aligned. Do not reuse weak-answer-derived features for a pre-October direct router. Report pipeline token/parameter cost separately from training cost and CPU routing overhead.
