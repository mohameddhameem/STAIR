# Non-LLM routing

The routing decisions use classical or small finite-action models; retrieval and answering still use LLM executors.

**Start with the [complete results index](results/README.md)** (updated31 October2026):914 explicitly separated result rows, code, validation locks, per-task decisions, audits and English reporting figures.

## Latest: MedAgentBench and Jonathan-cache HotpotQA comparisons

- [MedAgentBench English reporting figure](results/figures/medagentbench/medagentbench_nonllm_vs_jonathan.png) and [PDF](results/figures/medagentbench/medagentbench_nonllm_vs_jonathan.pdf): three GRPO architectures plus seven supervised classical families, compared with Jonathan's LLM router.
- [MedAgentBench classical report](results/medagentbench_classical_v1/REPORT.md): Linear/RBF SVM and logistic regression reach63.33% task success at77.68% full weak-first cost in the gated, validation-selected round2 setting; Jonathan is63.33%/81.47%. Fifteen both-fail tasks avoid ineffective escalation; failures remain. Seeds0/1/2 agree on this fixed replay. This is supervised reward-weighted classification, **not GRPO**.
- [MedAgentBench GRPO report](results/medagentbench_grpo_v1/REPORT.md):24 fresh fits on the same patient-disjoint90/60/150 split and cached views/outcomes; A1 reward-minus-group-mean update, no std/KL/PPO. Seed17 MLP64 reproduces T1's decisions, but repeat seeds vary substantially. [Training protocol/code](experiments/medagentbench/README.md).
- [HotpotQA English reporting figure](results/figures/hotpotqa_v2/nonllm_jonathan_aligned_comparison_v2.png): frozen Jonathan output cache, shared4814/1110/1481 split, standard lenient and Table5 budget. SS76.77%; MLP128→64 SFT matches that accuracy at98.18% cost at the validation-budget point. [Alignment report](results/hotpotqa_jonathan_aligned_v1/REPORT.md). Source executor quantization is unconfirmed; this is not independent executor reproduction.

All these tests actually run the trained router and replay frozen task-executor outcomes; no new medical episodes are generated. Costs exclude decision-model/risk-scorer compute consistently with the reference readout. Test sets were previously inspected; results are exploratory. [Minimal record-free audit data](results/medagentbench_replay_inputs/README.md) supports patient-group, cost and pass/fail checks without raw clinical views or patient identifiers.

## Earlier independently generated GGUF cohort

The corrected HotpotQA500-question GGUF experiment is complete: direct SS is **71.8% lenient accuracy**. Six classical router families with three seeds trained successfully, but all validation-selected operating points fell back to fixed SS. No improvement from dynamic routing is claimed. The new corrected-cache GRPO rerun also reaches71.8% at100% SS cost by learning to choose SS for every test question. Its exact expected-reward control reaches71.4% at83.79% SS cost. [GRPO report](results/menu_ws_grpo_v1/REPORT.md). Historical GRPO results remain separately labeled.

| Retrieval / answering | Lenient % | Cost % direct SS |
|---|---:|---:|
| Qwen7B / Llama3B |54.0|9.62|
| Qwen7B / Llama70B |62.2|12.60|
| Qwen72B / Llama3B |65.8|97.26|
| Qwen72B / Llama70B |71.8|100.00|

These results use Q4_K_M GGUF, K=1/greedy/no fusion, common0.2 test IDs and token-derived FLOP units. They are not the newer common0.3 SLR golden-set reproduction, nor automatically eligible for pooled BF16/AWQ comparisons. See the [protocol boundaries and archived results](results/README.md).

- [Additional methods: success/gain predictors and unscaled policy gradients](experiments/menu_ws_gguf/ALTERNATIVES.md)
- [Current four-model code](experiments/menu_ws_gguf/README.md)
- [Historical non-LLM improvements and GRPO](experiments/router_improvements/README.md)
- [Weak-first cascade](CASCADE.md)
- [Historical setup audit](COMMON_SETUP.md)
- [Earlier baseline/deployment instructions](HISTORICAL_README.md)
- [Current upstream common framework](../common/README.md)

Large model files, raw caches and checkpoints remain outside Git. No cluster jobs are submitted by this results publication.

Latest exploratory alternatives: a per-family validation-selected success-prediction tree scores72.4% at96.77% SS cost; augmented exact reward preserves71.8% at89.25% SS cost. The global validation-selected family instead scores71.2% at85.61% cost. Small/reused test-set gains are not stable-improvement evidence. [Complete comparison](results/menu_ws_alternatives_v1/REPORT.md).

Further completed studies add four-path paired routing, strict nested OOF stacking, held-out CRC-style gates and fully charged weak-first cascades. OOF stacking scores72.0% at90.70% SS cost; the OOF match-SS selection scores71.8% at89.77%. Weak-first cascade at the validation95% target scores71.4% at74.13%. All-escalate cascade costs109.61%; calibration targets are not per-test guarantees. [Full four-study report](results/menu_ws_next_v1/REPORT.md).
