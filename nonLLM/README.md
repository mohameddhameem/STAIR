# Non-LLM routing

The routing decisions use classical or small finite-action models; retrieval and answering still use LLM executors.

**Start with the [complete results index](results/README.md)** (updated24 October2026). It links the corrected four-model experiment, historical GRPO/multi-sample/architecture experiments, per-question outputs, protocols and invalidated runs.

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
