# Instruct GGUF full-run launch

The full 7,405-question experiment started on serverA on 2026-10-15, using the
official 1.5B/7B Instruct GGUF Q4_K_M weights and llama.cpp. ServerB is not used.
Physical GPUs 0/1/3 are running. GPU2 failed CUDA allocation with an uncorrectable
ECC error; reset is pending permission to close blocking monitoring clients.
All questions remain assigned through disjoint work queues across healthy GPUs.

Temperatures are fixed to public AgentTTS HotpotQA code: retrieval .9, QA .7,
fusion .3, maximum 512 new tokens. There is no temperature sweep. Each pair has
one 40-candidate QA batch and all 40 prefix-fusion budgets.

The archived profile records model revisions and verified hashes. Validation
covered three short controls per model, 40 unique seeded native draws, grounded
reader controls, and 60,244-token long-input checks on both models. The actual
pipeline completed 32 fusion outcomes on two training questions. A separate
actual-worker smoke completed all 160 pair/budget outputs and answer reading
for training index 0; those valid cached outputs can be reused by the full run.
Full-corpus conservative fusion context bounds passed for all 7,405 questions.

These files document launch validation, not held-out benchmark results. Final
fixed-policy and retrospective oracle results are pending. Live status is in
`/data1/zmcheng/cs707_sampling40_instruct_gguf/state.json`; use `gguf_status.py`
to read committed progress. Details are in
[`GGUF_RUN.md`](../../experiments/adaptive_sampling/GGUF_RUN.md).
