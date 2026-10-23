> Complete. [Report](../../results/menu_ws_grpo_v1/REPORT.md): GRPO71.8% at100% SS cost; exact expected-reward control71.4% at83.79% SS cost.

# Corrected four-model non-LLM GRPO

Source cache: `runs/nonllm_common_ws_gguf_v2`, whose SS is71.8% on common0.2 test500. Output: `runs/nonllm_common_ws_grpo_v1`. This run does not switch to common0.3 slr-* golden-set IDs or prices.

The frozen executors are Qwen7B/72B retrieval and Llama3B/70B answering. No new answers are generated. A retrieval head sees only the question and supplied pool; the QA head sees the question plus the selected retrieval evidence and model tag. Input features are training-only TF-IDF (up to25K terms per stage) plus scaled numeric statistics. Two linear Bernoulli heads are non-LLM models.

Task reward is author lenient correctness minus lambda times actual executed retrieval+QA token units divided by training-mean direct SS units. Strong retrieval is selected directly; there is no mandatory weak-first execution cost. Lambda is0/.05/.15/.4. The cached four-route outcome grid supplies training rewards, not deployment features.

Three seeds0/1/2; group8/32 give24 GRPO streams. Twelve exact expected-reward streams provide an enumeration control. All start from the same per-seed masked cheapest-correct BCE warm start (maximum30 epochs, patience5, best validation BCE).

GRPO samples route actions, computes group-relative standardized advantages, and uses a clipped PPO surrogate (clip0.2, two updates, KL0.01 to the rollout policy, entropy0.002). A constant-reward group has zero advantages. This is our finite-action adaptation, not LLM token-level TRL GRPO. More route samples do not create additional LLM answers.

Maximum300 RL epochs; validation task loss is negative expected correctness-minus-cost reward. Check every10 epochs (also epoch1), keep best, early-stop after six non-improving checks and at least60 epochs. Log task train/val loss separately from the sampled surrogate loss. Model/optimizer/RNG states are retained for continuation.

Validation averages the three seeds' stage probabilities, then selects operating points for maximum accuracy, SS budget, and cheapest cost retaining95% of validation SS accuracy. Report learned-only and fixed-fallback selections separately; report when no learned policy meets a constraint. A validation budget does not guarantee the same budget on test. The validation lock precedes test features/scoring; test has previously been inspected, so results are descriptive.

CPU feature extraction and ensemble decision latency are included in ms_router. CPU and training work are excluded from the parameter-FLOP cost proxy, consistently with the classical experiment.

Commands from the workspace directory containing STAIR and the local data:

```bash
python STAIR/nonLLM/experiments/menu_ws_gguf/test_grpo.py
python STAIR/nonLLM/experiments/menu_ws_gguf/grpo.py --out runs/nonllm_common_ws_grpo_v1
python STAIR/nonLLM/experiments/menu_ws_gguf/status_grpo.py
```

Training uses up to four free local GPUs without interrupting other jobs. The local datasets, executor caches, feature arrays and model checkpoints remain outside Git.
