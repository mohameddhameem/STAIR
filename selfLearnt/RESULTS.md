# Results so far (cluster run, seed 0)

HotpotQA, the team test set (AgentTTS's 500 questions, dev [500:1000]); accuracy = lenient (team standard, AgentTTS's
ruler); cost = AgentTTS units per question, counted on the tokens actually processed (`stair_common.cost`). Seed 0 only:
with 500 questions an accuracy carries about ±2 points of sampling noise. The per-question tables (`results/common/`
layout) are in [`results/common/`](../results/common/) on this branch, with the fine-tuning loss curves in
[`results/common/history/`](../results/common/history/).

| Experiment | Policy | Accuracy | Units / question |
|---|---|---|---|
| AL-01 AgentTTS | **paper-best** (their reported setting: Qwen2.5 72B x1 -> Llama 3.2 3B x40 + fusion) | **0.744** | 1,984 |
| AL-01 AgentTTS | searched (their search re-run, Qwen2.5 72B as planner instead of o3-mini) | 0.638 | 3,790 |
| AL-02 | always-strongest (Qwen2.5 72B x1 -> Llama 3.1 70B x1 + fusion, AgentTTS's pipeline) | 0.692 | 5,142 |
| AL-03 | fixed: Qwen 7B -> Llama 3B (untrained, our prompts, one pass) | 0.400 | 181 |
| AL-03 | fixed: Qwen 7B -> Llama 8B | 0.494 | 190 |
| AL-03 | fixed: Qwen 7B -> Llama 70B | 0.584 | 276 |
| AL-03 | fixed: Qwen 32B -> Llama 3B / 8B / 70B | 0.564 / 0.606 / 0.678 | 805 / 815 / 913 |
| AL-03 | fixed: Qwen 72B -> Llama 3B / 8B / 70B | 0.614 / 0.690 / **0.730** | 1,808 / 1,814 / 1,875 |
| AL-03 | rulebook, starting rules (hidden-state probes) | 0.512 | 531 |
| AL-03 | rulebook, learnt rules | 0.566 | 609 |
| AL-03 | oracle within the rulebook's choices | 0.646 | 243 |

## Fine-tuning so far (early stopping on validation loss, `results/common/history/`)

| Model | Role | Epochs run | Best epoch (kept) | Val loss |
|---|---|---|---|---|
| Llama 3.2 1B / 3.2 3B / 3.1 8B | answerer | 2 | 1 | 0.354 / 0.270 / 0.273 |
| Llama 3.1 70B (QLoRA, 2,000 questions) | answerer | 2 | 1 | 0.258 |
| Qwen2.5 0.5B | retriever | 3 | 2.5 | 0.127 |
| Qwen3.5 0.8B | retriever | 2 | 1 | 0.044 |
| Qwen2.5 1.5B / 3B / 7B | retriever | 2 | 2 | 0.058 / 0.034 / 0.024 |

Answerers stop improving after one epoch (validation loss rises in the second, so the epoch-1 checkpoint is kept);
retrievers keep improving through epoch 2. Qwen2.5 32B and 72B (QLoRA) are still training.

AL-04 .. AL-06 (fine-tuned models + self-learnt rulebook) are still running: they wait for the QLoRA fine-tuning of
Qwen2.5 32B and 72B; the other nine models are fine-tuned.

## Reading so far

- **Reproduction.** AgentTTS's reported best setting scores 0.744 here (paper: 0.742), so the pipeline, data and ruler
  reproduce their setup. Their search re-run with an open planner and a 50-question score table picks a worse, costlier
  setting (0.638 at 3,790 units).
- **Our single-pass pipeline is efficient.** Qwen 72B -> Llama 70B with our prompts, one answer: 0.730 at 1,875 units,
  against 0.692 at 5,142 for the same two models in AgentTTS's pipeline.
- **AL-03's rulebook does not yet beat fixed pipelines.** The learnt rules (0.566 at 609 units) are dominated by the fixed
  Qwen 7B -> Llama 70B pipeline (0.584 at 276). Even the oracle over the rulebook's choices (0.646) stays below the best
  fixed pipeline (0.730): on escalation the rulebook *merges* the shortlists of the retrievers it has used rather than
  switching to the larger retriever's own shortlist, which appears to cap accuracy. Under investigation before AL-04 .. AL-06
  reach rule learning, since they share the mechanism.
