# STAIR
RightRoute: STAge-level Instance Routing

This branch (`systemOne`) holds the System One judge line of work: see [systemOne/README.md](systemOne/README.md).
Other lines: [`nonLLM`](https://github.com/mohameddhameem/STAIR/tree/nonLLM),
[`selfLearnt`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt).

## Side by side: AgentTTS and the three STAIR lines

Snapshot 2026-10-17. Each column is quoted from its own branch or paper; see the notes below before comparing numbers
across columns.

| | **AgentTTS** (baseline) | **systemOne** (this branch) | **nonLLM** | **selfLearnt** |
|---|---|---|---|---|
| Idea | An LLM planner searches one global setting (models + sample counts) under a compute budget | A small typed decision model (judge) decides per question whether the large model is needed | Cheap non-LLM classifiers route each stage weak/strong; per-question sampling budgets | No router: fine-tuned model ladders + a rulebook learnt by micro-experiments escalate per question |
| Decides | Once, for all questions | Per question | Per question and per stage | Per question and per stage |
| Decision model | o3-mini planner (offline search) | nano-jev 33M; Strands Decider 2B / 4B | TF-IDF LR / SVM / NB, random forest | Thresholded signals (confidence, agreement, probes) |
| Models | Qwen2.5 7B / 32B / 72B retrieve; Llama 3.2 3B / 3.1 8B / 70B answer | Qwen2.5 1.5B / 7B / 14B (exp. 1–7, 9); Llama 3B ×1–40 fused vs 70B (exp. 8) | Qwen2.5 1.5B / 7B | Qwen2.5 0.5B–72B retrievers; Llama 3B / 8B / 70B (Colab: Qwen 7B / 32B) |
| Test set | 500 HotpotQA questions (dev [500:1000]) | 7,405 dev (exp. 7, 9); **AgentTTS's 500** (exp. 8) | Team split, 1,481 test | **AgentTTS's 500** (cluster); 400 (Colab) |
| Metric | Gold contained in the response (lenient) | F1 (exp. 7, 9); LLM-graded short answer, plus lenient (exp. 8) | Lenient + strict; EM/F1 after extraction | Lenient (cluster); "corrected" EM / F1 ≥ 0.5 / contained (Colab) |
| Cost | AgentTTS budget units | GFLOPs; AgentTTS units in exp. 8 | Pipeline units 2.0 / 5.7 / 5.7 / 9.4 | AgentTTS units + FLOPs; Colab cost index |
| Best result so far | Best setting Qwen 72B ×1 → Llama 3B ×40 + fusion: 0.742 reported; **0.746 lenient** in the team's re-score at **~938 units** (~101% of strong-everything) | AgentTTS's 500: Llama 70B on nano-jev's top 5, **0.748 graded** (0.680 lenient) at **96 units** (10.3%); 3B ×40 fused **0.756 lenient** at 220 units. 7,405 dev: judge AIQ 0.673 → 0.708 (2B decider) | Routers +0.9 to +1.3 points over cost-matched fixed policies (stored grid, lenient) | Colab: fine-tuned 0.8B retriever → Qwen 32B 84.0% at 43.5% of an AgentTTS-style setup's compute |
| Status | Published | Done (exp. 1–9) | Instruct sampling run in progress | Cluster run not started at snapshot |

### On AgentTTS's own 500 questions (the only shared test set)

| Setup | Retrieval units | Total units | Lenient | Graded |
|---|---|---|---|---|
| AgentTTS best setting (team re-score) | 814 (Qwen 72B) | ~938 | 0.746 | not graded yet |
| systemOne: nano-jev top 5 → Llama 70B | **2.9** (nano-jev) | **96** | 0.680 | **0.748** |
| systemOne: nano-jev top 5 → Llama 3B ×40 fused | 2.9 | 220 | **0.756** | 0.632 |
| selfLearnt workstreams 1–5 | | | pending | |

- **Cost:** systemOne's retrieval is about **280× cheaper** than AgentTTS's 72B retrieval, and the whole pipeline
  **4–10× cheaper**.
- **Accuracy is not settled.** On the lenient metric the fused 3B edges AgentTTS (0.756 vs 0.746), but lenient
  scoring rewards long answers: under the LLM grader the 70B (0.748) clearly beats the fused 3B (0.632). AgentTTS's
  answers have not been graded the same way, so no accuracy win over AgentTTS is claimed.

### Notes on comparing

- Only systemOne's experiment 8 and selfLearnt's cluster workstreams use AgentTTS's 500 questions. nonLLM's 1,481
  and systemOne's 7,405 are different test sets; their numbers cannot be put on one scale with AgentTTS's.
- Lenient, strict, graded and "corrected" accuracy differ by up to ~0.14 on the same answers (systemOne exp. 8).
- Sources: [systemOne/RESULTS.md](systemOne/RESULTS.md);
  [nonLLM results](https://github.com/mohameddhameem/STAIR/blob/b6302ce/nonLLM/results/measured_v1/REPORT.md);
  [selfLearnt README](https://github.com/mohameddhameem/STAIR/blob/dcaa60b/selfLearnt/README.md);
  [AgentTTS](https://github.com/FairyFali/AgentTTS).
