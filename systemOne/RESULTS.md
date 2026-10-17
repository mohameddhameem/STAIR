# Results

One line per run; the full write-up of each is in [docs/experiments/](docs/experiments/), and the index with
published reference points in [docs/experiments.md](docs/experiments.md). Report files are in [results/](results/).
Experiments 1–7 and 9 score answer F1 on the 7,405 HotpotQA dev questions; AIQ is comparable only within one model
pair.

## Experiments 1–6: experts and feature routers (2026-10-05/06)

| Run | What | Main result |
|---|---|---|
| 1 | Pair A: Qwen2.5-1.5B (top 2 passages) / 7B (all 10); logistic regression routers | Always small F1 61.3 at 1,051 GFLOPs; always large 66.9 at 22,762; best router AIQ 0.665 (entropy 0.662, random 0.641, oracle 0.755) |
| 2 | Cascade routers (decide after the small answer) | AIQ 0.672, the only routers significantly above entropy |
| 3 | 7B reading nano-jev's top 3 / top 5 | Better and cheaper: F1 69.2 at 7,292 / 70.0 at 11,598 GFLOPs |
| 4 | Large = Qwen2.5-14B (pair B) | F1 71.9; router AIQ 0.697 |
| 5 | Small = Qwen2.5-3B (pair C) | Router AIQ 0.681 |
| 6 | Three-stage cascade 1.5B → 7B (top 3) → 14B | F1 72.6 at 14,243 GFLOPs (15% reach the 14B) |

## Experiment 7: the System One judge (2026-10-07)

Fine-tuned nano-jev (33M) on pair A; 3 seeds, scores averaged. ([write-up](docs/experiments/2026-10-07-system-one-router.md),
[report](results/system-one/eval/A/report.json))

| Policy | AIQ | Judge cost |
|---|---|---|
| Entropy threshold | 0.662 | |
| Best logistic regression before the answer | 0.665 | |
| Zero-shot nano-jev ("is the answer correct?") | 0.664 | 33 GFLOPs |
| **Judge qp** (question + top-2 passages) | **0.673** | 32 GFLOPs |
| Judge qpa (adds the small answer) | 0.673 | 33 GFLOPs |

qp minus the best logistic regression before the answer: [+0.004, +0.011]. On pair B the best judge edges the best cascade router ([+0.000, +0.006]); on pair C (a different small model) the gain is small and not clearly above entropy.
No router reaches always-large's F1 at its operating point.

## Experiment 8: the AgentTTS setting (2026-10-14/15)

AgentTTS's 500 HotpotQA test questions, five answer options, cost in AgentTTS units (strong-everything = 929).
Re-scored with an LLM grader (Qwen2.5-14B) on each model's short committed answer.
([write-up](docs/experiments/2026-10-15-experiment-8-rescored.md), [report](results/exp8-rescore/))

| Option | Lenient | **Graded** | Units |
|---|---|---|---|
| Llama 3.2 3B, one answer | 0.608 | 0.524 | 7.5 |
| 3B, 10 samples fused | 0.726 | 0.588 | 64.9 |
| 3B, 40 samples fused | 0.756 | 0.632 | 220.0 |
| **Llama 3.1 70B (4-bit), one answer** | 0.680 | **0.748** | 96.0 |
| Oracle (cheapest correct option) | | 0.816 | 30.3 |

The judge picks the 70B for 500 of 500 (qp) and 499 of 500 (qpa) questions; its AIQ ties the fixed-policy hull
(qp 0.701) or slightly beats it (qpa 0.713). AgentTTS's own 74.6 is lenient and not comparable until its answers
are graded the same way.

## Experiment 9: Amazon's Strands Decider as the judge (2026-10-16/17)

2B (Qwen3.5 torso) and 4B-active (Gemma 4 E4B) deciders, zero-shot and fine-tuned (LoRA + pointer head, 3 seeds).
([write-up](docs/experiments/2026-10-17-experiment-9.md), [results](results/exp9/))

| Setting | Judge | AIQ | With judge cost | Minus nano-jev qp |
|---|---|---|---|---|
| Exp. 7, pair A | 2B fine-tuned, qpa | 0.708 | **0.702** | [+0.030, +0.039] |
| | 4B fine-tuned, qpa | **0.713** | 0.700 | [+0.035, +0.045] |
| | 4B zero-shot | 0.693 | 0.683 | [+0.016, +0.024] |
| | 2B fine-tuned, qp | 0.680 | 0.676 | [+0.004, +0.011] |
| | nano-jev qp | 0.673 | 0.673 | |
| Exp. 8, graded | best decider (2B zero-shot, qpa) | 0.700 (cost included) | | hull 0.701; nano-jev qpa 0.713 |

The decider gains mostly from reading the small model's answer; on experiment 8 (3,000 training questions, few
examples for the expensive options) no decider beats the hull.
