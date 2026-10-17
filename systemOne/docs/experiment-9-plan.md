# Experiment 9 plan: Amazon's Strands Decider as the System One judge

**RQ9: does a 2B–4B decision model judge routing better than the 33M nano-jev, by enough to pay for itself?**
Both zero-shot and fine-tuned on the same labels.

## Why

- **Experiment 7, pair A:** the nano-jev judge (qp) reached AIQ 0.673, against 0.665 for the matched logistic
  regression router and 0.662 for entropy. That is the report's central result, but the margin is small.
- **Experiment 8, graded labels:** nano-jev learned to send almost every question to the 70B. Its AIQ ties the
  fixed-policy hull (qp 0.701), while the oracle reaches 0.816. A stronger judge could claim part of that gap.
- **A new model family:** Amazon released the Strands Decider family on 2026-10-01
  ([announcement](https://strandsagents.com/blog/introducing-strands-decider/),
  [repo](https://github.com/strands-labs/strands-decider), Apache-2.0). It is the same kind of model as nano-jev,
  a typed decision model built after Marc Brooker's
  [System One post](https://brooker.co.za/blog/2026/09/28/engineering-system-one.html), but about 60–120× larger.

## The model family

| | |
|---|---|
| Architecture | An LLM torso with its text head removed. A pointer head (about 1M parameters) scores each option's hidden state against the `<answer>` position, and a rank-16 LoRA adapts the torso. One calibrated temperature per question kind (`noul`, `choice`, `score`) |
| Checkpoints used | `StrandsAgents/strands-decider-2B-hobson-v21` (Qwen3.5-2B torso, `choice` temperature 0.818) and `StrandsAgents/strands-decider-E4B-gemma4-v1-2610` (Gemma 4 E4B, 0.775). Both have a 4,096-token window |
| Other checkpoints | `…-2B-hobson-v19`, `…-2B-qwen3.5-v1-2610`, Gemma 4 `E2B`, `12B` and `26B-A4B` |
| Training data | About 115k rows from 25+ public datasets, synthetic rows and distillation. **HotpotQA is evaluation only** ([data/sources.md](https://github.com/strands-labs/strands-decider/blob/main/data/sources.md)), so our test sets are clean. MuSiQue, another multi-hop dataset, is in it |
| Reported | Accuracy around 72% on JevBench public v19, a median latency of about 106–115 ms on an RTX 3090, and "a confidence of 0.9 or more is right about 95% of the time" |

## Design

- **The wrapper,** `stair.decider.DeciderJudge`:
  - it has `judge.Judge`'s interface, so both experiments' training and evaluation code run unchanged;
  - an item becomes a decider `choice` question, rendered and truncated exactly as the package's own engine does;
  - `decider-smoke` checks the wrapper against that engine before anything long runs.
- **Zero-shot:** the checkpoint's own calibrated probabilities.
  - Experiment 7: nano-jev's two zero-shot questions, "is this claim supported?" and "is the proposed answer
    correct?".
  - Experiment 8: the five-option "which answer option" question.
- **Fine-tuned:** nano-jev's protocol, applied to the LoRA and the pointer head only:
  - grouped cross-entropy on the same labels, keeping the epoch with the best dev NLL, over 3 seeds;
  - learning rate 1e-4 and 2 epochs, where nano-jev used 5e-5 and 3.
- **Cost:**
  - each judge pays 2 × parameters × tokens: 2B for the Qwen model, 4B active for E4B, against nano-jev's 0.0334B;
  - experiment 7 reports AIQ with and without that cost;
  - experiment 8 adds it to every question.
- **Comparisons:** paired bootstrap against nano-jev, in the same report. Experiment 7 reuses its trained nano-jev
  judges; experiment 8 reads them through `reference_judges`.
- **One difference to report:** the decider keeps its 4,096-token window, while nano-jev cut the state at 512
  tokens. For experiment 7's top-2 passages this rarely binds; for experiment 8's top 5 it can.

## Hypotheses

- **H9a.** The fine-tuned 2B decider beats nano-jev qp on pair A by at least 0.005 AIQ, with a paired 95% interval
  above zero.
- **H9b.** In experiment 8, a fine-tuned decider's AIQ beats the fixed-policy hull, with an interval above zero,
  where nano-jev qp only tied it.
- **H9c.** Zero-shot deciders beat the entropy router with their judge cost included.

## Runs

GPU job sheet: [gpu-jobs-exp9.md](gpu-jobs-exp9.md), about 25–30 GPU-hours on OMEGA in all.
