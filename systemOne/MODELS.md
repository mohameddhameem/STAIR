# The published judges

**Status:** two families of judges are on Hugging Face. Both are trained on the golden set, with 3 seeds each.
- **nano-jev** (33M, MIT), published 2026-10-25: two judges for pair A.
- **The Strands Decider** (2B and E4B, Apache-2.0), published 2026-10-26: eight judges, four for pair A and four for
  menu A.

**Layout of every repo:**
- seed 0 is at the root, and seeds 1 and 2 are in `seed1/` and `seed2/`;
- each folder's `systemone.json` names its setting, its input set and its thresholds, picked on `slr-val`.

## nano-jev judges (pair A)

| Router | Hugging Face | Reads | Decides |
|---|---|---|---|
| qp | [mohameddhameem/systemone-jev-qp](https://huggingface.co/mohameddhameem/systemone-jev-qp) | the question and the small model's top 2 passages | before the small model runs |
| qpa | [mohameddhameem/systemone-jev-qpa](https://huggingface.co/mohameddhameem/systemone-jev-qpa) | the same, plus the small model's answer | after the small model answers |

**The loading code** is [github.com/mohameddhameem/systemone-jev](https://github.com/mohameddhameem/systemone-jev), a
trimmed copy of `src/stair/` for inference (`systemone_jev.route.Router`). The score check, the figures and the card
template are there too.

### What they are

- **Base:** [`sdmlai/nano-jev`](https://huggingface.co/sdmlai/nano-jev) v1.0, a 33M MiniLM cross-encoder. It is
  MIT-licensed, and so are the published weights.
- **Pair A:**
  - small: Qwen2.5-1.5B-Instruct on nano-jev's top 2 paragraphs;
  - large: Qwen2.5-7B-Instruct on all 10.
- **Label:** `large-helps`: the small model is wrong and the large one is right, by the standard lenient ruler.
- **Training:**
  - data: the 4,814 `slr-train` questions, with 10% held out to pick the epoch by NLL;
  - loss: grouped-softmax cross-entropy;
  - settings: lr 5e-5, batch 16, up to 3 epochs.
- **Licence of the training data:** HotpotQA is CC BY-SA 4.0. Whether share-alike applies to weights trained on it is
  not settled, and the cards do not claim that it does not.
- **Provenance:**
  - the weights come from OMEGA job 43118 (`judge train ... --save-model`, `configs/system-one-slr.toml`);
  - job 43119 re-evaluated them.

### Their results on `slr-test`

These are on the 1,481 test questions. The 1,303 questions nano-jev never saw are shown in brackets.

| Judge | 3-seed ensemble | Single seeds, mean ± sd | Seed 0 (the Hub root) |
|---|---|---|---|
| qp | 0.674 (0.670) | 0.672 ± 0.001 | 0.670 |
| qpa | 0.674 (0.670) | 0.671 ± 0.002 | 0.668 |

**Baselines:**

| Baseline | AIQ |
|---|---|
| Entropy | 0.665 |
| LR router before the answer | 0.669 |
| LR cascade router | 0.681 |
| Random | 0.638 |
| Oracle | 0.758 |

**Reading:**
- None of the judge–baseline differences is significant, so the judges tie the baselines. The cards claim no
  accuracy gain.
- [RESULTS.md](RESULTS.md) lists nano-jev qpa at 0.677. That is the experiment 7/9 run, whose weights were not
  saved. The retrain that saved them scores 0.674, within the seed spread.
- **Score check:** the saved scores were computed in bf16 on a GPU.
  - On an L40S, `route.py` reproduces them exactly.
  - On a CPU in fp32, they differ by up to 7.6e-3, and at most 2 of the 1,481 decisions flip.

**What the Jev programme adds beyond the cards** ([write-up](docs/experiments/2026-10-25-jev-novelty.md)): fine-tuning
from nano-jev gives a better router than starting from the same 33M MiniLM without decision training. This is
confirmed on MuSiQue and 2WikiMultiHopQA. The cards report only the HotpotQA golden set.

## Strands Decider judges (pair A and menu A)

| Setting | 2B, qp | 2B, qpa | E4B, qp | E4B, qpa |
|---|---|---|---|---|
| Pair A | [systemone-decider-2b-qp](https://huggingface.co/mohameddhameem/systemone-decider-2b-qp) | [systemone-decider-2b-qpa](https://huggingface.co/mohameddhameem/systemone-decider-2b-qpa) | [systemone-decider-e4b-qp](https://huggingface.co/mohameddhameem/systemone-decider-e4b-qp) | [systemone-decider-e4b-qpa](https://huggingface.co/mohameddhameem/systemone-decider-e4b-qpa) |
| Menu A | [systemone-decider-2b-qp-menu-a](https://huggingface.co/mohameddhameem/systemone-decider-2b-qp-menu-a) | [systemone-decider-2b-qpa-menu-a](https://huggingface.co/mohameddhameem/systemone-decider-2b-qpa-menu-a) | [systemone-decider-e4b-qp-menu-a](https://huggingface.co/mohameddhameem/systemone-decider-e4b-qp-menu-a) | [systemone-decider-e4b-qpa-menu-a](https://huggingface.co/mohameddhameem/systemone-decider-e4b-qpa-menu-a) |

**The loading code** is [github.com/mohameddhameem/systemone-decider](https://github.com/mohameddhameem/systemone-decider)
(`systemone_decider.route.PairRouter` and `MenuRouter`). The training code, the score check and the card templates
are there too.

### What they are

- **Base:** the [Strands Decider](https://github.com/strands-labs/strands-decider) checkpoints. The 2B runs on a
  Qwen3.5-2B torso, and the E4B on Gemma 4 E4B. A pointer head scores each option of a typed question.
  - Fine-tuning trains the LoRA (rank 16 in the 2B, 48 in the E4B) and the head.
  - The weights are Apache-2.0, as the checkpoints are.
- **Pair A**, as for nano-jev: one question, "which model should answer?", with the `large-helps` label.
  - `qp` reads the question and the small model's top 2 passages; `qpa` also reads its answer.
  - The threshold is the cheapest point on `slr-val` within 1% (relative) of always-large's accuracy.
- **Menu A** is jonathan's SLR setup: Qwen2.5 7B / 72B retrieve, and Llama 3.2 3B / 3.1 70B answer. One model answers
  both stage questions, weak first.
  - **Retrieval:** do the weak retriever's passages hold what is needed?
  - **Answer:** will the weak answerer do? `qp` reads that branch's passages; `qpa` also reads the weak answer.
  - **Labels and threshold grid** are jonathan's (`ret_labels` / `qa_labels`, steps of 0.05). The cards credit the
    SLR setup.
  - **Three rules ship:**
    - `near_strong`, the default: the cheapest point within 1% of always-strong's accuracy on `slr-val`;
    - his `peak` and `parity`.
- **Training:**
  - data: the 4,814 `slr-train` questions, with 10% held out to pick the epoch by NLL;
  - loss: grouped-softmax cross-entropy;
  - settings: the plans of experiments 9 and 10.
- **Licence of the training data:** as for nano-jev above.
- **Provenance:** a retrain of experiments 9 and 10's fine-tuned deciders, with the same plans and seeds, keeping the
  weights.
  - Pair A: jobs 43342 / 43343, eval 43344.
  - Menu A: jobs 43345 / 43346, eval 43347.
  - Reports are in [results/decider-release](results/decider-release/), and the comparison with the experiment runs
    is in [the golden-set write-up](docs/experiments/2026-10-21-slr-standard.md) ("Published weights").

### Their results on `slr-test` (1,481 questions)

**Pair A.** A judge's own call is priced at its measured tokens: 1.2–1.3 for the 2B, and 4.3–4.4 for the E4B.

| Judge | 3-seed ensemble | Single seeds, mean ± sd | Seed 0 with its price | Seed 0 at its threshold |
|---|---|---|---|---|
| 2B qp | 0.682 | 0.679 ± 0.004 | 0.666 | 64.82, 2.7% below always-large, at 24% of its cost |
| 2B qpa | 0.697 | 0.694 ± 0.001 | 0.682 | 64.96, 2.5% below, at 24% |
| E4B qp | 0.682 | 0.681 ± 0.001 | 0.645 | 64.89, 2.6% below, at 54% |
| E4B qpa | 0.710 | 0.705 ± 0.005 | 0.655 | 65.70, 1.4% below, at 56% |

The baselines are the nano-jev table's: entropy 0.665, the best LR router 0.681, random 0.638 and oracle 0.758.
Always-large scores 66.64 at 10 a question.

**Menu A.** Prices are weak-first, from 80 (all WW) to 1,009 (all SS). The ensemble's three passes cost 19.3–20.2 a
question for the E4B, and 4.5–4.7 for the 2B.

| Judge | 3-seed ensemble | With its price | Single seeds, mean ± sd | Seed 0, `near_strong` (price + judge) |
|---|---|---|---|---|
| E4B qpa | 0.739 | 0.732 | 0.733 ± 0.002 | 75.62 at 603.1 + 6.9 |
| 2B qpa | 0.731 | 0.730 | 0.727 ± 0.001 | 75.76 at 661.1 + 1.7 |
| E4B qp | 0.722 | 0.716 | 0.720 ± 0.001 | 75.08 at 591.2 + 7.0 |
| 2B qp | 0.722 | 0.720 | 0.719 ± 0.001 | 74.81 at 552.7 + 1.7 |

The references:
- jonathan's SFT (label) router scores 0.724, and his SFT (label) + reward router 0.726;
- the fixed paths' hull scores 0.693;
- always-strong scores 76.77 at 929.

**Reading:**
- **Pair A:** with its price, one 2B qpa judge beats entropy for all three seeds, and ties the LR routers that read
  the small answer. The E4B judges lose their lead to their price.
- **Menu A, before its price:** the E4B qpa ensemble beats both of jonathan's routers ([+0.007, +0.022],
  [+0.007, +0.021]).
- **Menu A, with its price, the verdict depends on the draw.**
  - This retrain beats both routers narrowly ([+0.001, +0.016], [+0.0002, +0.014]).
  - The experiment 10 run tied them ([−0.002, +0.015], [−0.003, +0.013]).
  - Read together: "ties or narrowly beats". [RESULTS.md](RESULTS.md) and the root README quote the experiment's
    0.736 / 0.730.
- **Draws:** the retrain lands within 0.002 of the experiment runs' seed means on both settings.
- **`near_strong`'s 1% holds on validation only.** On test, seed 0 lands 1.3–2.6% below always-strong, at 60–71% of
  its price.
  - For seed 1 of 2B qp, E4B qp and E4B qpa it escalates everything. Their retrieval scores crowd above 0.95, past
    the 0.05 grid.
- **Score check** (job 44210): on a GPU, in bf16 with the judge's batches of 16, all 24 folders reproduce every saved
  test score exactly.
  - `MenuRouter.retrieval` and `answer` score one item at a time, which pads differently. They moved scores by up to
    0.013 on the checked items.

## Not published

- **The experiment 8 option judge.** Its weights stay on the cluster.
- **The zero-shot deciders** need no release: they are the Strands Decider checkpoints as published.
- **History:**
  - Until 2026-10-23, this page planned to publish `stair-judge-*` judges trained on the earlier 7,405-question
    setting.
  - Its note that nano-jev is Apache-2.0 was wrong: nano-jev v1.0 is MIT.
  - Until 2026-10-26, this page listed the Strands Decider judges as not published.
