# The published judges

**Status: published on 2026-10-25.** These are two nano-jev judges for pair A on the golden set, with 3 seeds each.

| Router | Hugging Face | Reads | Decides |
|---|---|---|---|
| qp | [mohameddhameem/systemone-jev-qp](https://huggingface.co/mohameddhameem/systemone-jev-qp) | the question and the small model's top 2 passages | before the small model runs |
| qpa | [mohameddhameem/systemone-jev-qpa](https://huggingface.co/mohameddhameem/systemone-jev-qpa) | the same, plus the small model's answer | after the small model answers |

**Where the files are:**
- **The loading code** is [github.com/mohameddhameem/systemone-jev](https://github.com/mohameddhameem/systemone-jev),
  a trimmed copy of `src/stair/` for inference (`systemone_jev.route.Router`). The score check, the figures and the
  card template are there too.
- **Seeds:** seed 0 is at the root of each Hugging Face repo, and seeds 1 and 2 are in `seed1/` and `seed2/`.
- **Thresholds:** each folder's `systemone.json` names its input set and its operating threshold, picked on `slr-val`.

## What they are

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

## Their results on `slr-test`

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

## Not published

- The Strands Decider judges (2B and E4B) and the experiment 8 option judge. Their weights stay on the cluster.
- **History:** until 2026-10-23 this page planned to publish `stair-judge-*` judges trained on the earlier
  7,405-question setting. Its note that nano-jev is Apache-2.0 was wrong: nano-jev v1.0 is MIT.
