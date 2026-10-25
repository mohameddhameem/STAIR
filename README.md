# STAIR
RightRoute: STAge-level Instance Routing

Want an experiment run on the cluster? See [experiments/](experiments/README.md).

This branch (`systemOne`) holds the System One judge line of work: see [systemOne/README.md](systemOne/README.md).
- **Other lines:**
  - [`jonathan`](https://github.com/mohameddhameem/STAIR/tree/jonathan): Stage-Level Routing, and the HotpotQA golden
    set;
  - [`nonLLM`](https://github.com/mohameddhameem/STAIR/tree/nonLLM);
  - [`selfLearnt`](https://github.com/mohameddhameem/STAIR/tree/selfLearnt).
- **The team standard and shared code:** [common/](common/README.md).

## Side by side: AgentTTS and the STAIR lines

**Snapshot 2026-10-23.** Each column is quoted from its own branch at the commit listed under [Sources](#sources), or
from the paper. Read the notes before comparing numbers across columns: test sets, rulers and cost scales differ.

| | **AgentTTS** (baseline) | **jonathan** (SLR) | **systemOne** (this branch) | **nonLLM** | **selfLearnt** |
|---|---|---|---|---|---|
| Idea | An LLM planner searches one global setting (models + sample counts) under a compute budget | A learned router at each stage decides whether the weak model's output is enough (weak-first cascade) | A small typed decision model (judge) reads the question and the evidence, and optionally the weak answer, then decides whether to pay for the stronger model | Cheap non-LLM classifiers route each stage weak/strong; per-question sampling budgets | No router: fine-tuned model ladders + a rulebook learnt by micro-experiments escalate per question |
| Decides | Once, for all questions | Per question and per stage | Per question (and per stage on menu A) | Per question and per stage | Per question and per stage |
| Decision model | o3-mini planner (offline search) | Qwen2.5-1.5B + LoRA (SFT on labels; + reward) | nano-jev 33M; Strands Decider 2B / E4B (LoRA) | TF-IDF LR / SVM / NB, random forest | Thresholded signals (confidence, agreement, probes) |
| Models | Qwen2.5 7B / 32B / 72B retrieve; Llama 3.2 3B / 3.1 8B / 70B answer | Config B (= menu A): Qwen2.5 7B / 72B retrieve, Llama 3.2 3B / 3.1 70B answer. Config A: Qwen2.5 1.5B / 7B | Menu A (exp. 10); nano-jev top 5 → Llama 3B ×1–40 fused or 70B (exp. 8); Qwen2.5 1.5B / 7B (exp. 7, 9) | Qwen2.5 1.5B / 7B (Config A's grid) | Menu A, fine-tuned; a ladder from nano-jev 33M to Qwen 72B; Colab: up to Qwen 32B |
| Test set | AgentTTS's 500 (dev [500:1000]) | **Golden set: `slr-test`**, 1,481 of the 7,405 dev questions | **`slr-test`** since 2026-10-21; AgentTTS's 500 kept as history | A 1,481-question test split of the same sizes (routers); a 500-question training cohort (sampling study) | AgentTTS's 500 (cluster); 400 (Colab) |
| Metric | Gold contained in the response (lenient) | The standard lenient ruler | The standard lenient ruler | The grid's stored lenient + strict; EM/F1 after extraction | Lenient + EM / F1 (cluster); "corrected" EM / F1 ≥ 0.5 / contained (Colab) |
| Cost | Its normalised budget (Table 5) | B: Table 5 prices per call, weak-first, strong-everything 929, router not charged. A: its own token-normalised scale | As jonathan's B, plus each judge's own price | Pipeline units 2.0 / 5.7 / 5.7 / 9.4 (routers); parameters × tokens (sampling) | AgentTTS units (cluster); a cost index (Colab) |
| Best result so far | Qwen 72B ×1 → Llama 3B ×40 + fusion: 0.742 reported, **0.746** lenient in the team's re-score, at a price of **859** (92% of strong-everything) | B: SFT (label) **76.70 at 835** and + reward 75.96 at 665, against always-strong 76.77 at 929. A: 59.55 at 182.3 against always-strong 59.55 at 250.1 | nano-jev top 5 → 3B ×20 fused **75.08 at 31.6** (3.4% of 929), tied with always-strong. E4B decider routing menu A: AIQ **0.736** against jonathan's 0.724 / 0.726 (better before its price, tied with it) | Routers +0.85 to +1.30 points over a cost-matched mix of fixed policies (best: 41.66% at 51% of SS's cost). Sampling cohort, gold-assisted oracle (not deployable): 82.8% against SS ×40's 37.8% at 46–58% less cost | Colab: fine-tuned 0.8B retriever → Qwen 32B 84.0% at cost index 43.5 (an AgentTTS-style pick = 100) |
| Status | Published | Done. B's rows reproduce exactly (checked by systemOne); A's router rows need files still on the lab box | Done (exp. 1–10 and the Jev programme); nano-jev qp / qpa [published on Hugging Face](systemOne/MODELS.md) | Full sampling run stopped at the user's request on 2026-10-18, progress saved | Cluster run (AL-01 … AL-06, on AgentTTS's 500) not started: GPU smoke test pending |

### On the golden set (`slr-test`, 1,481 questions, standard lenient ruler, Table 5 prices)

The only test set two lines share today: jonathan's routers and systemOne's experiments 7–10.

| Policy | Line | Lenient | Price | % of 929 |
|---|---|---|---|---|
| Always-strong (Qwen 72B retrieval → Llama 70B) | jonathan | 76.77 | 929 | 100% |
| SLR SFT (label) | jonathan | 76.70 | 834.8 | 90% |
| SLR SFT (label) + reward | jonathan | 75.96 | 664.7 | 72% |
| E4B decider (mean of 3 seeds) at jonathan's operating rule | systemOne | 76.70 | 737.5 (757.7 with its own calls) | 79% |
| nano-jev top 5 → Llama 3B, 20 samples fused | systemOne | 75.08 | 31.6 | 3.4% |
| nano-jev top 5 → Llama 70B | systemOne | 68.33 | 121.6 | 13% |
| WS (Qwen 7B retrieval → Llama 70B) | jonathan's grid | 64.01 | 192 | 21% |
| Oracle: the cheapest correct path | | 81.5 | 205.5 | 22% |

- **Cheap retrieval + sampling:** 3B ×20 fused minus always-strong is −1.69 points [−4.19, +0.68], a tie. It ties
  both of jonathan's routers too, at 3–5% of their price. A tie is not equality: the interval allows always-strong to be
  up to 4 points better.
- **Routers on menu A** (AIQ over weak-first prices 80–1,009):

  | Router | AIQ | With the router's own price |
  |---|---|---|
  | E4B decider | 0.736 | 0.730 |
  | 2B decider | 0.731 | 0.730 |
  | SLR + reward | 0.726 | (uncharged) |
  | SLR label | 0.724 | (uncharged) |
  | nano-jev | 0.719 | 0.719 |

  The deciders beat jonathan's routers before their own price, and tie them with it.
- **Why nano-jev's evidence lags the 72B's:**
  - With jonathan's prompt and his 72B's selections, our 70B scores 76.23, against his 76.77.
  - With nano-jev's top 5 it scores 69.82: retrieval, not the answer prompt, is the gap.
  - The 72B writes about two question-focused references, and the answer is already in them for 85% of questions.
- **nonLLM's routers** use a split of the same sizes, but Config A's models, the grid's own lenient and their own cost
  scale, so they are not in this table.
- **selfLearnt** reports on AgentTTS's 500 questions, not yet on the golden set.

### Corrections to the 2026-10-17 snapshot

- **AgentTTS's cost.** "~938 units, ~101% of strong-everything" mixed scales. Table 5 is AgentTTS's normalised
  budget, 3u − 2 per call of u units, not units.
  - In AgentTTS units its best setting costs 302: 97% of strong-everything's 311.
  - At Table 5 prices it costs 859: 92% of 929.
- **3B ×40 fused.** "220 units" counted the prompt once per sample. Under AgentTTS's Eq. 8, samples from one prompt
  pay it once: 71 units.
- **"280× cheaper retrieval, 4–10× cheaper overall".** This compared units with Table 5 prices.
  - On one scale, nano-jev's ranking of the 100-passage pool costs 2.9 units against the 72B's 272.
  - At Table 5 prices it is 6.6 against 814.
  - The like-for-like comparison is the golden-set table above.

### Notes on comparing

- **Test sets:**
  - jonathan and systemOne share `slr-test`.
  - AgentTTS and selfLearnt's cluster plan use AgentTTS's 500. systemOne's earlier experiment 8 did too; see
    [its write-up](systemOne/docs/experiments/2026-10-18-common-redo.md).
  - nonLLM's routers use a 1,481-question split of the same sizes. Whether it is the same split was not checked here.
- **Rulers:** lenient, strict, graded and "corrected" accuracy differ by up to ~0.14 on the same answers (systemOne
  exp. 8). The grid's stored lenient also differs from the standard ruler: Config A's always-strong scores 59.55 on the
  standard ruler.
- **Cost:** Table 5 prices, AgentTTS units, pipeline units and cost indices are different scales.

### Sources

- **AgentTTS:** [the paper's code](https://github.com/FairyFali/AgentTTS).
- **jonathan:** [`jonathan/README.md` at `5b69cce`](https://github.com/mohameddhameem/STAIR/blob/5b69cce/jonathan/README.md).
- **systemOne:** [systemOne/RESULTS.md](systemOne/RESULTS.md) and
  [the golden-set write-up](systemOne/docs/experiments/2026-10-21-slr-standard.md).
- **nonLLM, at `f48aeae`:**
  - [routers](https://github.com/mohameddhameem/STAIR/blob/f48aeae/nonLLM/results/measured_v1/REPORT.md);
  - [sampling cohort](https://github.com/mohameddhameem/STAIR/blob/f48aeae/nonLLM/results/adaptive_sampling40_instruct_gguf/cohort_500/oracle_cost/comparison.md).
- **selfLearnt:**
  - [README at `c53035c`](https://github.com/mohameddhameem/STAIR/blob/c53035c/selfLearnt/README.md);
  - [Colab results at `0aa1743`](https://github.com/mohameddhameem/STAIR/blob/0aa1743/selfLearnt/colab_dev/README.md).
