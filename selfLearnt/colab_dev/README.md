# Development history (Google Colab, 28 Sep – 13 Oct 2026)

This branch (`selfLearnt-colab-dev`) keeps the development work behind the self-learnt switching pipeline. Everything here ran
in Google Colab on one A100, with models and results on Google Drive. **To run the method, use the full-scale pipeline
on the `selfLearnt` branch (`selfLearnt/`)**; this folder is the evidence for how it was designed.

| Folder | Contents |
|---|---|
| `notebooks/` | The 33 Colab notebooks, with their outputs, as run |
| `code/` | The Python modules the notebooks call (copied from Drive), and `make_nbs.py`, which generates the notebooks |
| `analysis/` | Three CPU re-analyses run on 13 Oct from saved outputs (notebook 27 with a fixed decision rule; answer checks as a signal; 0.5B → 0.8B skipping 0.6B) |
| `docs/Experiment_Plan_and_Log.md` | The full plan and experiment log: design of every set written before its run, then its results |

Models, data and run folders (tens of GB) stay on Google Drive, under `0_Small_Big_Model_Switching/`.

## What each set asked, and what it found

Accuracy below is on HotpotQA (hard level), 20 candidate passages per question. Sets 1–4 use exact match on 200 test
questions. From Set 5 onward a corrected label is also used (exact match, token F1 ≥ 0.5, or the gold answer contained),
on 400 questions. "AgentTTS's pick" is the fixed setup the AgentTTS planner chose: Qwen 7B retrieves, Qwen 32B answers.

| Set | Notebooks | Question | Result |
|---|---|---|---|
| 1 | 00–06 | Does per-question switching (Qwen 3B / 7B) beat AgentTTS's single configuration? | No: best single setup 0.545 EM; router v1 0.41 |
| 2 | 04b, 06, 07 | Fixed router (2a); chain-of-thought answering (2b) | Router discriminates but does not beat the best fixed setup; CoT lifts the 3B by 0.08–0.18 EM |
| 3 | 08, 09 | Is the gap in answering or in retrieval? | Retrieval: with the right passages the 3B with CoT answers as well as the 7B; CoT closes a third of the retrieval gap |
| 4 | 10–12 | Broad sweep: 3 retrievers × 3 answerers × CoT × shortlist settings; AgentTTS with a Qwen 32B planner | Best fixed (Qwen 7B → Qwen 32B, CoT) 68% at 5.8x; planner's rules 60% at 6.7x |
| 5 | 13–20 | Can a router that reads the question, its metadata or hidden states beat a fixed setup? | Question text, topics, embeddings: at chance. A hidden-state probe on Llama 8B's answer matches AgentTTS's pick at −27% compute (corrected label) |
| 6 | 21 | Next-experiments plan: retry retrieval, revise answers, shorter context | One remedy helped: when the retriever probe flags missing passages, retry and let Qwen 32B answer (78.1% vs 76.8% at −8% compute); unstable in transfer |
| 7 | 22–24 | Fine-tune the retriever (LoRA) to shortlist the gold passages | Qwen 3B on 12,000 questions: 84.2% at −42% compute, the gold-passage ceiling; Qwen 7B no better. An answer gate on top trades accuracy for compute evenly |
| 7d | 25a, 25b | Can sub-1B retrievers do as well? | Qwen3.5 0.8B matches 3B (84.0%, −0.3 points) at a quarter of its retrieval compute; Qwen3 0.6B about 1 point behind |
| 8 | 26 | Qwen2.5 0.5B trained longer; cascade 0.5B → 0.6B → 0.8B; Llama → Qwen with votes | 0.5B reaches 82.8% passages found (0.8B 90.8%). Its confidence flags bad shortlists (0.90); escalating on it beats random (+1.4) but only matches 0.8B alone. Answer escalation and votes add nothing |
| 9 | 27 | Can the 0.5B learn where each question should go? Ensembles, calibrated escalation, Sure / Unsure | Learned route collapsed to the majority label: Llama's failures cannot be predicted before it answers (0.54–0.59). The Sure flag is honest (91.5% right when Sure) |
| 9 (CPU) | `analysis/` | Diagnosis of Set 9 | One bug (repetition penalty in generation) found but not the cause; with a fixed decision rule the learned start still matches random. Answer checks after Llama answers reach 0.76 and give the first answer switch that beats random |
| 10 | 28 | A self-learnt rulebook: soft rules tuned by micro-experiments, hard guardrails | Built and tested on saved outputs (starting rulebook +4.6 points over random escalation); not run in Colab (compute units ran out). Moved to the full-scale pipeline on the `selfLearnt` branch |

## Headline results (400 test questions)

All rows come from notebook 26's run on the same 400 HotpotQA test questions (20 candidate passages each).

| Retriever → answerer | What it represents | Lenient accuracy² | Exact match | Cost index¹ |
|---|---|---|---|---|
| Untrained Qwen 7B (step by step, picks 3 passages, one try) → Qwen 32B | AgentTTS-style fixed pick (stand-in from our scaled-down search, Colab model menu) | 76.8% | 57.5% | 100 |
| Fine-tuned 3B → Qwen 32B | Fine-tuned retriever, largest answerer in the Colab menu | 84.5%³ | 64.2%³ | 57.6 |
| Fine-tuned 0.8B → Qwen 32B | Fine-tuned retriever, largest answerer in the Colab menu: best observed balance | 84.0% | 63.5% | 43.5 |
| Fine-tuned 0.6B → Qwen 32B | Fine-tuned retriever, largest answerer in the Colab menu: cheaper | 83.2%³ | 62.7%³ | 42.2 |
| Fine-tuned 0.5B → Qwen 32B | Cheapest fine-tuned retriever, largest answerer in the Colab menu | 79.5% | 60.0% | 41.7 |
| Fine-tuned 0.8B → Llama 8B | Fine-tuned retriever, small answerer | 77.2% | 54.8% | 16.9 |
| 0.5B; if unsure → 0.6B → 0.8B; Qwen 32B answers | Confidence-triggered retrieval switching (49% escalated; +1.4 points over random escalation of the same share, 95% interval +0.6 to +2.3) | 83.5% | 63.2% | 44.9 |
| Qwen 72B → Llama 70B | Strongest at both stages (full AgentTTS menu) | Not run in Colab: cluster workstream 2 | – | – |
| Qwen 72B ×1 → Llama 3B ×40 + fusion | AgentTTS's own HotpotQA pick (their code; 0.742 reported) | Not run in Colab: cluster workstream 1 | – | – |

¹ Estimated inference compute (2 × parameters × tokens, retrieval and answering) relative to the stand-in fixed pick.
Training is excluded.
² Our corrected label: exact match, or token F1 ≥ 0.5, or the gold answer contained in the answer. Yes/no answers must match
exactly.
³ Re-merged and re-run in notebook 26. The earlier runs gave 84.2% / 64.0% (3B, notebook 23a) and 83.0% / 62.5% (0.6B,
notebook 25a).

- On 400 questions, differences under about 1.5 points aren't reliable. The 3B, 0.8B and 0.6B rows are statistically tied:
  the 0.8B is −0.3 points from the 3B (−1.3 to +0.8).
- Every fine-tuned retriever from the 0.6B upwards beats the stand-in by 6–8 points at 42–58% of its compute.
- The last two rows will come from the cluster run, scored on AgentTTS's 500 test questions with 100 candidates each.
  Those numbers won't be directly comparable with the rows above.

## Running a notebook again

Each notebook was written for Colab with a Google Drive folder `MyDrive/0_EngD_assignments/0_Small_Big_Model_Switching`
holding `code/`, `models/`, `data/` and `runs/`. Notebooks check that Drive serves the exact code version they were made for
(SHA-256), copy the code locally, and resume where they stopped. `code/make_nbs.py` regenerates all notebooks.
