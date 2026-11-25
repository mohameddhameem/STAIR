# Report changes: System One sections (Dhameem)

## 26 Nov 2026: final pass on the team's Overleaf version

The files here are now the team's Overleaf version (as of 26 Nov), with the changes below.

- **Bibliography (the "TODO" author labels in the PDF):**
  - Overleaf still had the old placeholder entries. Now merged: the Overleaf bib is the base (`iec60812` and every other
    team entry kept).
  - 12 entries replaced by the verified versions from 19 Nov: `r2reasoner2025`, `routingsurvey2026`, `nanojev2026`,
    `strandsdecider2026`, `dekoninck2025unified`, `he2025cognify`, `yue2025masrouter`, `liao2025rsd`,
    `pan2025specreason`, `shao2025dot`, `kadavath2022know`, `bajaj2016msmarco`.
  - `chen2023frugalgpt` changed to `@article`, so its venue (TMLR) prints. No TODO is left in the bib.
- **Review markup removed:** the teal `\upd` / `\updnote` / `updblock` notes and their definitions are gone; the text
  is unchanged. `\ifshowupdates` stays, because Andy's purple notes use it.
- **Confirmed:** the team's edits to our parts. Table 1's new cost basis (task-model compute as % of Strong alone,
  decision compute reported separately) gives 79.4 (HotpotQA) and 72.6 (MedAgentBench = 59.0 × 1.23) for the
  System One row. Both are correct.
- **Small content fixes:**
  - **Notes to Table 1:** the judges' own compute is now reported, as the caption promises (E4B three passes +20.2
    units = 2.2 points, 81.6% with them; the point needs the three-seed ensemble; nano-jev's price rounds to zero).
  - **MedAgentBench judge table and paragraph:** the ×1.23 conversion to Table 1's basis (59.0 → 72.6), and the GRPO
    54.1 marked as "of always-escalate".
  - **§4.2:** "trained on the same per-stage labels as SFT (label) *on HotpotQA*". The implementation appendix now
    states the MedAgentBench label.
  - **Compute appendix:** our TODO is replaced. The cluster keeps no per-job accounting history, so per-run times stand
    in for a GPU-hour total.
- **TODOs left for teammates:**
  - Jonathan: early grid card (compute appendix);
  - Jonathan: 72B-AWQ serving card;
  - Andy and Ziming: GPU-hours.

## 19 Nov 2026: System One sections filled

Changes to `Final_Report.tex` and `final_refs.bib`. Teammates' sections are unchanged, except for two items flagged
below. (On 19 Nov every change was marked in teal in the PDF; that markup was removed on 26 Nov.)

## Updated (corrections to existing text)

| Where | Before | Now | Why |
|---|---|---|---|
| **Table 1**, System One row, HotpotQA cost | 79.4% | **81.6%**, with a † footnote | 79.4% left out the judge's own three passes, but the caption counts routing overhead. The point also needs the 3-seed ensemble; no single seed reaches it. |
| **Table 1**, System One row, trained parameters | "LoRA" | **109M** \| 33M | Counted from the published E4B adapter (104.6M LoRA + 3.9M head). nano-jev is fully fine-tuned (33.4M). |
| **§6 Results**, "Routers reach always-Strong accuracy…" | 79.4% for the E4B judge | 81.6% (its own passes included) | Same fix as Table 1. |
| **§4.2**, System One judge paragraph | Judge reads qp "before the Weak run" | Rewritten for the Weak-first, per-stage framing | Now says what each stage's question reads (HotpotQA retrieval / answer; MedAgentBench task + Weak attempt). It also says the judge uses the same per-stage labels as SFT (label), is averaged over seeds and pays for its own compute. |
| **App., "Judges vs. routers"** | "once their price is counted all are at 0.730"; both judges "edge the LLM routers" | Corrected, with paired intervals | With price the judges fall to 0.730 and **tie** the LLM routers (0.724 / 0.726, router compute uncharged). The 2B beats only the label router before price. The published retrain supports "ties or narrowly beats". |
| **App. Implementation**, Strands Decider | "pointer head (≈1M)" | 1.05M (2B), 3.9M (E4B); LoRA 16.8M / 104.6M | ≈1M was the 2B's head only. |
| **Related Work**, "Repeated steps" (shared with Ziming) | Longer TRIM sentence | Shortened, same meaning | Saves a line; the Intro already says TRIM is evaluated on math reasoning. |

## Added

- **Related Work:** a new paragraph, *Escalation signals and decision models*. It places the System One judges against
  the Weak model's own signals (P(True), semantic entropy, self-verification), Self-RAG / Adaptive-RAG, and small
  decision models (nano-jev, Strands Decider).
- **Reproducibility Statement:**
  - the STAIR URL and commit;
  - the System One pre-registration (operating points and split hashes committed before each test read);
  - the published judges on Hugging Face and the code repo.
- **Appendix, System One judges** (replaces the TODO):
  - the AIQ definition;
  - **Table "System One judges on HotpotQA":** main executors and the preliminary 1.5B/7B pair, with AIQ, AIQ with
    price and operating points;
  - **Table "System One judges on MedAgentBench, and judge cost":** canonical and pure-temperature tables, zero-shot
    judges, then parameters, GFLOPs, latency and price;
  - paragraphs: *Preliminary pair*, *MedAgentBench* (incl. the paired comparison with the GRPO router), *What the typed
    decisions add* (the pre-registered MuSiQue / 2Wiki study: what was confirmed, what failed), *Contamination*.
- **App. Implementation:** the training protocol on both benchmarks (labels, epoch selection, seeds, threshold rules,
  bootstrap).
- **App. Compute:** one sentence on where and how long the judges train.
- **App. Negative findings:** three System One items (typed head and few labels; risk control; judges escalating
  everything at the LLM routers' operating rules).

## Removed

- **Related Work:** the TODO on "conformal". This paragraph never used the word, and our only conformal-style result
  (Learn-then-Test risk control) is negative. So please **drop "conformal" from the slide's gap wording** too.
- **Reproducibility:** the `public URL / commit hash` TODO (filled).
- **Appendix:** the `Dhameem: move the full judge table here` TODO (filled).
- **App. Negative findings:** the bullet "LLM fusion … hurt small models; the typed decision head did not beat a plain
  classifier head" is split in two. The fusion half is **unchanged**; the typed-head half moved to its own System One
  item.

## Bibliography (`final_refs.bib`)

Not highlighted in the PDF. Every entry below was checked against its publisher's page or a citation index.

- **TODO / placeholder entries fixed:**
  - `routingsurvey2026` = Moslem & Kelleher, *Dynamic Model Routing and Cascading for Efficient LLM Inference: A
    Survey*, TMLR 2026. It contains the exact "largely unexplored" sentence we quote.
  - `r2reasoner2025`: authors and the arXiv title (*Route-and-Reason*).
  - `he2025cognify`: authors; arXiv 2502.08056.
  - `pan2025specreason`: NeurIPS 2025, authors.
  - `dekoninck2025unified`: ICML 2025, PMLR 267.
  - `yue2025masrouter`: ACL 2025, all authors.
  - `liao2025rsd`: all authors.
- **Author list completed:** `shao2025dot` (WWW 2025).
- **Added:** `nanojev2026`, `strandsdecider2026`, `kadavath2022know`, `farquhar2024semantic`, `asai2024selfrag`,
  `jeong2024adaptiverag`, `trivedi2022musique`, `ho2020constructing`, `angelopoulos2021ltt`, `bajaj2016msmarco`.
- **Still open:** `ahn2026atf` (Jonathan's working paper).

## For teammates to check (not edited)

- **Jonathan, Table 1:** the LLM router costs (89.9 / 69.7) don't include router compute, although the caption says
  "including routing overhead". Either charge it or reword the caption.
- **Jonathan, Table 1 footnote ‡:** "77 + 3 = 80 units \todo{confirm}" is confirmed by the golden-set ladder (WW = 80,
  8.6% of 929), so the TODO can go.
- **Owner of "LLM fusion of sampled answers hurt small models":** this seems to conflict with experiment 8. There,
  Llama-3.2-3B with 20 fused samples on nano-jev's top 5 scores 75.08 at 3.4% of always-Strong's price, tying
  always-Strong. Is the claim about a different (smaller) model?
- **§6, MedAgentBench p-values:** we hold per-task decisions for both the judges and the LLM routers. A paired McNemar
  test against always-Strong can replace the unpaired p = 0.20 / 0.10 without another test read. Dhameem can run it.
