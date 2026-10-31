# 2WikiMultiHopQA Pre-Registration — Stage-Level Routing
**Dated 31 October 2026 · DRAFT for Jonathan's line-by-line review — not registered, nothing launched, until he approves.**
Settings below were confirmed with Jonathan one by one on 31 Oct (§12 records each). Where a setting is still his to decide it is marked **[OPEN]**.

---

## 1. Purpose and the claims this dataset carries
2WikiMultiHopQA (2Wiki) is the second AgentTTS dataset, after HotpotQA, used to validate stage-level routing **without** the risk factor (the risk factor is validated on MedAgentBench and GPAgentBench-2K, where a high-risk tier exists). Two claims:
1. **Reproduction** — AgentTTS's published 2Wiki method is reproduced with their own code on their own instances, so the comparison row is theirs, not our reading of theirs.
2. **Routing vs sampling** — with the same four models, the same instances and the same scorers, a per-instance routing decision at **K = 1, greedy** is compared with AgentTTS's per-dataset choice plus test-time sampling (**×50 + fuser**). The only difference between the AgentTTS row and ours is the sampling; same pipeline, same models, same data, same rulers.
A third, descriptive claim: 2Wiki's four question types (compositional / comparison / bridge-comparison / inference) are reported separately, as the dataset's real difference from HotpotQA.

## 2. Data
- **Source file:** official 2Wiki `dev.json` (12,576 questions, all labelled), from `data_ids_april7.zip` (zip sha256 `95df2bf5…`; dev.json sha256 `79f77ae1…`), on the lab at `/mnt/hdd/jonathan/data/2wiki_official/`. Recorded in `common/stair_common/splits/MANIFEST.json` (commit 262dc65).
- **Why not AgentTTS's `tts_train.json`:** it is the official **test** split with answers withheld (first `_id` 8a65901a…); the registry committed over it was over unlabeled data and was corrected 31 Oct.
- **Team registry (commit 262dc65):** test = dev[0:500] · val = dev[500:1000] · train_extra = dev[1000:] minus every question sharing a gold Wikidata entity with test or val → **500 / 500 / 9,744**. Generated and checked by `common/tools/twowiki_entity_split.py`.
- **AgentTTS's 100 pooled instances** (`tts_test.json`) = dev[0:100] exactly, inside our test 500 — the "AgentTTS slice", as on HotpotQA. Fields: `question`, `answers` (gold + Wikidata aliases), `ctxs_candidate` (100–104 single-sentence passages), `ctxs_gt` (gold passages: 2 for 79 questions, 4 for 20, 6 for 1). Type mix of the 100: compositional 39 · comparison 25 · bridge-comparison 21 · inference 15.
- **What AgentTTS did not release:** their 2Wiki test file `tts_test_2_samples500.json` (loaded by `archon_main.py --test`) is absent from the repo. By their generator (`data/retrieval_QA_data_generator.py`) it would be dev[100:600] with pools we do not have. **Consequence, stated on every slide that uses it:** the 100 are AgentTTS's *search-training* set; their published 2Wiki number (0.72, Table 1) is on an unreleased set. The reproduction anchor is therefore their own reported value on these 100 (Fig. 4(c), training-size 100 point — value to be read off the figure and marked approximate), and the row is labelled **"AgentTTS (reproduced), their 100"**.
- **Pools beyond the 100** do not exist and are rebuilt by their construction (§7).

## 3. Models — one grid for everything
| Stage | Weak (W) | Strong (S) |
|---|---|---|
| Retrieval | Qwen/Qwen2.5-7B-Instruct | Qwen/Qwen2.5-72B-Instruct-AWQ |
| Answering (QA) | meta-llama/Llama-3.2-3B-Instruct | meta-llama/Llama-3.1-70B-Instruct (4-bit) |

Identical to the HotpotQA matched-model grid (Config B). AgentTTS's 2Wiki pick (72B ×1 retrieval → Llama-3.2-3B generation) is our **SW** cell with sampling added. All four checkpoints were used for the HotpotQA grid; the run script asserts each local path exists before anything launches (no assumption). Router base: Qwen/Qwen2.5-1.5B-Instruct, as on HotpotQA.

## 4. Row A — AgentTTS (reproduced), their 100
- **Code:** AgentTTS repo commit 3f0c2df, `code_others/archon_main.py`, unchanged (byte-faithful, as the HotpotQA rep40 was). Environment: the same `archon` environment used for the HotpotQA reproduction (path on the lab to be recorded here before launch).
- **Configuration — their own, from the command block in `archon_main.py`:** `--generate --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 --generate_model_name meta-llama/Llama-3.2-3B-Instruct --generate_samples 50` (test budget 4). We run it **without** `--test` so it loads `tts_test.json` (the 100). **×50**, not the ×40 of HotpotQA and not the "×5" on deck slide 212, which is wrong and will be corrected.
- **Sampling — theirs, unchanged:** retrieve T 0.9 / 512 tokens; merge T 0.7 / 512; generate T 0.7 / 512 × 50; fuser T 0.3 / 512. One pass; their seeds; no second seed.
- **Inputs:** their pools as shipped (no rebuild, no reshuffle); file sha256 recorded here before launch.
- **Outputs kept:** every sampled generation, the fused answer, the retrieval selections — so the row can be re-scored under any ruler without re-running.
- **Scoring:** their script prints "Gen Exact Match / Gen F1 / Ret F1" through `evaluation_2wiki.metric_evaluation(response, example['answers'])` — a file **absent from the repo** for both HotpotQA and 2Wiki. The author's statement (Fali Wang, 11 Aug 2026, reconfirmed 21 Oct) that this is a lenient contains-the-gold check, not strict EM, applies to 2Wiki by construction (same function). 2Wiki differs from HotpotQA in one respect: they pass the **alias list**, so any alias in the output counts. Our standard ruler is run over the same `answers` list (§6).

## 5. Our rows — same 100 (lab), then the registry split (cluster)
- **Four fixed paths WW / WS / SW / SS** — one call per stage, **K = 1, greedy**, no fuser. Prompts: our HotpotQA prompts with the short-answer line (the prompt register is the reference; byte-identical text, dataset name aside — any wording change is a registered deviation).
- **Router, zero-shot:** the HotpotQA label-free champion (fresh adapter, GRPO(Reward), reward router prompt — `grpo_ret_qa_a1_fresh_ep6_objective`, best adapters) applied to 2Wiki **without retraining**, at its two **frozen** operating points: greedy 0.50/0.50/0.50 and the cap point 0.80/0.05/0.05. One read on the 100; one read on the test 500 once labels exist. No threshold is chosen on 2Wiki for this row.
- **Costs:** AgentTTS's Eq. 4 unit budgets, 2Wiki entry = HotpotQA entry (`AgentTTS.py`: Retrieval 814, QA 115) → same certified `PATH` dict (imported, never retyped). Reported as % of always-SS, as on every board.
- **Per-type readout** (four types) for every scorer — reported, never used to tune anything.

## 6. Scorers — every row, the same four
| Column | Definition | Implementation |
|---|---|---|
| **Lenient** (headline, AgentTTS-comparable) | gold or any alias appears in the output, standard ruler | certified `stair_common.score.lenient` over the `answers` list |
| **Strict EM** | normalised output equals a gold string | certified `stair_common.score.em` |
| **QA F1** (their "Gen F1" / "F1") | token-overlap F1, best over golds | certified `stair_common.score.f1` |
| **Ret F1** (their "Ret F1") | set F1 between selected passage ids and gold passage ids | their `precision_recall_f1(ctxs_gt_ids, ids)` logic — `utils.py` is absent from their repo; re-implemented from the definition, stated as such |

Columns are labelled **Ret F1** and **QA F1**; "F1" alone is never printed. Alias-EM is **not** reported (decided 31 Oct).
**[OPEN — gold list for Strict EM and QA F1.]** The certified `score.py` gives every ruler the same gold list and takes the best over aliases (the HotpotQA convention). On 2Wiki that makes "Strict EM" alias-aware (a diacritic variant such as *Malgorzata* / *Małgorzata* counts). Proposed: keep the certified behaviour — one gold list for all four columns, no re-implementation — and say so in the column note. Alternative: Strict EM against the single `answer` string only. Jonathan to decide; the choice is recorded here before any scoring.

## 7. Pools for the rest of the registry (prerequisite for §8) — CPU, lab, this week
- **Construction = theirs** (`data/retrieval_QA_data_generator.py`): Elasticsearch `multi_match` over `title` + `text` of the sentence corpus built from the dev contexts, top-100 sentences per question, gold sentences appended if absent, shuffled; `answers` = gold + aliases from `id_aliases.json`.
- **Evidence the construction matches:** rebuild pools for the 100 as well and compare with the shipped pools — gold recovered 100/100 required; candidate overlap (Jaccard) reported. Pools for dev[100:600] are produced by the same run (their test 500, reconstructed — used only in the deferred Row C, §13).
- **Caveat on every use:** same method, not provably the same passages; ES version and analyzer recorded.

## 8. Label pass — the four paths on the registry split (cluster)
- **Order:** test 500 + val 500 first (~20 lab-card-hours equivalent); then **train_extra by position**, in blocks of 1,000 (the first 1,000 of the 9,744, then 1,001–2,000, …). Nothing in the training slice is chosen after seeing results.
- **How many training blocks:** decided by the HotpotQA learning curve (§9), not assumed — label up to the block where HotpotQA's val curve flattened, plus one block of margin. The number is written into this file before the training run.
- Outputs: per-instance four-path outcomes under all four scorers + costs → Excel for Jonathan's check before any training. Greedy decoding; determinism spot-check (5 instances twice, byte-identical) at phase start, as registered for MedAgentBench.

## 9. HotpotQA learning curve (decides §8's budget; lab; val only)
Train the HotpotQA champion recipe (fresh adapter, GRPO(Reward), same seeds, same steps-per-example) on the first **1,000 / 2,000 / 3,000 / 4,814** training instances by position; read each on **validation only** (composed lenient accuracy at cost, standard ruler). No test read is consumed. Stop rule for 2Wiki: the smallest size whose val accuracy is within **1 point** of the 4,814 run defines the flattening point.
**Prediction:** flattening between 2,000 and 3,000. If the curve is still rising at 4,814, 2Wiki labels all of train_extra.

## 10. Training on 2Wiki — two routers, each with one held-out test read
1. **Fresh adapter (primary):** the HotpotQA champion recipe unchanged — GRPO(Reward), fresh adapter, reward router prompt, same reward (R′ = lenient-ruler tokenF1 − 0.10·C/C_SS, risk term dormant), same G and schedule. Thresholds: STORE → SWEEP → LOCK on val, Rule B. One test read.
2. **HotpotQA-initialised adapter (secondary, if the cluster window allows before the deadline):** same recipe, initialised from the champion's adapters instead of fresh. One test read. Slides name it "HotpotQA-initialised adapter", never a code.
3. **Zero-shot champion** (§5) is the third point: no adaptation.

## 11. Predictions (written before any run)
- Row A lenient on the 100 ≈ AgentTTS's Fig. 4(c) training-size-100 value, within ±5 points (n = 100: one instance = 1 point); Row A strict EM ≈ 0 (prose outputs); Row A QA F1 low for the same reason.
- Our SS (K = 1 greedy) lenient on the 100 ≥ Row A − 3 points; SS > SW > WS ≥ WW on lenient; Ret F1 S ≫ W.
- Zero-shot champion: between best fixed path and oracle on accuracy at < 70 % of SS cost at the cap point; no claim of a beat.
- Fresh adapter > zero-shot on test 500; HotpotQA-initialised ≈ fresh (within 2 points), any gain on compositional / bridge-comparison, none on comparison / inference.
- Type gap: inference lowest lenient for every row.

## 12. Settings confirmed with Jonathan, 31 Oct 2026 (one by one)
1. Reproduction uses the **exact** AgentTTS instances (their 100) and their shipped pools. ✔
2. One set of settings and models for **all** our experiments — the HotpotQA grid (§3); only difference from AgentTTS is **K = 1 greedy** vs their K > 1 + fuser. ✔
3. K > 1 rows follow AgentTTS exactly — "×N + fuser" with N from their code per dataset: HotpotQA ×40, **2Wiki ×50**. ✔
4. Scorers: lenient and strict EM, **plus** QA F1 and Ret F1 because AgentTTS reports them; alias-EM dropped. ✔ (gold-list convention for Strict EM — **[OPEN]**, §6)
5. Label pass on the cluster; training-set size decided from a HotpotQA learning curve, not assumed. ✔
6. Fresh adapter primary; HotpotQA-initialised secondary (recommendation accepted 31 Oct; order confirmed). ✔
7. Dataset analysis precedes runs — done 31 Oct (profile, two real instances, deck slides 103–104). ✔
8. Prompts and Python reviewed line by line by Jonathan before any launch; no tokens or credentials in chat or files; commits only to branch `jonathan`. (standing rules)

## 13. Deferred, recorded so they are not forgotten
- **Row C:** Row A re-run on the reconstructed dev[100:600] pools — "AgentTTS test (reconstructed)" against the published 0.72. After the deadline.
- **`splits.py`** still regenerates the 2Wiki registry from `tts_train.json`; must be repointed to `dev.json` (Dhameem; Jonathan drafts the note).
- Deck corrections: slide 212 "×5" → "×50"; slides 98/101 TaskBench "gated" → "included — gates 1–3".
- TaskBench: official JARVIS evaluation check, 20-instance timing pilot, metric decision, sub-domain split — after 2Wiki.

## 14. Outcome log (appended as results arrive; §§1–13 are not rewritten)
*(empty)*

---
*Prepared 31 Oct 2026 from: AgentTTS repo 3f0c2df (`code_others/archon_main.py`, `AgentTTS.py`, `data/retrieval_QA_data_generator.py`), the corrected 2Wiki registry (commit 262dc65), `twowiki_profile.py` output on the lab, and the HotpotQA GRPO pre-registration (20 Oct) for the recipe, rulers and frozen champion points.*
