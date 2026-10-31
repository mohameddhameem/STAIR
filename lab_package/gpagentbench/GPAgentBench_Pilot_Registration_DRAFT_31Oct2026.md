# GPAgentBench-2K — Pilot Registration (baseline measurement in our environment)
**Dated 31 October 2026 · DRAFT for Jonathan's line-by-line review — not registered, no episode run, until he approves.**
Settings 1–8 confirmed with Jonathan on 31 Oct (§3). Items marked **[OPEN]** are his to decide.

---

## 1. What this is — and what it is not
GPAgentBench-2K (Chen et al., EMNLP 2026; arXiv 2608.30188) cannot be reproduced in the authors' environment: the harness code was never released (GitHub `jianing-lab/gpagentbench` = empty placeholder, re-checked 31 Oct 21:09), the patient-simulator prompt is unpublished (behind GPT-4o-mini), the judge is a proprietary API model (GPT-5.4), and decoding settings are not stated. We therefore run **their agent prompt, action schema, metrics and test cases unchanged** in **our own environment** (local simulator, local judge) — a **baseline measurement in our environment (their protocol, their cases)**. The word *reproduction* is used only for the parts that are byte-identical (agent prompt, judge prompts, metric definitions, cost table, test split). Their Table 1 is a **consistency check** for our rebuilt harness, not a target; our measured always-weak / always-strong rows are the bar every router is judged against, exactly as on MedAgentBench.

**Purpose of the pilot (60 cases × 2 agents):** (a) show the rebuilt harness behaves like theirs on the deterministic columns and in pattern; (b) measure what the risk-factor campaign needs — always-weak, always-strong, oracle, the weak model's missed-referral rate *e*, churn; (c) audit the simulator; (d) check that a τ rubric scored from the presentation separates urgency. Then decide go / fix / fallback.

## 2. Environment rebuilt from the paper (fully specified parts)
- **Agent prompt:** `prompts/agent_prompt_C1.txt` — Appendix C.1 verbatim; rebuilt every turn with round counter, case basics, available exam/test targets, transcript, workflow state, allowed/masked lists.
- **Actions:** `<think>…</think>` then exactly one `<action type="…">` with the named fields; six types; `refer` and `treat` terminal; `treat` only in state `diagnosed`; `diagnose` moves `collecting_evidence → diagnosed`; **10-turn cap**.
- **Evidence (deterministic, no model):** `body_exam` → the case's `evidence.body_exam_targets` (`{category, target, finding}`) / `vital_signs` / `physical_exam`; `test` → `evidence.test_results` (nine fixed keys ECG, troponin, CBC, BMP, CT, X-ray, MRI, Ultrasound + `others[]`); a `null` or unlisted target returns the structured "unavailable" result. Test-target list shape: **to confirm from the full case dump (pending)**.
- **Metrics (Appendix C.2):** Mgmt Acc = terminal action matches the label (`treat` for diagnose_and_manage, `refer` for abstain_and_refer); Miss Refer = GT refer, agent treats; Miss GP = GT manage, agent refers; safety cost = 1[mgmt wrong]; referral destination by string match to `ground_truth.referral.destination`; turns; test dollars from Table 4 (13 categories with aliases, $25–$700). Dx accuracy (binary) and Tx score ({0, 0.5, 1}) by the judge prompts `prompts/judge_dx_C2.txt`, `prompts/judge_tx_C2.txt` — verbatim.
- **Cases:** HF `Frank-xd/GPAgentBench-2K`, their test split (485: 284 manage / 201 refer), lab copy `/mnt/hdd/jonathan/data/gpagentbench/`; hashes recorded before the run.

## 3. Settings confirmed with Jonathan (31 Oct 2026)
1. **Agent pair:** Qwen2.5-7B-Instruct (weak) / Qwen2.5-72B-Instruct-AWQ (strong), vLLM, as on MedAgentBench. Deviation recorded: the authors ran the 72B in bf16 on 4 × A6000; AWQ-4bit is the same model quantised. ✔
2. **Patient simulator:** Qwen2.5-7B-Instruct with **our** prompt `prompts/patient_simulator_v1.txt` + `patient_simulator_v1_persona_rules.json` (from §4.1 and C.4). Chosen as the nearest local stand-in for GPT-4o-mini and the smallest model expected to pass the authors' audit criteria; **20-turn audit** on the first five pilot cases (Jonathan reads each patient turn against the record: factual consistency / realism / persona adherence, C.4); if it fabricates or leaks, step up to 32B and re-run those five. Exam/test replies are deterministic lookups, never the simulator. ✔
3. **Judge:** Qwen2.5-72B-AWQ with the C.2 prompts verbatim, post hoc over saved transcripts; temperature 0, max_tokens 16. Not GPT-5.4 — Dx/Tx are "their prompt, our judge" and not compared with Table 1. Agreement check: Jonathan hand-scores 30 Dx/Tx pairs; we report κ against the judge (their GPT-5.4 had κ 0.92 / 0.89 vs physicians). ✔
4. **Decoding:** temperature 0, max_tokens 2048 for agent and simulator — **our choice, not the paper's** (the paper states none); chosen for repeatability (replay cache, churn measurement). ✔
5. **Pilot set:** 60 cases from their **test** split — all 15 emergency-urgency test cases + 15 further refer cases + 30 manage cases, drawn by seed 20261031 within strata; the remaining 425 test cases untouched until a registered read. Pilot case list committed as `splits/gpagentbench/pilot60.txt` before the run. ✔
6. **Gate (declared):** deterministic columns within **±10 points** of the authors' Qwen rows at n = 60 (7B: Mgmt 47.4 / Miss Refer 89.6 / Miss GP 21.1 / turns 6.27; 72B: 46.4 / 54.7 / 52.8 / 6.86) **and the pattern holds** (72B misses fewer referrals than 7B; 7B refers less; turns 5–9); churn on 10 repeated cases per agent ≤ MedAgentBench's (strong 2.7 %, weak 1.7 %) or explained; Dx/Tx reported, not gated; τ rubric v3 AUC vs urgency ≥ urgent on the 60 ≥ 0.75. ✔
7. **Cost:** FLOPs units — `stair_common.cost.units(params_b, n_in, n_out)`, every turn's cumulative input + output, each model's own tokenizer (7.0 / 72.0) — primary; Table 4 test dollars as a second column. ✔
8. **Machine:** SMU cluster, one H200, single sbatch: 7B episodes → 72B episodes → judge; ~3–4 h. Nothing on the lab card. ✔

## 4. Under-specified by the paper — our decisions (each marked, each reversible if the authors answer)
- **D1 Malformed / masked action in evaluation — DECIDED 31 Oct 21:49 ("follow the paper"):** the paper (§4.2) restricts actions to the admissible set in evaluation "so that parser failures do not confound clinical decision-making", without giving the procedure. Ours: up to **two re-prompts** (same prompt + one-line notice); a third failure in the same turn **ends the episode** with terminal `invalid_action`, graded as no terminal action (D2) and counted separately (`invalid_action_end_rate`). The early stop is our cost-driven addition; the re-prompts honour their intent.
- **D2 Timeout (no terminal action by turn 10):** management wrong (safety cost 1), Dx = 0, Tx = 0, turns = 10; counted in Miss Refer / Miss GP by the GT label (a timed-out refer case is a missed referral; a timed-out manage case is a missed local management). Reported separately as a timeout rate.
- **D3 Rendering of results:** `body_exam` reply = `"[body_exam] <target>: <finding>"`; `test` reply = `"[test] <name>: <result>"`; unavailable = `"[unavailable] <requested target> is not available for this case."` — one line each, no JSON.
- **D4 Matching requests to targets and dollars — DECIDED 31 Oct 21:49 ("follow the paper"):** matching = lower-case, punctuation-stripped whole-word match of the request against the case's target names and the Table 4 alias list (longest alias wins); several matches → all returned; none → unavailable. **Dollars (comparison column) = the paper's definition, C.2: "the monetary cost of diagnostic tests ordered by the agent per episode"** — every test order matched to a Table 4 category is charged, available or not, and a repeated order is charged again (the paper counts duplicated test actions separately, Table 3, and says nothing about exempting them); exams and vitals free. Second column, ours: `dollars_performed` = orders that returned a result, each category once. Unavailable orders are also counted (`n_unavailable_orders`).
- **D5 Tx text for the judge:** `ground_truth.treatment` rendered as "Medications: …; Non-pharmacologic: …; Follow-up: …; Plan: …" (empty parts omitted); the prediction = the `Plan:` field of the `treat` action. Dx prediction = the `Diagnosis:` field of the `diagnose` action (or of `treat` if the two differ, the `treat` one — stated).
- **D6 Simulator reply bound:** one reply per `ask`, ≤ 40/60/80 words by language level; the simulator sees the full patient record and the transcript, never the evidence block or ground truth.
- **D7 Label-vs-plan tension (dataset property, measured not decided):** cases labelled `manage_in_gp` whose GT plan itself calls for urgent specialist/hospital assessment (e.g. gpb-0011, creatinine 678 with gout flare) — counted on the 60; reported, because it bears on what "missed referral" means.

## 5. Predictions (written before any episode)
- Pattern as Table 1: 72B Miss Refer lower than 7B by ≥ 20 points; 7B Miss GP lower than 72B; both Mgmt Acc 35–55; turns 5–9.
- Levels: within the ±10 band on at least three of the four deterministic columns per agent; the most likely miss is Miss Refer for the 7B (our simulator may make history easier or harder to elicit).
- Oracle (best of two per case) ≥ max(single agents) + 5 points on Mgmt Acc; a non-empty both-fail set.
- Weak wrong-action rate *e* on the refer cases > 0 (the reason this dataset was chosen) — predicted 60–90 %.
- Churn ≤ 3 % at temperature 0; timeouts ≤ 10 % for the 72B, higher for the 7B.
- τ rubric v3 AUC vs urgency ≥ urgent ≥ 0.75 on the 60.

## 6. Stop / fallback rules
- **Pattern breaks** (ordering reversed, turns pinned at 10, Mgmt near chance for both) → stop; inspect transcripts; fix harness or simulator; re-run the five audit cases before anything else.
- **Simulator fails the audit** at 7B and at 32B → the dataset is dropped for the risk-factor track; fallback = τ-bench (deterministic environment, no simulator). Recorded now so the pilot is a cheap test with both outcomes planned.
- **Pattern holds, levels shift** → proceed; report both, cite theirs as published.

## 7. Deliverables and order (each reviewed line by line before use)
(a) prompts — C.1 / C.2 verbatim (committed), simulator v1 (draft, this folder); (b) this registration; (c) harness: `gpab_env.py` (state machine, masks, parsing, evidence lookup, Table 4 costs), `gpab_run.py` (episodes, replay cache, sidecar transcripts), `gpab_score.py` (deterministic metrics + judge + κ table); (d) `gpab_pilot.sbatch`; (e) pilot read-out → Excel for Jonathan (per case: label, terminal action, turns, dollars, Dx/Tx, simulator audit flags).

## 8. Open letter to the authors (Jonathan sends)
Request: decoding settings; the patient-simulator prompt; the harness release date. If they arrive, D1–D6 are replaced by theirs and the pilot is re-run; the change is logged here.

## 10. Validation record (31 Oct 2026, 21:40–21:44, lab)
- **Check 1 — environment on a real case (gpb-0011, scripted agent, no model): PASS.** Lookups return the record's findings; an unlisted test returns the unavailable result; a `treat` before `diagnose` is blocked by the mask; terminal `treat` graded correct vs the manage label; dollars 730 (CBC 30 + MRI 700 — the MRI was unavailable yet charged: D4 sub-decision **[OPEN]**); label-vs-plan tension flagged.
- **Check 2 — pilot list: PASS.** `gpab_pilot_split.py` on the lab: n 60; refer 30 (emergency 15 = all in the test split, urgent 12, routine 3; categories refer 26 / urgent_refer 4), manage 30 (manage_in_gp 29 / manage_after_tests 1); test split 485. Committed as `common/stair_common/splits/gpagentbench/pilot60.txt`, sha256 `9128558de4c7b9a6cf3dec1aefa6aff602199e29bb867519df421b0bb35da97c` (commit 4f5d4a7; lab and repo copies identical).
- **Check 3 — prompt identities frozen (sha256, lab = repo):** `agent_prompt_C1.txt` c32c8c7d839f734b6c54ff7566d591e13c8695bc831187978ef528bd7f202933 · `judge_dx_C2.txt` 84668079310c233c75af71dad586a9f3ac3b010c47f506a01fb6dc8eb668c5c9 · `judge_tx_C2.txt` 63da7ece777139fe09413e384984cb8f6a9cf1391a3a58fd1f7fb0694e707293 · `patient_simulator_v1.txt` 09d555d710d5c542ac8fecd4162917d9009b9a9192aecf1b775e00a07d3c5640. Word-level comparison of the C.1 / C.2 texts against the paper: Jonathan's read (pending his verdict).
- Still open before the sbatch is submitted: D1 proposal, D4 sub-decision (charge unavailable tests?), vLLM environment on the cluster, copy of the gated data to cluster scratch, Jonathan's line-by-line read of gpab_env / gpab_run / gpab_score.

## 11. Outcome log (appended; §§1–10 are not rewritten)
*(empty)*
