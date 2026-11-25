# Fixed-budget allocation over a bundle of queries — registration (DRAFT, 17 Nov 2026, 20:05 SGT)
ERP I component requested by Prof. Archan (item 2, email of 17 Nov): "how to perform overall allocation of a fixed token/cost budget
(a cumulative constraint) given an expected bundle (and distribution) of queries, as opposed to optimising the cost function for a
single query." Datasets: HotpotQA and MedAgentBench only (Jonathan, 17 Nov 18:26). Status: registered before any run; script
`budget_allocation.py` v1.1 awaiting Jonathan's line-by-line read; nothing has been executed on real data.

## 1. Research question (Jonathan, 17 Nov 19:54)
How to improve accuracy within a fixed budget equal to the always-strong cost — and must the whole budget be spent to reach the
highest accuracy, or is the peak reached below it? Baselines: always-strong (100 % spend) and the oracle (unconstrained ceiling), plus
the oracle under the same budget.

## 2. Formulation
Bundle of N queries, actions a ∈ {W, S} per stage (plus H on MedAgentBench's high-risk tier). Maximise Σᵢ E[correctᵢ(aᵢ)] subject to
Σᵢ costᵢ(aᵢ) ≤ B. Lagrangian relaxation ⇒ per-query rule "maximise p̂ᵢ(a) − λ·cᵢ(a)"; with a calibrated score p̂ᵢ = P(weak is enough) this
is a threshold: deliver weak iff p̂ᵢ ≥ θ; higher θ ⇒ more escalation ⇒ more spend. Our routers already decide through such thresholds
(R′ = correctness − 0.10·cost fixes λ for every query). The budget component sets θ for the bundle. Accuracy as a function of spend
need not be monotone: escalation lowers accuracy wherever the weak path is right and the strong path wrong (task-1 reversal on
MedAgentBench; 4–6 % of HotpotQA instances) — hence the question whether the peak lies below B.

## 3. Inputs (all stored; no model is run; no new test exposure beyond re-reading declared test outputs under new policies)
- **HotpotQA**: `runs/config_b/router_data_standard.jsonl` (test 1,481 / val 400; `em_standard` = [WW, WS, SW, SS] lenient), prices
  [80, 195, 894, 1009], C_SS 929 (Table 5 convention). Router = GRPO (Reward) + three insights (the F6g read, 11 Nov): TEST scores
  `test_scores/ret_grpo_fresh_guided.jsonl`, `qa_grpo_fresh_guided.jsonl`; VAL scores `grpo_ret_qa_a1_fresh_ep6_guided/ret_val_scores.jsonl`,
  `qa_val_scores.jsonl`. One θ drives the three gates (ret, qa|W, qa|S) — the one-parameter family through the greedy point.
- **MedAgentBench**: `runs/mab/training1_table.json` (test 150 / val 60; qw, qs, cw, cs), `tau_scores_v2.json` (tier τ ≥ 0.8),
  Training 1 router TEST `training1_router/test_scores.jsonl` (pDeliver), VAL `training1_router/val_scores.jsonl`. deliver = cw,
  escalate = cw + cs (weak-first), hand-off = cw (the weak attempt is already run; human time is a second resource, not FLOPs).
  Training 3 (round-2 router) as a sensitivity if its val/test scores are on disk.
- Validation files are REQUIRED: θ* and the pacing curve come from validation only; nothing is tuned on a bundle's outcomes.

## 4. Design
- **Bundles**: 1,000 bootstrap samples with replacement; N = 200 (HotpotQA) / 100 (MedAgentBench); a second run with `--shift`
  (Dirichlet(1,…,1) re-weighting of groups: HotpotQA question type if present in the rows, MedAgentBench task type) = "the expected
  distribution of queries" varying from the validation mix.
- **Budgets**: B ∈ {40, 50, 60, 70, 80, 90, 100, 110, 120} % of the bundle's always-strong cost; B = 100 % is the headline.
- **Policies**: spend-all (largest θ with spend ≤ B) · spend-to-peak (θ* = validation argmax accuracy, tie → cheaper; capped by B;
  remainder banked; MedAgentBench: remainder's instances not escalated, instead up to K_H ∈ {0, 5, 10, 15} hand-offs to the tier's
  lowest-pDeliver queries, person assumed correct — Training 6 convention, stated) · per-query (θ₀ = 0.5 greedy, blind to B) · pacing
  (online: θ_t = min(θ*, largest θ whose validation spend ≤ remaining budget per remaining query)) · oracle_B (greedy knapsack on true
  outcomes within B) · random_B · oracle (unconstrained) · always_SS.
- **Endpoints**: primary — accuracy at B = 100 % of spend-all vs spend-to-peak (if spend-to-peak ≥ spend-all, spending it all is not
  optimal); s* = spend at θ* and the accuracy there vs at 100 %; MedAgentBench: tier failures with no human at B = 100 % by K_H.
  Secondary — violation rate P(spend > B) and overshoot of per-query under shift; regret of pacing to spend-to-peak and to oracle_B;
  the full accuracy–spend curve over θ (bundle means) with its peak.
- **Reporting**: one figure per dataset (accuracy vs spend: curve + policy points at B = 100 %, with always-strong, oracle_B, oracle as
  reference lines); one table at B = 100 %; the violation plot under shift. SEs are over bundles (bootstrap SE, not instance SE).

## 5. Predictions (written before the run)
- HotpotQA: θ* on validation lies at a spend of 60–75 %; the test curve plateaus at ≈ always-strong (76.8 ± 1) from ~70 % spend to
  100 %; spend-to-peak ≈ spend-all ± 1 at B = 100 % (within noise) — the budget's last 25–30 % buys no accuracy; best achievable at
  B = 100 % is always-strong + 0–2. oracle_B at 100 % ≈ unconstrained oracle (≈ 86) since the oracle costs ~35 %.
- MedAgentBench: θ* at a spend of 50–60 %; the curve PEAKS (≈ 60–62) and FALLS toward 54 at 100 %; spend-all at B = 100 % ≈ 54–56,
  spend-to-peak ≈ 60–62 (+5 or more over spend-all) — spending it all on strong calls is wrong here; with K_H ≥ 10, tier failures with no
  human → ≈ 0 and accuracy rises further; oracle_B at 100 % = unconstrained oracle 63.3 (its cost ~53 %).
- Per-query (θ₀ = 0.5) violates B = 60 % on a large share of shifted bundles on both datasets (its realised spend varies with the mix);
  pacing loses ≤ 2 pts to spend-to-peak at B = 100 % and ≤ 3 pts at tight budgets.

## 6. Decision rules
- "Spend it all?" answered per dataset by the primary endpoint at B = 100 % (bootstrap SE over bundles): spend-to-peak > spend-all by
  more than 2 SE ⇒ NO (peak below budget); |difference| ≤ 2 SE ⇒ indifferent on accuracy, the remainder is pure saving; spend-all >
  spend-to-peak by > 2 SE ⇒ YES (not predicted; would mean the validation θ* under-escalates).
- The cumulative-constraint claim is made only from spend-to-peak / pacing (which satisfy B by construction) against per-query's
  measured violation rate; no post-hoc θ.
- Stop rule: if the validation and test curves disagree in shape (peak position differs by > 20 spend points), report both and do not
  claim a peak location — the router's calibration, not the allocator, is then the finding.

## 7. Three checks before the run (as for every run): sha of the script and of every input file recorded in the output manifest; a
smoke run with `--bundles 20` whose bounds order holds (oracle ≥ oracle_B ≥ allocators ≥ random; always_SS = spend-all at B = 100 %);
validation files present (the script refuses otherwise). Outputs: `budget_<dataset>.json`, `_per_bundle.csv`, `.xlsx` (Jonathan's check
copy), under `runs/budget/<dataset>[_shift]/`.

## 8. Smoke results and AMENDMENTS (17 Nov 2026, 21:33–21:45 SGT; 20 bundles each on the lab; numbers are checks, not results)
- Checks passed: script sha ce79d1c5… on the lab = repo; validation files present (`grpo_ret_qa_a1_fresh_ep6_guided/{ret,qa}_val_scores.jsonl`;
  `training1_router/val_scores.jsonl`); bounds order holds; always_SS = spend-all at B = 100 % on both datasets.
- Full-test anchors (CPU): HotpotQA always-strong 76.77, four-cell oracle **81.5** (Config B 7B/72B, lenient) — the deck's "+9.1 at a
  third of the cost" belongs to the preliminary 1.5B/7B pair; the ERP paper must cite 81.5 for this pair.
- Finding 1 — **saturation**: Training 1's pDeliver has 39 distinct levels on test (44 of 150 ≥ 0.999, 50 ≤ 0.001 → 63 % saturated;
  val 23 levels, 37 of 60). Smoke: θ* (exact argmax) = 1.0 → 95.6 % spend, 63.05 vs greedy 55.8 % / 62.05 — the curve is flat
  (+1.0 pt for +40 cost points). A threshold family needs a ranking; a saturated router offers few levels, so bundle allocation
  cannot place spend finely. **Amendment A:** the allocation router on MedAgentBench is **Training 3 (round-2; ECE 0.04, AUC 0.91)**
  with its own table (`training3_table.json`, escalation = cw2 + cs) and score files (`training3_router/{val,test}_scores.jsonl`, on
  disk); Training 1 is kept as the contrast that shows the saturation effect. Script v1.2: `--run training3_router` (table auto-selected),
  `score_levels` diagnostic in the manifest.
- Finding 2 — **exact argmax over-spends**: HotpotQA θ* = 0.95 → 90.6 % spend for 75.45 vs the registered cap-70 point 75.15.
  **Amendment B (tie rule made operational):** θ* = the CHEAPEST θ whose validation accuracy is within ONE validation instance of the
  maximum (1/400 = 0.25 pt HotpotQA; 1/60 = 1.67 pt MedAgentBench). Script v1.2 `theta_star(vcurve, n_val)`.
- HotpotQA behaves as predicted: accuracy rises to always-strong (75.47 = SS on the bundles) and not above; spend-all vs spend-to-peak
  0.02 apart at B = 100 %. Expected answer: the peak is the bar, reached at ~70 % with Amendment B; the remainder is saving.
- Predictions unchanged except: MedAgentBench's primary comparison is now made with Training 3 (prediction: θ* at 55–70 % spend;
  spend-to-peak > spend-all at B = 100 % by > 2 SE; curve falls toward 54 at 100 %); Training 1 reported as the saturated contrast.
- Script v1.2 sha f0b2f302… (committed 17 Nov 21:50). Real run after Jonathan's read: both datasets, plain and `--shift`, 1,000 bundles;
  MedAgentBench with `--run training3_router --kh 0 5 10 15` and `--run training1_router` as contrast.

## 9. Deferred extension — K > 1 as an allocation option (Jonathan, 17 Nov 21:40: "we can use K > 1 within fixed budget, not just only
selection of strong or weak. This can be next experiment.")
Under a cumulative budget, k weak calls with a vote become one more purchasable action per query (1 weak · k weak + vote · 1 strong, each
priced); the allocator still decides which resource each query gets — right routing over a richer menu, consistent with the 13 Nov scope
decision that rejected sampling as a substitute for routing. NOT part of ERP I. Order after 27 Nov: 2Wiki first (sampling cells already
scripted in `twowiki_newpaths_cells.py` §15-B, never run; AgentTTS evidence that weak×50 beats always-strong there), then decide on
HotpotQA (AgentTTS ≈ always-strong → little expected) and MedAgentBench (k weak episodes ≈ one strong at k ≈ 5; stochastic
trajectories, vote over episodes — methodologically different). Mentioned in the ERP I proposal (possible methods) and final report
(future work) as the extension of the budget component.

## 10. FIRST FULL RUN (17 Nov 2026, 21:46 SGT; script v1.2 f0b2f302…; 1,000 bundles; lab) — read, and AMENDMENT C
**HotpotQA (N = 200; val 1,110 scored; θ* = 0.92 → 86.1 % val spend).** Curve: 71.67 @ 39.5 % · 73.75 @ 49.7 · 74.79 @ 59.4 · 75.56 @
69.3 · 76.38 @ 79.3 · 76.62 @ 89.0 · 76.69 @ 100 (= always-strong 76.69; test-curve peak 76.69 @ 91.9 %). At B = 100 %: spend-all 76.69
@ 100 % vs spend-to-peak 76.54 @ 85.5 % (Δ 0.15, bootstrap SE ≈ 0.14 → indifferent) ⇒ **highest accuracy = the bar; 14.5 % of the
budget is pure saving.** oracle_B = oracle 81.41 @ 21.5 % (the Config B four-cell ceiling). Per-query θ₀ = 0.5: 74.35 @ 56.7 %; violates
B = 40 / 50 % on 100 / 99 % of bundles, B = 60 % on 14 %. Pacing = offline allocator within 0.1 pt; its violation rate at tight budgets
(15–64 %) is the price of not seeing the bundle (overshoot ≤ 0.07 %). `--shift` was a no-op (rows carry no question type → one group).
**MedAgentBench, Training 3 (N = 100; θ* = 0.33 on 60 val instances).** At B = 100 %: spend-all 54.04 @ 100 % (= always-strong) ·
spend-to-peak 59.97 @ 91 % · registered greedy θ = 0.5 (per-query) 63.34 @ 93.7 % = test oracle 63.34 · with the saved budget turned
into hand-offs (K_H = 5 / 10 / 15): **64.94 @ 77.6 % · 69.12 @ 67.2 % · 71.48 @ 62.8 %, tier failures with no human 13.9 → 9.0 / 4.8 /
2.4**; oracle_B with hand-offs 68.3 / 73.1 / 76.3. **Spending the whole budget on strong calls is the WORST policy at B = 100 %.**
Shifted bundles: same picture (spend-to-peak 59.4; θ = 0.5 62.95). **Training 1 contrast:** θ* = 1.0 (deliver only p = 1.000) → 62.66 @
94.9 %; greedy 0.5 61.34 @ 54.4 %; nothing placeable in between (saturation); with K_H = 15: 71.56 @ 76.0 % — the hand-off effect is
router-independent; the saturation shows as coarse spend placement.
**Caveats found in the read:** (i) MedAgentBench spend was measured against the router's escalation price (cw2 + cs for Training 3), not
always-strong (cw + cs) — hence θ = 0.5 at "93.7 %" where the October read reported 81.5 %. **Amendment C (v1.3, sha 9075fa2b…):**
budget denominator = always-strong cw + cs per instance for both routers; the 'always_SS' row = escalate-all at the router's own
escalation price. All MedAgentBench spend figures above shift down (~12 pts for Training 3); the ordering is unaffected. Re-run required.
(ii) θ* from 60 validation instances did not transfer on Training 3 (val 0.33 → test 59.97 vs greedy 0.5 → 63.34): the registration's
stop-rule case for the peak LOCATION — reported as such; the peak's EXISTENCE (63 → 54 as spend → 100 %) is not in doubt. (iii) HotpotQA
shift needs a question-type key; pending the row keys.

## 11. SECOND RUN — MedAgentBench on the corrected cost axis (17 Nov 2026, 21:53 SGT; v1.3 sha 9075fa2b…; 1,000 bundles; N = 100) — FINAL
Denominator = always-strong cw + cs. Check: the registered greedy point θ = 0.5 (per-query) reads **63.34 @ 81.4 %** = the October held-out
read (63.33 @ 81.5 %). **Training 3, B = 100 %:** spend-all (= escalate-all at round 2) 54.04 @ 86.9 % · spend-to-peak (val θ* = 0.33)
59.97 @ 79.0 % · θ = 0.5 63.34 @ 81.4 % · spend-to-peak + K_H = 5 / 10 / 15 hand-offs **64.94 @ 67.4 % · 69.12 @ 58.4 % · 71.48 @ 54.6 %**,
tier failures with no human 13.9 → 9.0 / 4.8 / 2.4 · oracle (two models) 63.34 @ 44.2 % · oracle_B + 15 hand-offs 76.28 @ 43.4 % (1.0
tier failures). Pacing = spend-to-peak exactly at B = 100 %. **Shifted task mix:** spend-to-peak + 15 → 68.77 @ 60.0 % (SE 0.37); θ = 0.5
62.95; same ordering. **Training 1 (contrast)** unchanged from §10 (its escalation price = the denominator): 62.66 @ 94.9 %; + 15 hand-offs
71.56 @ 76.0 %. **Answer of record (MedAgentBench):** the accuracy peak lies at ≈ 45–80 % of the always-strong budget; spending the
remainder on strong calls lowers accuracy to always-strong's 54; spending it on hand-offs raises accuracy to 71.5 at 55 % of the budget
and cuts unseen high-risk failures by 83 %. Decision rule §6: spend-to-peak (with or without hand-offs) > spend-all by ≫ 2 SE ⇒ NO, do not
spend it all. Caveat (ii) of §10 stands: θ* from 60 validation instances (59.97) under-performs the registered greedy point (63.34) by 3.4
pts — both reported; the peak's existence is not in doubt. Pending: HotpotQA `--shift` with question types (v1.4, `--data tts_test.json`).

## 12. HotpotQA SHIFT RUN (17 Nov 2026, 21:57 SGT; v1.4 sha 3ae99642…; `--data tts_test.json`; groups = HotpotQA's own labels, bridge
1,184 / comparison 297; 1,000 bundles, N = 200) — EXPERIMENT CLOSED
Under a Dirichlet-shifted type mix: always-strong 76.13 · spend-to-peak 75.93 @ 76.1 % · pacing 75.93 @ 76.1 % · per-query (θ₀ = 0.5)
74.18 @ **45.8 %** (plain mix: 56.7 %) — the fixed per-query threshold's realised spend moves 11 points with the mix and violates B = 60 %
on 12.3 % of shifted bundles (overshoot 0.34 %); bundle allocators 0 %; pacing 25 % at B = 60 % with overshoot 0.02 %. Oracle 80.15 @
17.5 %. Same curve shape as the plain run (§10).
**Results of record for ERP I (both datasets, B = always-strong cost):**
- HotpotQA: highest accuracy = the always-strong bar (76.7); the router reaches it at ~85 % of the budget (within 0.15 pt from ~80 %);
  the remainder buys no accuracy. Cumulative-budget compliance requires a bundle-level threshold (offline or paced); the per-query
  threshold cannot guarantee it under a changing mix.
- MedAgentBench: highest accuracy is reached at ~45–80 % of the budget (63.3 at the registered router point, 81 %); spending the rest on
  strong calls lowers it to 54.0; spending it on hand-offs raises it to 71.5 at 55 % of the budget with unseen high-risk failures 13.9 →
  2.4 per 100 tasks; robust to the router (Training 1: 71.6) and to the task mix (68.8).
- Caveats carried into the write-up: θ* from a 60-instance validation set under-performs the registered greedy point by 3.4 pts on
  MedAgentBench (both shown); the person is assumed correct (Training 6 convention); bootstrap SEs are over bundles drawn from one
  held-out set, not independent samples; all inputs are stored outputs of declared test reads (a second analysis of the same outcomes).
Output folders (lab): `runs/budget/{hotpotqa, hotpotqa_shift_v14, mab_t3_v13, mab_t3_v13_shift, mab_t1_v13}` (+ `mab_t1` from §10, identical).

## 13. K > 1 FOR THE WEAK ANSWERER WITHIN THE FIXED BUDGET — HotpotQA (registered 17 Nov 2026, 22:40 SGT; DRAFT until Jonathan's read; nothing launched)
**Rule (Jonathan, 21:45):** the strong model is called at most once per stage; if the weak model is chosen it may be called k times,
k ∈ {1, 5, 10, 20, 40}, answers combined by majority vote (no fuser). **Scope (21:41):** HotpotQA and MedAgentBench for ERP I; the K > 1
cells are made on HotpotQA only — on MedAgentBench a weak "call" is a whole agent episode (k = 5 ≈ one strong call; voting over
trajectories is a different method), so MedAgentBench stays W / S / H. **Budget (22:29):** the allocation is read inside the fixed budget
= the always-strong cost of the test set, exactly as on slide 150; the cells are the measurement step only.
**Cells** (`hotpotqa_samplek_cells.py` v1.0 + `hotpotqa_samplek.sbatch`; Config B runner unchanged; weak answerer Llama-3.2-3B): for every
question of slr-val (1,110) and slr-test (1,481), WW×40 (stored weak-retriever picks → 40 sampled weak answers) and SW×40 (stored
strong-retriever picks → 40 sampled weak answers); one batched generate per question per context; temperature 0.7, top-p 0.9 (OUR settings;
nothing to match), seed 20261117 + position; every sample stored. Smoke `--end 10` first. ~1–1.5 h on one H200 per split.
**Scoring** (`hotpotqa_samplek_score.py` v1.0, CPU): vote@k over the first k samples (normalize_answer; tie → earliest), lenient ruler
against the gold aliases (= em_standard's ruler); reference WW/WS/SW/SS on the same questions; prices by the Table 5 per-call
convention — WW×k = 77 + 3k, SW×k = 811 + 3k (WW 80, WS 195, SW 894, SS 1,009) — measured tokens alongside. Equal-price comparisons the
scorer prints: WW×40 (197) vs WS (195); SW×40 (931) vs SS (1,009).
**Allocation** (budget_allocation.py extension, after the cells exist): per question the menu is weak×k (k from the ladder) or strong×1,
each priced; the family "deliver weak×k iff the router's score ≥ θ, else strong" with one (k, θ) per set chosen on validation subject to
the budget; plotted as five curves (k = 1 = today's slide 150) on the same axes; fixed paths WW×k and SW×k as points; oracle over the
extended menu. Budget = always-strong cost of the test set (100 %); every reported point ≤ 100 % spend.
**Predictions:** vote@k lifts the weak answerer by a few points and saturates by k ≈ 10–20; SW×40 < SS (AgentTTS reproduced ≈ SS on
HotpotQA); WW×40 < WS at equal price; at the always-strong budget the best k > 1 family gains ≤ 1 pt over k = 1. Expected answer: on
HotpotQA, more weak calls do not substitute for one strong call — measured, not argued. Decision rule: if any k > 1 family beats k = 1 at
B = 100 % by > 2 bootstrap SE, K > 1 enters the menu for HotpotQA; otherwise it is reported as a negative with the equal-price pairs.
**Settings to confirm one by one (not assumed):** (1) temperature 0.7 / top-p 0.9; (2) K = 40 with the ladder {1, 5, 10, 20, 40}; (3) the Table
5 per-call price of 3 units per weak answer call as the headline accounting (measured tokens as the alternative).
- **17 Nov 2026 22:35 — §13 SETTINGS CONFIRMED by Jonathan one by one:** (1) temperature 0.7 / top-p 0.9 (22:33), temperature sweep only as a
  conditional ablation if K > 1 shows a gain; (2) K = 40, ladder {1, 5, 10, 20, 40} (22:34) after the budget check — SW×40 = 931 < 1,009 per
  question, 92 % of the set's budget if every question took it, ceiling k = 66; (3) per-call price 3 units per weak answer call as the headline
  (the harsher price for K), measured tokens alongside (22:35, "OK if this is within the budget" — it is, see (2)).
- **17 Nov 2026 22:48 — §13 inputs staged on the cluster.** First smoke (job 367962) cancelled before start: `~/slr/hotpotqa_stores/` held only
  `retr_strong.jsonl`. `retr_weak.jsonl` located on the lab at `runs/config_b/` (6.6 MB, 8 Oct) and copied via laptop; shas on the cluster = lab:
  retr_weak 50296feb…, retr_strong be53e8af… (same build, same 110-passage pool). No H200 free at 22:47; the smoke queues overnight.
- **17 Nov 2026 22:49 — §13 SMOKE SUBMITTED by Jonathan (cluster job 367972; `SPLIT=test ARGS="--end 10"`, OUT `samplek_smoke`; pending, priority).** Pass
  criteria: 20 rows in `samples_test.jsonl` (10 × {W, S}) with 40 samples each; `n_seq_out` 40, 0 truncations; samples read as short answers. Then
  `SPLIT=val` and `SPLIT=test` full runs into `samplek_k40` (same OUT, no ARGS).
- **18 Nov 2026 08:57 — §13 SMOKE PASSED (job 367972, COMPLETED 1 m 17 s, 17 Nov 22:51).** 20/20 rows (10 × {W, S}), 40 samples each, `n_seq_out`
  40, 0 truncations, 1–4 s per batched call; samples are short HotpotQA answers; on easy questions all 40 agree (vote = greedy), so K > 1 can
  gain only where the weak answerer is uncertain. Measured output ≈ 3–7 tokens per sample (far below the 3-unit per-call price). Full splits
  `SPLIT=val` / `SPLIT=test` into `samplek_k40` to be submitted by Jonathan.
- **18 Nov 2026 08:58 — §13 FULL RUNS SUBMITTED by Jonathan: job 368176 (`SPLIT=val`, 1,110) and 368177 (`SPLIT=test`, 1,481), OUT `samplek_k40`; both
  pending (priority). Logs under `runs/hotpotqa/samplek_smoke/` (submission directory).**
- **18 Nov 2026 11:55 — §13 CELLS COMPLETE (jobs 368176 val, 368177 test; ~1.1 s per question on one H200).** `samplek_k40/`: samples_val 2,220 rows,
  samples_test 2,962 rows (= 1,110 × 2 and 1,481 × 2), 40 samples each, 0 truncations. Packed (455 KB) for the lab; scoring and allocation next.

## 14. §13 READ — K > 1 FOR THE WEAK ANSWERER, HotpotQA (18 Nov 2026, 12:03 SGT; cells `samplek_k40` 368176/368177; `hotpotqa_samplek_score.py`; `budget_allocation.py` v1.6 `--full --samplek --k-weak k`) — CLOSED, NEGATIVE
**Vote@k (test 1,481; val 1,110 in brackets):** WW×1 50.17 (49.82) · ×5 50.24 (50.54) · ×10 50.84 (50.72) · ×20 50.64 (51.17) · ×40 **50.44** (51.71); greedy
WW 50.30 (52.25). SW×1 61.72 (60.27) · ×5 61.17 (62.25) · ×10 62.05 (62.97) · ×20 61.99 (63.06) · ×40 **62.12** (63.15); greedy SW 61.78 (63.15).
Majority voting converges to the weak model's own greedy answer — it removes sampling noise, it does not add knowledge. **Equal-price pairs
(Table 5 per-call):** WW×40 at 197 units = 50.44 vs WS at 195 = **64.01** (one strong answer wins by 13.6 pts); SW×40 at 931 = 62.12 vs SS at
1,009 = 76.77. Measured output ≈ 3–7 tokens per sample (per-call is the conservative price).
**Allocation within the always-strong budget (`--full`, one bundle = the test set):** IDENTICAL for k ∈ {1, 5, 10, 20, 40} — spend-to-peak
76.64 @ 85.5 %, spend-all 76.77 @ 100 %, per-query (θ = 0.5) 74.48 @ 56.8 %; only the oracle moves (81.50 → 82.11 / 81.84 / 81.77 / 81.77).
Cause, verified on the stored scores: the router's chosen paths contain NO weak answer — θ = 0.5: WS 53.5 % + SS 46.5 %; θ = 0.92: WS 18.0 % +
SS 82.0 % (WW = SW = 0 %). The router's only live decision on HotpotQA is the retrieval choice; the weak answerer is never trusted, and the
vote results show that judgement was right. **Decision rule (§13): no k > 1 family beats k = 1 ⇒ K > 1 does not enter the HotpotQA menu;
the conditional temperature ablation is not triggered.** Statement of record: *within the always-strong budget on HotpotQA, more weak calls do
not substitute for one strong call; the strong call is the only purchase that moves accuracy — the opposite of 2Wiki, where AgentTTS's 3B×50
beats the strong model; the dataset decides.* k = 1 run reproduces slide 150 (76.64 / 76.77 / 74.48 on the full set vs 76.54 / 76.69 / 74.35
bundle means). Note: the `--full` run shows `groups {'all'}` because `--data` was not passed (type is irrelevant without `--shift`).
**Budget component CLOSED for ERP I.** Deliverables next: slides (proposal addition; final report section incl. K > 1 negative), paper §.

## 15. TEMPERATURE ABLATION for K = 40 (registered 18 Nov 2026, 12:10 SGT; Jonathan 12:06 "can we sweep temperatures to check if we can improve
accuracy within the budget?"; on the LAB GPU, 12:06 "can we do this in the lab?")
Design: temperatures {0.3, 1.0, 1.3} added to the 0.7 of §13; top-p 0.9, K = 40, both contexts, both splits; `hotpotqa_samplek_cells.py` v1.1
`--runner lab` (same script; the lab's `run_labels4.py` = same API) via `hotpotqa_samplek_lab_sweep.sh`; 10-question smoke first. Selection rule:
the temperature with the best vote@40 on VALIDATION, read once on test; allocation as §13/§14 with `--samplek` from that temperature.
Predictions: higher temperature raises diversity but lowers per-sample accuracy — vote@40 moves by at most ±2 pts vs 0.7; the 13.6-pt equal-price
gap to one strong answer (WW×40 vs WS) does not close; no k > 1 family beats k = 1 within the always-strong budget at any temperature. Decision
rule as §13. If a temperature beats 0.7 by > 2 pts on validation vote@40 (not predicted), it is read on test and the allocation re-run with it.
- **18 Nov 2026 12:12 — WRONG-MODEL INCIDENT (caught at the smoke; no result affected).** The first lab smoke printed `[weak] Qwen/Qwen2.5-1.5B-Instruct
  loaded`: the lab runner `run_labels4.py` defaults `WEAK_MODEL` to Qwen2.5-1.5B and `QA_WEAK = os.environ.get("SLR_QA_WEAK", WEAK_MODEL)`; v1.1 of the
  cells script and the sweep driver never set `SLR_QA_WEAK`. The Config B cells were produced on the CLUSTER (`run_labels_agenttts.py`, `QA_WEAK =
  meta-llama/Llama-3.2-3B-Instruct`; `configb_stores.tgz` in `runs/config_b/` is the cluster bundle), so no lab Config B run had ever exercised this
  default. The sweep (T = 0.3 val) had started on the wrong model: stopped by Jonathan; `samplek_k40_t0.3` and `samplek_lab_smoke` deleted (`ls`
  showed only `samplek_k40` afterwards). Id confirmed from the lab's own AgentTTS reference (`agenttts_archon_hotpotqa_reference.py` lines 659–670:
  `--generate_model_name meta-llama/Llama-3.2-3B-Instruct`).
- **FIX — `hotpotqa_samplek_cells.py` v1.2 (sha 80868c0bf477…), `hotpotqa_samplek_lab_sweep.sh` (sha 4a625e962b07…), commit bc8b709:** hard guard
  `assert "llama-3.2-3b" in L.QA_WEAK.lower()` before any model load; the driver requires `SLR_QA_WEAK` to be exported (`${SLR_QA_WEAK:?…}`) instead
  of relying on the runner default. Only the weak answerer is loaded by this script (retrieval = stored Config B picks; strong model never called), so
  `SLR_QA_WEAK` is the only model variable that matters.
- **18 Nov 2026 12:19 — RE-SMOKE PASSED** (`--end 10`, T = 0.3, `SLR_QA_WEAK=meta-llama/Llama-3.2-3B-Instruct`): guard passed, `[weak]
  meta-llama/Llama-3.2-3B-Instruct loaded` (cached; 65 s), `SAMPLEK DONE`; 20 rows, 40 samples per cell; position 4 ctx W: first five samples all
  "Greenwich Village" (low-T agreement as expected). Smoke folder deleted.
- **18 Nov 2026 12:20:52 — SWEEP STARTED** on the lab (tmux `sweep`; log `runs/hotpotqa/sweep_18nov.log`): `hotpotqa_samplek_lab_sweep.sh 0.3 1.0 1.3`,
  val then test per temperature, OUT `runs/hotpotqa/samplek_k40_t{0.3,1.0,1.3}`. Input shas identical to the §13 cluster cells: `tts_test.json`
  f65d5813…, `retr_weak` 50296feb…, `retr_strong` be53e8af…; lab runner `run_labels4.py` 318fb60c…. The only variable across the four temperatures is T.
  Expected ≈ 5–6 h total. Next: `hotpotqa_samplek_score.py --cells <folder>` per temperature (inside slr-env), selection on validation per the §15 rule.
- **18 Nov 2026 12:50 — PRICE CORRECTION to §13/§14 (found while drafting the ERP I slide; my error).** `budget_allocation.py` prices the paths
  weak-first: WW 80 = 77 + 3, WS 195 = 77 + 118, SW 894 = 77 + 814 + 3, SS 1,009 = 77 + 814 + 118 — a strong-retrieval path includes the weak
  retrieval call that ran first, so "strong retrieval" costs **891**, not 811. `hotpotqa_samplek_score.py` v1.0 used `S_RET = 811`, so every SW×k
  price in §13/§14 is 80 units low. Corrected SW×k = 891 + 3k: 894 / 906 / 921 / 951 / **1,011** for k = 1 / 5 / 10 / 20 / 40 (v1.0 wrote 814 / 826 /
  841 / 871 / 931). WW×k = 77 + 3k is correct. **Consequences:** (1) the §13 budget check "SW×40 = 931 < 1,009" was wrong — SW×40 at 1,011 exceeds one
  SS call by 2 units; SW×20 (951) is the largest k inside it. (2) Accuracies unchanged (prices do not enter the vote). (3) Allocation results of §14
  unchanged: the router selected no WW/SW path at any k, so no selected cost used the wrong price; only the oracle's COST at k > 1 can move by a few
  units (its accuracy cannot). (4) The §14 reading is strengthened: SW×40 is now both dearer than SS and 14.65 pts worse (62.12 vs 76.77); the equal-
  price pair becomes SW×20 at 951 = 61.99 vs SS at 1,009 = 76.77. Fix: scorer v1.1 (sha below), `budget_allocation.py` docstring/help; the
  `samplek_summary.json` / `samplek_outcomes.jsonl` of the T = 0.7 cells to be RE-SCORED on the lab (seconds) before any number is quoted; §14's
  statement of record stands with the corrected prices. ERP I slides carry the corrected prices.

## 16. AMENDMENT D — THE BUDGET IS THE STRONG MODEL ALONE (Jonathan, 18 Nov 2026 12:53: "spent cost should be more than 100 % of strong, not 86.9")
**What was wrong.** v1.3–v1.6 priced the budget weak-first: HotpotQA 1,009 = 77 (weak retrieval, run first) + 814 + 118; MedAgentBench cw + cs per task.
Under that convention "escalate every task" can never exceed 100 % by construction (it read 86.9 % on MedAgentBench only because the Training 3
table's weak pass cw2 is cheaper than cw). But a deployer who always uses the strong model does not run the weak model at all, so the budget
Prof. Archan's question refers to is the **strong model alone**: **929 per HotpotQA question (814 + 115 — the trainer's C_SS; corrected 13:30 from 932, which had left the weak answer call of 3 units in)** and **cs per MedAgentBench task**. The
weak pass is overhead the router must pay for out of that budget; escalate-all pays weak + strong on every instance and therefore reads **> 100 %**.
**Change — `budget_allocation.py` v1.7 (sha 4f3199b9bd7a…): `--budget strong-only` (new default) | `weak-first` (reproduces v1.3–v1.6).** Only the
denominator `css` changes (HotpotQA PRICE[3] − 80 = 929; MedAgentBench cs); per-path prices, outcomes, thresholds, θ* rule, policies unchanged.
Consequences by arithmetic before any re-run: HotpotQA always-SS/escalate-all = 1,009 / 929 = **108.6 %** (inadmissible); every earlier spend %
multiplies by 1,009 / 929 = 1.086 → spend-to-peak 85.5 → **92.9 %**, per-query 56.8 → **61.7 %**, oracle 21.5 → 23.4 %; "spend-all" becomes the
largest θ whose spend ≤ 929 per question (a new point, < always-SS). MedAgentBench rows scale by (cw + cs)/cs per task (table on the lab).
**Predictions.** HotpotQA: spend-to-peak reaches the always-strong accuracy (76.6–76.8) within the strong-only budget at ≈ 93 %; the budget-feasible
"spend-all" point sits ≤ 0.3 pt below always-SS; the answer "the whole budget need not be spent" stands. MedAgentBench: the router's point and the
hand-off rows stay below 100 % (they were ≤ 81 % weak-first and cw ≪ cs); escalate-all > 100 %; the hand-off rows remain the only policies above
the two-model oracle. **Re-runs (lab, CPU, beside the GPU sweep), outputs in NEW folders, old folders kept:** `hotpotqa_full_d` (`--full`),
`hotpotqa_d` (bundles N = 200 × 1,000, figure), `mab_t3_d` (`--run training3_router --kh 0 5 10 15`, 1,000 bundles), `mab_t3_full_d` (`--full`).
Slides 150–151 and the paper section are rebuilt from these; §10–§12 numbers remain on record as the weak-first convention.

## 17. ERP I EXPERIMENT PLAN — ROUTING WITHIN THE BUDGET, COST AS A CONSTRAINT (Jonathan, 18 Nov 2026 13:05–13:28; "keep the same routing method but the cost is the constraint, not optimization factor"; "this is only for ERP I")
**Scope:** HotpotQA and MedAgentBench; budget = the strong model alone on the whole held-out set (Amendment D). Routing method unchanged; the only
change is the cost component of the reward. Whether this enters the paper is decided later.
**(1) K = 1 — reward without the cost term.** R′ᵢ = Qᵢ(token-F1) − max(0, Cᵢ/C_SS,ᵢ − 1) [− λt·hand-off term, MedAgentBench only]. The per-instance
guard max(0, Cᵢ/C_SS,ᵢ − 1) is kept (Jonathan 13:25: "we shouldn't avoid over spending compared to the budget"); in the trainer's prices it is 0 on all
four K = 1 paths (WW 80, WS 195, SW 894, SS = C_SS 929) and bites only on W×k paths (SW×k > 929 for k ≥ 13). The budget then enters ONLY at inference,
as the set-level threshold (`budget_allocation.py --budget strong-only`). Trainer: `lab_package/hotpotqa/trainers/grpo_trainer_v17_lam.py` v1.7
(sha 0f45a36a740a…) = the three-insights trainer with a `--lam` flag (default LAMBDA 0.10, old commands reproduce identically) and the guard; the
λ-free run uses `--lam 0`. **Before launch on the lab:** diff the lab's trainer file that produced `grpo_ret_qa_a1_fresh_ep6_guided` against the repo's
v1.5 (the lab also had a "v1.6 twoinsights" variant) — v1.7 must be a 7-line diff on THAT file; Jonathan's line-by-line read; three checks; register.
Command (to validate): identical to the three-insights launch — `--baseline` OMITTED (the 0.6892 bar is a cost-inclusive R′ and is not comparable;
the bar for this run is the validation always-SS mean token-F1 / lenient, read from the trainer's own eval lines) `--from-scratch --epochs 6 --prompts
guided --lam 0 --outdir /mnt/hdd/jonathan/runs/config_b/grpo_ret_qa_a1_fresh_ep6_guided_lam0`; seed 17, G = 8. ≈ 2–3 h on the lab GPU, after the
temperature sweep. Then: store → one validation sweep → `budget_allocation.py --budget strong-only --full` with the new tag; one declared held-out read.
**Predictions (K = 1, λ = 0):** the router reaches the always-strong accuracy within the strong-only budget at a spend close to the current router's
spend-to-peak (≈ 93 %); it does not beat always-strong by more than noise (the two-model oracle bounds it at 81.5). Risk: on the ~52 % of questions
where all four paths tie on F1 the λ = 0 reward has zero gradient, so the router's scores may rank those questions less usefully for the set-level
threshold — if its spend at the bar is HIGHER than the λ = 0.10 router's, the cost term was a useful tie-breaker and that is the reported finding.
**(2) K > 1 as a routing action (HotpotQA; after (1)).** At the answer stage the router chooses among {W×1, W×5, W×10, W×20, W×K_max, S×1}; the strong
model is never repeated. Reward as in (1) (the guard now matters), trained from the STORED vote@k outcomes (no new model calls); budget = the set-level
constraint at inference. Settings to be confirmed one by one before the code is written: K_max (40 = stored samples; 50 needs a new sampling run),
temperature (best validation vote@40 from §15), action set. **Prediction:** gain ≤ +0.6 pt over K = 1 (the enlarged-menu oracle's headroom, §14).
**Slides:** 150–152 rebuilt from the strong-only re-runs (§16) and (1)–(2); V01-084's 151–152 are placeholders in the weak-first convention until then.
- **18 Nov 2026 13:31 — §16 WHOLE-SET RUNS UNDER THE STRONG-ONLY BUDGET (lab, v1.7 sha 74776cac125a…, `--full`; new folders `hotpotqa_full_d`,
  `mab_t3_full_d`).** Header `budget strong-only`, css 929 / cs. **HotpotQA (n = 1,481):** always-SS 76.77 @ 100 (baseline; the router's escalate-all
  path costs **108.6 % — infeasible**, violation 100 %) · spend-all **76.77 @ 99.9 %** (largest feasible θ = 0.96) · spend-to-peak **76.64 @ 92.9 %**
  (val θ* 0.92; test spend 92.87) · per-query θ = 0.5 74.48 @ 61.7 % · pacing 76.64 @ 92.9 · oracle 81.50 @ 23.3 % · random_B 74.61 @ 100. Prediction
  (§16) met: bar reached inside the budget at ≈ 93 %; nothing above it. **MedAgentBench (n = 150):** always-strong 54.00 @ 100 · escalate-all
  **106.9 % — infeasible** · spend-all **62.67 @ 99.7 %** (+8.67 vs always-strong, within budget) · spend-to-peak 60.00 @ 97.3 % (val θ* 0.33) ·
  **per-query θ = 0.5: 63.33 @ 100.2 % — 0.2 % OVER the strict budget (violation)**, so the registered router point is admissible only through the
  set-level threshold · test-curve peak θ 0.39: 63.33 @ 100.23 % (over) · oracle 63.33 @ 54.0 % · hand-offs (`--kh` counts are per the 150-task set:
  5 / 10 / 15 = 3.3 / 6.7 / 10 per 100) spend-to-peak + 5 / 10 / 15 → 63.33 @ 87.8 / 66.67 @ 77.5 / 70.00 @ 69.0; tier failures with no human 21 → 16 /
  11 / 6; oracle_B + 15 = 73.33 @ 54.0. Prediction (§16) met: router point and hand-off rows < 100 %; escalate-all > 100 %; hand-off rows the only
  policies above the two-model oracle. Next: `mab_t3_full_d2` with `--kh 0 8 15 23` (≈ 5 / 10 / 15 per 100) and the two 1,000-bundle runs for the
  figure (`hotpotqa_d`, `mab_t3_d`); slides 150–151 rebuilt from these.
- **18 Nov 2026 14:20 — ROUTER CORRECTION + v1.8 (Jonathan 14:12: "we need to use 75.89 at 69.7 % with GRPO (Reward), label-free").** The HotpotQA
  budget runs so far used the THREE-INSIGHTS router (`grpo_fresh_guided`, F6g: greedy 74.48 @ 61.7 %, cap 75.15 @ 68.3 %). The paper's headline
  router is the LABEL-FREE CHAMPION (fresh adapter, goal-stated prompt, `grpo_ret_qa_a1_fresh_ep6_objective`; test_scores tag `grpo_fresh_objective`;
  28 Oct read: greedy 74.21 @ 54.2 %, locked cap point t = 0.80/0.05/0.05 → **75.89 @ 69.7 %**, evaluation accounting with SS = 929). ERP I slides
  150–151 switch to the champion; F6g runs stay on record as the 17 Nov sensitivity. **v1.8 (sha 2eed562c7aa6…): `--theta-locked T_RET T_QAW T_QAS`**
  adds one policy row `locked` = the router at its registered thresholds (decision rule identical to `grpo_pair_sweep.py` lines 36–37), priced on the
  strong-only axis; nothing else changes. Note: on this axis the locked point will read ABOVE 69.7 % because the SS path costs 1,009 here vs 929 in the
  sweep's accounting (the weak pass is paid). Runs: `hotpotqa_champ_full_d` (`--full --theta-locked 0.80 0.05 0.05`) and `hotpotqa_champ_d` (N = 200 ×
  1,000). Prediction: champion curve ≥ F6g curve at every spend, same bar (76.8) at ≤ 100 %; locked point ≈ 75.9 @ 72–75 %.
- **18 Nov 2026 14:17 — CHAMPION WHOLE-SET RUN (lab, v1.8 sha 2eed562c7aa6…, `hotpotqa_champ_full_d`, `--theta-locked 0.80 0.05 0.05`).** The
  registered points REPRODUCE EXACTLY on the strong-only axis: per-query (greedy) **74.21 @ 54.2 %**, locked cap point **75.89 @ 69.7 %** — i.e. the
  28 Oct "evaluation accounting" (C_SS 929, weak-first path prices) IS the strong-only budget convention; the 14:20 note's expectation that the locked
  point would read above 69.7 % was wrong and is withdrawn. Under the budget: spend-all = spend-to-peak **76.70 @ 97.7 %** (largest feasible θ);
  the curve's own peak θ = 0.98 is **76.84 @ 104.2 %** — above always-SS (76.77) but 4 % over budget, inadmissible; validation θ* = 0.98 (103.9 %),
  so the cap, not θ*, sets the point; pacing 76.70 @ 100.0 (one 0.03 % overshoot); oracle 81.50 @ 23.3; escalate-all 108.6 % infeasible. Prediction
  (14:20) met: champion ≥ F6g at the budget (76.70 vs 76.77 @ 99.9 — equal within 0.1), same bar. **Training 1 B = 100 (`mab_t1_full_d`):** spend-all
  = spend-to-peak 61.33 @ 76.3 % (saturated probabilities: no gain over its registered 61.33 @ 66.6 %), pacing 62.00 @ 93.7 %, oracle 63.33 @ 59.6 %,
  escalate-all 123.0 %. Slides 150–151 switch the HotpotQA router to the champion.
- **18 Nov 2026 15:03 — ANSWER-STAGE FIXED-k ON THE HEADLINE ROUTER (lab; scorer v1.1 re-score `samplek_k40_scored_v11`: SW×k prices 894 / 906 / 921 /
  951 / 1,011, accuracies as §14; `budget_allocation.py` v1.8 `--full --samplek --k-weak k`, folders `hotpotqa_champ_full_d_k{5,10,20,40}`).** Router rows
  IDENTICAL for every k: spend-all = spend-to-peak 76.70 @ 97.7 %, per-instance 74.21 @ 54.2 %, locked 75.89 @ 69.7 % (the headline router, like the
  three-insights one, keeps no answer stage weak on these paths). Oracle with W×k at the answer stage: 82.11 @ 24.8 % (k = 5), 81.84 @ 25.7 (10),
  81.77 @ 28.2 (20), 81.77 @ 33.1 (40) — ceiling +0.6 pt over the two-model oracle 81.50, as §14. §17(2) prediction stands. Slide 153 fixed-k row filled.
  **Gap made explicit by Jonathan (14:49–14:53): k has only been measured at the ANSWER stage (WW×k, SW×k); the RETRIEVAL stage (weak retriever ×k,
  i.e. WS and WW with k retrievals) is unmeasured, and WS is the path the router buys most (53 % at θ = 0.5). The router cannot be offered k at the
  retrieval stage without those outcomes → retrieval-stage k measurement to be registered (§18, draft below) before the k-router (§17(2)).**

## 18. k WEAK RETRIEVALS — the retrieval-stage outcomes for the k-router (registered 18 Nov 2026, 15:20 SGT; DRAFT until Jonathan's line-by-line read; nothing submitted)
**Why (Jonathan 14:49–15:02):** k has only been measured at the answer stage (WW×k, SW×k), the two paths the K = 1 router never buys; the path it
buys most is WS (53 % at θ = 0.5), where k means repeating the WEAK RETRIEVER. The router can only decide among actions it has outcomes for, so
these cells are the training signal, not a hand-picked experiment. **Settings (Jonathan 15:10, one by one):** (1) Config B models; (2) k_ret ∈ {1, 5,
10} — 20/40 excluded (WS = 1,658 / 3,198 > 929 per-instance budget); k_ret = 1 = the stored greedy cell; (3) combination NOT fixed — union and
vote (≥ ⌈k/2⌉) both measured so the router may choose; (4) retriever sampling T 0.7 / top-p 0.9 (answer-stage temperature may follow tonight's
sweep); (5) answer stage measured both ways on every combined context — strong ×1 greedy (WS(k_ret)) and weak ×40 samples (WW(k_ret, k_ans));
(6) splits val 1,110 / test 1,481; (7) cluster, two H200s, one job per split; nothing to copy (runner, data, ids, clone already there); (8) prices:
per-call weak-first as the headline axis (WS(k_ret) = 77 k_ret + 118 → 503 / 888; WW = 77 k_ret + 3 k_ans) AND measured tokens → AgentTTS units
alongside (a union context is longer; the per-call price understates it).
**Code:** `hotpotqa_retrk_cells.py` v1.0 (delta on the reviewed answer-stage script; `# RETRK` marks the new parts; model guards assert the
Config B ids; resume-safe per phase) + `hotpotqa_retrk.sbatch` (SPLIT env, OUT `runs/hotpotqa/retrk_k10`). Phases: retr (7B, 10 samples/question,
RETR_OUT 768) → qa_strong (70B fp16, greedy, 4 contexts) → qa_weak (3B, 40 samples × 4 contexts). Estimate ≈ 2.5–3 h per split.
**Checks before submission:** hashes (script, runner = the answer-stage job's runner sha), 10-question smoke (`ARGS="--end 10"`): retrk rows
have 10 pick-lists each with 1–20 passage numbers inside the pool, u5 ⊇ v5, u10 ⊇ u5; qa_strong rows are short answers; samples rows 40 strings;
manifest models = Qwen2.5-7B / Llama-3.2-3B / Llama-3.1-70B.
**Predictions:** union lifts WS by +1 to +2 pt at k_ret = 5 on validation (2Wiki §15-A: +1.5), ≤ +0.5 more at 10; vote lifts less than union;
WW(k_ret, k_ans) stays within ±0.5 of the answer-stage result. Decision rule: if WS(k_ret = 5, union or vote) ≥ 65.5 on validation, the
retrieval-stage k action enters the k-router's menu as a live option; otherwise HotpotQA's statement stays "k buys nothing; the strong call is the
only purchase", with the retrieval stage now measured rather than assumed. Scorer (`hotpotqa_retrk_score.py`) and the k-router trainer follow.
- **19 Nov 2026 17:35 — §18 SMOKE READ (job 368524, 10 val questions, 23.6 min, exit 0) + PILOT AMENDMENT.** Content correct: 10 pick-lists per
  question (1–2 passages, inside the pool), v5 ⊆ u5, 40 strong answers, 40 × 40 weak samples, models Qwen2.5-7B / Llama-3.2-3B / Llama-3.1-70B.
  Speed is the problem: the retrieval prompt is 12.8–15.2k tokens (110-passage pool) and the 10-wide sampled decode runs until the longest sample
  stops (≤ 768 tokens) → **32–236 s per question (mean ≈ 116 s)**; full val ≈ 36 h, test ≈ 48 h — infeasible by 27 Nov. Answer phases: 2 min per
  40 calls each (≈ 3.7 h per phase per split). **Pilot (Jonathan 17:34: "pilot first and decide"):** validation positions 0–399 of slr-val only;
  `--retr-out 320` for the sampled retrievals (greedy cell ≈ 100 tokens; a cut-off sample still parses the [n] it emitted — recorded in the
  manifest next to the runner's 768); retr phase in two one-GPU chunks (0–200, 200–400; v1.1 writes one chunk file each), then qa_strong + qa_weak
  on two GPUs. Expected ≈ 3 h + 1.5 h. The §18 decision rule is read on this pilot (WS(k_ret = 5) ≥ 65.5 on validation → live option → test run;
  else settled on validation, no test run). Files: cells v1.1 sha 771b2f65b20b…? (see git), `hotpotqa_retrk_pilot.sbatch`, scorer v1.1 (chunk files).
  ERP I priority unchanged: the pilot is a bonus; λ = 0 retrain, paper section and decks come first.
- **19 Nov 2026 17:53 — λ = 0 RETRAIN (§17(1)) SMOKE PASSED AND RUN LAUNCHED (lab, tmux `lam0`).** Pre-launch: lab trainer that trained the headline
  router = `grpo_trainer_v15_backup.py` sha cddc2533e0bb = repo `grpo_trainer_v15.py` (byte-identical); v1.7 (sha 0f45a36a740a) = that file + the
  seven registered lines, diff read by Jonathan; copied beside the lab's helper modules; py_compile clean; GPU idle; root disk 100 % (1.4 G free —
  outputs go to /mnt/hdd; administrator to be told). Smoke (`--smoke --from-scratch --prompts objective --lam 0`): FINAL record shows the reward
  `tokenF1 - max(0, C_path/C_SS - 1) - 0.0*C_path/C_SS`, lambda 0.0 / lambda_default 0.1, prompts objective, fresh adapters, rows sha 6bbff27e…,
  c_ss 929; objective 0.679 → 0.730 over 3 smoke epochs with mean cost 468 → 677 (the router escalates more once nothing in the reward says
  "cheap" — expected). Incident: the first smoke survived an SSH drop and ran beside the tmux one for ~1 min; killed (PID 1170188); smoke folder
  deleted before the real run. **Run:** `--from-scratch --epochs 6 --prompts objective --lam 0`, no `--baseline` (registered), seed 17, G 8, out
  `grpo_ret_qa_a1_fresh_ep6_objective_lam0`. Identical to the headline run except the reward's cost term. Cosmetic: the FINAL record's inner
  `version` string still says v1.5 (header v1.7; the reward string identifies the code) — fixed in the repo copy after launch, not in the running file.
  Temperature sweep (§15): died at T = 0.3 val question 1,036 (CUDA OOM — a foreign 16 GB process on the 44 GB card); resume-safe; restarts after this run.
- **19 Nov 2026 21:48 — §17(1) λ = 0 RETRAIN: RESULT (negative for λ = 0; the router still needs the cost term).** Run `grpo_ret_qa_a1_fresh_ep6_objective_lam0`
  (trainer v1.7, `--lam 0`, 6 epochs, 124 min, exit clean). Training: from step 100 validation mean F1 0.757 / lenient 76.76 % = the always-strong level
  on validation, at mean cost 926–929 (= the strong price); p(W) at retrieval 0.001–0.06 in epochs 0–1 — the router converged to "escalate (nearly)
  everything"; later epochs drift cheaper (cost 690–900, p(W) ≤ 0.29) at the same accuracy, but the best-objective checkpoint is the all-strong one.
  Store: `grpo_lam0_test_store.py` (sha 710e3726e788…, = the registered store script with run folder and tag changed; best adapters) → test_scores
  `*_grpo_fresh_objective_lam0.jsonl`. **Allocation** (`budget_allocation.py` v1.8 `--full --budget strong-only`, `hotpotqa_lam0_full_d`): per-instance
  (greedy) **76.77 @ 108.6 % — identical to escalate-all, over budget**; spend-all = spend-to-peak **75.49 @ 97.1 %** vs the λ = 0.10 router's 76.70 @
  97.7 % (**−1.21 pt at the same spend**); curve peak 76.77 only at 108.2 %; val θ* = 0.10 at 105.4 % (the validation curve never fits the budget);
  pacing 75.89 @ 100.0. Guard max(0, C/C_SS − 1) never fired (as predicted: 0 on all K = 1 paths). **Reading (statement of record):** removing the
  per-instance cost term does not make the router "budget-ready"; it removes the only training signal on the ~52 % tied questions and the strong ≥
  weak majority, so the scores become nearly flat and the set-level threshold has little to rank. Cost as a CONSTRAINT at inference fixes the operating
  point; the cost TERM in training is what makes the router allocatable. Part 2 sub-question (a): the router still needs the cost term. The §17
  prediction ("reaches the bar at ≈ 93 %") was wrong in the optimistic direction; the registered risk ("tie-breaker") is the mechanism observed.
- **20 Nov 2026 13:43 — §15 SWEEP: SECOND RESTART (incident, no result affected).** The 19 Nov 21:59 restart ran in a plain SSH window, not in tmux
  `sweep` (the `(attached)` client was a shell inside the tmux window, mistaken for the run); it died when the SSH session closed at ≈ 22:20, with
  T = 0.3 test at 393 / 1,481 stored. Restarted inside tmux `sweep` (green status bar verified) with the same command and `SLR_QA_WEAK`; resumed
  2,220 val entries and 393 test entries, recomputed nothing; log `runs/hotpotqa/samplek_sweep.log`. Hashes identical to the 18 Nov start.
- **20 Nov 2026 15:05 — §15 READ, T = 0.3 (scorer v1.1 sha 865e75c37fd4…, out `samplek_k40_t0.3_scored`; val 2,220 / test 2,962 rows = complete).**
  **Vote@k val (test in brackets):** WW×1 51.17 (50.44) · ×5 51.17 (50.44) · ×10 51.53 (50.57) · ×20 51.53 (50.44) · ×40 **51.80** (50.71); greedy
  WW 52.25 (50.30). SW×1 62.88 (62.39) · ×5 62.70 (62.26) · ×10 62.79 (62.26) · ×20 62.70 (62.19) · ×40 **62.79** (62.12); greedy SW 63.15 (61.78).
  vs T = 0.7 validation vote@40: WW 51.80 vs 51.71 (+0.09), SW 62.79 vs 63.15 (−0.36) — inside the predicted ±2 pts; the selection rule does not
  fire. Equal-price pairs unchanged in substance: WW×40 at 197 = 50.71 vs WS at 195 = 64.01 (−13.3); SW×20 at 951 = 62.19 vs SS at 1,009 = 76.77
  (−14.6). Reading: lower temperature makes the samples agree with the greedy answer more often (the k = 1 sample is already within 1 pt of greedy),
  so voting has even less to remove; it adds no knowledge. T = 1.0 and 1.3 follow (expected ≈ 18:20); one read each, same rule.
- **24 Nov 2026 13:39 — §15 READ, T = 1.0 and T = 1.3 (sweep `SWEEP DONE` 20 Nov 18:27; scorer v1.1; outs `samplek_k40_t1.0_scored`, `samplek_k40_t1.3_scored`;
  all six cell files complete: val 2,220 / test 2,962 rows each). §15 CLOSED — the selection rule does not fire; the registered T = 0.7 cells stand.**
  **T = 1.0, vote@k val (test):** WW×1 47.48 (48.41) · ×5 49.37 (49.83) · ×10 50.27 (50.17) · ×20 50.90 (50.17) · ×40 **51.71** (50.84);
  SW×1 57.03 (60.43) · ×5 59.28 (60.90) · ×10 60.81 (60.97) · ×20 61.62 (62.26) · ×40 **62.07** (62.59).
  **T = 1.3, vote@k val (test):** WW×1 44.14 (44.63) · ×5 48.20 (47.81) · ×10 49.19 (49.49) · ×20 50.90 (50.91) · ×40 **50.54** (50.71);
  SW×1 52.34 (56.18) · ×5 58.74 (58.34) · ×10 60.09 (60.09) · ×20 61.53 (62.39) · ×40 **61.44** (62.73).
  **Validation vote@40 vs T = 0.7 (51.71 / 63.15):** T = 0.3 +0.09 / −0.36; T = 1.0 +0.00 / −1.08; T = 1.3 −1.17 / −1.71 — all inside the predicted ±2 pts;
  no temperature beats 0.7 by > 2 pts, so no new test read is used for allocation. **Reading (statement of record):** temperature moves the single-sample
  accuracy a lot (WW×1 val 51.17 → 47.48 → 44.14 for T = 0.3 → 1.0 → 1.3) and the vote at k = 40 recovers exactly that loss and no more: every
  temperature converges to the same ceiling, the weak model's own greedy answer (WW 52.25 / SW 63.15 on val). Voting repairs what temperature
  breaks; it does not add knowledge. Equal-price pairs unchanged in substance at every T (test): WW×40 at 197 = 50.4–50.8 vs WS at 195 = 64.01;
  SW×20 at 951 = 62.0–62.4 vs SS at 1,009 = 76.77. Answer-stage K > 1 on HotpotQA is closed negative at all four temperatures; the only open
  K > 1 question is retrieval-stage k (§18 pilot, cluster job 371659).

## 19. TEMPERATURE SWEEP FOR THE RETRIEVAL-STAGE k PILOT (WS path) — registered 24 Nov 2026, 14:00 SGT, BEFORE any result is read
(Jonathan 13:58 "I think we should run temperature sweep for WS and K > 1"; 13:45 question "why we didn't test with WS" → the §15 sweep varied k on the
weak ANSWERER, which WS does not have; k on WS can only enter at the weak RETRIEVER, which is the §18 pilot, run at T = 0.7 only.)
**Design.** Same as the §18 pilot in every respect except the sampling temperature of the weak retriever: the first 400 validation questions
(positions 0–399 of slr-val), k_ret 10 (levels 5 / 10), union and vote combination, seed 20261118, fp16, `--retr-out 320`, cells script
`hotpotqa_retrk_cells.py` v1.1 (sha f48a6a9e56c6…, unchanged); temperatures {0.3, 1.0, 1.3} around the pilot's 0.7 (top-p 0.9 throughout).
The strong answerer is greedy, so T changes only which passages are picked. qa_weak is NOT run in the sweep (WW with k on retrieval is not the
question). New sbatch `hotpotqa_retrk_tsweep.sbatch` (sha 2b4258d09fae…, commit bc96c79): differs from the pilot's only in TEMP from the
environment → `--temp $TEMP`, OUT `retrk_tsweep_val400_t<T>` (one folder per T, never the pilot's), and a `qa_strong`-only phase. Three checks
done before submission: hashes (sbatch 2b4258d09fae, cells f48a6a9e56c6), `bash -n` OK, no pre-existing sweep folders; the registered pilot
(same code path at T = 0.7, 400 questions, exit 0) stands as the smoke; the `--temp` plumbing is verified from the first running job's banner.
**Quantity read.** WS(k_ret = 5 / 10, union / vote) on the 400 validation questions per temperature, scored by `hotpotqa_retrk_score.py` v1.1
against WS(k = 1) = 65.05 (registered cell on the same positions as the scorer reports it).
**Predictions.** A hotter retriever yields more distinct picks (a wider union, longer context) but lower per-sample pick quality: union WS moves by
at most ±2 pts across temperatures; vote WS falls with T (as the answer-stage vote did in §15); no temperature beats the §18 decision threshold
(WS(k = 5) ≥ 65.5 on validation) if T = 0.7 does not — temperature is a second-order knob on this mechanism. **Decision rule** unchanged from §18:
any temperature whose WS(k = 5) on validation reaches 65.5 is read once on test; otherwise the sweep closes the retrieval-stage question at all T.
**Timing.** ERP I is due 27 Nov; if the sweep has not finished by then it is reported as running / future work, and the paper's K > 1 statement
rests on §13–§15 and the §18 pilot at T = 0.7.
- **24 Nov 2026 14:00 — SUBMITTED** (cluster, researchshort, 1 × H200 each): T = 0.3 chunks 0–200 / 200–400 = jobs **379571 / 379572**;
  T = 1.0 = **379573 / 379574**; T = 1.3 = **379575 / 379576**. All PD (Priority), no start estimate yet. The `qa_strong` job (2 × H200) per
  temperature follows after both of its chunks finish.
- **24 Nov 2026 14:04 — §18 PILOT READ (POSITIVE for union; negative for vote).** Cells: cluster jobs 370037 / 370038 (retr, 71 min each) and
  371659 (qa, 51 min), folder `retrk_pilot_val400` (tgz sha d1965e5f…, counts 200 / 200 / 1,600 × 4 / 200 × 2); scorer v1.1 (ce59da7077de…),
  out `retrk_pilot_val400_scored`. Validation positions 0–399; k_ret = 1 reference on the same questions: WW 58.5 @ 80, WS 70.0 @ 195,
  SW 66.5 @ 894, SS 78.75 @ 1,009. **WS with k on the retriever, one greedy strong answer:** u5 (union of 5 sampled retrievals) **72.50**
  (+2.50) @ 503 per-call; u10 **73.75** (+3.75) @ 888; v5 (vote ≥ ⌈k/2⌉) 66.50 (−3.50); v10 67.50 (−2.50). **WW(k_ret, k_ans) on the union
  context:** u5 55.75 / 57.75 / 57.75 / 57.25 / 57.75 (k_ans 1 / 5 / 10 / 20 / 40) vs greedy WW 58.5 at k_ret = 1; u10 56.00 … 58.25 — the weak
  answerer does not profit from the wider context. Measured units (AgentTTS-normalised FLOPs) u5-S 152.1, u10-S 259.5 (k_ret = 1 reference
  measured units not yet reported — re-score with `--meta-retr1/--meta-qa1`). **Reading (provisional):** union adds evidence the strong
  answerer uses; vote discards it. The §18 decision rule fires (WS(k = 5) ≥ k = 1 + 0.45 on validation). Before any claim: (1) measured cost
  of the union context vs WS/SS, (2) paired significance vs WS(k = 1) on the 400 (10 questions = +2.5 pts), (3) one held-out test read of the
  union action — the next registered step. Nothing on test has been read.
- **24 Nov 2026 14:10 — §18 MEASURED COST (scorer v1.3, sha 7597fbebbb75…; v1.2 implemented the documented `--meta-retr1/--meta-qa1`, v1.3
  added `--meta-retr1s` for SS and charges the retrieval PROMPT once in full for a k_ret < batch context — v1.1/v1.2 pro-rated the prompt, which
  made u5 look cheaper than one WS call (×0.71), an artefact). Measured units = AgentTTS-normalised FLOPs from the Config B meta sidecars
  (`stair_common.cost.units`; retr 7 B / 72 B, QA 3 B / 70 B; the 72 B strong-retriever count is Jonathan's to confirm). Same 400 validation
  questions: **WS(k = 1) 214.0; u5 237.7 (×1.11); u10 259.5 (×1.21); v5 219.3; v10 231.5; SS 1,825.8 (×8.53 WS)**. Reading: in measured compute
  the union action is cheap — u10 buys +3.75 pts for 21 % more than one WS call, at 14 % of SS — because a k-sample retrieval batch processes
  the 12k-token pool prompt once (Eq. 8) and only the generated picks and the longer union context are extra. The per-call price axis (u10 = 888
  vs SS 1,009, 88 %) counts each sample as a full call and is the conservative bound; both axes are reported. SS measures ×8.5 of WS (vs ×5.2
  per-call): the 72 B retriever reading the pool dominates. Accuracies unchanged from the 14:04 read. Still open before any claim: paired
  significance vs WS(k = 1) on the 400; one held-out test read.
- **24 Nov 2026 14:12 — §18 PAIRED SIGNIFICANCE on the 400 (`analysis/retrk_paired_test.py`: exact two-sided McNemar on discordant pairs;
  10,000-resample paired bootstrap of the mean difference, seed 20261124; inputs `retrk_outcomes.jsonl` + `router_data_standard.jsonl`).**
  vs WS(k = 1) = 70.0: **u10 +3.75 (wins 20 / losses 5, p = 0.004, CI [+1.5, +6.25])**; u5 +2.50 (16 / 6, p = 0.053, CI [+0.25, +4.75]);
  v5 −3.50 (4 / 18, p = 0.004); v10 −2.50 (4 / 14, p = 0.031). u10 vs u5 +1.25 (8 / 3, p = 0.23, n.s.). u10 vs SS −5.00 (9 / 29, p = 0.002).
  Mechanism: of the 43 questions SS answers and WS(k = 1) misses, u10 recovers 18 (u5: 13) while breaking 5 of WS's 280 correct answers.
  **Reading:** the union of 10 sampled retrievals is a real gain on validation (p < 0.01); the 5-sample union is borderline at n = 400; vote is
  significantly harmful (it discards evidence); u10 remains significantly below SS. Gate 2 cleared for u10. Remaining gate: one held-out test
  read of the union action (u10 primary, u5 secondary) — registered next step; nothing on test has been read.
- **24 Nov 2026 15:34 — §18 HELD-OUT TEST RUN, retrieval phase STARTED on the LAB GPU (tmux `retrk`; log `runs/hotpotqa/retrk_test1481.log`;
  out `retrk_test1481`).** Jonathan 15:11–15:13: the union result contradicts AgentTTS's selected strategy (Strong retrieval ×1 → Weak answerer ×40 +
  fuser, chosen on their 50 training instances), so the claim is validated like for like: (a) the union action on all 1,481 held-out test questions,
  (b) AgentTTS's design on our pipeline and the same questions (§20, to be registered when the fuser model/prompt are confirmed from the
  reference code — never assumed). Script v1.2 (sha 14f9018c1850…, commit 284003a): `--runner lab` imports the lab runner (run_labels4.py,
  same API); the only change is the import switch; sampling, prompts, combination and storage are untouched. Env guards: SLR_RETR_WEAK =
  Qwen/Qwen2.5-7B-Instruct, SLR_QA_WEAK = Llama-3.2-3B-Instruct, SLR_QA_STRONG = Llama-3.1-70B-Instruct (asserted at start; only the retriever
  loads in this phase). Settings identical to the pilot: k_ret 10 (levels 5 / 10), T 0.7 / top-p 0.9, seed 20261118, fp16, `--retr-out 320`,
  SLR_POOL_K 110, no shuffle; `--split test --start 0 --end 1481` (one chunk file `retrk_test_0-1481.jsonl`; resume-safe). Smoke (10 q.,
  `retrk_smoke_lab`): 10 rows, 15.9–16.7 s/question → full run ≈ 6.8 h, expected done ≈ 22:30. The qa_strong phase (70B) runs on the cluster
  afterwards (the pilot sbatch with SPLIT=test; qa_weak not run — WW with k on retrieval is not the question). **Predictions (test, 1,481):**
  u10 − WS(k = 1) in [+1.5, +6.25] (the validation CI), point estimate ≈ +3; u5 ≈ +2; vote variants negative; u10 remains below SS (76.77);
  measured cost ×1.1–1.3 of WS. Decision rule: u10 beats WS(k = 1) on test with McNemar p < 0.05 → ③ stands as "retrieval-stage K > 1 pays";
  otherwise ③ is reduced to the answer-stage negative and the validation result is reported as not replicated.

## 20. AGENTTTS'S SELECTED HOTPOTQA STRATEGY ON OUR PIPELINE — registered 24 Nov 2026, 15:45 SGT, BEFORE any result is read
(Jonathan 15:06: "with n = 50, AgentTTS selected the best combination (Strong for Ret and K = 40 Weak for Answer). This experiment is showing
contradictory to AgentTTS findings. So, we have to validate this claim.")
**Their design, read from their own code and record on the cluster (`~/slr/agenttts/code_others/archon_hotpotqa.py` 440–480, 655–675;
`archon/completions/components/Fuser.py`; `.../components/prompts.py` 24–65), never assumed:** retrieval Qwen2.5-72B-Instruct-AWQ ×1 (T 0.9) →
generation Llama-3.2-3B-Instruct ×40 (T 0.7, 512 tokens) → **fuser = the same 3B** (T 0.3, 512 tokens, 1 sample); "best config … 0.742". Fuser
messages: system "You are a helpful assistant who fuses multiple answers"; user = `make_fuser_prompt` (no critiques) whose "latest user query" is
`conversation[-1]["content"]` = "Query: " + question + "References: " + "\n[i] passage" over the WHOLE candidate pool (565–574), then "Once again,
the query is: …", "Responses from models:" and the numbered 40 candidates. top_p never set in their utils (vLLM default 1.0). Their metric: lenient
contains-gold (author-confirmed). Their strategy was selected on 50 training instances (SE ≈ 7 pts at n = 50).
**Our reproduction of it:** the generation layer = our stored SW×40 samples (Strong retrieval ×1 → 3B ×40 at T 0.7; `samplek_k40`, cluster) — one
recorded difference: our samples used top-p 0.9, theirs 1.0. New: `hotpotqa_fuser_cells.py` v1.0 (the fusion call only; prompt byte-faithful
including their f-string line-continuation whitespace; 3B, T 0.3 / top-p 1.0 / 512 tokens, seed 20261124 + i), `hotpotqa_fuser.sbatch` (any GPU),
`hotpotqa_fuser_score.py` v1.0 (lenient; same questions; vote@40 on the same samples; WW/WS/SW/SS k = 1; exact McNemar + bootstrap; mean fused
length and measured units reported). Per-call price SW×40 + fuser = 891 + 120 + 3 = 1,014 (> SS 1,009).
**Quantity read:** accuracy of the fused answer on the 1,481 held-out test questions (validation 1,110 as a second read if time allows).
**Predictions:** fused accuracy between the vote@40 (62.1) and WS (64.0), most likely 60–66; below SS (76.77) by > 10 pts; fuser vs vote within
±3 pts (the fuser reads the same 40 candidates; it can only re-select or paraphrase them). **Decision rule:** if their design on our pipeline beats
our union action (u10 on test) or always-Strong, learning ③ is withdrawn and the slide says so; if it lands as predicted, ③ stands and the paper
states the contradiction with its explanation (selection on n = 50; a weak Strong baseline in their pipeline; retrieval F1 vs final-answer
yardstick). **Three checks before submission:** hashes (script + sbatch after pull), `bash -n`, 10-question smoke (`ARGS="--end 10"`, separate OUT);
the first full-run log line must show `ctx S, fuser over first 40 samples, T 0.3 top_p 1.0`.
- **24 Nov 2026 15:47 — §19 SWEEP CANCELLED (jobs 379571–379576, never started; estimates 25 Nov 02:44 / 08:44 / 14:44 on `honey`).** Reason: the
  §18 test retrieval moved to the lab GPU (15:34) and the sweep would hold the only H200 node exactly when the §18 test qa_strong job and the §20
  fuser read need it before the 27 Nov deadline. The sweep is deferred, not dropped: resubmit with the same command after the §18/§20 reads.
  §20 smoke submitted: job 379772 (`ARGS="--end 10"`, any GPU, OUT `fuser_smoke`).
- **24 Nov 2026 15:50 — §20 SMOKE 379772 FAILED on a Tesla V100-SXM3-32GB (`torch.AcceleratorError: no kernel image is available for execution on the
  device`): slr-env's torch build has no sm_70 kernels. Not a script fault — the banner and model load printed (`test: 10 questions, ctx S, fuser
  over first 40 samples, T 0.3 top_p 1.0 max_new 512`; `[fuser] unsloth/Llama-3.2-3B-Instruct loaded`). Resubmitted pinned to an Ampere card:
  job 379776 (`--gres=gpu:a5000:1`). Rule from now on for any-GPU jobs: request a named Ampere-or-newer type (a5000 / 3090 / a40 / l40s / a100 /
  h100 / h200), never V100 / P100 / RTX 8000.** Model id recorded: the cluster runner's QA_WEAK is `unsloth/Llama-3.2-3B-Instruct` (a re-upload of
  Meta's Llama-3.2-3B-Instruct weights) — the same runner and id that produced the stored SW×40 samples, so fuser = generator, as in their design.
- **24 Nov 2026 16:05 — §20 FULL RUN STARTED on the LAB GPU (tmux `fuser`; out `runs/hotpotqa/fuser_k40_lab`; log `fuser_k40_lab.log`), sharing the
  card with the §18 retrieval run.** Reason: every Ampere-or-newer cluster card was occupied (smoke 379776 estimated 23:46; left queued as backup,
  to be cancelled). Lab smoke (`fuser_smoke_lab`, 10 q., 2.0 min): banner `ctx S, fuser over first 40 samples, T 0.3 top_p 1.0 max_new 512`,
  `[fuser] meta-llama/Llama-3.2-3B-Instruct loaded`, 10 fused paragraphs (28–191 words; they tend to continue the last candidate rather than restate
  — their design's behaviour, recorded). Inputs: the lab copy of `samplek_k40/samples_test.jsonl` (2,962 rows; sha printed in manifest). Model id on
  the lab: `meta-llama/Llama-3.2-3B-Instruct` (the cluster id is the `unsloth/` re-upload of the same weights); script sha 03ccbede0068, seed
  20261124 + i. Expected ≈ 12 s/question shared → ≈ 5 h, done ≈ 21:30. Read: `hotpotqa_fuser_score.py` v1.0 on the lab (CPU).
- **25 Nov 2026 11:21 — LAB UNREACHABLE since 24 Nov ≈ 19:30 (ssh/ping time out; traceroute dies after the campus hop 10.172.1.50); the two lab
  runs (§18 test retrieval in tmux `retrk`, §20 fuser in tmux `fuser`) cannot be checked. Cluster copies submitted so neither read depends on the
  lab:** §20 full fuser run job **381472** (`SPLIT=test`, A5000, OUT `runs/hotpotqa/fuser_k40`; the cluster smoke 379776 had completed clean,
  1:51, 10 rows); §18 test retrieval **jobs 381477–381484** (8 chunks of 200 on 1 × H200, `hotpotqa_retrk_test.sbatch` sha 780b01023208 —
  the pilot sbatch with `--split test`, OUT `retrk_test1481`, qa phase restricted to `qa_strong`). Whichever copy (lab or cluster) finishes
  first feeds the read; the other is cancelled, and the registration records which was used.
