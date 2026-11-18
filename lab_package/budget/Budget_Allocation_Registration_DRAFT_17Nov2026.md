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
