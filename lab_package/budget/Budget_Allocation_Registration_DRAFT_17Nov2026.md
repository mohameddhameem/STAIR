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
