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
