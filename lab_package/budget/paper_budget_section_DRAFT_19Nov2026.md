# §6.x  Allocating a fixed budget over a set of instances  — DRAFT 19 Nov 2026 (for RightRoute_final_report; ERP I)

*Owner: Jonathan. Notation as in §3.1 / Eq. (2). Numbers: whole held-out sets unless stated; ± from 1,000 bootstrap bundles. Blanks `[__]` are filled from runs in progress (λ = 0 retrain, retrieval-stage k pilot).*

---

## 6.x Allocating a fixed budget over a set of instances

Eq. (2) optimises cost per instance: the term λ·Cᵢ/C_SS,ᵢ pushes every decision toward the cheaper path, and the router's operating point is whatever that pressure produces. A deployer, however, usually has a *cumulative* constraint — a token or cost budget for an expected set of instances — and asks a different question: *given that budget, what is the highest accuracy the router can reach, and must the whole budget be spent to reach it?* This section answers that question for the two datasets of §5 with the routers already trained; the only new mechanism is how the budget is enforced.

**Budget.** We take the budget to be the cost of the Strong model alone on the whole held-out set, B = Σᵢ C_S,ᵢ — what a deployer who never uses the Weak model would pay (929 units per HotpotQA question; the Strong agent's cost per MedAgentBench task). Because RightRoute runs the Weak model first and escalates (§3.1), the Weak pass is paid out of the same budget: escalating *every* stage costs more than B (108.6 % on HotpotQA, 106.9 % on MedAgentBench) and is not an admissible policy. This is the strict reading of "always-Strong cost"; the per-instance normalisation of Eq. (2) is unchanged.

**From per-instance to set-level decisions.** The router's score at a stage is a probability p_W that the Weak result suffices. Per-instance routing (the operating points of §6.1–6.3) thresholds p_W at a fixed value, θ = 0.5 or the locked thresholds of the validation sweep. Under a set budget we instead choose **one threshold θ for the whole set**: the stages are ranked by p_W, and θ is the largest value whose total spend fits B (*spend-all*), or the cheapest θ whose validation accuracy is within one instance of the validation peak (*spend-to-peak*). By Lagrangian duality a set constraint Σᵢ Cᵢ ≤ B is equivalent to a per-instance price λ* on cost, so the threshold family is the inference-time dual of the λ term in Eq. (2): the budget decides the price, the router's ranking decides who pays it. Thresholds are set on validation only; every number below is a re-read of the stored held-out outputs of §6 — no new model call.

**Results (Figure X, Table Y).** On HotpotQA, with the label-free router of §6.1 (GRPO (Reward), fresh adapter), spend-all reaches **76.70 % at 97.7 %** of B and spend-to-peak the same point — the always-Strong bar (76.77 %) and no more; the threshold family passes the bar (76.84 %) only at 104 % of B, i.e. outside the budget. The two registered per-instance points stay where §6.1 reported them (74.21 @ 54.2 %; 75.89 @ 69.7 % at the locked thresholds), and the two-model oracle (81.50 @ 23.3 %) shows the headroom a perfect router would have. On MedAgentBench, with the round-2 router of §6.3, spend-all reaches **62.67 % at 99.7 %** of B — **+8.7 points above always-Strong (54.00 %)** inside its own budget — while the router's registered per-instance point (63.33 %) sits 0.2 % *over* the strict budget and is admissible only through the set-level threshold. The cost-optimised router of §6.2 (Training 1, cost in the reward, W/S only) reaches 61.33 % at 66.6 % of B: within 2 points of the two-model ceiling (63.33 %, the best any W/S policy can do, since 36.7 % of tasks fail under both models) for two-thirds of the budget; the remaining third of B is worth +1.3 points.

**Must the whole budget be spent?** No — and it cannot be spent on escalation alone. On HotpotQA the last 7 % of B buys 0.1 point; on MedAgentBench the router's accuracy is capped by the two-model ceiling well before the budget is exhausted. What the unspent remainder buys differs by dataset: on HotpotQA it is saving; on MedAgentBench it is **human escalation**. Converting the remainder into escalations of the least-confident high-risk tasks (τᵢ ≥ 0.8; 30 of the 150 held-out tasks) to a person — 5 / 10 / 15 of the 30 — gives 63.3 / 66.7 / 70.0 % at 88 / 78 / 69 % of B and reduces unseen high-risk failures from 21 to 16 / 11 / 6 (the person is counted correct, Qᵢ = 1 as in Eq. 2; human time is not priced in B). Each such escalation is worth about +0.7 point and −2 % of spend, because it both fixes a task that both models fail and removes a Strong-agent call. These rows are the only policies above the two-model ceiling, and they are the budget-level form of the risk term in Eq. (2).

**Cost as a constraint rather than an objective.** If the budget is the constraint, the cost term in Eq. (2) is no longer needed as an optimisation target. We retrain the HotpotQA router with λ = 0 — same recipe, prompt, seed and data; only the cost term removed, the per-instance guard max(0, Cᵢ/C_SS,ᵢ − 1) kept (it is zero on all four K = 1 paths) — and allocate it under the same set budget. [__ Result: spend-all __ @ __ %, spend-to-peak __ @ __ %; compared with the λ = 0.10 router's 76.70 @ 97.7 %. Reading: the λ = 0 router reaches the bar at __ % of B — the cost term was / was not doing useful work as a tie-breaker on the ~52 % of questions where all four paths tie on F1. __]

**More than one Weak call (K > 1).** Within the same budget a router may also buy *repeated* Weak calls where it keeps a stage Weak — k samples with a majority vote — instead of one Strong call; the Strong model is never repeated. At the answer stage this never pays on HotpotQA: majority voting over k ∈ {5, 10, 20, 40} samples of the 3B answerer converges to its own greedy answer (±0.5 point), and at equal price one Strong answer wins by 13.6 points (WW×40 at 197 units = 50.4 % vs WS at 195 = 64.0 %) and 14.8 points (SW×20 at 951 = 62.0 % vs SS at 1,009 = 76.8 %); the allocation under B is identical for every k because the router keeps no answer stage Weak above ~60 % of B, and an oracle allowed to choose k gains only +0.6 point (82.11 vs 81.50). The path the router does buy is WS, where k means repeating the Weak *retriever*; those outcomes are measured on a validation pilot (k ∈ {5, 10}, picks combined by union or vote) [__ WS(k=5) = __ / __ (union / vote) vs 64.0 at k = 1 __], which decides whether a retrieval-stage k action enters the router's menu. This is the opposite of 2WikiMultihopQA, where AgentTTS's 3B × 50 + fuser beats the Strong model (70.4 vs 65.8 % of their 500): whether repeated Weak calls pay is a property of the dataset, which is why the router must be able to learn it rather than assume it.

**Summary.** A fixed set budget changes the question from "what does each instance cost?" to "what does the set buy?". With one threshold per set, RightRoute reaches the always-Strong accuracy on HotpotQA and exceeds it by 8.7 points on MedAgentBench, in both cases inside a budget that escalate-all would overrun; the highest accuracy does not require the whole budget, and above the two-model ceiling only a person moves the number.

---

### Figure X (caption)
Accuracy within a fixed budget equal to the Strong model alone (B = 100 %); 1,000 bootstrap bundles per dataset (N = 200 HotpotQA questions, N = 100 MedAgentBench tasks); the curve is the router's threshold family, points are policies at B. Left, HotpotQA (label-free router of §6.1): the budget buys the always-Strong bar and no more; the curve passes it only over budget. Right, MedAgentBench (round-2 router of §6.3): the router beats the bar inside the budget; escalate-all is over budget; human escalation of the least-confident high-risk tasks is the only policy above the two-model ceiling. *(= slide 150 of V01-085, `fig150.png`.)*

### Table Y — cost as optimisation target vs cost as constraint (whole held-out sets; spend in % of the Strong model alone)

| MedAgentBench (150 tasks) | Acc. % | Spend % | Cost treated as |
|---|---|---|---|
| Always-Strong | 54.00 | 100 | — |
| Training 1 — GRPO, cost in the reward, W/S, θ = 0.5 (§6.2) | 61.33 | 66.6 | optimised |
| Training 1 — same router, spend-all under the budget | 61.33 | 76.3 | constraint |
| Round-2 router (W/S/person), spend-all under the budget | **62.67** | **99.7** | constraint |
| Round-2 router, registered per-instance point θ = 0.5 | 63.33 | 100.2 (over) | — |
| Two-model oracle (W/S ceiling) | 63.33 | 54.0 | — |
| Escalate every task (Weak pass + Strong) | 54.00 | 106.9 (over) | — |

| HotpotQA (1,481 questions) | Acc. % | Spend % | Cost treated as |
|---|---|---|---|
| Always-Strong | 76.77 | 100 | — |
| Label-free router — registered per-instance point θ = 0.5 | 74.21 | 54.2 | optimised |
| Label-free router — locked thresholds (headline, §6.1) | 75.89 | 69.7 | optimised |
| Label-free router — spend-all = spend-to-peak under the budget | **76.70** | **97.7** | constraint |
| Label-free router — curve peak (θ = 0.98), over budget | 76.84 | 104.2 (over) | — |
| Same recipe, λ = 0 — spend-all under the budget | [__] | [__] | constraint |
| Two-model oracle (W/S ceiling) | 81.50 | 23.3 | — |
| Escalate every stage (Weak pass + Strong) | 76.77 | 108.6 (over) | — |

### Reproducibility note (for the appendix)
Budget allocation: `lab_package/budget/budget_allocation.py` v1.8 (`--budget strong-only`, `--full`, `--theta-locked`); registration `Budget_Allocation_Registration_DRAFT_17Nov2026.md` §1–§18 (design, amendments A–D, runs, K > 1 measurement, λ = 0 retrain, retrieval-stage pilot). Routers: HotpotQA `grpo_ret_qa_a1_fresh_ep6_objective` (label-free champion), λ = 0 variant `..._lam0` (trainer v1.7 = v1.5 + `--lam`); MedAgentBench `training3_router` (round 2) and `training1_router`. All allocation rows are re-reads of stored declared held-out outputs.

### Author contributions (budget section)
J. Ahn: design of the set-level allocation (budget definition, threshold family, spend-all / spend-to-peak, human-escalation conversion), the cost-as-constraint reformulation (λ = 0 retrain, over-budget guard), the K > 1 routing-action design and its measurement, all runs and registrations; [teammates: none on this section — the routers reused are those of §6, whose attribution stands].
