# MedAgentBench Routing Campaign — Results Write-up

27 Oct 2026 · Jonathan (Dohyung Ahn)

A 1.5B-parameter router trained with GRPO beats the strong-model-on-everything baseline on both axes of the held-out test — +7.3 accuracy points at 46% cost savings — and a round-2 early-escalation variant reaches the test's oracle accuracy; a learned risk penalty reproduced a hard risk gate at higher cost, grounding the campaign's central claim: risk belongs at the decision layer, not in the gradient.

## Research question and setup

Can a small router learn, from outcomes alone, when a cheap model's work is good enough to deliver — and does bolting risk awareness onto that router beat building risk into its training? The testbed is MedAgentBench: 300 electronic-health-record tasks (10 task types x 30 instances) executed against a simulated FHIR patient-record server, graded by the benchmark's own reference-solution checkers.

The architecture is a weak-first cascade. The weak model (Qwen2.5-7B-Instruct) attempts every task. A 1.5B router (Qwen2.5-1.5B-Instruct with a LoRA adapter, trained from scratch by GRPO with no labels) reads the task text plus a view of the weak attempt and makes one decision: deliver the weak result, or escalate — re-run the task with the strong model (Qwen2.5-72B-Instruct-AWQ). The router never sees the grader's verdict; its only training signal is a reward combining correctness and cost.

## Method

**Patient-disjoint split.** The 300 instances divide into calibration (150) and held-out test (150), 15 per task type per side, with no patient appearing on both sides (the original instance-random split shared 68 of ~98 patients and was replaced; overlap verified 0). Calibration further divides into train (90) and validation (60). Every tuning decision — threshold sweeps, operating points, risk-rule parameters — was made on validation and declared before any test contact.

**Cost model.** Cost is measured in normalized FLOPs units: cost.units(parameters, input tokens, output tokens), tokenized with each served model's own tokenizer over every round's cumulative input. The reward's denominator is the per-instance measured always-strong cost, which eliminated all 300 false penalty activations that a global constant produced.

**Reward.** R' = correctness (the benchmark's own pass/fail grade) - 0.10 x (cost / always-strong cost on that instance) - a hinge penalty if the chosen path costs more than always-strong - lambda_t x max(0, tau - c), where tau is a pre-computed per-instance risk score and c is the router's own deliver-confidence. The risk term is zero in Training 1 and active only in Training 2.

**Replay training.** Both paths were executed once per instance and stored; training replays recorded outcomes, so every GRPO run is execution-free — a full training takes ~6 minutes on one GPU and consumes no new episodes. Repeatability was measured first (weak 1.7% / strong 2.7% instance churn between runs; the router itself is deterministic at temperature 0), giving +/-0.33 / +/-0.67-point noise floors on all accuracy numbers.

**Risk scores.** tau = (15 - (Q1+Q2+Q3))/12 over a three-question rubric (Impact, Detectability, Reversibility, each 1-5), floored at 0.10, capped at 0.95, with Q1 = 1 (a medication prescription or recommendation) forcing tau >= 0.8. All 300 instances were scored by the locally served 72B and frozen before any test read; the scoring prompt needed exactly one registered revision (two scope clarifications: reversibility means the clinical consequence, not the database record), and the result was hand-validated at tier level — the high-risk tier is the two medication-order task types.

**Pre-registration.** Every experiment was registered before it ran: design, predictions, kill criteria, and evaluation scope. Each trained router was read on the held-out test exactly once.

## Baselines and the band

Always-weak scores 11.67 and always-strong 56.67 on all 300 instances (run 2: 56.00; our AWQ serving — an independent serving via DeepInfra scored 58.00, placing the paper's published 51.33, from a provider-retired environment, as the outlier). The per-instance oracle — the best choice on every task — reaches 65.00 at 52.6% of always-strong cost, so routing has nine points of accuracy headroom at roughly half the bill. On the held-out test 150, the same quantities are: always-weak 12.00, always-strong 54.00, oracle 63.33 — every result below is measured on those same 150 instances.

Two structural facts make routing valuable here. First, the task-1 reversal: on patient-name lookups the weak model scores 29/30 while the strong model scores 2/30 — the cheap model is better and ten times cheaper, so always-strong is not even the accuracy ceiling. Second, the hopeless set: on 105 of 300 instances (35%) both models fail; escalation there buys nothing at up to 5.5x the cost (task 9: 726 vs 4,009 units at equal zero accuracy). A capable router must find both: deliver what the weak model wins, and decline to pay for escalations that cannot fix the answer.

## Training 1 — the risk-free router

Training 1 beats always-strong on both axes of the held-out test: 61.33% accuracy at 54.1% of always-strong cost (+7.33 accuracy points, -46% cost), within 2.0 points of the oracle. Validation predicted 60.0 at 54.8%; the test delivered 61.33 at 54.1% — the declared operating point generalized. The learned partition matches the structural analysis exactly: deliver task 1 (the reversal) and the hopeless types, escalate the strong-only types. The hopeless-set behavior was predicted as emergent before training — when both paths fail, the correctness terms cancel in the group advantage and only the cost gradient remains — and appeared unprompted, with no stated principle in the router's prompt.

**The risk guardrail readout.** The same trained router was tallied twice on its one test pass: as-is, and with a hard gate forcing escalation wherever tau >= 0.8 (the two medication-order task types, 30 test instances). At the primary operating point the gate changed accuracy by exactly zero and raised cost from 54.1% to 81.0% of always-strong — its only overrides were task-9 instances, where the strong model also scores zero. At the declared cost-lean operating point, however, the gate protected: the looser threshold had begun delivering medication orders, and the gate restored them (+6.7 points of high-risk-tier accuracy). The measured characterization: a hard risk gate is free where the router already escalates, pure cost where both models fail, and protective exactly when cost pressure starts trading away high-risk tasks.

## Training 2 — risk in the reward

Training 2 activated the gradient-level risk term -lambda_t x max(0, tau - c) on the deliver branch, dormant below tau = 0.8, with everything else identical to Training 1. All three declared penalty weights (lambda_t in {0.5, 1.0, 2.0}) converged to the identical policy: deliver task 1 only, escalate everything else. Its high-risk-tier outcome is numerically identical to the hard gate's (0 of 12 validation instances delivered, same accuracy, same cost) — the learned penalty reproduced the override rule — but the pressure spilled over through the shared adapter weights: the router never learned to deliver the hopeless types cheaply, and its reward fell to 0.2927 against Training 1's 0.3667.

The registered predictions resolved cleanly. The hoped-for differential (keep delivering task 9, where escalation buys nothing) did not occur: while the router's confidence was still mid-range early in training, the penalty swept the whole high tier to escalation, and the low-confidence/high-penalty loop locked it there at every lambda_t. The registered gaming risk (inflating confidence to dodge the penalty) was not observed. On the held-out test (a declared completeness read, policy frozen beforehand): 63.33% at 94.5% of always-strong cost — the prediction, registered before the read, said "exactly 63.33 at ~94-95%."

## Training 3 — the round-2 router

A headroom analysis of stored transcripts showed that 161 of 300 weak episodes burn all 8 rounds before failing, and that cutting escalated episodes at round 2 would recover up to 5.7 cost points with decisions held fixed — the measured price of letting every weak attempt finish. Training 3 tested whether a router could decide at round 2: its view is the weak attempt cut at the second round (a finished answer when the episode ended early; otherwise only "still working" plus the round-2 action), and escalation pays only the round-2 cost plus the strong run. With restart semantics, this remains fully replay-trained.

The router did not reproduce Training 1's decisions at lower cost — cheap escalation changed the optimal policy. Hopeless-delivery's cost advantage shrank from ~3,300 units to the round-2 residue, escalating the borderline lab-order type became reward-positive, and the optimum moved to an operationally legible rule: deliver only what the weak model has already finished and reliably finishes correctly; escalate anything still working at round 2. On the held-out test: 63.33% — equal to the test oracle — at 81.5% of always-strong cost, beating its own (much higher) bar of always-escalate-at-round-2. It dominates Training 2 at the same accuracy with 13.0 points less cost, and satisfies the risk guardrail for free, since every high-risk instance escalates.

## The held-out test frontier

Every policy below is measured on the same 150 held-out instances; each trained router was read exactly once. Cost is total compute as a percentage of always-strong's; R' is the trained objective (higher is better; Training 3's escalation is priced at its round-2 cut).

| Policy | Accuracy % | Cost (% of always-strong) | Mean reward R' |
| --- | --- | --- | --- |
| Always-weak | 12.00 | 18.7 | 0.0949 |
| Always-strong | 54.00 | 100 | 0.1640 |
| Oracle (upper bound) | 63.33 | — | — |
| Training 1 — risk-free router (primary operating point) | 61.33 | 54.1 | 0.3766 |
| Training 1 + risk guardrail (tau >= 0.8 -> always strong) | 61.33 | 81.0 | 0.3488 |
| Training 1 — cost-lean operating point | 60.00 | 51.2 | 0.3876 |
| Training 2 — risk in the reward | 63.33 | 94.5 | 0.2778 |
| Training 3 — the round-2 router | 63.33 | 81.5 | 0.4588 |

The trained frontier has two points: Training 1 as the cost point (both-axes win over always-strong) and Training 3 as the accuracy point (the oracle, at four-fifths of always-strong's bill). On all 300 instances the always-strong baseline is 56.67 (run 2: 56.00); its 54.00 here is the same serving restricted to the test half — within one table, every number comes from the same instance set.

## Decision quality beyond accuracy and cost

**Calibration.** The router's deliver-probability is a strong decision score for every trained router (AUC 0.91-1.00 against actual weak success), but the routers differ sharply on literal calibration. Training 1's probabilities are saturated and literally overconfident (calibration error 0.34): it confidently delivers the hopeless set, which is reward-correct behavior that literal calibration counts as error. Trainings 2 and 3, which escalate everything uncertain, have nearly honest literal probabilities (error 0.03-0.04) — their confidence approximates the real probability the weak model succeeds. Calibration honesty and cost-efficiency trade off through the hopeless set. Saturation is high everywhere (77-83% of probabilities outside 5-95%), so each router offers only a few usable operating points — decisive decisions, a coarse dial.

**Latency.** One routing decision is one forward pass of the resident 1.5B router: ~19 ms median, ~25 ms at the 95th percentile — negligible beside episodes that run seconds to minutes.

**Repeatability.** The routing decision is deterministic: identical inputs give identical decisions. All pipeline variation comes from the executor models (+/-0.33 / +/-0.67-point noise floors), which affect the always-strong baseline equally — routing adds zero run-to-run variability.

## Findings

1. **A small learned router beats the strong-model-everywhere baseline on both axes at once.** +7.3 accuracy points at 46% cost savings on the held-out test — possible because always-strong is not the accuracy ceiling (the task-1 reversal) and not the cost floor (the hopeless set).
2. **Risk belongs at the decision layer, not in the gradient.** The learned risk penalty converged to the hard gate's exact high-tier behavior while degrading the rest of the policy through shared weights; the readout-time gate achieves identical protection cheaper, is auditable, and can be retuned without retraining.
3. **A hard risk gate is risk-aware but capability-blind — and its price is now measured.** +27 cost points for zero accuracy when the router is well calibrated; genuinely protective only against a router tuned too aggressively for cost.
4. **Early escalation, not reward-shaping, wins the accuracy corner.** Deciding at round 2 — before a doomed weak attempt burns its budget — reaches oracle accuracy at four-fifths of always-strong's cost, dominating the risk-trained router, with an operationally explainable rule.
5. **Structure for what you know, learning for what the reward can discover, hard rules only for governance.** Two insights entered through architecture, the third (don't pay for escalations that cannot fix the answer) was predicted as emergent and appeared unprompted; the one hand-coded rule stayed outside training as an inference-time constraint. Stated principles in the router's prompt were never needed — on HotpotQA they measurably hurt.

## Training 5 — the router decides the hand-off (human in the loop; 29 Oct)

Prof. Archan's second action item (28 Oct) asked for a router that can decide not to use any AI model and route the task to a person. Training 5 adds a third action to the same fresh-adapter GRPO(Reward) router: on high-risk instances (τ ≥ 0.8; tasks 5 and 9; 30 held-out test instances) the router chooses W (deliver the weak attempt), S (escalate to the strong model) or H (route to a person); elsewhere it remains the two-way router. The prompt gains one option line ("H) Hand the task to a person"); the system sentence is unchanged. The router decides from the task text and the weak attempt only — it never sees ground truth or a human price at test time. Ground truth prices the three branches in training exactly as it prices W and S: R′(H) = 𝟙[both models fail] − 𝟙[strong would succeed] − λ·C_W/C_SS,ᵢ, so a needed hand-off is rewarded and an unnecessary one (the strong model would have solved it) is penalised; the risk term became a risk-weighted failure penalty on whichever AI option was chosen, −λt·τᵢ·𝟙[τᵢ ≥ 0.8]·𝟙[option failed], with no term on H. A fresh adapter never sampled H often enough to learn it, so the first 10 of 40 epochs sample actions uniformly (v1.3c); nothing else changed. Two variants were read once each on the held-out test: 5a (λt = 0, no risk term) and 5b (λt = 1).

Results on the 30 high-risk test instances (greedy = primary operating point; graded after the fact: *needed* = neither model would have completed the task, *unnecessary* = the strong model would have): 5a routes 15 to a person (15 needed / 0 unnecessary — all of task 9, precision 1.0), delivers the other 15 (task 5) and leaves **15 high-risk failures that no person sees**; 65.33 % overall. 5b routes 27 to a person (19 needed / 8 unnecessary), escalates 3, delivers none and leaves **2 failures no person sees**; 76.00 % overall. At its secondary point (route to a person iff P(H) ≥ 0.8, θ chosen on validation) 5b hands over 17 (16 / 1) with 5 unseen failures, 74.00 %. The three-way oracle is 21 hand-offs, 0 unseen failures, 77.33 %. The reading: the router finds the both-fail task type on its own; the risk term is what stops it from delivering the cheap task-5 attempts (15 silent failures → 2), and the price of that caution is unnecessary hand-offs (8 greedy, 1 at the 0.8 point) — the governance dial.

Reporting conventions (registration entry of 29 Oct). *Accuracy* counts an instance routed to a person as completed correctly (the person is assumed to answer correctly); with no person available (the hand-off falls back to the strong model) the same decisions give 5a 55.33 % and 5b 63.33 % — Training 2's and the two-way oracle's number. *Cost* is AI cost plus **human cost**: one human execution per hand-off, priced at k × that instance's strong-call cost with k = 1 on the board — a convention, not a measurement; it is the floor (a person costs no less than the strong call it replaces) and it makes escalation and routing to a person cost the same, so the two routes are compared on outcome only. In a real environment the price is human time × loaded rate in the same unit. Totals: 5a 48.1 AI + 26.9 human = 75.0 % of always-strong; 5b 62.9 + 31.6 = 94.5 % (66.8 + 27.7 = 94.5 at the 0.8 point) — the same cost as Training 2 with 12.7 more accuracy points; at k = 1.5 the totals are 88.4 % and 110.4 %. On the high-risk tier a human execution so priced is 4.6× the weak attempt (2,574 vs 560 units); task 9, where the strong model spends 4,211 units per instance for 0/15, is where routing to a person replaces a guaranteed failure at the same price. Every number replays from the committed files with `training5_board_replay.py`.

What this does not show: 5b's unnecessary hand-offs are all task-5 instances the strong model would have solved; whether a router can separate those from the task-5 both-fail cases from the weak attempt alone is open (the two look alike to it). The human-cost convention is a floor; the trade-off a deployment faces is the slope (≈ 14–16 board points per +0.5 of k), not the k = 1 number.

## Limitations and threats to validity

- **Single seed per training.** Each training ran once (seed 17); Training 2's lambda_t-insensitivity across three runs and Training 1's exact fresh-start reproduction are partial substitutes, not replacements, for seed repeats.
- **Small evaluation sets.** 150 test instances (15 per task type) give coarse per-type resolution; the borderline lab-order type is the known noisy case (3/6 strong on validation vs ~23% overall) and was Training 1's one miss against the oracle.
- **Type-level decisions.** The routers' policies are effectively per-task-type on this benchmark, because within-type structure is homogeneous; datasets with within-type heterogeneity are where instance-level routing must still prove itself.
- **Risk scoring.** The rubric scorer is a mid-size model that required one scope clarification and systematically compresses severity; hand validation was at tier level by a non-clinician, with fine Q2/Q3 calibration flagged for clinician review.
- **Serving.** All results use an AWQ-quantized strong model on one GPU; the paper's exact environment is retired, so reproduction rests on two independent servings agreeing (56.0-58.0).
- **Deferred check.** Partial-credit correctness was designed but not run (its trigger — starved learning — never arose); the mechanical scheme is on file.

## Future work and provenance

**Scoped next chapter — mid-episode handoff.** Does continuing the strong model from the weak model's partial state beat restarting it? This cannot be replayed (every stored strong episode starts from scratch), so it requires live rollouts and its own registration. The round-2 architecture, the per-round cost store, and the restart baseline it must beat (63.33 at 81.5%) are in hand.

**Other openings.** A third benchmark to test the recipe's transfer (the full pipeline — split, costs, weak pilot, replay training, one test read — is now a repeatable procedure); clinician review of the risk rubric's fine calibration; seed repeats for the headline runs.

**Provenance.** Every number in this write-up chains to primary artifacts: the benchmark's own graders (unmodified dispatcher; the separately distributed reference solutions, obtained from the authors' README link and byte-verified between installations), the patient-disjoint split committed with manifest hashes, the frozen risk-score file (sha-256 in every manifest), per-run manifests with table hashes and adapter paths, and a pre-registration log that records every declared decision, prediction, and amendment in order. All scripts and the log are on the team repository's campaign branch.
