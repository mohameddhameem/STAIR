# MedAgentBench non-LLM replay: Jonathan-aligned Training 1 and 3

Same committed replay views, outcomes, 90/60/150 patient-disjoint split, benchmark pass/fail grading, reward/cost accounting, and fresh GRPO A1 mechanics. Primary seed17; seeds0/1/2 are independent repeats. No executor calls or training, FHIR server, raw episode generation, or SFT warm start.

## What is aligned

Weak Qwen2.5-7B vs strong Qwen2.5-72B-AWQ; one deliver/escalate decision. Training1 sees completed weak attempt; Training3 sees round2 view, deliver pays full weak cost and escalate pays round2 weak cost + full strong cost. Both normalize report cost by the same full-episode always-escalate sum(cw+cs). This is not HotpotQA Table5/929 or direct-strong-only accounting.

Fresh classifier, G8; advantage=reward-group mean, no std normalization, no KL/PPO; entropy.03; AdamW1e-5, accum8, clip1,40epochs; best rounded validation delivered-path reward every10 optimizer updates. Features fitted only on train90; only view content is used, with TF-IDF and numerical text statistics. Parameterization and pretrained knowledge differ from the LLM router by design.

## Primary seed17, fixed theta0.50, no risk gate

| Setting | Router | Accuracy | Cost / full always-escalate | Deliver /150 |
|---|---|---:|---:|---:|
| T1 | Always weak | 12.00% | 18.72% | 150 |
| T1 | Always escalate | 54.00% | 100.00% | 0 |
| T1 | Oracle | 63.33% | 48.46% | 73 |
| T1 | Jonathan LLM | 61.33% | 54.11% | 60 |
| T1 | linear | 52.00% | 78.74% | 35 |
| T1 | mlp64 | 61.33% | 54.11% | 60 |
| T1 | mlp128_64 | 50.67% | 46.86% | 79 |
| T3 | Always weak | 12.00% | 18.72% | 150 |
| T3 | Always escalate | 54.00% | 86.92% | 0 |
| T3 | Oracle | 63.33% | 43.90% | 73 |
| T3 | Jonathan LLM | 63.33% | 81.47% | 15 |
| T3 | linear | 35.33% | 60.51% | 75 |
| T3 | mlp64 | 61.33% | 48.38% | 60 |
| T3 | mlp128_64 | 61.33% | 48.38% | 60 |

## Primary seed17, threshold selected on validation reward

| Setting | Router | Theta | Test accuracy | Cost / full always-escalate |
|---|---|---:|---:|---:|
| T1 | linear | 0.49 | 44.00% | 63.79% |
| T1 | mlp64 | 0.50 | 61.33% | 54.11% |
| T1 | mlp128_64 | 0.51 | 61.33% | 54.11% |
| T3 | linear | 0.53 | 63.33% | 81.47% |
| T3 | mlp64 | 0.52 | 63.33% | 58.45% |
| T3 | mlp128_64 | 0.50 | 61.33% | 48.38% |

## Interpretation and limits

Fixed theta0.50 matches the reference decision rule, but classifiers do not share LLM probability calibration. The additional validation-reward threshold is chosen only on calib_sweep, then applied to test. All models/thresholds were locked before a single test inference per model. Gate and OP-B views reuse the same probabilities. The source campaign test results were already public: this is a paired replay, not a freshly secret test.

Decision-model FLOPs and frozen risk-score generation are excluded from the supplied cost proxy, as on the replay board. Batch CPU feature+decision timing is measured separately; it is not streaming per-request latency. No clinical performance or safety claim follows from this small replay. Uncertainty should be clustered by patient rather than task.

Scope is Training1 andTraining3, two AI actions. The newer Training5 human-handoff policies require different action space, failure penalties and human-cost assumptions and are not pooled here. Full seed/point/risk-gate results are in results.csv; independent0/1/2 mean/range in seed_repeats.json.

## Risk-gate paired readout

The lower-cost ungated round2 point delivers 15 high-risk weak failures. When the same tau>=.8 forced-escalation guardrail is applied, those apparent savings disappear. The gate uses the author's frozen risk-score cache; risk-score generation cost is excluded for both router families.

| Setting | MLP64 seed17 point | Gate | Accuracy | Cost % | High-risk wrong deliveries |
|---|---|---|---:|---:|---:|
| T1 | OP_A_fixed_050 | False | 61.33% | 54.11% | 15 |
| T1 | OP_A_fixed_050 | True | 61.33% | 80.99% | 0 |
| T1 | val_reward_selected | False | 61.33% | 54.11% | 15 |
| T1 | val_reward_selected | True | 61.33% | 80.99% | 0 |
| T3 | OP_A_fixed_050 | False | 61.33% | 48.38% | 15 |
| T3 | OP_A_fixed_050 | True | 61.33% | 71.39% | 0 |
| T3 | val_reward_selected | False | 63.33% | 58.45% | 15 |
| T3 | val_reward_selected | True | 63.33% | 81.47% | 0 |

## Seed repeat sensitivity

These are independent seeds0/1/2, with no ensemble, not a selection of the best seed. Seed17 is the source-matching primary run. Results vary substantially under the fixed low learning rate and small90-task training set; a single matching seed does not establish robust reproduction.

| Setting | Router | Point | Mean accuracy (range) | Mean cost % |
|---|---|---|---:|---:|
| T1 | linear | OP_A_fixed_050 | 35.11% (31.33–39.33) | 53.36% |
| T1 | linear | val_reward_selected | 48.89% (31.33–61.33) | 72.57% |
| T1 | mlp64 | OP_A_fixed_050 | 53.33% (52.00–54.00) | 80.09% |
| T1 | mlp64 | val_reward_selected | 47.56% (30.00–61.33) | 55.36% |
| T1 | mlp128_64 | OP_A_fixed_050 | 42.44% (12.00–61.33) | 57.61% |
| T1 | mlp128_64 | val_reward_selected | 56.44% (54.00–61.33) | 84.70% |
| T3 | linear | OP_A_fixed_050 | 47.78% (41.33–56.67) | 62.28% |
| T3 | linear | val_reward_selected | 54.67% (53.33–56.67) | 74.91% |
| T3 | mlp64 | OP_A_fixed_050 | 47.11% (33.33–54.00) | 68.35% |
| T3 | mlp64 | val_reward_selected | 58.00% (54.00–62.67) | 67.64% |
| T3 | mlp128_64 | OP_A_fixed_050 | 43.11% (12.00–63.33) | 54.70% |
| T3 | mlp128_64 | val_reward_selected | 57.11% (54.00–63.33) | 69.76% |

Paired patient-cluster bootstrap intervals (2,000 resamples) are in paired_patient_comparisons.json. They address patient clustering on this replay; they do not remove model-selection multiplicity, prior public-test knowledge, or seed sensitivity.
