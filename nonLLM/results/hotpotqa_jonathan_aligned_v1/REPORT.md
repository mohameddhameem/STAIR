# Jonathan-aligned non-LLM routers

Pinned Jonathan Config B executor outputs and common slr-train/val/test (4,814/1,110/1,481). Standard lenient. All trained routers are non-generative linear classifiers. Three seeds averaged before validation threshold selection.

Stage GRPO combines the production stage reward with the committed exploratory GRPO update. It is an adaptation, not a reproduction claim for the slide-only 76.23% result. Query4 uses the exploratory script's separate official-EM/hinge reward. No executor training or new answers.

Costs below exclude classifier CPU work and LLM-router calls, matching configb_test_readout.py. They are Table5 budget prices, not measured token units. SS direct929; weak-first SS1009. Author training SS929 simplification is retained and disclosed.

## Validation-selected maximum-accuracy policies

| Method | Accuracy | Cost | % SS | Rescues / harms vs SS |
|---|---:|---:|---:|---:|
| fixed_WW | 50.30% | 80.0 | 8.61% | 49 / 441 |
| fixed_WS | 64.01% | 192.0 | 20.67% | 20 / 209 |
| fixed_SW | 61.78% | 817.0 | 87.94% | 48 / 270 |
| fixed_SS | 76.77% | 929.0 | 100.00% | 0 / 0 |
| cascade_always_escalate | 76.77% | 1009.0 | 108.61% | 0 / 0 |
| oracle_cheapest_correct_cascade | 81.50% | 216.6 | 23.32% | 70 / 0 |
| sft | 76.50% | 1001.2 | 107.78% | 2 / 6 |
| sft_reward_qa | 76.57% | 1001.6 | 107.82% | 2 / 5 |
| sft_reward_both | 76.64% | 1002.8 | 107.94% | 2 / 4 |
| grpo_stage_cold | 76.37% | 1000.4 | 107.69% | 1 / 7 |
| sft_grpo_stage | 76.03% | 990.3 | 106.60% | 1 / 12 |
| sft_grpo_stage_extended | 76.03% | 990.3 | 106.60% | 1 / 12 |
| grpo_query4_cold | 63.74% | 463.6 | 49.90% | 34 / 227 |
| sft_grpo_query4 | 50.30% | 80.0 | 8.61% | 49 / 441 |
| rwsft_query4 | 64.35% | 451.8 | 48.63% | 34 / 218 |

## Validation-selected SS-budget policies

The budget constraint is applied on validation. Test cost is reported as observed; it is not guaranteed by the validation constraint.

| Method | Test accuracy | Test cost | % SS |
|---|---:|---:|---:|
| sft | 69.75% | 460.1 | 49.52% |
| sft_reward_qa | 69.75% | 460.3 | 49.55% |
| sft_reward_both | 71.37% | 560.3 | 60.32% |
| grpo_stage_cold | 67.72% | 405.0 | 43.59% |
| sft_grpo_stage | 74.41% | 897.3 | 96.59% |
| sft_grpo_stage_extended | 74.41% | 897.3 | 96.59% |
| grpo_query4_cold | 63.74% | 463.6 | 49.90% |
| sft_grpo_query4 | 50.30% | 80.0 | 8.61% |
| rwsft_query4 | 64.35% | 451.8 | 48.63% |

## Reference two-stage reward row

The pinned configb_pair_sweep.py selects the both-reward peak thresholds (0.95,0.60,1.00) on validation. Applying its committed ret_reward/qa_reward scores to test gives 76.23% at76.98% of SS cost, excluding router calls. This reproduces the slide's accuracy, but not its80.5% cost. The source training scripts compute exact expected reward with SFT warm starts. This is evidence for the both-reward row; it does not establish the slide used sampled GRPO. See author_four_pairs.json and author_readout.txt.

The exact-reward non-LLM counterpart here is sft_reward_both. Sampled stage GRPO is reported separately. No claim of matching the slide's unreconciled cost convention is made.

## Training duration

42 completed fits across three seeds. SFT and exact reward use three epochs; sampled GRPO uses two. The extended stage GRPO arms allowed30 epochs with validation patience5: all six stage/seed fits peaked at epoch1 and stopped at epoch6. Extension did not change the selected policy.

## Limits

The reference LLM-router readout is SFT 76.70%/834.8, SFT+reward(QA only) 75.96%/664.7, SS76.77%/929. Same executor outcomes and split make this comparison much closer than the earlier local GGUF test500 experiment. Router family/features/optimization batch implementation differ. This test was inspected in earlier research; results are descriptive. No final-test choice of checkpoint or threshold. All objectives, including cheap95 and SS-budget, are in results.csv.

ms_batch_decision_per_question includes feature-array loading and model/probability evaluation, excludes TF-IDF feature construction; it is not online end-to-end decision latency.
