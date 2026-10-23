# Non-LLM GRPO continuation to 300 epochs

54 seed-specific streams: base/aug9 × exact/G16/G32 × λ0/.15/.4 × seeds17/29/43. Every replay matched original epoch120 weights bit for bit before uninterrupted continuation. Full optimizer and RNG states saved at checkpoints.

Primary: unchanged review-validated lenient reconstruction. Original mint_labels.py unavailable. Same regenerated Instruct cohort: SS test52.94%, val52.52%. Test previously inspected; comparisons descriptive. Exact means exact expected reward, not exact-match scoring.

| Family / objective | G | λ | Selected epoch | Val % | Old val % | Test % | Old test % | Cost % SS | Old cost % SS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_exact_cheap95 | 0 | 0.0 | 30 | 49.91 | 49.91 | 49.16 | 49.16 | 67.36 | 67.36 |
| base_exact_max_accuracy | 0 | 0.0 | 120 | 52.70 | 52.70 | 51.92 | 51.92 | 114.47 | 114.47 |
| base_exact_ss_budget | 0 | 0.15 | 240 | 52.07 | 51.98 | 50.84 | 51.59 | 90.35 | 96.05 |
| base_grpo_cheap95 | 16 | 0.0 | 240 | 49.91 | 49.91 | 49.49 | 49.02 | 69.06 | 67.45 |
| base_grpo_max_accuracy | 16 | 0.15 | 30 | 52.79 | 52.79 | 51.86 | 51.86 | 119.92 | 119.92 |
| base_grpo_ss_budget | 32 | 0.15 | 300 | 52.25 | 51.62 | 51.86 | 51.86 | 100.16 | 101.75 |
| aug9_exact_cheap95 | 0 | 0.0 | 30 | 49.91 | 49.91 | 49.16 | 49.16 | 68.25 | 68.25 |
| aug9_exact_max_accuracy | 0 | 0.0 | 180 | 52.70 | 52.61 | 52.46 | 52.67 | 117.92 | 120.47 |
| aug9_exact_ss_budget | 0 | 0.4 | 300 | 52.07 | 51.89 | 51.79 | 51.25 | 96.51 | 90.88 |
| aug9_grpo_cheap95 | 32 | 0.0 | 30 | 49.91 | 49.91 | 49.43 | 49.56 | 70.22 | 69.97 |
| aug9_grpo_max_accuracy | 16 | 0.15 | 300 | 52.70 | 52.70 | 52.40 | 52.73 | 118.32 | 121.41 |
| aug9_grpo_ss_budget | 32 | 0.4 | 240 | 51.89 | 51.62 | 51.25 | 50.78 | 99.06 | 97.58 |

All checkpoints and deployment thresholds selected on validation before test inference. Costs include sunk weak calls, use parameter-weighted recorded-token proxies, and exclude router/training overhead. More training does not remove the cascade cost floor.

Earlier search includes G8 and confidence/aug5; this extension focuses base/aug9 and G16/G32. Each family comparison includes its old checkpoints and hence prior validation winner for base/aug9 if G16/G32/exact; G8 prior winners can be outside extension.
