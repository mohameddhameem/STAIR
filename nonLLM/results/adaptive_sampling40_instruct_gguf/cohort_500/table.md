# Matched 500-question reference

500 distinct training questions with complete WW/SW results; prioritize already complete WS/SS without inspecting correctness, then fixed-seed sample remaining eligible questions. Descriptive availability-based cohort, not a uniform full-dataset sample or held-out score.

W=1.5B, S=7B. First letter: retrieval; second: answer/fusion.

Strict extracted EM. Every question has all four pairs and k=1..40.

| Pair | Any successful k | First k=1 | 2-5 | 6-10 | 11-20 | 21-40 |
|---|---:|---:|---:|---:|---:|---:|
| WW | 225 | 95 | 81 | 15 | 19 | 15 |
| WS | 279 | 145 | 81 | 21 | 16 | 16 |
| SW | 257 | 121 | 83 | 20 | 23 | 10 |
| SS | 288 | 164 | 80 | 26 | 8 | 10 |

Counts overlap across pairs. This training availability-based cohort is not a uniform sample or held-out benchmark.

| Pair | Best shared k on this cohort | Strict hits | Retrospective budget oracle hits |
|---|---|---:|---:|
| WW | [13] | 111 | 225 |
| WS | [1] | 145 | 279 |
| SW | [3] | 142 | 257 |
| SS | [8] | 168 | 288 |

Gold-assisted oracle selection is descriptive, not a deployable or cost-constrained policy. Strict EM can miss aliases; long-budget successes need manual interpretation.
