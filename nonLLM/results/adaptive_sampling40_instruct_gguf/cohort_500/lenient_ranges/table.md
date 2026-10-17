# Lenient EM budget distribution

Scores use the same gold-independent v2 extracted answers. Lenient EM reuses gguf_analyze.loose: official normalization, guarded bidirectional word-boundary containment. Locally defined; not certified identical to the missing upstream scorer.

Frozen availability-based training cohort, not held-out accuracy. Success sets and ranges use gold retrospectively. Min/max bounds do not imply every intervening budget succeeds. Each k uses the first k of the same forty candidates and its own fusion.

W = 1.5B, S = 7B. First letter retrieval; second answer/fusion.

| Pair | First 1 | First 2–5 | First 6–10 | First 11–20 | First 21–40 | Never correct |
|---|---:|---:|---:|---:|---:|---:|
| WW | 126 | 93 | 21 | 26 | 17 | 217 |
| WS | 191 | 90 | 24 | 15 | 22 | 158 |
| SW | 151 | 101 | 27 | 22 | 12 | 187 |
| SS | 215 | 87 | 24 | 10 | 10 | 154 |
| any_pair | 311 | 69 | 16 | 10 | 8 | 86 |

`any_pair` permits switching both model pair and budget after seeing gold.

| Pair | Last 1 | Last 2–5 | Last 6–10 | Last 11–20 | Last 21–40 |
|---|---:|---:|---:|---:|---:|
| WW | 11 | 9 | 11 | 19 | 233 |
| WS | 8 | 17 | 10 | 15 | 292 |
| SW | 6 | 5 | 5 | 14 | 283 |
| SS | 8 | 10 | 5 | 15 | 308 |
| any_pair | 5 | 3 | 2 | 9 | 395 |

| Pair | Best shared k | Shared hits | Any-k hits | Median [min, max] among successful questions | Non-contiguous sets |
|---|---|---:|---:|---|---:|
| WW | [11, 13] | 139 | 283 | [2, 39] | 228 |
| WS | [1] | 191 | 342 | [1.0, 39.0] | 264 |
| SW | [12] | 180 | 313 | [2, 39] | 263 |
| SS | [1] | 215 | 346 | [1.0, 40.0] | 294 |

Median min and max are separate marginal medians, not the success interval of a representative question. Per-question exact sets and contiguous intervals are in per_question.csv; full 1–40 histograms are in histograms.csv.
