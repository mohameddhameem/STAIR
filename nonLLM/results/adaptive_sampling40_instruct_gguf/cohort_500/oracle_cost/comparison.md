# Retrospective oracle versus SS x40

Billion model parameters times logical prompt/output tokens; same proxy as gguf_analyze.py. QA shared prompt counted once. Includes one selected retrieval, k answer completions, one selected fusion and one selected extraction.

Gold-assisted retrospective oracle on a frozen availability-based training cohort. No search/controller cost. Assumes correct choice is known before execution; not the cost of exhaustively searching all configurations. Not measured GPU time, FLOPs or dollars. Fusion prompt tokens counted in full, independent of incidental cache reuse.

If every pair/budget is incorrect, use SS x40, the same as baseline (86 questions). Never drop failed questions.

| Policy | Correct / 500 | Lenient EM | Gain (pp) | Total cost reduction | Generation cost reduction | Mean k |
|---|---:|---:|---:|---:|---:|---:|
| ss40 | 189 | 37.8% | 0.0 | 0.00% | 0.00% | 40.000 |
| min_k_then_cost | 414 | 82.8% | 45.0 | 45.63% | 45.98% | 8.972 |
| min_cost_then_k | 414 | 82.8% | 45.0 | 57.75% | 58.18% | 11.620 |
