# Higher-concurrency diagnostic benchmark

Same first 16 training-extra question IDs; alternating Qwen7/Qwen72 evidence branches; original task messages, model, tool-disabled frozen template, greedy decoding, max512 generated tokens and shared16K Q8 KV. Both servers configured with 16 slots; client concurrency varied. Admission reserves prompt+512 tokens, so long requests may reduce effective concurrency.

| GPU pair | Active requests | Wall seconds | Requests/s | Generated tokens/s |
|---|---:|---:|---:|---:|
| 2/3 | 8 | 71.26 | 0.225 | 22.33 |
| 2/3 | 16 | 35.79 | 0.447 | 34.53 |
| 0/1 | 4 | 47.25 | 0.339 | 24.53 |
| 0/1 | 8 | 42.47 | 0.377 | 27.60 |
| 0/1 | 16 | 34.24 | 0.467 | 35.28 |

Selected runtime: 16 requests per two-GPU replica, two replicas (32 maximum). GPU0/1 token throughput improves from24.53 to35.28 tokens/s (1.44x); GPU2/3 from22.33 at8 requests to34.53 at16. Small finite batches and variable answer lengths make these diagnostic figures, not full-run speed guarantees. Batched numerical differences change some text and token counts. No accuracy labels were used to choose runtime.

Shared KV pool remains16384 tokens per replica, with each individual request retaining its16384-token limit. No context truncation or sampling/model changes. Existing corrected caches remain; every new record identifies server_parallel=16 and kv_unified=true. Runtime can be set via CS707_REQUESTS_PER_REPLICA (1 to32).
