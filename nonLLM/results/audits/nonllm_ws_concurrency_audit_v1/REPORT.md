# Llama70 concurrency benchmark

Eight fixed training-extra questions with Qwen7 evidence, same weights, prompt, frozen tool-disabled chat template, greedy decoding, maximum 512 generated tokens, Q8 KV and no truncation. Server: two A5000 GPUs, 4 slots, unified 16384-token KV pool, original batch256/ubatch64.

| Active requests | Wall seconds | Requests/s | Generated tokens/s | Exact text matches to serial |
|---:|---:|---:|---:|---:|
| 1 | 70.83 | 0.113 | 11.94 | 8/8 |
| 2 | 41.36 | 0.193 | 14.77 | 6/8 |
| 4 | 35.78 | 0.224 | 17.08 | 6/8 |

Four concurrent requests approximately halve this sample's elapsed time. Generated-token throughput improves by 1.43x. Different outputs contain different token counts, so request throughput is not a strictly equal-token comparison. Only eight diagnostic training cases were used; this is not a whole-run ETA or accuracy estimate. Batched floating-point computation changes 2/8 texts compared with this serial baseline. Runtime choice is based on speed and memory, not validation/test accuracy.

Applied runtime: two replicas, each serving up to four concurrent requests with shared 16384-token KV capacity. A client token budget reserves actual rendered prompt length plus 512 output tokens per active request, delaying long requests when necessary. A single writer per replica commits complete records, retaining crash-safe JSONL caching. Context/token limits, model, decoding, prompt and metric settings are unchanged. Cached serial records are retained; new records explicitly identify server_parallel=4 and kv_unified=true. All router policies share the resulting executor cache. No bitwise serial-equivalence claim is made.
