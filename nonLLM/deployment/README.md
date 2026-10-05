# GGUF inference setup

Three Q4_K_M weights are stored outside Git under `/ssd1/zmcheng/cs707/models`.
`models.json` records repository revisions, upstream base models, exact file sizes,
and SHA-256 hashes. All weight hashes match the Hugging Face LFS metadata.

- 7B Base: `mradermacher/Qwen2.5-7B-GGUF` (4,683,073,984 bytes).
- 1.5B Base: `mradermacher/Qwen2.5-1.5B-GGUF` (986,048,544 bytes).
- 1.5B Instruct: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` (1,117,320,736 bytes).

The requested official Base GGUF repository IDs returned HTTP 401; community
model cards identify the exact requested upstream Qwen Base models.
Tokenizers/configuration files are available under `models/tokenizers/`.
Base models should use text completion; Instruct models can use chat templates.
GGUF quantized generation may differ from the previously measured dataset grid.
GGUF files cannot directly replace the Transformers model used for LoRA training
in the reference `sft_router.py`.

## llama.cpp

Standalone CUDA build, without modifying the existing Python training environment:
`tools/llama.cpp`, commit `d89651a7b205c03c4a0b13cd0646d400dc929f79`.
CUDA 12.4, architecture SM86 (RTX A5000).

Re-run the temporary checks:

```bash
python3 /ssd1/zmcheng/cs707/STAIR/nonLLM/deployment/gguf_smoke.py --gpu 0
```

This loads one model at a time with context 1024, one slot, two CPU threads and
32 output tokens, checks available VRAM, and terminates each server it starts.
Outputs and server logs go to `runs/gguf_smoke/` outside Git.
This is an inference/API smoke check, not a QA accuracy or throughput benchmark.

Start a persistent 7B Base completion service when sufficient VRAM is free:

```bash
CUDA_VISIBLE_DEVICES=0 /ssd1/zmcheng/cs707/tools/llama.cpp/build/bin/llama-server \
  -m /ssd1/zmcheng/cs707/models/Qwen2.5-7B-GGUF/Qwen2.5-7B.Q4_K_M.gguf \
  -ngl 99 -c 4096 -np 1 -b 128 -ub 128 -t 2 --host 127.0.0.1 --port 18087
```

Use `/v1/completions` for Base models. `/completion` is the native API used by
the smoke script; Instruct can use `/v1/chat/completions`.
Larger context and concurrent requests need extra KV-cache memory.

## SGLang

Current upstream documentation supports `gguf` on NVIDIA CUDA, and the server
arguments include `--load-format gguf` and `--quantization gguf`.
SGLang is not installed or tested in this environment. The following is a
starting recipe for an isolated SGLang environment, not a verified local command:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /ssd1/zmcheng/cs707/models/Qwen2.5-7B-GGUF/Qwen2.5-7B.Q4_K_M.gguf \
  --tokenizer-path /ssd1/zmcheng/cs707/models/tokenizers/Qwen2.5-7B \
  --load-format gguf --quantization gguf --dtype float16 \
  --context-length 2048 --mem-fraction-static 0.35 \
  --max-total-tokens 2048 --max-running-requests 1 \
  --disable-cuda-graph --host 127.0.0.1 --port 18088
```

Re-check available VRAM before launch. The static memory fraction refers to
GPU capacity and covers weights plus the KV pool; account for other GPU jobs
and runtime overhead. GGUF model/quantization compatibility still needs an actual
SGLang load check on the installed release.

Sources:
- https://docs.sglang.io/docs/advanced_features/quantization
- https://docs.sglang.io/docs/advanced_features/server_arguments
- https://github.com/ggml-org/llama.cpp
- https://qwen.readthedocs.io/en/v2.5/run_locally/llama.cpp.html

## Verified local smoke run (2026-10-05)

All three models loaded and generated successfully on GPU 0 (RTX A5000), while
existing experiments remained running. Context 1024, single request:

| Model | GPU process memory after load | Short-run generation tokens/s |
|---|---:|---:|
| 7B Base | 4498 MiB | 42.95 |
| 1.5B Base | 1218 MiB | 105.06 |
| 1.5B Instruct | 1218 MiB | 100.22 |

GPU memory was measured after loading, not a peak-memory measurement. Speed was
measured on only 32 generated tokens for Base and 8 for Instruct with concurrent
GPU jobs, so it is not a reliable performance benchmark.
All three generated Paris in response to a France-capital prompt. All temporary
servers were shut down and their GPU memory released. Full local results:
`/ssd1/zmcheng/cs707/runs/gguf_smoke/results.json`.
