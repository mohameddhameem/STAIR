"""Cost in AgentTTS units, the team's one cost measure.

AgentTTS (Sec. 4.1, Eq. 2-4) normalises inference FLOPs so that one pass of the smallest model on the cheapest task is
1 unit: Llama-3.2-3B with 128 prompt + 64 generated tokens. A call costs

    units = 2 x params x (prompt_tokens + generated_tokens) / (2 x 3e9 x 192)

with the tokens actually measured for that call. Sum it over every model call a question makes: retrieval, answering,
fusion, and the router or judge. Limit, stated with every number: like AgentTTS, it counts parameter FLOPs only,
not attention, so small models on long prompts are undercounted (about 60% for a 0.5B on a 100-passage prompt).

Samples share their prompt. S samples drawn from one prompt are one call: the prompt counts once and every sample's
generated tokens count, units(params_b, prompt_tokens, sum of the samples' generated tokens). This is AgentTTS's own
accounting (App. A.2, Eq. 8: FLOPs = prompt + S x decode, because "Transformer models can exploit batching to share
the cost of prompt encoding across multiple decoding passes"). A call with a new prompt, such as fusing the samples,
counts its own prompt.
"""

UNIT_PARAMS_B = 3.0
UNIT_TOKENS = 128 + 64
UNIT_GFLOPS = 2 * UNIT_PARAMS_B * UNIT_TOKENS  # 1,152 GFLOPs


def units(params_b: float, prompt_tokens: int, generated_tokens: int) -> float:
    """AgentTTS units of one call of a model with `params_b` billion parameters. For several samples from one prompt,
    pass the prompt once and the samples' generated tokens summed (AgentTTS App. A.2, Eq. 8)."""
    return 2 * params_b * (prompt_tokens + generated_tokens) / UNIT_GFLOPS


# AgentTTS's Table 5 (S = 1 sample), HotpotQA: per-call units with their prompts, for checking a reproduction.
AGENTTTS_TABLE5_HOTPOTQA = {
    "retrieve": {"Qwen2.5-72B": 814, "Qwen2.5-32B": 361, "Qwen2.5-7B": 77, "Llama-3.1-70B": 791, "Llama-3.1-8B": 89,
                 "Llama-3.2-3B": 32},
    "answer": {"Qwen2.5-72B": 118, "Qwen2.5-32B": 51, "Qwen2.5-7B": 10, "Llama-3.1-70B": 115, "Llama-3.1-8B": 11,
               "Llama-3.2-3B": 3},
}
