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

HotpotQA golden set (Jonathan's Stage-Level Routing, which the team follows): costs are AgentTTS's Table 5 prices, on
its normalised budget scale (Eq. 3-4), not units. A call is priced at the paper's Table 4 lengths, weak-first, and every
call that ran is counted: TABLE5_HOTPOTQA, SLR_PATH_PRICES, STRONG_EVERYTHING. For a model or call that Table 5 does
not list, budget() is the same formula (3 x units - 2, floored at 0).
"""

UNIT_PARAMS_B = 3.0
UNIT_TOKENS = 128 + 64
UNIT_GFLOPS = 2 * UNIT_PARAMS_B * UNIT_TOKENS  # 1,152 GFLOPs


def units(params_b: float, prompt_tokens: int, generated_tokens: int) -> float:
    """AgentTTS units of one call of a model with `params_b` billion parameters. For several samples from one prompt,
    pass the prompt once and the samples' generated tokens summed (AgentTTS App. A.2, Eq. 8)."""
    return 2 * params_b * (prompt_tokens + generated_tokens) / UNIT_GFLOPS


def budget(params_b: float, prompt_tokens: int, generated_tokens: int) -> float:
    """AgentTTS's normalised budget of one call (Eq. 3-4): the 3B samples, sharing one 128-token prompt, of equal FLOPs.
    That is 3 x units - 2, floored at 0 for calls cheaper than one 3B sample (the formula goes negative there). Samples
    from one prompt are one call, as in units(). At Table 4's lengths it reproduces Table 5."""
    return max(0.0, 3 * units(params_b, prompt_tokens, generated_tokens) - 2)


# AgentTTS's Table 5 (S = 1 sample), HotpotQA: the golden set's per-call prices. Not units: these are AgentTTS's
# normalised budget (Eq. 3-4), the number of 3B samples sharing one 128-token prompt with the same FLOPs, which is
# 3 x units - 2 for one call. At the paper's Table 4 lengths (retrieve 2,048 + 128 tokens, answer 256 + 64),
# round(budget(params_b, prompt, generated)) gives every entry: Qwen2.5-72B retrieval is 272 units and 814 here.
TABLE5_HOTPOTQA = {
    "retrieve": {"Qwen2.5-72B": 814, "Qwen2.5-32B": 361, "Qwen2.5-7B": 77, "Llama-3.1-70B": 791, "Llama-3.1-8B": 89,
                 "Llama-3.2-3B": 32},
    "answer": {"Qwen2.5-72B": 118, "Qwen2.5-32B": 51, "Qwen2.5-7B": 10, "Llama-3.1-70B": 115, "Llama-3.1-8B": 11,
               "Llama-3.2-3B": 3},
}
AGENTTTS_TABLE5_HOTPOTQA = TABLE5_HOTPOTQA  # the 0.2 name

# Menu A's four paths as the golden set prices them (Jonathan's configb_test_readout.py): weak retrieval Qwen2.5-7B,
# strong Qwen2.5-72B; weak answer Llama-3.2-3B, strong Llama-3.1-70B. Weak-first, so an escalated stage also pays for
# the weak call that ran before it. The router is not charged.
_R, _A = TABLE5_HOTPOTQA["retrieve"], TABLE5_HOTPOTQA["answer"]
SLR_PATH_PRICES = {
    "WW": _R["Qwen2.5-7B"] + _A["Llama-3.2-3B"],                                              # 80
    "WS": _R["Qwen2.5-7B"] + _A["Llama-3.2-3B"] + _A["Llama-3.1-70B"],                        # 195
    "SW": _R["Qwen2.5-7B"] + _R["Qwen2.5-72B"] + _A["Llama-3.2-3B"],                          # 894
    "SS": _R["Qwen2.5-7B"] + _R["Qwen2.5-72B"] + _A["Llama-3.2-3B"] + _A["Llama-3.1-70B"],    # 1,009
}
STRONG_EVERYTHING = _R["Qwen2.5-72B"] + _A["Llama-3.1-70B"]  # 929: one strong retrieval and one strong answer
