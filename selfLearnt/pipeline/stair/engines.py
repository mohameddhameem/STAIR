"""Model engines: vLLM for generation (tensor parallel, LoRA adapters, token log-probabilities, n samples per prompt) and for
last-token hidden states (pooling runner, used by the probes of workstream 3); a mock engine for CPU dry runs."""
import json
import os

import random
import re
import zlib

from .common import log


def _chat_kwargs(thinking):
    return {"enable_thinking": thinking} if thinking is not None else {}


# FlashInfer compiles its sampling kernels on first use (needs a CUDA toolkit matching PyTorch's); on the cluster that
# compile fails (CUDA 12.4 module vs PyTorch built for CUDA 13). vLLM's own PyTorch sampler needs no compiling.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


class VLLMEngine:
    def __init__(self, model, max_model_len, gpu_util=0.88, seed=0, lora_path=None, pooling=False):
        """model: a resolved model entry (path, tensor_parallel, quantization, no_thinking)."""
        from vllm import LLM
        tp = int(model.get("tensor_parallel", 1))
        vis = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
        if tp > 1 and len(vis) > 1:
            tp = len(vis)        # a split model uses every GPU the job got (2 large or 4 smaller cards)
        # PyTorch's expandable segments (set in the job scripts for training) clash with vLLM's own memory pool: on two
        # GPUs the CUDA-graph capture ran out of memory. vLLM's worker processes inherit this variable, so drop it here.
        if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF")
        kw = dict(model=model["path"], max_model_len=int(max_model_len), gpu_memory_utilization=gpu_util,
                  seed=seed, tensor_parallel_size=tp, enable_prefix_caching=not pooling)
        if tp > 1:
            # vLLM's custom all-reduce needs direct GPU-to-GPU access, which these PCIe nodes do not give ("custom_all_reduce
            # ... invalid argument"); NCCL's all-reduce works everywhere.
            kw["disable_custom_all_reduce"] = True
        if model.get("quantization") or "awq" in model["path"].lower():
            kw["quantization"] = model.get("quantization") or "awq"
        if lora_path:
            kw.update(enable_lora=True, max_lora_rank=int(model.get("lora_rank", 16)), max_loras=1)
        if pooling:
            self.llm = _pooling_llm(LLM, kw)
        else:
            self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.thinking = False if model.get("no_thinking") else None
        self.lora = None
        if lora_path:
            from vllm.lora.request import LoRARequest
            self.lora = LoRARequest("adapter", 1, lora_path)
        self.max_len = int(max_model_len)

    def encode(self, system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **_chat_kwargs(self.thinking))
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def encode_full(self, system, user, answer):
        """Prompt + the model's own output, for hidden states after generation."""
        return self.encode(system, user) + list(self.tok(answer, add_special_tokens=False)["input_ids"])

    def generate(self, prompts, n, temperature, max_tokens, seeds, meta=None):
        """prompts: token-id lists. Returns, per prompt, a list of n tuples (text, out_tokens, mean_logprob).
        A prompt longer than the context is cut from the left side of its references (the end, with the instruction, is kept)."""
        from vllm import SamplingParams
        sps, inputs = [], []
        for p, s in zip(prompts, seeds):
            room = self.max_len - max_tokens - 8
            if len(p) > room:
                p = p[:room // 2] + p[-(room - room // 2):]
            inputs.append({"prompt_token_ids": p})
            sps.append(SamplingParams(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, max_tokens=max_tokens, seed=s,
                                      logprobs=1, repetition_penalty=1.0))
        kw = {"lora_request": self.lora} if self.lora else {}
        outs = self.llm.generate(inputs, sps, use_tqdm=False, **kw)
        return [[(c.text, len(c.token_ids), _mean_logprob(c)) for c in o.outputs] for o in outs]

    def hidden(self, prompts, meta=None):
        """Last-token hidden state (final layer) for each token-id list."""
        room = self.max_len - 8
        inputs = [{"prompt_token_ids": (p if len(p) <= room else p[:room // 2] + p[-(room - room // 2):])} for p in prompts]
        if hasattr(self.llm, "embed"):
            outs = self.llm.embed(inputs, use_tqdm=False)
            return [list(map(float, o.outputs.embedding)) for o in outs]
        outs = self.llm.encode(inputs, use_tqdm=False)
        return [list(map(float, o.outputs.data.reshape(-1).tolist())) for o in outs]


def _pooling_llm(LLM, kw):
    """Hidden-state runner; the argument names changed across vLLM releases, so try the known spellings."""
    errs = []
    try:
        from vllm.config import PoolerConfig
        pc = PoolerConfig(pooling_type="LAST", normalize=False)
    except Exception as e:  # noqa: BLE001
        pc, errs = None, [f"PoolerConfig: {e}"]
    for extra in ({"runner": "pooling", "convert": "embed", "pooler_config": pc}, {"runner": "pooling", "override_pooler_config": pc},
                  {"task": "embed", "override_pooler_config": pc}, {"task": "embed"}):
        extra = {k: v for k, v in extra.items() if v is not None}
        try:
            return LLM(**kw, **extra)
        except TypeError as e:
            errs.append(f"{list(extra)}: {e}")
        except Exception as e:  # noqa: BLE001
            if "unexpected keyword" in str(e) or "not a valid" in str(e) or "Unrecognized" in str(e):
                errs.append(f"{list(extra)}: {e}")
                continue
            raise
    raise RuntimeError("could not start vLLM as a hidden-state (pooling) runner:\n" + "\n".join(errs))


def _mean_logprob(c):
    n = len(c.token_ids)
    if not n:
        return None
    cum = getattr(c, "cumulative_logprob", None)
    if cum is None:
        lps = getattr(c, "logprobs", None)
        if not lps:
            return None
        cum = 0.0
        for tid, d in zip(c.token_ids, lps):
            e = d.get(tid) if d else None
            if e is None:
                return None
            cum += e.logprob
    return float(cum) / n


# ==================================================================================
class MockEngine:
    """Fake model for dry runs. Output quality grows with model size and with fine-tuning; confidence and hidden states
    carry a (noisy) signal of quality so that every downstream step has something to learn from."""

    def __init__(self, params=1e9, finetuned=False, **kw):
        import math
        self.skill = min(0.92, 0.42 + 0.07 * math.log10(max(params, 1e8) / 1e8) + (0.12 if finetuned else 0.0))
        self.thinking = None

    def encode(self, system, user):
        return ((system or "") + " " + user).split()

    def encode_full(self, system, user, answer):
        return self.encode(system, user) + (answer or "").split()

    def _rng(self, p, s):
        return random.Random(s ^ zlib.crc32(" ".join(map(str, p[-80:])).encode() + str(len(p)).encode()))

    def generate(self, prompts, n, temperature, max_tokens, seeds, meta=None):
        res = []
        for p, s, m in zip(prompts, seeds, meta or [{}] * len(prompts)):
            rng = self._rng(p, s)
            outs = []
            for _ in range(n):
                t, good = self._one(rng, m or {}, temperature)
                k = max(1, len(t.split()) + rng.randint(1, 12))
                outs.append((t, k, -rng.uniform(0.02, 0.25) if good else -rng.uniform(0.15, 0.9)))
            res.append(outs)
        return res

    def _one(self, rng, m, temperature):
        kind, sk = m.get("kind"), self.skill
        if kind in ("ret", "ret_atts"):
            gt = [g + 1 for g in m["gt"]]
            sel = [g for g in gt if rng.random() < sk + 0.15]
            sel += [rng.randint(1, m["n_refs"]) for _ in range(rng.randint(0, 2 if kind == "ret_atts" else 1))]
            if not sel:
                sel = [rng.randint(1, m["n_refs"])]
            good = set(gt) <= set(sel)
            if kind == "ret":
                return "Selected references: " + " ".join(f"[{i}]" for i in sel), good
            return "Query: ...\nSelected references: " + " ".join(f"[{i}] some details" for i in sel), good
        if kind in ("qa", "qa_direct", "qa_free"):
            good = rng.random() < sk * (1.0 if m.get("has_gold", True) else 0.55)
            a = m["gold"] if good else rng.choice(["unknown", "Paris", "1999", "John Smith", "no"])
            if kind == "qa_direct":
                return f"Answer: {a}", good
            if kind == "qa_free":
                return f"Based on the references, the answer is {a}. Further context follows.", good
            return f"Reasoning...\nAnswer: {a}\nConfidence: {rng.randint(70, 99) if good else rng.randint(40, 95)}", good
        if kind == "fuse":
            resp = m.get("responses", [])
            golds = [r for r in resp if m["gold"].lower() in r.lower()]
            good = (len(golds) * 2 > len(resp)) or (golds and rng.random() < sk)
            return (f"The best supported answer is {m['gold']}." if good else "The responses disagree; most likely unknown."), bool(good)
        if kind == "atts_plan":
            cands = []
            for _ in range(m.get("batch", 10)):
                r = rng.choice(m["ret_models"])
                g = rng.choice(m["gen_models"])
                cands.append({"Retrieval": {"model": r, "samples": rng.choice([1, 2, 3, 5, 10])},
                              "Question Answering": {"model": g, "samples": rng.choice([1, 3, 5, 10, 20, 40])}})
            return json.dumps({"candidates": cands}), True
        if kind == "atts_insight":
            return "1. Prefer the largest retriever at one sample. 2. Answer samples level off around 20. 3. Balance budget.", True
        if kind == "plan":
            return json.dumps(m.get("mock_plan", [])), True
        return "ok", True

    def hidden(self, prompts, meta=None):
        out = []
        for p, m in zip(prompts, meta or [{}] * len(prompts)):
            rng = self._rng(p, 7)
            y = float(m.get("label", 0.5))
            out.append([rng.gauss(0, 1) + (1.2 * (y - 0.5) if i < 4 else 0.0) for i in range(16)])
        return out


def make_engine(cfg, model, max_model_len=None, lora_path=None, pooling=False):
    if cfg["backend"] == "mock":
        return MockEngine(params=model.get("params", 1e9), finetuned=bool(lora_path) or model.get("finetuned", False))
    mml = max_model_len or model.get("max_model_len") or cfg["gpu"]["max_model_len"]
    log(f"loading {model.get('label', model['path'])} (vLLM{', LoRA ' + lora_path if lora_path else ''}{', hidden states' if pooling else ''}; context {mml})")
    return VLLMEngine(model, mml, gpu_util=_gpu_util(cfg["gpu"]["gpu_util"]), lora_path=lora_path, pooling=pooling)


def _gpu_util(want):
    """vLLM refuses to start when less than `want` x GPU memory is free. When this process already used the GPU (training
    earlier in the same run, e.g. `--stage all`), PyTorch keeps some memory: ask for what is actually free, minus a margin."""
    import sys
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return want
    free_gpu()
    free, total = torch.cuda.mem_get_info()
    got = min(want, free / total - 0.03)
    if got < want:
        log(f"GPU memory already in use by this process: vLLM gets {got:.2f} of the GPU instead of {want:.2f}")
    return max(got, 0.3)


def free_gpu(*objs):
    try:
        import gc
        import torch
        for o in objs:
            del o
        gc.collect()
        torch.cuda.empty_cache()
        try:
            from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


def strip_tags(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
