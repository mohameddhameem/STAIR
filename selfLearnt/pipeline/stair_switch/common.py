"""Shared pieces: prompts (unchanged from the development notebooks), metrics, file helpers, BM25, model engines, voting."""
import json
import os
import random
import re
import string
import time
import zlib
from collections import Counter

import numpy as np

# ==================================================================================
# Prompts (identical to the Colab development code, so results stay comparable)
# ==================================================================================
RET_SYSTEM = "You are a helpful assistant that return the beneficial references to the query."
QA_SYSTEM = "You are a helpful assistant."
SHORT_ANSWER = ("\nEnd your response with a final line in exactly this form: 'Answer: <short answer>' - "
                "the answer only (a name, date, number or short phrase; 'yes' or 'no' for yes/no questions).")
COT_INSTRUCTION = ("\nThink step by step before answering: first write out the fact from the references that answers "
                   "the first part of the question, then use it to find the fact that answers the second part, "
                   "then state the answer.")
RET_COT_INSTRUCTION = ("Think step by step before selecting: (1) state what the first part of the query asks and which "
                       "reference number contains that fact; (2) using that fact, state what the second part needs and which "
                       "reference number contains it. Then end with one final line in exactly this form, listing only the "
                       "reference numbers you select:\nSelected references: [n] [n]")
CONF_RET = ("\nAfter your selection, add one last line in exactly this form: 'Confidence: <0-100>' - how sure you are, in percent, "
            "that the selected references contain everything needed to answer the query.")
CONF_ANS = ("\nAfter the Answer line, add one last line in exactly this form: 'Confidence: <0-100>' - how sure you are, in percent, "
            "that the answer is correct.")
DIRECT = ("Select only the references needed to answer the query. Answer with one line in exactly this form, listing only the "
          "reference numbers you select:\nSelected references: [n] [n]")


def build_query(ex):
    q = "Query: " + ex["question"] + "References: "
    for i, ref in enumerate(ex["ctxs_candidate"]):
        q += f"\n[{i + 1}] " + ref
    return q


def direct_prompt(q):
    """The fine-tuned retrievers' prompt (input of training and of every later shortlist)."""
    return ("You have been provided with a query and a set of references. "
            "Your task is to select references that comprehensively cover the information needed to answer the query. "
            f"Here is the query and references: \n {build_query(q)}.\n" + DIRECT)


def baseline_ret_prompt(q):
    """AgentTTS's pick, as run in the development sweep: step-by-step retrieval, select 3, one try (+ stated confidence)."""
    p = ("You have been provided with a query and a set of references. "
         "Your task is to select references that comprehensively cover the information needed to answer the query. "
         f"Here is the query and references: \n {build_query(q)}.\n" + RET_COT_INSTRUCTION)
    p = p[:-len(RET_COT_INSTRUCTION)] + (
        "Think step by step before selecting: (1) state what the first part of the query asks and which "
        "reference number contains that fact; (2) using that fact, state what the second part needs and which "
        "reference number contains it. Select exactly 3 references in total: those from steps 1 and 2 plus the "
        "next most useful ones. Then end with one final line in exactly this form, listing only the 3 reference "
        "numbers you select:\nSelected references: [n] [n] [n]")
    return p + CONF_RET


def target_text(q):
    return "Selected references: " + " ".join(f"[{i + 1}]" for i in sorted(q["ret_answers"]))


def qa_prompt(question, refs):
    qr = "Query: " + question + "\nReferences:\n"
    for i, r in enumerate(refs):
        qr += f"[{i + 1}] " + r + "\n"
    return (f"You will be provided with a set of references related to the query. "
            f"Your task is to answer the query using the provided references. If the references are insufficient, supplement your response with your own knowledge. "
            f"Here is the query and references: \n {qr}." + COT_INSTRUCTION + SHORT_ANSWER + CONF_ANS)


def selected_refs(ex, sel):
    refs = ex["ctxs_candidate"]
    return [refs[i - 1] for i in sel if 1 <= i <= len(refs)]


def selection_text(text):
    i = text.lower().rfind("selected references:")
    return text[i:] if i >= 0 else text


def picks_of(text, n_refs):
    out = []
    for x in re.findall(r"\[(\d+)\]", text):
        i = int(x)
        if 1 <= i <= n_refs and i not in out:
            out.append(i)
    return out


def key(sel):
    return tuple(sorted(int(i) for i in sel))


# ==================================================================================
# Metrics
# ==================================================================================
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em_f1(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    em = float(p == g)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return em, em
    pt, gt = p.split(), g.split()
    same = sum((Counter(pt) & Counter(gt)).values())
    if same == 0:
        return em, 0.0
    pr, rc = same / len(pt), same / len(gt)
    return em, 2 * pr * rc / (pr + rc)


def lenient(pred, gold):
    """Corrected label: exact match, or token F1 >= 0.5, or the gold answer contained in the answer. Yes/no must match exactly."""
    em, f1 = em_f1(pred, gold)
    if em:
        return 1.0
    p, g = normalize_answer(pred), normalize_answer(gold)
    if g in ("yes", "no", "noanswer") or p in ("yes", "no", "noanswer", ""):
        return 0.0
    return float(f1 >= 0.5 or (" " + g + " ") in (" " + p + " "))


_CONF = re.compile(r"\(?\bconfidence\b\s*(?:level|score)?\s*[:=]?\s*(\d{1,3})(?:\.\d+)?\s*%?\)?\.?", flags=re.I)


def stated_conf(text):
    hits = _CONF.findall(text or "")
    return None if not hits else min(100, int(hits[-1])) / 100.0


def answer_of(text):
    text = _CONF.sub("", text or "").rstrip()
    hits = re.findall(r"answer\s*:\s*(.+)", text, flags=re.I)
    if hits:
        a = hits[-1].strip()
    else:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        a = lines[-1] if lines else ""
    return re.sub(r"[*_`]+", "", a).strip().strip(".").strip()[:200]


def agree(a, b):
    na, nb = normalize_answer(a or ""), normalize_answer(b or "")
    if not na or not nb:
        return False
    return na == nb or em_f1(a, b)[1] >= 0.8


def vote(votes):
    """Majority by agreement; ties go to the earlier vote (the order says who is preferred)."""
    sup = [sum(agree(v["prediction"], w["prediction"]) or i == j for j, w in enumerate(votes)) for i, v in enumerate(votes)]
    best = max(range(len(votes)), key=lambda i: (sup[i], -i))
    return votes[best], sup[best]


def ensemble(sels):
    """Shortlists from several retrievers: all agree -> it; a strict majority -> it; else the union."""
    sels = [key(s) for s in sels]
    if len(set(sels)) == 1:
        return sels[0]
    cnt = Counter(sels)
    best = max(cnt.values())
    tops = [s for s, c in cnt.items() if c == best]
    if best >= 2 and len(tops) == 1:
        return tops[0]
    return key(set().union(*sels))


# ==================================================================================
# Files
# ==================================================================================
def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def lp_(x):
    return -1.0 if x is None else float(x)


# ==================================================================================
# BM25 (distractor passages)
# ==================================================================================
class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        from scipy import sparse
        from sklearn.feature_extraction.text import CountVectorizer
        self.vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
        tf = self.vec.fit_transform(docs).tocsc()
        n, dl = tf.shape[0], np.asarray(tf.sum(axis=1)).ravel()
        df = np.diff(tf.indptr)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean())) * idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()

    def search(self, query, size):
        q = self.vec.transform([query])
        q.data[:] = 1.0
        s = (q @ self.W).toarray().ravel()
        size = min(size, len(s) - 1)
        top = np.argpartition(-s, size)[:size]
        return top[np.argsort(-s[top])]


# ==================================================================================
# Engines: vLLM (with token log-probabilities, tensor parallel) or a mock for dry runs
# ==================================================================================
class VLLMEngine:
    def __init__(self, path, max_model_len=8192, gpu_util=0.88, tensor_parallel=1, seed=0, quantization=None):
        from vllm import LLM
        kw = dict(model=path, max_model_len=max_model_len, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, seed=seed,
                  tensor_parallel_size=tensor_parallel)
        if quantization or "awq" in path.lower():
            kw["quantization"] = quantization or "awq"
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.thinking = None          # False for Qwen3 / Qwen3.5 models (no thinking block)

    def encode(self, system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"enable_thinking": self.thinking} if self.thinking is not None else {}
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompts, n, temperature, max_tokens, seeds, meta=None):
        from vllm import SamplingParams
        sps = [SamplingParams(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, max_tokens=max_tokens, seed=s, logprobs=1,
                              repetition_penalty=1.0) for s in seeds]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps, use_tqdm=False)
        return [[(c.text, len(c.token_ids), _mean_logprob(c)) for c in o.outputs] for o in outs]


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


class MockEngine:
    """Fake model for dry runs: plausible outputs (some right, some wrong), token log-probabilities and stated confidence."""

    def __init__(self, skill=0.6, **kw):
        self.skill, self.thinking = skill, None

    def encode(self, system, user):
        return (system + " " + user).split()

    def generate(self, prompts, n, temperature, max_tokens, seeds, meta=None):
        res = []
        for p, s, m in zip(prompts, seeds, meta or [{}] * len(prompts)):
            rng = random.Random(s ^ zlib.crc32(" ".join(map(str, p)).encode()))       # varies by prompt, reproducible
            outs = []
            for _ in range(n):
                kind = m.get("kind")
                if kind == "ret":
                    sel = [g + 1 for g in m["gt"] if rng.random() < self.skill + 0.25]
                    sel += [rng.randint(1, m["n_refs"]) for _ in range(rng.randint(0, 1))]
                    t = "Selected references: " + " ".join(f"[{i}]" for i in sel)
                    good = len(sel) >= 2
                elif kind == "qa":
                    good = rng.random() < self.skill + 0.2
                    t = "Reasoning...\nAnswer: " + (m["gold"] if good else rng.choice(["unknown", "Paris", "1999", "John Smith"]))
                    t += f"\nConfidence: {rng.randint(70, 99) if good else rng.randint(40, 95)}"
                elif kind == "plan":
                    t = json.dumps([{"action": "param", "name": "votes", "value": 3, "why": "mock"}])
                    good = True
                else:
                    t, good = "ok", True
                outs.append((t, max(1, len(t.split()) + rng.randint(3, 30)), -rng.uniform(0.02, 0.3) if good else -rng.uniform(0.1, 0.9)))
            res.append(outs)
        return res


def make_engine(cfg, model, **kw):
    """model: a model entry of the config (path, tensor_parallel, ...)."""
    if cfg["backend"] == "mock":
        return MockEngine(skill=0.7 if model.get("params", 1e9) > 2e10 else 0.55)
    e = VLLMEngine(model["path"], max_model_len=kw.get("max_model_len", cfg["gpu"]["max_model_len"]), gpu_util=cfg["gpu"]["gpu_util"],
                   tensor_parallel=model.get("tensor_parallel", 1), quantization=model.get("quantization"))
    if model.get("no_thinking"):
        e.thinking = False
    return e


def free_gpu():
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
