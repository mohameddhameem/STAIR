#!/usr/bin/env python
"""
ssw.py - Small/Big model Switching on HotpotQA (Qwen2.5-3B vs Qwen2.5-7B).

One module, used by four notebooks. Every GPU stage runs as its own process
(`python ssw.py <command> ...`) so GPU memory is fully released between models.

  build_data   CPU  random dev/test samples, N candidate paragraphs (gold + BM25 distractors)
  retrieve     GPU  Step 0: retrieval samples for every question with one model (S or L)
  answer       GPU  Step 0: answer samples + fusion with one model, on the selections of both retrievers
  table        CPU  Step 0: per-question x configuration table, costs, signals, go/no-go check
  agenttts     GPU  notebook 05: AgentTTS search (repo prompts) on the dev table -> one global config
  switching    GPU  notebooks 06/07: cascade, classifier and natural-language rule routers -> config per question
  speedcheck   GPU  live, one-question-at-a-time latency of chosen policies

`--backend mock` replaces every model with a fast fake, so the whole flow can be tested on CPU.
Prompts follow code_others/archon_hotpotqa.py and AgentTTS.py from the AgentTTS repo; the deviations
(short-answer line, fuser sees the question instead of all paragraphs) are marked "DEVIATION".
"""
import argparse
import itertools
import json
import math
import os
import random
import re
import string
import subprocess
import sys
import threading
import time
import zlib
from collections import Counter

import numpy as np

# ==================================================================================
# Models and search space
# ==================================================================================
DEFAULT_MODELS = {
    "S": {"id": "Qwen/Qwen2.5-3B-Instruct", "dir": "Qwen2.5-3B-Instruct", "params": 3.09e9, "nominal": 3e9},
    "L": {"id": "Qwen/Qwen2.5-7B-Instruct", "dir": "Qwen2.5-7B-Instruct", "params": 7.62e9, "nominal": 7e9},
}
RET_KS = [1, 3]      # retrieval samples scored
ANS_KS = [1, 5]      # answer samples scored
PROBE = "S1-S5"      # cheap first pass used by the "after a probe" routers


def cfg_key(r, rk, a, ak):
    return f"{r}{rk}-{a}{ak}"


def parse_cfg(key):
    """'S1-L5' -> ('S', 1, 'L', 5); a 'c' after the answer model marks chain-of-thought answers ('L1-Lc5')."""
    m = re.fullmatch(r"([SL])(\d+)-([SL]c?)(\d+)", key)
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


def base(model_key):
    """'Lc' -> 'L': the model behind a (possibly chain-of-thought) answer key."""
    return model_key[0]


def is_cot(model_key):
    return model_key.endswith("c")


ALL_CFGS = [cfg_key(r, rk, a, ak) for r in "SL" for rk in RET_KS for a in "SL" for ak in ANS_KS]


def describe_cfg(key, models=DEFAULT_MODELS):
    r, rk, a, ak = parse_cfg(key)
    size = lambda k: f"{models[base(k)]['nominal'] / 1e9:.0f}B"
    return f"retrieval {size(r)} x{rk}, answer {size(a)}{' CoT' if is_cot(a) else ''} x{ak}"


# ==================================================================================
# Prompts (repo wording; DEVIATIONS marked)
# ==================================================================================
RET_SYSTEM = "You are a helpful assistant that return the beneficial references to the query."
QA_SYSTEM = "You are a helpful assistant."
FUSER_SYSTEM = "You are a helpful assistant who fuses multiple answers"
SHORT_ANSWER = ("\nEnd your response with a final line in exactly this form: 'Answer: <short answer>' - "
                "the answer only (a name, date, number or short phrase; 'yes' or 'no' for yes/no questions).")  # DEVIATION


def build_query(ex):
    q = "Query: " + ex["question"] + "References: "
    for i, ref in enumerate(ex["ctxs_candidate"]):
        q += f"\n[{i + 1}] " + ref
    return q


def retrieval_prompt(ex, cot=False):
    if cot:  # SET 3 ADDITION: chain-of-thought retrieval (not in the paper); same references, same task
        return (f"You have been provided with a query and a set of references. "
                f"Your task is to select references that comprehensively cover the information needed to answer the query. "
                f"Here is the query and references: \n {build_query(ex)}.\n" + RET_COT_INSTRUCTION)
    return (f"You have been provided with a query and a set of references. "
            f"Your task is to select references that comprehensively cover the information needed to answer the query. "
            f"Your response should include the original query and a list of selected references in the following format: 'Query: [query]\nSelected references: [reference number] [reference details]'."
            f"Here is the query and references: \n {build_query(ex)}.")


RET_COT_INSTRUCTION = ("Think step by step before selecting: (1) state what the first part of the query asks and which "
                       "reference number contains that fact; (2) using that fact, state what the second part needs and which "
                       "reference number contains it. Then end with one final line in exactly this form, listing only the "
                       "reference numbers you select:\nSelected references: [n] [n]")


def selection_text(text):
    """For chain-of-thought retrieval: only the part after the LAST 'Selected references:' counts,
    so reference numbers mentioned while reasoning are not counted as selections."""
    i = text.lower().rfind("selected references:")
    return text[i:] if i >= 0 else text


COT_INSTRUCTION = ("\nThink step by step before answering: first write out the fact from the references that answers "
                   "the first part of the question, then use it to find the fact that answers the second part, "
                   "then state the answer.")  # SET 2b ADDITION: chain-of-thought (not in the paper)


def qa_prompt(question, refs, cot=False):
    qr = "Query: " + question + "\nReferences:\n"
    for i, r in enumerate(refs):
        qr += f"[{i + 1}] " + r + "\n"
    return (f"You will be provided with a set of references related to the query. "
            f"Your task is to answer the query using the provided references. If the references are insufficient, supplement your response with your own knowledge. "
            f"Here is the query and references: \n {qr}." + (COT_INSTRUCTION if cot else "") + SHORT_ANSWER)


def fuser_prompt(question, drafts):
    query = "Query: " + question  # DEVIATION: the repo passes the question + all candidate paragraphs here
    pad = " " * 12
    p = (f"You have been provided with a set of responses from various open-source models to the latest user query, which is {query}."
         f"{pad}Your task is to synthesize these responses into a single, high-quality response. "
         f"{pad}It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. "
         f"{pad}Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. "
         f"{pad}Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n")
    p += f"Once again, the query is: {query}\n"
    p += "Responses from models:"
    for i, d in enumerate(drafts):
        p += f"\n{i + 1}. {d}"
    return p + SHORT_ANSWER


def merge(candidates):
    """Repo Merge component: keep the most common list length, then the most frequent ids (1-based)."""
    ids, lengths = [], []
    for c in candidates:
        nums = re.findall(r"\[(\d+)\]", c)
        ids.extend(nums)
        lengths.append(len(nums))
    if ids:
        n = Counter(lengths).most_common()[0][0]
        sel = [e[0] for e in Counter(ids).most_common()[0:n]]
    else:
        sel = ["1"]
    return tuple(sorted({int(e) for e in sel}))


def selected_refs(ex, sel):
    refs = ex["ctxs_candidate"]
    return [refs[i - 1] for i in sel if 1 <= i <= len(refs)]


# ==================================================================================
# Metrics (official HotpotQA EM/F1 on the extracted short answer)
# ==================================================================================
def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def extract_answer(text):
    hits = re.findall(r"answer\s*:\s*(.+)", text, flags=re.I)
    if hits:
        a = hits[-1].strip()
    else:
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        a = lines[-1] if lines else ""
    a = re.sub(r"[*_`]+", "", a).strip().strip(".").strip()
    return a[:200]


def em_f1(pred, gold):
    p, g = normalize_answer(pred), normalize_answer(gold)
    em = float(p == g)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        return em, em
    pt, gt = p.split(), g.split()
    common = Counter(pt) & Counter(gt)
    same = sum(common.values())
    if same == 0:
        return em, 0.0
    pr, rc = same / len(pt), same / len(gt)
    return em, 2 * pr * rc / (pr + rc)


def cover_em(text, gold):
    g, t = normalize_answer(gold), normalize_answer(text)
    return float(bool(g) and re.search(r"(^| )" + re.escape(g) + r"( |$)", t) is not None)


def ret_f1(gt, pred):
    gt, pred = set(gt), set(pred)
    inter = len(gt & pred)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gt) if gt else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def compute_budget(S, M_large, Np1, Nd1, M_small=3e9, Np2=128, Nd2=64):
    """AgentTTS Eq. 4, same code as the repo."""
    alpha, beta1, beta2, beta3 = M_large / M_small, Np1 / Nd1, Np2 / Nd2, Np1 / Np2
    return beta2 * ((alpha * beta3 / beta1) * (beta1 + S) - 1)


# ==================================================================================
# Small utilities
# ==================================================================================
def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)


def gpu_name():
    try:
        out = subprocess.run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null",
                             shell=True, capture_output=True, text=True).stdout.strip()
        return out.splitlines()[0] if out else "none"
    except Exception:
        return "none"


class Status:
    """Writes status.json (stage, done/total, elapsed, ETA) so progress can be followed from Drive."""

    def __init__(self, run_dir, stage, total, time_limit_min=None):
        self.path, self.stage, self.total = os.path.join(run_dir, "status.json"), stage, total
        self.t0, self.limit = time.time(), (time_limit_min * 60 if time_limit_min else None)
        self.update(0)

    def update(self, done, note=""):
        el = time.time() - self.t0
        eta = el / done * (self.total - done) if done else None
        write_json(self.path, {"stage": self.stage, "done": done, "total": self.total, "elapsed_s": round(el),
                               "eta_s": round(eta) if eta is not None else None, "note": note,
                               "updated": time.strftime("%Y-%m-%d %H:%M:%S")})

    def over_limit(self):
        return self.limit is not None and time.time() - self.t0 > self.limit


class GPUMonitor:
    def __init__(self):
        self.mem, self.util, self._stop = [], [], threading.Event()
        self.ok = subprocess.run("which nvidia-smi", shell=True, capture_output=True).returncode == 0

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                m, u = out.strip().splitlines()[0].split(",")
                self.mem.append(float(m))
                self.util.append(float(u))
            except Exception:
                pass
            self._stop.wait(1.0)

    def __enter__(self):
        if self.ok:
            threading.Thread(target=self._loop, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._stop.set()

    def summary(self):
        return {"peak_vram_mb": max(self.mem) if self.mem else None,
                "mean_gpu_util_pct": float(np.mean(self.util)) if self.util else None}


# ==================================================================================
# LLM engines (vLLM or mock)
# ==================================================================================
class VLLMEngine:
    def __init__(self, path, max_model_len=8192, gpu_util=0.85, seed=0, thinking=None):
        from vllm import LLM
        cap = gpu_name().split(",")[-1].strip()
        dtype = "half" if cap and cap[0].isdigit() and float(cap) < 8.0 else "auto"
        self.llm = LLM(model=path, max_model_len=max_model_len, dtype=dtype, gpu_memory_utilization=gpu_util,
                       enable_prefix_caching=True, seed=seed)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.thinking = thinking

    def encode(self, system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kw = {"enable_thinking": self.thinking} if self.thinking is not None else {}
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompts, n, temperature, max_tokens, seeds, meta=None):
        from vllm import SamplingParams
        sps = [SamplingParams(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                              max_tokens=max_tokens, seed=s) for s in seeds]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps, use_tqdm=False)
        return [[(c.text, len(c.token_ids)) for c in o.outputs] for o in outs]


class MockEngine:
    """Fake model: produces plausible outputs from `meta` so every stage and file can be tested on CPU."""

    def __init__(self, size="S", **kw):
        self.size, self.max_model_len, self.thinking = size, 10 ** 9, None
        self.skill = 0.62 if size == "L" else 0.45

    def encode(self, system, user):
        return (system + " " + user).split()

    def generate(self, prompts, n, temperature, max_tokens, seeds, meta=None):
        res = []
        for p, s, m in zip(prompts, seeds, meta or [{}] * len(prompts)):
            rng = random.Random(s)
            outs = []
            for _ in range(n):
                kind = m.get("kind")
                if kind == "ret":
                    sel = [g + 1 for g in m["gt"] if rng.random() < self.skill + 0.2]
                    sel += [rng.randint(1, m["n_refs"]) for _ in range(rng.randint(0, 2))]
                    t = f"Query: {m['q']}\nSelected references: " + " ".join(f"[{i}] ..." for i in sel)
                elif kind in ("qa", "fuse"):
                    ok = rng.random() < min(0.95, self.skill * 0.6 + 0.5 * m.get("hit", 0.5) + (0.08 if m.get("cot") else 0))
                    t = "Reasoning...\nAnswer: " + (m["gold"] if ok else rng.choice(["unknown", "Paris", "1999", "John Smith"]))
                elif kind == "agent_json":
                    cands = [{"Retrieval": {"model": rng.choice(m["choices"][0]), "samples": rng.choice([1, 2, 3, 4])},
                              "Question Answering": {"model": rng.choice(m["choices"][1]), "samples": rng.choice([1, 3, 5, 6])}}
                             for _ in range(m.get("k", 2))]
                    t = "<think>mock reasoning</think>" + json.dumps({"candidates": cands})
                elif kind == "agent_text":
                    t = "<think>mock reasoning</think>1. Prefer the larger retriever.\n2. QA gains level off after 5 samples."
                elif kind == "rules" and m.get("v2"):
                    acts = m["actions"]
                    t = (f"<rules>\nIF agree < 0.6 THEN {acts[-1]}\nIF compare = yes AND words > 12 THEN {acts[1]}\n"
                         f"IF yesno = yes THEN {acts[0]}\nOTHERWISE {acts[0]}\n</rules>")
                elif kind == "rules":
                    acts = m["actions"]
                    t = ("<rules>\n1. IF the question compares two things (which/who ... or) THEN use " + acts[-1] +
                         "\n2. IF the answer agreement is below 0.6 THEN use " + acts[-1] +
                         "\n3. OTHERWISE use " + acts[0] + "\n</rules>")
                elif kind == "route":
                    acts = m["actions"]
                    t = rng.choice(acts)
                else:
                    t = "ok"
                outs.append((t, max(1, len(t.split()) + rng.randint(3, 30))))
            res.append(outs)
        time.sleep(0.0005 * len(prompts))
        return res


def make_engine(backend, path, size="S", **kw):
    if backend == "mock":
        return MockEngine(size=size)
    return VLLMEngine(path, **kw)


def split_thinking(text):
    m = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    if m:
        return m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    if "</think>" in text:
        a, b = text.split("</think>", 1)
        return a.replace("<think>", "").strip(), b.strip()
    return "", text.strip()


# ==================================================================================
# Data
# ==================================================================================
_TOKEN = r"(?u)\b\w\w+\b"


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        from scipy import sparse
        from sklearn.feature_extraction.text import CountVectorizer
        self.vec = CountVectorizer(lowercase=True, token_pattern=_TOKEN, dtype=np.float32)
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
        top = np.argpartition(-s, size)[:size]
        return top[np.argsort(-s[top])]


def cmd_build_data(args):
    with open(args.hotpot_dev, encoding="utf-8") as f:
        data = json.load(f)
    seen, corpus = set(), []
    for item in data:
        for _, sents in item["context"]:
            t = " ".join(sents)
            if t not in seen:
                seen.add(t)
                corpus.append(t)
    log(f"HotpotQA dev: {len(data)} questions, {len(corpus)} paragraphs; building BM25 ...")
    bm25 = BM25(corpus)
    rng = random.Random(args.seed)
    picked = rng.sample(range(len(data)), args.dev_n + args.test_n)
    splits = {"dev": picked[:args.dev_n], "test": picked[args.dev_n:]}
    os.makedirs(args.out_dir, exist_ok=True)
    for split, idxs in splits.items():
        out = []
        for i in idxs:
            ex = data[i]
            ctx = {t: " ".join(s) for t, s in ex["context"]}
            gold = []
            for title, _ in ex["supporting_facts"]:
                if ctx[title] not in gold:
                    gold.append(ctx[title])
            cands = gold.copy()
            for j in bm25.search(ex["question"], args.n_candidates + 10):
                if len(cands) >= args.n_candidates:
                    break
                if corpus[j] not in cands:
                    cands.append(corpus[j])
            rng.shuffle(cands)
            out.append({"qid": ex["_id"], "question": ex["question"], "answer": ex["answer"], "type": ex.get("type"),
                        "ret_answers": [cands.index(g) for g in gold], "ctxs_candidate": cands})
        path = os.path.join(args.out_dir, f"{split}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        toks = np.mean([sum(len(c) for c in e["ctxs_candidate"]) / 4 for e in out])
        log(f"wrote {path}: {len(out)} questions x {args.n_candidates} paragraphs (~{toks:.0f} tokens of paragraphs each)")
    write_json(os.path.join(args.out_dir, "data_config.json"), vars(args))


def load_questions(data_dir, splits=("dev", "test")):
    qs = []
    for sp in splits:
        with open(os.path.join(data_dir, f"{sp}.json"), encoding="utf-8") as f:
            for i, ex in enumerate(json.load(f)):
                ex["split"], ex["i"] = sp, i
                qs.append(ex)
    return qs


# ==================================================================================
# Step 0: retrieval
# ==================================================================================
def cmd_retrieve(args):
    models = json.load(open(args.models_json))
    m = models[args.model_key]
    run = args.run_dir
    os.makedirs(os.path.join(run, "step0"), exist_ok=True)
    cot = getattr(args, "cot", False)
    tag = args.model_key + ("c" if cot else "")        # retrieval key: S, L, Sc or Lc
    max_tok = 1024 if cot else 512
    out_path = os.path.join(run, "step0", f"retrieval_{tag}.jsonl")
    qs = load_questions(args.data_dir)
    done = {r["qid"] for r in read_jsonl(out_path)}
    todo = [q for q in qs if q["qid"] not in done]
    log(f"retrieve{' (chain-of-thought)' if cot else ''} | {m['id']} | {len(qs)} questions ({len(done)} already done) | {max(RET_KS)} samples")
    st = Status(run, f"retrieve_{tag}", len(qs), args.time_limit_min)
    t0 = time.time()
    flops = 0.0
    with GPUMonitor() as mon:
        eng = make_engine(args.backend, m["path"], size=args.model_key, max_model_len=args.max_model_len,
                          gpu_util=args.gpu_util, seed=args.seed)
        load_s = time.time() - t0
        t1 = time.time()
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            prompts = [eng.encode(RET_SYSTEM, retrieval_prompt(q, cot=cot)) for q in chunk]
            meta = [{"kind": "ret", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"]), "q": q["question"]} for q in chunk]
            outs = eng.generate(prompts, max(RET_KS), 0.9, max_tok, [args.seed * 100003 + zlib.crc32(q["qid"].encode()) % 99991 for q in chunk], meta)
            rows = []
            for q, p, o in zip(chunk, prompts, outs):
                rows.append({"qid": q["qid"], "split": q["split"], "model": tag, "prompt_tokens": len(p),
                             "samples": [{"text": t, "out_tokens": k, **({"sel_text": selection_text(t),
                                                                           "has_final_line": "selected references:" in t.lower()} if cot else {})}
                                         for t, k in o]})
                flops += 2 * m["params"] * (len(p) + sum(k for _, k in o))
            append_jsonl(out_path, rows)
            st.update(len(done) + c + len(chunk))
            log(f"  {len(done) + c + len(chunk)}/{len(qs)}")
            if st.over_limit():
                log("time limit reached - stopping cleanly (rerun to resume)")
                break
        gen_s = time.time() - t1
    write_json(os.path.join(run, "stages", f"retrieve_{tag}_{int(time.time())}.json"),
               {"stage": "retrieve", "model": tag, "gpu": gpu_name(), "load_s": load_s, "gen_s": gen_s,
                "flops": flops, **mon.summary()})
    log(f"retrieve done: {gen_s:.0f}s generation + {load_s:.0f}s load")


# ==================================================================================
# Step 0: answer + fuse (one model, selections of both retrievers)
# ==================================================================================
def cmd_answer(args):
    models = json.load(open(args.models_json))
    m = models[args.model_key]
    run = args.run_dir
    cot = getattr(args, "cot", False)
    tag = args.model_key + ("c" if cot else "")          # answer key written into the table: S, L, Sc or Lc
    gold = getattr(args, "gold", False)                  # SET 3: answer from the two gold paragraphs (no retrieval)
    ret_ks = getattr(args, "ret_ks", None) or ([1] if (cot or gold) else RET_KS)   # CoT is added on the 1-sample retrievals only
    if gold:
        ret_ks = [1]
    suffix = getattr(args, "out_suffix", "") or ""
    max_tok = 512 if cot else 384
    qs = {q["qid"]: q for q in load_questions(args.data_dir)}
    rets = {}
    if gold:
        for qid in qs:
            rets[(qid, "G")] = {"prompt_tokens": 0, "samples": []}
    else:
        for rk_model in (getattr(args, "ret_tags", None) or ["S", "L"]):
            for r in read_jsonl(os.path.join(run, "step0", f"retrieval_{rk_model}.jsonl")):
                rets[(r["qid"], rk_model)] = r
    ret_models = sorted({k[1] for k in rets})
    if not ret_models:
        raise SystemExit("no retrieval files found for " + str(getattr(args, "ret_tags", None)))
    out_path = os.path.join(run, "step0", f"answers_{tag}{suffix}.jsonl")
    samp_path = os.path.join(run, "step0", f"answer_samples_{tag}{suffix}.jsonl")
    n_rows = len(ret_models) * len(ret_ks) * len(ANS_KS)
    per_q = Counter(r["qid"] for r in read_jsonl(out_path))
    todo = [qid for qid in qs if per_q.get(qid, 0) < n_rows and all((qid, rm) in rets for rm in ret_models)]
    if os.path.exists(out_path):  # drop partial rows so unfinished questions are recomputed cleanly
        keep = [r for r in read_jsonl(out_path) if per_q[r["qid"]] >= n_rows]
        open(out_path, "w").close()
        append_jsonl(out_path, keep)
    log(f"answer{' (chain-of-thought)' if cot else ''} | {m['id']} | retrievers {ret_models} | {len(qs)} questions "
        f"({len(qs) - len(todo)} done) | {max(ANS_KS)} answer samples + fusion")
    st = Status(run, f"answer_{tag}{suffix}", len(qs), args.time_limit_min)
    t0 = time.time()
    flops_ans = flops_fuse = 0.0
    with GPUMonitor() as mon:
        eng = make_engine(args.backend, m["path"], size=args.model_key, max_model_len=args.max_model_len,
                          gpu_util=args.gpu_util, seed=args.seed)
        load_s = time.time() - t0
        t1 = time.time()
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            plan, uniq, a_prompts, a_meta, a_keys = [], {}, [], [], []
            for qid in chunk:
                q = qs[qid]
                for rm in ret_models:
                    texts = [s.get("sel_text", s["text"]) for s in rets[(qid, rm)]["samples"]]
                    for rk in ret_ks:
                        sel = tuple(sorted(i + 1 for i in q["ret_answers"])) if rm == "G" else merge(texts[:rk])
                        plan.append((qid, rm, rk, sel))
                        key = (qid, sel)
                        if key not in uniq:
                            uniq[key] = len(a_prompts)
                            a_prompts.append(eng.encode(QA_SYSTEM, qa_prompt(q["question"], selected_refs(q, sel), cot=cot)))
                            hit = len({i - 1 for i in sel} & set(q["ret_answers"])) / max(1, len(q["ret_answers"]))
                            a_meta.append({"kind": "qa", "gold": q["answer"], "hit": hit, "cot": cot})
                            a_keys.append(key)
            a_outs = eng.generate(a_prompts, max(ANS_KS), 0.7, max_tok,
                                  [args.seed * 7919 + zlib.crc32(repr(k).encode()) % 99991 for k in a_keys], a_meta)
            srows = []
            for key, p, o in zip(a_keys, a_prompts, a_outs):
                srows.append({"qid": key[0], "sel": list(key[1]), "model": tag, "prompt_tokens": len(p),
                              "samples": [{"text": t, "out_tokens": k, "short": extract_answer(t)} for t, k in o]})
                flops_ans += 2 * m["params"] * (len(p) + sum(k for _, k in o))
            append_jsonl(samp_path, srows)
            f_index, f_prompts, f_meta = {}, [], []
            for qid, rm, rk, sel in plan:
                q = qs[qid]
                outs = a_outs[uniq[(qid, sel)]]
                for ak in ANS_KS:
                    fk = (qid, sel, ak)
                    if fk in f_index:
                        continue
                    drafts = [t for t, _ in outs[:ak]]
                    f_index[fk] = len(f_prompts)
                    f_prompts.append(eng.encode(FUSER_SYSTEM, fuser_prompt(q["question"], drafts)))
                    hit = np.mean([em_f1(extract_answer(t), q["answer"])[0] for t in drafts])
                    f_meta.append({"kind": "fuse", "gold": q["answer"], "hit": float(hit), "cot": cot})
            f_outs = eng.generate(f_prompts, 1, 0.3, 384, [args.seed * 104729 + i for i in range(len(f_prompts))], f_meta)
            for p, o in zip(f_prompts, f_outs):
                flops_fuse += 2 * m["params"] * (len(p) + o[0][1])
            rows = []
            for qid, rm, rk, sel in plan:
                q = qs[qid]
                a_i = uniq[(qid, sel)]
                outs = a_outs[a_i]
                ret = rets[(qid, rm)]
                for ak in ANS_KS:
                    j = f_index[(qid, sel, ak)]
                    text, f_tok = f_outs[j][0]
                    short = extract_answer(text)
                    em, f1 = em_f1(short, q["answer"])
                    rows.append({
                        "qid": qid, "split": q["split"], "config": cfg_key(rm, rk, tag, ak),
                        "ret_model": rm, "ret_k": rk, "ans_model": tag, "ans_k": ak,
                        "selected": [i - 1 for i in sel], "gold_ids": q["ret_answers"],
                        "ret_prompt_tokens": ret["prompt_tokens"],
                        "ret_output_tokens": sum(s["out_tokens"] for s in ret["samples"][:rk]),
                        "ans_prompt_tokens": len(a_prompts[a_i]), "ans_output_tokens": sum(k for _, k in outs[:ak]),
                        "fuse_prompt_tokens": len(f_prompts[j]), "fuse_output_tokens": f_tok,
                        "prediction": short, "gold": q["answer"], "EM": em, "F1": f1, "cover_EM": cover_em(text, q["answer"]),
                        "Ret_F1": ret_f1(q["ret_answers"], [i - 1 for i in sel]),
                    })
            append_jsonl(out_path, rows)
            done_n = len(qs) - len(todo) + c + len(chunk)
            st.update(done_n)
            log(f"  {done_n}/{len(qs)} | answer prompts {len(a_prompts)}, fuser prompts {len(f_prompts)}")
            if st.over_limit():
                log("time limit reached - stopping cleanly (rerun to resume)")
                break
        gen_s = time.time() - t1
    write_json(os.path.join(run, "stages", f"answer_{tag}{suffix}_{int(time.time())}.json"),
               {"stage": "answer", "model": tag, "suffix": suffix, "ret_models": ret_models, "gpu": gpu_name(), "load_s": load_s, "gen_s": gen_s,
                "flops": flops_ans + flops_fuse, "flops_answer": flops_ans, "flops_fuse": flops_fuse, **mon.summary()})
    log(f"answer done: {gen_s:.0f}s generation + {load_s:.0f}s load")


# ==================================================================================
# Step 0: table, signals, go/no-go
# ==================================================================================
YESNO_START = ("is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ", "has ", "have ", "had ", "will ", "would ", "should ")
COMPARE_WORDS = ("which", "first", "older", "younger", "earlier", "later", "more", "less", "larger", "smaller",
                 "longer", "shorter", "higher", "lower", "both", "same", "either")


def question_features(q):
    t = q.lower()
    words = q.split()
    return {
        "f_words": len(words),
        "f_yesno": float(t.startswith(YESNO_START)),
        "f_compare": float(any(re.search(rf"\b{w}\b", t) for w in COMPARE_WORDS) or " or " in t),
        "f_digit": float(bool(re.search(r"\d", q))),
        "f_caps": float(sum(1 for w in words[1:] if w[:1].isupper())),
        "f_quote": float('"' in q),
    }


def cost_columns(df, models):
    P = {k: v["params"] for k, v in models.items()}
    df["flops_ret"] = [2 * P[base(m)] * (a + b) for m, a, b in zip(df.ret_model, df.ret_prompt_tokens, df.ret_output_tokens)]
    df["flops_ans"] = [2 * P[base(m)] * (a + b) for m, a, b in zip(df.ans_model, df.ans_prompt_tokens, df.ans_output_tokens)]
    df["flops_fuse"] = [2 * P[base(m)] * (a + b) for m, a, b in zip(df.ans_model, df.fuse_prompt_tokens, df.fuse_output_tokens)]
    df["flops"] = df.flops_ret + df.flops_ans + df.flops_fuse
    return df


def cmd_table(args):
    import pandas as pd
    models = json.load(open(args.models_json))
    run = args.run_dir
    rows = []
    for a in ("S", "L", "Sc", "Lc"):
        rows += read_jsonl(os.path.join(run, "step0", f"answers_{a}.jsonl"))
    if not rows:
        raise SystemExit("no answer rows - run retrieve and answer first")
    df = cost_columns(pd.DataFrame(rows), models)
    # measured seconds per FLOP per (stage, model) -> estimated GPU seconds
    spf = {}
    for fn in os.listdir(os.path.join(run, "stages")):
        s = json.load(open(os.path.join(run, "stages", fn)))
        if s.get("stage") in ("retrieve", "answer"):
            a = spf.setdefault((s["stage"], s["model"]), [0.0, 0.0, s.get("gpu"), 0.0])
            a[0] += s["gen_s"]
            a[1] += s["flops"]
            a[3] = max(a[3], s.get("peak_vram_mb") or 0)
    sec = lambda st, mk: spf[(st, mk)][0] / spf[(st, mk)][1] if (st, mk) in spf and spf[(st, mk)][1] else np.nan
    df["est_gpu_s"] = [fr * sec("retrieve", rm) + (fa + ff) * sec("answer", am)
                       for fr, fa, ff, rm, am in zip(df.flops_ret, df.flops_ans, df.flops_fuse, df.ret_model, df.ans_model)]
    # Eq. 4 budget with measured average lengths (what the AgentTTS agent reasons with)
    # (retrieval lengths from all rows; answer lengths separately for plain and CoT answers, so adding CoT rows
    #  leaves the budgets of the plain setups exactly as they were)
    cot_rows = df.ans_model.str.endswith("c")
    plain = df[~cot_rows]
    Np_r, Nd_r = plain.ret_prompt_tokens.mean(), (plain.ret_output_tokens / plain.ret_k).mean()
    Np_a, Nd_a = plain.ans_prompt_tokens.mean(), (plain.ans_output_tokens / plain.ans_k).mean()
    lengths = {"Retrieval": [round(Np_r), max(1, round(Nd_r))], "Question Answering": [round(Np_a), max(1, round(Nd_a))]}
    if cot_rows.any():
        c = df[cot_rows]
        lengths["Question Answering (CoT)"] = [round(c.ans_prompt_tokens.mean()), max(1, round((c.ans_output_tokens / c.ans_k).mean()))]
    qa_len = lambda am: lengths["Question Answering (CoT)" if is_cot(am) else "Question Answering"]
    df["budget_eq4"] = [compute_budget(rk, models[rm]["nominal"], *lengths["Retrieval"]) +
                        compute_budget(ak, models[base(am)]["nominal"], *qa_len(am))
                        for rm, rk, am, ak in zip(df.ret_model, df.ret_k, df.ans_model, df.ans_k)]
    # signals from the cheap probe (S retrieval x3, S answers x5 on the 1-sample S selection)
    qs = {q["qid"]: q for q in load_questions(args.data_dir)}
    ret_s = {r["qid"]: r for r in read_jsonl(os.path.join(run, "step0", "retrieval_S.jsonl"))}
    samp_s = {}
    for r in read_jsonl(os.path.join(run, "step0", "answer_samples_S.jsonl")):
        samp_s[(r["qid"], tuple(r["sel"]))] = r
    sig = []
    for qid, q in qs.items():
        feats = {"qid": qid, "split": q["split"], "question": q["question"], "answer": q["answer"], "type": q.get("type"),
                 **question_features(q["question"])}
        if qid in ret_s:
            sets = [set(int(x) for x in re.findall(r"\[(\d+)\]", s["text"])) for s in ret_s[qid]["samples"]]
            pairs = list(itertools.combinations(sets, 2))
            feats["s_ret_agree"] = float(np.mean([len(a & b) / len(a | b) if a | b else 1.0 for a, b in pairs])) if pairs else 1.0
            sel1 = merge([ret_s[qid]["samples"][0]["text"]])
            sm = samp_s.get((qid, sel1))
            if sm:
                shorts = [normalize_answer(s["short"]) for s in sm["samples"]]
                top, cnt = Counter(shorts).most_common(1)[0]
                feats["s_ans_agree"] = cnt / len(shorts)
                feats["s_ans_unknown"] = float(any(w in top for w in ("unknown", "not ", "cannot", "no information", "unclear")))
        sig.append(feats)
    sig = pd.DataFrame(sig)
    out = os.path.join(run, "table")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "per_question.csv"), index=False)
    sig.to_csv(os.path.join(out, "signals.csv"), index=False)
    agg = df.groupby(["split", "config"]).agg(EM=("EM", "mean"), F1=("F1", "mean"), cover_EM=("cover_EM", "mean"),
                                              Ret_F1=("Ret_F1", "mean"), flops=("flops", "mean"),
                                              est_gpu_s=("est_gpu_s", "mean"), budget_eq4=("budget_eq4", "mean"),
                                              n=("EM", "size")).reset_index()
    agg.to_csv(os.path.join(out, "configs.csv"), index=False)
    write_json(os.path.join(out, "table_meta.json"), {
        "lengths": lengths, "models": models, "ret_ks": RET_KS, "ans_ks": ANS_KS, "probe": PROBE,
        "stage_seconds": {f"{k[0]}_{k[1]}": v[0] for k, v in spf.items()},
        "peak_vram_mb": {f"{k[0]}_{k[1]}": v[3] for k, v in spf.items()}, "gpu": next((v[2] for v in spf.values()), None)})
    # go / no-go: oracle vs best single configuration, on dev
    md = [f"# Step 0 summary - {os.path.basename(run.rstrip('/'))}\n",
          f"GPU: {next((v[2] for v in spf.values()), 'n/a')} · configs: {df.config.nunique()} · questions: "
          f"{sig.split.value_counts().to_dict()}\n",
          f"Measured lengths used for Eq. 4: retrieval Np={lengths['Retrieval'][0]}, Nd={lengths['Retrieval'][1]}; "
          f"QA Np={lengths['Question Answering'][0]}, Nd={lengths['Question Answering'][1]}"
          + (f"; CoT QA Np={lengths['Question Answering (CoT)'][0]}, Nd={lengths['Question Answering (CoT)'][1]}"
             if "Question Answering (CoT)" in lengths else "") + "\n"]
    for split in ("dev", "test"):
        a = agg[agg.split == split].sort_values("EM", ascending=False)
        if a.empty:
            continue
        md.append(f"\n## {split}: configurations (mean per question)\n")
        md.append("| config | meaning | EM | F1 | Ret_F1 | TFLOPs/q | est. GPU s/q | Eq.4 budget |\n|---|---|---|---|---|---|---|---|")
        for r in a.itertuples():
            md.append(f"| {r.config} | {describe_cfg(r.config, models)} | {r.EM:.3f} | {r.F1:.3f} | {r.Ret_F1:.3f} | "
                      f"{r.flops / 1e12:.1f} | {r.est_gpu_s:.2f} | {r.budget_eq4:.0f} |")
        d = df[df.split == split]
        piv_em = d.pivot_table(index="qid", columns="config", values="EM")
        piv_fl = d.pivot_table(index="qid", columns="config", values="flops")
        best_cfg = a.iloc[0].config
        orc_em, orc_fl = [], []
        for qid in piv_em.index:
            ok = [c for c in piv_em.columns if piv_em.loc[qid, c] == 1]
            c = min(ok, key=lambda c: piv_fl.loc[qid, c]) if ok else piv_fl.loc[qid].idxmin()
            orc_em.append(piv_em.loc[qid, c])
            orc_fl.append(piv_fl.loc[qid, c])
        cheapest = a.sort_values("flops").iloc[0]
        best = a.iloc[0]
        md.append(f"\n**Oracle router ({split}):** EM {np.mean(orc_em):.3f} at {np.mean(orc_fl) / 1e12:.1f} TFLOPs/q · "
                  f"best single config {best_cfg}: EM {best.EM:.3f} at {best.flops / 1e12:.1f} TFLOPs/q · "
                  f"cheapest {cheapest.config}: EM {cheapest.EM:.3f} at {cheapest.flops / 1e12:.1f} TFLOPs/q")
        l_only = piv_em[[c for c in piv_em.columns if c.startswith("L") or "-L" in c]].max(axis=1)
        s_only = piv_em[[c for c in piv_em.columns if c.startswith("S") and "-S" in c]].max(axis=1)
        md.append(f"\nQuestions only a config using the 7B gets right: {int(((l_only == 1) & (s_only == 0)).sum())} of "
                  f"{len(piv_em)} · only all-3B gets right: {int(((s_only == 1) & (l_only == 0)).sum())}")
        if split == "dev":
            gain, saving = np.mean(orc_em) - best.EM, 1 - np.mean(orc_fl) / best.flops
            room = gain >= 0.05 or (gain >= 0 and saving >= 0.3)
            md.append(f"\n**Go/no-go:** on dev the oracle router is {gain:+.3f} EM vs the best single config, at "
                      f"{saving:.0%} less compute. " + ("There is room for per-question switching - go ahead with per-question switching."
                      if room else "Little room for a router - consider more paragraphs or a larger model before switching."))
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    plot_configs(agg, os.path.join(out, "configs_em_vs_flops.png"))


# ==================================================================================
# Plot helpers (dataviz reference palette, light mode)
# ==================================================================================
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8})
    return plt


def pareto(points):
    """points: list of (cost, score) -> indices on the frontier (cost asc, strictly better score)."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    front, best = [], -1e9
    for i in order:
        if points[i][1] > best + 1e-12:
            front.append(i)
            best = points[i][1]
    return front


def plot_configs(agg, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, split in zip(axes, ("dev", "test")):
        a = agg[agg.split == split]
        if a.empty:
            continue
        for ci, (pat, name) in enumerate([(r"-S\d", "answer 3B"), (r"-L\d", "answer 7B"),
                                          (r"-Sc\d", "answer 3B, CoT"), (r"-Lc\d", "answer 7B, CoT")]):
            sub = a[a.config.str.contains(pat, regex=True)]
            if sub.empty:
                continue
            ax.scatter(sub.flops / 1e12, sub.EM, s=48, color=SERIES[ci], edgecolor=SURFACE, linewidth=2, zorder=3, label=name)
        for r in a.itertuples():
            ax.annotate(r.config, (r.flops / 1e12, r.EM), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK2)
        f = pareto(list(zip(a.flops, a.EM)))
        fr = a.iloc[f].sort_values("flops")
        ax.plot(fr.flops / 1e12, fr.EM, color=INK2, lw=1.2, ls="--", zorder=2, label="Pareto frontier")
        ax.set_xscale("log")
        from matplotlib.ticker import ScalarFormatter
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xlabel("TFLOPs per question (log)")
        ax.set_title(f"{split}: {int(a.n.max())} questions", color=INK, loc="left")
    axes[0].set_ylabel("EM")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ==================================================================================
# Table access for Scripts 1 and 2
# ==================================================================================
class Table:
    def __init__(self, table_dir):
        import pandas as pd
        self.dir = table_dir
        self.df = pd.read_csv(os.path.join(table_dir, "per_question.csv"))
        self.sig = pd.read_csv(os.path.join(table_dir, "signals.csv"))
        self.meta = json.load(open(os.path.join(table_dir, "table_meta.json")))
        self.models = self.meta["models"]
        self.agg = self.df.groupby(["split", "config"]).agg(EM=("EM", "mean"), F1=("F1", "mean"), Ret_F1=("Ret_F1", "mean"),
                                                            flops=("flops", "mean"), est_gpu_s=("est_gpu_s", "mean"),
                                                            budget_eq4=("budget_eq4", "mean")).reset_index()
        self.cfgs = sorted(self.df.config.unique())
        self.em = {sp: self.df[self.df.split == sp].pivot_table(index="qid", columns="config", values="EM") for sp in ("dev", "test")}
        self.f1 = {sp: self.df[self.df.split == sp].pivot_table(index="qid", columns="config", values="F1") for sp in ("dev", "test")}
        self.fl = {sp: self.df[self.df.split == sp].pivot_table(index="qid", columns="config", values="flops") for sp in ("dev", "test")}
        self.sec = {sp: self.df[self.df.split == sp].pivot_table(index="qid", columns="config", values="est_gpu_s") for sp in ("dev", "test")}

    def config_stats(self, split, cfg):
        r = self.agg[(self.agg.split == split) & (self.agg.config == cfg)].iloc[0]
        return {"EM": float(r.EM), "F1": float(r.F1), "Ret_F1": float(r.Ret_F1), "flops": float(r.flops),
                "est_gpu_s": float(r.est_gpu_s), "budget_eq4": float(r.budget_eq4)}


def bootstrap_diff(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), (n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.mean(a) - np.mean(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


class AgentLLM:
    """Batched chat for the agent (Script 1 and the rule writer in Script 2)."""

    def __init__(self, backend, path, thinking, max_model_len, gpu_util, seed, max_tokens):
        self.eng = make_engine(backend, path, size="agent", max_model_len=max_model_len, gpu_util=gpu_util, seed=seed,
                               thinking=thinking if thinking else None)
        self.max_tokens, self.calls, self.in_tok, self.out_tok, self.seconds = max_tokens, 0, 0, 0, 0.0

    def chat(self, items):
        """items: list of (system, user, seed, meta) -> list of (thinking, answer)."""
        t = time.time()
        prompts = [self.eng.encode(s, u) for s, u, _, _ in items]
        limit = getattr(self.eng, "max_model_len", 10 ** 9) - self.max_tokens - 8
        too_long = [len(p) for p in prompts if len(p) > limit]
        if too_long:
            raise SystemExit(f"agent prompt of {max(too_long)} tokens exceeds the context ({limit}); lower --max_evidence")
        outs = self.eng.generate(prompts, 1, 0.6, self.max_tokens, [x[2] for x in items], [x[3] for x in items])
        self.seconds += time.time() - t
        self.calls += len(items)
        self.in_tok += sum(len(p) for p in prompts)
        self.out_tok += sum(o[0][1] for o in outs)
        return [split_thinking(o[0][0]) for o in outs]


def extract_json(text):
    try:
        s, e = text.find("{"), text.rfind("}") + 1
        return json.loads(text[s:e])
    except Exception:
        return None


# ==================================================================================
# Script 1: AgentTTS (ported from code_others/AgentTTS.py + agent/agent_class.py)
# ==================================================================================
SUBTASKS = ["Retrieval", "Question Answering"]
TASK_DESC = ("This task Retreval-then-generate has two subtasks. The first subtask is retrieval: given a question and a set of "
             "{n} text chunks, the model needs to select the most relevant chunks that can help answer the question. The second "
             "subtask is question answering: based on the retrieved chunks, the model should generate a response that answers "
             "the question accurately.")


class AgentTTSRun:
    def __init__(self, table, budget, seed, max_trials, batch_size, n_chunks, allow_cot=False, cfgs=None):
        self.T, self.budget, self.seed, self.max_trials, self.batch_size = table, budget, seed, max_trials, batch_size
        ms = table.models
        self.cfgs = cfgs or table.cfgs
        self.name_of = {k: ms[k]["id"] for k in "SL"}
        if allow_cot:   # Set 2b control: chain-of-thought answering appears as two extra QA "models"
            self.name_of.update({k + "c": ms[k]["id"] + " (chain-of-thought)" for k in "SL"})
        self.key_of = {v: k for k, v in self.name_of.items()}
        self.size = {name: ms[base(k)]["nominal"] for k, name in self.name_of.items()}
        qa = ["S", "Sc", "L", "Lc"] if allow_cot else ["S", "L"]
        self.model_choices = [[self.name_of["S"], self.name_of["L"]], [self.name_of[k] for k in qa]]
        L = table.meta["lengths"]
        self.lengths = L
        self.Np = [L["Retrieval"][0], L["Question Answering"][0]]
        self.Nd = [L["Retrieval"][1], L["Question Answering"][1]]
        self.M_dict = {st: max(self.size[m] for m in ch) for st, ch in zip(SUBTASKS, self.model_choices)}
        self.task_desc = TASK_DESC.format(n=n_chunks)
        self.model_space = {st: {"model_choice": ch, "prompt_length": Np, "generation_length": Nd}
                            for st, ch, Np, Nd in zip(SUBTASKS, self.model_choices, self.Np, self.Nd)}
        if allow_cot:
            Np_c, Nd_c = L["Question Answering (CoT)"]
            self.task_desc += (" For question answering, models marked '(chain-of-thought)' reason step by step before "
                               "answering; they produce longer generations, so their budget uses their own lengths.")
            self.model_space["Question Answering"]["chain_of_thought_models"] = {"prompt_length": Np_c, "generation_length": Nd_c}
        ex = {st: {"model": ch[-1], "samples": 1, "budget": round(compute_budget(1, self.size[ch[-1]], *self.len_of(st, ch[-1])), 1)}
              for st, ch in zip(SUBTASKS, self.model_choices)}
        self.example = {"candidates": [ex]}
        self.metrics, self.main = ["Gen_EM", "Ret_F1"], "Gen_EM"
        self.history, self.experience, self.trials, self.log = [], [], [], []
        self.done, self.recent_added, self.evaluated = False, 0, set()

    def len_of(self, subtask, model):
        """(prompt, generation) lengths used in Eq. 4; chain-of-thought answers have their own."""
        if subtask == "Question Answering" and is_cot(self.key_of.get(model, "S")):
            return tuple(self.lengths["Question Answering (CoT)"])
        return tuple(self.lengths[subtask])

    # --- budget (repo) ---
    def calc_budget(self, params):
        return sum(compute_budget(params[st]["samples"], self.size[params[st]["model"]], *self.len_of(st, params[st]["model"]))
                   for st in SUBTASKS)

    def conv_budget_to_samples(self, budget, m, Np, Nd):
        alpha, beta1, beta2, beta3 = m / 3e9, Np / Nd, 128 / 64, Np / 128
        return beta1 * (((budget / beta2) + 1) / (alpha * beta3) - 1)

    # --- environment: snap to the grid like the repo's fuzzy_match, dev scores only ---
    def env_run(self, params):
        try:
            rm, am = self.key_of[params["Retrieval"]["model"]], self.key_of[params["Question Answering"]["model"]]
            rs, as_ = float(params["Retrieval"]["samples"]), float(params["Question Answering"]["samples"])
        except Exception:
            return None, None
        best, best_d = None, None
        for cfg in self.cfgs:          # the recorded grid for these two models (repo: fuzzy_match on its CSV)
            r, rk, a, ak = parse_cfg(cfg)
            if (r, a) != (rm, am) or self.T.config_stats("dev", cfg)["budget_eq4"] > self.budget:
                continue
            d1 = self.M_dict["Retrieval"] / self.size[self.name_of[rm]] * 4 - 3
            d2 = self.M_dict["Question Answering"] / self.size[self.name_of[am]] * 4 - 3
            d = abs(rk - rs) / d1 + abs(ak - as_) / d2
            if best_d is None or d < best_d:
                best, best_d = cfg, d
        if best is None or best_d > 5:
            return None, None
        st = self.T.config_stats("dev", best)
        r, rk, a, ak = parse_cfg(best)
        snapped = {"Retrieval": {"model": self.name_of[r], "samples": rk}, "Question Answering": {"model": self.name_of[a], "samples": ak}}
        return snapped, {"Gen_EM": round(st["EM"], 3), "Ret_F1": round(st["Ret_F1"], 3), "_config": best}

    def evaluate(self, cands, source):
        added = 0
        for c in cands:
            if self.done:
                break
            if not isinstance(c, dict) or any(s not in c for s in SUBTASKS):
                continue
            try:
                if self.calc_budget(c) > self.budget:
                    self.log.append({"event": "over_budget", "candidate": c})
                    continue
            except Exception:
                continue
            snapped, score = self.env_run(c)
            if snapped is None:
                self.log.append({"event": "no_match", "candidate": c})
                continue
            cfg = score.pop("_config")
            cost = self.calc_budget(snapped)
            new = cfg not in self.evaluated
            self.evaluated.add(cfg)
            self.trials.append({"trial": len(self.trials) + 1, "source": source, "config": cfg, "requested": c,
                                "dev_EM": score["Gen_EM"], "dev_Ret_F1": score["Ret_F1"], "budget_eq4": cost, "new": new})
            if score[self.main] > 0:
                self.history.append({"params": snapped, "score": score, "cost": cost})
                added += 1
            if len(self.evaluated) >= self.max_trials:
                self.done = True
        self.recent_added = added

    # --- init candidates (Insight 1), repo logic ---
    def init_candidates(self):
        split = [self.budget * 0.85, self.budget * 0.15]
        cands = []
        for st, bsub, ch in zip(SUBTASKS, split, self.model_choices):
            max_size = self.M_dict[st]
            low_others = sum(compute_budget(1, self.size[ch2[0]], *self.len_of(st2, ch2[0]))
                             for st2, ch2 in zip(SUBTASKS, self.model_choices) if st2 != st)
            high_others = sum(compute_budget(1, self.size[ch2[-1]], *self.len_of(st2, ch2[-1]))
                              for st2, ch2 in zip(SUBTASKS, self.model_choices) if st2 != st)
            avail = self.budget - low_others
            b1 = compute_budget(1, max_size, *self.len_of(st, ch[-1]))
            b = b1 if b1 < avail else avail
            for model in ch:
                S = self.conv_budget_to_samples(b, self.size[model], *self.len_of(st, model))
                if S < 1:
                    continue
                c = {st: {"model": model, "samples": int(min(S, 60))}}
                for st2, ch2, Np2, Nd2 in zip(SUBTASKS, self.model_choices, self.Np, self.Nd):
                    if st2 != st:
                        m2 = ch2[-1] if self.budget - b > high_others else ch2[0]
                        c[st2] = {"model": m2, "samples": 1}
                cands.append({s: c[s] for s in SUBTASKS})
        return cands

    def hist_str(self, h):
        return "\n".join(f"Parameters: {x['params']}, Score: {x['score']}, Used budget: {x['cost']:.2f}, total budget: {self.budget}" for x in h)

    def experience_prompt(self, init):
        sysm = "You are an expert in parameter optimization."
        if init:
            return sysm, f"""I am tuning test-time parameters for a complex task that can be broken down into multiple simple subtasks. The parameters include the model size and the number of samples used for each subtask. Generally, increasing the number of samples improves performance. The goal is to find the best parameter settings that maximize performance within a fixed budget.

            I will provide the task name, description, budget, and initial search history. Please help me summarize insights from the search history in the following aspect:

            1. The search history includes initial trials that use different model sizes under the same unit budget. In these trials, smaller models are allowed more samples, while larger models have fewer samples due to their higher cost. Please help me identify which model size we should explore further in each subtask. If the large model performs significantly better than the small model, we will choose the large one. Otherwise, if its performance is similar or only slightly better, we will prefer the smaller model.
            2. As the number of samples increases, performance tends to improve and then level off. Please identify the search direction for finding the optimal number of samples for each subtask, especially the point after which more samples no longer improve results.
            3. The total compute budget is shared across all subtasks and they are interdependent. If we could not assign the optimal model and sample count to every subtask, consider the following strategy. Leveraging your planning capability, prioritize each subtask, try different combinations of budget allocations for the other tasks, and balance the allocations across subtasks.

            ###Task name: hotpotqa
            ###Task description: {self.task_desc}
            ###Subtasks: {SUBTASKS}
            ###Budget: {self.budget}
            ###Metrics: {self.metrics}, Main metric: {self.main}

            ###Search history: {self.hist_str(self.history[-self.recent_added:] if self.recent_added else self.history)}

            Please generate your response in most concise wording.

            ###Response:
            """
        insight2 = "As the number of samples increases, performance tends to improve and then level off. Please identify the search direction for finding the optimal number of samples for each subtask, especially the point after which more samples no longer improve results."
        insight3 = "The total compute budget is shared across all subtasks and they are interdependent. If we could not assign the optimal model and sample count to every subtask, consider the following strategy. Prioritize each subtask and try different combinations of the budget allocation in other tasks and Balance the allocations among subtasks. "
        k = self.recent_added
        old, recent = (self.history[:-k], self.history[-k:]) if k else (self.history, [])
        return sysm, f"""I am tuning test-time parameters for a complex task that can be broken down into multiple simple subtasks. The parameters include the model size and the number of samples used for each subtask. Generally, increasing the number of samples improves performance. The goal is to find the best parameter settings that maximize performance within a fixed budget.

            I will provide the task name, description, budget, and both old and recent search history.

            Please help me summarize insights from the recent search history in the following three aspects:

            1. {insight2}

            2. {insight3}

            3. If you observe that the performance variance across the search history is small, please suggest that future searches focus more on exploration strategies such as crossover, mutation, and random search, rather than strictly following the existing search patterns.

            ###Task name: hotpotqa
            ###Task description: {self.task_desc}
            ###Subtasks: {SUBTASKS}
            ###Budget: {self.budget}
            ###Metrics: {self.metrics}, Main metric: {self.main}

            ###Old search history: {self.hist_str(old)}
            ###Recent search history: {self.hist_str(recent)}

            Please generate your response in most concise wording, item by item.

            ###Response:
            """

    def candidate_prompt(self):
        exp = "\n".join(self.experience[:1] + self.experience[-2:])
        return "You are an expert in parameter optimization.", f"""I am tuning test-time parameters for a complex task that can be broken down into multiple simple subtasks. The parameters include the model size and the number of samples used for each subtask. Generally, increasing the number of samples improves performance. The goal is to find the best parameter settings that maximize performance within a fixed budget.

        I will provide the task name, task description, a list of subtasks and the model choices available for each, the total budget, available samples, the budget calculation formula, evaluation metrics (including the main metric to optimize), search history (previous configurations tried), and insights from previous experiments.

        The budget for a subtask is calculated using this code:
        ```python
        def compute_budget(num_samples, model_size, prompt_length, generation_length, M_small=3e9, Np2=128, Nd2=64):
            # Compute the FLOPs-based budget of generating num_samples samples on a large model,
            # normalized to the FLOPs of a small model's unit budget.
            alpha = model_size / M_small
            beta1 = prompt_length / generation_length
            beta2 = Np2 / Nd2  # normalize to same decode length as unit config
            beta3 = prompt_length / Np2

            budget = beta2 * ((alpha * beta3 / beta1) * (beta1 + num_samples) - 1)
            return budget
        ```
        Model sizes: {', '.join(f"{n} = {self.size[n]:.0e}" for n in self.name_of.values())}.
        The total budget is the sum of the budgets across all subtasks.

        Using the information I provide, please generate {self.batch_size} new candidate configurations that follow the insights and stay within the total budget. The output must be in strict JSON format, using the structure shown below:

        {json.dumps(self.example, indent=2)}

        Inputs:
        ###Task name: hotpotqa
        ###Task description: {self.task_desc}
        ###Subtasks, model choices, and prompt_length, generation_length: {self.model_space}
        ###Available samples: [1, 100]
        ###Total budget: {self.budget}
        ###Budget equation per subtask: as described above
        ###Metrics: {self.metrics}
        ###Main metric: {self.main}
        ###Search history: {self.hist_str(self.history)}
        ###Insights: {exp}

        Your response should return exactly {self.batch_size} new candidates, in strict JSON format only.

        ###Response:
        """

    def best(self):
        if not self.trials:
            return None
        return sorted(self.trials, key=lambda t: (-t["dev_EM"], t["budget_eq4"]))[0]["config"]


def cmd_agenttts(args):
    import pandas as pd
    T = Table(args.table_dir)
    run = args.run_dir
    os.makedirs(run, exist_ok=True)
    n_chunks = len(json.load(open(os.path.join(args.data_dir, "dev.json")))[0]["ctxs_candidate"])
    cfgs = [c for c in T.cfgs if args.allow_cot or not is_cot(parse_cfg(c)[2])]
    # budgets are quantiles of the PLAIN (non-CoT) setups' costs, so a CoT-allowed run gets exactly the Set 1 budgets
    plain = [c for c in T.cfgs if not is_cot(parse_cfg(c)[2])]
    dev = T.agg[(T.agg.split == "dev") & T.agg.config.isin(plain)]
    dev_budgets = sorted(dev.budget_eq4)
    budgets = [round(float(np.quantile(dev_budgets, q)), 1) for q in args.budget_quantiles]
    log(f"AgentTTS | {len(cfgs)} setups{' (chain-of-thought allowed)' if args.allow_cot else ''} | budgets (Eq. 4 units, "
        f"quantiles {args.budget_quantiles} of setup costs): {budgets} | seeds {args.seeds}")
    runs = [AgentTTSRun(T, b, s, args.max_trials, args.batch_size, n_chunks, args.allow_cot, cfgs)
            for b in budgets for s in range(args.seeds)]
    for r in runs:
        r.evaluate(r.init_candidates(), "init (Insight 1)")
    st = Status(run, "agenttts", args.iterations, args.time_limit_min)
    agent = AgentLLM(args.backend, args.agent_path, args.thinking, args.max_model_len, args.gpu_util, args.seed, args.agent_max_tokens)
    for it in range(args.iterations):
        active = [r for r in runs if not r.done]
        if not active:
            break
        items = [(*r.experience_prompt(init=(it == 0)), 1000 * it + i, {"kind": "agent_text"}) for i, r in enumerate(active)]
        for r, (think, ans) in zip(active, agent.chat(items)):
            r.experience.append(ans)
            r.log.append({"event": "guidelines", "iteration": it + 1, "text": ans, "thinking": think})
        items = [(*r.candidate_prompt(), 2000 + 1000 * it + i,
                  {"kind": "agent_json", "choices": r.model_choices, "k": r.batch_size}) for i, r in enumerate(active)]
        for r, (think, ans) in zip(active, agent.chat(items)):
            js = extract_json(ans)
            r.log.append({"event": "candidates", "iteration": it + 1, "text": ans, "thinking": think, "parsed": js is not None})
            r.evaluate((js or {}).get("candidates", []), f"agent round {it + 1}")
        st.update(it + 1, f"{sum(r.done for r in runs)}/{len(runs)} runs finished")
        log(f"  round {it + 1}: distinct configs tried per run {[len(r.evaluated) for r in runs]}")
        if st.over_limit():
            log("time limit reached - stopping cleanly")
            break
    # results
    rows = []
    for r in runs:
        best = r.best()
        rec = {"budget_eq4": r.budget, "seed": r.seed, "trials": len(r.trials), "distinct_configs": len(r.evaluated), "chosen": best}
        if best:
            d, t = T.config_stats("dev", best), T.config_stats("test", best)
            rec.update({"dev_EM": d["EM"], "test_EM": t["EM"], "test_F1": t["F1"], "test_TFLOPs": t["flops"] / 1e12,
                        "test_est_gpu_s": t["est_gpu_s"]})
        rows.append(rec)
        write_json(os.path.join(run, "traces", f"budget{r.budget}_seed{r.seed}.json"),
                   {"budget": r.budget, "seed": r.seed, "trials": r.trials, "log": r.log, "chosen": best})
    res = pd.DataFrame(rows)
    # dev-optimal global config per budget (exhaustive search of the dev table)
    opt = []
    for b in budgets:
        dev_all = T.agg[(T.agg.split == "dev") & T.agg.config.isin(cfgs)]   # every setup the agent may choose
        feas = dev_all[dev_all.budget_eq4 <= b].sort_values(["EM", "flops"], ascending=[False, True])
        c = feas.iloc[0].config
        t = T.config_stats("test", c)
        opt.append({"budget_eq4": b, "dev_optimal": c, "dev_EM": float(feas.iloc[0].EM), "test_EM": t["EM"], "test_F1": t["F1"],
                    "test_TFLOPs": t["flops"] / 1e12, "test_est_gpu_s": t["est_gpu_s"]})
    opt = pd.DataFrame(opt)
    res.to_csv(os.path.join(run, "agenttts_runs.csv"), index=False)
    opt.to_csv(os.path.join(run, "dev_optimal.csv"), index=False)
    write_json(os.path.join(run, "agent_usage.json"), {"calls": agent.calls, "input_tokens": agent.in_tok,
                                                        "output_tokens": agent.out_tok, "seconds": agent.seconds})
    md = [f"# AgentTTS global configuration - {os.path.basename(run.rstrip('/'))}\n",
          f"Setups searched: {len(cfgs)}" + (" (chain-of-thought answering allowed)" if args.allow_cot else " (no chain-of-thought, as in the paper)") + "\n",
          f"Agent: {args.agent_path} (thinking={args.thinking}) · {agent.calls} calls, {agent.in_tok} in / {agent.out_tok} out tokens, "
          f"{agent.seconds:.0f}s · max {args.max_trials} distinct configs per run\n",
          "\n## Dev-optimal global config (exhaustive, the ceiling for any single config)\n",
          "| budget (Eq.4) | config | meaning | dev EM | test EM | test F1 | test TFLOPs/q | test est. GPU s/q |\n|---|---|---|---|---|---|---|---|"]
    for r in opt.itertuples():
        md.append(f"| {r.budget_eq4:.0f} | {r.dev_optimal} | {describe_cfg(r.dev_optimal, T.models)} | {r.dev_EM:.3f} | {r.test_EM:.3f} | "
                  f"{r.test_F1:.3f} | {r.test_TFLOPs:.1f} | {r.test_est_gpu_s:.2f} |")
    md += ["\n## AgentTTS agent (one row per budget x seed)\n",
           "| budget | seed | distinct configs tried | chosen | dev EM | test EM | test TFLOPs/q | matches dev-optimal |\n|---|---|---|---|---|---|---|---|"]
    for r in res.itertuples():
        o = opt[opt.budget_eq4 == r.budget_eq4].iloc[0].dev_optimal
        md.append(f"| {r.budget_eq4:.0f} | {r.seed} | {r.distinct_configs} | {r.chosen} | {getattr(r, 'dev_EM', float('nan')):.3f} | "
                  f"{getattr(r, 'test_EM', float('nan')):.3f} | {getattr(r, 'test_TFLOPs', float('nan')):.1f} | {'yes' if r.chosen == o else ('tie on dev EM' if abs(getattr(r, 'dev_EM', -1) - opt[opt.budget_eq4 == r.budget_eq4].iloc[0].dev_EM) < 1e-9 else 'no')} |")
    md.append(f"\nTraces (every trial, guidelines and the agent's thinking) are in `traces/`. Test scores are read from the Step 0 table; "
              f"the agent only ever saw dev scores.")
    with open(os.path.join(run, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


# ==================================================================================
# Script 2: per-question switching
# ==================================================================================
FEATS_PRE = ["f_words", "f_yesno", "f_compare", "f_digit", "f_caps", "f_quote"]
FEATS_POST = FEATS_PRE + ["s_ret_agree", "s_ans_agree", "s_ans_unknown"]


class Evaluator:
    """Scores a policy {qid: config} on a split; post-probe policies pay for the probe unless they keep its answer."""

    def __init__(self, T):
        self.T = T
        self.ref = T.config_stats("dev", sorted(T.cfgs, key=lambda c: T.config_stats("dev", c)["flops"])[0])["flops"]

    def score(self, split, choice, probe=False, overhead_flops=None):
        em, f1, fl, sec = [], [], [], []
        for qid, c in choice.items():
            em.append(self.T.em[split].loc[qid, c])
            f1.append(self.T.f1[split].loc[qid, c])
            f = self.T.fl[split].loc[qid, c]
            s = self.T.sec[split].loc[qid, c]
            if probe and c != PROBE:
                f += self.T.fl[split].loc[qid, PROBE]
                s += self.T.sec[split].loc[qid, PROBE]
            if overhead_flops:
                f += overhead_flops.get(qid, 0.0)
            fl.append(f)
            sec.append(s)
        return {"EM": float(np.mean(em)), "F1": float(np.mean(f1)), "flops": float(np.mean(fl)),
                "rel_cost": float(np.mean(fl) / self.ref), "est_gpu_s": float(np.mean(sec)), "_em": em}

    def utility(self, s, lam):
        return s["EM"] - lam * s["rel_cost"]


def per_question_utility(T, split, lam, ref, cfgs, probe=False):
    em, fl = T.em[split], T.fl[split]
    u = em[cfgs].copy()
    for c in cfgs:
        extra = fl[PROBE] if (probe and c != PROBE) else 0
        u[c] = em[c] - (lam + 1e-6) * (fl[c] + extra) / ref   # the tiny term breaks ties toward the cheaper config
    return u


def cmd_switching(args):
    import pandas as pd
    T = Table(args.table_dir)
    run = args.run_dir
    os.makedirs(run, exist_ok=True)
    E = Evaluator(T)
    sig = T.sig.set_index("qid")
    dev_q, test_q = list(T.em["dev"].index), list(T.em["test"].index)
    lams = args.lambdas
    dev_agg = {c: T.config_stats("dev", c) for c in T.cfgs}
    results, decisions = [], {}

    def add(policy, setting, lam, choice, probe=False, overhead=None, extra=None):
        s = E.score("test", choice, probe, overhead)
        d = E.score("dev", extra["dev_choice"], probe, extra.get("dev_overhead")) if extra and "dev_choice" in extra else None
        results.append({"policy": policy, "setting": setting, "lambda": lam, "test_EM": s["EM"], "test_F1": s["F1"],
                        "test_TFLOPs": s["flops"] / 1e12, "rel_cost": s["rel_cost"], "test_est_gpu_s": s["est_gpu_s"],
                        "dev_EM": d["EM"] if d else None, "dev_rel_cost": d["rel_cost"] if d else None,
                        "configs_used": dict(Counter(choice.values())), "_em": s["_em"]})
        decisions[(policy, setting, lam)] = choice

    def add_agenttts_picks():   # the global configs chosen by the AgentTTS agent in notebook 05, if given
        if args.script1_dir and os.path.exists(os.path.join(args.script1_dir, "agenttts_runs.csv")):
            s1 = pd.read_csv(os.path.join(args.script1_dir, "agenttts_runs.csv"))
            for r in s1.dropna(subset=["chosen"]).itertuples():
                if r.chosen in T.cfgs:
                    add(f"AgentTTS agent (budget {r.budget_eq4:.0f}, seed {r.seed})", "fixed", None, {q: r.chosen for q in test_q})

    if args.router_version == 2:
        switching_v2(args, T, E, sig, dev_q, test_q, dev_agg, add, run)
        add_agenttts_picks()
        report_switching(args, T, E, results, decisions, run)
        return

    # ---- fixed baselines ----
    cheapest = min(T.cfgs, key=lambda c: dev_agg[c]["flops"])
    all_L = max([c for c in T.cfgs if c.startswith("L") and "-L" in c], key=lambda c: (dev_agg[c]["EM"], -dev_agg[c]["flops"]))
    add("always 3B (cheapest config)", "fixed", None, {q: cheapest for q in test_q})
    add("always 7B (best all-7B config on dev)", "fixed", None, {q: all_L for q in test_q})
    for lam in lams:
        g = max(T.cfgs, key=lambda c: (dev_agg[c]["EM"] - lam * dev_agg[c]["flops"] / E.ref, -dev_agg[c]["flops"]))
        add("dev-optimal global config", "fixed", lam, {q: g for q in test_q}, extra={"dev_choice": {q: g for q in dev_q}})
    # ---- oracle ----
    orc = {}
    for q in test_q:
        ok = [c for c in T.cfgs if T.em["test"].loc[q, c] == 1]
        orc[q] = min(ok or T.cfgs, key=lambda c: T.fl["test"].loc[q, c])
    add("oracle router (uses test answers)", "oracle", None, orc)

    # ---- cascade ----
    taus = [0.2, 0.4, 0.6, 0.8, 1.01]
    for lam in lams:
        best = None
        for tau in taus:
            for esc in T.cfgs:
                ch = {q: (PROBE if sig.loc[q, "s_ans_agree"] >= tau and not sig.loc[q, "s_ans_unknown"] else esc) for q in dev_q}
                u = E.utility(E.score("dev", ch, probe=True), lam)
                if best is None or u > best[0]:
                    best = (u, tau, esc, ch)
        _, tau, esc, dch = best
        tch = {q: (PROBE if sig.loc[q, "s_ans_agree"] >= tau and not sig.loc[q, "s_ans_unknown"] else esc) for q in test_q}
        add(f"cascade (agree<{tau:g} -> {esc})", "after probe", lam, tch, probe=True, extra={"dev_choice": dch})

    # ---- classifier ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    for setting, feats, probe in (("before running", FEATS_PRE, False), ("after probe", FEATS_POST, True)):
        Xd = sig.loc[dev_q, feats].fillna(0).values.astype(float)
        Xt = sig.loc[test_q, feats].fillna(0).values.astype(float)
        sc = StandardScaler().fit(Xd)
        Xd, Xt = sc.transform(Xd), sc.transform(Xt)
        for lam in lams:
            U = per_question_utility(T, "dev", lam, E.ref, T.cfgs, probe)
            # actions: configs that are the best choice for at least 2 dev questions, plus the global best
            y_all = U.idxmax(axis=1)
            g = U.mean().idxmax()
            acts = sorted({c for c, n in Counter(y_all).items() if n >= 2} | {g})
            y = U[acts].idxmax(axis=1).loc[dev_q].values
            if len(set(y)) < 2:
                dch, tch = {q: y[0] for q in dev_q}, {q: y[0] for q in test_q}
            else:
                best_c, best_u = 1.0, None
                for C in (0.1, 0.3, 1.0, 3.0):   # leave-one-out on dev
                    pred = []
                    for i in range(len(dev_q)):
                        m = np.ones(len(dev_q), bool)
                        m[i] = False
                        if len(set(y[m])) < 2:
                            pred.append(y[m][0])
                            continue
                        clf = LogisticRegression(C=C, max_iter=2000).fit(Xd[m], y[m])
                        pred.append(clf.predict(Xd[i:i + 1])[0])
                    u = E.utility(E.score("dev", dict(zip(dev_q, pred)), probe), lam)
                    if best_u is None or u > best_u:
                        best_c, best_u, dch = C, u, dict(zip(dev_q, pred))
                clf = LogisticRegression(C=best_c, max_iter=2000).fit(Xd, y)
                tch = dict(zip(test_q, clf.predict(Xt)))
            add("classifier", setting, lam, tch, probe, extra={"dev_choice": dch})

    # ---- natural-language rule router ----
    if not args.skip_nl:
        nl_rules_router(args, T, E, sig, dev_q, test_q, dev_agg, add, run)

    add_agenttts_picks()

    report_switching(args, T, E, results, decisions, run)


def nl_rules_router(args, T, E, sig, dev_q, test_q, dev_agg, add, run):
    models = T.models
    agent = AgentLLM(args.backend, args.agent_path, args.thinking, args.max_model_len, args.agent_gpu_util, args.seed,
                     args.agent_max_tokens)
    applier = make_engine(args.backend, models["S"]["path"], size="S", max_model_len=4096, gpu_util=args.applier_gpu_util,
                          seed=args.seed)
    # a small action menu: the dev Pareto configs (cost vs EM) plus the probe
    pts = [(dev_agg[c]["flops"], dev_agg[c]["EM"]) for c in T.cfgs]
    acts = sorted({T.cfgs[i] for i in pareto(pts)} | {PROBE}, key=lambda c: dev_agg[c]["flops"])
    menu = "\n".join(f"- {c}: {describe_cfg(c, models)} | dev EM {dev_agg[c]['EM']:.2f} | cost {dev_agg[c]['flops'] / E.ref:.1f}x the cheapest"
                     for c in acts)
    rules_log = []

    def signals_text(q, setting):
        s = sig.loc[q]
        t = f"Question: {s.question}"
        if setting == "after probe":
            t += (f"\nSignals from a cheap 3B first pass: retrieval agreement {s.s_ret_agree:.2f}, answer agreement "
                  f"{s.s_ans_agree:.2f}, answer looks like 'unknown': {'yes' if s.s_ans_unknown else 'no'}")
        return t

    def apply(rules, qids, setting, seed):
        items = []
        for q in qids:
            user = (f"Routing rules:\n{rules}\n\nAvailable routes: {', '.join(acts)}\n\n{signals_text(q, setting)}\n\n"
                    f"Follow the rules. Reply with the route name only.")
            items.append((user, q))
        prompts = [applier.encode("You route questions to a processing option.", u) for u, _ in items]
        outs = applier.generate(prompts, 1, 0.0, 12, [seed] * len(prompts), [{"kind": "route", "actions": acts}] * len(prompts))
        choice, over = {}, {}
        for (u, q), p, o in zip(items, prompts, outs):
            m = re.search(r"[SL]\d-[SL]\d", o[0][0])
            choice[q] = m.group(0) if m and m.group(0) in acts else acts[0]
            over[q] = 2 * models["S"]["params"] * (len(p) + o[0][1])
        return choice, over

    for setting in ("before running", "after probe"):
        probe = setting == "after probe"
        for lam in args.nl_lambdas:
            U = per_question_utility(T, "dev", lam, E.ref, acts, probe)
            # evidence for the rule writer: every "mixed" question (some routes right, some wrong) first - they carry the
            # routing signal - then others, up to --max_evidence lines so the prompt stays within the agent's context
            mixed = [q for q in dev_q if 0 < T.em["dev"].loc[q, acts].sum() < len(acts)]
            rest = [q for q in dev_q if q not in set(mixed)]
            rng = random.Random(args.seed)
            rng.shuffle(mixed)
            rng.shuffle(rest)
            shown = (mixed + rest)[:args.max_evidence]
            evidence = []
            for q in shown:
                ok = [c for c in acts if T.em["dev"].loc[q, c] == 1]
                evidence.append(f"- {signals_text(q, setting).replace(chr(10), ' | ')} | correct routes: {', '.join(ok) or 'none'} | best: {U.loc[q].idxmax()}")
            base = (f"We answer multi-hop questions with a retrieve-then-answer pipeline and can run it in several ways "
                    f"(small Qwen 3B or large Qwen 7B, 1 or more samples). Routes, cheapest first:\n{menu}\n\n"
                    f"Goal: for each NEW question pick the route that maximises accuracy minus {lam} x cost (cost in units of "
                    f"the cheapest route{'; with probe signals the cheap first pass is always paid and route ' + PROBE + ' reuses it' if probe else ''}).\n"
                    f"Write at most {args.max_rules} plain-English rules of the form 'IF <condition on the question"
                    f"{' or the signals' if probe else ''}> THEN use <route>', checked in order, the last one a default. "
                    f"Conditions must be checkable from the question text{' and signals' if probe else ''} alone. Put the rules between <rules> and </rules>.\n\n"
                    f"Evidence from {len(shown)} of {len(dev_q)} development questions (all questions where the routes disagree come first):\n" + "\n".join(evidence))
            best = None
            rules, feedback = None, ""
            for rnd in range(args.nl_rounds):
                user = base + (f"\n\nYour previous rules:\n{rules}\n\nResult on the development questions: {feedback}\n"
                               f"Revise the rules to improve accuracy minus cost." if rules else "")
                think, ans = agent.chat([("You write short, general routing rules from evidence.", user, 31 * rnd + 7,
                                          {"kind": "rules", "actions": acts})])[0]
                m = re.search(r"<rules>(.*?)</rules>", ans, flags=re.S)
                rules = (m.group(1) if m else ans).strip()
                dch, dov = apply(rules, dev_q, setting, args.seed)
                s = E.score("dev", dch, probe, dov)
                u = E.utility(s, lam)
                wrong = [q for q in dev_q if T.em["dev"].loc[q, dch[q]] == 0 and U.loc[q].max() > U.loc[q, dch[q]]][:8]
                waste = [q for q in dev_q if T.em["dev"].loc[q, dch[q]] == 1 and U.loc[q].idxmax() != dch[q]][:6]
                feedback = (f"EM {s['EM']:.3f}, cost {s['rel_cost']:.2f}x, utility {u:.3f}. "
                            f"Wrong where a better route existed: " + "; ".join(f"'{sig.loc[q, 'question']}' got {dch[q]}, best {U.loc[q].idxmax()}" for q in wrong) +
                            ". Correct but more expensive than needed: " + "; ".join(f"'{sig.loc[q, 'question']}' got {dch[q]}, cheaper {U.loc[q].idxmax()}" for q in waste))
                rules_log.append({"setting": setting, "lambda": lam, "round": rnd + 1, "rules": rules, "thinking": think,
                                  "dev_EM": s["EM"], "dev_rel_cost": s["rel_cost"], "dev_utility": u})
                log(f"  NL rules | {setting} | lambda {lam} | round {rnd + 1}: dev EM {s['EM']:.3f}, cost {s['rel_cost']:.2f}x")
                if best is None or u > best[0]:
                    best = (u, rules, dch, dov)
            _, rules, dch, dov = best
            tch, tov = apply(rules, test_q, setting, args.seed)
            add("NL rule router", setting, lam, tch, probe, tov, extra={"dev_choice": dch, "dev_overhead": dov})
            with open(os.path.join(run, f"rules_{setting.replace(' ', '_')}_lam{lam}.txt"), "w", encoding="utf-8") as f:
                f.write(rules + "\n")
    write_json(os.path.join(run, "nl_rules_log.json"), rules_log)
    write_json(os.path.join(run, "agent_usage.json"), {"calls": agent.calls, "input_tokens": agent.in_tok,
                                                        "output_tokens": agent.out_tok, "seconds": agent.seconds})


# ==================================================================================
# Script 2, router v2 (Set 2a / 2b): a short route menu, checkable conditions, enforced rule cap
# ==================================================================================
V2_FEATURES = {  # name -> (signals.csv column, kind, meaning shown to the rule writer)
    "yesno": ("f_yesno", "bool", "the question is a yes/no question"),
    "compare": ("f_compare", "bool", "the question compares or chooses between things (which/older/both/or ...)"),
    "words": ("f_words", "num", "number of words in the question"),
    "digit": ("f_digit", "bool", "the question contains a number"),
    "names": ("f_caps", "num", "number of capitalised words after the first (roughly: named entities)"),
    "agree": ("s_ans_agree", "num", "3B trial run: share of its 5 answers equal to its most common answer (0.2-1.0)"),
    "ret_agree": ("s_ret_agree", "num", "3B trial run: overlap of the paragraphs chosen by its 3 retrieval samples (0-1)"),
    "unknown": ("s_ans_unknown", "bool", "3B trial run: its most common answer says unknown / cannot tell"),
}
PRE_FEATS = ["yesno", "compare", "words", "digit", "names"]
POST_FEATS = PRE_FEATS + ["agree", "ret_agree", "unknown"]
_COND = re.compile(r"^\s*([a-z_]+)\s*(<=|>=|=|<|>)\s*([a-z0-9.]+)\s*$")


def v2_values(sig, q, names):
    out = {}
    for n in names:
        col, kind, _ = V2_FEATURES[n]
        v = sig.loc[q, col]
        v = 0.0 if v != v else float(v)  # NaN -> 0
        out[n] = ("yes" if v >= 0.5 else "no") if kind == "bool" else (round(v, 2) if n in ("agree", "ret_agree") else int(v))
    return out


def v2_text(vals):
    return "; ".join(f"{k}={v}" for k, v in vals.items())


def parse_rules_v2(text, routes, names, max_rules):
    """Keeps only well-formed 'IF <feature> <op> <value> [AND ...] THEN <route>' lines (at most max_rules-1)
    plus one 'OTHERWISE <route>'. Returns (rules as [(conditions, route)], default route, cleaned text, dropped count)."""
    rules, default, dropped = [], None, 0
    for line in text.splitlines():
        line = re.sub(r"^[\s\-*\d.)]+", "", line).strip()   # drop list markers at the START only (routes end in digits)
        if not line:
            continue
        m = re.match(r"^OTHERWISE\s+(?:use\s+)?([SL]\d-[SL]c?\d)", line, flags=re.I)
        if m:
            default = default or (m.group(1) if m.group(1) in routes else None)
            continue
        m = re.match(r"^IF\s+(.+?)\s+THEN\s+(?:use\s+)?([SL]\d-[SL]c?\d)\b", line, flags=re.I)
        if not m or m.group(2) not in routes:
            dropped += 1
            continue
        conds, ok = [], True
        for part in re.split(r"\s+AND\s+", m.group(1), flags=re.I):
            cm = _COND.match(part.lower())
            if not cm or cm.group(1) not in names:
                ok = False
                break
            conds.append((cm.group(1), cm.group(2), cm.group(3)))
        if ok and len(rules) < max_rules - 1:
            rules.append((conds, m.group(2)))
        else:
            dropped += 1
    default = default or routes[0]
    clean = "\n".join(f"{i + 1}. IF " + " AND ".join(f"{a} {o} {v}" for a, o, v in c) + f" THEN {r}" for i, (c, r) in enumerate(rules))
    clean += ("\n" if rules else "") + f"{len(rules) + 1}. OTHERWISE {default}"
    return rules, default, clean, dropped


def apply_rules_v2(rules, default, vals):
    def holds(a, o, v):
        x = vals[a]
        if isinstance(x, str) or v in ("yes", "no"):
            return str(x) == v if o == "=" else False
        try:
            v = float(v)
        except ValueError:
            return False
        return {"<": x < v, "<=": x <= v, ">": x > v, ">=": x >= v, "=": abs(x - v) < 1e-9}[o]
    for conds, route in rules:
        if all(holds(*c) for c in conds):
            return route
    return default


def switching_v2(args, T, E, sig, dev_q, test_q, dev_agg, add, run):
    routes = [r for r in args.routes if r in T.cfgs]
    missing = [r for r in args.routes if r not in T.cfgs]
    if missing:
        raise SystemExit(f"routes not in the results table: {missing}")
    probe_ok = PROBE in routes
    log(f"router v2 | routes {routes} | probe {PROBE} {'in menu' if probe_ok else 'NOT in menu: after-probe routers skipped'}")
    lams = args.lambdas
    # fixed baselines: every route alone, and the best single setup over the WHOLE table (AgentTTS-style)
    for r in routes:
        add(f"always {r} ({describe_cfg(r, T.models)})", "fixed", None, {q: r for q in test_q}, extra={"dev_choice": {q: r for q in dev_q}})
    for lam in lams:
        g = max(T.cfgs, key=lambda c: (dev_agg[c]["EM"] - lam * dev_agg[c]["flops"] / E.ref, -dev_agg[c]["flops"]))
        add("dev-optimal global config", "fixed", lam, {q: g for q in test_q}, extra={"dev_choice": {q: g for q in dev_q}})
    for name, pool in (("oracle over all setups (hindsight)", T.cfgs), ("oracle over the routes (hindsight)", routes)):
        orc = {}
        for q in test_q:
            ok = [c for c in pool if T.em["test"].loc[q, c] == 1]
            orc[q] = min(ok or pool, key=lambda c: T.fl["test"].loc[q, c])
        add(name, "oracle", None, orc)
    # cascade: probe first, escalate to one route when the 3B answers disagree
    if probe_ok:
        for lam in lams:
            best = None
            for tau in [0.2, 0.4, 0.6, 0.8, 1.01]:
                for esc in routes:
                    ch = {q: (PROBE if sig.loc[q, "s_ans_agree"] >= tau and not sig.loc[q, "s_ans_unknown"] else esc) for q in dev_q}
                    u = E.utility(E.score("dev", ch, probe=True), lam)
                    if best is None or u > best[0]:
                        best = (u, tau, esc, ch)
            _, tau, esc, dch = best
            tch = {q: (PROBE if sig.loc[q, "s_ans_agree"] >= tau and not sig.loc[q, "s_ans_unknown"] else esc) for q in test_q}
            add(f"cascade (agree<{tau:g} -> {esc})", "after probe", lam, tch, probe=True, extra={"dev_choice": dch})
    # classifier over the routes
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    settings = [("before running", PRE_FEATS, False)] + ([("after probe", POST_FEATS, True)] if probe_ok else [])
    for setting, names, probe in settings:
        cols = [V2_FEATURES[n][0] for n in names]
        Xd = StandardScaler().fit(sig.loc[dev_q, cols].fillna(0).values.astype(float))
        Xdev = Xd.transform(sig.loc[dev_q, cols].fillna(0).values.astype(float))
        Xtest = Xd.transform(sig.loc[test_q, cols].fillna(0).values.astype(float))
        for lam in lams:
            U = per_question_utility(T, "dev", lam, E.ref, routes, probe).loc[dev_q]
            y = U.idxmax(axis=1).values
            # cost-sensitive: weight = what a wrong route can cost on that question. Questions every route solves, or
            # none does, carry (almost) no weight, so they no longer drown the label in "cheapest route" (router v1 flaw).
            w = (U.max(axis=1) - U.min(axis=1)).values
            w = w / w.max() if w.max() > 0 else np.ones(len(w))

            def fit_predict(C, mask, X):
                live = mask & (w > 1e-3)
                if len(set(y[live])) < 2:
                    lab = y[live][0] if live.any() else routes[0]
                    return np.array([lab] * len(X))
                return LogisticRegression(C=C, max_iter=2000).fit(Xdev[live], y[live], sample_weight=w[live]).predict(X)
            best_c, best_u = 1.0, None
            for C in (0.1, 0.3, 1.0, 3.0):
                pred = []
                for i in range(len(dev_q)):
                    m = np.ones(len(dev_q), bool)
                    m[i] = False
                    pred.append(fit_predict(C, m, Xdev[i:i + 1])[0])
                u = E.utility(E.score("dev", dict(zip(dev_q, pred)), probe), lam)
                if best_u is None or u > best_u:
                    best_c, best_u, dch = C, u, dict(zip(dev_q, pred))
            tch = dict(zip(test_q, fit_predict(best_c, np.ones(len(dev_q), bool), Xtest)))
            add("classifier (cost-weighted)", setting, lam, tch, probe, extra={"dev_choice": dch})
    if not args.skip_nl:
        nl_rules_v2(args, T, E, sig, dev_q, test_q, dev_agg, add, run, routes, settings)


def nl_rules_v2(args, T, E, sig, dev_q, test_q, dev_agg, add, run, routes, settings):
    models = T.models
    agent = AgentLLM(args.backend, args.agent_path, args.thinking, args.max_model_len, args.agent_gpu_util, args.seed,
                     args.agent_max_tokens)
    reader = make_engine(args.backend, models["S"]["path"], size="S", max_model_len=4096, gpu_util=args.applier_gpu_util,
                         seed=args.seed)
    menu = "\n".join(f"- {c}: {describe_cfg(c, models)} | dev EM {dev_agg[c]['EM']:.2f} | cost {dev_agg[c]['flops'] / E.ref:.1f}x the cheapest setup"
                     for c in routes)
    log_rows = []

    def read_rules(clean, qids, names, seed):
        """The 3B model reads the rules and the question's feature values and names a route."""
        prompts, over = [], {}
        for q in qids:
            user = (f"Routing rules (check them in order; use the first rule whose conditions all hold):\n{clean}\n\n"
                    f"Feature values of this question: {v2_text(v2_values(sig, q, names))}\n\n"
                    f"Which route does the first matching rule give? Reply with the route name only, one of: {', '.join(routes)}.")
            prompts.append(reader.encode("You apply routing rules exactly as written.", user))
        outs = reader.generate(prompts, 1, 0.0, 12, [seed] * len(prompts), [{"kind": "route", "actions": routes}] * len(prompts))
        choice = {}
        for q, p, o in zip(qids, prompts, outs):
            m = re.search(r"[SL]\d-[SL]c?\d", o[0][0])
            choice[q] = m.group(0) if m and m.group(0) in routes else routes[0]
            over[q] = 2 * models["S"]["params"] * (len(p) + o[0][1])
        return choice, over

    for setting, names, probe in settings:
        feat_doc = "\n".join(f"- {n}: {V2_FEATURES[n][2]}" for n in names)
        for lam in args.nl_lambdas:
            U = per_question_utility(T, "dev", lam, E.ref, routes, probe)
            # route accuracy on dev by feature value: the aggregate evidence the rules should be built from
            rows = []
            for n in names:
                col, kind, _ = V2_FEATURES[n]
                v = sig.loc[dev_q, col].fillna(0)
                buckets = ([("yes", v >= 0.5), ("no", v < 0.5)] if kind == "bool" else
                           [(f"<= {v.median():g}", v <= v.median()), (f"> {v.median():g}", v > v.median())])
                for label, mask in buckets:
                    qq = list(v[mask].index)
                    if qq:
                        rows.append(f"| {n} {label} | {len(qq)} | " + " | ".join(f"{T.em['dev'].loc[qq, r].mean():.2f}" for r in routes) + " |")
            table = (f"| feature value | questions | " + " | ".join(f"EM {r}" for r in routes) + " |\n|---|---|" + "---|" * len(routes)
                     + "\n" + "\n".join(rows))
            mixed = [q for q in dev_q if 0 < T.em["dev"].loc[q, routes].sum() < len(routes)]
            rest = [q for q in dev_q if q not in set(mixed)]
            rng = random.Random(args.seed)
            rng.shuffle(mixed)
            rng.shuffle(rest)
            shown = (mixed + rest)[:args.max_evidence]
            evidence = "\n".join(f"- {v2_text(v2_values(sig, q, names))} | correct routes: "
                                 f"{', '.join(r for r in routes if T.em['dev'].loc[q, r] == 1) or 'none'} | best: {U.loc[q].idxmax()}"
                                 for q in shown)
            base_prompt = (
                f"We answer multi-hop questions with a retrieve-then-answer pipeline and can run each question through one of "
                f"these routes (cheapest first):\n{menu}\n\n"
                f"Goal: choose a route per NEW question to maximise accuracy minus {lam} x cost (cost in units of the cheapest setup"
                + (f"; with trial-run features the cheap trial {PROBE} is always paid, and choosing {PROBE} reuses it" if probe else "")
                + ").\n\nYou may only use these features, with these exact names:\n" + feat_doc +
                f"\n\nWrite the COMPLETE routing policy, at most {args.max_rules} lines, checked in order:\n"
                f"  IF <feature> <op> <value> [AND <feature> <op> <value>] THEN <route>\n  ...\n  OTHERWISE <route>\n"
                f"where <op> is one of < <= > >= = and yes/no features use '= yes' or '= no'. Nothing else counts: lines in "
                f"another form are ignored, and only the first {args.max_rules - 1} IF-lines are kept. Put the policy between "
                f"<rules> and </rules>.\n\nRoute accuracy on the {len(dev_q)} development questions, split by feature value:\n{table}\n\n"
                f"Individual development questions ({len(shown)} of {len(dev_q)}, those where routes disagree first):\n{evidence}")
            best, clean, feedback = None, None, ""
            for rnd in range(args.nl_rounds):
                user = base_prompt + (f"\n\nYour previous policy (it will be REPLACED by your new one):\n{clean}\n\nIts result on the "
                                      f"development questions: {feedback}\nWrite an improved complete policy." if clean else "")
                think, ans = agent.chat([("You write short, precise routing policies from evidence.", user, 31 * rnd + 7,
                                          {"kind": "rules", "actions": routes, "v2": True})])[0]
                m = re.search(r"<rules>(.*?)</rules>", ans, flags=re.S)
                rules, default, clean, dropped = parse_rules_v2(m.group(1) if m else ans, routes, names, args.max_rules)
                exact = {q: apply_rules_v2(rules, default, v2_values(sig, q, names)) for q in dev_q}
                read, rover = read_rules(clean, dev_q, names, args.seed)
                s_ex, s_rd = E.score("dev", exact, probe), E.score("dev", read, probe, rover)
                u = E.utility(s_ex, lam)
                fidelity = float(np.mean([exact[q] == read[q] for q in dev_q]))
                used = Counter(exact.values())
                feedback = (f"EM {s_ex['EM']:.3f}, cost {s_ex['rel_cost']:.2f}x, utility {u:.3f}; routes used "
                            + ", ".join(f"{r}: {used.get(r, 0)}" for r in routes)
                            + "; per route, share of its questions answered correctly: "
                            + ", ".join(f"{r} {np.mean([T.em['dev'].loc[q, r] for q in dev_q if exact[q] == r]):.2f}" for r in routes if used.get(r)))
                log_rows.append({"setting": setting, "lambda": lam, "round": rnd + 1, "rules_raw": ans, "rules": clean,
                                 "n_rules": len(rules) + 1, "dropped_lines": dropped, "thinking": think,
                                 "dev_EM_exact": s_ex["EM"], "dev_rel_cost_exact": s_ex["rel_cost"], "dev_utility_exact": u,
                                 "dev_EM_3B_reads": s_rd["EM"], "dev_rel_cost_3B_reads": s_rd["rel_cost"], "fidelity_3B_vs_exact": fidelity})
                log(f"  NL rules v2 | {setting} | lambda {lam} | round {rnd + 1}: {len(rules) + 1} rules ({dropped} lines dropped), "
                    f"dev EM {s_ex['EM']:.3f} at {s_ex['rel_cost']:.2f}x (exact); 3B follows them on {fidelity:.0%} of questions")
                if best is None or u > best[0]:
                    best = (u, rules, default, clean, exact, read, rover)
            _, rules, default, clean, dexact, dread, drover = best
            texact = {q: apply_rules_v2(rules, default, v2_values(sig, q, names)) for q in test_q}
            tread, trover = read_rules(clean, test_q, names, args.seed)
            add("NL rules v2 (applied exactly)", setting, lam, texact, probe, extra={"dev_choice": dexact})
            add("NL rules v2 (3B reads the rules)", setting, lam, tread, probe, trover, extra={"dev_choice": dread, "dev_overhead": drover})
            with open(os.path.join(run, f"rules_v2_{setting.replace(' ', '_')}_lam{lam}.txt"), "w", encoding="utf-8") as f:
                f.write(clean + "\n")
            log(f"  NL rules v2 | {setting}: 3B follows the chosen rules on "
                f"{np.mean([texact[q] == tread[q] for q in test_q]):.0%} of test questions")
    write_json(os.path.join(run, "nl_rules_log.json"), log_rows)
    write_json(os.path.join(run, "agent_usage.json"), {"calls": agent.calls, "input_tokens": agent.in_tok,
                                                        "output_tokens": agent.out_tok, "seconds": agent.seconds})


def report_switching(args, T, E, results, decisions, run):
    import pandas as pd
    df = pd.DataFrame(results)
    ref_rows = {r["lambda"]: r for r in results if r["policy"] == "dev-optimal global config"}
    cis = []
    for r in results:
        g = ref_rows.get(r["lambda"]) or next(iter(ref_rows.values()))
        d, lo, hi = bootstrap_diff(r["_em"], g["_em"])
        cis.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    df["EM_vs_global_95CI"] = cis
    out = df.drop(columns=["_em"])
    out.to_csv(os.path.join(run, "policies.csv"), index=False)
    rows = []
    for (p, s, lam), ch in decisions.items():
        for q, c in ch.items():
            rows.append({"policy": p, "setting": s, "lambda": lam, "qid": q, "config": c,
                         "EM": T.em["test"].loc[q, c], "question": T.sig.set_index("qid").loc[q, "question"]})
    pd.DataFrame(rows).to_csv(os.path.join(run, "test_decisions.csv"), index=False)
    # plot
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    kinds = [("fixed", "fixed config", SERIES[0]), ("before running", "router: question only", SERIES[1]),
             ("after probe", "router: after 3B probe", SERIES[2]), ("oracle", "oracle", MUTED)]
    for setting, label, col in kinds:
        sub = out[out.setting == setting]
        if sub.empty:
            continue
        ax.scatter(sub.test_TFLOPs, sub.test_EM, s=46, color=col, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
    real = out[out.setting != "oracle"].reset_index(drop=True)  # the oracle is a reference, not an achievable policy
    front = pareto(list(zip(real.test_TFLOPs, real.test_EM)))
    fr = real.iloc[front].sort_values("test_TFLOPs")
    ax.plot(fr.test_TFLOPs, fr.test_EM, color=INK2, ls="--", lw=1.2, zorder=2, label="Pareto frontier")
    for r in fr.itertuples():
        ax.annotate(r.policy.split(" (")[0], (r.test_TFLOPs, r.test_EM), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK2)
    n = len(T.em["test"])
    ax.set_xscale("log")
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.set_xlabel("TFLOPs per question on test (log, routing costs included)")
    ax.set_ylabel("EM on test")
    ax.set_title(f"Accuracy vs compute - {n} test questions (±1 SE ≈ {np.sqrt(0.25 / n):.2f} EM)", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(run, "policies_em_vs_flops.png"), dpi=150)
    plt.close(fig)
    md = [f"# Per-question switching (router v{getattr(args, 'router_version', 1)}) - {os.path.basename(run.rstrip('/'))}\n",
          f"Test questions: {n}. Costs include the probe (after-probe routers) and the rule-reading call (NL router). "
          f"'EM vs global' = test EM minus the dev-optimal global config at the same lambda, with a paired bootstrap 95% interval.\n",
          "| policy | setting | lambda | test EM | test F1 | TFLOPs/q | cost vs cheapest | est. GPU s/q | EM vs global [95% CI] | routes used |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in out.sort_values(["setting", "lambda", "test_TFLOPs"], na_position="first").iterrows():
        md.append(f"| {r.policy} | {r.setting} | {'' if pd.isna(r['lambda']) else r['lambda']} | {r.test_EM:.3f} | {r.test_F1:.3f} | "
                  f"{r.test_TFLOPs:.1f} | {r.rel_cost:.2f}x | {r.test_est_gpu_s:.2f} | {r.EM_vs_global_95CI} | "
                  f"{', '.join(f'{k}:{v}' for k, v in sorted(r.configs_used.items()))} |")
    with open(os.path.join(run, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


# ==================================================================================
# Live speed check: chosen policies, one question at a time
# ==================================================================================
def run_config_live(q, cfg, engines, seed):
    """Runs one configuration end to end for one question; returns (prediction, seconds by stage)."""
    r, rk, a, ak = parse_cfg(cfg)
    t = {}
    t0 = time.perf_counter()
    er = engines[r]
    p = er.encode(RET_SYSTEM, retrieval_prompt(q))
    outs = er.generate([p], rk, 0.9, 512, [seed], [{"kind": "ret", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"]), "q": q["question"]}])[0]
    sel = merge([o[0] for o in outs])
    t["retrieval"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    ea = engines[base(a)]
    cot = is_cot(a)
    p = ea.encode(QA_SYSTEM, qa_prompt(q["question"], selected_refs(q, sel), cot=cot))
    drafts = [o[0] for o in ea.generate([p], ak, 0.7, 512 if cot else 384, [seed], [{"kind": "qa", "gold": q["answer"], "hit": 0.5, "cot": cot}])[0]]
    t["answer"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = ea.encode(FUSER_SYSTEM, fuser_prompt(q["question"], drafts))
    text = ea.generate([p], 1, 0.3, 384, [seed], [{"kind": "fuse", "gold": q["answer"], "hit": 0.5}])[0][0][0]
    t["fuse"] = time.perf_counter() - t0
    return extract_answer(text), t


def cmd_speedcheck(args):
    import pandas as pd
    models = json.load(open(args.models_json))
    run = args.run_dir
    dec = pd.read_csv(os.path.join(args.switching_dir, "test_decisions.csv"))
    pol = pd.read_csv(os.path.join(args.switching_dir, "policies.csv"))
    qs = {q["qid"]: q for q in load_questions(args.data_dir, ("test",))}
    qids = list(qs)[:args.n_questions]
    # policies to time: fixed baselines + the best router per setting (highest test EM at lowest cost)
    pick = []
    for setting in ("fixed", "before running", "after probe"):
        sub = pol[(pol.setting == setting) & ~pol.policy.str.startswith("AgentTTS")].sort_values(["test_EM", "test_TFLOPs"], ascending=[False, True])
        pick += [tuple(x) for x in sub[["policy", "setting", "lambda"]].head(2 if setting == "fixed" else 1).values]
    engines = {"S": make_engine(args.backend, models["S"]["path"], size="S", max_model_len=8192, gpu_util=args.gpu_util_S, seed=0),
               "L": make_engine(args.backend, models["L"]["path"], size="L", max_model_len=8192, gpu_util=args.gpu_util_L, seed=0)}
    run_config_live(qs[qids[0]], PROBE, engines, 0)  # warm-up
    rows = []
    for policy, setting, lam in pick:
        d = dec[(dec.policy == policy) & (dec.setting == setting)]
        d = d[d["lambda"].isna()] if pd.isna(lam) else d[d["lambda"] == lam]
        route = dict(zip(d.qid, d.config))
        for qid in qids:
            cfg = route.get(qid)
            if cfg is None:
                continue
            total = {"retrieval": 0.0, "answer": 0.0, "fuse": 0.0}
            if setting == "after probe" and cfg != PROBE:
                _, tp = run_config_live(qs[qid], PROBE, engines, 1)
                for k in total:
                    total[k] += tp[k]
            pred, tt = run_config_live(qs[qid], cfg, engines, 1)
            for k in total:
                total[k] += tt[k]
            rows.append({"policy": policy, "setting": setting, "lambda": lam, "qid": qid, "config": cfg,
                         "seconds": sum(total.values()), **{f"s_{k}": v for k, v in total.items()},
                         "EM": em_f1(pred, qs[qid]["answer"])[0]})
        log(f"  timed {policy} ({setting})")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run, "speedcheck_per_question.csv"), index=False)
    s = df.groupby(["policy", "setting", "lambda"], dropna=False).agg(
        questions=("seconds", "size"), mean_s=("seconds", "mean"), median_s=("seconds", "median"),
        p95_s=("seconds", lambda x: float(np.percentile(x, 95))), EM=("EM", "mean")).reset_index()
    s.to_csv(os.path.join(run, "speedcheck.csv"), index=False)
    print(s.to_string(index=False))


# ==================================================================================
# Set 3 diagnostics: retrieval quality and answering ceiling
# ==================================================================================
def cmd_diagnose(args):
    """Tables for Set 3: (A) retrieval quality per retriever (plain / chain-of-thought), (B) accuracy of every
    answerer on every retrieval source incl. the two gold paragraphs, (C) paired differences with 95% intervals."""
    import pandas as pd
    models = json.load(open(args.models_json))
    run = args.run_dir
    qs = {q["qid"]: q for q in load_questions(args.data_dir)}
    step0 = os.path.join(run, "step0")
    stage_s = {}
    for fn in os.listdir(os.path.join(run, "stages")):
        st = json.load(open(os.path.join(run, "stages", fn)))
        if st.get("stage") == "retrieve":
            stage_s[st["model"]] = stage_s.get(st["model"], 0) + st["gen_s"]
    # ---- A: retrieval quality
    rq = []
    for fn in sorted(os.listdir(step0)):
        m = re.match(r"retrieval_([SL]c?)\.jsonl$", fn)
        if not m:
            continue
        tag = m.group(1)
        rows = read_jsonl(os.path.join(step0, fn))
        for k in (1, 3):
            for r in rows:
                q = qs.get(r["qid"])
                if q is None:
                    continue
                texts = [x.get("sel_text", x["text"]) for x in r["samples"]][:k]
                sel = [i - 1 for i in merge(texts)]
                g = set(q["ret_answers"])
                hit = len(g & set(sel))
                rq.append({"retriever": tag, "k": k, "qid": r["qid"], "split": q["split"],
                           "both": float(hit == len(g)), "one": float(0 < hit < len(g)), "none": float(hit == 0),
                           "Ret_F1": ret_f1(q["ret_answers"], sel), "n_selected": len(sel),
                           "out_tokens": np.mean([x["out_tokens"] for x in r["samples"][:k]]),
                           "final_line": np.mean([x.get("has_final_line", True) for x in r["samples"][:k]]),
                           "flops": 2 * models[base(tag)]["params"] * (r["prompt_tokens"] + sum(x["out_tokens"] for x in r["samples"][:k])),
                           "gpu_s_3samples": stage_s.get(tag, np.nan) / max(1, len(rows))})
    rq = pd.DataFrame(rq)
    # ---- B: answers
    ans = []
    for fn in sorted(os.listdir(step0)):
        if fn.startswith("answers_") and fn.endswith(".jsonl"):
            ans += read_jsonl(os.path.join(step0, fn))
    ad = pd.DataFrame(ans).drop_duplicates(["qid", "ret_model", "ret_k", "ans_model", "ans_k"])
    ad = ad[ad.ret_k == 1]
    ad["flops"] = [2 * models.get(base(r), {"params": 0})["params"] * (a + b) + 2 * models[base(m)]["params"] * (c + d + e + f)
                   for r, m, a, b, c, d, e, f in zip(ad.ret_model, ad.ans_model, ad.ret_prompt_tokens, ad.ret_output_tokens,
                                                     ad.ans_prompt_tokens, ad.ans_output_tokens, ad.fuse_prompt_tokens, ad.fuse_output_tokens)]
    order_r = [r for r in ["S", "Sc", "L", "Lc", "G"] if r in set(ad.ret_model)]
    names = {"S": "3B", "Sc": "3B + CoT", "L": "7B", "Lc": "7B + CoT", "G": "gold paragraphs (no retrieval)"}
    cols = [(m, k) for m in ["S", "Sc", "L", "Lc"] for k in (1, 5) if ((ad.ans_model == m) & (ad.ans_k == k)).any()]

    def cell(sub):
        return f"{100 * sub.EM.mean():.1f}%" if len(sub) else "-"
    md = [f"# Set 3 diagnostics - {os.path.basename(run.rstrip('/'))}\n",
          f"Questions: {len(qs)} (dev + test pooled unless marked). Retrieval uses 1 sample unless marked. "
          "Gold = the two paragraphs HotpotQA marks as needed, given directly to the answerer.\n"]
    if not rq.empty:
        md += ["\n## A. Retrieval quality (did the retriever hand over both gold paragraphs?)\n",
               "| retriever | samples merged | both gold found | one found | none found | paragraph F1 | paragraphs picked | output tokens/sample | TFLOPs/q | final line present |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for tag in [r for r in ["S", "Sc", "L", "Lc"] if r in set(rq.retriever)]:
            for k in (1, 3):
                x = rq[(rq.retriever == tag) & (rq.k == k)]
                md.append(f"| {names[tag]} | {k} | {100 * x.both.mean():.1f}% | {100 * x.one.mean():.1f}% | {100 * x.none.mean():.1f}% | "
                          f"{x.Ret_F1.mean():.3f} | {x.n_selected.mean():.1f} | {x.out_tokens.mean():.0f} | {x.flops.mean() / 1e12:.1f} | "
                          f"{100 * x.final_line.mean():.0f}% |")
    for split, lab in ((None, "dev + test, 400 questions"), ("test", "test only, 200 questions")):
        d = ad if split is None else ad[ad.split == split]
        md += [f"\n## B. Answer accuracy (EM) by retrieval source and answerer - {lab}\n",
               "| retrieval source | " + " | ".join(f"{names[m]} x{k}" for m, k in cols) + " |",
               "|---|" + "---|" * len(cols)]
        for r in order_r:
            md.append(f"| {names[r]} | " + " | ".join(cell(d[(d.ret_model == r) & (d.ans_model == m) & (d.ans_k == k)]) for m, k in cols) + " |")
    # ---- C: paired differences (pooled), 5 answers + fusion
    piv = ad[ad.ans_k == 5].pivot_table(index="qid", columns=["ret_model", "ans_model"], values="EM")

    def diff(a, b):
        if a not in piv.columns or b not in piv.columns:
            return None
        both = piv[[a, b]].dropna()
        if both.empty:
            return None
        d, lo, hi = bootstrap_diff(both[a].values, both[b].values)
        return f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] (n={len(both)})"
    comps = [("Does CoT retrieval help? (same answerer)", [(("Sc", m), ("S", m)) for m in ["S", "Sc", "L", "Lc"]]),
             ("3B + CoT retrieval vs plain 7B retrieval (same answerer)", [(("Sc", m), ("L", m)) for m in ["S", "Sc", "L", "Lc"]]),
             ("How much does perfect retrieval add over the 7B retriever? (same answerer)", [(("G", m), ("L", m)) for m in ["S", "Sc", "L", "Lc"]]),
             ("Answerer gaps with perfect retrieval (gold paragraphs)", [(("G", "L"), ("G", "S")), (("G", "Sc"), ("G", "S")),
                                                                         (("G", "L"), ("G", "Sc")), (("G", "Lc"), ("G", "L"))])]
    md.append("\n## C. Paired differences, 5 answers + fusion (EM points, 95% bootstrap interval, dev + test)\n")
    for title, pairs in comps:
        lines = []
        for a, b in pairs:
            dd = diff(a, b)
            if dd:
                lines.append(f"| {names[a[0]]} -> {names[a[1]]} vs {names[b[0]]} -> {names[b[1]]} | {dd} |")
        if lines:
            md += [f"\n**{title}**\n", "| comparison (retriever -> answerer) | difference |", "|---|---|"] + lines
    out = os.path.join(run, "diagnose")
    os.makedirs(out, exist_ok=True)
    if not rq.empty:
        rq.to_csv(os.path.join(out, "retrieval_quality_per_question.csv"), index=False)
    ad.to_csv(os.path.join(out, "answers_per_question.csv"), index=False)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


# ==================================================================================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build_data")
    b.add_argument("--hotpot_dev", required=True)
    b.add_argument("--out_dir", required=True)
    b.add_argument("--dev_n", type=int, default=40)
    b.add_argument("--test_n", type=int, default=100)
    b.add_argument("--n_candidates", type=int, default=20)
    b.add_argument("--seed", type=int, default=42)

    def common(p, gpu=True):
        p.add_argument("--run_dir", required=True)
        p.add_argument("--data_dir", required=True)
        p.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--time_limit_min", type=float, default=None)
        if gpu:
            p.add_argument("--models_json", required=True)
            p.add_argument("--max_model_len", type=int, default=8192)
            p.add_argument("--gpu_util", type=float, default=0.85)
            p.add_argument("--chunk", type=int, default=20)

    for name in ("retrieve", "answer"):
        p = sub.add_parser(name)
        common(p)
        p.add_argument("--model_key", choices=["S", "L"], required=True)
        if name == "retrieve":
            p.add_argument("--cot", action="store_true", help="Set 3: chain-of-thought retrieval (writes retrieval_Sc / retrieval_Lc)")
        if name == "answer":
            p.add_argument("--cot", action="store_true", help="Set 2b: chain-of-thought answers on the 1-sample retrievals")
            p.add_argument("--gold", action="store_true", help="Set 3: answer from the two gold paragraphs (no retrieval)")
            p.add_argument("--ret_tags", nargs="+", default=None, help="retrieval files to answer on (default S L)")
            p.add_argument("--ret_ks", type=int, nargs="+", default=None, help="retrieval samples to merge (default 1 3; CoT 1)")
            p.add_argument("--out_suffix", default="", help="suffix for the answers file, e.g. __gold")
    dg = sub.add_parser("diagnose")
    common(dg, gpu=False)
    dg.add_argument("--models_json", required=True)
    t = sub.add_parser("table")
    common(t, gpu=False)
    t.add_argument("--models_json", required=True)

    def agent_args(p):
        p.add_argument("--table_dir", required=True)
        p.add_argument("--agent_path", required=True)
        p.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode for the agent")
        p.add_argument("--agent_max_tokens", type=int, default=1536)
        p.add_argument("--max_model_len", type=int, default=16384)

    a1 = sub.add_parser("agenttts")
    common(a1, gpu=False)
    agent_args(a1)
    a1.add_argument("--gpu_util", type=float, default=0.85)
    a1.add_argument("--budget_quantiles", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    a1.add_argument("--seeds", type=int, default=3)
    a1.add_argument("--iterations", type=int, default=4)
    a1.add_argument("--batch_size", type=int, default=2)
    a1.add_argument("--max_trials", type=int, default=8)
    a1.add_argument("--allow_cot", action="store_true", help="Set 2b control: let the agent choose chain-of-thought answering")

    a2 = sub.add_parser("switching")
    common(a2, gpu=False)
    agent_args(a2)
    a2.set_defaults(max_model_len=32768)
    a2.add_argument("--models_json", required=True)
    a2.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.01, 0.03, 0.1])
    a2.add_argument("--nl_lambdas", type=float, nargs="+", default=[0.03])
    a2.add_argument("--nl_rounds", type=int, default=5)
    a2.add_argument("--max_rules", type=int, default=8)
    a2.add_argument("--agent_gpu_util", type=float, default=0.5)
    a2.add_argument("--applier_gpu_util", type=float, default=0.3)
    a2.add_argument("--skip_nl", action="store_true")
    a2.add_argument("--max_evidence", type=int, default=90, help="dev questions shown to the rule writer")
    a2.add_argument("--router_version", type=int, choices=[1, 2], default=1, help="1 = Set 1 router; 2 = Set 2 router")
    a2.add_argument("--routes", nargs="+", default=["S1-S5", "S1-L5", "L1-L5"], help="router v2 route menu")
    a2.add_argument("--script1_dir", default=None)

    sc = sub.add_parser("speedcheck")
    common(sc, gpu=False)
    sc.add_argument("--models_json", required=True)
    sc.add_argument("--switching_dir", required=True)
    sc.add_argument("--n_questions", type=int, default=10)
    sc.add_argument("--gpu_util_S", type=float, default=0.3)
    sc.add_argument("--gpu_util_L", type=float, default=0.5)

    args = ap.parse_args()
    if hasattr(args, "run_dir"):
        os.makedirs(os.path.join(args.run_dir, "stages"), exist_ok=True)
    {"build_data": cmd_build_data, "retrieve": cmd_retrieve, "answer": cmd_answer, "table": cmd_table,
     "agenttts": cmd_agenttts, "switching": cmd_switching, "speedcheck": cmd_speedcheck, "diagnose": cmd_diagnose}[args.cmd](args)


if __name__ == "__main__":
    main()
