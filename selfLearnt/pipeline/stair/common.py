"""Shared pieces: prompts, metrics, compute accounting, file helpers, BM25, voting and ensembles.

Two families of prompts:
  * AgentTTS's own prompts (workstreams 1 and 2), copied verbatim from github.com/FairyFali/AgentTTS
    (code_others/archon_hotpotqa.py and archon/completions/components/prompts.py, Apache-2.0);
  * our prompts (workstreams 3-5): shortlist as reference numbers, short answers ('Answer: ...').
"""
import json
import os
import random
import re
import string
import time
from collections import Counter

import numpy as np

# ==================================================================================
# AgentTTS prompts (verbatim, including their spacing)
# ==================================================================================
ATTS_RET_SYSTEM = "You are a helpful assistant that return the beneficial references to the query."
ATTS_QA_SYSTEM = "You are a helpful assistant."
ATTS_FUSE_SYSTEM = "You are a helpful assistant who fuses multiple answers"


def build_query(ex):
    """AgentTTS's query text: 'Query: <question>References: \\n[1] ...' (no space before 'References', as in their code)."""
    q = "Query: " + ex["question"] + "References: "
    for i, ref in enumerate(ex["ctxs_candidate"]):
        q += f"\n[{i + 1}] " + ref
    return q


def atts_ret_prompt(ex):
    return (f"You have been provided with a query and a set of references. "
            f"Your task is to select references that comprehensively cover the information needed to answer the query. "
            f"Your response should include the original query and a list of selected references in the following format: 'Query: [query]\nSelected references: [reference number] [reference details]'."
            f"Here is the query and references: \n {build_query(ex)}.")


def atts_merge(candidates):
    """AgentTTS's Merge: reference numbers voted across the retrieval samples; keep as many as the most common shortlist length."""
    ref_ids, lengths = [], []
    for c in candidates:
        nums = re.findall(r'\[(\d+)\]', c)
        ref_ids.extend(nums)
        lengths.append(len(nums))
    if ref_ids:
        L = Counter(lengths).most_common()[0][0]
        return [int(e[0]) for e in Counter(ref_ids).most_common()[0:L]]
    return [1]


def atts_selected_refs(ex, ids):
    """AgentTTS's generator input: references whose line contains '[id]', in reference order, text after the first ']'
    (their code: reference.split(']')[1]; a line may be added twice if it matches twice)."""
    lines = build_query(ex).split("References:")[1].strip().split("\n")
    out = []
    for line in lines:
        for rid in ids:
            if f"[{rid}]" in line:
                out.append(line.split(']')[1].strip())
    return out


def atts_qa_prompt(question, refs):
    qr = "Query: " + question + "\nReferences:\n"
    for i, r in enumerate(refs):
        qr += f"[{i + 1}] " + r + "\n"
    return (f"You will be provided with a set of references related to the query. "
            f"Your task is to answer the query using the provided references. If the references are insufficient, supplement your response with your own knowledge. "
            f"Here is the query and references: \n {qr}.")


def atts_fuse_prompt(ex, responses):
    """Archon's fuser prompt; the 'query' is the whole first message, i.e. the question with all candidate references."""
    query = build_query(ex)
    prompt = f"You have been provided with a set of responses from various open-source models to the latest user query, which is {query}.\
            Your task is to synthesize these responses into a single, high-quality response. \
            It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. \
            Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. \
            Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n"
    prompt += f"Once again, the query is: {query}\n"
    prompt += "Responses from models:"
    for i, r in enumerate(responses):
        prompt += f"\n{i + 1}. {r}"
    return prompt


# ==================================================================================
# Our prompts (workstreams 3-5)
# ==================================================================================
RET_SYSTEM = ATTS_RET_SYSTEM
QA_SYSTEM = ATTS_QA_SYSTEM
SHORT_ANSWER = ("\nEnd your response with a final line in exactly this form: 'Answer: <short answer>' - "
                "the answer only (a name, date, number or short phrase; 'yes' or 'no' for yes/no questions).")
COT_INSTRUCTION = ("\nThink step by step before answering: first write out the fact from the references that answers "
                   "the first part of the question, then use it to find the fact that answers the second part, "
                   "then state the answer.")
CONF_ANS = ("\nAfter the Answer line, add one last line in exactly this form: 'Confidence: <0-100>' - how sure you are, in percent, "
            "that the answer is correct.")
DIRECT_RET = ("Select only the references needed to answer the query. Answer with one line in exactly this form, listing only the "
              "reference numbers you select:\nSelected references: [n] [n]")
DIRECT_ANS = ("\nAnswer with one line in exactly this form: 'Answer: <short answer>' - the answer only (a name, date, number or "
              "short phrase; 'yes' or 'no' for yes/no questions).")


def direct_ret_prompt(ex):
    """Prompt of the fine-tuned retrievers (training input and every later shortlist)."""
    return ("You have been provided with a query and a set of references. "
            "Your task is to select references that comprehensively cover the information needed to answer the query. "
            f"Here is the query and references: \n {build_query(ex)}.\n" + DIRECT_RET)


def ret_target(ex):
    return "Selected references: " + " ".join(f"[{i + 1}]" for i in sorted(ex["ret_answers"]))


def _qr(question, refs):
    qr = "Query: " + question + "\nReferences:\n"
    for i, r in enumerate(refs):
        qr += f"[{i + 1}] " + r + "\n"
    return qr


def cot_qa_prompt(question, refs):
    """Untrained answerers (workstream 3): step by step, short answer, stated confidence."""
    return (f"You will be provided with a set of references related to the query. "
            f"Your task is to answer the query using the provided references. If the references are insufficient, supplement your response with your own knowledge. "
            f"Here is the query and references: \n {_qr(question, refs)}." + COT_INSTRUCTION + SHORT_ANSWER + CONF_ANS)


def direct_qa_prompt(question, refs):
    """Fine-tuned answerers (workstreams 4-5): training input and inference prompt; target 'Answer: <gold>'."""
    return (f"You will be provided with a set of references related to the query. "
            f"Your task is to answer the query using the provided references. If the references are insufficient, supplement your response with your own knowledge. "
            f"Here is the query and references: \n {_qr(question, refs)}." + DIRECT_ANS)


def ans_target(ex):
    return "Answer: " + ex["answer"]


def selected_refs(ex, sel):
    refs = ex["ctxs_candidate"]
    return [refs[i - 1] for i in sel if 1 <= i <= len(refs)]


def selection_text(text):
    i = (text or "").lower().rfind("selected references:")
    return text[i:] if i >= 0 else (text or "")


def picks_of(text, n_refs):
    out = []
    for x in re.findall(r"\[(\d+)\]", text or ""):
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
    s = (s or "").lower()
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


def cover_em(response, gold):
    """AgentTTS-style accuracy: the normalised gold answer appears (as whole words) in the normalised response.
    AgentTTS reports 'EM' on free-text fused responses (0.742 on HotpotQA); its metric file is not in the repository, and
    containment is the reading consistent with that number. Applied to every workstream's final text."""
    g = normalize_answer(gold)
    return float(bool(g) and (" " + g + " ") in (" " + normalize_answer(response) + " "))


def lenient(pred, gold):
    """Our corrected label (development notebooks): exact match, or token F1 >= 0.5, or the gold answer contained. Yes/no exact."""
    em, f1 = em_f1(pred, gold)
    if em:
        return 1.0
    p, g = normalize_answer(pred), normalize_answer(gold)
    if g in ("yes", "no", "noanswer") or p in ("yes", "no", "noanswer", ""):
        return 0.0
    return float(f1 >= 0.5 or (" " + g + " ") in (" " + p + " "))


def scores(answer, gold):
    """The team's scores of one final answer (stair_common.score: the standard lenient ruler, official EM / F1).
    acc = lenient, the headline. Gold may be one answer or a list of aliases."""
    from stair_common.score import score
    s = score(answer or "", gold)
    return {"acc": float(s["lenient"]), "lenient": float(s["lenient"]), "EM": float(s["em"]), "F1": float(s["f1"])}


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
# Compute
# ==================================================================================
def atts_budget(S, M, Np, Nd, M_small=3e9, Np2=128, Nd2=64):
    """AgentTTS's normalised budget of S samples from a model with M parameters (unit = one 3B pass on a 128/64-token task)."""
    alpha, beta1, beta2, beta3 = M / M_small, Np / Nd, Np2 / Nd2, Np / Np2
    return max(0.0, beta2 * ((alpha * beta3 / beta1) * (beta1 + S) - 1))


def units(params_b, prompt_tokens, out_tokens):
    """AgentTTS units of one call (stair_common.cost.units): 2 x params x tokens / 1,152 GFLOPs, nominal size in billions."""
    from stair_common.cost import units as _u
    return _u(float(params_b), int(prompt_tokens), int(out_tokens))


TABLE4_LENGTHS = {"retrieve": (2048, 128), "answer": (256, 64)}   # AgentTTS Table 4: prompt / generated tokens per sample


def price(cfg, v, stage, samples=1):
    """The HotpotQA golden set's price of one call (stair_common.cost, common 0.3.0): AgentTTS's Table 5 entry for a listed
    model and one sample, else cost.budget() at Table 4's lengths (S samples from one prompt are one call). A fine-tuned
    variant is priced as its base model; the router / rules are not charged."""
    from stair_common.cost import TABLE5_HOTPOTQA, budget
    m = cfg["models"][v.split(":")[0]]
    k = m["label"].replace(" ", "-")
    if samples == 1 and k in TABLE5_HOTPOTQA[stage]:
        return float(TABLE5_HOTPOTQA[stage][k])
    p, g = TABLE4_LENGTHS[stage]
    return float(budget(float(m["params_b"]), p, g * samples))


def price_measured(cfg, v, prompt_tokens, out_tokens):
    """A call Table 5 does not cover (AgentTTS's fusion): cost.budget() at its measured tokens."""
    from stair_common.cost import budget
    return float(budget(float(cfg["models"][v.split(":")[0]]["params_b"]), int(prompt_tokens), int(out_tokens)))


def flops(params, prompt_tokens, out_tokens, n=1, shared_prompt=True):
    """2 x params x tokens. n samples of one prompt: the prompt is counted once when shared (prefix cache), else n times."""
    return 2.0 * params * ((prompt_tokens if shared_prompt else n * prompt_tokens) + out_tokens)


# ==================================================================================
# Files
# ==================================================================================
def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                try:
                    out.append(json.loads(l))
                except json.JSONDecodeError:          # a line cut by a job that stopped mid-write
                    continue
    return out


def append_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + f".{os.getpid()}.tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    os.replace(path + f".{os.getpid()}.tmp", path)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + f".{os.getpid()}.tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)
    os.replace(path + f".{os.getpid()}.tmp", path)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def lp_(x):
    return -1.0 if x is None else float(x)


def seed_of(*parts):
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode()) % 999983


# ==================================================================================
# BM25 (candidate passages)
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


def shuffle_rng(seed):
    return random.Random(seed)
