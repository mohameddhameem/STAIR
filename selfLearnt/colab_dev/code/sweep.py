#!/usr/bin/env python3
"""
Set 4 ('Set_1__3-modality-size'): broad sweep of retriever-answerer pairings with three sizes per role.

  retrievers  q3 = Qwen2.5-3B, q7 = Qwen2.5-7B, q32 = Qwen2.5-32B (4-bit AWQ)
  answerers   l3 = Llama-3.2-3B, l8 = Llama-3.1-8B, q32 = Qwen2.5-32B (4-bit AWQ)
  variables   CoT (off/on, both steps together) x per-try selection size (silent / 3 / 5)
              x final shortlist (vote = paper's merge / top2 / all = union)

Commands (each runs in its own process; all resumable):
  status     CPU  what is done and what is pending
  retrieve   GPU  one retriever, all 6 prompt variants, 3 tries each
  answer     GPU  one answerer with or without CoT, on every final shortlist of the matching retrievals
  plan       GPU  AgentTTS baseline per condition + per-question switching (rules written by the planner model)
  report     CPU  per-question JSONL, setup tables, Excel workbook, charts, summary.md

Reuses prompts, merge rule, metrics, budget formula and the AgentTTS search from ssw.py (Sets 1-3), unchanged.
"""
import argparse
import itertools
import json
import os
import random
import re
import sys
import time
import zlib
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S  # noqa: E402

RETRIEVERS = ["q3", "q7", "q32"]
ANSWERERS = ["l3", "l8", "q32"]
COTS = [0, 1]
PICKS = ["silent", "3", "5"]          # per-try selection size the retriever is told to pick ("silent" = paper prompt)
MERGES = ["vote", "top2", "all"]      # final shortlist rule ("vote" = paper's merge, "all" = union of the tries)
RET_KS = [1, 3]
ANS_KS = [1, 5]
N_TRIES = 3
MAIN_RK, MAIN_AK = 3, 5
SMALL = {"q3", "l3"}
LABEL = {"q3": "Qwen 3B", "q7": "Qwen 7B", "q32": "Qwen 32B", "l3": "Llama 3B", "l8": "Llama 8B"}
SIZE_WORD = {"q3": "small", "q7": "medium", "q32": "large", "l3": "small", "l8": "medium"}
PICK_LABEL = {"silent": "paper default", "3": "3", "5": "5"}
MERGE_LABEL = {"vote": "paper default (vote)", "top2": "2", "all": "all"}
log = S.log


# ==================================================================================
# Engines
# ==================================================================================
class Engine(S.VLLMEngine):
    def __init__(self, path, max_model_len=8192, gpu_util=0.88, seed=0):
        from vllm import LLM
        kw = dict(model=path, max_model_len=max_model_len, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, seed=seed)
        if "awq" in path.lower():
            try:
                self.llm = LLM(quantization="awq", **kw)
            except (TypeError, ValueError):
                self.llm = LLM(**kw)
        else:
            self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len, self.thinking = max_model_len, None


    def generate(self, prompts, n, temperature, max_tokens, seeds, meta=None):
        """Like ssw.VLLMEngine.generate, plus the mean log-probability the model gave to its own output tokens."""
        from vllm import SamplingParams
        sps = [SamplingParams(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                              max_tokens=max_tokens, seed=s, logprobs=1) for s in seeds]
        outs = self.llm.generate([{"prompt_token_ids": p} for p in prompts], sps, use_tqdm=False)
        return [[(c.text, len(c.token_ids), mean_logprob(c)) for c in o.outputs] for o in outs]


def mean_logprob(c):
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


class Mock4(S.MockEngine):
    """Fake model for dry runs: adds a token log-probability and, when asked for, a 'Confidence: N' line."""

    def generate(self, prompts, n, temperature, max_tokens, seeds, meta=None):
        outs = super().generate(prompts, n, temperature, max_tokens, seeds, meta)
        res = []
        for o, s, m in zip(outs, seeds, meta or [{}] * len(outs)):
            rng = random.Random(s + 17)
            row = []
            for t, k in o:
                good = bool(m.get("gold")) and ("Answer: " + str(m.get("gold"))) in t
                if m.get("conf"):
                    c = rng.randint(70, 98) if good else rng.randint(35, 90)
                    t += f"\nConfidence: {c}"
                row.append((t, k, -rng.uniform(0.05, 0.4) if good else -rng.uniform(0.1, 0.9)))
            res.append(row)
        return res


def make_engine(backend, models, key, **kw):
    if backend == "mock":
        return Mock4(size="S" if key in SMALL else "L")
    return Engine(models[key]["path"], **kw)


# ==================================================================================
# Confidence: stated by the model ('Confidence: N') and measured (token probability, agreement between tries)
# ==================================================================================
CONF_RET = ("\nAfter your selection, add one last line in exactly this form: 'Confidence: <0-100>' - how sure you are, in percent, "
            "that the selected references contain everything needed to answer the query.")
CONF_ANS = ("\nAfter the Answer line, add one last line in exactly this form: 'Confidence: <0-100>' - how sure you are, in percent, "
            "that the answer is correct.")
_CONF = re.compile(r"\(?\bconfidence\b\s*(?:level|score)?\s*[:=]?\s*(\d{1,3})(?:\.\d+)?\s*%?\)?\.?", flags=re.I)


def stated_conf(text):
    """Last 'Confidence: N' in the text as 0-1, or None when the model gave none."""
    hits = _CONF.findall(text or "")
    if not hits:
        return None
    return min(100, int(hits[-1])) / 100.0


def strip_conf(text):
    return _CONF.sub("", text or "").rstrip()


def answer_of(text):
    """Short answer, with any confidence statement removed first so it cannot leak into the answer."""
    return S.extract_answer(strip_conf(text))


def token_prob(lp):
    return None if lp is None else float(np.exp(lp))


def asks_conf(cot, pick):
    """The paper-default retrieval prompt (no CoT, no number given) is left untouched: measured confidence only."""
    return bool(cot) or pick != "silent"


class Planner:
    """Batched chat for the planner model (AgentTTS agent and rule writer)."""

    def __init__(self, backend, models, key, max_model_len, max_tokens, seed=0):
        self.eng = make_engine(backend, models, key, max_model_len=max_model_len, seed=seed)
        self.max_model_len, self.max_tokens = max_model_len, max_tokens
        self.calls, self.in_tok, self.out_tok, self.seconds = 0, 0, 0, 0.0

    def chat(self, items):
        t = time.time()
        prompts = [self.eng.encode(s, u) for s, u, _, _ in items]
        limit = self.max_model_len - self.max_tokens - 8
        if any(len(p) > limit for p in prompts) and not isinstance(self.eng, S.MockEngine):
            raise SystemExit(f"planner prompt of {max(len(p) for p in prompts)} tokens exceeds the context ({limit})")
        outs = self.eng.generate(prompts, 1, 0.6, self.max_tokens, [x[2] for x in items], [x[3] for x in items])
        self.seconds += time.time() - t
        self.calls += len(items)
        self.in_tok += sum(len(p) for p in prompts)
        self.out_tok += sum(o[0][1] for o in outs)
        return [S.split_thinking(o[0][0]) for o in outs]


# ==================================================================================
# Retrieval: prompts, per-try selections, final shortlists
# ==================================================================================
def vid(model, cot, pick):
    return f"{model}_c{cot}_p{pick}"


def ret_prompt(q, cot, pick):
    if not cot:
        p = S.retrieval_prompt(q)                      # the paper's prompt, unchanged when pick == "silent"
        if pick != "silent":
            anchor = "needed to answer the query. "
            assert anchor in p
            p = p.replace(anchor, anchor + f"Select exactly {pick} references. ", 1) + CONF_RET
        return p
    p = S.retrieval_prompt(q, cot=True)                # Set 3 chain-of-thought retrieval prompt when pick == "silent"
    if pick != "silent":
        assert p.endswith(S.RET_COT_INSTRUCTION)
        n = int(pick)
        p = p[:-len(S.RET_COT_INSTRUCTION)] + (
            "Think step by step before selecting: (1) state what the first part of the query asks and which "
            "reference number contains that fact; (2) using that fact, state what the second part needs and which "
            f"reference number contains it. Select exactly {n} references in total: those from steps 1 and 2 plus the "
            f"next most useful ones. Then end with one final line in exactly this form, listing only the {n} reference "
            "numbers you select:\nSelected references: " + " ".join(["[n]"] * n))
    return p + CONF_RET


def sel_text(sample, cot):
    return S.selection_text(sample["text"]) if cot else sample["text"]


def picks_of(text, n_refs):
    out = []
    for x in re.findall(r"\[(\d+)\]", text):
        i = int(x)
        if 1 <= i <= n_refs and i not in out:
            out.append(i)
    return out


def shortlists(row, cot, n_refs):
    """Final shortlists (1-based passage numbers) for every (tries merged, rule)."""
    texts = [sel_text(s, cot) for s in row["samples"]]
    per_try = [picks_of(t, n_refs) for t in texts]
    out = {}
    for rk in RET_KS:
        vote = tuple(i for i in S.merge(texts[:rk]) if 1 <= i <= n_refs) or (1,)     # paper's merge, as in Sets 1-3
        votes, first = Counter(), {}
        for t, p in enumerate(per_try[:rk]):
            for pos, i in enumerate(p):
                votes[i] += 1
                first.setdefault(i, (t, pos))
        ranked = sorted(votes, key=lambda i: (-votes[i], first[i]))
        out[(rk, "vote")] = vote
        out[(rk, "top2")] = tuple(sorted(ranked[:2])) or (1,)
        out[(rk, "all")] = tuple(sorted(ranked)) or (1,)
    return out


def load_models(path):
    return json.load(open(path))


CODE_VERSION = "2: confidence lines in prompts, token probabilities recorded"


def check_version(run):
    """One run folder = one version of the prompts. Refuses to mix outputs made with different prompt versions."""
    p = os.path.join(run, "code_version.txt")
    step = os.path.join(run, "step")
    have = os.path.isdir(step) and any(f.endswith(".jsonl") for f in os.listdir(step))
    old = open(p).read().strip() if os.path.exists(p) else ("1: before confidence tracking" if have else None)
    if old is None:
        os.makedirs(run, exist_ok=True)
        with open(p, "w") as f:
            f.write(CODE_VERSION)
    elif old != CODE_VERSION:
        raise SystemExit(f"This run folder was made with an earlier version of the prompts ({old!r}); the code is now {CODE_VERSION!r}. "
                         "Mixing them would make the setups incomparable. Set NEW_RUN = True in the notebook to start a fresh run folder.")


def ret_path(run, v):
    return os.path.join(run, "step", f"retrieval_{v}.jsonl")


def ans_path(run, model, cot):
    return os.path.join(run, "step", f"answers_{model}_c{cot}.jsonl")


def cmd_retrieve(args):
    models = load_models(args.models_json)
    check_version(args.run_dir)
    m = models[args.model]
    run = args.run_dir
    os.makedirs(os.path.join(run, "step"), exist_ok=True)
    qs = S.load_questions(args.data_dir)
    variants = [(c, p) for c in COTS for p in PICKS]
    pending = [(c, p) for c, p in variants if len(S.read_jsonl(ret_path(run, vid(args.model, c, p)))) < len(qs)]
    if not pending:
        log(f"retrieve {args.model}: all 6 variants already complete")
        return
    st = S.Status(run, f"retrieve_{args.model}", len(variants) * len(qs), args.time_limit_min)
    t0 = time.time()
    with S.GPUMonitor() as mon:
        eng = make_engine(args.backend, models, args.model, max_model_len=args.max_model_len, gpu_util=args.gpu_util)
        load_s, t1, flops, n_done, stopped = time.time() - t0, time.time(), 0.0, 0, False
        for cot, pick in pending:
            v = vid(args.model, cot, pick)
            done = {r["qid"] for r in S.read_jsonl(ret_path(run, v))}
            todo = [q for q in qs if q["qid"] not in done]
            max_tok = 1024 if (cot or pick != "silent") else 512
            log(f"retrieve {v} | {m['id']} | {len(todo)} of {len(qs)} questions to do | {N_TRIES} tries")
            for c in range(0, len(todo), args.chunk):
                chunk = todo[c:c + args.chunk]
                prompts = [eng.encode(S.RET_SYSTEM, ret_prompt(q, cot, pick)) for q in chunk]
                meta = [{"kind": "ret", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"]), "q": q["question"],
                         "conf": asks_conf(cot, pick)} for q in chunk]
                seeds = [zlib.crc32((v + q["qid"]).encode()) % 999983 for q in chunk]
                outs = eng.generate(prompts, N_TRIES, 0.9, max_tok, seeds, meta)
                rows = []
                for q, p, o in zip(chunk, prompts, outs):
                    rows.append({"qid": q["qid"], "split": q["split"], "variant": v, "prompt_tokens": len(p),
                                 "samples": [{"text": t, "out_tokens": k, "hit_cap": k >= max_tok,
                                              "has_final_line": "selected references:" in t.lower(),
                                              "stated_conf": stated_conf(t) if asks_conf(cot, pick) else None,
                                              "token_prob": token_prob(lp)} for t, k, lp in o]})
                    flops += 2 * m["params"] * (len(p) + sum(x[1] for x in o))
                S.append_jsonl(ret_path(run, v), rows)
                n_done += len(chunk)
                st.update(n_done, v)
                if st.over_limit():
                    stopped = True
                    break
            log(f"  {v}: {len(S.read_jsonl(ret_path(run, v)))}/{len(qs)}")
            if stopped:
                log("time limit reached - stopping cleanly (run again to continue)")
                break
        gen_s = time.time() - t1
    S.write_json(os.path.join(run, "stages", f"retrieve_{args.model}_{int(time.time())}.json"),
                 {"stage": "retrieve", "model": args.model, "gpu": S.gpu_name(), "load_s": load_s, "gen_s": gen_s,
                  "questions_x_variants": n_done, "flops": flops, **mon.summary()})
    log(f"retrieve {args.model} done: {gen_s:.0f}s generation + {load_s:.0f}s load")


# ==================================================================================
# Answering: every distinct final shortlist of the retrievals with the same CoT setting
# ==================================================================================
def needed_shortlists(run, qs, cot):
    """qid -> sorted list of distinct final shortlists over all complete-so-far retrieval rows with this CoT setting."""
    need = {q["qid"]: set() for q in qs}
    n_refs = {q["qid"]: len(q["ctxs_candidate"]) for q in qs}
    for r in RETRIEVERS:
        for pick in PICKS:
            for row in S.read_jsonl(ret_path(run, vid(r, cot, pick))):
                if row["qid"] in need:
                    need[row["qid"]].update(shortlists(row, cot, n_refs[row["qid"]]).values())
    return {k: sorted(v) for k, v in need.items()}


def cmd_answer(args):
    models = load_models(args.models_json)
    check_version(args.run_dir)
    m = models[args.model]
    run, cot = args.run_dir, int(args.cot)
    qs = S.load_questions(args.data_dir)
    need = needed_shortlists(run, qs, cot)
    path = ans_path(run, args.model, cot)
    done = {(r["qid"], tuple(r["sel"])) for r in S.read_jsonl(path)}
    todo = [(q, [s for s in need[q["qid"]] if (q["qid"], s) not in done]) for q in qs]
    todo = [(q, sels) for q, sels in todo if sels]
    total = sum(len(v) for v in need.values())
    n_todo = sum(len(s) for _, s in todo)
    log(f"answer {args.model} c{cot} | {m['id']} | {total} distinct shortlists, {n_todo} to do | {max(ANS_KS)} answers + fusion")
    if not todo:
        return
    max_tok = 512 if cot else 384
    st = S.Status(run, f"answer_{args.model}_c{cot}", total, args.time_limit_min)
    t0 = time.time()
    with S.GPUMonitor() as mon:
        eng = make_engine(args.backend, models, args.model, max_model_len=args.max_model_len, gpu_util=args.gpu_util)
        load_s, t1, flops, n_done = time.time() - t0, time.time(), 0.0, total - n_todo
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            keys, prompts, meta = [], [], []
            for q, sels in chunk:
                gold = set(q["ret_answers"])
                for sel in sels:
                    keys.append((q, sel))
                    prompts.append(eng.encode(S.QA_SYSTEM, S.qa_prompt(q["question"], S.selected_refs(q, sel), cot=bool(cot)) + CONF_ANS))
                    hit = len({i - 1 for i in sel} & gold) / max(1, len(gold))
                    meta.append({"kind": "qa", "gold": q["answer"], "hit": hit, "cot": bool(cot), "conf": True})
            seeds = [zlib.crc32(repr((args.model, cot, q["qid"], sel)).encode()) % 999983 for q, sel in keys]
            drafts = eng.generate(prompts, max(ANS_KS), 0.7, max_tok, seeds, meta)
            f_prompts, f_meta, f_index = [], [], []
            for i, ((q, sel), d) in enumerate(zip(keys, drafts)):
                for ak in ANS_KS:
                    texts = [strip_conf(x[0]) for x in d[:ak]]          # the fuser sees the drafts without their confidence lines
                    f_prompts.append(eng.encode(S.FUSER_SYSTEM, S.fuser_prompt(q["question"], texts) + CONF_ANS))
                    hit = float(np.mean([S.em_f1(S.extract_answer(t), q["answer"])[0] for t in texts]))
                    f_meta.append({"kind": "fuse", "gold": q["answer"], "hit": hit, "cot": bool(cot), "conf": True})
                    f_index.append((i, ak))
            fused = eng.generate(f_prompts, 1, 0.3, 384, [s + 7 * ak for s, (_, ak) in zip([seeds[i] for i, _ in f_index], f_index)], f_meta)
            rows = {}
            for (i, ak), fp, fo in zip(f_index, f_prompts, fused):
                q, sel = keys[i]
                text, ftok, flp = fo[0]
                short = answer_of(text)
                em, f1 = S.em_f1(short, q["answer"])
                r = rows.setdefault(i, {"qid": q["qid"], "sel": list(sel), "model": args.model, "cot": cot,
                                        "prompt_tokens": len(prompts[i]), "out_tokens": [x[1] for x in drafts[i]],
                                        "draft_answers": [answer_of(x[0]) for x in drafts[i]],
                                        "draft_conf": [stated_conf(x[0]) for x in drafts[i]],
                                        "draft_token_prob": [token_prob(x[2]) for x in drafts[i]], "fused": {}})
                r["fused"][str(ak)] = {"prompt_tokens": len(fp), "out_tokens": ftok, "prediction": short, "EM": em, "F1": f1,
                                       "cover_EM": S.cover_em(strip_conf(text), q["answer"]),
                                       "stated_conf": stated_conf(text), "token_prob": token_prob(flp)}
                flops += 2 * m["params"] * (len(fp) + ftok)
            for i, r in rows.items():
                flops += 2 * m["params"] * (r["prompt_tokens"] + sum(r["out_tokens"]))
            S.append_jsonl(path, [rows[i] for i in sorted(rows)])
            n_done += len(keys)
            st.update(n_done)
            log(f"  {n_done}/{total} shortlists answered")
            if st.over_limit():
                log("time limit reached - stopping cleanly (run again to continue)")
                break
        gen_s = time.time() - t1
    S.write_json(os.path.join(run, "stages", f"answer_{args.model}_c{cot}_{int(time.time())}.json"),
                 {"stage": "answer", "model": args.model, "cot": cot, "gpu": S.gpu_name(), "load_s": load_s, "gen_s": gen_s,
                  "flops": flops, **mon.summary()})
    log(f"answer {args.model} c{cot} done: {gen_s:.0f}s generation + {load_s:.0f}s load")


# ==================================================================================
# Status: what is complete, what is pending
# ==================================================================================
def sweep_status(run, data_dir):
    qs = S.load_questions(data_dir)
    n = len(qs)
    ret = {vid(r, c, p): len(S.read_jsonl(ret_path(run, vid(r, c, p)))) for r in RETRIEVERS for c in COTS for p in PICKS}
    stages = []
    for r in RETRIEVERS:
        d = sum(ret[vid(r, c, p)] for c in COTS for p in PICKS)
        stages.append({"stage": "retrieve", "model": r, "cot": None, "done": d, "total": 6 * n, "complete": d >= 6 * n})
    all_ret = all(s["complete"] for s in stages)
    for a in ANSWERERS:
        for c in COTS:
            need = needed_shortlists(run, qs, c)
            done = {(x["qid"], tuple(x["sel"])) for x in S.read_jsonl(ans_path(run, a, c))}
            tot = sum(len(v) for v in need.values())
            d = sum((q, s) in done for q, v in need.items() for s in v)
            stages.append({"stage": "answer", "model": a, "cot": c, "done": d, "total": tot,
                           "complete": all_ret and tot > 0 and d >= tot})
    return {"questions": n, "stages": stages, "complete": all(s["complete"] for s in stages),
            "plan_done": os.path.exists(os.path.join(run, "plan", "agenttts.jsonl"))}


def cmd_status(args):
    st = sweep_status(args.run_dir, args.data_dir)
    S.write_json(os.path.join(args.run_dir, "sweep_status.json"), st)
    print(f"{'stage':10s} {'model':10s} {'CoT':4s} {'done':>8s} {'total':>8s}  state")
    for s in st["stages"]:
        note = "complete" if s["complete"] else ("waiting for retrievals" if s["stage"] == "answer" and s["done"] >= s["total"] else "PENDING")
        print(f"{s['stage']:10s} {LABEL[s['model']]:10s} {'-' if s['cot'] is None else ('on' if s['cot'] else 'off'):4s} "
              f"{s['done']:8d} {s['total']:8d}  {note}")
    print("SWEEP COMPLETE" if st["complete"] else "sweep not complete yet - run the notebook again to continue")


# ==================================================================================
# Per-question table
# ==================================================================================
def nanmean(v):
    v = [x for x in v if x is not None and x == x]
    return float(np.mean(v)) if v else None


def setup_id(r, a, cot, pick, merge, rk, ak):
    return f"{r}>{a}|c{cot}|p{pick}|{merge}|r{rk}a{ak}"


def build_per_question(run, data_dir, models):
    import pandas as pd
    qs = {q["qid"]: q for q in S.load_questions(data_dir)}
    P = {k: v["params"] for k, v in models.items()}
    rows = []
    for cot in COTS:
        ans = {a: {(x["qid"], tuple(x["sel"])): x for x in S.read_jsonl(ans_path(run, a, cot))} for a in ANSWERERS}
        for r in RETRIEVERS:
            for pick in PICKS:
                for row in S.read_jsonl(ret_path(run, vid(r, cot, pick))):
                    q = qs.get(row["qid"])
                    if q is None:
                        continue
                    gold = set(q["ret_answers"])
                    n_refs = len(q["ctxs_candidate"])
                    per_try = [set(picks_of(sel_text(s, cot), n_refs)) for s in row["samples"]]
                    pairs = list(itertools.combinations(per_try, 2))
                    agree = float(np.mean([len(x & y) / len(x | y) if x | y else 1.0 for x, y in pairs])) if pairs else 1.0
                    r_conf = [s.get("stated_conf") for s in row["samples"]]
                    r_tp = [s.get("token_prob") for s in row["samples"]]
                    for (rk, merge), sel in shortlists(row, cot, n_refs).items():
                        sel0 = {i - 1 for i in sel}
                        hit = len(sel0 & gold)
                        found = "both" if hit == len(gold) else ("one" if hit else "none")
                        r_out = sum(s["out_tokens"] for s in row["samples"][:rk])
                        f_ret = 2 * P[r] * (row["prompt_tokens"] + r_out)
                        for a in ANSWERERS:
                            x = ans[a].get((row["qid"], sel))
                            if x is None:
                                continue
                            for ak in ANS_KS:
                                fz = x["fused"][str(ak)]
                                a_out = sum(x["out_tokens"][:ak])
                                f_ans = 2 * P[a] * (x["prompt_tokens"] + a_out)
                                f_fuse = 2 * P[a] * (fz["prompt_tokens"] + fz["out_tokens"])
                                drafts = [S.normalize_answer(d) for d in x.get("draft_answers", [])[:ak]]
                                rows.append({
                                    "qid": row["qid"], "split": q["split"], "setup": setup_id(r, a, cot, pick, merge, rk, ak),
                                    "retriever": r, "answerer": a, "cot": cot, "pick": pick, "merge": merge, "rk": rk, "ak": ak,
                                    "EM": fz["EM"], "F1": fz["F1"], "prediction": fz["prediction"], "gold": q["answer"],
                                    "found": found, "n_passages": len(sel), "Ret_F1": S.ret_f1(q["ret_answers"], sel0),
                                    "ret_agree": agree, "ans_agree": (Counter(drafts).most_common(1)[0][1] / len(drafts)) if drafts else 1.0,
                                    "ret_conf": nanmean(r_conf[:rk]), "ret_token_prob": nanmean(r_tp[:rk]),
                                    "ans_conf": fz.get("stated_conf"), "ans_token_prob": fz.get("token_prob"),
                                    "draft_conf": nanmean(x.get("draft_conf", [])[:ak]),
                                    "ret_prompt_tokens": row["prompt_tokens"], "ret_out_tokens": r_out,
                                    "ans_prompt_tokens": x["prompt_tokens"], "ans_out_tokens": a_out,
                                    "flops_ret": f_ret, "flops_ans": f_ans, "flops_fuse": f_fuse, "flops": f_ret + f_ans + f_fuse})
    return pd.DataFrame(rows)


def aggregate(df, by):
    g = df.groupby(by, sort=False)
    out = g.agg(accuracy=("EM", "mean"), F1=("F1", "mean"), TFLOPs=("flops", lambda x: x.mean() / 1e12),
                found_both=("found", lambda x: (x == "both").mean()), found_none=("found", lambda x: (x == "none").mean()),
                n_passages=("n_passages", "mean"), Ret_F1=("Ret_F1", "mean"), n=("EM", "size"),
                ans_conf=("ans_conf", lambda x: __import__("pandas").to_numeric(x, errors="coerce").mean()),
                ret_conf=("ret_conf", lambda x: __import__("pandas").to_numeric(x, errors="coerce").mean())).reset_index()
    fb = df[df.found == "both"].groupby(by, sort=False).EM.mean().rename("acc_when_found")
    nf = df[df.found != "both"].groupby(by, sort=False).EM.mean().rename("acc_when_not_found")
    return out.merge(fb.reset_index(), on=by, how="left").merge(nf.reset_index(), on=by, how="left")


KEYS = ["setup", "retriever", "answerer", "cot", "pick", "merge", "rk", "ak"]


def setups_table(df):
    """One row per setup: pooled dev+test, plus dev and test accuracy and compute."""
    t = aggregate(df, KEYS)
    for sp in ("dev", "test"):
        s = df[df.split == sp].groupby("setup").agg(**{f"{sp}_accuracy": ("EM", "mean"), f"{sp}_TFLOPs": ("flops", lambda x: x.mean() / 1e12)})
        t = t.merge(s.reset_index(), on="setup", how="left")
    main = t[(t.rk == MAIN_RK) & (t.ak == MAIN_AK)]
    ref = main.TFLOPs.min() if len(main) else t.TFLOPs.min()
    t["compute_x_cheapest"] = t.TFLOPs / ref
    return t, ref


# ==================================================================================
# Report: JSONL, Excel, charts, summary
# ==================================================================================
def pretty_main(t):
    m = t[(t.rk == MAIN_RK) & (t.ak == MAIN_AK)].copy()
    m["order"] = [(RETRIEVERS.index(r), ANSWERERS.index(a), c, PICKS.index(p), MERGES.index(mg))
                  for r, a, c, p, mg in zip(m.retriever, m.answerer, m.cot, m.pick, m["merge"])]
    m = m.sort_values("order")
    return __import__("pandas").DataFrame({
        "Retriever-answerer setup": [f"{LABEL[r]} - {LABEL[a]}" for r, a in zip(m.retriever, m.answerer)],
        "Sizes": [f"{SIZE_WORD[r]}-{SIZE_WORD[a]}" for r, a in zip(m.retriever, m.answerer)],
        "CoT": ["Y" if c else "N" for c in m.cot],
        "Per-attempt shortlist": [PICK_LABEL[p] for p in m.pick],
        "Final shortlist": [MERGE_LABEL[x] for x in m["merge"]],
        "Overall accuracy": m.accuracy.values, "Overall compute (x cheapest)": m.compute_x_cheapest.values,
        "Retrieval found both": m.found_both.values, "Accuracy when found": m.acc_when_found.values,
        "Accuracy when not found": m.acc_when_not_found.values, "Passages in final shortlist": m.n_passages.values,
        "Stated answer confidence": m.ans_conf.values, "Confidence minus accuracy": (m.ans_conf - m.accuracy).values,
        "Stated retrieval confidence": m.ret_conf.values,
        "Dev accuracy": m.dev_accuracy.values, "Test accuracy": m.test_accuracy.values, "F1": m.F1.values,
        "TFLOPs per question": m.TFLOPs.values, "Questions": m.n.values, "setup id": m.setup.values})


def retrieval_quality(run, data_dir, models):
    import pandas as pd
    qs = {q["qid"]: q for q in S.load_questions(data_dir)}
    rows = []
    for r in RETRIEVERS:
        for cot in COTS:
            for pick in PICKS:
                data = S.read_jsonl(ret_path(run, vid(r, cot, pick)))
                if not data:
                    continue
                per_try_n, one_pick, cap, final = [], [], [], []
                acc = {(rk, mg): [] for rk in RET_KS for mg in MERGES}
                for row in data:
                    q = qs[row["qid"]]
                    gold, n_refs = set(q["ret_answers"]), len(q["ctxs_candidate"])
                    for s in row["samples"]:
                        k = len(picks_of(sel_text(s, cot), n_refs))
                        per_try_n.append(k)
                        one_pick.append(k <= 1)
                        cap.append(s.get("hit_cap", False))
                        final.append(s.get("has_final_line", True))
                    for key, sel in shortlists(row, cot, n_refs).items():
                        hit = len({i - 1 for i in sel} & gold)
                        acc[key].append((hit == len(gold), hit == 0, len(sel)))
                out_tok = np.mean([s["out_tokens"] for row in data for s in row["samples"]])
                base = {"Retriever": LABEL[r], "CoT": "Y" if cot else "N", "Per-attempt shortlist": PICK_LABEL[pick],
                        "Passages picked per try": float(np.mean(per_try_n)), "Tries that picked 0 or 1": float(np.mean(one_pick)),
                        "Output tokens per try": float(out_tok), "Tries cut off by length cap": float(np.mean(cap)),
                        "Final line present (CoT)": float(np.mean(final)) if cot else None,
                        "TFLOPs per question (3 tries)": float(np.mean([2 * models[r]["params"] * (row["prompt_tokens"] + sum(s["out_tokens"] for s in row["samples"])) for row in data]) / 1e12)}
                for (rk, mg), v in acc.items():
                    a = np.array(v, float)
                    rows.append({**base, "Tries merged": rk, "Final shortlist": MERGE_LABEL[mg], "Found both": a[:, 0].mean(),
                                 "Found none": a[:, 1].mean(), "Passages in final shortlist": a[:, 2].mean(), "Questions": len(a)})
    return pd.DataFrame(rows)


def main_effects(t):
    import pandas as pd
    m = t[(t.rk == MAIN_RK) & (t.ak == MAIN_AK)]
    rows = []
    for var, name, lab in (("retriever", "Retriever", LABEL), ("answerer", "Answerer", LABEL), ("cot", "CoT", {0: "N", 1: "Y"}),
                           ("pick", "Per-attempt shortlist", PICK_LABEL), ("merge", "Final shortlist", MERGE_LABEL)):
        for lv, g in m.groupby(var, sort=False):
            rows.append({"Variable": name, "Level": lab[lv], "Mean accuracy": g.accuracy.mean(), "Best accuracy": g.accuracy.max(),
                         "Mean found both": g.found_both.mean(), "Mean compute (x cheapest)": g.compute_x_cheapest.mean(), "Setups": len(g)})
    return pd.DataFrame(rows)


BINS = [(0.0, 0.5, "below 50"), (0.5, 0.7, "50-69"), (0.7, 0.9, "70-89"), (0.9, 1.01, "90-100")]


def confidence_tables(run, data_dir):
    """How well each confidence signal separates right from wrong decisions.
    Returns (summary, calibration bins) as DataFrames; one decision = one distinct answer, or one retrieval try."""
    import pandas as pd
    qs = {q["qid"]: q for q in S.load_questions(data_dir)}
    items = []
    for a in ANSWERERS:
        for cot in COTS:
            for x in S.read_jsonl(ans_path(run, a, cot)):
                if x["qid"] not in qs or str(MAIN_AK) not in x["fused"]:
                    continue
                f = x["fused"][str(MAIN_AK)]
                d = [S.normalize_answer(z) for z in x.get("draft_answers", [])]
                base = {"step": "answer", "model": LABEL[a], "CoT": "Y" if cot else "N", "correct": float(f["EM"])}
                for sig, v in (("stated by the model (final answer)", f.get("stated_conf")),
                               ("stated by the model (mean of the 5 drafts)", nanmean(x.get("draft_conf", []))),
                               ("token probability (final answer)", f.get("token_prob")),
                               ("agreement of the 5 answers", Counter(d).most_common(1)[0][1] / len(d) if d else None)):
                    items.append({**base, "signal": sig, "value": v})
    for r in RETRIEVERS:
        for cot in COTS:
            for pick in PICKS:
                for row in S.read_jsonl(ret_path(run, vid(r, cot, pick))):
                    q = qs.get(row["qid"])
                    if q is None:
                        continue
                    gold, n_refs = set(q["ret_answers"]), len(q["ctxs_candidate"])
                    base = {"step": "retrieval", "model": LABEL[r], "CoT": "Y" if cot else "N"}
                    tries = []
                    for sm in row["samples"]:
                        p = {i - 1 for i in picks_of(sel_text(sm, cot), n_refs)}
                        tries.append(p)
                        ok = float(gold <= p)
                        items.append({**base, "correct": ok, "signal": "stated by the model (per try)", "value": sm.get("stated_conf")})
                        items.append({**base, "correct": ok, "signal": "token probability (per try)", "value": sm.get("token_prob")})
                    pairs = list(itertools.combinations(tries, 2))
                    agree = float(np.mean([len(x & y) / len(x | y) if x | y else 1.0 for x, y in pairs])) if pairs else None
                    final = {i - 1 for i in shortlists(row, cot, n_refs)[(MAIN_RK, "vote")]}
                    items.append({**base, "correct": float(gold <= final), "signal": "agreement of the 3 tries", "value": agree})
    d = pd.DataFrame(items)
    if d.empty:
        return d, d
    summ, cal = [], []
    for (step, model, cot, sig), g in d.groupby(["step", "model", "CoT", "signal"], sort=False):
        have = g[g.value.notna()]
        rec = {"Step": step, "Model": model, "CoT": cot, "Confidence signal": sig, "Decisions": len(g),
               "Share with a value": len(have) / len(g) if len(g) else 0.0,
               "Share correct": g.correct.mean()}
        if len(have) >= 10:
            v, c = have.value.astype(float).values, have.correct.values
            rec["Mean confidence"] = float(v.mean())
            if sig.startswith("stated"):
                rec["Overconfidence (confidence minus correct)"] = float(v.mean() - c.mean())
            if 0 < c.sum() < len(c):
                from sklearn.metrics import roc_auc_score
                rec["Separation (AUC, 0.5 = useless, 1 = perfect)"] = float(roc_auc_score(c, v))
            med = float(np.median(v))
            hi, lo = have[have.value.astype(float) > med], have[have.value.astype(float) <= med]
            if len(hi) and len(lo):
                rec["Correct when confidence above median"] = hi.correct.mean()
                rec["Correct when confidence at or below median"] = lo.correct.mean()
            if sig.startswith("stated"):
                for lo_, hi_, lab in BINS:
                    b = have[(have.value.astype(float) >= lo_) & (have.value.astype(float) < hi_)]
                    if len(b):
                        cal.append({"Step": step, "Model": model, "CoT": cot, "Confidence signal": sig, "Stated confidence": lab,
                                    "Decisions": len(b), "Mean stated confidence": float(b.value.astype(float).mean()), "Share correct": b.correct.mean()})
        summ.append(rec)
    return pd.DataFrame(summ), pd.DataFrame(cal)


def confidence_chart(cal, out):
    if cal is None or len(cal) == 0:
        return
    plt = S._plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, step, sig, title in ((axes[0], "answer", "stated by the model (final answer)", "Answerer: stated confidence vs share of answers correct"),
                                 (axes[1], "retrieval", "stated by the model (per try)", "Retriever: stated confidence vs share of tries with both passages")):
        d = cal[(cal.Step == step) & (cal["Confidence signal"] == sig)]
        ax.plot([0, 1], [0, 1], color=S.MUTED, ls=":", lw=1.2, zorder=1)
        ax.text(0.03, 0.08, "below the dotted line = overconfident", fontsize=7.5, color=S.MUTED)
        for i, model in enumerate(d.Model.unique()):
            for cot, ls, mk in (("N", "--", "o"), ("Y", "-", "o")):
                g = d[(d.Model == model) & (d.CoT == cot)].sort_values("Mean stated confidence")
                if len(g):
                    ax.plot(g["Mean stated confidence"], g["Share correct"], ls=ls, marker=mk, ms=4 + 5 * np.sqrt(g.Decisions / max(1, d.Decisions.max())).values.mean(),
                            color=S.SERIES[i % len(S.SERIES)], label=f"{model}, CoT {'on' if cot == 'Y' else 'off'}", zorder=3,
                            markerfacecolor=S.SERIES[i % len(S.SERIES)] if cot == "Y" else S.SURFACE)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        fmt = __import__("matplotlib.ticker").ticker.PercentFormatter(1.0)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xlabel("Stated confidence")
        ax.set_title(title, loc="left", color=S.INK, fontsize=9.5)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("Share actually correct")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "5_confidence_calibration.png"), dpi=150)
    plt.close(fig)


def make_charts(t, rq, out):
    plt = S._plt()
    os.makedirs(out, exist_ok=True)
    m = t[(t.rk == MAIN_RK) & (t.ak == MAIN_AK)]
    col = dict(zip(RETRIEVERS, S.SERIES[:3]))
    mark = dict(zip(ANSWERERS, ["o", "s", "^"]))
    # 1. accuracy vs compute
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for r in RETRIEVERS:
        for a in ANSWERERS:
            for c in COTS:
                g = m[(m.retriever == r) & (m.answerer == a) & (m.cot == c)]
                ax.scatter(g.compute_x_cheapest, g.accuracy, s=42, marker=mark[a], zorder=3, linewidth=1.4,
                           facecolor=col[r] if c else S.SURFACE, edgecolor=col[r])
    pts = list(zip(m.compute_x_cheapest, m.accuracy))
    fr = m.iloc[S.pareto(pts)].sort_values("compute_x_cheapest")
    ax.plot(fr.compute_x_cheapest, fr.accuracy, color=S.INK2, ls="--", lw=1.2, zorder=2)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=col[r], label=f"retriever {LABEL[r]}") for r in RETRIEVERS]
    h += [Line2D([], [], marker=mark[a], ls="", color=S.INK2, label=f"answerer {LABEL[a]}") for a in ANSWERERS]
    h += [Line2D([], [], marker="o", ls="", markerfacecolor=S.INK2, color=S.INK2, label="CoT on (filled)"),
          Line2D([], [], marker="o", ls="", markerfacecolor=S.SURFACE, color=S.INK2, label="CoT off (hollow)"),
          Line2D([], [], ls="--", color=S.INK2, label="best for its cost")]
    ax.legend(handles=h, frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_xscale("log")
    ax.set_xlabel("Compute per question (x cheapest setup, log scale)")
    ax.set_ylabel("Accuracy (exact match)")
    ax.set_title(f"Accuracy vs compute: {len(m)} setups, {int(m.n.max()) if len(m) else 0} questions", loc="left", color=S.INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "1_accuracy_vs_compute.png"), dpi=150)
    plt.close(fig)
    # 2. retrieval found both
    if len(rq):
        q3 = rq[rq["Tries merged"] == MAIN_RK]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
        groups = [(p, mg) for p in PICKS for mg in MERGES]
        for ax, c in zip(axes, ["N", "Y"]):
            for j, r in enumerate(RETRIEVERS):
                ys = []
                for p, mg in groups:
                    g = q3[(q3.Retriever == LABEL[r]) & (q3.CoT == c) & (q3["Per-attempt shortlist"] == PICK_LABEL[p]) & (q3["Final shortlist"] == MERGE_LABEL[mg])]
                    ys.append(g["Found both"].mean() if len(g) else np.nan)
                ax.bar(np.arange(len(groups)) + (j - 1) * 0.27, ys, 0.25, color=col[r], label=LABEL[r], zorder=3)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([f"pick {PICK_LABEL[p].replace('paper default', 'default')}\nkeep {mg}" for p, mg in groups], fontsize=7.5)
            ax.set_title(f"CoT {'on' if c == 'Y' else 'off'}", loc="left", color=S.INK)
            ax.yaxis.set_major_formatter(__import__("matplotlib.ticker").ticker.PercentFormatter(1.0))
        axes[0].set_ylabel("Final shortlist contains both needed passages")
        axes[0].legend(frameon=False, fontsize=8, title="retriever", title_fontsize=8)
        fig.suptitle("Retrieval success by retriever, per-attempt shortlist and final shortlist (3 tries)", x=0.01, ha="left", color=S.INK)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "2_retrieval_found_both.png"), dpi=150)
        plt.close(fig)
    # 3. heatmap: pairs x conditions
    conds = [(c, p, mg) for c in COTS for p in PICKS for mg in MERGES]
    pairs = [(r, a) for r in RETRIEVERS for a in ANSWERERS]
    Z = np.full((len(pairs), len(conds)), np.nan)
    for i, (r, a) in enumerate(pairs):
        for j, (c, p, mg) in enumerate(conds):
            g = m[(m.retriever == r) & (m.answerer == a) & (m.cot == c) & (m.pick == p) & (m["merge"] == mg)]
            if len(g):
                Z[i, j] = g.accuracy.iloc[0]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.grid(False)
    im = ax.imshow(Z, aspect="auto", cmap="Blues", vmin=np.nanmin(Z) if np.isfinite(Z).any() else 0, vmax=np.nanmax(Z) if np.isfinite(Z).any() else 1)
    hi = np.nanmax(Z) if np.isfinite(Z).any() else 1
    lo = np.nanmin(Z) if np.isfinite(Z).any() else 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(Z[i, j]):
                ax.text(j, i, f"{100 * Z[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if (Z[i, j] - lo) > 0.6 * (hi - lo) else S.INK)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels([f"{LABEL[r]} - {LABEL[a]}" for r, a in pairs], fontsize=8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([f"{'CoT' if c else 'no CoT'}\npick {PICK_LABEL[p].replace('paper default', 'default')}\nkeep {mg}" for c, p, mg in conds], fontsize=7)
    ax.axvline(8.5, color=S.SURFACE, lw=3)
    ax.set_title("Accuracy (%) by retriever-answerer pair and condition", loc="left", color=S.INK)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "3_accuracy_heatmap.png"), dpi=150)
    plt.close(fig)
    # 4. main effects
    me = main_effects(t)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = [f"{v}: {lv}" for v, lv in zip(me.Variable, me.Level)]
    y = np.arange(len(me))[::-1]
    ax.hlines(y, me["Mean accuracy"], me["Best accuracy"], color=S.GRID, lw=3, zorder=2)
    ax.scatter(me["Mean accuracy"], y, color=S.SERIES[0], zorder=3, label="mean over setups")
    ax.scatter(me["Best accuracy"], y, color=S.SERIES[1], zorder=3, marker="D", label="best setup")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.xaxis.set_major_formatter(__import__("matplotlib.ticker").ticker.PercentFormatter(1.0))
    ax.set_xlabel("Accuracy (exact match)")
    ax.set_title("Effect of each variable (mean and best accuracy across the setups that use that level)", loc="left", color=S.INK, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "4_effect_of_each_variable.png"), dpi=150)
    plt.close(fig)


def write_excel(path, sheets):
    import pandas as pd
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, d in sheets.items():
            if d is None or len(d) == 0:
                continue
            d.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "B2"
            for j, c in enumerate(d.columns, 1):
                letter = ws.cell(row=1, column=j).column_letter
                width = max(len(str(c)), int(d[c].astype(str).str.len().quantile(0.9)) if len(d) else 0)
                ws.column_dimensions[letter].width = min(60, max(9, width + 2))
                low = str(c).lower()
                if d[c].dtype.kind == "f":
                    pct = any(w in low for w in ("accuracy", "found", "tries that", "cut off", "final line", "share", "correct", "confidence"))
                    for cell in ws[letter][1:]:
                        cell.number_format = "0.0%" if pct else "0.00"


def cmd_report(args):
    import pandas as pd
    models = load_models(args.models_json)
    run = args.run_dir
    out = os.path.join(run, "results")
    os.makedirs(out, exist_ok=True)
    st = sweep_status(run, args.data_dir)
    df = build_per_question(run, args.data_dir, models)
    if df.empty:
        print("no finished retrieval+answer pairs yet - nothing to report")
        return
    df.to_json(os.path.join(out, "per_question.jsonl"), orient="records", lines=True, force_ascii=False)
    t, ref = setups_table(df)
    t.to_json(os.path.join(out, "setups.jsonl"), orient="records", lines=True)
    main = pretty_main(t)
    rq = retrieval_quality(run, args.data_dir, models)
    rq.to_json(os.path.join(out, "retrieval_quality.jsonl"), orient="records", lines=True)
    me = main_effects(t)
    conf, cal = confidence_tables(run, args.data_dir)
    if len(conf):
        conf.to_json(os.path.join(out, "confidence.jsonl"), orient="records", lines=True)
    sheets = {"Main table": main, "All setups": t.drop(columns=["setup"]).assign(setup=t.setup), "Retrieval quality": rq, "By variable": me,
              "Confidence": conf, "Calibration": cal}
    plan = os.path.join(run, "plan")
    for name, fn in (("AgentTTS baseline", "agenttts.jsonl"), ("Switching", "switching.jsonl"), ("Rules", "rules.jsonl"),
                     ("Router decisions", "decisions.jsonl")):
        rows = S.read_jsonl(os.path.join(plan, fn))
        if rows:
            sheets[name] = pd.DataFrame(rows)
    times = []
    for fn in sorted(os.listdir(os.path.join(run, "stages"))) if os.path.isdir(os.path.join(run, "stages")) else []:
        s = json.load(open(os.path.join(run, "stages", fn)))
        times.append({"stage": s.get("stage"), "model": LABEL.get(s.get("model"), s.get("model")), "CoT": s.get("cot"),
                      "generation seconds": s.get("gen_s"), "load seconds": s.get("load_s"), "gpu": s.get("gpu")})
    sheets["Stage times"] = pd.DataFrame(times)
    sheets["Read me"] = pd.DataFrame({"Item": ["Questions", "Main table", "Compute", "Per-attempt shortlist", "Final shortlist", "CoT", "Found both",
                                               "Confidence", "Per-question data", "Sweep complete"],
                                      "Meaning": [f"{st['questions']} (dev + test pooled unless a column says dev or test)",
                                                  f"{MAIN_RK} retrieval tries, {MAIN_AK} answer tries + fusion. 'All setups' also has 1-try versions.",
                                                  f"FLOPs = 2 x parameters x tokens, relative to the cheapest main-table setup ({ref:.1f} TFLOPs per question)",
                                                  "how many passages the retriever is told to pick in each try; 'paper default' = no number given",
                                                  "'paper default (vote)' = paper's merge; '2' = two most-voted passages; 'all' = every passage any try picked",
                                                  "Y = chain-of-thought in both retrieval and answering; N = neither",
                                                  "the final shortlist contains both passages needed to answer",
                                                  "stated = the 'Confidence: N' line the model wrote (not asked in the paper-default retrieval prompt); token probability = "
                                                  "average probability the model gave its own output; agreement = how much repeated tries agree. Sheets 'Confidence' and 'Calibration'.",
                                                  "results/per_question.jsonl (one row per question x setup)", str(st["complete"])]})
    write_excel(os.path.join(out, "set4_results.xlsx"), sheets)
    make_charts(t, rq, os.path.join(out, "charts"))
    confidence_chart(cal, os.path.join(out, "charts"))
    # summary.md
    pct = lambda x: "-" if x != x else f"{100 * x:.1f}%"
    md = [f"# Set 4 sweep - {os.path.basename(run.rstrip('/'))}\n",
          f"Questions: {st['questions']} (dev + test pooled). Main rows use {MAIN_RK} retrieval tries and {MAIN_AK} answer tries + fusion. "
          f"Sweep complete: **{st['complete']}** ({len(main)} of {len(RETRIEVERS) * len(ANSWERERS) * len(COTS) * len(PICKS) * len(MERGES)} main setups have results).\n",
          "\n## Best setups by accuracy\n",
          "| Retriever-answerer | CoT | Per-attempt shortlist | Final shortlist | Accuracy | Compute (x cheapest) | Found both | Acc. when found | Acc. when not found |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in main.sort_values("Overall accuracy", ascending=False).head(12).iterrows():
        md.append(f"| {r['Retriever-answerer setup']} | {r['CoT']} | {r['Per-attempt shortlist']} | {r['Final shortlist']} | {pct(r['Overall accuracy'])} | "
                  f"{r['Overall compute (x cheapest)']:.1f}x | {pct(r['Retrieval found both'])} | {pct(r['Accuracy when found'])} | {pct(r['Accuracy when not found'])} |")
    md += ["\n## Paper-default condition (no CoT, no number given, vote merge)\n",
           "| Retriever-answerer | Accuracy | Compute (x cheapest) | Found both | Acc. when found | Acc. when not found |", "|---|---|---|---|---|---|"]
    d = main[(main.CoT == "N") & (main["Per-attempt shortlist"] == "paper default") & (main["Final shortlist"] == MERGE_LABEL["vote"])]
    for _, r in d.iterrows():
        md.append(f"| {r['Retriever-answerer setup']} | {pct(r['Overall accuracy'])} | {r['Overall compute (x cheapest)']:.1f}x | "
                  f"{pct(r['Retrieval found both'])} | {pct(r['Accuracy when found'])} | {pct(r['Accuracy when not found'])} |")
    md += ["\n## Effect of each variable (main setups)\n", "| Variable | Level | Mean accuracy | Best accuracy | Mean found both | Mean compute |", "|---|---|---|---|---|---|"]
    for _, r in me.iterrows():
        md.append(f"| {r['Variable']} | {r['Level']} | {pct(r['Mean accuracy'])} | {pct(r['Best accuracy'])} | {pct(r['Mean found both'])} | {r['Mean compute (x cheapest)']:.1f}x |")
    if len(conf):
        md += ["\n## Confidence: does it tell right from wrong?\n",
               "Stated = the 'Confidence: N' the model wrote. Separation 0.5 = no better than chance, 1.0 = perfect. "
               "The paper-default retrieval prompt is left untouched, so it has no stated confidence.\n",
               "| Step | Model | CoT | Signal | Mean confidence | Share correct | Overconfidence | Separation | Correct when above median | at or below |", "|---|---|---|---|---|---|---|---|---|---|"]
        num = lambda r, k, f: "-" if k not in r or r[k] != r[k] or r[k] is None else f.format(r[k])
        for _, r in conf.iterrows():
            md.append(f"| {r['Step']} | {r['Model']} | {r['CoT']} | {r['Confidence signal']} | {num(r, 'Mean confidence', '{:.2f}')} | {pct(r['Share correct'])} | "
                      f"{num(r, 'Overconfidence (confidence minus correct)', '{:+.2f}')} | {num(r, 'Separation (AUC, 0.5 = useless, 1 = perfect)', '{:.2f}')} | "
                      f"{num(r, 'Correct when confidence above median', '{:.0%}')} | {num(r, 'Correct when confidence at or below median', '{:.0%}')} |")
    for title, fn in (("AgentTTS baseline (per condition)", "agenttts.md"), ("Per-question switching", "switching.md")):
        p = os.path.join(plan, fn)
        if os.path.exists(p):
            md.append(f"\n## {title}\n\n" + open(p, encoding="utf-8").read())
    md.append("\nFull tables: `results/set4_results.xlsx`. Per-question rows: `results/per_question.jsonl`. Charts: `results/charts/`.")
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


# ==================================================================================
# Plan stage 1: AgentTTS baseline per condition (search ported in ssw.py, generalised to 3 models per role)
# ==================================================================================
class SweepAgentRun(S.AgentTTSRun):
    def __init__(self, cond, grid, models, lengths, budget, seed, max_trials, batch_size, n_chunks):
        self.cond, self.grid = cond, grid                 # grid[(r, rk, a, ak)] = {"dev": {...}, "test": {...}, "budget": float}
        self.budget, self.seed, self.max_trials, self.batch_size = budget, seed, max_trials, batch_size
        keys = list(dict.fromkeys(RETRIEVERS + ANSWERERS))
        self.name_of = {k: models[k]["id"] for k in keys}
        self.key_of = {v: k for k, v in self.name_of.items()}
        self.size = {self.name_of[k]: models[k]["nominal"] for k in keys}
        self.model_choices = [[self.name_of[k] for k in RETRIEVERS], [self.name_of[k] for k in ANSWERERS]]
        self.lengths = lengths
        self.Np = [lengths["Retrieval"][0], lengths["Question Answering"][0]]
        self.Nd = [lengths["Retrieval"][1], lengths["Question Answering"][1]]
        self.M_dict = {st: max(self.size[m] for m in ch) for st, ch in zip(S.SUBTASKS, self.model_choices)}
        self.task_desc = S.TASK_DESC.format(n=n_chunks)
        self.model_space = {st: {"model_choice": ch, "prompt_length": Np, "generation_length": Nd}
                            for st, ch, Np, Nd in zip(S.SUBTASKS, self.model_choices, self.Np, self.Nd)}
        ex = {st: {"model": ch[-1], "samples": 1, "budget": round(S.compute_budget(1, self.size[ch[-1]], *self.len_of(st, ch[-1])), 1)}
              for st, ch in zip(S.SUBTASKS, self.model_choices)}
        self.example = {"candidates": [ex]}
        self.metrics, self.main = ["Gen_EM", "Ret_F1"], "Gen_EM"
        self.history, self.experience, self.trials, self.log = [], [], [], []
        self.done, self.recent_added, self.evaluated = False, 0, set()

    def len_of(self, subtask, model):
        return tuple(self.lengths[subtask])

    def env_run(self, params):
        try:
            rm, am = self.key_of[params["Retrieval"]["model"]], self.key_of[params["Question Answering"]["model"]]
            rs, as_ = float(params["Retrieval"]["samples"]), float(params["Question Answering"]["samples"])
        except Exception:
            return None, None
        best, best_d = None, None
        for (r, rk, a, ak), g in self.grid.items():
            if (r, a) != (rm, am) or g["budget"] > self.budget:
                continue
            d1 = self.M_dict["Retrieval"] / self.size[self.name_of[rm]] * 4 - 3
            d2 = self.M_dict["Question Answering"] / self.size[self.name_of[am]] * 4 - 3
            d = abs(rk - rs) / d1 + abs(ak - as_) / d2
            if best_d is None or d < best_d:
                best, best_d = (r, rk, a, ak), d
        if best is None or best_d > 5:
            return None, None
        r, rk, a, ak = best
        st = self.grid[best]["dev"]
        snapped = {"Retrieval": {"model": self.name_of[r], "samples": rk}, "Question Answering": {"model": self.name_of[a], "samples": ak}}
        return snapped, {"Gen_EM": round(st["EM"], 3), "Ret_F1": round(st["Ret_F1"], 3), "_config": cfg_name(best)}


def cfg_name(k):
    r, rk, a, ak = k
    return f"{r}x{rk}>{a}x{ak}"


def cfg_words(k):
    r, rk, a, ak = k
    return f"{LABEL[r]} retrieves x{rk}, {LABEL[a]} answers x{ak}"


def cond_name(c):
    cot, pick, merge = c
    return f"c{cot}|p{pick}|{merge}"


def cond_words(c):
    cot, pick, merge = c
    return f"CoT {'on' if cot else 'off'}, per-attempt shortlist {PICK_LABEL[pick]}, final shortlist {MERGE_LABEL[merge]}"


def build_grids(df, models):
    """Per condition: grid of the 36 (retriever, tries, answerer, tries) configs with dev/test stats and Eq. 4 budgets."""
    grids = {}
    for cond, d in df.groupby(["cot", "pick", "merge"], sort=False):
        Np_r, Nd_r = d.ret_prompt_tokens.mean(), (d.ret_out_tokens / d.rk).mean()
        Np_a, Nd_a = d.ans_prompt_tokens.mean(), (d.ans_out_tokens / d.ak).mean()
        lengths = {"Retrieval": [round(Np_r), max(1, round(Nd_r))], "Question Answering": [round(Np_a), max(1, round(Nd_a))]}
        grid = {}
        for (r, rk, a, ak), g in d.groupby(["retriever", "rk", "answerer", "ak"], sort=False):
            st = {}
            for sp in ("dev", "test"):
                x = g[g.split == sp]
                st[sp] = {"EM": float(x.EM.mean()), "F1": float(x.F1.mean()), "Ret_F1": float(x.Ret_F1.mean()), "flops": float(x.flops.mean())}
            st["budget"] = (S.compute_budget(int(rk), models[r]["nominal"], *lengths["Retrieval"]) +
                            S.compute_budget(int(ak), models[a]["nominal"], *lengths["Question Answering"]))
            grid[(r, int(rk), a, int(ak))] = st
        grids[(int(cond[0]), str(cond[1]), str(cond[2]))] = (grid, lengths)
    return grids


PC = "planner's stated confidence"


def run_agenttts(args, df, models, planner, ref, out):
    all_q = S.load_questions(args.data_dir)
    n_chunks = len(all_q[0]["ctxs_candidate"])
    args.n_dev = sum(q["split"] == "dev" for q in all_q)
    grids = build_grids(df, models)
    runs = []
    for cond, (grid, lengths) in grids.items():
        costs = sorted(g["budget"] for g in grid.values())
        for b in [round(float(np.quantile(costs, q)), 1) for q in args.budget_quantiles]:
            for s in range(args.seeds):
                runs.append(SweepAgentRun(cond, grid, models, lengths, b, s, args.max_trials, args.batch_size, n_chunks))
    log(f"AgentTTS baseline | {len(grids)} conditions x {len(args.budget_quantiles)} budgets x {args.seeds} seed(s) = {len(runs)} searches, "
        f"each over {len(next(iter(grids.values()))[0])} configs")
    for r in runs:
        r.evaluate(r.init_candidates(), "init (Insight 1)")
    for it in range(args.iterations):
        active = [r for r in runs if not r.done]
        if not active:
            break
        items = [(*r.experience_prompt(init=(it == 0)), 1000 * it + i, {"kind": "agent_text"}) for i, r in enumerate(active)]
        for r, (think, ans) in zip(active, planner.chat(items)):
            r.experience.append(ans)
            r.log.append({"event": "guidelines", "iteration": it + 1, "text": ans})
        items = [(*r.candidate_prompt(), 2000 + 1000 * it + i, {"kind": "agent_json", "choices": r.model_choices, "k": r.batch_size})
                 for i, r in enumerate(active)]
        for r, (think, ans) in zip(active, planner.chat(items)):
            js = S.extract_json(ans)
            r.log.append({"event": "candidates", "iteration": it + 1, "text": ans, "parsed": js is not None})
            r.evaluate((js or {}).get("candidates", []), f"agent round {it + 1}")
        log(f"  AgentTTS round {it + 1}: {sum(r.done for r in runs)}/{len(runs)} searches finished")
    # after the search (outside the paper's procedure): ask the planner how sure it is of its final choice
    ask = [r for r in runs if r.best()]
    items = []
    for i, r in enumerate(ask):
        hist = "\n".join(f"- {t['config']}: dev accuracy {t['dev_EM']:.3f}, budget {t['budget_eq4']:.0f}" for t in r.trials)
        items.append(("You are an expert in parameter optimization.",
                      f"You searched for the best configuration of a retrieve-then-answer pipeline within a budget of {r.budget}. Configurations "
                      f"are written <retriever>x<tries>><answerer>x<tries>. You tried {len(r.evaluated)} of {len(r.grid)} possible configurations, "
                      f"each scored on {args.n_dev} development questions:\n{hist}\n\nYour final choice is {r.best()}.\n"
                      "How sure are you, in percent, that no better configuration exists within the budget? Give one sentence of reasoning, "
                      "then a last line in exactly this form: 'Confidence: <0-100>'.", 9000 + i, {"kind": "agent_text", "conf": True}))
    said = {}
    if items:
        for r, (think, ans) in zip(ask, planner.chat(items)):
            said[id(r)] = (stated_conf(ans), strip_conf(ans).strip()[:400])
            r.log.append({"event": "confidence", "text": ans})
    name2key = {}
    rows = []
    for r in runs:
        for k in r.grid:
            name2key[cfg_name(k)] = k
        feas = {k: g for k, g in r.grid.items() if g["budget"] <= r.budget}
        opt = max(feas, key=lambda k: (feas[k]["dev"]["EM"], -feas[k]["dev"]["flops"]))
        best = r.best()
        rec = {"condition": cond_name(r.cond), "CoT": "Y" if r.cond[0] else "N", "Per-attempt shortlist": PICK_LABEL[r.cond[1]],
               "Final shortlist": MERGE_LABEL[r.cond[2]], "budget (Eq.4)": r.budget, "seed": r.seed,
               "configs tried": len(r.evaluated), "of configs": len(r.grid)}
        if best:
            k = name2key[best]
            g = r.grid[k]
            rec.update({"AgentTTS choice": cfg_words(k), "dev accuracy": g["dev"]["EM"], "test accuracy": g["test"]["EM"],
                        "test compute (x cheapest)": g["test"]["flops"] / ref})
            others = sorted({t["dev_EM"] for t in r.trials if t["config"] != best}, reverse=True)
            rec["planner's stated confidence"] = said.get(id(r), (None, ""))[0]
            rec["dev accuracy lead over runner-up tried"] = (g["dev"]["EM"] - others[0]) if others else None
            rec["planner's reason"] = said.get(id(r), (None, ""))[1]
        rec.update({"dev-optimal": cfg_words(opt), "dev-optimal test accuracy": r.grid[opt]["test"]["EM"],
                    "dev-optimal test compute (x cheapest)": r.grid[opt]["test"]["flops"] / ref,
                    "found dev-optimal": bool(best) and (name2key[best] == opt or abs(r.grid[name2key[best]]["dev"]["EM"] - r.grid[opt]["dev"]["EM"]) < 1e-9)})
        rows.append(rec)
        S.write_json(os.path.join(out, "traces", f"agenttts_{cond_name(r.cond).replace('|', '_')}_b{r.budget}_s{r.seed}.json"),
                     {"condition": cond_words(r.cond), "budget": r.budget, "trials": r.trials, "log": r.log, "chosen": best})
    open(os.path.join(out, "agenttts.jsonl"), "w").close()
    S.append_jsonl(os.path.join(out, "agenttts.jsonl"), rows)
    top = [x for x in rows if x["budget (Eq.4)"] == max(y["budget (Eq.4)"] for y in rows if y["condition"] == x["condition"])]
    md = [f"One AgentTTS search per condition and budget; the agent picks one retriever, one answerer and the number of tries for all questions. "
          f"Planner: {models[args.planner]['id']}. Found the dev-optimal setup in {sum(x['found dev-optimal'] for x in rows)} of {len(rows)} searches. "
          "Largest budget shown; all budgets are in the Excel sheet.\n",
          "| CoT | Per-attempt shortlist | Final shortlist | AgentTTS choice | Test accuracy | Compute (x cheapest) | Dev-optimal | Found it | Planner's confidence |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in top:
        md.append(f"| {x['CoT']} | {x['Per-attempt shortlist']} | {x['Final shortlist']} | {x.get('AgentTTS choice', '-')} | "
                  f"{100 * x.get('test accuracy', float('nan')):.1f}% | {x.get('test compute (x cheapest)', float('nan')):.1f}x | {x['dev-optimal']} | "
                  f"{'yes' if x['found dev-optimal'] else 'no'} | "
                  + ("-" if x.get(PC) is None else f"{100 * x[PC]:.0f}%") + " |")
    with open(os.path.join(out, "agenttts.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return rows


# ==================================================================================
# Plan stage 2: per-question switching, with rules written by the planner
# ==================================================================================
FEATURES = {
    "yesno": ("bool", "the question can be answered yes or no"),
    "compare": ("bool", "the question compares two things (which / older / more / both / or ...)"),
    "words": ("num", "number of words in the question"),
    "digit": ("bool", "the question contains a digit"),
    "names": ("num", "number of capitalised words after the first (roughly: named entities)"),
    "ret_agree": ("num", "after the small retriever's 3 tries: how much the tries picked the same passages (0 = all different, 1 = identical)"),
    "ans_agree": ("num", "after the small pipeline's 5 answers: share equal to the most common answer (0.2-1.0)"),
    "ans_conf": ("num", "the small answerer's own stated confidence in its final answer (0-1)"),
    "ret_conf": ("num", "the small retriever's own stated confidence that its selection is complete (0-1)"),
}
PRE = ["yesno", "compare", "words", "digit", "names"]
POST = PRE + ["ret_agree", "ans_agree", "ans_conf", "ret_conf"]
_COND = re.compile(r"^\s*([a-z_]+)\s*(<=|>=|=|<|>)\s*([a-z0-9.]+)\s*$")


def parse_rules(text, routes, names, max_rules):
    low = {r.lower(): r for r in routes}
    rules, default, dropped, default_conf = [], None, 0, None
    for line in text.splitlines():
        line = re.sub(r"^[\s\-*\d.)]+", "", line).strip()
        if not line:
            continue
        cf = re.search(r"\[\s*(\d{1,3})\s*%?\s*\]\s*\.?\s*$", line)
        conf = min(100, int(cf.group(1))) / 100.0 if cf else None
        if cf:
            line = line[:cf.start()].strip()
        m = re.match(r"^OTHERWISE\s+(?:use\s+)?(\S+)", line, flags=re.I)
        if m:
            if default is None:
                default = low.get(m.group(1).strip(".,;`'\"").lower())
                default_conf = conf if default else None
            continue
        m = re.match(r"^IF\s+(.+?)\s+THEN\s+(?:use\s+)?(\S+)", line, flags=re.I)
        route = low.get(m.group(2).strip(".,;`'\"").lower()) if m else None
        if not route:
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
            rules.append((conds, route, conf))
        else:
            dropped += 1
    default = default or routes[0]
    tag = lambda c: "" if c is None else f" [{round(100 * c)}]"
    clean = "\n".join(f"{i + 1}. IF " + " AND ".join(f"{a} {o} {v}" for a, o, v in c) + f" THEN {r}{tag(cf)}" for i, (c, r, cf) in enumerate(rules))
    clean += ("\n" if rules else "") + f"{len(rules) + 1}. OTHERWISE {default}{tag(default_conf)}"
    return rules, default, clean, dropped, default_conf


def frontier_routes(stats, cap):
    """stats: {route: (dev_cost, dev_EM)} -> up to `cap` routes on the dev frontier, cheapest first."""
    names = list(stats)
    idx = S.pareto([stats[n] for n in names])
    fr = sorted((names[i] for i in idx), key=lambda n: stats[n][0])
    if len(fr) > cap:
        keep = sorted(set(np.linspace(0, len(fr) - 1, cap).round().astype(int)))
        fr = [fr[i] for i in keep]
    if len(fr) < 2:
        cheapest = min(names, key=lambda n: stats[n][0])
        fr = sorted(set(fr + [cheapest] + [max(names, key=lambda n: stats[n][1])]), key=lambda n: stats[n][0])
    return fr


def run_switching(args, df, planner, ref, out):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    qs = {q["qid"]: q for q in S.load_questions(args.data_dir)}
    main = df[(df.rk == MAIN_RK) & (df.ak == MAIN_AK)]
    feat_q = {qid: S.question_features(q["question"]) for qid, q in qs.items()}
    qcol = {"yesno": "f_yesno", "compare": "f_compare", "words": "f_words", "digit": "f_digit", "names": "f_caps"}
    dev_q = [q for q in qs if qs[q]["split"] == "dev"]
    test_q = [q for q in qs if qs[q]["split"] == "test"]
    scopes = []                                              # (name, words, {route: setup id}, probe setup id)
    probe_pair = (RETRIEVERS[0], ANSWERERS[0])
    for cond, d in main.groupby(["cot", "pick", "merge"], sort=False):
        cond = (int(cond[0]), str(cond[1]), str(cond[2]))
        ids = {f"{r}-{a}": setup_id(r, a, cond[0], cond[1], cond[2], MAIN_RK, MAIN_AK) for r in RETRIEVERS for a in ANSWERERS}
        scopes.append((cond_name(cond), cond_words(cond), ids, ids[f"{probe_pair[0]}-{probe_pair[1]}"]))
    all_ids = sorted(main.setup.unique())
    em = main.pivot_table(index="qid", columns="setup", values="EM")
    fl = main.pivot_table(index="qid", columns="setup", values="flops")
    ragree = main.pivot_table(index="qid", columns="setup", values="ret_agree")
    aagree = main.pivot_table(index="qid", columns="setup", values="ans_agree")
    import pandas as _pd
    num_main = main.assign(ans_conf=_pd.to_numeric(main.ans_conf, errors="coerce"), ret_conf=_pd.to_numeric(main.ret_conf, errors="coerce"))
    aconf = num_main.pivot_table(index="qid", columns="setup", values="ans_conf", dropna=False)
    rconf = num_main.pivot_table(index="qid", columns="setup", values="ret_conf", dropna=False)
    sig_tab = {"ret_agree": ragree, "ans_agree": aagree, "ans_conf": aconf, "ret_conf": rconf}

    def usable(names, probe):
        """Drops a signal the probe setup does not have (e.g. no stated retrieval confidence for the paper-default prompt)."""
        keep = []
        for n in names:
            if n in sig_tab:
                col = sig_tab[n][probe] if probe in sig_tab[n].columns else None
                if col is None or col.reindex(dev_q + test_q).notna().mean() < 0.8:
                    continue
            keep.append(n)
        return keep
    dev_stats = {s: (fl.loc[dev_q, s].mean(), em.loc[dev_q, s].mean()) for s in all_ids}
    g_routes = frontier_routes(dev_stats, args.max_routes + 1)
    g_probe = setup_id(probe_pair[0], probe_pair[1], 0, "silent", "vote", MAIN_RK, MAIN_AK)
    if g_probe in all_ids:
        scopes.append(("all conditions", "any setup from the whole sweep", {f"G{i + 1}": s for i, s in enumerate(g_routes)}, g_probe))

    def describe(sid):
        pair, c, p, mg, _ = sid.split("|")
        r, a = pair.split(">")
        return f"{LABEL[r]} retrieves, {LABEL[a]} answers ({cond_words((int(c[1:]), p[1:], mg))})"

    def feats(q, names, probe):
        v = {}
        for n in names:
            if n in qcol:
                x = feat_q[q][qcol[n]]
                v[n] = ("yes" if x >= 0.5 else "no") if FEATURES[n][0] == "bool" else int(x)
            else:
                x = sig_tab[n].loc[q, probe]
                if x != x:                                   # the model gave no value for this question: use the dev median
                    x = float(np.nanmedian(sig_tab[n].loc[dev_q, probe]))
                v[n] = round(float(x), 2)
        return v

    def num(v):
        return [float(x == "yes") if isinstance(x, str) else float(x) for x in v.values()]

    def holds(vals, a, o, v):
        x = vals[a]
        if isinstance(x, str) or v in ("yes", "no"):
            return str(x) == v if o == "=" else False
        try:
            v = float(v)
        except ValueError:
            return False
        return {"<": x < v, "<=": x <= v, ">": x > v, ">=": x >= v, "=": abs(x - v) < 1e-9}[o]

    def apply(rules, default, vals):
        """-> (route, index of the rule that fired; len(rules) = the OTHERWISE line)."""
        for i, (conds, route, _) in enumerate(rules):
            if all(holds(vals, *c) for c in conds):
                return route, i
        return default, len(rules)

    def score(choice, routes_ids, split_q, probe, use_probe):
        e = np.mean([em.loc[q, routes_ids[choice[q]]] for q in split_q])
        f = np.mean([fl.loc[q, routes_ids[choice[q]]] + (fl.loc[q, probe] if use_probe and routes_ids[choice[q]] != probe else 0.0) for q in split_q])
        return float(e), float(f)

    results, rule_rows, jobs, decisions = [], [], [], []
    for name, words, ids, probe in scopes:
        ids = {k: v for k, v in ids.items() if v in em.columns}
        if len(ids) < 2 or probe not in em.columns:
            continue
        stats = {k: dev_stats[v] for k, v in ids.items()}
        routes = frontier_routes(stats, args.max_routes) if name != "all conditions" else list(ids)
        rid = {k: ids[k] for k in routes}
        best_fixed, rid_all = {}, dict(ids)

        def make_add(name, words, rid_all, probe, best_fixed):
            def add(policy, setting, lam, choice, use_probe, dev_choice=None, conf=None):
                te, tf = score(choice, rid_all, test_q, probe, use_probe)
                row = {"scope": name, "scope description": words, "policy": policy, "setting": setting, "lambda": lam,
                       "test accuracy": te, "test compute (x cheapest)": tf / ref, "test TFLOPs": tf / 1e12,
                       "routes used": ", ".join(f"{k}: {v}" for k, v in sorted(Counter(choice.values()).items()))}
                if dev_choice:
                    de, dfl = score(dev_choice, rid_all, dev_q, probe, use_probe)
                    row.update({"dev accuracy": de, "dev compute (x cheapest)": dfl / ref})
                if lam in best_fixed:
                    b = best_fixed[lam]
                    d, lo, hi = S.bootstrap_diff([em.loc[q, rid_all[choice[q]]] for q in test_q], [em.loc[q, rid_all[b]] for q in test_q])
                    row["accuracy vs best fixed route [95% CI]"] = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                if conf:
                    cv = {q: conf.get(q) for q in test_q if conf.get(q) is not None}
                    for q in test_q:
                        decisions.append({"scope": name, "policy": policy, "setting": setting, "lambda": lam, "qid": q, "route": choice[q],
                                          "route setup": rid_all[choice[q]], "router confidence": conf.get(q), "correct": float(em.loc[q, rid_all[choice[q]]])})
                    if cv:
                        row["router confidence (mean)"] = float(np.mean(list(cv.values())))
                        med = float(np.median(list(cv.values())))
                        hi_q, lo_q = [q for q in cv if cv[q] > med], [q for q in cv if cv[q] <= med]
                        if hi_q and lo_q:
                            row["accuracy when router confidence above median"] = float(np.mean([em.loc[q, rid_all[choice[q]]] for q in hi_q]))
                            row["accuracy when router confidence at or below median"] = float(np.mean([em.loc[q, rid_all[choice[q]]] for q in lo_q]))
                results.append(row)
            return add
        add = make_add(name, words, rid_all, probe, best_fixed)

        for k in routes:
            add(f"always {k}", "fixed", None, {q: k for q in test_q}, False, {q: k for q in dev_q})
        util = {lam: {k: stats[k][1] - lam * stats[k][0] / ref for k in routes} for lam in args.lambdas}
        for lam in args.lambdas:
            best_fixed[lam] = max(routes, key=lambda k: (util[lam][k], -stats[k][0]))
            add("best fixed route (chosen on dev)", "fixed", lam, {q: best_fixed[lam] for q in test_q}, False, {q: best_fixed[lam] for q in dev_q})
        orc = {q: min([k for k in routes if em.loc[q, rid[k]] == 1] or routes, key=lambda k: fl.loc[q, rid[k]]) for q in test_q}
        add("perfect router (hindsight)", "oracle", None, orc, False)
        for setting, names, use_probe in (("question only", PRE, False), ("after small pipeline", usable(POST, probe), True)):
            Xd = np.array([num(feats(q, names, probe)) for q in dev_q])
            Xt = np.array([num(feats(q, names, probe)) for q in test_q])
            sc = StandardScaler().fit(Xd)
            Xd_, Xt_ = sc.transform(Xd), sc.transform(Xt)
            for lam in args.lambdas:
                U = np.array([[em.loc[q, rid[k]] - (lam + 1e-6) * (fl.loc[q, rid[k]] + (fl.loc[q, probe] if use_probe and rid[k] != probe else 0)) / ref
                               for k in routes] for q in dev_q])
                y = np.array([routes[i] for i in U.argmax(1)])
                w = U.max(1) - U.min(1)
                w = w / w.max() if w.max() > 0 else np.ones(len(w))

                def fit_predict(C, mask, X, proba=False):
                    live = mask & (w > 1e-3)
                    if len(set(y[live])) < 2:
                        lab = np.array([y[live][0] if live.any() else routes[0]] * len(X))
                        return (lab, [None] * len(X)) if proba else lab
                    clf = LogisticRegression(C=C, max_iter=2000).fit(Xd_[live], y[live], sample_weight=w[live])
                    return (clf.predict(X), clf.predict_proba(X).max(1).tolist()) if proba else clf.predict(X)
                best_c, best_u, dch = 1.0, None, None
                for C in (0.1, 1.0, 3.0):
                    pred = []
                    for i in range(len(dev_q)):
                        mk = np.ones(len(dev_q), bool)
                        mk[i] = False
                        pred.append(fit_predict(C, mk, Xd_[i:i + 1])[0])
                    ch = dict(zip(dev_q, pred))
                    e, f = score(ch, rid_all, dev_q, probe, use_probe)
                    if best_u is None or e - lam * f / ref > best_u:
                        best_c, best_u, dch = C, e - lam * f / ref, ch
                lab, pr = fit_predict(best_c, np.ones(len(dev_q), bool), Xt_, proba=True)
                add("classifier (cost-weighted)", setting, lam, dict(zip(test_q, lab)), use_probe, dch, conf=dict(zip(test_q, pr)))
            if not args.skip_rules:
                jobs.append({"scope": name, "words": words, "setting": setting, "names": names, "use_probe": use_probe, "routes": routes,
                             "rid": rid, "rid_all": rid_all, "probe": probe, "stats": stats, "add": add, "best": None, "clean": None, "feedback": ""})
    # ---- rules written by the planner, all scopes batched per round ----
    lam = args.rule_lambda
    for rnd in range(args.rule_rounds if jobs else 0):
        items = []
        for ji, j in enumerate(jobs):
            routes, rid, names, probe = j["routes"], j["rid"], j["names"], j["probe"]
            menu = "\n".join(f"- {k}: {describe(rid[k])} | dev accuracy {j['stats'][k][1]:.2f} | cost {j['stats'][k][0] / ref:.1f}x the cheapest setup" for k in routes)
            rows = []
            for n in names:
                vals = {q: feats(q, [n], probe)[n] for q in dev_q}
                if FEATURES[n][0] == "bool":
                    buckets = [("= yes", [q for q in dev_q if vals[q] == "yes"]), ("= no", [q for q in dev_q if vals[q] == "no"])]
                else:
                    med = float(np.median(list(vals.values())))
                    buckets = [(f"<= {med:g}", [q for q in dev_q if vals[q] <= med]), (f"> {med:g}", [q for q in dev_q if vals[q] > med])]
                for lab, qq in buckets:
                    if qq:
                        rows.append(f"| {n} {lab} | {len(qq)} | " + " | ".join(f"{em.loc[qq, rid[k]].mean():.2f}" for k in routes) + " |")
            table = "| feature value | questions | " + " | ".join(f"accuracy {k}" for k in routes) + " |\n|---|---|" + "---|" * len(routes) + "\n" + "\n".join(rows)
            mixed = [q for q in dev_q if 0 < sum(em.loc[q, rid[k]] for k in routes) < len(routes)]
            rest = [q for q in dev_q if q not in set(mixed)]
            rng = random.Random(7)
            rng.shuffle(mixed)
            rng.shuffle(rest)
            shown = (mixed + rest)[:args.max_evidence]
            evidence = "\n".join("- " + "; ".join(f"{a}={b}" for a, b in feats(q, names, probe).items()) + " | correct routes: "
                                 + (", ".join(k for k in routes if em.loc[q, rid[k]] == 1) or "none") for q in shown)
            prompt = (
                "We answer multi-hop questions with a retrieve-then-answer pipeline and can run each question through one of these routes "
                f"(cheapest first):\n{menu}\n\nGoal: choose a route per NEW question to maximise accuracy minus {lam} x cost (cost in units of the cheapest setup"
                + ("; the small pipeline is always run first to get its agreement features, and its cost is paid on top unless you choose that same route" if j["use_probe"] else "")
                + ").\n\nYou may only use these features, with these exact names:\n" + "\n".join(f"- {n}: {FEATURES[n][1]}" for n in names) +
                f"\n\nWrite the COMPLETE routing policy, at most {args.max_rules} lines, checked in order:\n"
                "  IF <feature> <op> <value> [AND <feature> <op> <value>] THEN <route>\n  ...\n  OTHERWISE <route>\n"
                "where <op> is one of < <= > >= = and yes/no features use '= yes' or '= no'. Use the route names exactly as listed. End every line with "
                "your confidence (0-100) that the line sends its questions to a route that answers them correctly, in square brackets, for example "
                "'IF words > 15 THEN <route> [70]'. Lines in another "
                f"form are ignored, and only the first {args.max_rules - 1} IF-lines are kept. Put the policy between <rules> and </rules>.\n\n"
                f"Route accuracy on the {len(dev_q)} development questions, split by feature value:\n{table}\n\n"
                f"Individual development questions ({len(shown)} of {len(dev_q)}, those where routes disagree first):\n{evidence}"
                + (f"\n\nYour previous policy (it will be REPLACED by your new one):\n{j['clean']}\n\nIts result on the development questions: "
                   f"{j['feedback']}\nWrite an improved complete policy." if j["clean"] else ""))
            items.append(("You write short, precise routing policies from evidence.", prompt, 31 * rnd + ji, {"kind": "rules", "actions": routes, "v2": True}))
        for j, (think, ans) in zip(jobs, planner.chat(items)):
            routes, rid_all, names, probe = j["routes"], j["rid_all"], j["names"], j["probe"]
            m = re.search(r"<rules>(.*?)</rules>", ans, flags=re.S)
            rules, default, clean, dropped, dconf = parse_rules(m.group(1) if m else ans, routes, names, args.max_rules)
            ch = {q: apply(rules, default, feats(q, names, probe))[0] for q in dev_q}
            e, f = score(ch, rid_all, dev_q, probe, j["use_probe"])
            u = e - lam * f / ref
            used = Counter(ch.values())
            j["clean"] = clean
            j["feedback"] = (f"accuracy {e:.3f}, cost {f / ref:.2f}x, utility {u:.3f}; routes used " + ", ".join(f"{k}: {used.get(k, 0)}" for k in routes))
            rule_rows.append({"scope": j["scope"], "setting": j["setting"], "round": rnd + 1, "rules kept": len(rules) + 1, "lines dropped": dropped,
                              "dev accuracy": e, "dev compute (x cheapest)": f / ref, "dev utility": u, "rules": clean, "raw output": ans, "chosen": False})
            if j["best"] is None or u > j["best"][0]:
                j["best"] = (u, rules, default, clean, ch, len(rule_rows) - 1, dconf)
        log(f"  switching rules round {rnd + 1}: {len(jobs)} policies written")
    os.makedirs(os.path.join(out, "rules"), exist_ok=True)
    for j in jobs:
        if j["best"] is None:
            continue
        _, rules, default, clean, dch, idx, dconf = j["best"]
        rule_rows[idx]["chosen"] = True
        # per rule: the planner's stated confidence, and how the rule actually did on the dev questions it fired on
        fired = {q: apply(rules, default, feats(q, j["names"], j["probe"])) for q in dev_q}
        stated = [c for _, _, c in rules] + [dconf]
        lines, measured = [], []
        for i in range(len(rules) + 1):
            qq = [q for q in dev_q if fired[q][1] == i]
            acc = float(np.mean([em.loc[q, j["rid_all"][fired[q][0]]] for q in qq])) if qq else None
            measured.append(acc)
            text = clean.splitlines()[i] if i < len(clean.splitlines()) else ""
            lines.append(f"{text}\n      planner's confidence: {'not given' if stated[i] is None else f'{round(100 * stated[i])}%'}; "
                         f"on dev it fired for {len(qq)} questions" + ("" if acc is None else f", {round(100 * acc)}% answered correctly"))
        tap = {q: apply(rules, default, feats(q, j["names"], j["probe"])) for q in test_q}
        tch = {q: r for q, (r, _) in tap.items()}
        conf = {q: (stated[i] if stated[i] is not None else measured[i]) for q, (_, i) in tap.items()}
        # `add` closes over the scope's own rid_all/probe/best_fixed
        j["add"]("rules written by the planner", j["setting"], lam, tch, j["use_probe"], dch, conf=conf)
        rule_rows[idx]["rules with confidence"] = "\n".join(lines)
        legend = "\n".join(f"{k} = {describe(j['rid'][k])}" for k in j["routes"])
        fn = f"{j['scope'].replace('|', '_').replace(' ', '_')}__{j['setting'].replace(' ', '_')}.txt"
        with open(os.path.join(out, "rules", fn), "w", encoding="utf-8") as f:
            f.write(f"Scope: {j['words']}\nSetting: {j['setting']}\nFeatures available: {', '.join(j['names'])}\n\nRoutes:\n{legend}\n\n"
                    "Rules (checked in order; the number in brackets is the planner's own confidence):\n" + "\n".join(lines) + "\n")
    for fn, rows in (("switching.jsonl", results), ("rules.jsonl", rule_rows), ("decisions.jsonl", decisions)):
        open(os.path.join(out, fn), "w").close()
        S.append_jsonl(os.path.join(out, fn), rows)
    res = pd.DataFrame(results)
    md = ["Per-question routing among the routes that are best for their cost on dev. 'question only' uses the question's wording; "
          "'after small pipeline' also uses how much the small retriever's tries and the small answerer's answers agree, and pays for that run. "
          f"Rules are written by the planner model and applied exactly by code. Test = {len(test_q)} questions, so differences under about 10 points are noise.\n"]
    if len(res):
        md += ["| Scope | Best fixed route | Classifier, question only | Classifier, after small pipeline | Planner's rules, question only | Planner's rules, after small pipeline | Perfect router |",
               "|---|---|---|---|---|---|---|"]
        cell = lambda d: "-" if d.empty else f"{100 * d.iloc[0]['test accuracy']:.0f}% at {d.iloc[0]['test compute (x cheapest)']:.1f}x"
        lam0 = args.rule_lambda if args.rule_lambda in args.lambdas else args.lambdas[0]
        for name in res.scope.unique():
            r = res[res.scope == name]
            md.append(f"| {r.iloc[0]['scope description']} | {cell(r[(r.policy.str.startswith('best fixed')) & (r['lambda'] == lam0)])} | "
                      f"{cell(r[(r.policy.str.startswith('classifier')) & (r.setting == 'question only') & (r['lambda'] == lam0)])} | "
                      f"{cell(r[(r.policy.str.startswith('classifier')) & (r.setting == 'after small pipeline') & (r['lambda'] == lam0)])} | "
                      f"{cell(r[(r.policy.str.startswith('rules')) & (r.setting == 'question only')])} | "
                      f"{cell(r[(r.policy.str.startswith('rules')) & (r.setting == 'after small pipeline')])} | {cell(r[r.setting == 'oracle'])} |")
        md.append(f"\nCost weight shown: lambda = {lam0}. The rules themselves are in `plan/rules/` and in the Excel sheet 'Rules'.")
        hi, lo = "accuracy when router confidence above median", "accuracy when router confidence at or below median"
        if hi in res.columns:
            md += ["\n**Is the router's own confidence informative?** Average over all scopes. Classifier confidence = the probability it gives its chosen "
                   "route; for the planner's rules = the confidence the planner wrote on the rule that fired (its measured dev accuracy if none was written).\n",
                   "| Router | Setting | Mean confidence | Accuracy when confidence above median | Accuracy when at or below |", "|---|---|---|---|---|"]
            g = res[res[hi].notna()].groupby(["policy", "setting"], sort=False)
            for (pol, st_), x in g:
                md.append(f"| {pol} | {st_} | {100 * x['router confidence (mean)'].mean():.0f}% | {100 * x[hi].mean():.0f}% | {100 * x[lo].mean():.0f}% |")
            md.append("\nEvery routed test question with its route, the router's confidence and whether the answer was right: `plan/decisions.jsonl` (Excel sheet 'Router decisions').")
    with open(os.path.join(out, "switching.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    return results, rule_rows


def cmd_plan(args):
    models = load_models(args.models_json)
    run = args.run_dir
    out = os.path.join(run, "plan")
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    df = build_per_question(run, args.data_dir, models)
    if df.empty:
        raise SystemExit("no results yet - run the sweep first")
    st = sweep_status(run, args.data_dir)
    if not st["complete"]:
        log("WARNING: the sweep is not complete; planning on the setups that are finished")
    _, ref = setups_table(df)
    ref *= 1e12
    t0 = time.time()
    planner = Planner(args.backend, models, args.planner, args.max_model_len, args.planner_max_tokens)
    run_agenttts(args, df, models, planner, ref, out)
    results, rule_rows = run_switching(args, df, planner, ref, out)
    S.write_json(os.path.join(out, "planner_usage.json"), {"planner": models[args.planner]["id"], "calls": planner.calls, "input_tokens": planner.in_tok,
                                                           "output_tokens": planner.out_tok, "seconds": planner.seconds, "total_seconds": time.time() - t0})
    S.write_json(os.path.join(run, "stages", f"plan_{int(time.time())}.json"), {"stage": "plan", "model": args.planner, "gpu": S.gpu_name(),
                                                                                "gen_s": planner.seconds, "load_s": None})
    print(open(os.path.join(out, "agenttts.md"), encoding="utf-8").read())
    print(open(os.path.join(out, "switching.md"), encoding="utf-8").read())
    for fn in sorted(os.listdir(os.path.join(out, "rules"))):
        print(f"\n--- {fn} ---\n" + open(os.path.join(out, "rules", fn), encoding="utf-8").read())


# ==================================================================================
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, gpu=True):
        p.add_argument("--run_dir", required=True)
        p.add_argument("--data_dir", required=True)
        p.add_argument("--models_json", required=True)
        p.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
        p.add_argument("--time_limit_min", type=float, default=None)
        if gpu:
            p.add_argument("--max_model_len", type=int, default=8192)
            p.add_argument("--gpu_util", type=float, default=0.88)
            p.add_argument("--chunk", type=int, default=10)
    p = sub.add_parser("status")
    common(p, gpu=False)
    p = sub.add_parser("report")
    common(p, gpu=False)
    p = sub.add_parser("retrieve")
    common(p)
    p.add_argument("--model", choices=RETRIEVERS, required=True)
    p = sub.add_parser("answer")
    common(p)
    p.add_argument("--model", choices=ANSWERERS, required=True)
    p.add_argument("--cot", type=int, choices=[0, 1], required=True)
    p = sub.add_parser("plan")
    common(p, gpu=False)
    p.add_argument("--planner", default="q32")
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--planner_max_tokens", type=int, default=1200)
    p.add_argument("--budget_quantiles", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_trials", type=int, default=8)
    p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.03])
    p.add_argument("--rule_lambda", type=float, default=0.03)
    p.add_argument("--rule_rounds", type=int, default=3)
    p.add_argument("--max_rules", type=int, default=8)
    p.add_argument("--max_routes", type=int, default=4)
    p.add_argument("--max_evidence", type=int, default=100)
    p.add_argument("--skip_rules", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.run_dir, "stages"), exist_ok=True)
    {"status": cmd_status, "retrieve": cmd_retrieve, "answer": cmd_answer, "plan": cmd_plan, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
