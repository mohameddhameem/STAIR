"""Saved model outputs for the rule-based workstreams (3-5) on the dev and test questions.

Every retriever variant shortlists every question once (greedy). For each workstream, every shortlist a rulebook can end on is
listed (the first retriever always runs, then up to max_ret_members-1 larger ones, in order; shortlists ensembled), plus each
retriever alone (fixed pipelines). Every answerer variant answers every listed shortlist once (greedy); answerers above the
first in a ladder also draw `answer_samples` sampled answers (the votes). The micro-experiments then replay any rulebook on
these outputs without running a model again.

Untrained variants ('q7') use AgentTTS's retrieval prompt and the step-by-step answer prompt with stated confidence;
fine-tuned variants ('q7:ft') use the prompts they were trained on."""
import itertools
import os

from . import data as D
from .common import (QA_SYSTEM, RET_SYSTEM, answer_of, append_jsonl, atts_merge, atts_ret_prompt, cot_qa_prompt, direct_qa_prompt, direct_ret_prompt, ensemble, key,
                     log, picks_of, read_jsonl, scores, seed_of, selected_refs, selection_text, stated_conf, write_jsonl)
from .config import serve_entry, variant, workstreams
from .engines import free_gpu, make_engine
from .progress import Tracker, mark_done

SPLITS = ("dev", "test")


def step_dir(cfg):
    d = os.path.join(cfg["run_dir"], "step")
    os.makedirs(d, exist_ok=True)
    return d


def fname(v):
    return v.replace(":", "-")


def f_ret(cfg, split, v):
    return os.path.join(step_dir(cfg), f"{split}_ret_{fname(v)}.jsonl")


def f_ans(cfg, v):
    return os.path.join(step_dir(cfg), f"answers_{fname(v)}.jsonl")


def f_smp(cfg, v):
    return os.path.join(step_dir(cfg), f"samples_{fname(v)}.jsonl")


def f_reach(cfg, split, ws):
    return os.path.join(step_dir(cfg), f"{split}_reach_{ws}.jsonl")


def ret_prompt(m, q):
    return direct_ret_prompt(q) if m["finetuned"] else atts_ret_prompt(q)


def qa_prompt(m, q, sel):
    refs = selected_refs(q, sel)
    return direct_qa_prompt(q["question"], refs) if m["finetuned"] else cot_qa_prompt(q["question"], refs)


def ret_variants(cfg):
    out = []
    for ws in workstreams(cfg, "rules"):
        for v in cfg["workstreams"][ws]["retrievers"]:
            if v not in out:
                out.append(v)
    return out


def ans_variants(cfg):
    out = []
    for ws in workstreams(cfg, "rules"):
        for v in cfg["workstreams"][ws]["answerers"]:
            if v not in out:
                out.append(v)
    return out


# ==================================================================================
def retrieve(cfg, v):
    m = variant(cfg, v)
    todo_all = []
    for sp in SPLITS:
        have = {r["qid"] for r in read_jsonl(f_ret(cfg, sp, v))}
        todo_all += [(sp, q) for q in D.load(cfg, sp) if q["qid"] not in have]
    total = sum(len(D.load(cfg, sp)) for sp in SPLITS)
    tr = Tracker(cfg, "retrieve", v, total, "questions")
    tr.tick(total - len(todo_all))
    if todo_all:
        eng = make_engine(cfg, serve_entry(m), lora_path=m["lora_path"])
        P = m["params"]
        chunk = cfg["gpu"]["chunk"]
        for c in range(0, len(todo_all), chunk):
            part = todo_all[c:c + chunk]
            prompts = [eng.encode(RET_SYSTEM, ret_prompt(m, q)) for _, q in part]
            outs = eng.generate(prompts, 1, 0.0, 48 if m["finetuned"] else 512, [seed_of("ret", v, q["qid"]) for _, q in part],
                                [{"kind": "ret" if m["finetuned"] else "ret_atts", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"])} for _, q in part])
            by = {}
            for (sp, q), p, o in zip(part, prompts, outs):
                text, k, lp = o[0]
                if m["finetuned"]:
                    sel = key(picks_of(selection_text(text), len(q["ctxs_candidate"])) or (1,))
                else:                   # untrained: AgentTTS's own reading of a retrieval output (their Merge on one sample)
                    sel = key(i for i in atts_merge([text]) if 1 <= i <= len(q["ctxs_candidate"])) or (1,)
                by.setdefault(sp, []).append({"qid": q["qid"], "variant": v, "sel": list(sel), "text": text[:6000], "prompt_tokens": len(p), "out_tokens": k,
                                              "flops": 2 * P * (len(p) + k), "found_both": float({i + 1 for i in q["ret_answers"]} <= set(sel)),
                                              "mean_logprob": lp})
            for sp, rows in by.items():
                append_jsonl(f_ret(cfg, sp, v), rows)
            tr.tick(total - len(todo_all) + c + len(part))
        free_gpu(eng)
    tr.finish()
    mark_done(cfg, "retrieve", v)


# ==================================================================================
def chains(ladder, max_members):
    """Member lists a rulebook can run: the first retriever, then larger ones in order (skipping allowed)."""
    out = [[ladder[0]]]
    for n in range(1, max_members):
        for comb in itertools.combinations(ladder[1:], n):
            out.append([ladder[0], *comb])
    return out


def reach(cfg, only=None):
    for ws in workstreams(cfg, "rules"):
        if only and ws != only:
            continue
        W = cfg["workstreams"][ws]
        lad = W["retrievers"]
        missing = [v for v in lad for sp in SPLITS if not read_jsonl(f_ret(cfg, sp, v))]
        if missing:
            log(f"{ws}: shortlists of {sorted(set(missing))} not there yet - skipped")
            continue
        for sp in SPLITS:
            R = {v: {r["qid"]: key(r["sel"]) for r in read_jsonl(f_ret(cfg, sp, v))} for v in lad}
            out = set()
            for q in R[lad[0]]:
                for v in lad:
                    out.add((q, R[v][q]))
                for ch in chains(lad, int(cfg["rules"]["max_ret_members"])):
                    out.add((q, ensemble([R[v][q] for v in ch])))
            write_jsonl(f_reach(cfg, sp, ws), [{"qid": q, "sel": list(s)} for q, s in sorted(out)])
            log(f"{ws} {sp}: {len(out)} reachable shortlists ({len(out) / max(1, len(R[lad[0]])):.1f} per question)")
        mark_done(cfg, "reach", ws)


def needs(cfg, v, samples=False):
    """Shortlists one answerer must answer (all workstreams using it; for samples only those where it is above the first rung)."""
    jobs, seen = [], set()
    for ws in workstreams(cfg, "rules"):
        W = cfg["workstreams"][ws]
        if v not in W["answerers"] or (samples and W["answerers"].index(v) == 0):
            continue
        for sp in SPLITS:
            for r in read_jsonl(f_reach(cfg, sp, ws)):
                k = (r["qid"], key(r["sel"]))
                if k not in seen:
                    seen.add(k)
                    jobs.append(k)
    return sorted(jobs)


def answer(cfg, v):
    m = variant(cfg, v)
    qm = D.qmap(cfg, SPLITS)
    jobs = needs(cfg, v)
    have = {(r["qid"], key(r["sel"])) for r in read_jsonl(f_ans(cfg, v))}
    todo = [k for k in jobs if k not in have]
    tr = Tracker(cfg, "answer", v, len(jobs), "shortlists")
    tr.tick(len(jobs) - len(todo))
    log(f"{v}: {len(todo)} answers to generate ({len(have)} done)")
    if todo:
        eng = make_engine(cfg, serve_entry(m), lora_path=m["lora_path"])
        P = m["params"]
        chunk = cfg["gpu"]["chunk"]
        mx = 24 if m["finetuned"] else 512
        for c in range(0, len(todo), chunk):
            part = todo[c:c + chunk]
            prompts = [eng.encode(QA_SYSTEM, qa_prompt(m, qm[q], s)) for q, s in part]
            outs = eng.generate(prompts, 1, 0.0, mx, [0] * len(part),
                                [{"kind": "qa_direct" if m["finetuned"] else "qa", "gold": qm[q]["answer"],
                                  "has_gold": {i + 1 for i in qm[q]["ret_answers"]} <= set(s)} for q, s in part])
            rows = []
            for (q, s), p, o in zip(part, prompts, outs):
                text, k, lp = o[0]
                pred = answer_of(text)
                rows.append({"qid": q, "sel": list(s), "variant": v, "prompt_tokens": len(p), "out_tokens": k, "flops": 2 * P * (len(p) + k),
                             "text": text[:6000], "prediction": pred, **scores(pred, pred, qm[q]["answer"]), "stated_conf": stated_conf(text),
                             "hit_cap": k >= mx, "mean_logprob": lp})
            append_jsonl(f_ans(cfg, v), rows)
            tr.tick(len(jobs) - len(todo) + c + len(part))
        free_gpu(eng)
    tr.finish()
    mark_done(cfg, "answer", v)


def sample(cfg, v):
    n = int(cfg["rules"]["answer_samples"])
    m = variant(cfg, v)
    qm = D.qmap(cfg, SPLITS)
    jobs = needs(cfg, v, samples=True)
    tr = Tracker(cfg, "sample", v, max(1, len(jobs)), "shortlists")
    if jobs:
        have = {(r["qid"], key(r["sel"])) for r in read_jsonl(f_smp(cfg, v)) if len(r["samples"]) >= n}
        todo = [k for k in jobs if k not in have]
        tr.tick(len(jobs) - len(todo))
        if todo:
            eng = make_engine(cfg, serve_entry(m), lora_path=m["lora_path"])
            P = m["params"]
            chunk = cfg["gpu"]["chunk"]
            mx = 24 if m["finetuned"] else 512
            for c in range(0, len(todo), chunk):
                part = todo[c:c + chunk]
                prompts = [eng.encode(QA_SYSTEM, qa_prompt(m, qm[q], s)) for q, s in part]
                outs = eng.generate(prompts, n, 0.7, mx, [seed_of("s", v, q) for q, _ in part],
                                    [{"kind": "qa_direct" if m["finetuned"] else "qa", "gold": qm[q]["answer"],
                                      "has_gold": {i + 1 for i in qm[q]["ret_answers"]} <= set(s)} for q, s in part])
                rows = []
                for (q, s), p, o in zip(part, prompts, outs):
                    ss = []
                    for t, k, lp in o:
                        pred = answer_of(t)
                        ss.append({"prediction": pred, **scores(pred, pred, qm[q]["answer"]), "out_tokens": k, "mean_logprob": lp})
                    rows.append({"qid": q, "sel": list(s), "variant": v, "prompt_tokens": len(p), "samples": ss})
                append_jsonl(f_smp(cfg, v), rows)
                tr.tick(len(jobs) - len(todo) + c + len(part))
            free_gpu(eng)
    tr.finish()
    mark_done(cfg, "sample", v)
