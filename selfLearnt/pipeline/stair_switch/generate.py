"""Model runs (vLLM): shortlists from every retriever, the AgentTTS baseline retrieval, answers and sampled answers on every
shortlist a rulebook can reach. All resumable: rows are appended per chunk, keyed by (question, shortlist)."""
import os
import zlib

from . import data as D
from .common import (QA_SYSTEM, RET_SYSTEM, answer_of, append_jsonl, baseline_ret_prompt, direct_prompt, em_f1, ensemble, free_gpu, key,
                     lenient, log, make_engine, picks_of, qa_prompt, read_jsonl, selected_refs, selection_text, stated_conf, write_jsonl)


def step_dir(cfg):
    d = os.path.join(cfg["run_dir"], "step")
    os.makedirs(d, exist_ok=True)
    return d


def ret_file(cfg, split, rkey):
    return os.path.join(step_dir(cfg), f"{split}_retrieval_{rkey}.jsonl")


def retrieve(cfg, split, entry, baseline=False):
    """Shortlists of one retriever on one question set. baseline=True: AgentTTS's pick (untrained, step-by-step, one try)."""
    qs = D.load(cfg, split)
    path = ret_file(cfg, split, entry["key"])
    done = {r["qid"] for r in read_jsonl(path)}
    todo = [q for q in qs if q["qid"] not in done]
    if not todo:
        log(f"{split} {entry['key']}: complete")
        return
    eng = make_engine(cfg, entry)
    if not baseline:
        eng.thinking = False if entry.get("no_thinking") else eng.thinking
    P = entry["params"]
    chunk = cfg["gpu"].get("chunk", 256)
    for c in range(0, len(todo), chunk):
        part = todo[c:c + chunk]
        prompts = [eng.encode(RET_SYSTEM, baseline_ret_prompt(q) if baseline else direct_prompt(q)) for q in part]
        meta = [{"kind": "ret", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"])} for q in part]
        seeds = [zlib.crc32((entry["key"] + q["qid"]).encode()) % 999983 for q in part]
        outs = eng.generate(prompts, 1, 0.9 if baseline else 0.0, 1024 if baseline else 48, seeds, meta)
        rows = []
        for q, p, o in zip(part, prompts, outs):
            text, k, lp = o[0][0], o[0][1], (o[0][2] if len(o[0]) > 2 else None)
            sel = key(picks_of(selection_text(text), len(q["ctxs_candidate"])) or (1,))
            rows.append({"qid": q["qid"], "retriever": entry["key"], "sel": list(sel), "text": text, "prompt_tokens": len(p), "out_tokens": k,
                         "flops": 2 * P * (len(p) + k), "found_both": float({i + 1 for i in q["ret_answers"]} <= set(sel)), "mean_logprob": lp})
        append_jsonl(path, rows)
        log(f"  {split} {entry['key']}: {len(done) + c + len(part)}/{len(qs)}")
    del eng
    free_gpu()


def ladder_states(ladder, start):
    """Retrievers run, in order, when a rulebook escalates to `start` and keeps escalating. The cheapest is always run."""
    base = ladder[0]
    if start == base:
        st = [[base]]
        for r in ladder[1:]:
            st.append(st[-1] + [r])
        return st
    i = ladder.index(start)
    st = [[base, start]]
    for r in ladder[i + 1:]:
        st.append(st[-1] + [r])
    return st


def reach(cfg, split):
    """Every shortlist a rulebook (or a fixed pipeline) can end on: each retriever alone, every escalation chain, the baseline."""
    ladder = [r["key"] for r in cfg["retrievers"]]
    R = {r: {x["qid"]: x for x in read_jsonl(ret_file(cfg, split, r))} for r in ladder}
    base = read_jsonl(ret_file(cfg, split, cfg["baseline"]["key"])) if split == "test" else []
    out = set()
    for q in R[ladder[0]]:
        for r in ladder:
            out.add((q, key(R[r][q]["sel"])))
        for s in ladder:
            for members in ladder_states(ladder, s):
                out.add((q, ensemble([R[m][q]["sel"] for m in members])))
    for x in base:
        out.add((x["qid"], key(x["sel"])))
    rows = [{"qid": q, "sel": list(s)} for q, s in sorted(out)]
    write_jsonl(os.path.join(step_dir(cfg), f"{split}_reach.jsonl"), rows)
    log(f"{split}: {len(rows)} reachable shortlists ({len(rows) / max(1, len(R[ladder[0]])):.2f} per question)")


def _jobs(cfg, splits):
    qmap, jobs, seen = {}, [], set()
    for sp in splits:
        for q in D.load(cfg, sp):
            qmap[q["qid"]] = q
        for r in read_jsonl(os.path.join(step_dir(cfg), f"{sp}_reach.jsonl")):
            k = (r["qid"], key(r["sel"]))
            if k not in seen:
                seen.add(k)
                jobs.append(k)
    return qmap, jobs


def answer(cfg, splits, model):
    """One greedy step-by-step answer (with stated confidence and token probability) on every reachable shortlist."""
    qmap, jobs = _jobs(cfg, splits)
    path = os.path.join(step_dir(cfg), f"answers_{model['key']}.jsonl")
    have = {(r["qid"], key(r["sel"])) for r in read_jsonl(path)}
    todo = [k for k in jobs if k not in have]
    log(f"{model['key']}: {len(todo)} answers to generate ({len(have)} already done)")
    if not todo:
        return
    eng = make_engine(cfg, model)
    P = model["params"]
    chunk = cfg["gpu"].get("chunk", 256)
    for c in range(0, len(todo), chunk):
        part = todo[c:c + chunk]
        prompts = [eng.encode(QA_SYSTEM, qa_prompt(qmap[q]["question"], selected_refs(qmap[q], s))) for q, s in part]
        outs = eng.generate(prompts, 1, 0.0, 512, [0] * len(part), [{"kind": "qa", "gold": qmap[q]["answer"]} for q, _ in part])
        rows = []
        for (q, s), p, o in zip(part, prompts, outs):
            text, k = o[0][0], o[0][1]
            pred = answer_of(text)
            em, f1 = em_f1(pred, qmap[q]["answer"])
            rows.append({"qid": q, "sel": list(s), "model": model["key"], "prompt_tokens": len(p), "out_tokens": k, "flops": 2 * P * (len(p) + k),
                         "text": text, "prediction": pred, "EM": em, "F1": f1, "lenient": lenient(pred, qmap[q]["answer"]),
                         "stated_conf": stated_conf(text), "hit_cap": k >= 512, "mean_logprob": o[0][2] if len(o[0]) > 2 else None})
        append_jsonl(path, rows)
        log(f"  {model['key']}: {min(c + chunk, len(todo))}/{len(todo)}")
    del eng
    free_gpu()


def sample(cfg, splits, model, n):
    """n sampled answers (temperature 0.7) from the big answerer on every reachable shortlist: the votes."""
    qmap, jobs = _jobs(cfg, splits)
    path = os.path.join(step_dir(cfg), f"samples_{model['key']}.jsonl")
    have = {(r["qid"], key(r["sel"])) for r in read_jsonl(path) if len(r["samples"]) >= n}
    todo = [k for k in jobs if k not in have]
    log(f"{model['key']}: {len(todo)} shortlists x {n} samples to generate")
    if not todo:
        return
    eng = make_engine(cfg, model)
    P = model["params"]
    chunk = cfg["gpu"].get("chunk", 256)
    for c in range(0, len(todo), chunk):
        part = todo[c:c + chunk]
        prompts = [eng.encode(QA_SYSTEM, qa_prompt(qmap[q]["question"], selected_refs(qmap[q], s))) for q, s in part]
        outs = eng.generate(prompts, n, 0.7, 512, [zlib.crc32((q + "s").encode()) % 999983 for q, _ in part],
                            [{"kind": "qa", "gold": qmap[q]["answer"]} for q, _ in part])
        rows = []
        for (q, s), p, o in zip(part, prompts, outs):
            ss = []
            for t, k, *rest in o:
                pred = answer_of(t)
                em, f1 = em_f1(pred, qmap[q]["answer"])
                ss.append({"prediction": pred, "EM": em, "F1": f1, "lenient": lenient(pred, qmap[q]["answer"]), "out_tokens": k,
                           "mean_logprob": rest[0] if rest else None})
            rows.append({"qid": q, "sel": list(s), "model": model["key"], "prompt_tokens": len(p), "samples": ss,
                         "flops": 2 * P * (len(ss) * len(p) + sum(x["out_tokens"] for x in ss))})
        append_jsonl(path, rows)
        log(f"  {model['key']} samples: {min(c + chunk, len(todo))}/{len(todo)}")
    del eng
    free_gpu()
